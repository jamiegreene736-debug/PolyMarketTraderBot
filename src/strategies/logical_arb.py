import asyncio
from loguru import logger
from src.strategies.base import BaseStrategy
from src import fees


class LogicalArbStrategy(BaseStrategy):
    """
    Logical (Combinatorial) Arbitrage Strategy
    -------------------------------------------
    Groups related markets (same event) and checks if the sum of YES prices
    violates logical constraints.

    If YES prices sum to < 1.0 - threshold:
        Buy ALL outcomes. One must pay $1.00. Cost < $1.00. Guaranteed profit.

    Example: 3 candidates in an election priced at $0.30 + $0.30 + $0.30 = $0.90
             Buy all three for $0.90 total. One pays $1.00. Profit = $0.10 (11%).
    """

    async def run(self):
        if not self.enabled:
            return

        min_arb = self.config.get("min_arb_pct", 0.03)
        order_size = self.config.get("order_size_usdc", 100)
        min_outcomes = self.config.get("min_outcomes", 2)

        event_groups = await self.market_data.get_grouped_markets()

        if not event_groups:
            self.log("No grouped markets found for arbitrage scanning")
            return

        self.log(f"Scanning {len(event_groups)} event groups for logical arbitrage")

        # Collect every (event_id, market, slug) tuple that needs a BBO, then
        # fetch them all concurrently in one gather call instead of nested
        # sequential awaits across hundreds of markets.
        eligible_groups = {
            eid: ms for eid, ms in event_groups.items() if len(ms) >= min_outcomes
        }
        fetch_plan = []
        for event_id, markets in eligible_groups.items():
            for market in markets:
                slug = self.market_data.get_slug(market)
                if slug:
                    fetch_plan.append((event_id, market, slug))

        bbos = await asyncio.gather(
            *[self.market_data.get_bbo(slug) for _, _, slug in fetch_plan]
        )

        # Re-bucket results by event_id
        by_event: dict[str, list] = {}
        for (event_id, market, slug), bbo in zip(fetch_plan, bbos):
            by_event.setdefault(event_id, []).append((market, slug, bbo))

        for event_id, entries in by_event.items():
            slugs = []
            ask_prices = []
            questions = []

            for market, slug, bbo in entries:
                question = self.market_data.get_question(market)
                if not bbo:
                    continue

                try:
                    ask = float(bbo.get("ask", {}).get("price", 1.0))
                except (TypeError, ValueError):
                    continue

                slugs.append(slug)
                ask_prices.append(ask)
                questions.append(question)

            if len(slugs) < min_outcomes:
                continue

            total_cost = sum(ask_prices)

            # Account for taker fees on every leg
            total_fees = sum(
                fees.taker_fee(round(order_size / max(p, 0.01), 2), p)
                for p in ask_prices
            )
            net_profit = 1.0 - total_cost - (total_fees / order_size if order_size > 0 else 0)
            profit_pct = round(net_profit * 100, 2)

            if net_profit <= min_arb:
                continue

            self.log(
                f"ARB FOUND on event {event_id}: "
                f"total_cost=${total_cost:.4f} | profit={profit_pct}% | "
                f"outcomes={len(slugs)}"
            )

            # Check we have enough capital for all legs
            total_needed = order_size * len(slugs)
            if not self.capital_manager.can_allocate(self.name, total_needed):
                self.log(f"Not enough capital for {len(slugs)}-leg arb (need ${total_needed})", level="warning")
                continue

            # Place all legs simultaneously
            placed = 0
            for slug, ask, question in zip(slugs, ask_prices, questions):
                if not self.capital_manager.allocate(self.name, order_size):
                    break

                shares = round(order_size / ask, 2)
                order_id = await self.order_manager.place_order(
                    market_slug=slug,
                    question=question,
                    intent="ORDER_INTENT_BUY_LONG",
                    price=ask,
                    quantity=shares,
                    strategy=self.name,
                )

                if order_id:
                    placed += 1
                    self.log(f"  Leg {placed}: BUY {shares} @ ${ask} on '{question[:45]}'")
                else:
                    self.capital_manager.release(self.name, order_size)

            if placed == len(slugs):
                self.log(f"Full arb entered: {placed} legs, cost=${total_cost:.4f}, expected profit={profit_pct}%")
            elif placed > 0:
                self.log(f"Partial arb entered ({placed}/{len(slugs)} legs) - WARNING: incomplete arb is risky", level="warning")
