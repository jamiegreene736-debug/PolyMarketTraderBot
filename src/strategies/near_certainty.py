import asyncio
from src.strategies.base import BaseStrategy
from src import fees


class NearCertaintyStrategy(BaseStrategy):
    """
    Near-Certainty Bond Strategy
    ----------------------------
    Buys YES contracts priced >= min_price on markets resolving within
    max_hours. Holds to $1.00 at resolution.

    Fee-aware: taker fees are calculated and deducted from expected profit.
    Only enters when net profit after fees exceeds min_net_return_pct.

    Example at p=$0.94, resolves in 24h:
      Taker fee  = 0.05 × 0.94 × 0.06 = $0.00282/share
      Net profit = 1.00 - 0.94 - 0.00282 = $0.0572/share (6.1%)
    """

    async def run(self):
        if not self.enabled:
            return

        min_price        = self.config.get("min_price", 0.93)
        max_hours        = self.config.get("max_hours_to_resolution", 8760)   # default 1 year
        min_volume       = self.config.get("min_market_volume", 0)
        fallback_size    = self.config.get("order_size_usdc", 50)
        min_net_return   = self.config.get("min_net_return_pct", 1.0)
        use_kelly        = self.config.get("use_kelly_sizing", True)
        kelly_frac       = self.config.get("kelly_fraction", 0.25)

        # If max_hours is large (>720), scan all active markets instead of resolving-soon
        if max_hours >= 720:
            markets = await self.market_data.get_markets()
        else:
            markets = await self.market_data.get_markets_resolving_soon(
                max_hours=max_hours, min_volume=min_volume
            )

        if not markets:
            self.log("No active markets found")
            return

        self.log(f"Scanning {len(markets)} markets for near-certainty YES (price>=${min_price})")
        entered = 0

        # Batch-fetch all BBOs in parallel up front. Previously this loop did
        # ~500 sequential awaits per tick, which could take longer than the
        # poll interval itself on slow ticks.
        market_slugs = [(m, self.market_data.get_slug(m)) for m in markets]
        market_slugs = [(m, s) for m, s in market_slugs if s]
        bbos = await asyncio.gather(
            *[self.market_data.get_bbo(s) for _, s in market_slugs]
        )

        for (market, slug), bbo in zip(market_slugs, bbos):
            question = self.market_data.get_question(market)

            if not bbo:
                continue

            try:
                best_bid = float(bbo.get("bid", {}).get("price", 0))
                best_ask = float(bbo.get("ask", {}).get("price", 1))
                mid      = float(bbo.get("mid", (best_bid + best_ask) / 2))
            except (TypeError, ValueError):
                continue

            if best_bid < min_price:
                continue

            # Fee-aware profitability check (use mid for fee calc)
            net_profit_pct = fees.net_profit_pct_near_certainty(mid)
            if net_profit_pct < min_net_return:
                self.log(f"Skipping {slug}: net return {net_profit_pct:.2f}% < {min_net_return}% minimum")
                continue

            hours_left = self.market_data._hours_to_resolution(market)
            fee_cost   = fees.taker_fee_per_share(mid)

            self.log(
                f"Opportunity: {question[:55]} | "
                f"mid=${mid:.4f} fee=${fee_cost:.4f} | "
                f"net={net_profit_pct:.2f}% | {hours_left:.1f}h left"
            )

            if self.order_manager.get_market_order_count(slug) > 0:
                continue

            # Kelly sizing: win_prob ≈ mid; net return = net_profit_pct/100
            if use_kelly:
                order_size = self.capital_manager.kelly_size(
                    self.name,
                    win_prob=mid,
                    net_return_pct=net_profit_pct / 100,
                    kelly_fraction=kelly_frac,
                    min_size=fallback_size,   # always at least order_size_usdc so we clear 5-share min
                    max_size=fallback_size,
                )
            else:
                order_size = fallback_size

            if not self.capital_manager.can_allocate(self.name, order_size):
                self.log("Capital limit reached", level="warning")
                break

            # Price aggressively above the ask to guarantee an immediate taker fill.
            # On a CLOB this still fills at the resting ask price — we don't overpay.
            taker_price = min(round(best_ask + 0.03, 4), 0.99)
            shares = round(order_size / taker_price, 2)

            if not self.capital_manager.allocate(self.name, order_size):
                break

            order_id = await self.order_manager.place_order(
                market_slug=slug,
                question=question,
                intent="ORDER_INTENT_BUY_LONG",
                price=taker_price,
                quantity=shares,
                strategy=self.name,
                tif="TIME_IN_FORCE_FILL_OR_KILL",
            )

            if order_id:
                self.log(
                    f"BUY {shares:.1f} shares @ ${best_ask:.4f} | "
                    f"effective cost ${fees.effective_taker_cost_per_share(best_ask):.4f}/share | "
                    f"net return {net_profit_pct:.2f}% | '{question[:45]}'"
                )
                entered += 1
            else:
                self.capital_manager.release(self.name, order_size)

        if entered:
            self.log(f"Entered {entered} near-certainty position(s) this tick")
