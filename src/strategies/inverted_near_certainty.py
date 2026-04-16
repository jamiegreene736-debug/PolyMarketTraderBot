"""
Inverted Near-Certainty Strategy
----------------------------------
The mirror image of Near-Certainty. Instead of buying YES contracts
priced near $1.00, this buys NO contracts on markets where YES is
priced very LOW (near $0.00) — meaning NO is nearly certain to win.

Example:
  Market: "Will candidate X win?" YES @ $0.04
  → NO is priced at $0.96 — nearly certain to resolve YES=No
  → Buy NO (equivalent to selling YES short) at $0.96
  → Hold to $1.00 resolution. Net profit after fees ≈ 3.6%

This doubles the near-certainty opportunity set at zero extra complexity.
The fee structure is identical — taker fees are very small at extreme prices.
"""

from src.strategies.base import BaseStrategy
from src import fees


class InvertedNearCertaintyStrategy(BaseStrategy):

    async def run(self):
        if not self.enabled:
            return

        max_yes_price    = self.config.get("max_yes_price", 0.07)
        max_hours        = self.config.get("max_hours_to_resolution", 72)
        min_volume       = self.config.get("min_market_volume", 1000)
        fallback_size    = self.config.get("order_size_usdc", 50)
        min_net_return   = self.config.get("min_net_return_pct", 1.0)
        use_kelly        = self.config.get("use_kelly_sizing", True)
        kelly_frac       = self.config.get("kelly_fraction", 0.25)

        # If max_hours is large (>720), scan all active markets
        if max_hours >= 720:
            markets = await self.market_data.get_markets()
        else:
            markets = await self.market_data.get_markets_resolving_soon(
                max_hours=max_hours, min_volume=min_volume
            )

        if not markets:
            self.log("No active markets found")
            return

        self.log(f"Scanning {len(markets)} markets for near-certain NO (YES<=${max_yes_price})")
        entered = 0

        for market in markets:
            slug     = self.market_data.get_slug(market)
            question = self.market_data.get_question(market)

            if not slug:
                continue

            bbo = await self.market_data.get_bbo(slug)
            if not bbo:
                continue

            try:
                best_bid = float(bbo.get("bid", {}).get("price", 0))
                best_ask = float(bbo.get("ask", {}).get("price", 1))
            except (TypeError, ValueError):
                continue

            # YES must be priced very low — meaning NO is near certain
            if best_ask > max_yes_price:
                continue

            # NO price derived from YES mid (more stable than ask)
            mid = float(bbo.get("mid", best_ask))
            no_price = round(1 - mid, 4)

            # Fee-aware profit (buying NO at no_price, resolves at 1.00)
            net_profit_pct = fees.net_profit_pct_near_certainty(no_price)
            if net_profit_pct < min_net_return:
                continue

            hours_left = self.market_data._hours_to_resolution(market)
            fee_cost   = fees.taker_fee_per_share(no_price)

            self.log(
                f"NO opportunity: {question[:55]} | "
                f"YES=${mid:.4f} → NO=${no_price:.4f} fee=${fee_cost:.4f} | "
                f"net={net_profit_pct:.2f}% | {hours_left:.1f}h left"
            )

            if self.order_manager.get_market_order_count(slug) > 0:
                continue

            # Kelly sizing: win_prob ≈ no_price; net return = net_profit_pct/100
            if use_kelly:
                order_size = self.capital_manager.kelly_size(
                    self.name,
                    win_prob=no_price,
                    net_return_pct=net_profit_pct / 100,
                    kelly_fraction=kelly_frac,
                    min_size=10.0,
                    max_size=fallback_size * 2,
                )
            else:
                order_size = fallback_size

            if not self.capital_manager.can_allocate(self.name, order_size):
                self.log("Capital limit reached", level="warning")
                break

            # Price aggressively to cross the spread and get an immediate taker fill
            taker_price = min(round(no_price + 0.03, 4), 0.99)
            shares = round(order_size / taker_price, 2)

            if not self.capital_manager.allocate(self.name, order_size):
                break

            order_id = await self.order_manager.place_order(
                market_slug=slug,
                question=question,
                intent="ORDER_INTENT_BUY_SHORT",
                price=taker_price,
                quantity=shares,
                strategy=self.name,
                tif="TIME_IN_FORCE_FILL_OR_KILL",
            )

            if order_id:
                self.log(
                    f"BUY NO {shares:.1f} shares @ ${no_price:.4f} | "
                    f"net return {net_profit_pct:.2f}% | '{question[:45]}'"
                )
                entered += 1
            else:
                self.capital_manager.release(self.name, order_size)

        if entered:
            self.log(f"Entered {entered} inverted near-certainty position(s)")
