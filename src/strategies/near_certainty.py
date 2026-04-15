from loguru import logger
from src.strategies.base import BaseStrategy


class NearCertaintyStrategy(BaseStrategy):
    """
    Near-Certainty Bond Strategy
    ----------------------------
    Finds markets resolving within X hours where YES is priced >= min_price
    (e.g. $0.93). Buys and holds to $1.00 at resolution.

    Example: Buy YES at $0.95 on a market resolving in 12 hours.
             Return = (1.00 - 0.95) / 0.95 = 5.3% in 12 hours.
    """

    async def run(self):
        if not self.enabled:
            return

        min_price = self.config.get("min_price", 0.93)
        max_hours = self.config.get("max_hours_to_resolution", 48)
        min_volume = self.config.get("min_market_volume", 1000)
        order_size = self.config.get("order_size_usdc", 50)

        markets = await self.market_data.get_markets_resolving_soon(
            max_hours=max_hours, min_volume=min_volume
        )

        if not markets:
            self.log("No markets resolving soon found")
            return

        self.log(f"Scanning {len(markets)} markets resolving within {max_hours}h")

        for market in markets:
            slug = self.market_data.get_slug(market)
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

            # We want to BUY YES at a price >= min_price
            # If best_ask <= 1.0 and best_bid >= min_price, opportunity exists
            if best_bid < min_price:
                continue

            expected_return = round((1.0 - best_ask) / best_ask * 100, 2)
            hours_left = self.market_data._hours_to_resolution(market)

            self.log(
                f"Opportunity: {question[:60]} | "
                f"bid=${best_bid} ask=${best_ask} | "
                f"return={expected_return}% | {hours_left:.1f}h left"
            )

            if not self.capital_manager.can_allocate(self.name, order_size):
                self.log("Capital limit reached, skipping", level="warning")
                break

            # Skip if we already have an open order on this market
            if self.order_manager.get_market_order_count(slug) > 0:
                self.log(f"Already have open order on {slug}, skipping")
                continue

            allocated = self.capital_manager.allocate(self.name, order_size)
            if not allocated:
                break

            shares = round(order_size / best_ask, 2)
            order_id = await self.order_manager.place_order(
                market_slug=slug,
                question=question,
                intent="ORDER_INTENT_BUY_LONG",
                price=best_ask,
                quantity=shares,
                strategy=self.name,
            )

            if order_id:
                self.log(
                    f"Placed BUY {shares} shares @ ${best_ask} on '{question[:50]}' "
                    f"(expected +{expected_return}%)"
                )
            else:
                self.capital_manager.release(self.name, order_size)
