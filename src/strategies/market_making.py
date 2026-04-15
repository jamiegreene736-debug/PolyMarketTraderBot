from loguru import logger
from src.strategies.base import BaseStrategy


class MarketMakingStrategy(BaseStrategy):
    """
    Market Making Strategy
    ----------------------
    Places limit orders on both sides (bid and ask) of high-volume markets.
    Earns the spread on every fill. Refreshes orders periodically to stay
    close to the current mid price.

    Example: Market mid = $0.50
             Place BUY @ $0.485 and SELL @ $0.515 (3-cent spread each side)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_refresh: dict[str, float] = {}

    async def run(self):
        if not self.enabled:
            return

        import asyncio
        now = asyncio.get_event_loop().time()

        min_volume = self.config.get("min_daily_volume", 10000)
        spread = self.config.get("spread_pct", 0.03)
        max_orders = self.config.get("max_open_orders_per_market", 4)
        order_size = self.config.get("order_size_usdc", 100)
        num_markets = self.config.get("num_markets", 5)
        refresh_interval = self.config.get("refresh_interval_seconds", 60)

        markets = await self.market_data.get_markets_by_volume(
            min_volume=min_volume, top_n=num_markets
        )

        if not markets:
            self.log("No markets found with sufficient volume")
            return

        self.log(f"Making markets on {len(markets)} markets")

        for market in markets:
            slug = self.market_data.get_slug(market)
            question = self.market_data.get_question(market)

            if not slug:
                continue

            last_refresh = self._last_refresh.get(slug, 0)
            if (now - last_refresh) < refresh_interval:
                continue

            bbo = await self.market_data.get_bbo(slug, force=True)
            if not bbo:
                continue

            try:
                best_bid = float(bbo.get("bid", {}).get("price", 0))
                best_ask = float(bbo.get("ask", {}).get("price", 1))
            except (TypeError, ValueError):
                continue

            if best_bid <= 0 or best_ask >= 1:
                continue

            mid = round((best_bid + best_ask) / 2, 4)
            our_bid = round(max(0.01, mid - spread / 2), 4)
            our_ask = round(min(0.99, mid + spread / 2), 4)

            # Cancel stale orders before placing new ones
            await self.order_manager.cancel_stale_orders(slug, current_mid=mid, max_drift=0.05)

            current_count = self.order_manager.get_market_order_count(slug)
            if current_count >= max_orders:
                continue

            # Place BUY (long YES at bid)
            if self.capital_manager.can_allocate(self.name, order_size):
                shares_bid = round(order_size / our_bid, 2)
                if self.capital_manager.allocate(self.name, order_size):
                    oid = await self.order_manager.place_order(
                        market_slug=slug,
                        question=question,
                        intent="ORDER_INTENT_BUY_LONG",
                        price=our_bid,
                        quantity=shares_bid,
                        strategy=self.name,
                    )
                    if not oid:
                        self.capital_manager.release(self.name, order_size)

            # Place SELL (short YES at ask = buy NO)
            if self.capital_manager.can_allocate(self.name, order_size):
                shares_ask = round(order_size / (1 - our_ask), 2)
                if self.capital_manager.allocate(self.name, order_size):
                    oid = await self.order_manager.place_order(
                        market_slug=slug,
                        question=question,
                        intent="ORDER_INTENT_BUY_SHORT",
                        price=round(1 - our_ask, 4),
                        quantity=shares_ask,
                        strategy=self.name,
                    )
                    if not oid:
                        self.capital_manager.release(self.name, order_size)

            self._last_refresh[slug] = now
            self.log(f"Quotes on '{question[:50]}': bid=${our_bid} ask=${our_ask} mid=${mid}")
