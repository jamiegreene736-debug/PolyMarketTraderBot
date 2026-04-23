import asyncio
from src.strategies.base import BaseStrategy
from src import fees


class MarketMakingStrategy(BaseStrategy):
    """
    Market Making Strategy (Rebate-Aware)
    ---------------------------------------
    Places limit orders on both sides of high-volume markets and earns:
      1. The bid/ask spread on every round trip
      2. Maker rebates (25% of taker fees) on every fill

    Fee model at mid=$0.50:
      Rebate per fill = 0.0125 × qty × 0.50 × 0.50 = $0.003125/share
      Round-trip rebate = $0.00625/share on top of spread income

    Since we earn rebates as makers, the minimum viable spread is just
    1 cent — far tighter than the old 3-cent default. We now scan 20
    markets instead of 5 to maximize rebate volume.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_refresh: dict[str, float] = {}

    async def run(self):
        if not self.enabled:
            return

        now = asyncio.get_event_loop().time()

        min_volume       = self.config.get("min_daily_volume", 10000)
        num_markets      = self.config.get("num_markets", 20)
        spread_pct       = self.config.get("spread_pct", 0.02)
        max_orders       = self.config.get("max_open_orders_per_market", 4)
        order_size       = self.config.get("order_size_usdc", 100)
        refresh_interval = self.config.get("refresh_interval_seconds", 60)

        markets = await self.market_data.get_markets_by_volume(
            min_volume=min_volume, top_n=num_markets
        )

        if not markets:
            self.log("No markets found with sufficient volume")
            return

        self.log(f"Market making on {len(markets)} markets")

        for market in markets:
            slug     = self.market_data.get_slug(market)
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

            if best_bid <= 0 or best_ask >= 1 or best_ask <= best_bid:
                continue

            mid = round((best_bid + best_ask) / 2, 4)

            # Use the larger of configured spread or minimum viable spread
            half_spread = max(spread_pct / 2, fees.min_viable_spread(mid) / 2)
            our_bid = round(max(0.01, mid - half_spread), 4)
            our_ask = round(min(0.99, mid + half_spread), 4)

            # Calculate expected rebate to log profitability
            shares_per_order = round(order_size / mid, 2)
            rebate_per_rt    = fees.market_making_rebate_round_trip(mid, shares_per_order)
            spread_income    = (our_ask - our_bid) * shares_per_order

            await self.order_manager.cancel_stale_orders(slug, current_mid=mid, max_drift=0.04)

            current_count = self.order_manager.get_market_order_count(slug)
            if current_count >= max_orders:
                continue

            # Place BUY side (long YES at bid)
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

            # Place SELL side (short YES at ask = buy NO)
            if (
                self.order_manager.get_market_order_count(slug) < max_orders
                and self.capital_manager.can_allocate(self.name, order_size)
            ):
                no_price   = round(1 - our_ask, 4)
                shares_ask = round(order_size / max(no_price, 0.01), 2)
                if self.capital_manager.allocate(self.name, order_size):
                    oid = await self.order_manager.place_order(
                        market_slug=slug,
                        question=question,
                        intent="ORDER_INTENT_BUY_SHORT",
                        price=no_price,
                        quantity=shares_ask,
                        strategy=self.name,
                    )
                    if not oid:
                        self.capital_manager.release(self.name, order_size)

            self._last_refresh[slug] = now
            self.log(
                f"Quotes: '{question[:40]}' | "
                f"bid=${our_bid} ask=${our_ask} mid=${mid} | "
                f"spread=${spread_income:.3f} + rebate=${rebate_per_rt:.3f}/RT"
            )
