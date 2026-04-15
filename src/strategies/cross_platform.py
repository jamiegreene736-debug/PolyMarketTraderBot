import os
from src.strategies.base import BaseStrategy
from src.kalshi_client import KalshiClient
from src import fees


class CrossPlatformArbStrategy(BaseStrategy):
    """
    Cross-Platform Arbitrage: Polymarket.us vs Kalshi
    --------------------------------------------------
    Finds the same event priced differently on both platforms.
    Buys the cheaper side and sells (buys NO on) the expensive side.

    Example:
      Polymarket: "Will Fed cut rates in June?" YES @ $0.42
      Kalshi:     Same event YES @ $0.48
      → Buy Polymarket @ $0.42, Buy NO on Kalshi @ $0.52
      → Guaranteed $0.06 profit per share (minus taker fees on both legs)

    Setup:
      1. Create Kalshi account at kalshi.com
      2. Go to Account → API Keys → Generate RSA Key Pair
      3. Download private key, base64-encode it:
           python3 -c "import base64; print(base64.b64encode(open('key.pem','rb').read()).decode())"
      4. Add to Railway Variables:
           KALSHI_API_KEY_ID=your_key_id
           KALSHI_PRIVATE_KEY=base64_encoded_pem
      5. Set cross_platform_arb.enabled: true in config.yaml
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kalshi: KalshiClient | None = None
        self._kalshi_markets_cache: list = []
        self._kalshi_cache_time: float = 0

    def _get_kalshi_client(self) -> KalshiClient | None:
        if self._kalshi:
            return self._kalshi

        key_id     = os.getenv("KALSHI_API_KEY_ID")
        private_key = os.getenv("KALSHI_PRIVATE_KEY")

        if not key_id or not private_key:
            self.log("Missing KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY env vars", level="warning")
            return None

        self._kalshi = KalshiClient(
            key_id=key_id,
            private_key_b64=private_key,
            dry_run=self.client.dry_run,
        )
        return self._kalshi

    async def run(self):
        if not self.enabled:
            return

        min_arb_pct  = self.config.get("min_arb_pct", 0.03)
        order_size   = self.config.get("order_size_usdc", 100)

        kalshi = self._get_kalshi_client()
        if not kalshi:
            return

        # Fetch Kalshi markets (cached for 5 minutes)
        import asyncio
        now = asyncio.get_event_loop().time()
        if (now - self._kalshi_cache_time) > 300:
            try:
                self._kalshi_markets_cache = await kalshi.get_markets(limit=500)
                self._kalshi_cache_time = now
                self.log(f"Fetched {len(self._kalshi_markets_cache)} Kalshi markets")
            except Exception as e:
                self.log(f"Failed to fetch Kalshi markets: {e}", level="error")
                return

        poly_markets = await self.market_data.get_markets()
        if not poly_markets:
            return

        self.log(f"Scanning {len(poly_markets)} Polymarket vs {len(self._kalshi_markets_cache)} Kalshi markets")
        found = 0

        for poly_market in poly_markets:
            poly_slug     = self.market_data.get_slug(poly_market)
            poly_question = self.market_data.get_question(poly_market)

            if not poly_slug or not poly_question:
                continue

            # Find best matching Kalshi market
            best_match  = None
            best_score  = 0.0

            for km in self._kalshi_markets_cache:
                kalshi_title = km.get("title") or km.get("question") or ""
                score = kalshi.match_score(poly_question, kalshi_title)
                if score > best_score:
                    best_score  = score
                    best_match  = km

            if not best_match or best_score < 0.65:
                continue

            kalshi_ticker = best_match.get("ticker")
            kalshi_title  = best_match.get("title", "")

            # Get prices on both sides
            poly_bbo   = await self.market_data.get_bbo(poly_slug)
            kalshi_bbo = await kalshi.get_bbo(kalshi_ticker)

            if not poly_bbo or not kalshi_bbo:
                continue

            try:
                poly_ask   = float(poly_bbo.get("ask", {}).get("price", 1.0))
                poly_bid   = float(poly_bbo.get("bid", {}).get("price", 0.0))
                kalshi_ask = float(kalshi_bbo.get("ask") or 1.0)
                kalshi_bid = float(kalshi_bbo.get("bid") or 0.0)
            except (TypeError, ValueError):
                continue

            shares = round(order_size / max(poly_ask, kalshi_ask, 0.01), 2)

            # Check arb in both directions
            # Direction 1: Buy Polymarket YES, Sell Kalshi YES (buy NO on Kalshi)
            if kalshi_bid > poly_ask:
                profit = fees.arb_profit_after_fees(poly_ask, kalshi_bid, shares)
                if fees.is_arb_profitable(poly_ask, kalshi_bid, min_arb_pct):
                    self.log(
                        f"ARB: Buy Poly @ {poly_ask:.4f}, Sell Kalshi @ {kalshi_bid:.4f} | "
                        f"profit=${profit:.3f} | '{poly_question[:45]}'"
                    )
                    await self._execute_arb(
                        poly_slug=poly_slug, poly_question=poly_question,
                        poly_intent="ORDER_INTENT_BUY_LONG", poly_price=poly_ask,
                        kalshi_ticker=kalshi_ticker, kalshi_action="sell",
                        kalshi_side="yes", kalshi_price=kalshi_bid,
                        shares=shares, order_size=order_size,
                    )
                    found += 1

            # Direction 2: Buy Kalshi YES, Sell Polymarket YES (buy NO on Polymarket)
            elif poly_bid > kalshi_ask:
                profit = fees.arb_profit_after_fees(kalshi_ask, poly_bid, shares)
                if fees.is_arb_profitable(kalshi_ask, poly_bid, min_arb_pct):
                    self.log(
                        f"ARB: Buy Kalshi @ {kalshi_ask:.4f}, Sell Poly @ {poly_bid:.4f} | "
                        f"profit=${profit:.3f} | '{poly_question[:45]}'"
                    )
                    await self._execute_arb(
                        poly_slug=poly_slug, poly_question=poly_question,
                        poly_intent="ORDER_INTENT_BUY_SHORT",
                        poly_price=round(1 - poly_bid, 4),
                        kalshi_ticker=kalshi_ticker, kalshi_action="buy",
                        kalshi_side="yes", kalshi_price=kalshi_ask,
                        shares=shares, order_size=order_size,
                    )
                    found += 1

        if found:
            self.log(f"Executed {found} cross-platform arb(s) this tick")
        else:
            self.log(f"No profitable arb found (threshold={min_arb_pct*100:.1f}% after fees)")

    async def _execute_arb(self, poly_slug, poly_question, poly_intent, poly_price,
                            kalshi_ticker, kalshi_action, kalshi_side, kalshi_price,
                            shares, order_size):
        kalshi = self._get_kalshi_client()
        if not kalshi:
            return

        if not self.capital_manager.can_allocate(self.name, order_size * 2):
            self.log("Not enough capital for arb both legs", level="warning")
            return

        self.capital_manager.allocate(self.name, order_size * 2)

        # Leg 1: Polymarket
        poly_oid = await self.order_manager.place_order(
            market_slug=poly_slug,
            question=poly_question,
            intent=poly_intent,
            price=poly_price,
            quantity=shares,
            strategy=self.name,
        )

        # Leg 2: Kalshi
        try:
            result = await kalshi.place_order(
                ticker=kalshi_ticker,
                action=kalshi_action,
                side=kalshi_side,
                count=int(shares),
                yes_price=kalshi_price,
            )
            self.log(f"Kalshi leg placed: {result}")
        except Exception as e:
            self.log(f"Kalshi leg failed: {e}", level="error")
            if poly_oid:
                await self.order_manager.cancel_order(poly_oid)
            self.capital_manager.release(self.name, order_size * 2)
