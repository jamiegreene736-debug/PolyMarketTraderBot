import asyncio
from datetime import datetime, timezone
from loguru import logger
from src.client import PolymarketClient
from src import database as db


class MarketData:
    """
    Fetches and caches market data from Polymarket.us.

    Key API facts learned from diagnostic:
      - Fields: id, question, slug, endDate, active, closed, archived,
                marketType, outcomes, outcomePrices, marketSides, feeCoefficient
      - Prices come from outcomePrices (list of floats), NOT a BBO endpoint
      - Volume field does NOT exist — all active markets are treated equally
      - endDate is an ISO string: "2025-11-03T13:00:00Z"
      - active=False or closed=True means market is inactive
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self._markets_cache: list = []
        self._market_by_slug: dict = {}          # slug → market, rebuilt on each fetch
        self._markets_fetched_at: float = 0
        self._book_cache: dict = {}
        self._book_fetched_at: dict = {}
        self._markets_ttl = 60
        self._book_ttl = 10
        self._diagnosed = False

    async def get_markets(self, force: bool = False) -> list:
        now = asyncio.get_event_loop().time()
        if force or (now - self._markets_fetched_at) > self._markets_ttl:
            try:
                raw = await self.client.get_markets()
                if not raw and self._markets_cache:
                    msg = (
                        f"Market refresh returned no data; preserving "
                        f"{len(self._markets_cache)} cached active market(s)"
                    )
                    logger.warning(msg)
                    await db.log_to_db("WARNING", msg)
                    self._markets_fetched_at = now
                    return self._markets_cache

                # Filter to only active, non-closed, non-archived markets
                self._markets_cache = [
                    m for m in raw
                    if not self._is_closed(m)
                ]
                # Rebuild slug→market index so get_bbo() is O(1) instead of
                # scanning the whole list on every call (kills quadratic blowup
                # when strategies fetch BBO for hundreds of markets per tick).
                self._market_by_slug = {
                    slug: m for m in self._markets_cache
                    if (slug := self.get_slug(m))
                }
                self._markets_fetched_at = now

                msg = f"Fetched {len(raw)} markets total, {len(self._markets_cache)} active"
                logger.info(msg)
                await db.log_to_db("INFO", msg)

                if not self._diagnosed:
                    self._diagnosed = True
                    if self._markets_cache:
                        sample = self._markets_cache[0]
                        await db.log_to_db("INFO", f"Active market sample keys: {list(sample.keys())}")
                        await db.log_to_db("INFO", f"Active market sample: {str(sample)[:500]}")
                    elif raw:
                        # Got markets but all filtered out — show why
                        sample = raw[0]
                        await db.log_to_db("INFO", f"ALL FILTERED OUT. Sample keys: {list(sample.keys())}")
                        await db.log_to_db("INFO", f"Sample active={sample.get('active')} closed={sample.get('closed')} archived={sample.get('archived')} endDate={sample.get('endDate')}")
                    else:
                        await db.log_to_db("WARNING", "get_markets returned empty list — check SDK response parsing")

            except Exception as e:
                msg = f"Failed to fetch markets: {e}"
                logger.error(msg)
                await db.log_to_db("ERROR", msg)

        return self._markets_cache

    async def get_bbo(self, slug: str, force: bool = False) -> dict | None:
        """
        For Polymarket.us, prices come from outcomePrices on the market object.
        This method wraps that lookup in the same interface the strategies expect.
        Falls back to the API bbo endpoint if the market isn't cached.
        """
        # O(1) cache lookup via slug index — built when markets are fetched.
        cached = self._market_by_slug.get(slug)
        if cached is not None:
            return self._bbo_from_market(cached)

        # Fall back to BBO API endpoint
        now = asyncio.get_event_loop().time()
        last = self._book_fetched_at.get(slug, 0)
        if force or (now - last) > self._book_ttl:
            try:
                self._book_cache[slug] = await self.client.get_bbo(slug)
                self._book_fetched_at[slug] = now
            except Exception as e:
                logger.warning(f"Failed to fetch BBO for {slug}: {e}")
                return None
        return self._book_cache.get(slug)

    def _bbo_from_market(self, market: dict) -> dict | None:
        """
        Build a BBO dict from a market object.
        Prefers real bestBid/bestAsk fields; falls back to outcomePrices ± spread.
        """
        # Use actual BBO fields if present
        best_bid = market.get("bestBid") or market.get("best_bid")
        best_ask = market.get("bestAsk") or market.get("best_ask")
        if best_bid is not None and best_ask is not None:
            try:
                bid = float(best_bid)
                ask = float(best_ask)
                if 0 < bid < ask < 1:
                    return {
                        "bid": {"price": bid},
                        "ask": {"price": ask},
                        "mid": round((bid + ask) / 2, 4),
                    }
            except (TypeError, ValueError):
                pass

        # Fall back to outcomePrices (YES price = prices[0])
        prices = self.get_outcome_prices(market)
        if not prices:
            return None
        yes_price = prices[0]
        # Use a 2-cent spread — aggressive enough for taker fills on near-certainty
        spread = 0.01
        return {
            "bid": {"price": max(0.01, yes_price - spread)},
            "ask": {"price": min(0.99, yes_price + spread)},
            "mid": yes_price,
        }

    async def get_markets_by_volume(self, min_volume: float, top_n: int = 10) -> list:
        markets = await self.get_markets()
        open_markets = [m for m in markets if not self._is_closed(m)]

        # Sort by volume descending (uses real volume24hr/volumeNum fields)
        open_markets.sort(key=lambda m: self._get_volume(m), reverse=True)

        # Apply minimum volume filter if any markets have real volume data
        if min_volume > 0:
            has_volume = [m for m in open_markets if self._get_volume(m) >= min_volume]
            if has_volume:
                open_markets = has_volume

        result = open_markets[:top_n]
        top_vols = [(self.get_question(m)[:30], round(self._get_volume(m))) for m in result[:3]]
        msg = f"Volume filter: {len(open_markets)} open markets, top {len(result)} — top3={top_vols}"
        logger.info(msg)
        await db.log_to_db("INFO", msg)

        return result

    async def get_markets_resolving_soon(self, max_hours: float, min_volume: float = 0) -> list:
        markets = await self.get_markets()
        result = []
        no_time = 0
        for m in markets:
            hours_left = self._hours_to_resolution(m)
            if hours_left is None:
                no_time += 1
                continue
            if 0 < hours_left <= max_hours:
                result.append(m)
        result.sort(key=lambda m: self._hours_to_resolution(m) or 999)

        msg = (f"Resolution filter: {len(result)} markets resolving within {max_hours}h "
               f"({no_time} had no resolution time)")
        logger.info(msg)
        await db.log_to_db("INFO", msg)
        return result

    async def get_resolution_window_stats(self, max_hours: float) -> dict:
        markets = await self.get_markets()
        no_time = 0
        with_time = 0
        within_window = 0

        for m in markets:
            hours_left = self._hours_to_resolution(m)
            if hours_left is None:
                no_time += 1
                continue
            with_time += 1
            if 0 < hours_left <= max_hours:
                within_window += 1

        return {
            "active_markets": len(markets),
            "with_resolution_time": with_time,
            "missing_resolution_time": no_time,
            "within_window": within_window,
            "max_hours": max_hours,
        }

    async def get_grouped_markets(self) -> dict[str, list]:
        """
        For Polymarket.us, multi-outcome markets have multiple entries in
        outcomePrices. We treat each market with 2+ outcomes as its own group
        for logical arbitrage — checking if the prices sum to ~1.0.
        """
        markets = await self.get_markets()
        groups: dict[str, list] = {}

        for m in markets:
            # Group by category + question prefix for related markets
            event_id = (m.get("eventId") or m.get("groupId") or
                        m.get("seriesId") or m.get("event_id"))
            if event_id:
                groups.setdefault(str(event_id), []).append(m)
                continue

            # Also treat each multi-outcome market as its own self-contained group
            prices = self.get_outcome_prices(m)
            if len(prices) >= 2:
                slug = self.get_slug(m)
                if slug:
                    groups[f"multi_{slug}"] = [m]  # handled specially in logical_arb

        valid = {eid: ms for eid, ms in groups.items() if len(ms) >= 1}

        msg = f"Grouped markets: {len(valid)} groups found"
        logger.info(msg)
        await db.log_to_db("INFO", msg)
        return valid

    async def get_all_multi_outcome_markets(self) -> list:
        """Returns all active markets that have 2+ outcome prices."""
        markets = await self.get_markets()
        return [m for m in markets if len(self.get_outcome_prices(m)) >= 2]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_outcome_prices(self, market: dict) -> list[float]:
        """Extract outcome prices as floats from a market object."""
        raw = market.get("outcomePrices") or []
        if isinstance(raw, str):
            import json
            try:
                raw = json.loads(raw)
            except Exception:
                return []
        result = []
        for p in raw:
            try:
                result.append(float(p))
            except (TypeError, ValueError):
                pass
        return result

    def get_outcomes(self, market: dict) -> list[str]:
        """Get outcome names (e.g. ['Los Angeles', 'Tennessee'])."""
        raw = market.get("outcomes") or []
        if isinstance(raw, str):
            import json
            try:
                raw = json.loads(raw)
            except Exception:
                return []
        return [str(o) for o in raw]

    def _is_closed(self, market: dict) -> bool:
        if market.get("closed") is True:
            return True
        if market.get("active") is False:
            return True
        if market.get("archived") is True:
            return True
        # Check if endDate is in the past
        hours = self._hours_to_resolution(market)
        if hours is not None and hours <= 0:
            return True
        return False

    def get_market_liquidity(self, slug: str) -> float:
        """
        Return the liquidity (sum of resting order-book depth, in USDC) for a
        market. Returns 0.0 if the market isn't cached or has no liquidity field.
        Used as a cheap pre-trade depth gate.
        """
        m = self._market_by_slug.get(slug)
        if not m:
            return 0.0
        for key in ("liquidityNum", "liquidity", "liquidityUSDC"):
            val = m.get(key)
            if val is not None:
                try:
                    v = float(val)
                    if v >= 0:
                        return v
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _get_volume(self, market: dict) -> float:
        for key in ("volume24hr", "volume1wk", "volumeNum", "volume",
                    "volume24h", "daily_volume", "liquidity", "liquidityNum"):
            val = market.get(key)
            if val is not None:
                try:
                    v = float(val)
                    if v > 0:
                        return v
                except (ValueError, TypeError):
                    pass
        return 0.0

    def _hours_to_resolution(self, market: dict) -> float | None:
        for key in ("endDate", "closeTime", "resolutionTime", "expirationDate",
                    "end_date", "close_time", "gameStartTime", "closingTime"):
            val = market.get(key)
            if not val:
                continue
            try:
                if isinstance(val, (int, float)):
                    end = datetime.fromtimestamp(val, tz=timezone.utc)
                else:
                    end = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                now = datetime.now(tz=timezone.utc)
                return (end - now).total_seconds() / 3600
            except Exception:
                continue
        return None

    def get_slug(self, market: dict) -> str:
        for key in ("slug", "id", "conditionId", "condition_id", "marketSlug"):
            val = market.get(key)
            if val:
                return str(val)
        return ""

    def get_question(self, market: dict) -> str:
        for key in ("question", "title", "name", "description"):
            val = market.get(key)
            if val:
                return str(val)
        return ""
