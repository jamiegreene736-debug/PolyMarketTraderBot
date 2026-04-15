import asyncio
from datetime import datetime, timezone
from loguru import logger
from src.client import PolymarketClient
from src import database as db


class MarketData:
    """
    Fetches and caches market data.
    On first fetch, logs the raw structure of the first market to the
    dashboard so we can see exactly what field names the API uses.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self._markets_cache: list = []
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
                self._markets_cache = await self.client.get_markets()
                self._markets_fetched_at = now

                count = len(self._markets_cache)
                msg = f"Fetched {count} markets from API"
                logger.info(msg)
                await db.log_to_db("INFO", msg)

                # Log first market structure once so we can debug field names
                if not self._diagnosed and count > 0:
                    self._diagnosed = True
                    sample = self._markets_cache[0]
                    keys_msg = f"Market fields: {list(sample.keys())}"
                    vals_msg = f"Market sample: {str(sample)[:400]}"
                    logger.info(keys_msg)
                    logger.info(vals_msg)
                    await db.log_to_db("INFO", keys_msg)
                    await db.log_to_db("INFO", vals_msg)

            except Exception as e:
                msg = f"Failed to fetch markets: {e}"
                logger.error(msg)
                await db.log_to_db("ERROR", msg)

        return self._markets_cache

    async def get_bbo(self, slug: str, force: bool = False) -> dict | None:
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

    async def get_markets_by_volume(self, min_volume: float, top_n: int = 10) -> list:
        markets = await self.get_markets()
        open_markets = [m for m in markets if not self._is_closed(m)]
        filtered = [m for m in open_markets if self._get_volume(m) >= min_volume]
        filtered.sort(key=lambda m: self._get_volume(m), reverse=True)

        msg = (f"Volume filter: {len(open_markets)} open markets, "
               f"{len(filtered)} with volume >= {min_volume}, returning top {min(top_n, len(filtered))}")
        logger.info(msg)
        await db.log_to_db("INFO", msg)

        return filtered[:top_n]

    async def get_markets_resolving_soon(self, max_hours: float, min_volume: float = 0) -> list:
        markets = await self.get_markets()
        result = []
        no_time = 0
        for m in markets:
            if self._is_closed(m):
                continue
            hours_left = self._hours_to_resolution(m)
            if hours_left is None:
                no_time += 1
                continue
            if 0 < hours_left <= max_hours and self._get_volume(m) >= min_volume:
                result.append(m)
        result.sort(key=lambda m: self._hours_to_resolution(m) or 999)

        msg = (f"Resolution filter: {len(result)} markets resolving within {max_hours}h "
               f"({no_time} markets had no resolution time)")
        logger.info(msg)
        await db.log_to_db("INFO", msg)

        return result

    async def get_grouped_markets(self) -> dict[str, list]:
        markets = await self.get_markets()
        groups: dict[str, list] = {}

        # Try every plausible grouping key
        group_keys = ("eventId", "groupId", "seriesId", "event_id",
                      "group_id", "series_id", "eventSlug", "event_slug")

        for m in markets:
            if self._is_closed(m):
                continue
            event_id = None
            for key in group_keys:
                val = m.get(key)
                if val:
                    event_id = val
                    break
            if event_id:
                groups.setdefault(str(event_id), []).append(m)

        valid = {eid: ms for eid, ms in groups.items() if len(ms) >= 2}
        msg = f"Grouped markets: {len(valid)} events with 2+ outcomes"
        logger.info(msg)
        await db.log_to_db("INFO", msg)
        return valid

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_volume(self, market: dict) -> float:
        for key in ("volume24h", "volume", "volumeNum", "dailyVolume",
                    "volume_24h", "daily_volume", "liquidity", "liquidityNum",
                    "totalVolume", "total_volume"):
            val = market.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return 0.0

    def _is_closed(self, market: dict) -> bool:
        for key in ("closed", "resolved", "is_closed", "isResolved",
                    "is_resolved", "active"):
            val = market.get(key)
            if key == "active":
                # active=False means closed
                if val is False:
                    return True
            elif val:
                return True
        return False

    def _hours_to_resolution(self, market: dict) -> float | None:
        for key in ("endDate", "closeTime", "resolutionTime", "expirationDate",
                    "end_date", "close_time", "resolution_time", "expiration_date",
                    "endDateIso", "closingTime", "resolveDate", "resolve_date",
                    "expiresAt", "expires_at", "settlementTime", "settlement_time"):
            val = market.get(key)
            if not val:
                continue
            try:
                if isinstance(val, (int, float)):
                    end = datetime.fromtimestamp(val, tz=timezone.utc)
                else:
                    end = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                now = datetime.now(tz=timezone.utc)
                delta = (end - now).total_seconds() / 3600
                return delta
            except Exception:
                continue
        return None

    def get_slug(self, market: dict) -> str:
        for key in ("slug", "conditionId", "condition_id", "id",
                    "marketSlug", "market_slug", "ticker"):
            val = market.get(key)
            if val:
                return str(val)
        return ""

    def get_question(self, market: dict) -> str:
        for key in ("question", "title", "name", "description", "slug"):
            val = market.get(key)
            if val:
                return str(val)
        return ""
