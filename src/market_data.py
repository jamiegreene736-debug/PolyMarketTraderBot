import asyncio
from datetime import datetime, timezone
from loguru import logger
from src.client import PolymarketClient


class MarketData:
    """
    Fetches and caches market data to reduce API calls.
    Cache TTL: 60 seconds for market list, 10 seconds for order book.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self._markets_cache: list = []
        self._markets_fetched_at: float = 0
        self._book_cache: dict = {}
        self._book_fetched_at: dict = {}
        self._markets_ttl = 60
        self._book_ttl = 10

    async def get_markets(self, force: bool = False) -> list:
        now = asyncio.get_event_loop().time()
        if force or (now - self._markets_fetched_at) > self._markets_ttl:
            try:
                self._markets_cache = await self.client.get_markets()
                self._markets_fetched_at = now
                logger.debug(f"Fetched {len(self._markets_cache)} markets from API")
            except Exception as e:
                logger.error(f"Failed to fetch markets: {e}")
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
        filtered = [
            m for m in markets
            if self._get_volume(m) >= min_volume and not self._is_closed(m)
        ]
        filtered.sort(key=lambda m: self._get_volume(m), reverse=True)
        return filtered[:top_n]

    async def get_markets_resolving_soon(self, max_hours: float, min_volume: float = 0) -> list:
        markets = await self.get_markets()
        result = []
        for m in markets:
            if self._is_closed(m):
                continue
            hours_left = self._hours_to_resolution(m)
            if hours_left is None:
                continue
            if 0 < hours_left <= max_hours and self._get_volume(m) >= min_volume:
                result.append(m)
        result.sort(key=lambda m: self._hours_to_resolution(m) or 999)
        return result

    async def get_grouped_markets(self) -> dict[str, list]:
        """Group markets by event ID for logical arbitrage detection."""
        markets = await self.get_markets()
        groups: dict[str, list] = {}
        for m in markets:
            if self._is_closed(m):
                continue
            event_id = m.get("eventId") or m.get("groupId") or m.get("seriesId")
            if event_id:
                groups.setdefault(str(event_id), []).append(m)
        return {eid: ms for eid, ms in groups.items() if len(ms) >= 2}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_volume(self, market: dict) -> float:
        for key in ("volume24h", "volume", "volumeNum", "dailyVolume"):
            val = market.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return 0.0

    def _is_closed(self, market: dict) -> bool:
        return market.get("closed", False) or market.get("resolved", False)

    def _hours_to_resolution(self, market: dict) -> float | None:
        for key in ("endDate", "closeTime", "resolutionTime", "expirationDate"):
            val = market.get(key)
            if val:
                try:
                    if isinstance(val, str):
                        end = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    else:
                        end = datetime.fromtimestamp(val, tz=timezone.utc)
                    now = datetime.now(tz=timezone.utc)
                    delta = (end - now).total_seconds() / 3600
                    return delta
                except Exception:
                    continue
        return None

    def get_slug(self, market: dict) -> str:
        return market.get("slug") or market.get("conditionId") or market.get("id", "")

    def get_question(self, market: dict) -> str:
        return market.get("question") or market.get("title") or market.get("slug", "")
