import asyncio
from loguru import logger
from polymarket_us import AsyncPolymarketUS


class PolymarketClient:
    """
    Thin async wrapper around the official polymarket-us SDK.
    Adds retry logic, response normalization, and dry_run mode.
    """

    def __init__(self, key_id: str, secret_key: str, dry_run: bool = False):
        self.key_id = key_id
        self.secret_key = secret_key
        self.dry_run = dry_run
        self._client: AsyncPolymarketUS | None = None

    async def connect(self):
        self._client = AsyncPolymarketUS(
            key_id=self.key_id,
            secret_key=self.secret_key,
        )
        logger.info(f"Polymarket client connected (dry_run={self.dry_run})")

    async def close(self):
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass

    # ── Response normalization ────────────────────────────────────────────────

    def _to_dict(self, obj) -> dict:
        """Convert any SDK response object to a plain Python dict."""
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):       # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, "dict"):             # Pydantic v1
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {}

    def _to_list(self, obj) -> list:
        """
        Extract a list of dicts from any SDK markets response.
        Handles: plain list, dict with any common key, Pydantic model, iterable.
        """
        if obj is None:
            return []
        if isinstance(obj, list):
            return [self._to_dict(m) for m in obj]
        if isinstance(obj, dict):
            for key in ("markets", "data", "results", "items", "content"):
                if key in obj and isinstance(obj[key], list):
                    return [self._to_dict(m) for m in obj[key]]
            return []
        # Pydantic model or object — check common attribute names first
        for attr in ("markets", "data", "results", "items", "content"):
            val = getattr(obj, attr, None)
            if val is not None and not callable(val):
                try:
                    return [self._to_dict(m) for m in val]
                except TypeError:
                    pass
        # Last resort: try plain iteration (handles list-like models)
        try:
            items = list(obj)
            if items:
                return [self._to_dict(m) for m in items]
        except TypeError:
            pass
        return []

    def _parse_bbo(self, raw) -> dict:
        """
        Normalize a BBO response to {"bid": {"price": float}, "ask": {"price": float}}.
        Handles all the different formats the SDK might return.
        """
        if raw is None:
            return {}

        d = self._to_dict(raw) if not isinstance(raw, dict) else raw

        # Format 1: {"bid": {"price": "0.45"}, "ask": {"price": "0.55"}}  (already correct)
        if "bid" in d and isinstance(d["bid"], dict):
            return d

        # Format 2: {"bestBid": "0.45", "bestAsk": "0.55"}
        if "bestBid" in d or "bestAsk" in d:
            return {
                "bid": {"price": d.get("bestBid", 0)},
                "ask": {"price": d.get("bestAsk", 1)},
            }

        # Format 3: {"yes_bid": "0.45", "yes_ask": "0.55"}
        if "yes_bid" in d or "yes_ask" in d:
            return {
                "bid": {"price": d.get("yes_bid", 0)},
                "ask": {"price": d.get("yes_ask", 1)},
            }

        # Format 4: flat {"bid": "0.45", "ask": "0.55"}
        if "bid" in d and not isinstance(d["bid"], dict):
            return {
                "bid": {"price": d.get("bid", 0)},
                "ask": {"price": d.get("ask", 1)},
            }

        # Log unknown format once so we can debug further
        logger.debug(f"Unknown BBO format, keys: {list(d.keys())} — raw: {str(d)[:200]}")
        return d

    # ── Retry ─────────────────────────────────────────────────────────────────

    async def _retry(self, coro_fn, retries: int = 3, delay: float = 2.0):
        for attempt in range(retries):
            try:
                return await coro_fn()
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"API call failed (attempt {attempt + 1}/{retries}): {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"API call failed after {retries} attempts: {e}")
                    raise

    # ── Market Data ──────────────────────────────────────────────────────────

    async def get_markets(self, **kwargs) -> list:
        """
        Fetch markets from the SDK.
        The SDK takes parameters as a positional dict, NOT as kwargs.
        Response is a dict with a 'markets' key, or a Pydantic model.
        Paginates to fetch up to 500 markets total.
        """
        all_markets = []
        offset = 0
        limit = 100

        while True:
            params = {"limit": limit, "offset": offset, "active": True, "closed": False}
            try:
                raw = await self._retry(lambda: self._client.markets.list(params))
            except Exception as e:
                logger.warning(f"markets.list(dict) failed: {e} — trying no args")
                try:
                    raw = await self._retry(lambda: self._client.markets.list())
                except Exception as e2:
                    logger.error(f"markets.list() also failed: {e2}")
                    break

            page = self._to_list(raw)
            if not page:
                break
            all_markets.extend(page)
            if len(page) < limit:
                break  # last page
            offset += limit
            if offset >= 500:
                break

        # Filter out closed/inactive
        markets = [
            m for m in all_markets
            if m.get("active") is not False
            and m.get("closed") is not True
            and m.get("archived") is not True
        ]
        logger.info(f"get_markets: fetched {len(all_markets)} total, {len(markets)} active")
        return markets

    async def get_market(self, slug: str) -> dict:
        raw = await self._retry(lambda: self._client.markets.retrieve_by_slug(slug))
        return self._to_dict(raw)

    async def get_order_book(self, slug: str) -> dict:
        raw = await self._retry(lambda: self._client.markets.book(slug))
        return self._to_dict(raw)

    async def get_bbo(self, slug: str) -> dict:
        raw = await self._retry(lambda: self._client.markets.bbo(slug))
        return self._parse_bbo(raw)

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_balance(self) -> dict:
        raw = await self._retry(lambda: self._client.account.balances())
        return self._to_dict(raw)

    async def get_positions(self) -> list:
        raw = await self._retry(lambda: self._client.portfolio.positions())
        return self._to_list(raw)

    async def get_activities(self) -> list:
        raw = await self._retry(lambda: self._client.portfolio.activities())
        return self._to_list(raw)

    # ── Orders ────────────────────────────────────────────────────────────────

    async def get_open_orders(self) -> list:
        raw = await self._retry(lambda: self._client.orders.list())
        return self._to_list(raw)

    async def place_order(self, market_slug: str, intent: str, price: float,
                          quantity: float, order_type: str = "ORDER_TYPE_LIMIT",
                          tif: str = "TIME_IN_FORCE_GOOD_TILL_CANCEL") -> dict:
        if self.dry_run:
            fake_id = f"dry_run_{market_slug}_{intent}_{price}"
            logger.info(f"[DRY RUN] Order: {intent} {quantity} @ ${price} on {market_slug}")
            return {"id": fake_id, "status": "simulated", "dry_run": True}

        payload = {
            "marketSlug": market_slug,
            "intent": intent,
            "type": order_type,
            "price": {"value": str(round(price, 4)), "currency": "USD"},
            "quantity": int(quantity),
            "tif": tif,
        }
        raw = await self._retry(lambda: self._client.orders.create(payload))
        result = self._to_dict(raw)
        logger.info(f"Order placed: {intent} {quantity} @ ${price} on {market_slug} → id={result.get('id')}")
        return result

    async def cancel_order(self, order_id: str) -> bool:
        if self.dry_run:
            logger.info(f"[DRY RUN] Cancel order: {order_id}")
            return True
        try:
            await self._retry(lambda: self._client.orders.cancel(order_id))
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        if self.dry_run:
            logger.info("[DRY RUN] Cancel all orders")
            return True
        try:
            await self._retry(lambda: self._client.orders.cancel_all())
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel all orders: {e}")
            return False
