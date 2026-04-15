import asyncio
from loguru import logger
from polymarket_us import AsyncPolymarketUS


class PolymarketClient:
    """
    Thin async wrapper around the official polymarket-us SDK.
    Adds retry logic, error handling, and dry_run mode.
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
            await self._client.__aexit__(None, None, None)

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
        return await self._retry(lambda: self._client.markets.list(**kwargs))

    async def get_market(self, slug: str) -> dict:
        return await self._retry(lambda: self._client.markets.retrieve_by_slug(slug))

    async def get_order_book(self, slug: str) -> dict:
        return await self._retry(lambda: self._client.markets.book(slug))

    async def get_bbo(self, slug: str) -> dict:
        """Best bid and offer for a market."""
        return await self._retry(lambda: self._client.markets.bbo(slug))

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_balance(self) -> dict:
        return await self._retry(lambda: self._client.account.balances())

    async def get_positions(self) -> list:
        return await self._retry(lambda: self._client.portfolio.positions())

    async def get_activities(self) -> list:
        return await self._retry(lambda: self._client.portfolio.activities())

    # ── Orders ────────────────────────────────────────────────────────────────

    async def get_open_orders(self) -> list:
        return await self._retry(lambda: self._client.orders.list())

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
        result = await self._retry(lambda: self._client.orders.create(payload))
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
