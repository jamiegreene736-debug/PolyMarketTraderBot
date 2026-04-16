import asyncio
from loguru import logger
from src.client import PolymarketClient
from src import database as db


class OrderManager:
    """
    Manages all order placement, tracking, deduplication, and cancellation.
    Prevents duplicate orders and enforces rate limiting.
    """

    def __init__(self, client: PolymarketClient, max_concurrent: int = 20):
        self.client = client
        self.max_concurrent = max_concurrent
        self._open_orders: dict[str, dict] = {}     # order_id -> order info
        self._market_orders: dict[str, list] = {}   # market_slug -> [order_ids]
        self._lock = asyncio.Lock()
        self._request_times: list[float] = []
        self._rate_limit = 8                         # max requests per second

    async def place_order(self, market_slug: str, question: str, intent: str,
                          price: float, quantity: float, strategy: str,
                          tif: str = "TIME_IN_FORCE_GOOD_TILL_CANCEL") -> str | None:
        async with self._lock:
            if len(self._open_orders) >= self.max_concurrent:
                logger.warning(f"Max concurrent orders ({self.max_concurrent}) reached, skipping")
                return None

            if self._is_duplicate(market_slug, intent, price):
                logger.debug(f"Duplicate order skipped: {intent} @ {price} on {market_slug}")
                return None

        await self._rate_limit_wait()

        try:
            result = await self.client.place_order(
                market_slug=market_slug,
                intent=intent,
                price=price,
                quantity=quantity,
                tif=tif,
            )
            order_id = result.get("id")
            if not order_id:
                msg = f"[order] No order_id returned for {intent} @ ${price:.4f} on {market_slug} — result={result}"
                logger.warning(msg)
                await db.log_to_db("WARNING", msg)
                return None

            async with self._lock:
                self._open_orders[order_id] = {
                    "order_id": order_id,
                    "market_slug": market_slug,
                    "intent": intent,
                    "price": price,
                    "quantity": quantity,
                    "strategy": strategy,
                }
                self._market_orders.setdefault(market_slug, []).append(order_id)

            msg = f"[order] PLACED {intent} {quantity:.1f}x @ ${price:.4f} on '{question[:40]}' id={order_id}"
            logger.info(msg)
            await db.log_to_db("INFO", msg)

            await db.insert_trade(
                strategy=strategy,
                market_slug=market_slug,
                question=question,
                side=intent,
                price=price,
                quantity=quantity,
                order_id=order_id,
            )
            return order_id

        except Exception as e:
            msg = f"[order] FAILED {intent} @ ${price:.4f} on {market_slug}: {e}"
            logger.error(msg)
            await db.log_to_db("ERROR", msg)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        async with self._lock:
            order = self._open_orders.get(order_id)
            market_slug = order["market_slug"] if order else ""
        success = await self.client.cancel_order(order_id, market_slug)
        if success:
            async with self._lock:
                order = self._open_orders.pop(order_id, None)
                if order:
                    slug = order["market_slug"]
                    if slug in self._market_orders:
                        self._market_orders[slug] = [
                            oid for oid in self._market_orders[slug] if oid != order_id
                        ]
            await db.cancel_trade(order_id)
        return success

    async def cancel_market_orders(self, market_slug: str):
        order_ids = list(self._market_orders.get(market_slug, []))
        for oid in order_ids:
            await self.cancel_order(oid)

    async def cancel_stale_orders(self, market_slug: str, current_mid: float, max_drift: float = 0.05):
        """Cancel orders that have drifted more than max_drift from current mid price."""
        order_ids = list(self._market_orders.get(market_slug, []))
        for oid in order_ids:
            order = self._open_orders.get(oid)
            if order and abs(order["price"] - current_mid) > max_drift:
                logger.debug(f"Cancelling stale order {oid} (price={order['price']}, mid={current_mid})")
                await self.cancel_order(oid)

    async def mark_filled(self, order_id: str, pnl: float = 0.0):
        async with self._lock:
            order = self._open_orders.pop(order_id, None)
            if order:
                slug = order["market_slug"]
                if slug in self._market_orders:
                    self._market_orders[slug] = [
                        oid for oid in self._market_orders[slug] if oid != order_id
                    ]
        await db.close_trade(order_id, pnl)

    async def sync_from_exchange(self) -> int:
        """
        On bot startup, fetch all open orders from the exchange and repopulate
        _open_orders / _market_orders so position tracking survives restarts.

        Fields from the exchange are normalised; strategy and question are
        back-filled from the local DB where available.

        Returns the number of orders synced.
        """
        try:
            raw_orders = await self.client.get_open_orders()
        except Exception as e:
            logger.warning(f"sync_from_exchange: could not fetch open orders: {e}")
            return 0

        if not raw_orders:
            logger.info("sync_from_exchange: no open orders on exchange")
            return 0

        # Load DB metadata so we can restore strategy/question
        try:
            db_meta = await db.get_open_trades_metadata()
        except Exception:
            db_meta = {}

        synced = 0
        async with self._lock:
            for o in raw_orders:
                # Normalise id
                order_id = (o.get("id") or o.get("orderId") or
                            o.get("order_id") or "")
                if not order_id:
                    continue

                # Normalise market slug
                slug = (o.get("marketSlug") or o.get("market_slug") or
                        o.get("slug") or o.get("conditionId") or "")

                # Normalise intent / side
                intent = (o.get("intent") or o.get("side") or
                          o.get("orderType") or "ORDER_INTENT_BUY_LONG")

                # Normalise price — can be float, str, or nested {"value": "0.97"}
                raw_price = o.get("price", 0)
                if isinstance(raw_price, dict):
                    raw_price = raw_price.get("value", 0)
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    price = 0.0

                # Normalise quantity
                try:
                    quantity = float(o.get("quantity") or o.get("size") or 0)
                except (TypeError, ValueError):
                    quantity = 0.0

                # Back-fill from DB if we have a record
                meta = db_meta.get(order_id, {})
                strategy = meta.get("strategy", "synced")
                question  = meta.get("question", slug)

                # Skip if already tracked (e.g. bot placed it this session)
                if order_id in self._open_orders:
                    continue

                self._open_orders[order_id] = {
                    "order_id": order_id,
                    "market_slug": slug,
                    "intent": intent,
                    "price": price,
                    "quantity": quantity,
                    "strategy": strategy,
                    "question": question,
                }
                if slug:
                    self._market_orders.setdefault(slug, []).append(order_id)
                synced += 1

        logger.info(f"sync_from_exchange: synced {synced} open order(s) from exchange "
                    f"(total tracked: {len(self._open_orders)})")
        await db.log_to_db("INFO",
            f"[order_manager] Startup sync: {synced} open orders restored from exchange")
        return synced

    def get_open_positions(self, exclude_strategies: list[str] | None = None) -> list[dict]:
        """Return a snapshot of all tracked open positions, optionally filtering out strategies."""
        exclude = set(exclude_strategies or [])
        return [
            dict(order)
            for order in self._open_orders.values()
            if order.get("strategy") not in exclude
        ]

    def get_market_order_count(self, market_slug: str) -> int:
        return len(self._market_orders.get(market_slug, []))

    def get_total_open_orders(self) -> int:
        return len(self._open_orders)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _is_duplicate(self, market_slug: str, intent: str, price: float) -> bool:
        for order in self._open_orders.values():
            if (order["market_slug"] == market_slug
                    and order["intent"] == intent
                    and abs(order["price"] - price) < 0.01):
                return True
        return False

    async def _rate_limit_wait(self):
        now = asyncio.get_event_loop().time()
        self._request_times = [t for t in self._request_times if now - t < 1.0]
        if len(self._request_times) >= self._rate_limit:
            sleep_time = 1.0 - (now - self._request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self._request_times.append(asyncio.get_event_loop().time())
