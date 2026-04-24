import asyncio
import time
from loguru import logger
from src.client import PolymarketClient
from src import database as db
from py_clob_client.order_builder.constants import BUY


# Polymarket CLOB enforces a 5-share minimum per order. Orders below this
# are silently rejected by the exchange — we catch them locally and log
# a clear warning so the user can size up or fund more capital.
MIN_ORDER_SHARES = 5


class OrderManager:
    """
    Manages all order placement, tracking, deduplication, and cancellation.
    Prevents duplicate orders and enforces rate limiting.
    """

    def __init__(self, client: PolymarketClient, max_concurrent: int = 20,
                 market_data=None, min_liquidity_multiple: float = 3.0):
        self.client = client
        self.max_concurrent = max_concurrent
        self.market_data = market_data               # optional — used for liquidity gate
        self.min_liquidity_multiple = min_liquidity_multiple
        self.capital_manager = None                  # optional — attached after startup balance loads
        self._open_orders: dict[str, dict] = {}     # order_id -> order info
        self._market_orders: dict[str, list] = {}   # market_slug -> [order_ids]
        self._lock = asyncio.Lock()
        self._request_times: list[float] = []
        self._rate_limit = 8                         # max requests per second
        self.last_order_status = ""

    def attach_capital_manager(self, capital_manager):
        self.capital_manager = capital_manager

    async def place_order(self, market_slug: str, question: str, intent: str,
                          price: float, quantity: float, strategy: str,
                          execution_side: str = BUY,
                          tif: str = "TIME_IN_FORCE_GOOD_TILL_CANCEL") -> str | None:
        execution_side = str(execution_side or BUY).upper()
        is_buy_order = execution_side != "SELL"
        self.last_order_status = ""

        async with self._lock:
            if is_buy_order and len(self._open_orders) >= self.max_concurrent:
                logger.warning(f"Max concurrent orders ({self.max_concurrent}) reached, skipping")
                return None

            if is_buy_order and self._is_duplicate(market_slug, intent, price, strategy):
                logger.debug(f"Duplicate order skipped: {intent} @ {price} on {market_slug}")
                return None

        # Polymarket minimum order size guard. Reject before hitting the API
        # rather than letting the exchange silently drop the order.
        if quantity < MIN_ORDER_SHARES:
            notional_needed = MIN_ORDER_SHARES * price
            msg = (f"[order] SKIP {intent} on {market_slug}: qty={quantity:.2f} shares "
                   f"< Polymarket minimum ({MIN_ORDER_SHARES}). At price ${price:.4f} "
                   f"you need ${notional_needed:.2f} notional — raise order_size_usdc "
                   f"or fund more capital.")
            logger.warning(msg)
            await db.log_to_db("WARNING", msg)
            return None

        notional = max(0.0, float(price or 0.0) * float(quantity or 0.0))
        if is_buy_order:
            available_for_orders = self.get_available_order_capacity_usdc()
            if available_for_orders is not None and notional > available_for_orders:
                msg = (
                    f"[order] WAIT {intent} on {market_slug}: notional ${notional:.2f} "
                    f"exceeds exchange-safe capacity ${available_for_orders:.2f} "
                    f"(tracked active orders=${self.get_open_order_notional_usdc():.2f})."
                )
                logger.info(msg)
                await db.log_to_db("INFO", msg)
                return None

        # Liquidity gate: skip thin markets where we can't get in/out without
        # moving price against ourselves. Uses the market's `liquidity` field
        # as a proxy for total book depth.
        if is_buy_order and self.market_data is not None and self.min_liquidity_multiple > 0:
            liquidity = self.market_data.get_market_liquidity(market_slug)
            required = notional * self.min_liquidity_multiple
            # Skip the check if the market has no liquidity metadata at all
            # (some Polymarket markets don't expose it, and we'd rather trade).
            if liquidity > 0 and liquidity < required:
                msg = (f"[order] SKIP {intent} on {market_slug}: thin market "
                       f"(liquidity=${liquidity:.0f} < {self.min_liquidity_multiple}× "
                       f"notional ${notional:.2f} = ${required:.2f}).")
                logger.warning(msg)
                await db.log_to_db("WARNING", msg)
                return None

        await self._rate_limit_wait()

        try:
            result = await self.client.place_order(
                market_slug=market_slug,
                intent=intent,
                price=price,
                quantity=quantity,
                side=execution_side,
                tif=tif,
            )
            order_status = str(result.get("status") or "").lower()
            if order_status == "no_match":
                self.last_order_status = "no_match"
                order_type = str(result.get("order_type") or "").upper()
                msg = (
                    f"[order] NO_FILL {intent} {execution_side} {quantity:.1f}x "
                    f"@ ${price:.4f} type={order_type} on '{question[:40]}' — "
                    "no matching liquidity"
                )
                logger.info(msg)
                await db.log_to_db("INFO", msg)
                return None

            order_id = result.get("id")
            if not order_id:
                self.last_order_status = "no_order_id"
                msg = f"[order] No order_id returned for {intent} @ ${price:.4f} on {market_slug} — result={result}"
                logger.warning(msg)
                await db.log_to_db("WARNING", msg)
                return None

            order_type = str(result.get("order_type") or "").upper()
            is_resting_order = order_type not in {"FOK", "FAK"}
            if not is_resting_order:
                self.last_order_status = "submitted"
                msg = (
                    f"[order] SUBMITTED {intent} {execution_side} {quantity:.1f}x "
                    f"@ ${price:.4f} type={order_type} on '{question[:40]}' id={order_id}"
                )
                logger.info(msg)
                await db.log_to_db("INFO", msg)
                return order_id

            async with self._lock:
                self._open_orders[order_id] = {
                    "order_id": order_id,
                    "market_slug": market_slug,
                    "asset_id": result.get("asset_id") or "",
                    "intent": intent,
                    "execution_side": execution_side,
                    "price": price,
                    "quantity": quantity,
                    "strategy": strategy,
                    "placed_at": time.time(),
                }
                self._market_orders.setdefault(market_slug, []).append(order_id)

            msg = (
                f"[order] PLACED {intent} {execution_side} {quantity:.1f}x "
                f"@ ${price:.4f} on '{question[:40]}' id={order_id}"
            )
            logger.info(msg)
            await db.log_to_db("INFO", msg)

            await db.insert_trade(
                strategy=strategy,
                market_slug=market_slug,
                question=question,
                side=intent,
                execution_side=execution_side,
                price=price,
                quantity=quantity,
                order_id=order_id,
            )
            self.last_order_status = "placed"
            return order_id

        except Exception as e:
            self.last_order_status = "error"
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

    async def cancel_stale_orders(
        self,
        market_slug: str,
        current_mid: float,
        max_drift: float = 0.05,
    ):
        """
        Cancel orders that have drifted too far from the current fair price.

        BUY_LONG orders are priced in YES terms, so we compare them to the YES mid.
        BUY_SHORT orders are priced in NO terms, so we compare them to the NO mid.
        """
        order_ids = list(self._market_orders.get(market_slug, []))
        for oid in order_ids:
            order = self._open_orders.get(oid)
            if not order:
                continue

            intent = str(order.get("intent") or "")
            reference_price = current_mid
            if intent == "ORDER_INTENT_BUY_SHORT":
                reference_price = max(0.01, min(0.99, round(1.0 - current_mid, 4)))

            if abs(order["price"] - reference_price) > max_drift:
                logger.debug(
                    f"Cancelling stale order {oid} "
                    f"(price={order['price']}, ref={reference_price}, intent={intent})"
                )
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
            msg = f"sync_from_exchange: could not fetch open orders: {e}"
            logger.warning(msg)
            await db.log_to_db("WARNING", msg)
            raise RuntimeError(msg) from e

        if not raw_orders:
            msg = "sync_from_exchange: no open orders on exchange"
            logger.info(msg)
            await db.log_to_db("INFO", "[order_manager] Startup sync: no open orders on exchange")
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

                meta = db_meta.get(order_id, {})

                # Normalise market slug
                raw_slug = (o.get("marketSlug") or o.get("market_slug") or
                            o.get("slug") or o.get("conditionId") or "")
                slug = meta.get("market_slug") or raw_slug

                # Intent is the outcome token direction (YES/NO). Execution
                # side is whether this open order is buying or selling that
                # token. Keeping them separate prevents resting exits from
                # being treated as new entries after a restart.
                intent = (
                    meta.get("side") or o.get("intent") or o.get("orderType")
                    or "ORDER_INTENT_BUY_LONG"
                )
                execution_side = str(
                    o.get("execution_side") or meta.get("execution_side") or BUY
                ).upper()

                # Normalise price — can be float, str, or nested {"value": "0.97"}
                raw_price = o.get("price", 0)
                if isinstance(raw_price, dict):
                    raw_price = raw_price.get("value", 0)
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    price = 0.0

                # Normalise quantity
                quantity = self._remaining_order_size(o)

                # Back-fill from DB if we have a record
                strategy = meta.get("strategy", "synced")
                question  = meta.get("question", slug)

                # Skip if already tracked (e.g. bot placed it this session)
                if order_id in self._open_orders:
                    continue

                # Restore placed_at from DB timestamp if available
                db_ts = meta.get("timestamp")
                if db_ts:
                    try:
                        from datetime import datetime, timezone
                        placed_at = datetime.fromisoformat(db_ts).replace(
                            tzinfo=timezone.utc).timestamp()
                    except Exception:
                        placed_at = time.time()
                else:
                    placed_at = time.time()

                self._open_orders[order_id] = {
                    "order_id": order_id,
                    "market_slug": slug,
                    "asset_id": o.get("asset_id") or "",
                    "intent": intent,
                    "execution_side": execution_side,
                    "price": price,
                    "quantity": quantity,
                    "strategy": strategy,
                    "question": question,
                    "placed_at": placed_at,
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

    def get_pending_exit_order(self, market_slug: str, intent: str) -> dict | None:
        """Return an existing open SELL order for the same market/outcome, if any."""
        target_slug = str(market_slug or "").strip()
        target_intent = str(intent or "").strip()
        for order in self._open_orders.values():
            if str(order.get("execution_side") or "BUY").upper() != "SELL":
                continue
            if str(order.get("market_slug") or "").strip() != target_slug:
                continue
            if str(order.get("intent") or "").strip() != target_intent:
                continue
            return dict(order)
        return None

    def get_market_order_count(self, market_slug: str) -> int:
        return len(self._market_orders.get(market_slug, []))

    def get_total_open_orders(self) -> int:
        return len(self._open_orders)

    def get_open_order_notional_usdc(self) -> float:
        total = 0.0
        for order in self._open_orders.values():
            side = str(order.get("execution_side") or "BUY").upper()
            if side == "SELL":
                continue
            try:
                total += float(order.get("price") or 0.0) * float(order.get("quantity") or 0.0)
            except (TypeError, ValueError):
                continue
        return max(0.0, total)

    def get_available_order_capacity_usdc(self) -> float | None:
        if self.capital_manager is None:
            return None

        reserve = self.capital_manager.total_usdc * (self.capital_manager.reserve_pct / 100)
        tradeable = max(0.0, self.capital_manager.total_usdc - reserve)
        return max(0.0, tradeable - self.get_open_order_notional_usdc())

    def clear(self):
        """Wipe all in-memory position state. Called by Reset Data from the dashboard."""
        self._open_orders.clear()
        self._market_orders.clear()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _is_duplicate(self, market_slug: str, intent: str, price: float,
                      strategy: str) -> bool:
        """
        Dedup logic:
          - Within a single strategy: block if same slug + same intent + price within 1¢
            (preserves legacy behaviour — lets market_making re-quote at new prices).
          - Across different strategies: block ANY same slug + same intent, regardless
            of price. Prevents pile-on where near_certainty + whale_tracker + ai_trader
            all buy the same market on the same tick and triple the intended exposure.
        """
        for order in self._open_orders.values():
            if order["market_slug"] != market_slug:
                continue
            if order["intent"] != intent:
                continue
            if order.get("strategy") == strategy:
                if abs(order["price"] - price) < 0.01:
                    return True
            else:
                # Different strategy — block same-slug-same-side pile-on outright.
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

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _remaining_order_size(cls, order: dict) -> float:
        for key in ("remaining_size", "remainingSize", "size_remaining", "sizeRemaining"):
            remaining = cls._safe_float(order.get(key), -1.0)
            if remaining >= 0:
                return remaining

        size = cls._safe_float(order.get("quantity") or order.get("size"), -1.0)
        if size >= 0:
            return size

        original = cls._safe_float(
            order.get("original_size") or order.get("originalSize"),
            0.0,
        )
        matched = cls._safe_float(
            order.get("size_matched") or order.get("sizeMatched"),
            0.0,
        )
        if original > 0:
            return max(0.0, original - matched)
        return 0.0
