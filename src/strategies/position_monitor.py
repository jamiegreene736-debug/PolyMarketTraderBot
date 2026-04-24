"""
Position Monitor — Auto Take-Profit / Stop-Loss / Max Hold Time
---------------------------------------------------------------
Runs every tick and checks every open directional position against:
  1. Take-profit threshold
  2. Stop-loss threshold
  3. Maximum hold time (per strategy) — exits at market price when expired

TP/SL logic:
  LONG (BUY_LONG / YES position)
    TP: current_bid >= entry_price × (1 + tp_pct)  OR  current_bid >= 0.99 (near-certainty cap)
    SL: current_bid <= entry_price × (1 − sl_pct)
    Exit: place BUY_SHORT at current ask

  SHORT (BUY_SHORT / NO position)
    entry_no  = entry_price
    no_bid    = 1 − current_yes_ask
    TP: no_bid >= entry_no × (1 + tp_pct)  OR  no_bid >= 0.99
    SL: no_bid <= entry_no × (1 − sl_pct)
    Exit: place BUY_LONG at current bid

Max hold time:
  If a position has been open longer than the strategy's max_hold_hours,
  it is exited at market price regardless of TP/SL. This prevents trades
  from sitting open for days when the edge window has closed.
"""

import asyncio
import time
import re
from datetime import datetime
from src.strategies.base import BaseStrategy
from src import database as db
from py_clob_client.order_builder.constants import SELL


# Strategies whose orders are managed elsewhere (not TP/SL monitored).
# position_monitor is excluded from its own scan so it doesn't recursively
# try to "exit" the exit orders it just placed. Exchange-backed "live position"
# rows are real positions and should still be auto-exited with the default
# max-hold policy after a restart.
_EXCLUDED_STRATEGIES = {"market_making", "position_monitor"}

# Near-certainty strategies use an absolute TP near $1.00, not pct-based
_NEAR_CERTAINTY_STRATEGIES = {"near_certainty", "inverted_near_certainty"}
_NEAR_CERTAINTY_TP_THRESHOLD = 0.99   # exit when price reaches $0.99
_NEAR_CERTAINTY_INFER_THRESHOLD = 0.93

# Smart force-exit tuning. First attempt at MAX_HOLD tries a passive limit
# at the fair price (no overpay). After this many seconds of unfilled
# passive attempts, we escalate to aggressive taker.
_SOFT_EXIT_GRACE_SECONDS = 180        # 3 minutes


class PositionMonitorStrategy(BaseStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # attempt_key → (first_attempt_ts, attempt_count)
        self._exit_attempts: dict[str, tuple[float, int]] = {}

    def _normalize_market_key(self, value: str | None) -> str:
        raw = (value or "").strip().lower()
        if not raw:
            return ""
        raw = raw.replace("–", "-").replace("—", "-")
        return re.sub(r"[^a-z0-9]+", "", raw)

    def _infer_strategy(self, outcome: str, avg_price: float) -> str:
        if outcome == "YES" and avg_price >= _NEAR_CERTAINTY_INFER_THRESHOLD:
            return "near_certainty"
        if outcome == "NO" and avg_price >= _NEAR_CERTAINTY_INFER_THRESHOLD:
            return "inverted_near_certainty"
        return "live position"

    async def _build_managed_positions(self) -> list[dict]:
        user_address = (self.client.funder_address or self.client.signer_address or "").strip()
        raw_positions = await self.client.get_positions(user=user_address)
        if not raw_positions:
            return []

        local_refs = await db.get_open_trade_rows()
        local_refs.extend(self.order_manager.get_open_positions())

        local_by_market_side: dict[tuple[str, str], list[dict]] = {}
        for ref in local_refs:
            side_key = str(ref.get("intent") or ref.get("side") or "")
            if not side_key:
                continue
            for market_ref in (ref.get("market_slug"), ref.get("question")):
                market_key = self._normalize_market_key(market_ref)
                if market_key:
                    local_by_market_side.setdefault((market_key, side_key), []).append(ref)

        condition_ids = [
            str(p.get("conditionId") or "").strip()
            for p in raw_positions
            if p.get("conditionId")
        ]
        latest_buy_by_market_outcome: dict[tuple[str, str], float] = {}
        if user_address and condition_ids:
            trade_rows = await self.client.get_trades(
                user=user_address,
                markets=condition_ids,
                limit=min(max(len(condition_ids) * 20, 50), 1000),
                taker_only=False,
                side="BUY",
            )
            for trade in trade_rows:
                market_id = str(trade.get("conditionId") or "").strip()
                outcome_key = str(trade.get("outcome") or "").upper()
                if not market_id or not outcome_key:
                    continue
                try:
                    ts = float(trade.get("timestamp") or 0.0)
                except (TypeError, ValueError):
                    continue
                if ts > 1_000_000_000_000:
                    ts /= 1000.0
                latest_buy_by_market_outcome[(market_id, outcome_key)] = max(
                    latest_buy_by_market_outcome.get((market_id, outcome_key), 0.0),
                    ts,
                )

        overrides = await db.get_auto_close_overrides()
        positions: list[dict] = []
        for pos in raw_positions:
            condition_id = str(pos.get("conditionId") or "").strip()
            outcome = str(pos.get("outcome") or "").upper()
            intent = "ORDER_INTENT_BUY_LONG" if outcome == "YES" else "ORDER_INTENT_BUY_SHORT"
            local_match = None
            for market_ref in (pos.get("slug"), pos.get("title")):
                market_key = self._normalize_market_key(market_ref)
                if not market_key:
                    continue
                candidates = local_by_market_side.get((market_key, intent), [])
                if candidates:
                    local_match = candidates.pop(0)
                    break

            avg_price = float(pos.get("avgPrice") or 0.0)
            size = float(pos.get("size") or 0.0)
            strategy = (local_match or {}).get("strategy") or self._infer_strategy(outcome, avg_price)
            placed_at = (local_match or {}).get("placed_at")
            if placed_at is None:
                raw_ts = (local_match or {}).get("timestamp")
                if raw_ts:
                    try:
                        placed_at = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00")).timestamp()
                    except Exception:
                        placed_at = None
            if placed_at is None and condition_id and outcome:
                placed_at = latest_buy_by_market_outcome.get((condition_id, outcome))

            positions.append({
                "order_id": (local_match or {}).get("order_id") or "",
                "market_slug": pos.get("slug") or pos.get("title") or "",
                "question": (local_match or {}).get("question") or pos.get("title") or pos.get("slug") or "",
                "intent": intent,
                "price": avg_price,
                "quantity": size,
                "strategy": strategy,
                "placed_at": float(placed_at) if placed_at is not None else time.time(),
                "condition_id": condition_id,
                "override_active": (condition_id, outcome) in overrides,
            })

        return positions

    async def run(self):
        if not self.enabled:
            return

        tp_pct    = self.config.get("take_profit_pct", 0.15)
        sl_pct    = self.config.get("stop_loss_pct", 0.08)
        exit_size = self.config.get("exit_size_usdc", 200)

        # Per-strategy max hold times (hours → seconds). None = no limit.
        hold_cfg  = self.config.get("max_hold_hours", {})
        default_max = hold_cfg.get("default", 24) if isinstance(hold_cfg, dict) else 24

        def max_hold_seconds(strategy: str) -> float | None:
            if not isinstance(hold_cfg, dict):
                return default_max * 3600
            hours = hold_cfg.get(strategy)
            if hours is None:
                hours = hold_cfg.get("default", default_max)
            return float(hours) * 3600

        positions = [
            pos for pos in await self._build_managed_positions()
            if pos.get("strategy") not in _EXCLUDED_STRATEGIES
        ]
        if not positions:
            return

        now  = time.time()
        exits = 0

        # Batch-fetch BBO for every open position up front. Exit order placement
        # still runs sequentially below since each one mutates order/capital state.
        slugs = [pos["market_slug"] for pos in positions]
        bbos = await asyncio.gather(
            *[self.market_data.get_bbo(slug) for slug in slugs]
        )
        bbo_by_slug = dict(zip(slugs, bbos))

        for pos in positions:
            order_id    = pos["order_id"]
            slug        = pos["market_slug"]
            intent      = pos["intent"]
            entry_price = pos["price"]
            quantity    = pos["quantity"]
            strategy    = pos.get("strategy", "")
            placed_at   = pos.get("placed_at", now)
            age_hours   = (now - placed_at) / 3600
            if pos.get("override_active"):
                continue

            bbo = bbo_by_slug.get(slug)
            if not bbo:
                continue

            try:
                current_bid = float(bbo.get("bid", {}).get("price", 0))
                current_ask = float(bbo.get("ask", {}).get("price", 1))
            except (TypeError, ValueError):
                continue

            exit_intent = None
            exit_price  = None
            trigger     = None
            exit_tif    = "TIME_IN_FORCE_FILL_OR_KILL"

            # ── Check max hold time first ─────────────────────────────────
            mhs = max_hold_seconds(strategy)
            if mhs is not None and (now - placed_at) >= mhs:
                # Two-tier exit: first attempt is passive (no overpay, rests in
                # book as GTC). After the grace window, escalate to aggressive
                # taker. Previously every MAX_HOLD ate a guaranteed 2¢ haircut.
                overdue_seconds = max(0.0, (now - placed_at) - mhs)
                default_first_attempt = now - min(overdue_seconds, _SOFT_EXIT_GRACE_SECONDS)
                attempt_key = order_id or f"{slug}:{intent}"
                first_attempt_ts, attempts = self._exit_attempts.get(
                    attempt_key, (default_first_attempt, 0)
                )
                attempts += 1
                elapsed_exiting = now - first_attempt_ts
                soft_phase = elapsed_exiting < _SOFT_EXIT_GRACE_SECONDS

                if soft_phase:
                    trigger  = f"MAX_HOLD_SOFT({age_hours:.1f}h,try{attempts})"
                    exit_tif = "TIME_IN_FORCE_GOOD_TILL_CANCEL"
                    if intent == "ORDER_INTENT_BUY_LONG":
                        exit_intent = "ORDER_INTENT_BUY_LONG"
                        exit_price  = max(0.01, round(current_bid, 4))
                    else:
                        exit_intent = "ORDER_INTENT_BUY_SHORT"
                        exit_price  = max(0.01, round(1.0 - current_ask, 4))
                else:
                    trigger  = f"MAX_HOLD_HARD({age_hours:.1f}h,try{attempts})"
                    exit_tif = "TIME_IN_FORCE_FILL_OR_KILL"
                    if intent == "ORDER_INTENT_BUY_LONG":
                        exit_intent = "ORDER_INTENT_BUY_LONG"
                        exit_price  = max(0.01, round(current_bid - 0.02, 4))
                    else:
                        exit_intent = "ORDER_INTENT_BUY_SHORT"
                        exit_price  = max(0.01, round((1.0 - current_ask) - 0.02, 4))

                self._exit_attempts[attempt_key] = (first_attempt_ts, attempts)

            # ── TP / SL (only if max-hold didn't fire) ────────────────────
            elif intent == "ORDER_INTENT_BUY_LONG":
                is_nc = strategy in _NEAR_CERTAINTY_STRATEGIES
                tp_threshold = (_NEAR_CERTAINTY_TP_THRESHOLD if is_nc
                                else entry_price * (1 + tp_pct))
                sl_threshold = entry_price * (1 - sl_pct)

                if current_bid >= tp_threshold:
                    trigger     = "TP"
                    exit_intent = "ORDER_INTENT_BUY_LONG"
                    exit_price  = max(0.01, round(current_bid, 4))
                elif current_bid <= sl_threshold:
                    trigger     = "SL"
                    exit_intent = "ORDER_INTENT_BUY_LONG"
                    exit_price  = max(0.01, round(current_bid, 4))

            elif intent == "ORDER_INTENT_BUY_SHORT":
                entry_no     = round(entry_price, 4)
                current_no_bid = round(1.0 - current_ask, 4)
                is_nc = strategy in _NEAR_CERTAINTY_STRATEGIES
                tp_threshold = (_NEAR_CERTAINTY_TP_THRESHOLD if is_nc
                                else entry_no * (1 + tp_pct))
                sl_threshold = entry_no * (1 - sl_pct)

                if current_no_bid >= tp_threshold:
                    trigger     = "TP"
                    exit_intent = "ORDER_INTENT_BUY_SHORT"
                    exit_price  = max(0.01, round(current_no_bid, 4))
                elif current_no_bid <= sl_threshold:
                    trigger     = "SL"
                    exit_intent = "ORDER_INTENT_BUY_SHORT"
                    exit_price  = max(0.01, round(current_no_bid, 4))

            if not trigger:
                continue

            pnl_est = (
                (current_bid - entry_price) * quantity
                if intent == "ORDER_INTENT_BUY_LONG"
                else (round(1.0 - current_ask, 4) - entry_price) * quantity
            )

            self.log(
                f"{trigger} | {strategy} | {slug[:35]} | "
                f"entry=${entry_price:.4f} now=${current_bid:.4f} "
                f"age={age_hours:.1f}h est_pnl=${pnl_est:.2f}"
            )

            # Release the capital that was locked when the entry was placed.
            # For live account positions we no longer rely on an open entry order
            # existing in memory, so we place the exit directly and then retire
            # the original open trade row from the local DB.
            notional = round(entry_price * quantity, 2)
            self.capital_manager.release(strategy, notional)

            exit_qty = min(quantity, round(exit_size / max(exit_price, 0.01), 2))
            oid = await self.order_manager.place_order(
                market_slug=slug,
                question=pos.get("question", slug),
                intent=exit_intent,
                price=exit_price,
                quantity=exit_qty,
                strategy=self.name,
                execution_side=SELL,
                tif=exit_tif,
            )

            if oid:
                self.log(
                    f"Exit order posted for {slug} ({trigger}) — waiting for actual fill before "
                    f"counting realized P&L"
                )
                exits += 1
            else:
                self.log(f"Exit order failed for {slug} — position stays open", level="warning")

        if exits:
            self.log(f"Closed {exits} position(s) this tick")
