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
    entry_no  = 1 − entry_price
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
from src.strategies.base import BaseStrategy


# Strategies whose orders are managed elsewhere (not TP/SL monitored).
# position_monitor is excluded from its own scan so it doesn't recursively
# try to "exit" the exit orders it just placed.
_EXCLUDED_STRATEGIES = {"market_making", "position_monitor"}

# Near-certainty strategies use an absolute TP near $1.00, not pct-based
_NEAR_CERTAINTY_STRATEGIES = {"near_certainty", "inverted_near_certainty"}
_NEAR_CERTAINTY_TP_THRESHOLD = 0.99   # exit when price reaches $0.99

# Smart force-exit tuning. First attempt at MAX_HOLD tries a passive limit
# at the fair price (no overpay). After this many seconds of unfilled
# passive attempts, we escalate to aggressive taker.
_SOFT_EXIT_GRACE_SECONDS = 180        # 3 minutes


class PositionMonitorStrategy(BaseStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # order_id → (first_attempt_ts, attempt_count)
        self._exit_attempts: dict[str, tuple[float, int]] = {}

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

        positions = self.order_manager.get_open_positions(
            exclude_strategies=list(_EXCLUDED_STRATEGIES)
        )
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
                first_attempt_ts, attempts = self._exit_attempts.get(
                    order_id, (now, 0)
                )
                attempts += 1
                elapsed_exiting = now - first_attempt_ts
                soft_phase = (attempts == 1) or (elapsed_exiting < _SOFT_EXIT_GRACE_SECONDS)

                if soft_phase:
                    trigger  = f"MAX_HOLD_SOFT({age_hours:.1f}h,try{attempts})"
                    exit_tif = "TIME_IN_FORCE_GOOD_TILL_CANCEL"
                    if intent == "ORDER_INTENT_BUY_LONG":
                        exit_intent = "ORDER_INTENT_BUY_SHORT"
                        # Fair NO price at current bid; no 2¢ haircut.
                        exit_price  = max(0.01, round(1.0 - current_bid, 4))
                    else:
                        exit_intent = "ORDER_INTENT_BUY_LONG"
                        exit_price  = min(0.99, current_ask)
                else:
                    trigger  = f"MAX_HOLD_HARD({age_hours:.1f}h,try{attempts})"
                    exit_tif = "TIME_IN_FORCE_FILL_OR_KILL"
                    if intent == "ORDER_INTENT_BUY_LONG":
                        exit_intent = "ORDER_INTENT_BUY_SHORT"
                        no_price    = round(1.0 - current_bid, 4)
                        exit_price  = max(0.01, no_price - 0.02)
                    else:
                        exit_intent = "ORDER_INTENT_BUY_LONG"
                        exit_price  = min(0.99, current_ask + 0.02)

                self._exit_attempts[order_id] = (first_attempt_ts, attempts)

            # ── TP / SL (only if max-hold didn't fire) ────────────────────
            elif intent == "ORDER_INTENT_BUY_LONG":
                is_nc = strategy in _NEAR_CERTAINTY_STRATEGIES
                tp_threshold = (_NEAR_CERTAINTY_TP_THRESHOLD if is_nc
                                else entry_price * (1 + tp_pct))
                sl_threshold = entry_price * (1 - sl_pct)

                if current_bid >= tp_threshold:
                    trigger     = "TP"
                    exit_intent = "ORDER_INTENT_BUY_SHORT"
                    no_price    = round(1.0 - current_bid, 4)
                    exit_price  = max(0.01, no_price - 0.02)
                elif current_bid <= sl_threshold:
                    trigger     = "SL"
                    exit_intent = "ORDER_INTENT_BUY_SHORT"
                    no_price    = round(1.0 - current_bid, 4)
                    exit_price  = max(0.01, no_price - 0.02)

            elif intent == "ORDER_INTENT_BUY_SHORT":
                entry_no     = round(1.0 - entry_price, 4)
                current_no_bid = round(1.0 - current_ask, 4)
                is_nc = strategy in _NEAR_CERTAINTY_STRATEGIES
                tp_threshold = (_NEAR_CERTAINTY_TP_THRESHOLD if is_nc
                                else entry_no * (1 + tp_pct))
                sl_threshold = entry_no * (1 - sl_pct)

                if current_no_bid >= tp_threshold:
                    trigger     = "TP"
                    exit_intent = "ORDER_INTENT_BUY_LONG"
                    exit_price  = min(0.99, current_ask + 0.02)
                elif current_no_bid <= sl_threshold:
                    trigger     = "SL"
                    exit_intent = "ORDER_INTENT_BUY_LONG"
                    exit_price  = min(0.99, current_ask + 0.02)

            if not trigger:
                continue

            pnl_est = (
                (current_bid - entry_price) * quantity
                if intent == "ORDER_INTENT_BUY_LONG"
                else (round(1.0 - current_ask, 4) - round(1.0 - entry_price, 4)) * quantity
            )

            self.log(
                f"{trigger} | {strategy} | {slug[:35]} | "
                f"entry=${entry_price:.4f} now=${current_bid:.4f} "
                f"age={age_hours:.1f}h est_pnl=${pnl_est:.2f}"
            )

            await self.order_manager.cancel_order(order_id)

            # Release the capital that was locked when the entry was placed.
            # This must happen regardless of whether the exit order fills,
            # because the entry position is gone the moment we cancel it above.
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
                tif=exit_tif,
            )

            if oid:
                await self.order_manager.mark_filled(oid, pnl=pnl_est)
                # Done with this position — clear any exit-attempt state.
                self._exit_attempts.pop(order_id, None)
                exits += 1
            else:
                self.log(f"Exit order failed for {slug} — position stays open", level="warning")

        if exits:
            self.log(f"Closed {exits} position(s) this tick")
