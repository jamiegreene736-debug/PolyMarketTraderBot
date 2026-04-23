"""
Circuit Breaker
===============
Hard stops that pause or kill the bot when losses exceed configured thresholds.

Four independent triggers — any one fires and ALL trading stops:

  1. Daily loss limit      — total realized losses today exceed max_daily_loss_usdc
  2. Portfolio drawdown    — current balance is more than max_drawdown_pct below the
                             session-start balance (catches unrealized losses too)
  3. Consecutive losses    — N straight losing trades without a winner
  4. Rapid order rate      — more than max_orders_per_minute placed in any 60s window
                             (catches runaway loops / duplicate-trade bugs)

The breaker is checked in the main bot loop BEFORE each strategy tick.
If tripped, it sets _bot_state["status"] = "error" and logs the reason.
"""

from collections import deque
from datetime import datetime, timezone
from loguru import logger
from src import database as db


class CircuitBreaker:
    def __init__(self, config: dict, start_balance: float):
        cfg = config.get("circuit_breaker", {})

        # Thresholds
        self.max_daily_loss_usdc    = cfg.get("max_daily_loss_usdc", 50.0)
        self.max_drawdown_pct       = cfg.get("max_drawdown_pct", 0.20)      # 20%
        self.max_consecutive_losses = cfg.get("max_consecutive_losses", 5)
        self.max_orders_per_minute  = cfg.get("max_orders_per_minute", 15)

        # State
        self.session_start_balance  = start_balance
        self.session_start_at       = datetime.now(timezone.utc)
        self.day_start_balance      = start_balance
        self._today_date            = datetime.now(timezone.utc).date()

        self._consecutive_losses    = 0
        self._order_timestamps: deque[float] = deque()  # epoch seconds
        self._tripped               = False
        self._trip_reason           = ""

    # ── Called by the bot loop ────────────────────────────────────────────────

    async def check(self, current_balance: float, recent_pnl_list: list[float]) -> bool:
        """
        Returns True if it's safe to trade, False if the breaker has tripped.
        Call this at the top of every tick before running strategies.
        """
        if self._tripped:
            return False

        self._refresh_day(current_balance)
        self._update_consecutive(recent_pnl_list)

        # ── Trigger 1: daily loss limit ───────────────────────────────────────
        daily_loss = self.day_start_balance - current_balance
        if daily_loss >= self.max_daily_loss_usdc:
            await self._trip(
                f"Daily loss limit hit: lost ${daily_loss:.2f} today "
                f"(limit=${self.max_daily_loss_usdc:.2f})"
            )
            return False

        # ── Trigger 2: portfolio drawdown ─────────────────────────────────────
        drawdown = (self.session_start_balance - current_balance) / max(self.session_start_balance, 0.01)
        if drawdown >= self.max_drawdown_pct:
            await self._trip(
                f"Portfolio drawdown limit hit: down {drawdown*100:.1f}% from session start "
                f"(${self.session_start_balance:.2f} → ${current_balance:.2f}, "
                f"limit={self.max_drawdown_pct*100:.0f}%)"
            )
            return False

        # ── Trigger 3: consecutive losses ─────────────────────────────────────
        if self._consecutive_losses >= self.max_consecutive_losses:
            await self._trip(
                f"Consecutive loss limit hit: {self._consecutive_losses} losses in a row "
                f"(limit={self.max_consecutive_losses})"
            )
            return False

        # ── Trigger 4: rapid order rate ───────────────────────────────────────
        now = datetime.now(timezone.utc).timestamp()
        self._order_timestamps = deque(
            t for t in self._order_timestamps if now - t <= 60
        )
        if len(self._order_timestamps) >= self.max_orders_per_minute:
            await self._trip(
                f"Rapid order rate: {len(self._order_timestamps)} orders in the last 60s "
                f"(limit={self.max_orders_per_minute})"
            )
            return False

        return True

    def record_order(self):
        """Call once every time an order is successfully placed."""
        self._order_timestamps.append(datetime.now(timezone.utc).timestamp())

    def record_trade_result(self, pnl: float):
        """Call when a trade closes so we can track consecutive losses."""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset(self, current_balance: float):
        """Manual reset from the dashboard after the user reviews the situation."""
        self._tripped = False
        self._trip_reason = ""
        self._consecutive_losses = 0
        self._order_timestamps.clear()
        self.session_start_balance = current_balance
        self.session_start_at = datetime.now(timezone.utc)
        logger.info(f"Circuit breaker reset. New session balance: ${current_balance:.2f}")

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _refresh_day(self, current_balance: float):
        today = datetime.now(timezone.utc).date()
        if today != self._today_date:
            self._today_date = today
            self.day_start_balance = current_balance
            logger.info(f"Circuit breaker: new day — resetting day_start_balance to ${current_balance:.2f}")

    def _update_consecutive(self, recent_pnl_list: list[float]):
        """
        Update consecutive loss counter from a fresh list of recent closed PnLs.
        Walk from newest → oldest and count the tail of negatives.
        """
        if not recent_pnl_list:
            self._consecutive_losses = 0
            return
        count = 0
        for pnl in recent_pnl_list:
            if pnl < 0:
                count += 1
            else:
                break
        self._consecutive_losses = count

    async def _trip(self, reason: str):
        self._tripped = True
        self._trip_reason = reason
        msg = f"[CIRCUIT BREAKER TRIPPED] {reason}"
        logger.error(msg)
        await db.log_to_db("ERROR", msg)
