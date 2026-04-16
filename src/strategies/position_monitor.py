"""
Position Monitor — Auto Take-Profit / Stop-Loss
-------------------------------------------------
Runs every tick and checks every open directional position against the
current market price. When a position hits its TP or SL threshold, the
monitor places an exit order (the opposite side) to close the trade.

Market-making quotes are excluded — they are managed by the market-making
strategy itself via cancel_stale_orders.

TP/SL logic:
  LONG (BUY_LONG / YES position)
    entry_price  = price paid per share
    TP triggered  when current_bid >= entry_price × (1 + tp_pct)
    SL triggered  when current_bid <= entry_price × (1 − sl_pct)
    Exit action:  place BUY_SHORT (sell YES / buy NO) at current ask

  SHORT (BUY_SHORT / NO position)
    entry_no_price  = 1 − entry_price  (the YES price when we entered)
    current_no_bid  = 1 − current_yes_ask
    TP triggered     when current_no_bid >= entry_no_price × (1 + tp_pct)
    SL triggered     when current_no_bid <= entry_no_price × (1 − sl_pct)
    Exit action:  place BUY_LONG (sell NO / buy YES) at current bid
"""

import asyncio
from src.strategies.base import BaseStrategy


# Strategies whose orders should NOT be monitored for TP/SL
_EXCLUDED_STRATEGIES = {"market_making"}


class PositionMonitorStrategy(BaseStrategy):
    """
    Monitors all open directional positions and exits them at TP or SL.
    Designed to run on every tick (same poll_interval as the main loop).
    """

    async def run(self):
        if not self.enabled:
            return

        tp_pct    = self.config.get("take_profit_pct", 0.15)   # e.g. 0.15 = exit when up 15%
        sl_pct    = self.config.get("stop_loss_pct", 0.10)     # e.g. 0.10 = exit when down 10%
        exit_size = self.config.get("exit_size_usdc", 200)     # max USDC per exit order

        positions = self.order_manager.get_open_positions(
            exclude_strategies=list(_EXCLUDED_STRATEGIES)
        )

        if not positions:
            return

        exits = 0
        for pos in positions:
            order_id    = pos["order_id"]
            slug        = pos["market_slug"]
            intent      = pos["intent"]
            entry_price = pos["price"]
            quantity    = pos["quantity"]

            bbo = await self.market_data.get_bbo(slug)
            if not bbo:
                continue

            try:
                current_bid = float(bbo.get("bid", {}).get("price", 0))
                current_ask = float(bbo.get("ask", {}).get("price", 1))
            except (TypeError, ValueError):
                continue

            exit_intent  = None
            exit_price   = None
            trigger      = None

            if intent == "ORDER_INTENT_BUY_LONG":
                # YES position — profit when YES price rises
                tp_threshold = entry_price * (1 + tp_pct)
                sl_threshold = entry_price * (1 - sl_pct)

                if current_bid >= tp_threshold:
                    trigger     = "TP"
                    exit_intent = "ORDER_INTENT_BUY_SHORT"
                    no_price    = round(1.0 - current_bid, 4)
                    exit_price  = max(0.01, no_price - 0.02)   # aggressive taker (sell into bid)

                elif current_bid <= sl_threshold:
                    trigger     = "SL"
                    exit_intent = "ORDER_INTENT_BUY_SHORT"
                    no_price    = round(1.0 - current_bid, 4)
                    exit_price  = max(0.01, no_price - 0.02)

            elif intent == "ORDER_INTENT_BUY_SHORT":
                # NO position — profit when YES price falls (NO price rises)
                entry_no  = round(1.0 - entry_price, 4)
                current_no_bid = round(1.0 - current_ask, 4)   # NO bid ≈ 1 − YES ask

                tp_threshold = entry_no * (1 + tp_pct)
                sl_threshold = entry_no * (1 - sl_pct)

                if current_no_bid >= tp_threshold:
                    trigger     = "TP"
                    exit_intent = "ORDER_INTENT_BUY_LONG"
                    exit_price  = min(0.99, current_ask + 0.02)  # aggressive taker

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
                f"{trigger} hit | {pos.get('strategy','?')} | {slug[:35]} | "
                f"entry=${entry_price:.4f} current=${current_bid:.4f} "
                f"est_pnl=${pnl_est:.2f} | placing exit {exit_intent}"
            )

            # Cancel the original tracked order first (removes it from our books)
            await self.order_manager.cancel_order(order_id)

            # Place the exit order
            exit_qty = min(quantity, round(exit_size / max(exit_price, 0.01), 2))
            oid = await self.order_manager.place_order(
                market_slug=slug,
                question=pos.get("question", slug),
                intent=exit_intent,
                price=exit_price,
                quantity=exit_qty,
                strategy=self.name,
                tif="TIME_IN_FORCE_FILL_OR_KILL",
            )

            if oid:
                await self.order_manager.mark_filled(oid, pnl=pnl_est)
                exits += 1
            else:
                self.log(f"Exit order failed for {slug} — position remains open", level="warning")

        if exits:
            self.log(f"Closed {exits} position(s) via TP/SL this tick")
