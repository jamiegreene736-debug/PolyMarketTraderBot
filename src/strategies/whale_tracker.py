"""
Whale / Momentum Tracker
-------------------------
Detects large, sudden price moves in active markets — the fingerprint of
whale activity — and trades in the same direction before the broader market
fully adjusts.

How it works:
  Each tick the strategy records the mid-price of every market. After
  `lookback_ticks` ticks of history, it compares the current price to the
  price N ticks ago. When the move exceeds `min_move_pct`:

    price went UP  → buy YES  (momentum long — whale is buying YES)
    price went DOWN → buy NO  (momentum short — whale is buying NO)

  A second gate (volume rank) ensures we only follow momentum in the most
  liquid markets, where whales actually operate.

Why this works without on-chain wallet access:
  The Polymarket.us API does not expose individual wallet activity publicly.
  But large whale orders show up instantly as price movements. A 5%+ price
  jump in a liquid market within 60 seconds is almost always caused by a
  large single order — not organic drift. We catch the same signal that
  on-chain copy-traders target, with a 30–60 second lag instead of
  block-level speed.

Limitations:
  - Runs at poll_interval speed (default 30s) — not millisecond latency
  - May catch news-driven moves (fine — those are valid alpha too)
  - Doesn't distinguish whale from news; both are worth following
"""

import asyncio
from collections import deque
from src.strategies.base import BaseStrategy
from src import fees


class WhaleTrackerStrategy(BaseStrategy):
    """
    Momentum strategy that follows large sudden price moves in liquid markets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # slug → deque of (timestamp, mid_price) snapshots
        self._price_history: dict[str, deque] = {}

    async def run(self):
        if not self.enabled:
            return

        min_move_pct   = self.config.get("min_move_pct", 0.05)     # 5% price move triggers
        lookback_ticks = self.config.get("lookback_ticks", 4)       # compare to N ticks ago (~2 min)
        top_n_markets  = self.config.get("top_markets", 30)         # watch top N by volume
        fallback_size  = self.config.get("order_size_usdc", 50)
        use_kelly      = self.config.get("use_kelly_sizing", True)
        kelly_frac     = self.config.get("kelly_fraction", 0.20)    # more conservative for momentum
        min_net_return = self.config.get("min_net_return_pct", 0.5) # lower bar — momentum trades are short-lived

        now = asyncio.get_event_loop().time()

        # Use the top markets by volume for whale monitoring
        markets = await self.market_data.get_markets_by_volume(
            min_volume=0, top_n=top_n_markets
        )
        if not markets:
            return

        signals: list[tuple[str, str, float, float, float]] = []
        # (slug, question, mid, move_pct, direction_sign)

        for market in markets:
            slug = self.market_data.get_slug(market)
            if not slug:
                continue

            bbo = await self.market_data.get_bbo(slug)
            if not bbo:
                continue

            try:
                mid = float(bbo.get("mid", 0))
            except (TypeError, ValueError):
                continue

            if mid <= 0.01 or mid >= 0.99:
                continue

            # Record snapshot
            if slug not in self._price_history:
                self._price_history[slug] = deque(maxlen=lookback_ticks + 2)
            self._price_history[slug].append((now, mid))

            history = self._price_history[slug]
            if len(history) < lookback_ticks:
                continue  # not enough history yet

            past_mid = history[0][1]
            move = (mid - past_mid) / past_mid  # positive = price rose, negative = fell

            if abs(move) < min_move_pct:
                continue

            signals.append((slug, self.market_data.get_question(market), mid, move, bbo))

        if not signals:
            return

        # Sort by magnitude of move — biggest moves first
        signals.sort(key=lambda x: abs(x[3]), reverse=True)

        entered = 0
        for slug, question, mid, move, bbo in signals:
            if self.order_manager.get_market_order_count(slug) > 0:
                continue  # already in this market

            # Determine direction — follow the momentum
            if move > 0:
                # Price surged — buy YES (long)
                direction = "up"
                intent    = "ORDER_INTENT_BUY_LONG"
                ask = float(bbo.get("ask", {}).get("price", min(mid + 0.02, 0.99)))
                taker_price = min(round(ask + 0.02, 4), 0.99)
                win_prob    = min(mid + abs(move) / 2, 0.95)   # assume move continues partway
                net_ret_pct = fees.net_profit_pct_near_certainty(mid) / 100
            else:
                # Price crashed — buy NO (short)
                direction = "down"
                intent    = "ORDER_INTENT_BUY_SHORT"
                bid = float(bbo.get("bid", {}).get("price", max(mid - 0.02, 0.01)))
                no_price    = round(1.0 - bid, 4)
                taker_price = min(round(no_price + 0.02, 4), 0.99)
                win_prob    = min(1.0 - mid + abs(move) / 2, 0.95)
                net_ret_pct = fees.net_profit_pct_near_certainty(1.0 - mid) / 100

            if net_ret_pct < min_net_return / 100:
                continue

            # Kelly sizing
            if use_kelly:
                order_size = self.capital_manager.kelly_size(
                    self.name,
                    win_prob=win_prob,
                    net_return_pct=max(net_ret_pct, 0.005),
                    kelly_fraction=kelly_frac,
                    min_size=10.0,
                    max_size=fallback_size,
                )
            else:
                order_size = fallback_size

            if not self.capital_manager.can_allocate(self.name, order_size):
                self.log("Capital limit reached", level="warning")
                break

            self.log(
                f"WHALE SIGNAL {direction.upper()} {abs(move):.1%} move | "
                f"'{question[:45]}' | mid=${mid:.3f} → entry=${taker_price:.4f} | "
                f"kelly=${order_size:.0f}"
            )

            shares = round(order_size / taker_price, 2)

            if not self.capital_manager.allocate(self.name, order_size):
                break

            order_id = await self.order_manager.place_order(
                market_slug=slug,
                question=question,
                intent=intent,
                price=taker_price,
                quantity=shares,
                strategy=self.name,
                tif="TIME_IN_FORCE_FILL_OR_KILL",
            )

            if order_id:
                self.log(
                    f"MOMENTUM TRADE {intent} {shares:.1f}x @ ${taker_price:.4f} | "
                    f"move={move:+.1%} | '{question[:40]}'"
                )
                entered += 1
                # Clear history so we don't re-trigger immediately
                self._price_history[slug].clear()
            else:
                self.capital_manager.release(self.name, order_size)

        if entered:
            self.log(f"Whale tracker: {entered} momentum trade(s) this tick")
