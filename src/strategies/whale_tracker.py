"""
Ranked Whale Tracker
--------------------
Primary mode:
  1. Pull top holders for the highest-volume markets.
  2. Rank those wallets by recent realized P&L and win rate.
  3. Watch recent large BUY trades from the best-ranked wallets.
  4. Copy only the strongest, freshest signals with a strict daily cap.

Fallback mode:
  If wallet data is sparse or ranking is cold, fall back to the legacy
  momentum-following logic based on sudden price moves in liquid markets.

This keeps the strategy fully automated while still behaving sensibly when the
public data API doesn't return enough elite-wallet activity to justify a trade.
"""

import asyncio
import time
from collections import deque
from src.strategies.base import BaseStrategy
from src import database as db


class WhaleTrackerStrategy(BaseStrategy):
    """
    Follow high-signal whale activity using public holder/trade data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # slug -> deque[(ts, mid)]
        self._price_history: dict[str, deque] = {}
        # wallet -> ranking profile
        self._wallet_profiles: dict[str, dict] = {}
        self._wallet_ranked_at: float = 0.0
        # trade-key -> first seen monotonic timestamp
        self._seen_trade_keys: dict[str, float] = {}

    async def run(self):
        if not self.enabled:
            return

        max_trades_per_day = int(self.config.get("max_trades_per_day", 10))
        trades_today = await db.count_trades_today(self.name)
        if trades_today >= max_trades_per_day:
            self.log(
                f"Daily trade cap reached ({trades_today}/{max_trades_per_day})",
                level="warning",
            )
            return

        markets = await self.market_data.get_markets_by_volume(
            min_volume=0,
            top_n=self.config.get("top_markets", 300),
        )
        if not markets:
            return

        ranked_signals = await self._get_ranked_wallet_signals(markets)
        if ranked_signals:
            entered = await self._execute_ranked_signals(
                ranked_signals,
                remaining=max_trades_per_day - trades_today,
            )
            if entered:
                self.log(f"Whale tracker: {entered} ranked-wallet trade(s) this tick")
                return

        if self.config.get("fallback_to_momentum", True):
            entered = await self._run_momentum_fallback(markets)
            if entered:
                self.log(f"Whale tracker: {entered} momentum fallback trade(s) this tick")

    def _purge_seen_trades(self, now: float):
        ttl = max(float(self.config.get("trade_signal_cooldown_seconds", 1800)), 60.0)
        stale = [key for key, ts in self._seen_trade_keys.items() if now - ts > ttl]
        for key in stale:
            self._seen_trade_keys.pop(key, None)

    def _wallet_profile_from_closed_positions(self, positions: list[dict]) -> dict | None:
        min_closed = int(self.config.get("min_wallet_closed_trades", 3))
        if len(positions) < min_closed:
            return None

        pnls = []
        for pos in positions:
            try:
                pnls.append(float(pos.get("realizedPnl") or 0.0))
            except (TypeError, ValueError):
                continue

        if len(pnls) < min_closed:
            return None

        total_pnl = sum(pnls)
        wins = sum(1 for pnl in pnls if pnl > 0)
        total = len(pnls)
        win_rate = wins / total if total else 0.0
        avg_pnl = total_pnl / total if total else 0.0

        # Blend realized profit, consistency, and sample size into one score.
        score = (
            max(total_pnl, 0.0) * 1.2
            + win_rate * 60.0
            + min(total, 20) * 1.5
            + max(avg_pnl, 0.0) * 3.0
        )

        if total_pnl <= 0 or win_rate < float(self.config.get("min_ranked_wallet_win_rate", 0.50)):
            return None

        return {
            "score": round(score, 2),
            "total_pnl": round(total_pnl, 2),
            "wins": wins,
            "total": total,
            "win_rate": round(win_rate, 4),
        }

    async def _refresh_wallet_rankings(self, markets: list[dict]) -> dict[str, dict]:
        now = time.time()
        refresh_seconds = int(self.config.get("wallet_refresh_seconds", 1800))
        if self._wallet_profiles and (now - self._wallet_ranked_at) < refresh_seconds:
            return self._wallet_profiles

        top_holder_markets = int(self.config.get("top_holder_markets", 40))
        top_holders_per_market = int(self.config.get("top_holders_per_market", 5))
        max_ranked_wallets = int(self.config.get("max_ranked_wallets", 150))
        wallet_closed_positions_limit = int(self.config.get("wallet_closed_positions_limit", 20))

        condition_ids = [
            str(m.get("conditionId") or m.get("condition_id") or "")
            for m in markets[:top_holder_markets]
            if m.get("conditionId") or m.get("condition_id")
        ]
        if not condition_ids:
            self._wallet_profiles = {}
            self._wallet_ranked_at = now
            return self._wallet_profiles

        holder_batches = [
            condition_ids[i:i + 10] for i in range(0, len(condition_ids), 10)
        ]
        holder_responses = await asyncio.gather(
            *[
                self.client.get_top_holders(
                    batch,
                    limit=top_holders_per_market,
                    min_balance=int(self.config.get("min_holder_balance", 1)),
                )
                for batch in holder_batches
            ]
        )

        wallet_weights: dict[str, float] = {}
        for response in holder_responses:
            for token_group in response:
                for holder in token_group.get("holders", []):
                    wallet = str(holder.get("proxyWallet") or "").strip().lower()
                    if not wallet:
                        continue
                    try:
                        amount = float(holder.get("amount") or 0.0)
                    except (TypeError, ValueError):
                        amount = 0.0
                    wallet_weights[wallet] = max(wallet_weights.get(wallet, 0.0), amount)

        candidate_wallets = sorted(
            wallet_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max_ranked_wallets]
        if not candidate_wallets:
            self._wallet_profiles = {}
            self._wallet_ranked_at = now
            return self._wallet_profiles

        semaphore = asyncio.Semaphore(int(self.config.get("wallet_rank_concurrency", 12)))

        async def score_wallet(wallet: str, holder_amount: float):
            async with semaphore:
                positions = await self.client.get_closed_positions(
                    user=wallet,
                    limit=min(max(wallet_closed_positions_limit, 1), 50),
                )
                profile = self._wallet_profile_from_closed_positions(positions)
                if not profile:
                    return None
                profile["holder_amount"] = round(holder_amount, 2)
                profile["wallet"] = wallet
                return profile

        scored = await asyncio.gather(
            *[score_wallet(wallet, amount) for wallet, amount in candidate_wallets]
        )
        profiles = [profile for profile in scored if profile]
        profiles.sort(key=lambda profile: profile["score"], reverse=True)

        min_wallet_score = float(self.config.get("min_wallet_score", 70.0))
        self._wallet_profiles = {
            profile["wallet"]: profile
            for profile in profiles
            if profile["score"] >= min_wallet_score
        }
        self._wallet_ranked_at = now

        if self._wallet_profiles:
            best = next(iter(sorted(self._wallet_profiles.values(), key=lambda p: p["score"], reverse=True)), None)
            self.log(
                f"Wallet ranking refreshed: {len(self._wallet_profiles)} ranked wallets "
                f"(best score={best['score']:.1f}, pnl=${best['total_pnl']:.2f}, "
                f"win_rate={best['win_rate']*100:.0f}%)"
            )
        else:
            self.log("Wallet ranking refreshed: no wallets met score threshold")

        return self._wallet_profiles

    async def _get_ranked_wallet_signals(self, markets: list[dict]) -> list[dict]:
        profiles = await self._refresh_wallet_rankings(markets)
        if not profiles:
            return []

        now_wall = time.time()
        self._purge_seen_trades(now_wall)

        trade_markets = [
            str(m.get("conditionId") or m.get("condition_id") or "")
            for m in markets[: int(self.config.get("top_holder_markets", 40))]
            if m.get("conditionId") or m.get("condition_id")
        ]
        if not trade_markets:
            return []

        trade_rows = await self.client.get_trades(
            markets=trade_markets,
            limit=int(self.config.get("recent_trades_fetch_limit", 300)),
            taker_only=True,
            side="BUY",
            filter_type="CASH",
            filter_amount=float(self.config.get("min_whale_trade_usdc", 250.0)),
        )
        if not trade_rows:
            return []

        recent_window = float(self.config.get("recent_trade_window_seconds", 900))
        min_market_price = float(self.config.get("min_market_price", 0.10))
        max_market_price = float(self.config.get("max_market_price", 0.90))

        market_by_condition = {
            str(m.get("conditionId") or m.get("condition_id") or ""): m
            for m in markets
            if m.get("conditionId") or m.get("condition_id")
        }

        best_by_market: dict[tuple[str, str], dict] = {}
        for trade in trade_rows:
            wallet = str(trade.get("proxyWallet") or "").strip().lower()
            profile = profiles.get(wallet)
            if not profile:
                continue

            try:
                ts = float(trade.get("timestamp") or 0.0)
            except (TypeError, ValueError):
                continue
            if ts > 1_000_000_000_000:
                ts /= 1000.0
            age = now_wall - ts
            if age < 0 or age > recent_window:
                continue

            side = str(trade.get("side") or "").upper()
            outcome = str(trade.get("outcome") or "").upper()
            if side != "BUY" or outcome not in {"YES", "NO"}:
                continue

            condition_id = str(trade.get("conditionId") or "")
            market = market_by_condition.get(condition_id)
            slug = str(trade.get("slug") or (market or {}).get("slug") or "")
            if not slug or self.order_manager.get_market_order_count(slug) > 0:
                continue

            price = float(trade.get("price") or 0.0)
            size = float(trade.get("size") or 0.0)
            if price <= 0:
                continue
            notional = max(price * max(size, 0.0), float(self.config.get("min_whale_trade_usdc", 250.0)))

            mid = price if 0 < price < 1 else 0.0
            if mid and (mid < min_market_price or mid > max_market_price):
                continue

            intent = "ORDER_INTENT_BUY_LONG" if outcome == "YES" else "ORDER_INTENT_BUY_SHORT"
            trade_key = str(trade.get("transactionHash") or f"{wallet}:{condition_id}:{outcome}:{int(ts)}")
            if trade_key in self._seen_trade_keys:
                continue

            freshness = max(0.1, 1.0 - (age / recent_window))
            size_boost = min(notional / max(float(self.config.get("min_whale_trade_usdc", 250.0)), 1.0), 4.0)
            signal_score = profile["score"] * (1.0 + 0.15 * size_boost) * freshness

            signal = {
                "wallet": wallet,
                "wallet_name": trade.get("name") or trade.get("pseudonym") or wallet[:10],
                "wallet_score": profile["score"],
                "wallet_win_rate": profile["win_rate"],
                "wallet_total_pnl": profile["total_pnl"],
                "trade_key": trade_key,
                "slug": slug,
                "question": trade.get("title") or self.market_data.get_question(market or {}) or slug,
                "intent": intent,
                "price": price,
                "notional": notional,
                "age_seconds": age,
                "signal_score": signal_score,
            }
            bucket_key = (slug, intent)
            if bucket_key not in best_by_market or signal_score > best_by_market[bucket_key]["signal_score"]:
                best_by_market[bucket_key] = signal

        return sorted(best_by_market.values(), key=lambda signal: signal["signal_score"], reverse=True)

    async def _execute_ranked_signals(self, signals: list[dict], remaining: int) -> int:
        fallback_size = float(self.config.get("order_size_usdc", 5))
        entered = 0

        for signal in signals:
            if entered >= remaining:
                break

            slug = signal["slug"]
            bbo = await self.market_data.get_bbo(slug)
            if not bbo:
                continue

            try:
                yes_bid = float(bbo.get("bid", {}).get("price", 0.0))
                yes_ask = float(bbo.get("ask", {}).get("price", 0.0))
            except (TypeError, ValueError):
                continue

            if signal["intent"] == "ORDER_INTENT_BUY_LONG":
                taker_price = yes_ask or min(max(signal["price"], 0.01), 0.99)
            else:
                taker_price = max(0.01, round(1.0 - max(yes_bid, 0.01), 4))

            taker_price = min(max(taker_price, 0.01), 0.99)
            if taker_price <= 0:
                continue

            order_size = fallback_size
            if not self.capital_manager.can_allocate(self.name, order_size):
                self.log("Capital limit reached", level="warning")
                break

            shares = round(order_size / taker_price, 2)

            self.log(
                f"RANKED WHALE {signal['wallet_name']} score={signal['wallet_score']:.1f} "
                f"win_rate={signal['wallet_win_rate']*100:.0f}% pnl=${signal['wallet_total_pnl']:.2f} | "
                f"{signal['intent']} {shares:.1f}x @ ${taker_price:.4f} | "
                f"age={int(signal['age_seconds'])}s | '{signal['question'][:48]}'"
            )

            if not self.capital_manager.allocate(self.name, order_size):
                break

            order_id = await self.order_manager.place_order(
                market_slug=slug,
                question=signal["question"],
                intent=signal["intent"],
                price=taker_price,
                quantity=shares,
                strategy=self.name,
                tif="TIME_IN_FORCE_FILL_OR_KILL",
            )
            if order_id:
                entered += 1
                self._seen_trade_keys[signal["trade_key"]] = time.time()
            else:
                self.capital_manager.release(self.name, order_size)

        return entered

    async def _run_momentum_fallback(self, markets: list[dict]) -> int:
        min_move_pct = self.config.get("min_move_pct", 0.05)
        lookback_ticks = self.config.get("lookback_ticks", 4)
        fallback_size = self.config.get("order_size_usdc", 5)
        use_kelly = self.config.get("use_kelly_sizing", True)
        kelly_frac = self.config.get("kelly_fraction", 0.20)
        min_net_return = self.config.get("min_net_return_pct", 0.5)

        now = asyncio.get_event_loop().time()
        signals: list[tuple[str, str, float, float]] = []

        market_slugs = [(m, self.market_data.get_slug(m)) for m in markets]
        market_slugs = [(m, s) for m, s in market_slugs if s]
        bbos = await asyncio.gather(
            *[self.market_data.get_bbo(s) for _, s in market_slugs]
        )

        for (market, slug), bbo in zip(market_slugs, bbos):
            if not bbo:
                continue

            try:
                bid = float(bbo.get("bid", {}).get("price", 0.0))
                ask = float(bbo.get("ask", {}).get("price", 0.0))
            except (TypeError, ValueError):
                continue
            if bid <= 0 or ask <= 0:
                continue

            mid = round((bid + ask) / 2, 4)
            if mid <= 0.10 or mid >= 0.90:
                continue

            history = self._price_history.setdefault(slug, deque(maxlen=lookback_ticks + 2))
            history.append((now, mid))
            if len(history) < lookback_ticks:
                continue

            past_mid = history[0][1]
            if past_mid <= 0:
                continue
            move = (mid - past_mid) / past_mid
            if abs(move) < min_move_pct:
                continue
            signals.append((slug, self.market_data.get_question(market), mid, move))

        signals.sort(key=lambda item: abs(item[3]), reverse=True)
        entered = 0

        for slug, question, mid, move in signals:
            if self.order_manager.get_market_order_count(slug) > 0:
                continue

            if move > 0:
                intent = "ORDER_INTENT_BUY_LONG"
                taker_price = min(round(mid + 0.03, 4), 0.88)
                win_prob = min(mid + abs(move) / 2, 0.88)
            else:
                intent = "ORDER_INTENT_BUY_SHORT"
                no_fair = round(1.0 - mid, 4)
                taker_price = min(round(no_fair + 0.03, 4), 0.88)
                win_prob = min(1.0 - mid + abs(move) / 2, 0.88)

            net_ret_pct = max((win_prob - taker_price) / max(taker_price, 0.01), 0.0)
            if net_ret_pct < min_net_return / 100:
                continue

            if use_kelly:
                order_size = self.capital_manager.kelly_size(
                    self.name,
                    win_prob=win_prob,
                    net_return_pct=max(net_ret_pct, 0.005),
                    kelly_fraction=kelly_frac,
                    min_size=fallback_size,
                    max_size=fallback_size,
                )
            else:
                order_size = fallback_size

            if not self.capital_manager.can_allocate(self.name, order_size):
                self.log("Capital limit reached", level="warning")
                break

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
                entered += 1
                self._price_history[slug].clear()
            else:
                self.capital_manager.release(self.name, order_size)

        return entered
