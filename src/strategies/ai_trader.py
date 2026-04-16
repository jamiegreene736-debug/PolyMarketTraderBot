"""
AI Trading Strategy (Superforecaster)
--------------------------------------
Uses Claude to estimate the true probability of a market outcome
and trades when the market price significantly diverges from the
AI estimate — i.e., when the crowd has it wrong.

Pipeline each run:
  1. Filter active markets to "interesting" price range (10%–90%)
  2. Skip markets we already hold or recently analyzed
  3. Fetch recent news headlines for context (via NewsAPI)
  4. Ask Claude to estimate the true probability with reasoning
  5. If |AI_prob - market_price| > min_edge AND confidence ≥ medium → trade
  6. Buy YES (if AI thinks market is underpriced) or Buy NO (overpriced)

Runs on a longer interval (default: every 60 min) to keep API costs low.
Results cached per market for cache_ttl_seconds (default: 2 hours).

Required env vars:
  ANTHROPIC_API_KEY  — Anthropic API key
  NEWS_API_KEY       — NewsAPI key (optional but strongly recommended)
"""

import json
import asyncio

import anthropic

from src.strategies.base import BaseStrategy
from src.news_client import NewsClient
from src import fees, database as db


SUPERFORECASTER_PROMPT = """\
You are a professional prediction market forecaster with a track record of accurate probability estimation.

Market Question: {question}
Current market price (implied YES probability): {price:.1%}
Time until resolution: {time_desc}

Recent relevant news:
{news}

Using structured superforecasting methodology:
1. What is the base rate for this type of event?
2. What specific evidence supports YES? What supports NO?
3. How much should you trust the current market price as a signal?
4. What is your best probability estimate?

Respond with ONLY a JSON object — no markdown, no explanation outside the JSON:
{{
  "probability": <float 0.00–1.00>,
  "confidence": "<low|medium|high>",
  "edge": "<buy|sell|pass>",
  "reasoning": "<one or two sentences>"
}}

Definitions:
- "edge": "buy"  → you believe YES is MORE likely than the market implies (market underpriced)
- "edge": "sell" → you believe NO  is MORE likely than the market implies (market overpriced, buy NO)
- "edge": "pass" → market price looks fair or you are genuinely uncertain
- "low" confidence  → sparse evidence, ambiguous question, or rapidly changing situation
- "high" confidence → strong, recent, corroborating evidence from multiple sources\
"""


class AITradingStrategy(BaseStrategy):
    """
    Superforecaster strategy powered by Claude + live news context.
    Trades when AI probability estimate diverges significantly from market price.
    """

    def __init__(self, *args, news_client: NewsClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.news_client = news_client
        self._anthropic_client: anthropic.AsyncAnthropic | None = None
        self._estimate_cache: dict[str, tuple[float, dict]] = {}  # slug → (timestamp, estimate)
        self._cache_ttl = self.config.get("cache_ttl_seconds", 7200)       # 2 hours
        self._run_interval = self.config.get("run_interval_seconds", 3600)  # 1 hour
        self._last_run: float = 0

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self):
        if not self.enabled:
            return

        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_run
        if elapsed < self._run_interval:
            remaining = int(self._run_interval - elapsed)
            self.log(f"Next AI scan in {remaining // 60}m {remaining % 60}s")
            return

        self._last_run = now

        min_edge     = self.config.get("min_edge_pct", 0.12)          # need ≥12% edge
        min_price    = self.config.get("min_market_price", 0.10)      # skip near-zero markets
        max_price    = self.config.get("max_market_price", 0.90)      # skip near-certain markets
        order_size   = self.config.get("order_size_usdc", 50)
        max_per_run  = self.config.get("max_markets_per_run", 8)      # API cost control
        model        = self.config.get("model", "claude-haiku-4-5-20251001")

        markets = await self.market_data.get_markets()
        if not markets:
            self.log("No markets available for AI analysis")
            return

        # Build candidate list: binary, interesting price range, no open position
        candidates = []
        for m in markets:
            slug = self.market_data.get_slug(m)
            if not slug:
                continue
            if self.order_manager.get_market_order_count(slug) > 0:
                continue  # already holding

            # Skip recently cached (still within TTL)
            cached = self._estimate_cache.get(slug)
            if cached and (now - cached[0]) < self._cache_ttl:
                continue

            bbo = await self.market_data.get_bbo(slug)
            if not bbo:
                continue
            try:
                mid = float(bbo.get("mid", 0))
            except (TypeError, ValueError):
                continue
            if not (min_price <= mid <= max_price):
                continue

            candidates.append((m, mid))

        if not candidates:
            self.log("No new markets to analyze this run (all cached or filtered)")
            return

        # Prioritise markets closest to 50% — most uncertain, highest potential edge
        candidates.sort(key=lambda x: abs(x[1] - 0.5))
        candidates = candidates[:max_per_run]

        self.log(f"AI scan: analysing {len(candidates)} markets using {model}")

        entered = 0
        for market, mid in candidates:
            slug      = self.market_data.get_slug(market)
            question  = self.market_data.get_question(market)
            if not slug or not question:
                continue

            # ── Fetch news context ────────────────────────────────────────────
            news_lines: list[str] = []
            if self.news_client:
                news_lines = await self.news_client.get_headlines(question, max_articles=5)
            news_context = (
                "\n".join(f"  • {h}" for h in news_lines)
                if news_lines
                else "  (no recent news found — base your estimate on general knowledge)"
            )

            # ── Build time description ────────────────────────────────────────
            hours_left = self.market_data._hours_to_resolution(market)
            if hours_left is None:
                time_desc = "unknown"
            elif hours_left < 24:
                time_desc = f"{hours_left:.1f} hours"
            elif hours_left < 720:
                time_desc = f"{hours_left / 24:.1f} days"
            else:
                time_desc = f"{hours_left / 720:.1f} months"

            # ── Ask Claude ────────────────────────────────────────────────────
            try:
                estimate = await self._ask_claude(
                    model=model,
                    question=question,
                    price=mid,
                    time_desc=time_desc,
                    news_context=news_context,
                )
            except Exception as e:
                self.log(f"Claude error on '{question[:40]}': {e}", level="warning")
                continue

            if not estimate:
                continue

            self._estimate_cache[slug] = (now, estimate)

            ai_prob    = float(estimate.get("probability", mid))
            confidence = estimate.get("confidence", "low")
            edge_dir   = estimate.get("edge", "pass")
            reasoning  = estimate.get("reasoning", "")
            edge       = abs(ai_prob - mid)

            self.log(
                f"'{question[:50]}' | "
                f"market={mid:.0%} AI={ai_prob:.0%} edge={edge:.0%} "
                f"[{confidence}] → {edge_dir} | {reasoning[:90]}"
            )

            # ── Filter by edge and confidence ─────────────────────────────────
            if edge_dir == "pass" or confidence == "low":
                continue
            if edge < min_edge:
                continue
            if not self.capital_manager.can_allocate(self.name, order_size):
                self.log("Capital limit reached", level="warning")
                break

            # ── Determine trade direction and price ───────────────────────────
            bbo = await self.market_data.get_bbo(slug)
            if not bbo:
                continue

            if edge_dir == "buy":
                # Market underprices YES — buy YES aggressively
                ask = float(bbo.get("ask", {}).get("price", min(mid + 0.02, 0.99)))
                taker_price = min(round(ask + 0.03, 4), 0.99)
                intent = "ORDER_INTENT_BUY_LONG"
            else:
                # Market overprices YES — buy NO aggressively
                bid = float(bbo.get("bid", {}).get("price", max(mid - 0.02, 0.01)))
                no_price = round(1.0 - bid, 4)
                taker_price = min(round(no_price + 0.03, 4), 0.99)
                intent = "ORDER_INTENT_BUY_SHORT"

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
                    f"TRADE {intent} {shares:.1f}x @ ${taker_price:.4f} | "
                    f"AI={ai_prob:.0%} vs market={mid:.0%} edge={edge:.0%} "
                    f"conf={confidence} | '{question[:45]}'"
                )
                entered += 1
            else:
                self.capital_manager.release(self.name, order_size)

        summary = f"AI trader: {entered} trade(s) placed | {len(candidates)} markets analysed"
        self.log(summary)
        await db.log_to_db("INFO", f"[ai_trader] {summary}")

    # ── Claude call ───────────────────────────────────────────────────────────

    def _get_anthropic(self) -> anthropic.AsyncAnthropic:
        if self._anthropic_client is None:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._anthropic_client

    async def _ask_claude(
        self,
        model: str,
        question: str,
        price: float,
        time_desc: str,
        news_context: str,
    ) -> dict | None:
        """Call Claude and return the parsed probability estimate dict."""
        client = self._get_anthropic()

        prompt = SUPERFORECASTER_PROMPT.format(
            question=question,
            price=price,
            time_desc=time_desc,
            news=news_context,
        )

        message = await client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()

        # Strip markdown code fences if Claude wraps the JSON
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        return json.loads(raw)
