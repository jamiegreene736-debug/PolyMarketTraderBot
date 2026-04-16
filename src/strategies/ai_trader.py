"""
AI Trading Strategy (Superforecaster + Ensemble)
-------------------------------------------------
Uses Claude + (optionally) OpenAI GPT to estimate the true probability of a
market outcome and trades when the market price significantly diverges from
the AI consensus — i.e., when the crowd has it wrong.

Pipeline each run:
  1. Filter active markets to "interesting" price range (10%–90%)
  2. Skip markets we already hold or recently analysed
  3. Fetch recent news headlines for context (via NewsAPI)
  4. Ask Claude (and optionally GPT-4o) to estimate the true probability
  5. Bayesian calibration: damp the AI edge based on market category
  6. Ensemble gate: only trade when both models agree (if ensemble enabled)
  7. Kelly-size the position based on estimated edge and confidence
  8. Buy YES (if underpriced) or Buy NO (if overpriced)

Runs on a longer interval (default: every 60 min) to keep API costs low.

Required env vars:
  ANTHROPIC_API_KEY  — Anthropic API key
  OPENAI_API_KEY     — OpenAI key (optional; required for ensemble mode)
  NEWS_API_KEY       — NewsAPI key (optional but strongly recommended)
"""

import json
import asyncio
import os

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

# ── Bayesian category calibration ────────────────────────────────────────────
# How much to trust AI edge in each category (1.0 = full trust, 0.5 = half).
# These represent AI's historical hit rate vs. calibration in each domain.
# Lower = more sceptical of AI; conservative defaults, tunable in config.
DEFAULT_CALIBRATION = {
    "politics":  0.80,   # Elections, policy — AI is informed but markets are too
    "sports":    0.65,   # Game outcomes — high variance, AI limited advantage
    "crypto":    0.60,   # Price events — extremely noisy
    "finance":   0.70,   # Earnings, indices — partially predictable
    "weather":   0.55,   # Atmospheric events — AI has limited local data
    "default":   0.75,   # Everything else
}

_SPORTS_KEYWORDS   = {"nba", "nfl", "mlb", "nhl", "fifa", "super bowl", "world cup",
                       "championship", "playoff", "tournament", "match", "game",
                       "mvp", "standings", "league", "team", "player", "score"}
_POLITICS_KEYWORDS = {"election", "president", "congress", "senate", "vote", "poll",
                       "candidate", "party", "democrat", "republican", "primary",
                       "campaign", "governor", "minister", "parliament", "referendum"}
_CRYPTO_KEYWORDS   = {"bitcoin", "btc", "eth", "ethereum", "crypto", "token",
                       "defi", "nft", "blockchain", "altcoin", "stablecoin", "dex"}
_FINANCE_KEYWORDS  = {"stock", "market", "fed", "interest rate", "inflation", "gdp",
                       "earnings", "s&p", "nasdaq", "recession", "ipo", "treasury"}
_WEATHER_KEYWORDS  = {"hurricane", "storm", "tornado", "earthquake", "flood",
                       "temperature", "rainfall", "snowfall", "wildfire", "drought"}


def _detect_category(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in _SPORTS_KEYWORDS):
        return "sports"
    if any(kw in q for kw in _POLITICS_KEYWORDS):
        return "politics"
    if any(kw in q for kw in _CRYPTO_KEYWORDS):
        return "crypto"
    if any(kw in q for kw in _FINANCE_KEYWORDS):
        return "finance"
    if any(kw in q for kw in _WEATHER_KEYWORDS):
        return "weather"
    return "default"


def _apply_bayesian_calibration(
    ai_prob: float,
    market_price: float,
    category: str,
    calibration_map: dict,
) -> float:
    """
    Shrink the AI edge toward the market price based on category confidence.
    calibrated_prob = market_price + calibration_factor × (ai_prob − market_price)
    At factor=1.0 the AI estimate is taken at face value.
    At factor=0.5 we split the difference between AI and market.
    """
    factor = calibration_map.get(category, calibration_map.get("default", 0.75))
    return market_price + factor * (ai_prob - market_price)


class AITradingStrategy(BaseStrategy):
    """
    Superforecaster strategy powered by Claude + optional GPT-4o ensemble.
    Trades when AI probability estimate diverges significantly from market price.
    """

    def __init__(self, *args, news_client: NewsClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.news_client = news_client
        self._anthropic_client: anthropic.AsyncAnthropic | None = None
        self._openai_client = None   # lazy-loaded
        self._estimate_cache: dict[str, tuple[float, dict]] = {}  # slug → (timestamp, estimate)
        self._cache_ttl      = self.config.get("cache_ttl_seconds", 7200)
        self._run_interval   = self.config.get("run_interval_seconds", 3600)
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

        min_edge     = self.config.get("min_edge_pct", 0.12)
        min_price    = self.config.get("min_market_price", 0.10)
        max_price    = self.config.get("max_market_price", 0.90)
        max_per_run  = self.config.get("max_markets_per_run", 8)
        model        = self.config.get("model", "claude-haiku-4-5-20251001")
        ensemble_on  = self.config.get("ensemble_enabled", False)
        openai_model = self.config.get("openai_model", "gpt-4o-mini")
        use_kelly    = self.config.get("use_kelly_sizing", True)
        kelly_frac   = self.config.get("kelly_fraction", 0.25)
        fallback_size = self.config.get("order_size_usdc", 50)

        calibration_map = {**DEFAULT_CALIBRATION, **self.config.get("calibration", {})}

        markets = await self.market_data.get_markets()
        if not markets:
            self.log("No markets available for AI analysis")
            return

        candidates = []
        for m in markets:
            slug = self.market_data.get_slug(m)
            if not slug:
                continue
            if self.order_manager.get_market_order_count(slug) > 0:
                continue

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
            self.log("No new markets to analyse this run (all cached or filtered)")
            return

        candidates.sort(key=lambda x: abs(x[1] - 0.5))
        candidates = candidates[:max_per_run]

        mode = "ensemble (Claude + GPT)" if ensemble_on else "Claude only"
        self.log(f"AI scan: analysing {len(candidates)} markets | {mode} | {model}")

        entered = 0
        for market, mid in candidates:
            slug     = self.market_data.get_slug(market)
            question = self.market_data.get_question(market)
            if not slug or not question:
                continue

            # ── News context ──────────────────────────────────────────────────
            news_lines: list[str] = []
            if self.news_client:
                news_lines = await self.news_client.get_headlines(question, max_articles=5)
            news_context = (
                "\n".join(f"  • {h}" for h in news_lines)
                if news_lines
                else "  (no recent news found — base your estimate on general knowledge)"
            )

            # ── Time description ──────────────────────────────────────────────
            hours_left = self.market_data._hours_to_resolution(market)
            if hours_left is None:
                time_desc = "unknown"
            elif hours_left < 24:
                time_desc = f"{hours_left:.1f} hours"
            elif hours_left < 720:
                time_desc = f"{hours_left / 24:.1f} days"
            else:
                time_desc = f"{hours_left / 720:.1f} months"

            # ── Ask model(s) ─────────────────────────────────────────────────
            prompt_kwargs = dict(
                question=question, price=mid,
                time_desc=time_desc, news_context=news_context,
            )

            try:
                if ensemble_on:
                    claude_est, gpt_est = await asyncio.gather(
                        self._ask_claude(model=model, **prompt_kwargs),
                        self._ask_openai(model=openai_model, **prompt_kwargs),
                        return_exceptions=True,
                    )
                    estimate = self._merge_estimates(claude_est, gpt_est, mid)
                    if estimate is None:
                        self.log(f"Ensemble disagreement on '{question[:40]}' — skipping")
                        continue
                else:
                    estimate = await self._ask_claude(model=model, **prompt_kwargs)
            except Exception as e:
                self.log(f"AI error on '{question[:40]}': {e}", level="warning")
                continue

            if not estimate:
                continue

            self._estimate_cache[slug] = (now, estimate)

            raw_ai_prob = float(estimate.get("probability", mid))
            confidence  = estimate.get("confidence", "low")
            edge_dir    = estimate.get("edge", "pass")
            reasoning   = estimate.get("reasoning", "")

            # ── Bayesian calibration ──────────────────────────────────────────
            category    = _detect_category(question)
            ai_prob     = _apply_bayesian_calibration(raw_ai_prob, mid, category, calibration_map)
            edge        = abs(ai_prob - mid)

            self.log(
                f"'{question[:50]}' [{category}] | "
                f"market={mid:.0%} raw_AI={raw_ai_prob:.0%} calibrated={ai_prob:.0%} "
                f"edge={edge:.0%} [{confidence}] → {edge_dir} | {reasoning[:80]}"
            )

            if edge_dir == "pass" or confidence == "low":
                continue
            if edge < min_edge:
                continue

            # ── Kelly sizing ──────────────────────────────────────────────────
            if use_kelly:
                win_prob = ai_prob if edge_dir == "buy" else (1.0 - ai_prob)
                net_ret  = edge  # rough: edge ≈ expected net return
                order_size = self.capital_manager.kelly_size(
                    self.name, win_prob, net_ret,
                    kelly_fraction=kelly_frac,
                    min_size=10.0,
                    max_size=fallback_size * 2,
                )
            else:
                order_size = fallback_size

            if not self.capital_manager.can_allocate(self.name, order_size):
                self.log("Capital limit reached", level="warning")
                break

            # ── Trade direction ───────────────────────────────────────────────
            bbo = await self.market_data.get_bbo(slug)
            if not bbo:
                continue

            if edge_dir == "buy":
                ask = float(bbo.get("ask", {}).get("price", min(mid + 0.02, 0.99)))
                taker_price = min(round(ask + 0.03, 4), 0.99)
                intent = "ORDER_INTENT_BUY_LONG"
            else:
                bid = float(bbo.get("bid", {}).get("price", max(mid - 0.02, 0.01)))
                no_price    = round(1.0 - bid, 4)
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
                    f"conf={confidence} kelly=${order_size:.0f} | '{question[:40]}'"
                )
                entered += 1
            else:
                self.capital_manager.release(self.name, order_size)

        summary = f"AI trader: {entered} trade(s) placed | {len(candidates)} markets analysed"
        self.log(summary)
        await db.log_to_db("INFO", f"[ai_trader] {summary}")

    # ── Ensemble merge ────────────────────────────────────────────────────────

    def _merge_estimates(self, claude_est, gpt_est, market_price: float) -> dict | None:
        """
        Return a merged estimate only when both models agree on direction.
        Averages probabilities and takes the more conservative confidence.
        Returns None if models disagree or either call failed.
        """
        if isinstance(claude_est, Exception) or isinstance(gpt_est, Exception):
            return None
        if not claude_est or not gpt_est:
            return None

        c_dir = claude_est.get("edge", "pass")
        g_dir = gpt_est.get("edge", "pass")

        if c_dir == "pass" or g_dir == "pass":
            return None
        if c_dir != g_dir:
            return None   # models disagree — stay out

        # Average the probability estimates
        avg_prob = (float(claude_est.get("probability", market_price)) +
                    float(gpt_est.get("probability", market_price))) / 2

        # Use the more conservative confidence of the two
        conf_rank = {"low": 0, "medium": 1, "high": 2}
        c_conf = claude_est.get("confidence", "low")
        g_conf = gpt_est.get("confidence", "low")
        conservative_conf = c_conf if conf_rank[c_conf] <= conf_rank[g_conf] else g_conf

        return {
            "probability": avg_prob,
            "confidence": conservative_conf,
            "edge": c_dir,
            "reasoning": f"Claude: {claude_est.get('reasoning','')[:60]} | GPT: {gpt_est.get('reasoning','')[:60]}",
        }

    # ── Claude call ───────────────────────────────────────────────────────────

    def _get_anthropic(self) -> anthropic.AsyncAnthropic:
        if self._anthropic_client is None:
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
        client = self._get_anthropic()
        prompt = SUPERFORECASTER_PROMPT.format(
            question=question, price=price,
            time_desc=time_desc, news=news_context,
        )
        message = await client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        return json.loads(raw)

    # ── OpenAI call ───────────────────────────────────────────────────────────

    def _get_openai(self):
        if self._openai_client is None:
            try:
                import openai
            except ImportError:
                raise RuntimeError("openai package not installed — run: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable not set")
            self._openai_client = openai.AsyncOpenAI(api_key=api_key)
        return self._openai_client

    async def _ask_openai(
        self,
        model: str,
        question: str,
        price: float,
        time_desc: str,
        news_context: str,
    ) -> dict | None:
        client = self._get_openai()
        prompt = SUPERFORECASTER_PROMPT.format(
            question=question, price=price,
            time_desc=time_desc, news=news_context,
        )
        response = await client.chat.completions.create(
            model=model,
            max_tokens=300,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        return json.loads(raw)
