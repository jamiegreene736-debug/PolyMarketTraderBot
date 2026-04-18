"""
News Catalyst Strategy
-----------------------
Polls breaking news headlines and uses Claude to identify when a headline
definitively resolves an active Polymarket market. Places aggressive taker
orders before the market reprices.

How it works each run (default: every 120s):
  1. Fetch top US headlines via NewsClient.
  2. For each new headline not seen before, ask Claude: "Does any of these
     active markets get resolved by this headline? Which one, and which side?"
  3. For high-confidence matches, place an aggressive taker order on the
     winning side (YES or NO).
  4. Cache processed headlines to avoid re-trading on the same news.

Required env vars:
  NEWS_API_KEY       — NewsAPI key (newsapi.org)
  ANTHROPIC_API_KEY  — for Claude classification
"""

import asyncio
import json
import os
import re

import anthropic

from src.strategies.base import BaseStrategy
from src.news_client import NewsClient
from src import database as db


# Headlines that can't possibly resolve a market (sports recaps aside, most
# prediction markets care about discrete events). We still send all headlines
# to Claude, but we can pre-filter truly noisy ones with keyword heuristics.
_NOISE_PREFIXES = (
    "opinion:", "analysis:", "explainer:", "photos:", "video:",
    "watch:", "listen:", "live updates", "live blog",
)


CLASSIFIER_PROMPT = """\
You are identifying whether a breaking news headline definitively resolves a
prediction market. Be strict — only return a match if the headline makes the
market outcome obvious with high certainty.

Headline: {headline}
Source: {source}
Published: {published}

Active markets (slug — question):
{market_list}

For each market the headline clearly resolves, output an entry. If no market
is clearly resolved, output an empty list.

Respond with ONLY a JSON object — no markdown, no prose outside the JSON:
{{
  "matches": [
    {{
      "slug": "<market slug exactly as listed>",
      "direction": "<yes|no>",
      "confidence": "<low|medium|high>",
      "reasoning": "<one short sentence>"
    }}
  ]
}}

Rules:
- "yes" = the YES side wins. "no" = the NO side wins.
- Only return "high" confidence when the headline states the outcome as fact
  (e.g. "Biden signs bill", "Team A wins championship", "Company announces X").
- "low" confidence matches should NOT be included — return an empty list instead.
- If the headline is speculation, prediction, or preview, do NOT match.
- Return at most 3 matches per headline.\
"""


class NewsCatalystStrategy(BaseStrategy):
    """
    Breaking-news-driven trader. Matches headlines to active markets via Claude
    and takes aggressive positions on high-confidence resolutions.
    """

    def __init__(self, *args, news_client: NewsClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.news_client = news_client
        self._anthropic_client: anthropic.AsyncAnthropic | None = None
        self._last_run: float = 0
        # Hash of processed headline titles so we never re-trade the same news.
        self._processed_headlines: set[str] = set()
        # Slug cooldown to avoid trading the same market on multiple headlines
        # in rapid succession (e.g., 3 outlets reporting the same event).
        self._slug_cooldown: dict[str, float] = {}

    async def run(self):
        if not self.enabled:
            return

        if self.news_client is None or not self.news_client.api_key:
            self.log(
                "NEWS_API_KEY not set — news_catalyst cannot poll headlines",
                level="warning",
            )
            return

        if not os.getenv("ANTHROPIC_API_KEY"):
            self.log(
                "ANTHROPIC_API_KEY not set — news_catalyst cannot classify headlines",
                level="warning",
            )
            return

        run_interval = self.config.get("run_interval_seconds", 120)
        now = asyncio.get_event_loop().time()
        if (now - self._last_run) < run_interval:
            return
        self._last_run = now

        order_size      = self.config.get("order_size_usdc", 5)
        max_headlines   = self.config.get("max_headlines_per_run", 10)
        max_markets     = self.config.get("max_markets_in_prompt", 40)
        min_confidence  = self.config.get("min_confidence", "high")
        slug_cooldown_s = self.config.get("slug_cooldown_seconds", 3600)
        model           = self.config.get("model", "claude-haiku-4-5-20251001")
        country         = self.config.get("news_country", "us")

        # ── 1. Fetch top headlines ────────────────────────────────────────
        headlines = await self.news_client.get_top_headlines(
            country=country, max_articles=max_headlines
        )
        if not headlines:
            return

        new_headlines = [
            h for h in headlines
            if h["title"] not in self._processed_headlines
            and not any(h["title"].lower().startswith(p) for p in _NOISE_PREFIXES)
        ]
        if not new_headlines:
            return

        # ── 2. Prepare candidate markets ──────────────────────────────────
        all_markets = await self.market_data.get_markets()
        if not all_markets:
            return

        # Restrict to non-resolved, reasonably-priced markets to keep the prompt
        # focused and the decisions cheap. Headlines usually resolve things in
        # the uncertain zone; $0.99/$0.01 markets are already decided.
        candidate_markets = []
        for m in all_markets:
            slug = self.market_data.get_slug(m)
            question = self.market_data.get_question(m)
            if not slug or not question:
                continue
            if (now - self._slug_cooldown.get(slug, 0)) < slug_cooldown_s:
                continue
            prices = self.market_data.get_outcome_prices(m)
            if prices:
                yes = prices[0]
                if yes < 0.02 or yes > 0.98:
                    continue
            candidate_markets.append((slug, question))

        if not candidate_markets:
            return

        # Sort by how "close to 50/50" the market is — those are the juiciest
        # news-driven opportunities.
        candidate_markets = candidate_markets[:max_markets]
        market_list_str = "\n".join(
            f"  - {slug} — {q[:120]}" for slug, q in candidate_markets
        )

        self.log(
            f"Scanning {len(new_headlines)} new headlines against "
            f"{len(candidate_markets)} markets"
        )

        # ── 3. Classify each headline via Claude (parallel) ───────────────
        classify_tasks = [
            self._classify_headline(h, market_list_str, model)
            for h in new_headlines
        ]
        results = await asyncio.gather(*classify_tasks, return_exceptions=True)

        entered = 0
        for headline, result in zip(new_headlines, results):
            self._processed_headlines.add(headline["title"])

            if isinstance(result, Exception) or not result:
                continue

            matches = result.get("matches", [])
            if not matches:
                continue

            for match in matches[:3]:
                slug = match.get("slug", "")
                direction = match.get("direction", "").lower()
                confidence = match.get("confidence", "low").lower()
                reasoning = match.get("reasoning", "")

                if direction not in ("yes", "no"):
                    continue
                if confidence != min_confidence and confidence != "high":
                    continue
                if slug not in {s for s, _ in candidate_markets}:
                    continue  # Claude hallucinated a slug
                if (now - self._slug_cooldown.get(slug, 0)) < slug_cooldown_s:
                    continue

                placed = await self._place_catalyst_trade(
                    slug=slug,
                    direction=direction,
                    order_size=order_size,
                    headline=headline,
                    reasoning=reasoning,
                )
                if placed:
                    self._slug_cooldown[slug] = now
                    entered += 1

        if entered:
            self.log(f"News catalyst: {entered} trade(s) placed from breaking headlines")
            await db.log_to_db("INFO", f"[news_catalyst] {entered} trades from news")

        # ── 4. Keep processed-headlines set bounded ───────────────────────
        if len(self._processed_headlines) > 500:
            # Drop oldest half by re-creating from a slice of the most recent
            # (order is not preserved in set, but this caps memory growth).
            self._processed_headlines = set(
                list(self._processed_headlines)[-250:]
            )

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _place_catalyst_trade(
        self,
        slug: str,
        direction: str,
        order_size: float,
        headline: dict,
        reasoning: str,
    ) -> bool:
        bbo = await self.market_data.get_bbo(slug)
        if not bbo:
            return False

        try:
            best_bid = float(bbo.get("bid", {}).get("price", 0))
            best_ask = float(bbo.get("ask", {}).get("price", 1))
        except (TypeError, ValueError):
            return False

        # Aggressive taker — we want to beat the market's repricing.
        if direction == "yes":
            intent = "ORDER_INTENT_BUY_LONG"
            taker_price = min(round(best_ask + 0.03, 4), 0.99)
        else:
            intent = "ORDER_INTENT_BUY_SHORT"
            no_price    = round(1.0 - best_bid, 4)
            taker_price = min(round(no_price + 0.03, 4), 0.99)

        shares = round(order_size / max(taker_price, 0.01), 2)

        if not self.capital_manager.can_allocate(self.name, order_size):
            self.log("Capital limit reached", level="warning")
            return False
        if not self.capital_manager.allocate(self.name, order_size):
            return False

        self.log(
            f"NEWS TRADE {intent} {shares:.1f}x @ ${taker_price:.4f} | "
            f"slug={slug} | headline='{headline['title'][:80]}' | "
            f"reason='{reasoning[:80]}'"
        )

        oid = await self.order_manager.place_order(
            market_slug=slug,
            question=headline["title"][:100],
            intent=intent,
            price=taker_price,
            quantity=shares,
            strategy=self.name,
            tif="TIME_IN_FORCE_FILL_OR_KILL",
        )
        if oid:
            return True
        self.capital_manager.release(self.name, order_size)
        return False

    def _get_anthropic(self) -> anthropic.AsyncAnthropic:
        if self._anthropic_client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._anthropic_client

    async def _classify_headline(
        self, headline: dict, market_list_str: str, model: str
    ) -> dict | None:
        """Return {"matches": [...]} or None on any failure."""
        client = self._get_anthropic()
        prompt = CLASSIFIER_PROMPT.format(
            headline=headline["title"],
            source=headline.get("source", "Unknown"),
            published=headline.get("published", ""),
            market_list=market_list_str,
        )
        try:
            message = await client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
                timeout=20.0,
            )
        except Exception as e:
            self.log(f"Claude error classifying '{headline['title'][:60]}': {e}",
                     level="warning")
            return None

        raw = message.content[0].text.strip()
        # Strip markdown fences if Claude added any
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        # Sometimes Claude returns prose before the JSON; extract the first {...}.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            self.log(f"Could not parse Claude JSON: {raw[:200]}", level="warning")
            return None
