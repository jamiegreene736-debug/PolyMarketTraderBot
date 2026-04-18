"""
News Client
-----------
Fetches relevant headlines from NewsAPI.org for market context.
Used by AITradingStrategy to give Claude real-world information
before it estimates probabilities.

Requires: NEWS_API_KEY env var (free tier at newsapi.org)
Free tier: 100 requests/day, development use only.
"""

import asyncio
import re
import httpx
from loguru import logger

# Common words to strip when building a search query from a market question
_STOP_WORDS = {
    "will", "the", "a", "an", "in", "be", "of", "to", "is", "are",
    "was", "were", "have", "has", "had", "do", "does", "did", "can",
    "could", "would", "should", "may", "might", "that", "this", "for",
    "on", "at", "by", "from", "with", "about", "or", "and", "if",
    "not", "no", "who", "what", "when", "where", "which", "how",
    "any", "ever", "there", "their", "its", "it", "over", "under",
    "more", "most", "than", "then", "before", "after", "between",
    "win", "lose", "beat", "vs", "versus",
}


def _search_query(question: str, max_words: int = 6) -> str:
    """Extract the most meaningful keywords from a market question."""
    words = re.sub(r"[^\w\s]", "", question).split()
    keywords = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 2]
    return " ".join(keywords[:max_words])


class NewsClient:
    """Fetches relevant headlines from NewsAPI for a given market question."""

    BASE_URL = "https://newsapi.org/v2/everything"
    TOP_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: dict[str, tuple[float, list[str]]] = {}
        self._ttl = 3600  # cache headlines for 1 hour
        # Separate cache for top headlines (per-country) with a short TTL so
        # news_catalyst can react quickly to breaking stories.
        self._top_cache: dict[str, tuple[float, list[dict]]] = {}
        self._top_ttl = 60

    async def get_headlines(self, question: str, max_articles: int = 5) -> list[str]:
        """
        Return a list of recent headline strings relevant to a market question.
        Returns [] if API key is missing, quota is exceeded, or the call fails.
        """
        if not self.api_key:
            return []

        query = _search_query(question)
        if not query:
            return []

        now = asyncio.get_event_loop().time()
        cached = self._cache.get(query)
        if cached and (now - cached[0]) < self._ttl:
            return cached[1]

        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(
                    self.BASE_URL,
                    params={
                        "q": query,
                        "apiKey": self.api_key,
                        "pageSize": max_articles,
                        "sortBy": "publishedAt",
                        "language": "en",
                    },
                )
                if resp.status_code == 429:
                    logger.warning("NewsAPI rate limit hit — skipping news context")
                    return []
                resp.raise_for_status()
                data = resp.json()

                headlines = []
                for article in data.get("articles", [])[:max_articles]:
                    title = article.get("title", "")
                    source = article.get("source", {}).get("name", "Unknown")
                    published = (article.get("publishedAt") or "")[:10]
                    if title and title != "[Removed]":
                        headlines.append(f"[{published}] {source}: {title}")

                self._cache[query] = (now, headlines)
                return headlines

        except Exception as e:
            logger.debug(f"NewsAPI error for '{query}': {e}")
            return []

    async def get_top_headlines(
        self, country: str = "us", max_articles: int = 20
    ) -> list[dict]:
        """
        Return recent top headlines as structured dicts:
          {"title": str, "source": str, "published": str, "url": str}

        Used by news_catalyst to find potential market-resolving news.
        Returns [] on any error — this is best-effort alpha, not critical path.
        """
        if not self.api_key:
            return []

        now = asyncio.get_event_loop().time()
        cached = self._top_cache.get(country)
        if cached and (now - cached[0]) < self._top_ttl:
            return cached[1]

        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(
                    self.TOP_HEADLINES_URL,
                    params={
                        "country": country,
                        "apiKey": self.api_key,
                        "pageSize": max_articles,
                    },
                )
                if resp.status_code == 429:
                    logger.warning("NewsAPI top-headlines rate limit hit")
                    return []
                resp.raise_for_status()
                data = resp.json()

                headlines: list[dict] = []
                for article in data.get("articles", [])[:max_articles]:
                    title = article.get("title", "")
                    if not title or title == "[Removed]":
                        continue
                    headlines.append({
                        "title":     title,
                        "source":    article.get("source", {}).get("name", "Unknown"),
                        "published": (article.get("publishedAt") or "")[:19],
                        "url":       article.get("url", ""),
                    })

                self._top_cache[country] = (now, headlines)
                return headlines

        except Exception as e:
            logger.debug(f"NewsAPI top-headlines error: {e}")
            return []
