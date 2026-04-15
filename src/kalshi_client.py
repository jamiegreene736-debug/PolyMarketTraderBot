"""
Kalshi API Client
==================
Async client for the Kalshi prediction market API.
Uses RSA-PSS authentication.

Credentials needed (add to Railway environment variables):
  KALSHI_API_KEY_ID     — your Kalshi key ID
  KALSHI_PRIVATE_KEY    — your RSA private key (PEM format, base64-encoded for env var storage)

How to generate Kalshi keys:
  1. Go to kalshi.com → Account → API Keys
  2. Generate a new RSA key pair
  3. Copy the private key PEM, base64-encode it:
       python3 -c "import base64; print(base64.b64encode(open('key.pem','rb').read()).decode())"
  4. Set KALSHI_PRIVATE_KEY to the base64 output in Railway

Base URL: https://api.elections.kalshi.com/trade-api/v2
"""

import base64
import time
from loguru import logger

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiClient:
    def __init__(self, key_id: str, private_key_b64: str, dry_run: bool = False):
        self.key_id = key_id
        self.dry_run = dry_run
        self.base_url = KALSHI_BASE_URL

        pem_bytes = base64.b64decode(private_key_b64)
        self.private_key = serialization.load_pem_private_key(pem_bytes, password=None)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path: str) -> dict:
        timestamp = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(timestamp, method, path),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    async def get_markets(self, limit: int = 200) -> list:
        path = f"/markets?limit={limit}&status=open"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{self.base_url}/markets",
                headers=self._headers("GET", "/markets"),
                params={"limit": limit, "status": "open"},
            )
            resp.raise_for_status()
            return resp.json().get("markets", [])

    async def get_orderbook(self, ticker: str) -> dict:
        path = f"/markets/{ticker}/orderbook"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{self.base_url}{path}",
                headers=self._headers("GET", path),
            )
            resp.raise_for_status()
            return resp.json()

    async def get_bbo(self, ticker: str) -> dict | None:
        """Returns best bid/offer as normalized 0-1 prices."""
        try:
            book = await self.get_orderbook(ticker)
            yes_bids = book.get("orderbook", {}).get("yes", [])
            yes_asks = book.get("orderbook", {}).get("no", [])

            best_bid = float(yes_bids[0][0]) / 100 if yes_bids else None
            best_ask = 1 - float(yes_asks[0][0]) / 100 if yes_asks else None

            return {"bid": best_bid, "ask": best_ask}
        except Exception as e:
            logger.warning(f"Kalshi BBO failed for {ticker}: {e}")
            return None

    async def place_order(self, ticker: str, action: str, side: str,
                          count: int, yes_price: float) -> dict:
        if self.dry_run:
            logger.info(f"[DRY RUN] Kalshi order: {action} {side} {count}x {ticker} @ {yes_price}")
            return {"order": {"order_id": f"dry_{ticker}", "status": "resting"}}

        path = "/portfolio/orders"
        body = {
            "ticker": ticker,
            "action": action,         # "buy" or "sell"
            "side": side,             # "yes" or "no"
            "count": count,
            "type": "limit",
            "yes_price_dollars": str(round(yes_price, 4)),
        }
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{self.base_url}{path}",
                json=body,
                headers=self._headers("POST", path),
            )
            resp.raise_for_status()
            return resp.json()

    def normalize_title(self, text: str) -> set:
        """Normalize market title to a set of significant words for matching."""
        import re
        stop_words = {"will", "the", "a", "an", "in", "on", "at", "to", "of",
                      "be", "is", "by", "or", "and", "for", "yes", "no", "win",
                      "reach", "above", "below", "before", "after", "than"}
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        return {w for w in words if w not in stop_words and len(w) > 2}

    def match_score(self, poly_question: str, kalshi_title: str) -> float:
        """Word overlap score between a Polymarket question and Kalshi title."""
        poly_words   = self.normalize_title(poly_question)
        kalshi_words = self.normalize_title(kalshi_title)
        if not poly_words or not kalshi_words:
            return 0.0
        overlap = poly_words & kalshi_words
        return len(overlap) / min(len(poly_words), len(kalshi_words))
