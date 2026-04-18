"""
Polymarket Client — py-clob-client (polymarket.com) + Gamma API
----------------------------------------------------------------
Market data  → Gamma API (https://gamma-api.polymarket.com), no auth.
Trading      → py-clob-client (https://clob.polymarket.com), L2-signed.

Required env vars:
  POLY_PRIVATE_KEY      wallet private key (0x…)
  POLY_API_KEY          CLOB API key (UUID)
  POLY_API_SECRET       CLOB API secret
  POLY_API_PASSPHRASE   CLOB API passphrase

Intent ↔ token mapping:
  ORDER_INTENT_BUY_LONG  → BUY YES token  (clobTokenIds[0])
  ORDER_INTENT_BUY_SHORT → BUY NO  token  (clobTokenIds[1])

Public interface is identical to the old polymarket_us wrapper so
all strategies, order_manager, and market_data work without changes.
"""

import asyncio
import json
import os
import httpx
from loguru import logger
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds, OrderArgs, BalanceAllowanceParams, AssetType, OpenOrderParams,
)
from py_clob_client.order_builder.constants import BUY

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID  = 137  # Polygon mainnet


class PolymarketClient:
    """
    Async wrapper around py-clob-client + Gamma API.
    py-clob-client is synchronous; every call is wrapped in asyncio.to_thread()
    so it doesn't block the event loop.
    """

    def __init__(self, api_key: str, api_secret: str, api_passphrase: str,
                 private_key: str, funder_address: str = "", dry_run: bool = False):
        self.api_key        = api_key
        self.api_secret     = api_secret
        self.api_passphrase = api_passphrase
        self.private_key    = private_key
        self.funder_address = funder_address  # Gnosis Safe / proxy wallet that holds USDC
        self.dry_run        = dry_run
        self._client: ClobClient | None = None

        # py-clob-client (requests) picks up HTTPS_PROXY env var automatically,
        # routing CLOB order traffic through the proxy to bypass geoblocks.
        # The Gamma API (httpx below) is not geoblocked so needs no proxy.
        self._http = httpx.AsyncClient(
            timeout=12,
            headers={"User-Agent": "polymarket-bot/1.0"},
        )
        # slug → {"condition_id": str, "yes_token_id": str, "no_token_id": str}
        self._slug_tokens: dict[str, dict] = {}
        # condition_id → slug (reverse map for syncing open orders)
        self._cid_to_slug: dict[str, str] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self):
        creds = ApiCreds(
            api_key        = self.api_key,
            api_secret     = self.api_secret,
            api_passphrase = self.api_passphrase,
        )

        # If a proxy/funder wallet address is provided, use POLY_PROXY signing
        # (signature_type=1).  This tells the CLOB to check the proxy wallet's
        # USDC balance rather than the raw EOA signer's balance, so funds can
        # stay in the user's Polymarket account without being moved to the EOA.
        sig_type = None
        funder   = None
        if self.funder_address:
            sig_type = 1          # POLY_PROXY
            funder   = self.funder_address
            logger.info(
                f"CLOB proxy-wallet mode: funder={funder} sig_type={sig_type}"
            )

        self._client = await asyncio.to_thread(
            ClobClient,
            CLOB_HOST,
            chain_id       = CHAIN_ID,
            key            = self.private_key,
            creds          = creds,
            signature_type = sig_type,
            funder         = funder,
        )
        logger.info(f"Polymarket CLOB client connected (dry_run={self.dry_run})")

        # ── Account diagnostics (logged once at startup) ──────────────────────
        # Helps us understand exactly what the CLOB sees for this account:
        # which wallets/keys are registered, and where the balance actually lives.
        try:
            api_keys_resp = await asyncio.to_thread(self._client.get_api_keys)
            logger.info(f"CLOB get_api_keys: {api_keys_resp}")
        except Exception as e:
            logger.warning(f"CLOB get_api_keys failed: {e}")

        for stype, label in [(0, "EOA"), (1, "POLY_PROXY"), (2, "POLY_GNOSIS_SAFE")]:
            try:
                bal = await asyncio.to_thread(
                    self._client.get_balance_allowance,
                    BalanceAllowanceParams(
                        asset_type=AssetType.COLLATERAL,
                        signature_type=stype,
                    ),
                )
                logger.info(f"CLOB balance sig_type={stype} ({label}): {bal}")
            except Exception as e:
                logger.warning(f"CLOB balance sig_type={stype} ({label}) failed: {e}")

    async def setup_allowances(self):
        """
        Approve the CLOB exchange contracts to spend USDC and CTF tokens on behalf
        of the signer wallet.  Must be called once after funding the signer address.
        Safe to call on every startup — the CLOB is idempotent (no-op if already set).
        """
        try:
            await asyncio.to_thread(
                self._client.update_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            logger.info("CLOB allowance set: COLLATERAL (USDC)")
        except Exception as e:
            logger.warning(f"setup_allowances (COLLATERAL) failed: {e}")

        try:
            await asyncio.to_thread(
                self._client.update_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL),
            )
            logger.info("CLOB allowance set: CONDITIONAL (CTF tokens)")
        except Exception as e:
            logger.warning(f"setup_allowances (CONDITIONAL) failed: {e}")

    async def close(self):
        await self._http.aclose()

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _to_dict(self, obj) -> dict:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {}

    def _to_list(self, obj) -> list:
        if obj is None:
            return []
        if isinstance(obj, list):
            return [self._to_dict(m) for m in obj]
        if isinstance(obj, dict):
            for key in ("markets", "data", "results", "items"):
                if key in obj and isinstance(obj[key], list):
                    return [self._to_dict(m) for m in obj[key]]
        return []

    # ── Token cache ───────────────────────────────────────────────────────────

    def _cache_market_tokens(self, m: dict):
        """Extract clobTokenIds from a Gamma market dict and cache slug→tokens."""
        slug    = m.get("slug", "")
        cid     = m.get("conditionId") or m.get("condition_id") or ""
        raw_ids = m.get("clobTokenIds")
        if not slug or not raw_ids:
            return
        try:
            ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            if len(ids) >= 2:
                self._slug_tokens[slug] = {
                    "condition_id":  cid,
                    "yes_token_id":  str(ids[0]),
                    "no_token_id":   str(ids[1]),
                }
                if cid:
                    self._cid_to_slug[cid] = slug
        except Exception:
            pass

    async def _ensure_tokens(self, slug: str) -> dict | None:
        """Return cached slug→token entry, fetching from Gamma API if missing."""
        if slug not in self._slug_tokens:
            await self.get_market(slug)
        return self._slug_tokens.get(slug)

    def _asset_id_to_slug(self, asset_id: str) -> str:
        for slug, entry in self._slug_tokens.items():
            if asset_id in (entry.get("yes_token_id"), entry.get("no_token_id")):
                return slug
        return ""

    def _derive_intent(self, order: dict, asset_id: str) -> str:
        for entry in self._slug_tokens.values():
            if asset_id == entry.get("yes_token_id"):
                return "ORDER_INTENT_BUY_LONG"
            if asset_id == entry.get("no_token_id"):
                return "ORDER_INTENT_BUY_SHORT"
        side = (order.get("side") or "").upper()
        return "ORDER_INTENT_BUY_LONG" if side == "BUY" else "ORDER_INTENT_BUY_SHORT"

    # ── Retry ─────────────────────────────────────────────────────────────────

    async def _retry(self, coro_fn, retries: int = 3, delay: float = 2.0):
        for attempt in range(retries):
            try:
                return await coro_fn()
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"API call failed (attempt {attempt+1}/{retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"API call failed after {retries} attempts: {e}")
                    raise

    # ── Market Data (Gamma API, no auth) ─────────────────────────────────────

    async def get_markets(self, **kwargs) -> list:
        """
        Fetch active markets from the Gamma public API.
        Paginates up to 500 markets. Populates slug→token cache as a side-effect.
        """
        all_markets: list = []
        offset = 0
        limit  = 100

        while len(all_markets) < 500:
            try:
                resp = await self._http.get(
                    f"{GAMMA_API}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit":  limit,
                        "offset": offset,
                    },
                )
                resp.raise_for_status()
                page: list = resp.json()
            except Exception as e:
                logger.error(f"Gamma get_markets failed (offset={offset}): {e}")
                break

            if not page:
                break

            for m in page:
                self._cache_market_tokens(m)

            all_markets.extend(page)
            if len(page) < limit:
                break
            offset += limit

        logger.info(f"get_markets: {len(all_markets)} markets from Gamma API")
        return all_markets

    async def get_market(self, slug: str) -> dict:
        """Fetch a single market by slug from Gamma API."""
        try:
            resp = await self._http.get(
                f"{GAMMA_API}/markets",
                params={"slug": slug},
            )
            resp.raise_for_status()
            data = resp.json()
            markets = data if isinstance(data, list) else data.get("markets", [])
            if markets:
                m = markets[0] if isinstance(markets[0], dict) else self._to_dict(markets[0])
                self._cache_market_tokens(m)
                return m
        except Exception as e:
            logger.warning(f"get_market({slug}) failed: {e}")
        return {}

    async def get_order_book(self, slug: str) -> dict:
        tokens = await self._ensure_tokens(slug)
        if not tokens:
            return {}
        try:
            ob = await asyncio.to_thread(
                self._client.get_order_book, tokens["yes_token_id"]
            )
            return {
                "bids": [{"price": b.price, "size": b.size} for b in (ob.bids or [])],
                "asks": [{"price": a.price, "size": a.size} for a in (ob.asks or [])],
            }
        except Exception as e:
            logger.debug(f"get_order_book({slug}): {e}")
            return {}

    async def get_bbo(self, slug: str) -> dict:
        """
        Best bid/offer for the YES token of a market.
        Primary: live CLOB order book top-of-book.
        Fallback: midpoint ± 1¢.
        """
        tokens = await self._ensure_tokens(slug)
        if not tokens:
            return {}

        try:
            ob    = await asyncio.to_thread(
                self._client.get_order_book, tokens["yes_token_id"]
            )
            bids  = ob.bids or []
            asks  = ob.asks or []
            bid_p = float(bids[0].price) if bids else None
            ask_p = float(asks[0].price) if asks else None
            if bid_p is not None and ask_p is not None:
                return {"bid": {"price": bid_p}, "ask": {"price": ask_p}}
        except Exception as e:
            logger.debug(f"get_bbo({slug}) orderbook: {e}")

        # Fallback: midpoint
        try:
            mid_raw = await asyncio.to_thread(
                self._client.get_midpoint, tokens["yes_token_id"]
            )
            mid = float(
                mid_raw.get("mid", 0.5) if isinstance(mid_raw, dict) else mid_raw
            )
            return {
                "bid": {"price": round(max(0.01, mid - 0.01), 4)},
                "ask": {"price": round(min(0.99, mid + 0.01), 4)},
            }
        except Exception as e2:
            logger.debug(f"get_bbo({slug}) midpoint: {e2}")
            return {}

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_balance(self) -> dict:
        """
        Return USDC collateral balance as dollar float.
        Polymarket CLOB returns balance as a string in micro-USDC (6 decimals)
        e.g. {"balance": "36000000.0", "allowance": "..."} → $36.00
        """
        try:
            raw = await asyncio.to_thread(
                self._client.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            # Log raw on first call so we can see the exact shape
            logger.info(f"get_balance raw response: {raw}")

            if not isinstance(raw, dict):
                logger.warning(f"get_balance: unexpected response type {type(raw)}: {raw}")
                return {"balance": 0.0, "availableBalance": 0.0}

            # Try known key names
            bal_raw = None
            for key in ("balance", "availableBalance", "USDC", "usdc", "collateral"):
                v = raw.get(key)
                if v is not None:
                    try:
                        bal_raw = float(v)
                        break
                    except (TypeError, ValueError):
                        pass

            if bal_raw is None:
                logger.warning(f"get_balance: no balance key found in {list(raw.keys())}")
                return {"balance": 0.0, "availableBalance": 0.0}

            # Polymarket returns micro-USDC (6 decimals): 36000000 → $36.00
            # Guard: if < 100000 it's already dollar-denominated
            bal = bal_raw / 1_000_000 if bal_raw >= 100_000 else bal_raw
            logger.info(f"get_balance: raw={bal_raw} → ${bal:.2f} USDC")
            return {"balance": bal, "availableBalance": bal}
        except Exception as e:
            logger.warning(f"get_balance failed: {e}")
            return {"balance": 0.0, "availableBalance": 0.0}

    async def get_positions(self) -> list:
        return []

    async def get_activities(self) -> list:
        return []

    # ── Orders ────────────────────────────────────────────────────────────────

    async def get_open_orders(self) -> list:
        """
        Fetch open orders from the CLOB and normalise for order_manager.sync_from_exchange.
        """
        try:
            raw    = await asyncio.to_thread(self._client.get_orders, OpenOrderParams())
            orders = raw if isinstance(raw, list) else (raw or {}).get("data", []) or []
            result = []
            for o in orders:
                od       = o if isinstance(o, dict) else (vars(o) if hasattr(o, "__dict__") else {})
                asset_id = str(od.get("asset_id") or od.get("assetId") or "")
                slug     = self._asset_id_to_slug(asset_id) or asset_id
                oid      = od.get("id") or od.get("orderId") or od.get("order_id") or ""
                if not oid:
                    continue
                result.append({
                    "id":          oid,
                    "market_slug": slug,
                    "intent":      self._derive_intent(od, asset_id),
                    "price":       float(od.get("price", 0) or 0),
                    "quantity":    float(
                        od.get("size_matched") or od.get("original_size")
                        or od.get("size") or 0
                    ),
                })
            return result
        except Exception as e:
            logger.warning(f"get_open_orders failed: {e}")
            return []

    async def place_order(
        self,
        market_slug: str,
        intent: str,
        price: float,
        quantity: float,
        order_type: str = "ORDER_TYPE_LIMIT",
        tif: str = "TIME_IN_FORCE_GOOD_TILL_CANCEL",
    ) -> dict:
        if self.dry_run:
            fake_id = f"dry_run_{market_slug}_{intent}_{price:.4f}"
            logger.info(
                f"[DRY RUN] {intent} {quantity:.1f}x @ ${price:.4f} on {market_slug}"
            )
            return {"id": fake_id}

        tokens = await self._ensure_tokens(market_slug)
        if not tokens:
            logger.error(f"place_order: no token mapping for '{market_slug}'")
            return {}

        if intent == "ORDER_INTENT_BUY_LONG":
            token_id = tokens["yes_token_id"]
        elif intent == "ORDER_INTENT_BUY_SHORT":
            token_id = tokens["no_token_id"]
        else:
            logger.error(f"place_order: unknown intent '{intent}'")
            return {}

        args   = OrderArgs(
            token_id = token_id,
            price    = round(float(price), 4),
            size     = float(quantity),
            side     = BUY,
        )
        raw    = await asyncio.to_thread(self._client.create_and_post_order, args)
        result = raw if isinstance(raw, dict) else (vars(raw) if hasattr(raw, "__dict__") else {})
        oid    = (
            result.get("orderID") or result.get("order_id") or result.get("id") or ""
        )
        logger.info(
            f"Order placed: {intent} {quantity:.1f}x @ ${price:.4f} "
            f"on {market_slug} → id={oid}"
        )
        return {"id": oid, "raw": result}

    async def cancel_order(self, order_id: str, market_slug: str = "") -> bool:
        if self.dry_run:
            logger.info(f"[DRY RUN] Cancel {order_id}")
            return True
        try:
            await asyncio.to_thread(self._client.cancel, order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.warning(f"cancel_order({order_id}): {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        if self.dry_run:
            logger.info("[DRY RUN] Cancel all orders")
            return True
        try:
            await asyncio.to_thread(self._client.cancel_all)
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.warning(f"cancel_all_orders: {e}")
            return False
