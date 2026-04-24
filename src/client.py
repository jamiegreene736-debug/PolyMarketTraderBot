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
import re
import httpx
from loguru import logger
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds, OrderArgs, BalanceAllowanceParams, AssetType, OpenOrderParams,
    OrderType,
)
from py_clob_client.order_builder.constants import BUY, SELL

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CHAIN_ID  = 137  # Polygon mainnet
SIG_TYPE_EOA = 0
SIG_TYPE_POLY_PROXY = 1
SIG_TYPE_POLY_GNOSIS_SAFE = 2


class PolymarketClient:
    """
    Async wrapper around py-clob-client + Gamma API.
    py-clob-client is synchronous; every call is wrapped in asyncio.to_thread()
    so it doesn't block the event loop.
    """

    def __init__(self, api_key: str, api_secret: str, api_passphrase: str,
                 private_key: str, funder_address: str = "", signature_type: int | None = None,
                 dry_run: bool = False):
        self.api_key        = api_key
        self.api_secret     = api_secret
        self.api_passphrase = api_passphrase
        self.private_key    = private_key
        self.funder_address = funder_address  # Gnosis Safe / proxy wallet that holds USDC
        self.configured_signature_type = signature_type
        self.dry_run        = dry_run
        self.signature_type = SIG_TYPE_EOA
        self.signer_address = ""
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
        # Last-known-good snapshots. During upstream API hiccups, callers should
        # see stale-but-real account data instead of dangerous zero/empty values.
        self._last_balance: dict | None = None
        self._last_markets: list[dict] = []
        self._last_positions: dict[tuple, list] = {}
        self._last_closed_positions: dict[tuple, list] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _build_client(
        self,
        creds: ApiCreds,
        signature_type: int | None = None,
        funder: str | None = None,
    ) -> ClobClient:
        return await asyncio.to_thread(
            ClobClient,
            CLOB_HOST,
            chain_id=CHAIN_ID,
            key=self.private_key,
            creds=creds,
            signature_type=signature_type,
            funder=funder,
        )

    def _extract_balance_usdc(self, raw: dict | None) -> float:
        if not isinstance(raw, dict):
            return 0.0

        bal_raw = None
        for key in ("balance", "availableBalance", "USDC", "usdc", "collateral"):
            value = raw.get(key)
            if value is None:
                continue
            try:
                bal_raw = float(value)
                break
            except (TypeError, ValueError):
                continue

        if bal_raw is None:
            return 0.0

        return bal_raw / 1_000_000 if bal_raw >= 100_000 else bal_raw

    def _extract_proxy_address(self, raw: dict | None) -> str:
        if not isinstance(raw, dict):
            return ""

        for key in (
            "proxyWallet",
            "proxy_wallet",
            "proxyAddress",
            "funder",
            "makerAddress",
            "maker_address",
            "walletAddress",
            "address",
        ):
            value = raw.get(key)
            if isinstance(value, str) and value.startswith("0x"):
                return value

        return ""

    def _sig_label(self, signature_type: int) -> str:
        return {
            SIG_TYPE_EOA: "EOA",
            SIG_TYPE_POLY_PROXY: "POLY_PROXY",
            SIG_TYPE_POLY_GNOSIS_SAFE: "POLY_GNOSIS_SAFE",
        }.get(signature_type, f"UNKNOWN({signature_type})")

    async def connect(self):
        creds = ApiCreds(
            api_key        = self.api_key,
            api_secret     = self.api_secret,
            api_passphrase = self.api_passphrase,
        )

        effective_funder = self.funder_address.strip()
        preferred_sig_type = self.configured_signature_type

        if effective_funder and preferred_sig_type is not None:
            self._client = await self._build_client(
                creds,
                signature_type=preferred_sig_type,
                funder=effective_funder,
            )
            self.signature_type = preferred_sig_type
            self.funder_address = effective_funder
            self.signer_address = self._client.get_address() or ""
            logger.info(
                "Polymarket CLOB client connected "
                f"(dry_run={self.dry_run}, mode={self._sig_label(self.signature_type)}, "
                f"signer={self.signer_address}, funder={self.funder_address})"
            )
            return

        probe_client = await self._build_client(creds)
        self.signer_address = probe_client.get_address() or ""

        candidates: list[tuple[float, int, str]] = []
        for stype in (
            SIG_TYPE_EOA,
            SIG_TYPE_POLY_PROXY,
            SIG_TYPE_POLY_GNOSIS_SAFE,
        ):
            try:
                raw = await asyncio.to_thread(
                    probe_client.get_balance_allowance,
                    BalanceAllowanceParams(
                        asset_type=AssetType.COLLATERAL,
                        signature_type=stype,
                    ),
                )
                logger.info(
                    f"CLOB balance probe sig_type={stype} ({self._sig_label(stype)}): {raw}"
                )
                discovered = self._extract_proxy_address(raw)
                balance = self._extract_balance_usdc(raw)

                if stype == SIG_TYPE_EOA:
                    candidates.append((balance, stype, self.signer_address))
                elif discovered:
                    candidates.append((balance, stype, discovered))
                elif effective_funder:
                    candidates.append((balance, stype, effective_funder))
            except Exception as e:
                logger.warning(
                    f"CLOB balance probe sig_type={stype} ({self._sig_label(stype)}) failed: {e}"
                )

        if not effective_funder:
            try:
                api_keys_resp = await asyncio.to_thread(probe_client.get_api_keys)
                logger.info(f"CLOB get_api_keys raw: {api_keys_resp}")
                discovered = self._extract_proxy_address(api_keys_resp)
                if discovered:
                    effective_funder = discovered
                    for stype in (SIG_TYPE_POLY_PROXY, SIG_TYPE_POLY_GNOSIS_SAFE):
                        candidates.append((0.0, stype, discovered))
            except Exception as e:
                logger.warning(f"CLOB get_api_keys failed: {e}")

        if preferred_sig_type is None and effective_funder:
            # If the user supplied a funder but not a type, prefer Gnosis Safe
            # first because Polymarket's docs call it the common path for newer
            # proxy-backed accounts.
            candidates.append((0.0, SIG_TYPE_POLY_GNOSIS_SAFE, effective_funder))
            candidates.append((0.0, SIG_TYPE_POLY_PROXY, effective_funder))

        if effective_funder:
            trial_types = (
                [preferred_sig_type]
                if preferred_sig_type is not None
                else [SIG_TYPE_POLY_GNOSIS_SAFE, SIG_TYPE_POLY_PROXY]
            )
            for stype in trial_types:
                if stype is None or stype == SIG_TYPE_EOA:
                    continue
                try:
                    funded_client = await self._build_client(
                        creds,
                        signature_type=stype,
                        funder=effective_funder,
                    )
                    raw = await asyncio.to_thread(
                        funded_client.get_balance_allowance,
                        BalanceAllowanceParams(
                            asset_type=AssetType.COLLATERAL,
                            signature_type=stype,
                        ),
                    )
                    balance = self._extract_balance_usdc(raw)
                    logger.info(
                        "CLOB funded probe "
                        f"sig_type={stype} ({self._sig_label(stype)}) "
                        f"funder={effective_funder}: {raw}"
                    )
                    candidates.append((balance, stype, effective_funder))
                except Exception as e:
                    logger.warning(
                        "CLOB funded probe failed "
                        f"sig_type={stype} ({self._sig_label(stype)}) "
                        f"funder={effective_funder}: {e}"
                    )

        if candidates:
            # Highest observed balance wins. For balance ties, prefer the more
            # modern proxy-safe path, then proxy, then EOA.
            balance, chosen_sig_type, chosen_funder = max(
                candidates,
                key=lambda item: (
                    item[0],
                    2 if item[1] == SIG_TYPE_POLY_GNOSIS_SAFE else 1 if item[1] == SIG_TYPE_POLY_PROXY else 0,
                ),
            )
            if chosen_sig_type == SIG_TYPE_EOA:
                self._client = probe_client
                self.signature_type = SIG_TYPE_EOA
                self.funder_address = ""
            else:
                self._client = await self._build_client(
                    creds,
                    signature_type=chosen_sig_type,
                    funder=chosen_funder,
                )
                self.signature_type = chosen_sig_type
                self.funder_address = chosen_funder
            self.signer_address = self._client.get_address() or self.signer_address
            logger.info(
                "Polymarket CLOB client connected "
                f"(dry_run={self.dry_run}, mode={self._sig_label(self.signature_type)}, "
                f"signer={self.signer_address}, funder={self.funder_address or self.signer_address}, "
                f"observed_balance=${balance:.2f})"
            )
            await self._refresh_api_creds()
            return

        self._client = probe_client
        self.signature_type = SIG_TYPE_EOA
        self.funder_address = ""
        logger.info(
            "Polymarket CLOB client connected "
            f"(dry_run={self.dry_run}, mode=EOA, signer={self.signer_address})"
        )
        await self._refresh_api_creds()

    async def _refresh_api_creds(self):
        """
        Re-derive API credentials from the active signer on startup. Polymarket's
        quickstart recommends createOrDeriveApiKey() when signature errors are
        suspected, and this keeps the HMAC creds aligned with the current signer.
        Falls back to the env-provided creds if derivation is unavailable.
        """
        try:
            derived = await asyncio.to_thread(self._client.create_or_derive_api_creds)
            if not derived:
                raise RuntimeError("create_or_derive_api_creds returned no credentials")

            await asyncio.to_thread(self._client.set_api_creds, derived)
            self.api_key = derived.api_key
            self.api_secret = derived.api_secret
            self.api_passphrase = derived.api_passphrase
            logger.info(
                "CLOB API credentials refreshed from signer "
                f"(key={self.api_key[:8]}..., mode={self._sig_label(self.signature_type)})"
            )
        except Exception as e:
            logger.warning(
                "CLOB API credential refresh failed; continuing with configured env creds: "
                f"{e}"
            )

    async def setup_allowances(self):
        """
        Approve the CLOB exchange contracts to spend USDC and CTF tokens on behalf
        of the signer wallet.  Must be called once after funding the signer address.
        Safe to call on every startup — the CLOB is idempotent (no-op if already set).
        """
        try:
            await asyncio.to_thread(
                self._client.update_balance_allowance,
                BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=self.signature_type,
                ),
            )
            logger.info(
                "CLOB collateral balance/allowance refreshed "
                f"(sig_type={self.signature_type}, funder={self.funder_address or self.signer_address})"
            )
        except Exception as e:
            logger.warning(f"setup_allowances (COLLATERAL) failed: {e}")

        if self.signature_type != SIG_TYPE_EOA:
            logger.info(
                "Skipping CONDITIONAL allowance update in proxy-wallet mode; "
                "Polymarket manages token approvals for the funded wallet"
            )
            return

        try:
            await asyncio.to_thread(
                self._client.update_balance_allowance,
                BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    signature_type=self.signature_type,
                ),
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
                logger.error(f"Gamma get_markets failed (offset={offset}): {type(e).__name__}: {e!r}")
                if self._last_markets:
                    logger.warning(
                        f"Gamma get_markets using {len(self._last_markets)} cached market(s) after upstream failure"
                    )
                    return list(self._last_markets)
                raise

            if not page:
                break

            for m in page:
                self._cache_market_tokens(m)

            all_markets.extend(page)
            if len(page) < limit:
                break
            offset += limit

        if all_markets:
            self._last_markets = list(all_markets)
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
            # Polymarket documents getBalanceAllowance as a cached value. Refresh
            # the collateral snapshot first so proxy-wallet balances do not stay at 0.
            try:
                await asyncio.to_thread(
                    self._client.update_balance_allowance,
                    BalanceAllowanceParams(
                        asset_type=AssetType.COLLATERAL,
                        signature_type=self.signature_type,
                    ),
                )
            except Exception as refresh_error:
                logger.warning(f"get_balance refresh failed: {refresh_error}")

            raw = await asyncio.to_thread(
                self._client.get_balance_allowance,
                BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=self.signature_type,
                ),
            )
            # Log raw on first call so we can see the exact shape
            logger.info(
                f"get_balance raw response (sig_type={self.signature_type}): {raw}"
            )

            if not isinstance(raw, dict):
                logger.warning(f"get_balance: unexpected response type {type(raw)}: {raw}")
                return {"balance": 0.0, "availableBalance": 0.0}

            has_balance_key = any(
                raw.get(key) is not None
                for key in ("balance", "availableBalance", "USDC", "usdc", "collateral")
            )
            if not has_balance_key:
                logger.warning(f"get_balance: no balance key found in {list(raw.keys())}")
                return {"balance": 0.0, "availableBalance": 0.0}

            bal = self._extract_balance_usdc(raw)
            logger.info(f"get_balance: ${bal:.2f} USDC")
            self._last_balance = {"balance": bal, "availableBalance": bal}
            return dict(self._last_balance)
        except Exception as e:
            logger.warning(f"get_balance failed: {type(e).__name__}: {e!r}")
            if self._last_balance is not None:
                logger.warning("get_balance using cached balance after upstream failure")
                return {**self._last_balance, "stale": True}
            raise

    async def get_positions(
        self,
        user: str | None = None,
        markets: list[str] | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list:
        user = (user or self.funder_address or self.signer_address or "").strip()
        if not user:
            return []
        cache_key = (user.lower(), tuple(sorted(markets or [])), min(max(limit, 1), 500), max(offset, 0))

        try:
            params = {
                "user": user,
                "sizeThreshold": 0,
                "limit": min(max(limit, 1), 500),
                "offset": max(offset, 0),
                "sortBy": "CURRENT",
                "sortDirection": "DESC",
            }
            if markets:
                params["market"] = ",".join(markets)
            resp = await self._http.get(
                f"{DATA_API}/positions",
                params=params,
            )
            resp.raise_for_status()
            raw = resp.json()
            positions = raw if isinstance(raw, list) else []
            self._last_positions[cache_key] = list(positions)
            logger.info(f"get_positions: {len(positions)} live positions for {user}")
            return positions
        except Exception as e:
            logger.warning(f"get_positions failed: {type(e).__name__}: {e!r}")
            if cache_key in self._last_positions:
                cached = self._last_positions[cache_key]
                logger.warning(
                    f"get_positions using {len(cached)} cached position(s) after upstream failure"
                )
                return list(cached)
            return []

    async def get_closed_positions(
        self,
        limit: int = 50,
        offset: int = 0,
        user: str | None = None,
        markets: list[str] | None = None,
    ) -> list:
        user = (user or self.funder_address or self.signer_address or "").strip()
        if not user:
            return []
        cache_key = (user.lower(), tuple(sorted(markets or [])), min(max(limit, 1), 50), max(offset, 0))

        try:
            params = {
                "user": user,
                "limit": min(max(limit, 1), 50),
                "offset": max(offset, 0),
                "sortBy": "TIMESTAMP",
                "sortDirection": "DESC",
            }
            if markets:
                params["market"] = ",".join(markets)
            resp = await self._http.get(
                f"{DATA_API}/closed-positions",
                params=params,
            )
            resp.raise_for_status()
            raw = resp.json()
            positions = raw if isinstance(raw, list) else []
            self._last_closed_positions[cache_key] = list(positions)
            logger.info(f"get_closed_positions: {len(positions)} closed positions for {user}")
            return positions
        except Exception as e:
            logger.warning(f"get_closed_positions failed: {type(e).__name__}: {e!r}")
            if cache_key in self._last_closed_positions:
                cached = self._last_closed_positions[cache_key]
                logger.warning(
                    f"get_closed_positions using {len(cached)} cached row(s) after upstream failure"
                )
                return list(cached)
            return []

    async def get_trades(
        self,
        *,
        markets: list[str] | None = None,
        user: str | None = None,
        limit: int = 100,
        offset: int = 0,
        taker_only: bool = True,
        side: str | None = None,
        filter_type: str | None = None,
        filter_amount: float | None = None,
    ) -> list:
        params = {
            "limit": min(max(limit, 1), 10_000),
            "offset": max(offset, 0),
            "takerOnly": str(bool(taker_only)).lower(),
        }
        if markets:
            params["market"] = ",".join(markets)
        if user:
            params["user"] = user
        if side:
            params["side"] = side
        if filter_type and filter_amount is not None:
            params["filterType"] = filter_type
            params["filterAmount"] = max(float(filter_amount), 0.0)

        try:
            resp = await self._http.get(f"{DATA_API}/trades", params=params)
            resp.raise_for_status()
            raw = resp.json()
            trades = raw if isinstance(raw, list) else []
            logger.info(
                f"get_trades: {len(trades)} rows "
                f"(markets={len(markets or [])}, user={user or 'any'})"
            )
            return trades
        except Exception as e:
            logger.warning(f"get_trades failed: {type(e).__name__}: {e!r}")
            return []

    async def get_top_holders(
        self,
        markets: list[str],
        *,
        limit: int = 20,
        min_balance: int = 1,
    ) -> list:
        if not markets:
            return []

        try:
            resp = await self._http.get(
                f"{DATA_API}/holders",
                params={
                    "market": ",".join(markets),
                    "limit": min(max(limit, 1), 20),
                    "minBalance": max(int(min_balance), 0),
                },
            )
            resp.raise_for_status()
            raw = resp.json()
            holders = raw if isinstance(raw, list) else []
            logger.info(
                f"get_top_holders: {len(holders)} token groups for {len(markets)} markets"
            )
            return holders
        except Exception as e:
            logger.warning(f"get_top_holders failed: {e}")
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
            raw_dicts = [
                o if isinstance(o, dict) else (vars(o) if hasattr(o, "__dict__") else {})
                for o in orders
            ]

            # Startup sync happens before strategy market refresh. Hydrate the
            # slug/token cache first so restored SELL exits are matched to the
            # same market/outcome as the live position instead of looking like
            # anonymous token orders.
            needs_token_hydration = any(
                asset_id and not self._asset_id_to_slug(asset_id)
                for od in raw_dicts
                for asset_id in [str(
                    od.get("asset_id") or od.get("assetId")
                    or od.get("token_id") or od.get("tokenId") or ""
                )]
            )
            if needs_token_hydration:
                await self.get_markets()

            result = []
            for od in raw_dicts:
                asset_id = str(
                    od.get("asset_id") or od.get("assetId")
                    or od.get("token_id") or od.get("tokenId") or ""
                )
                slug     = self._asset_id_to_slug(asset_id) or asset_id
                oid      = od.get("id") or od.get("orderId") or od.get("order_id") or ""
                if not oid:
                    continue
                quantity = self._remaining_order_size(od)
                execution_side = str(
                    od.get("side") or od.get("orderSide") or od.get("order_side") or ""
                ).upper()
                result.append({
                    "id":             oid,
                    "market_slug":    slug,
                    "asset_id":       asset_id,
                    "intent":         self._derive_intent(od, asset_id),
                    "execution_side": execution_side,
                    "price":          float(od.get("price", 0) or 0),
                    "quantity":       quantity,
                })
            return result
        except Exception as e:
            logger.warning(f"get_open_orders failed: {type(e).__name__}: {e!r}")
            return []

    async def place_order(
        self,
        market_slug: str,
        intent: str,
        price: float,
        quantity: float,
        side: str = BUY,
        order_type: str = "ORDER_TYPE_LIMIT",
        tif: str = "TIME_IN_FORCE_GOOD_TILL_CANCEL",
    ) -> dict:
        if self.dry_run:
            fake_id = f"dry_run_{market_slug}_{intent}_{price:.4f}"
            order_type = self._order_type_from_tif(tif)
            logger.info(
                f"[DRY RUN] {intent} {quantity:.1f}x @ ${price:.4f} "
                f"type={order_type} on {market_slug}"
            )
            return {"id": fake_id, "order_type": order_type}

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
            side     = side,
        )
        order_type = self._order_type_from_tif(tif)
        signed_order = await asyncio.to_thread(self._client.create_order, args)
        try:
            raw = await asyncio.to_thread(
                self._client.post_order,
                signed_order,
                orderType=order_type,
            )
        except Exception as e:
            error_text = str(e)
            if (
                order_type in {OrderType.FAK, OrderType.FOK}
                and "no orders found to match" in error_text.lower()
            ):
                order_id_match = re.search(r"['\"]orderID['\"]:\s*['\"]([^'\"]+)['\"]", error_text)
                oid = order_id_match.group(1) if order_id_match else ""
                logger.info(
                    f"Order no-fill: {intent} {side} {quantity:.1f}x @ ${price:.4f} "
                    f"type={order_type} on {market_slug} — no matching liquidity"
                )
                return {
                    "id": oid,
                    "asset_id": token_id,
                    "order_type": order_type,
                    "status": "no_match",
                    "raw_error": error_text,
                }
            raise
        result = raw if isinstance(raw, dict) else (vars(raw) if hasattr(raw, "__dict__") else {})
        oid    = (
            result.get("orderID") or result.get("order_id") or result.get("id") or ""
        )
        logger.info(
            f"Order placed: {intent} {side} {quantity:.1f}x @ ${price:.4f} "
            f"type={order_type} "
            f"on {market_slug} → id={oid}"
        )
        return {"id": oid, "asset_id": token_id, "order_type": order_type, "raw": result}

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

    @staticmethod
    def _order_type_from_tif(tif: str) -> OrderType:
        value = str(tif or "").upper()
        if value in {"FOK", "TIME_IN_FORCE_FILL_OR_KILL"}:
            return OrderType.FOK
        if value in {
            "FAK",
            "IOC",
            "TIME_IN_FORCE_FILL_AND_KILL",
            "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
        }:
            return OrderType.FAK
        if value in {"GTD", "TIME_IN_FORCE_GOOD_TILL_DATE"}:
            return OrderType.GTD
        return OrderType.GTC

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _remaining_order_size(cls, order: dict) -> float:
        for key in ("remaining_size", "remainingSize", "size_remaining", "sizeRemaining"):
            remaining = cls._safe_float(order.get(key), -1.0)
            if remaining >= 0:
                return remaining

        size = cls._safe_float(order.get("quantity") or order.get("size"), -1.0)
        if size >= 0:
            return size

        original = cls._safe_float(
            order.get("original_size") or order.get("originalSize"),
            0.0,
        )
        matched = cls._safe_float(
            order.get("size_matched") or order.get("sizeMatched"),
            0.0,
        )
        if original > 0:
            return max(0.0, original - matched)
        return 0.0
