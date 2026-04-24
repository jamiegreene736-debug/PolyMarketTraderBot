"""
PolyMarket Trader Dashboard
============================
Runs the web dashboard AND the bot in the same process.
The bot loop runs as a background asyncio task so they share
memory directly — no cross-process database sync issues.
"""

import os
import json
import asyncio
import re
from contextlib import asynccontextmanager
from datetime import datetime

import yaml
import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader
import secrets
from loguru import logger
from py_clob_client.order_builder.constants import SELL

from src.logger import setup_logger
from src.client import PolymarketClient
from src.market_data import MarketData
from src.order_manager import OrderManager
from src.capital_manager import CapitalManager
from src import database as db
from src import fees
from src.ai_observer import AIObserver

from src.strategies.near_certainty import NearCertaintyStrategy
from src.strategies.inverted_near_certainty import InvertedNearCertaintyStrategy
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.logical_arb import LogicalArbStrategy
from src.strategies.cross_platform import CrossPlatformArbStrategy
from src.strategies.news_catalyst import NewsCatalystStrategy
from src.strategies.ai_trader import AITradingStrategy
from src.strategies.position_monitor import PositionMonitorStrategy
from src.strategies.whale_tracker import WhaleTrackerStrategy
from src.news_client import NewsClient
from src.circuit_breaker import CircuitBreaker

load_dotenv()

# ── Shared in-memory bot state ────────────────────────────────────────────────
_bot_state = {
    "status": "stopped",        # stopped | running | error
    "last_heartbeat": None,
    "last_error": None,
    "task": None,               # asyncio.Task reference
}

# Live references set when the bot starts so API routes can read them
_order_manager: "OrderManager | None" = None
_capital: "CapitalManager | None" = None
_circuit_breaker: "CircuitBreaker | None" = None
_client_ref: "PolymarketClient | None" = None
_market_data_ref: "MarketData | None" = None
_ai_observer: "AIObserver | None" = None
_account_cache = {
    "balance": 0.0,
    "balance_ts": 0.0,
    "positions": [],
    "positions_ts": 0.0,
    "closed_positions": [],
    "closed_positions_ts": 0.0,
}


def load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# ── Bot loop ──────────────────────────────────────────────────────────────────

async def run_bot_loop():
    global _order_manager, _capital, _circuit_breaker, _client_ref, _market_data_ref, _ai_observer
    config = load_config()
    bot_cfg = config.get("bot", {})

    api_key         = os.getenv("POLY_API_KEY", "").strip()
    api_secret      = os.getenv("POLY_API_SECRET", "").strip()
    api_passphrase  = os.getenv("POLY_API_PASSPHRASE", "").strip()
    private_key     = os.getenv("POLY_PRIVATE_KEY", "").strip()
    funder_address  = os.getenv("POLY_FUNDER_ADDRESS", "").strip()
    signature_type_raw = os.getenv("POLY_SIGNATURE_TYPE", "").strip()
    signature_type = None
    if signature_type_raw:
        try:
            signature_type = int(signature_type_raw)
        except ValueError:
            logger.warning(f"Invalid POLY_SIGNATURE_TYPE={signature_type_raw!r}; using auto-detect")

    if not api_key or not api_secret or not api_passphrase or not private_key:
        missing = [k for k, v in {
            "POLY_API_KEY":        api_key,
            "POLY_API_SECRET":     api_secret,
            "POLY_API_PASSPHRASE": api_passphrase,
            "POLY_PRIVATE_KEY":    private_key,
        }.items() if not v]
        _bot_state["status"] = "error"
        _bot_state["last_error"] = f"Missing env vars: {', '.join(missing)}"
        logger.error(_bot_state["last_error"])
        return

    client = PolymarketClient(
        api_key        = api_key,
        api_secret     = api_secret,
        api_passphrase = api_passphrase,
        private_key    = private_key,
        funder_address = funder_address,
        signature_type = signature_type,
        dry_run        = bot_cfg.get("dry_run", False),
    )
    await client.connect()
    _client_ref = client

    await db.log_to_db(
        "INFO",
        "CLOB auth: "
        f"signer={client.signer_address or 'unknown'} "
        f"funder={client.funder_address or 'EOA signer wallet'} "
        f"sig_type={client.signature_type}"
    )

    # Proxy-wallet accounts already have their allowances managed by
    # Polymarket. Direct EOAs need approvals on first live startup.
    if not bot_cfg.get("dry_run", False) and client.signature_type == 0:
        await client.setup_allowances()

    market_data = MarketData(client)
    _market_data_ref = market_data
    order_manager = OrderManager(
        client,
        max_concurrent=bot_cfg.get("max_concurrent_orders", 20),
        market_data=market_data,
        min_liquidity_multiple=bot_cfg.get("min_liquidity_multiple", 3.0),
    )

    # Fetch real balance at startup so capital manager is accurate from tick 1
    startup_balance = 0.0
    startup_balance_available = False
    startup_balance_timeout = float(bot_cfg.get("startup_balance_timeout_seconds", 20))
    try:
        msg = f"Startup: fetching exchange balance (timeout={startup_balance_timeout:.0f}s)"
        logger.info(msg)
        await db.log_to_db("INFO", msg)
        balance_data = await asyncio.wait_for(
            client.get_balance(),
            timeout=startup_balance_timeout,
        )
        for key in ("availableBalance", "balance", "usdc", "availableUsdc", "cashBalance"):
            val = balance_data.get(key)
            if val is not None:
                startup_balance = float(val)
                startup_balance_available = True
                break
        msg = f"Startup balance: ${startup_balance:.2f} USDC"
        logger.info(msg)
        await db.log_to_db("INFO", msg)
        if startup_balance > 0:
            _account_cache["balance"] = startup_balance
            _account_cache["balance_ts"] = asyncio.get_event_loop().time()
    except asyncio.TimeoutError:
        msg = f"Startup balance fetch timed out after {startup_balance_timeout:.0f}s; continuing with fallback balance"
        logger.warning(msg)
        await db.log_to_db("WARNING", msg)
    except Exception as e:
        msg = f"Could not fetch startup balance: {e}"
        logger.warning(msg)
        await db.log_to_db("WARNING", msg)

    dry_run = bot_cfg.get("dry_run", False)

    # In dry-run mode, use a realistic test balance if the real balance is $0
    if dry_run and startup_balance < 1.0:
        startup_balance = 36.0
        msg = "Dry-run: using simulated $36 balance for strategy testing"
        logger.info(msg)
        await db.log_to_db("INFO", msg)

    # In live mode, the CLOB's get_balance_allowance may return $0 for
    # email/magic-link accounts whose funds are held in a Gnosis Safe proxy
    # rather than a raw EOA. If a POLY_STARTING_BALANCE override is set and
    # the CLOB returned $0, use the override so strategies can attempt orders.
    # The exchange validates funds at execution time — if funds truly aren't
    # available, individual orders will fail and the bot will log/skip them.
    if not dry_run and startup_balance < 1.0:
        override = os.getenv("POLY_STARTING_BALANCE", "").strip()
        if override:
            try:
                startup_balance = float(override)
                msg = (
                    f"CLOB returned $0 balance — using POLY_STARTING_BALANCE override: "
                    f"${startup_balance:.2f}. Orders will be validated by the exchange."
                )
                logger.warning(msg)
                await db.log_to_db("WARNING", msg)
            except ValueError:
                msg = f"Invalid POLY_STARTING_BALANCE value: {override!r}"
                logger.warning(msg)
                await db.log_to_db("WARNING", msg)
        elif not startup_balance_available:
            msg = (
                "Startup balance unavailable from CLOB; pausing bot instead of "
                "starting with a fake $0 balance. Check upstream connectivity and retry Start."
            )
            _bot_state["status"] = "error"
            _bot_state["last_error"] = msg
            logger.warning(msg)
            await db.log_to_db("WARNING", msg)
            return

    capital = CapitalManager(
        total_usdc=startup_balance,
        strategy_config=config.get("strategies", {}),
        reserve_pct=config.get("capital", {}).get("reserve_pct", 10),
    )
    order_manager.attach_capital_manager(capital)

    news_client = NewsClient(api_key=os.getenv("NEWS_API_KEY", ""))

    # Expose to API routes
    _order_manager = order_manager
    _capital = capital
    startup_equity = await _get_account_equity(client, capital.total_usdc)
    if abs(startup_equity - capital.total_usdc) >= 0.01:
        msg = (
            f"Startup account equity: ${startup_equity:.2f} "
            f"(cash=${capital.total_usdc:.2f})"
        )
        logger.info(msg)
        await db.log_to_db("INFO", msg)
    _circuit_breaker = CircuitBreaker(config, start_balance=startup_equity)
    _ai_observer = AIObserver(config.get("ai_observer", {}))
    cleared_reports = await db.clear_ai_observer_reports()
    msg = f"AI observer session reset ({cleared_reports} stale report(s) cleared)"
    logger.info(msg)
    await db.log_to_db("INFO", msg)

    strategies = [
        # Run position monitor FIRST each tick so exits happen before new entries
        PositionMonitorStrategy("position_monitor", config["strategies"].get("position_monitor", {"enabled": True}),
                                client, market_data, order_manager, capital),
        NearCertaintyStrategy("near_certainty", config["strategies"]["near_certainty"],
                              client, market_data, order_manager, capital),
        InvertedNearCertaintyStrategy("inverted_near_certainty", config["strategies"]["inverted_near_certainty"],
                                      client, market_data, order_manager, capital),
        MarketMakingStrategy("market_making", config["strategies"]["market_making"],
                             client, market_data, order_manager, capital),
        LogicalArbStrategy("logical_arb", config["strategies"]["logical_arb"],
                           client, market_data, order_manager, capital),
        AITradingStrategy("ai_trader", config["strategies"].get("ai_trader", {"enabled": False}),
                          client, market_data, order_manager, capital,
                          news_client=news_client),
        WhaleTrackerStrategy("whale_tracker", config["strategies"].get("whale_tracker", {"enabled": True}),
                             client, market_data, order_manager, capital),
        CrossPlatformArbStrategy("cross_platform_arb", config["strategies"]["cross_platform_arb"],
                                 client, market_data, order_manager, capital),
        NewsCatalystStrategy("news_catalyst", config["strategies"]["news_catalyst"],
                             client, market_data, order_manager, capital,
                             news_client=news_client),
    ]

    enabled = [s for s in strategies if s.enabled]
    position_monitor = next((s for s in enabled if s.name == "position_monitor"), None)
    poll_interval = bot_cfg.get("poll_interval_seconds", 30)
    balance_refresh_counter = 0
    market_data_failures = 0

    # Restore open positions from exchange on every startup, but do not let a
    # slow CLOB call keep the dashboard stuck at "CLOB auth" forever.
    startup_sync_timeout = float(bot_cfg.get("startup_sync_timeout_seconds", 25))
    try:
        msg = f"Startup: syncing open exchange orders (timeout={startup_sync_timeout:.0f}s)"
        logger.info(msg)
        await db.log_to_db("INFO", msg)
        await asyncio.wait_for(
            order_manager.sync_from_exchange(),
            timeout=startup_sync_timeout,
        )
    except asyncio.TimeoutError:
        msg = f"Startup exchange-order sync timed out after {startup_sync_timeout:.0f}s; halting start to avoid trading with incomplete state"
        _bot_state["status"] = "error"
        _bot_state["last_error"] = msg
        logger.warning(msg)
        await db.log_to_db("WARNING", msg)
        return
    except Exception as e:
        msg = f"Startup exchange-order sync failed: {e}"
        _bot_state["status"] = "error"
        _bot_state["last_error"] = msg
        logger.warning(msg)
        await db.log_to_db("WARNING", msg)
        return

    msg = f"Bot started — active strategies: {[s.name for s in enabled]}"
    logger.info(msg)
    await db.log_to_db("INFO", msg)

    try:
        while _bot_state["status"] == "running":
            try:
                recent_closed = []

                # Refresh balance every 10 ticks
                # In dry-run mode, skip real balance updates — CLOB reports $0 because
                # funds are in the CTF Exchange contract, which would immediately trip the
                # circuit breaker against the simulated $36 startup balance.
                if balance_refresh_counter % 10 == 0 and not dry_run:
                    try:
                        balance_data = await asyncio.wait_for(
                            client.get_balance(),
                            timeout=float(bot_cfg.get("balance_refresh_timeout_seconds", 15)),
                        )
                        for key in ("availableBalance", "balance", "usdc", "availableUsdc", "cashBalance"):
                            val = balance_data.get(key)
                            if val is not None:
                                balance = float(val)
                                # Only update if CLOB returns a real value.
                                # $0 means the proxy wallet auth hasn't resolved yet —
                                # keeping the override prevents a false circuit-breaker trip.
                                if balance > 0:
                                    capital.update_balance(balance)
                                    _account_cache["balance"] = balance
                                    _account_cache["balance_ts"] = asyncio.get_event_loop().time()
                                # Snapshot against exchange-truth realized P&L, not local
                                # placeholder close rows from earlier buggy exit paths.
                                try:
                                    realized_pnl = _live_closed_summary(
                                        await _get_live_closed_positions(limit=150)
                                    ).get("total_pnl", 0.0)
                                except Exception:
                                    realized_pnl = 0.0
                                await db.snapshot_balance(balance, realized_pnl, 0.0)
                                msg = f"Balance refreshed: ${balance:.2f} USDC"
                                logger.info(msg)
                                await db.log_to_db("INFO", msg)
                                break
                    except Exception as e:
                        msg = f"Balance refresh failed: {e}"
                        logger.warning(msg)
                        await db.log_to_db("WARNING", msg)

                balance_refresh_counter += 1

                if position_monitor is not None:
                    await _reconcile_live_capital_allocations(
                        capital=capital,
                        client=client,
                        order_manager=order_manager,
                        position_monitor=position_monitor,
                    )

                # ── Circuit breaker check ─────────────────────────────────
                try:
                    recent_closed = sorted(
                        await _get_live_closed_positions(limit=150),
                        key=lambda pos: _polymarket_ts_seconds(pos.get("timestamp")),
                        reverse=True,
                    )
                    session_start_ts = (
                        _circuit_breaker.session_start_at.timestamp()
                        if _circuit_breaker is not None
                        else 0.0
                    )
                    recent_pnls = [
                        _safe_float(pos.get("realizedPnl"))
                        for pos in recent_closed
                        if _polymarket_ts_seconds(pos.get("timestamp")) >= session_start_ts
                    ]
                except Exception:
                    recent_pnls = []
                account_equity = await _get_account_equity(client, capital.total_usdc)
                cb_safe = await _circuit_breaker.check(account_equity, recent_pnls)
                if not cb_safe:
                    _bot_state["status"] = "error"
                    _bot_state["last_error"] = f"Circuit breaker: {_circuit_breaker.trip_reason}"
                    msg = (
                        "Circuit breaker halted new strategy execution; existing exchange "
                        "orders were left untouched for manual review."
                    )
                    logger.warning(msg)
                    await db.log_to_db("WARNING", msg)
                    break

                tick_msg = (
                    f"Tick #{balance_refresh_counter} | "
                    f"balance=${capital.total_usdc:.2f} | "
                    f"equity=${account_equity:.2f} | "
                    f"open_orders={order_manager.get_total_open_orders()} | "
                    f"strategies={[s.name for s in enabled]}"
                )
                logger.info(tick_msg)
                await db.log_to_db("INFO", tick_msg)

                try:
                    market_snapshot = await asyncio.wait_for(
                        market_data.get_markets(),
                        timeout=float(bot_cfg.get("market_refresh_timeout_seconds", 18)),
                    )
                except Exception as e:
                    market_snapshot = []
                    msg = f"Market data preflight failed: {e}"
                    logger.warning(msg)
                    await db.log_to_db("WARNING", msg)

                entries_paused = not bool(market_snapshot)
                if entries_paused:
                    market_data_failures += 1
                    msg = (
                        "Market data unavailable; skipping new-entry strategies this tick "
                        f"(failure {market_data_failures}/3)."
                    )
                    _bot_state["last_error"] = msg
                    logger.warning(msg)
                    await db.log_to_db("WARNING", msg)
                    if market_data_failures >= 3:
                        msg = (
                            "Polymarket market/data APIs unavailable for 3 consecutive ticks; "
                            "pausing bot to avoid trading against empty/stale market data."
                        )
                        _bot_state["status"] = "error"
                        _bot_state["last_error"] = msg
                        logger.warning(msg)
                        await db.log_to_db("WARNING", msg)
                        break
                else:
                    market_data_failures = 0

                for s in enabled:
                    try:
                        if entries_paused and s.name != "position_monitor":
                            continue
                        await s.run()
                    except Exception as e:
                        msg = f"[{s.name}] crashed: {e}"
                        logger.error(msg)
                        await db.log_to_db("ERROR", msg)

                if _ai_observer is not None:
                    session_start_ts = (
                        _circuit_breaker.session_start_at.timestamp()
                        if _circuit_breaker is not None
                        else None
                    )
                    _ai_observer.maybe_schedule(
                        balance=capital.total_usdc,
                        open_orders=order_manager.get_total_open_orders(),
                        bot_status=_bot_state.get("status", "unknown"),
                        last_error=_bot_state.get("last_error"),
                        closed_positions=recent_closed,
                        session_start_ts=session_start_ts,
                    )

                _bot_state["last_heartbeat"] = datetime.utcnow().isoformat()
                if not entries_paused:
                    _bot_state["last_error"] = None

            except Exception as e:
                _bot_state["last_error"] = str(e)
                msg = f"Bot loop error: {e}"
                logger.error(msg)
                await db.log_to_db("ERROR", msg)

            await asyncio.sleep(poll_interval)

    except asyncio.CancelledError:
        pass
    finally:
        # Do not cancel exchange orders on shutdown/error. Pending exits are
        # safety orders and must be left resting unless the user explicitly
        # asks to cancel them.
        logger.info("Bot loop stopped; exchange orders left untouched.")
    # Do NOT set status here — supervisor owns the status after a crash.
    # Only /api/stop sets status="stopped" (user-requested).


# ── FastAPI app ───────────────────────────────────────────────────────────────

async def _auto_restart_bot():
    """
    Supervisor loop: optionally starts the bot on server boot (if auto_start: true
    in config), then automatically restarts it if it crashes — with a 10-second
    cooldown between attempts so we don't spin-loop on a bad config.
    """
    # Respect auto_start config — default False so the user clicks Start themselves.
    cfg = load_config()
    if not cfg.get("bot", {}).get("auto_start", False):
        logger.info("auto_start=false — waiting for manual Start from dashboard")
        _bot_state["status"] = "stopped"
        while _bot_state.get("status") == "stopped":
            await asyncio.sleep(5)

    while True:
        logger.info("Auto-starting bot loop...")
        _bot_state["status"] = "running"
        _bot_state["last_error"] = None
        task = asyncio.create_task(run_bot_loop())
        _bot_state["task"] = task
        try:
            await task
        except asyncio.CancelledError:
            logger.info("Bot task cancelled (shutdown)")
            break
        except Exception as e:
            logger.error(f"Bot loop crashed unexpectedly: {e}")
            _bot_state["last_error"] = str(e)
        # If we get here the bot stopped (crash or clean stop via /api/stop)
        if _bot_state.get("status") == "stopped":
            # User explicitly stopped it — don't restart, just wait
            logger.info("Bot stopped by user — supervisor idle, waiting for manual start")
            while _bot_state.get("status") == "stopped":
                await asyncio.sleep(5)
        elif _bot_state.get("status") == "error":
            logger.warning("Bot entered error state — supervisor idle, waiting for manual Start")
            while _bot_state.get("status") == "error":
                await asyncio.sleep(5)
        else:
            logger.warning("Bot crashed — restarting in 10s...")
            _bot_state["status"] = "error"
            await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    await db.init_db()
    supervisor = asyncio.create_task(_auto_restart_bot())
    _bot_state["supervisor"] = supervisor
    yield
    # Shutdown: cancel supervisor and bot task cleanly
    supervisor.cancel()
    task = _bot_state.get("task")
    if task and not task.done():
        task.cancel()


app = FastAPI(title="PolyMarket Trader", docs_url=None, redoc_url=None, lifespan=lifespan)

jinja_env = Environment(loader=FileSystemLoader("templates"))
jinja_env.filters["tojson"] = json.dumps

security = HTTPBasic()
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "changeme")


def verify_password(credentials: HTTPBasicCredentials = Depends(security)):
    correct = secrets.compare_digest(credentials.password.encode(), DASHBOARD_PASSWORD.encode())
    if not correct:
        raise HTTPException(status_code=401, detail="Incorrect password",
                            headers={"WWW-Authenticate": "Basic"})
    return credentials


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_market_key(value: str | None) -> str:
    """
    Normalize market identifiers across slugs and human-readable titles.
    This lets us match:
      - slug vs question
      - punctuation differences
      - unicode dash vs ASCII dash
    """
    raw = _normalize_text(value)
    if not raw:
        return ""
    raw = raw.replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", "", raw)


def _iso_from_polymarket_timestamp(value) -> str:
    if value in (None, ""):
        return ""

    try:
        raw = float(value)
    except (TypeError, ValueError):
        return str(value)

    # Data API timestamps can be in seconds or milliseconds.
    if raw > 1_000_000_000_000:
        raw /= 1000.0

    try:
        return datetime.utcfromtimestamp(raw).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _trade_sort_key(row: dict) -> str:
    return str(row.get("resolved_at") or row.get("timestamp") or "")


def _format_age_label(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if minutes == 0:
        return f"{hours}h"
    return f"{hours}h {minutes}m"


async def _get_account_equity(client: PolymarketClient, cash_balance: float) -> float:
    """
    Include live position value for drawdown checks. Cash can drop when an
    order fills or collateral moves into a position, which is not a loss.
    """
    user_address = (client.funder_address or client.signer_address or "").strip()
    positions: list[dict] = []
    if user_address:
        try:
            positions = await asyncio.wait_for(
                client.get_positions(user=user_address),
                timeout=5.0,
            )
            if positions:
                _account_cache["positions"] = positions
                _account_cache["positions_ts"] = asyncio.get_event_loop().time()
        except Exception as e:
            logger.warning(f"Account equity position refresh failed, using cached positions: {e}")
            positions = list(_account_cache.get("positions") or [])

    position_value = 0.0
    for pos in positions:
        try:
            position_value += float(pos.get("currentValue") or 0.0)
        except (TypeError, ValueError):
            continue
    return max(0.0, float(cash_balance or 0.0) + position_value)


def _position_monitor_hold_hours() -> dict:
    cfg = load_config()
    pm_cfg = cfg.get("strategies", {}).get("position_monitor", {})
    return pm_cfg.get("max_hold_hours", {}) if isinstance(pm_cfg, dict) else {}


def _max_hold_seconds(strategy: str) -> float | None:
    hold_cfg = _position_monitor_hold_hours()
    if not isinstance(hold_cfg, dict):
        return None

    hours = hold_cfg.get(strategy)
    if hours is None:
        hours = hold_cfg.get("default")
    if hours is None:
        return None

    try:
        return float(hours) * 3600
    except (TypeError, ValueError):
        return None


def _infer_strategy_from_live_position(outcome: str, avg_price: float) -> str:
    cfg = load_config().get("strategies", {})
    near_min = float(cfg.get("near_certainty", {}).get("min_price", 0.93))
    inv_yes_max = float(cfg.get("inverted_near_certainty", {}).get("max_yes_price", 0.07))
    inv_no_min = max(0.0, 1.0 - inv_yes_max)

    if outcome == "YES" and avg_price >= near_min:
        return "near_certainty"
    if outcome == "NO" and avg_price >= inv_no_min:
        return "inverted_near_certainty"
    return "live position"


async def _get_cached_balance(max_age_seconds: float = 30.0) -> float:
    now = asyncio.get_event_loop().time()
    cached = _safe_float(_account_cache.get("balance"))
    cached_ts = _safe_float(_account_cache.get("balance_ts"))
    if cached > 0 and now - cached_ts <= max_age_seconds:
        return cached

    if _client_ref is not None:
        try:
            cash_data = await asyncio.wait_for(_client_ref.get_balance(), timeout=3.0)
            cash = _safe_float(
                cash_data.get("availableBalance")
                or cash_data.get("balance")
                or cash_data.get("usdc")
                or cash_data.get("availableUsdc")
                or cash_data.get("cashBalance")
            )
            if cash > 0:
                _account_cache["balance"] = cash
                _account_cache["balance_ts"] = now
                return cash
        except Exception as e:
            logger.warning(f"Dashboard balance refresh failed, using cached balance: {e}")

    if cached > 0:
        return cached
    if _capital is not None and _capital.total_usdc > 0:
        return float(_capital.total_usdc)
    return 0.0


async def _get_cached_live_positions(user_address: str, max_age_seconds: float = 30.0) -> list[dict]:
    now = asyncio.get_event_loop().time()
    cached = list(_account_cache.get("positions") or [])
    cached_ts = _safe_float(_account_cache.get("positions_ts"))
    if cached and now - cached_ts <= max_age_seconds:
        return cached

    if _client_ref is not None:
        try:
            positions = await asyncio.wait_for(
                _client_ref.get_positions(user=user_address),
                timeout=3.0,
            )
            if positions:
                _account_cache["positions"] = positions
                _account_cache["positions_ts"] = now
                return positions
        except Exception as e:
            logger.warning(f"Dashboard position refresh failed, using cached positions: {e}")

    if cached:
        return cached
    return []


def _open_order_position_fallback() -> list[dict]:
    if _order_manager is None:
        return []

    rows = []
    now = datetime.utcnow().timestamp()
    for order in _order_manager.get_open_positions():
        price = _safe_float(order.get("price"))
        quantity = _safe_float(order.get("quantity"))
        placed_at = _safe_float(order.get("placed_at"))
        strategy = order.get("strategy") or "open_order"
        intent = str(order.get("intent") or "")
        rows.append({
            "order_id": order.get("order_id") or "",
            "market_slug": order.get("market_slug") or "—",
            "title": order.get("question") or order.get("market_slug") or "—",
            "condition_id": "",
            "intent": intent,
            "price": price,
            "quantity": quantity,
            "current_value": 0.0,
            "cost_basis": round(price * quantity, 2),
            "outcome": "YES" if "BUY_LONG" in intent else "NO" if "BUY_SHORT" in intent else "—",
            "strategy": strategy,
            "closable": True,
            "estimated_pnl": 0.0,
            "override_active": False,
            "age_label": _format_age_label(max(0.0, now - placed_at) if placed_at else None),
            "max_hold_label": "—",
            "placed_at_ts": placed_at or 0,
            "max_hold_seconds": 0,
            "force_exit_in_label": "resting order",
            "force_exit_at": "",
            "force_exit_at_ts": 0,
        })
    return rows


async def _get_close_quote(
    market_slug: str,
    outcome: str,
    quantity: float,
    entry_price: float,
) -> dict:
    """
    Quote what Close Now can actually sell into right now.

    Uses the live token order book, not Gamma marks. If there is no bid, the
    displayed P&L should say so instead of showing a theoretical mark.
    """
    if _client_ref is None:
        return {}

    book = await _client_ref.get_outcome_order_book(market_slug, outcome)
    raw_bids = book.get("bids") or []
    bids = sorted(
        (
            {
                "price": _safe_float(level.get("price")),
                "size": _safe_float(level.get("size")),
            }
            for level in raw_bids
        ),
        key=lambda level: level["price"],
        reverse=True,
    )

    remaining = max(0.0, float(quantity or 0.0))
    fillable_qty = 0.0
    gross_proceeds = 0.0
    exit_fee = 0.0
    fee_rate_bps = _safe_float(book.get("fee_rate_bps"))
    best_bid = bids[0]["price"] if bids else 0.0
    best_bid_size = bids[0]["size"] if bids else 0.0

    for level in bids:
        price = level["price"]
        size = level["size"]
        if remaining <= 0:
            break
        if price < 0.01 or size <= 0:
            continue
        fill_qty = min(remaining, size)
        gross_proceeds += fill_qty * price
        exit_fee += fees.taker_fee_for_rate(fill_qty, price, fee_rate_bps)
        fillable_qty += fill_qty
        remaining -= fill_qty

    net_proceeds = gross_proceeds - exit_fee
    cost_basis = fillable_qty * float(entry_price or 0.0)
    net_pnl = net_proceeds - cost_basis
    avg_exit_price = gross_proceeds / fillable_qty if fillable_qty > 0 else 0.0
    full_cost_basis = float(quantity or 0.0) * float(entry_price or 0.0)
    resting_price = 0.01
    resting_gross = float(quantity or 0.0) * resting_price
    # A resting 1c sell is not executable immediately. We show an "if filled"
    # estimate separately so users can decide whether it is worth posting.
    resting_rebate = fees.maker_rebate_for_rate(
        float(quantity or 0.0),
        resting_price,
        fee_rate_bps,
    )

    return {
        "best_bid": best_bid,
        "best_bid_size": best_bid_size,
        "fee_rate_bps": fee_rate_bps,
        "fillable_qty": fillable_qty,
        "unfilled_qty": max(0.0, float(quantity or 0.0) - fillable_qty),
        "avg_exit_price": avg_exit_price,
        "gross_proceeds": gross_proceeds,
        "exit_fee": exit_fee,
        "net_proceeds": net_proceeds,
        "net_pnl": net_pnl,
        "has_liquidity": fillable_qty > 0,
        "fully_fillable": fillable_qty >= float(quantity or 0.0) - 1e-9,
        "resting_exit": {
            "price": resting_price,
            "quantity": float(quantity or 0.0),
            "gross_proceeds_if_filled": resting_gross,
            "maker_rebate_estimate": resting_rebate,
            "net_proceeds_if_filled": resting_gross,
            "net_pnl_if_filled": resting_gross - full_cost_basis,
            "net_pnl_after_rebate_estimate": resting_gross + resting_rebate - full_cost_basis,
        },
    }


async def _get_live_closed_positions(limit: int = 150) -> list[dict]:
    if _client_ref is None:
        return []

    now = asyncio.get_event_loop().time()
    cached = list(_account_cache.get("closed_positions") or [])
    cached_ts = _safe_float(_account_cache.get("closed_positions_ts"))
    if cached and now - cached_ts <= 45:
        return cached[:limit]

    positions: list[dict] = []
    offset = 0

    while len(positions) < limit:
        try:
            batch = await asyncio.wait_for(
                _client_ref.get_closed_positions(
                    limit=min(50, limit - len(positions)),
                    offset=offset,
                ),
                timeout=3.0,
            )
        except Exception as e:
            logger.warning(f"Dashboard closed-position refresh failed, using cached rows: {e}")
            return cached[:limit]
        if not batch:
            break
        positions.extend(batch)
        if len(batch) < 50:
            break
        offset += len(batch)

    if positions:
        _account_cache["closed_positions"] = positions
        _account_cache["closed_positions_ts"] = now
        return positions
    if cached:
        return cached[:limit]
    return positions


def _live_closed_summary(closed_positions: list[dict]) -> dict:
    realized_pnl = round(sum(_safe_float(p.get("realizedPnl")) for p in closed_positions), 2)
    wins = sum(1 for p in closed_positions if _safe_float(p.get("realizedPnl")) > 0)
    total_closed = len(closed_positions)
    win_rate = round((wins / total_closed) * 100, 1) if total_closed else 0.0
    today = datetime.utcnow().date().isoformat()
    trades_today = sum(
        1
        for p in closed_positions
        if (_iso_from_polymarket_timestamp(p.get("timestamp")) or "").startswith(today)
    )
    return {
        "total_pnl": realized_pnl,
        "win_rate": win_rate,
        "wins": wins,
        "total_closed": total_closed,
        "total_trades": total_closed,
        "trades_today": trades_today,
    }


def _polymarket_ts_seconds(value) -> float:
    ts = _safe_float(value)
    if ts > 1_000_000_000_000:
        ts /= 1000.0
    return ts


async def _reconcile_live_capital_allocations(
    capital: CapitalManager,
    client: PolymarketClient,
    order_manager: OrderManager,
    position_monitor: PositionMonitorStrategy,
):
    allocations: dict[str, float] = {}

    try:
        managed_positions = await position_monitor._build_managed_positions()
    except Exception as e:
        logger.warning(f"Capital reconciliation: could not load live positions: {e}")
        managed_positions = []

    for pos in managed_positions:
        strategy = str(pos.get("strategy") or "").strip()
        if not strategy or strategy == "position_monitor":
            continue
        try:
            notional = float(pos.get("price") or 0.0) * float(pos.get("quantity") or 0.0)
        except (TypeError, ValueError):
            continue
        if notional > 0:
            allocations[strategy] = allocations.get(strategy, 0.0) + notional

    try:
        db_meta = await db.get_open_trades_metadata()
    except Exception:
        db_meta = {}

    try:
        raw_orders = await client.get_open_orders()
    except Exception as e:
        logger.warning(f"Capital reconciliation: could not load open orders: {e}")
        raw_orders = []

    for o in raw_orders or []:
        order_id = str(o.get("id") or o.get("orderId") or o.get("order_id") or "").strip()
        meta = db_meta.get(order_id, {})
        strategy = str(meta.get("strategy") or "").strip()
        if not strategy or strategy == "position_monitor":
            continue

        execution_side = str(
            o.get("execution_side") or meta.get("execution_side") or ""
        ).upper().strip()
        if execution_side == "SELL":
            continue

        raw_price = o.get("price", 0)
        if isinstance(raw_price, dict):
            raw_price = raw_price.get("value", 0)
        try:
            price = float(raw_price or 0.0)
            quantity = float(o.get("quantity") or o.get("size") or 0.0)
        except (TypeError, ValueError):
            continue

        notional = price * quantity
        if notional > 0:
            allocations[strategy] = allocations.get(strategy, 0.0) + notional

    capital.reconcile(allocations)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(_=Depends(verify_password)):
    stats = await db.get_dashboard_stats()
    if _client_ref is not None:
        try:
            user_address = (_client_ref.funder_address or _client_ref.signer_address or "").strip()
            live_positions = await _get_cached_live_positions(user_address)
            stats["open_positions"] = len(live_positions) or (
                _order_manager.get_total_open_orders() if _order_manager is not None else 0
            )
        except Exception:
            pass
        try:
            stats.update(_live_closed_summary(await _get_live_closed_positions(limit=150)))
        except Exception:
            pass
    template = jinja_env.get_template("index.html")
    html = template.render(**stats)
    return HTMLResponse(content=html)


@app.get("/api/stats")
async def api_stats(_=Depends(verify_password)):
    stats = await db.get_dashboard_stats()
    if _client_ref is not None:
        try:
            user_address = (_client_ref.funder_address or _client_ref.signer_address or "").strip()
            live_positions = await _get_cached_live_positions(user_address)
            stats["open_positions"] = len(live_positions) or (
                _order_manager.get_total_open_orders() if _order_manager is not None else 0
            )
        except Exception:
            pass
        try:
            stats.update(_live_closed_summary(await _get_live_closed_positions(limit=150)))
        except Exception:
            pass
    return stats


@app.get("/api/diagnose")
async def api_diagnose(_=Depends(verify_password)):
    """
    Live diagnostic: queries the CLOB to find the correct proxy wallet address
    and which signature_type has a non-zero balance.
    Hit this in your browser to debug auth/balance issues.
    """
    import os as _os
    from src.client import PolymarketClient, CLOB_HOST, CHAIN_ID
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType

    api_key        = _os.getenv("POLY_API_KEY", "").strip()
    api_secret     = _os.getenv("POLY_API_SECRET", "").strip()
    api_passphrase = _os.getenv("POLY_API_PASSPHRASE", "").strip()
    private_key    = _os.getenv("POLY_PRIVATE_KEY", "").strip()

    results = {}

    try:
        creds = ApiCreds(
            api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase
        )
        clob = await asyncio.to_thread(
            ClobClient, CLOB_HOST, chain_id=CHAIN_ID, key=private_key, creds=creds
        )
        results["signer_address"] = clob.get_address()

        # API keys — shows which wallet/address the key is associated with
        try:
            results["api_keys"] = await asyncio.to_thread(clob.get_api_keys)
        except Exception as e:
            results["api_keys_error"] = str(e)

        # Balance for each sig type
        for stype, label in [(0, "EOA"), (1, "POLY_PROXY"), (2, "POLY_GNOSIS_SAFE")]:
            try:
                bal = await asyncio.to_thread(
                    clob.get_balance_allowance,
                    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=stype),
                )
                results[f"balance_sig{stype}_{label}"] = bal
            except Exception as e:
                results[f"balance_sig{stype}_{label}_error"] = str(e)

        # Also try update_balance_allowance with sig_type=1 (may reveal proxy address in response)
        try:
            upd = await asyncio.to_thread(
                clob.update_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=1),
            )
            results["update_allowance_sig1_response"] = upd
        except Exception as e:
            results["update_allowance_sig1_error"] = str(e)

    except Exception as e:
        results["clob_connect_error"] = str(e)

    return results


@app.get("/api/status")
async def api_status(_=Depends(verify_password)):
    state = dict(_bot_state)
    state.pop("task", None)
    state.pop("supervisor", None)

    # Auto-detect stale heartbeat
    hb = state.get("last_heartbeat")
    if state["status"] == "running" and hb:
        age = (datetime.utcnow() - datetime.fromisoformat(hb)).total_seconds()
        if age > 120:
            state["status"] = "error"
            state["last_error"] = f"No heartbeat for {int(age)}s"

    return state


@app.post("/api/start")
async def api_start(_=Depends(verify_password)):
    if _bot_state.get("status") == "running":
        task = _bot_state.get("task")
        if task and not task.done():
            return {"status": "running"}
    # Signal supervisor to resume (it polls for status != "stopped")
    _bot_state["status"] = "running"
    _bot_state["last_error"] = None
    logger.info("Bot start requested via dashboard")
    return {"status": "running"}


@app.post("/api/stop")
async def api_stop(_=Depends(verify_password)):
    # Set stopped BEFORE cancelling so supervisor doesn't auto-restart
    _bot_state["status"] = "stopped"
    task = _bot_state.get("task")
    if task and not task.done():
        task.cancel()
    logger.info("Bot stopped via dashboard")
    return {"status": "stopped"}


@app.post("/api/clear-logs")
async def api_clear_logs(_=Depends(verify_password)):
    async with __import__('aiosqlite').connect('bot_data.db') as conn:
        await conn.execute("DELETE FROM bot_logs")
        await conn.commit()
    return {"ok": True}


@app.post("/api/ai-alerts/acknowledge")
async def api_acknowledge_ai_alerts(_=Depends(verify_password)):
    count = await db.acknowledge_ai_observer_alerts()
    msg = f"AI observer alerts acknowledged ({count})"
    logger.info(msg)
    await db.log_to_db("INFO", msg)
    return {"ok": True, "count": count}


@app.post("/api/reset-data")
async def api_reset_data(_=Depends(verify_password)):
    """Wipe all trades, balance snapshots, logs, and in-memory positions."""
    task = _bot_state.get("task")
    if task and not task.done():
        raise HTTPException(status_code=400, detail="Stop the bot before resetting data")
    # Clear database
    async with __import__('aiosqlite').connect('bot_data.db') as conn:
        await conn.execute("DELETE FROM trades")
        await conn.execute("DELETE FROM balance_snapshots")
        await conn.execute("DELETE FROM bot_logs")
        await conn.execute("DELETE FROM ai_observer_reports")
        await conn.commit()
    # Clear in-memory state so the dashboard shows clean zeros immediately
    if _order_manager is not None:
        _order_manager.clear()
    if _capital is not None:
        _capital._allocated.clear()
    _account_cache.update({
        "balance": 0.0,
        "balance_ts": 0.0,
        "positions": [],
        "positions_ts": 0.0,
        "closed_positions": [],
        "closed_positions_ts": 0.0,
    })
    logger.info("All data reset via dashboard (DB + in-memory)")
    return {"ok": True}


@app.post("/api/auto-close-override")
async def api_auto_close_override(payload: dict, _=Depends(verify_password)):
    condition_id = str(payload.get("condition_id") or "").strip()
    outcome = str(payload.get("outcome") or "").upper().strip()
    active = bool(payload.get("active"))

    if not condition_id or outcome not in {"YES", "NO"}:
        raise HTTPException(status_code=400, detail="condition_id and outcome are required")

    await db.set_auto_close_override(condition_id, outcome, active)
    msg = (
        f"Auto-close {'paused' if active else 'resumed'} for "
        f"{condition_id[:12]}... {outcome}"
    )
    logger.info(msg)
    await db.log_to_db("INFO", msg)
    return {"ok": True, "condition_id": condition_id, "outcome": outcome, "active": active}


@app.get("/api/positions")
async def api_positions(_=Depends(verify_password)):
    """Return live Polymarket positions with balance breakdown."""
    if _client_ref is None:
        cash = float(_capital.total_usdc) if _capital is not None else 0.0
        positions = _open_order_position_fallback()
        position_value = sum(p["current_value"] for p in positions)
        return {
            "positions": positions,
            "cash_balance": round(cash, 2),
            "position_value": round(position_value, 2),
            "total": round(cash + position_value, 2),
            "source": "local",
        }

    user_address = (_client_ref.funder_address or _client_ref.signer_address or "").strip()
    cash = await _get_cached_balance()
    raw_positions = await _get_cached_live_positions(user_address)
    local_position_refs = await db.get_open_trade_rows()
    if _order_manager is not None:
        # In-memory state can carry a fresher strategy label, but the DB rows are
        # the durable source of placed_at/timestamp after fills and restarts.
        local_position_refs.extend(_order_manager.get_open_positions())

    if not raw_positions and _order_manager is not None and _order_manager.get_total_open_orders() > 0:
        positions = _open_order_position_fallback()
        position_value = sum(p["current_value"] for p in positions)
        return {
            "positions": positions,
            "cash_balance": round(cash, 2),
            "position_value": round(position_value, 2),
            "total": round(cash + position_value, 2),
            "source": "local_open_orders",
        }

    local_by_market_side: dict[tuple[str, str], list[dict]] = {}
    pending_exit_by_market_side: dict[tuple[str, str], dict] = {}
    if _order_manager is not None:
        for order in _order_manager.get_open_positions():
            if str(order.get("execution_side") or "").upper() != "SELL":
                continue
            side_key = str(order.get("intent") or order.get("side") or "")
            if not side_key:
                continue
            for market_ref in (order.get("market_slug"), order.get("question")):
                market_key = _normalize_market_key(market_ref)
                if market_key:
                    pending_exit_by_market_side.setdefault((market_key, side_key), order)

    for ref in local_position_refs:
        if str(ref.get("execution_side") or "").upper() == "SELL":
            continue
        if str(ref.get("strategy") or "") == "position_monitor":
            continue
        side_key = str(ref.get("intent") or ref.get("side") or "")
        if not side_key:
            continue
        for market_ref in (ref.get("market_slug"), ref.get("question")):
            market_key = _normalize_market_key(market_ref)
            if market_key:
                local_by_market_side.setdefault((market_key, side_key), []).append(ref)

    condition_ids = [
        str(p.get("conditionId") or "").strip()
        for p in raw_positions
        if p.get("conditionId")
    ]
    latest_buy_by_market_outcome: dict[tuple[str, str], float] = {}
    if user_address and condition_ids:
        trade_rows = await _client_ref.get_trades(
            user=user_address,
            markets=condition_ids,
            limit=min(max(len(condition_ids) * 20, 50), 1000),
            taker_only=False,
            side="BUY",
        )
        for trade in trade_rows:
            market_id = str(trade.get("conditionId") or "").strip()
            outcome_key = str(trade.get("outcome") or "").upper()
            if not market_id or not outcome_key:
                continue
            try:
                ts = float(trade.get("timestamp") or 0.0)
            except (TypeError, ValueError):
                continue
            if ts > 1_000_000_000_000:
                ts /= 1000.0
            key = (market_id, outcome_key)
            latest_buy_by_market_outcome[key] = max(latest_buy_by_market_outcome.get(key, 0.0), ts)

    override_keys = await db.get_auto_close_overrides()
    bbo_map: dict[str, dict] = {}
    if _market_data_ref is not None:
        slugs = [str(p.get("slug") or "").strip() for p in raw_positions]
        bbos = await asyncio.gather(
            *[_market_data_ref.get_bbo(slug) if slug else asyncio.sleep(0, result={}) for slug in slugs]
        )
        bbo_map = {slug: bbo or {} for slug, bbo in zip(slugs, bbos)}
    close_quote_map: dict[tuple[str, str], dict] = {}
    quote_tasks = []
    quote_keys = []
    for p in raw_positions:
        slug = str(p.get("slug") or "").strip()
        outcome = str(p.get("outcome") or "").upper()
        if not slug or outcome not in {"YES", "NO"}:
            continue
        quote_keys.append((slug, outcome))
        quote_tasks.append(
            _get_close_quote(
                slug,
                outcome,
                _safe_float(p.get("size")),
                _safe_float(p.get("avgPrice")),
            )
        )
    if quote_tasks:
        quote_results = await asyncio.gather(*quote_tasks)
        close_quote_map = {key: quote or {} for key, quote in zip(quote_keys, quote_results)}

    now = datetime.utcnow().timestamp()
    positions = []
    for p in raw_positions:
        condition_id = str(p.get("conditionId") or "").strip()
        outcome = str(p.get("outcome") or "").upper()
        intent = "ORDER_INTENT_BUY_LONG" if outcome == "YES" else "ORDER_INTENT_BUY_SHORT"
        market_slug = p.get("slug") or p.get("title") or "—"
        local_match = None
        match_key = _normalize_market_key(p.get("slug"))
        if match_key:
            candidates = local_by_market_side.get((match_key, intent), [])
            if candidates:
                local_match = candidates.pop(0)
        if local_match is None:
            title_key = _normalize_market_key(p.get("title"))
            if title_key:
                candidates = local_by_market_side.get((title_key, intent), [])
                if candidates:
                    local_match = candidates.pop(0)
        pending_exit = None
        for market_ref in (p.get("slug"), p.get("title")):
            market_key = _normalize_market_key(market_ref)
            if market_key:
                pending_exit = pending_exit_by_market_side.get((market_key, intent))
                if pending_exit:
                    break
        avg_price = float(p.get("avgPrice") or 0.0)
        size = float(p.get("size") or 0.0)
        current_value = float(p.get("currentValue") or (avg_price * size) or 0.0)
        strategy = (local_match or {}).get("strategy") or _infer_strategy_from_live_position(outcome, avg_price)
        bbo = bbo_map.get(str(p.get("slug") or "").strip(), {})
        try:
            current_bid = float((bbo.get("bid") or {}).get("price", 0.0))
        except (TypeError, ValueError):
            current_bid = 0.0
        try:
            current_ask = float((bbo.get("ask") or {}).get("price", 1.0))
        except (TypeError, ValueError):
            current_ask = 1.0
        placed_at = (local_match or {}).get("placed_at")
        if placed_at is None:
            raw_timestamp = (local_match or {}).get("timestamp")
            if raw_timestamp:
                try:
                    placed_at = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00")).timestamp()
                except Exception:
                    placed_at = None
        if placed_at is None and condition_id and outcome:
            placed_at = latest_buy_by_market_outcome.get((condition_id, outcome))
        age_seconds = None
        if placed_at is not None:
            try:
                placed_at = float(placed_at)
                age_seconds = max(0.0, now - placed_at)
            except (TypeError, ValueError):
                age_seconds = None
                placed_at = None
        max_hold_seconds = _max_hold_seconds(strategy)
        force_exit_at = None
        force_exit_in = None
        if age_seconds is not None and max_hold_seconds is not None:
            force_exit_at_ts = float(placed_at) + max_hold_seconds
            force_exit_in = force_exit_at_ts - now
            force_exit_at = datetime.utcfromtimestamp(force_exit_at_ts).isoformat()
        close_quote = close_quote_map.get((str(p.get("slug") or "").strip(), outcome), {})
        estimated_pnl = close_quote.get("net_pnl") if close_quote.get("has_liquidity") else None
        override_active = (condition_id, outcome) in override_keys
        pending_exit_order_id = (pending_exit or {}).get("order_id") or ""
        pending_exit_price = _safe_float((pending_exit or {}).get("price"))
        pending_exit_quantity = _safe_float((pending_exit or {}).get("quantity"))
        pending_exit_placed_at = _safe_float((pending_exit or {}).get("placed_at"))
        pending_exit_age = max(0.0, now - pending_exit_placed_at) if pending_exit_placed_at else None
        positions.append({
            "order_id": "",
            "market_slug": market_slug,
            "title": p.get("title") or p.get("slug") or "—",
            "condition_id": condition_id,
            "intent": intent,
            "price": avg_price,
            "quantity": size,
            "current_value": current_value,
            "cost_basis": round(avg_price * size, 2),
            "outcome": p.get("outcome") or "—",
            "strategy": strategy,
            "closable": False,
            "estimated_pnl": round(float(estimated_pnl), 2) if estimated_pnl is not None else None,
            "close_quote": {
                "has_liquidity": bool(close_quote.get("has_liquidity")),
                "fully_fillable": bool(close_quote.get("fully_fillable")),
                "best_bid": round(_safe_float(close_quote.get("best_bid")), 4),
                "best_bid_size": round(_safe_float(close_quote.get("best_bid_size")), 2),
                "fillable_qty": round(_safe_float(close_quote.get("fillable_qty")), 2),
                "unfilled_qty": round(_safe_float(close_quote.get("unfilled_qty")), 2),
                "avg_exit_price": round(_safe_float(close_quote.get("avg_exit_price")), 4),
                "gross_proceeds": round(_safe_float(close_quote.get("gross_proceeds")), 4),
                "exit_fee": round(_safe_float(close_quote.get("exit_fee")), 5),
                "net_proceeds": round(_safe_float(close_quote.get("net_proceeds")), 4),
                "net_pnl": round(_safe_float(close_quote.get("net_pnl")), 4),
                "fee_rate_bps": round(_safe_float(close_quote.get("fee_rate_bps")), 4),
                "resting_exit": {
                    "price": round(_safe_float((close_quote.get("resting_exit") or {}).get("price")), 4),
                    "quantity": round(_safe_float((close_quote.get("resting_exit") or {}).get("quantity")), 2),
                    "gross_proceeds_if_filled": round(_safe_float((close_quote.get("resting_exit") or {}).get("gross_proceeds_if_filled")), 4),
                    "maker_rebate_estimate": round(_safe_float((close_quote.get("resting_exit") or {}).get("maker_rebate_estimate")), 5),
                    "net_proceeds_if_filled": round(_safe_float((close_quote.get("resting_exit") or {}).get("net_proceeds_if_filled")), 4),
                    "net_pnl_if_filled": round(_safe_float((close_quote.get("resting_exit") or {}).get("net_pnl_if_filled")), 4),
                    "net_pnl_after_rebate_estimate": round(_safe_float((close_quote.get("resting_exit") or {}).get("net_pnl_after_rebate_estimate")), 4),
                },
            },
            "override_active": override_active,
            "pending_exit_order_id": pending_exit_order_id,
            "pending_exit_price": pending_exit_price,
            "pending_exit_quantity": pending_exit_quantity,
            "pending_exit_age_label": _format_age_label(pending_exit_age),
            "age_label": _format_age_label(age_seconds),
            "max_hold_label": _format_age_label(max_hold_seconds),
            "placed_at_ts": placed_at or 0,
            "max_hold_seconds": max_hold_seconds or 0,
            "force_exit_in_label": (
                "manual hold"
                if override_active
                else
                "exit pending"
                if pending_exit_order_id
                else
                "expired"
                if force_exit_in is not None and force_exit_in <= 0
                else _format_age_label(force_exit_in)
                if force_exit_in is not None
                else "—"
            ),
            "force_exit_at": force_exit_at or "",
            "force_exit_at_ts": force_exit_at_ts if age_seconds is not None and max_hold_seconds is not None else 0,
        })

    position_value = sum(p["current_value"] for p in positions)
    total = cash + position_value
    return {
        "positions": positions,
        "cash_balance": round(cash, 2),
        "position_value": round(position_value, 2),
        "total": round(total, 2),
        "source": "live",
    }


@app.get("/api/closed-trades")
async def api_closed_trades(_=Depends(verify_password)):
    """Return live account closed trades plus local cancelled trades, newest first."""
    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("""
            SELECT * FROM trades
            WHERE status IN ('closed', 'cancelled')
            ORDER BY COALESCE(resolved_at, timestamp) DESC
            LIMIT 100
        """) as cur:
            status_rows = [dict(r) for r in await cur.fetchall()]

        async with conn.execute("""
            SELECT * FROM trades
            ORDER BY COALESCE(resolved_at, timestamp) DESC
            LIMIT 300
        """) as cur:
            history_rows = [dict(r) for r in await cur.fetchall()]

    strategy_lookup: dict[tuple[str, str], dict] = {}
    for row in history_rows:
        side = row.get("side") or ""
        for market_ref in (row.get("market_slug"), row.get("question")):
            key = (_normalize_text(market_ref), side)
            if key[0] and key not in strategy_lookup:
                strategy_lookup[key] = row

    live_closed_rows: list[dict] = []
    live_closed_positions = await _get_live_closed_positions(limit=150)
    for pos in live_closed_positions:
        outcome = str(pos.get("outcome") or "").upper()
        side = "ORDER_INTENT_BUY_LONG" if outcome == "YES" else "ORDER_INTENT_BUY_SHORT"
        slug = pos.get("slug") or ""
        title = pos.get("title") or slug or "—"
        match = (
            strategy_lookup.get((_normalize_text(slug), side))
            or strategy_lookup.get((_normalize_text(title), side))
        )
        avg_price = _safe_float(pos.get("avgPrice"))
        total_bought = _safe_float(pos.get("totalBought"))
        quantity = _safe_float(pos.get("size"))
        if quantity <= 0 and avg_price > 0:
            quantity = total_bought / avg_price

        live_closed_rows.append({
            "timestamp": _iso_from_polymarket_timestamp(pos.get("timestamp")),
            "resolved_at": _iso_from_polymarket_timestamp(pos.get("timestamp")),
            "strategy": (match or {}).get("strategy") or "live_account",
            "market_slug": slug,
            "question": title,
            "side": side,
            "price": avg_price,
            "quantity": quantity,
            "order_id": (match or {}).get("order_id") or "",
            "status": "closed",
            "pnl": _safe_float(pos.get("realizedPnl")),
        })

    local_cancelled = [row for row in status_rows if row.get("status") == "cancelled"]
    # Closed-trade reporting should follow exchange truth. Historical local
    # "closed" rows may represent posted exit attempts, not actual realized
    # fills, so we do not use them for P&L cards or win-rate stats.
    closed = live_closed_rows
    all_trades = sorted(
        [*closed, *local_cancelled],
        key=_trade_sort_key,
        reverse=True,
    )
    trades = all_trades[:100]

    realized_pnl = round(sum(_safe_float(r.get("pnl")) for r in closed), 2)
    wins = sum(1 for r in closed if _safe_float(r.get("pnl")) > 0)
    total_closed = len(closed)
    win_rate = round((wins / total_closed) * 100, 1) if total_closed else 0.0

    strategy_rollup: dict[str, dict] = {}
    for row in closed:
        strategy = row.get("strategy") or "unknown"
        bucket = strategy_rollup.setdefault(strategy, {"strategy": strategy, "pnl": 0.0, "wins": 0, "total": 0})
        pnl = _safe_float(row.get("pnl"))
        bucket["pnl"] += pnl
        bucket["total"] += 1
        if pnl > 0:
            bucket["wins"] += 1

    strategy_stats = sorted(
        (
            {
                "strategy": s["strategy"],
                "pnl": round(s["pnl"], 2),
                "wins": s["wins"],
                "total": s["total"],
            }
            for s in strategy_rollup.values()
        ),
        key=lambda s: abs(s["pnl"]),
        reverse=True,
    )

    today = datetime.utcnow().date().isoformat()
    trades_today = sum(
        1
        for row in all_trades
        if str(row.get("resolved_at") or row.get("timestamp") or "").startswith(today)
    )

    return {
        "trades": trades,
        "summary": {
            "realized_pnl": realized_pnl,
            "wins": wins,
            "total_closed": total_closed,
            "win_rate": win_rate,
            "total_trades": len(all_trades),
            "trades_today": trades_today,
            "strategy_stats": strategy_stats,
        },
    }


@app.get("/api/circuit-breaker")
async def api_circuit_breaker_status(_=Depends(verify_password)):
    """Return circuit breaker state."""
    if _circuit_breaker is None:
        return {"tripped": False, "reason": "", "thresholds": {}}
    cb = _circuit_breaker
    return {
        "tripped": cb.tripped,
        "reason": cb.trip_reason,
        "thresholds": {
            "max_daily_loss_usdc": cb.max_daily_loss_usdc,
            "max_drawdown_pct": cb.max_drawdown_pct,
            "max_consecutive_losses": cb.max_consecutive_losses,
            "max_recent_loss_usdc": cb.max_recent_loss_usdc,
            "recent_loss_window": cb.recent_loss_window,
            "max_orders_per_minute": cb.max_orders_per_minute,
        },
        "state": {
            "consecutive_losses": cb._consecutive_losses,
            "session_start_balance": round(cb.session_start_balance, 2),
            "day_start_balance": round(cb.day_start_balance, 2),
        },
    }


@app.post("/api/circuit-breaker/reset")
async def api_circuit_breaker_reset(_=Depends(verify_password)):
    """Manually reset the circuit breaker after reviewing the situation."""
    if _circuit_breaker is None:
        raise HTTPException(status_code=503, detail="Bot not running")
    if _capital is None:
        raise HTTPException(status_code=503, detail="Bot not running")
    reset_value = _capital.total_usdc
    if _client_ref is not None:
        reset_value = await _get_account_equity(_client_ref, _capital.total_usdc)
    _circuit_breaker.reset(reset_value)
    # Also clear the bot error state so it can resume
    if _bot_state["status"] == "error" and "Circuit breaker" in (_bot_state.get("last_error") or ""):
        _bot_state["status"] = "running"
        _bot_state["last_error"] = None
    msg = "Circuit breaker manually reset via dashboard"
    logger.info(msg)
    await db.log_to_db("INFO", msg)
    return {"ok": True}


@app.post("/api/close-live-position")
async def api_close_live_position(payload: dict, _=Depends(verify_password)):
    """Sell a live position immediately if possible, otherwise optionally post a resting 1c exit."""
    if _order_manager is None or _client_ref is None:
        raise HTTPException(status_code=503, detail="Bot not running")

    market_slug = str(payload.get("market_slug") or "").strip()
    title = str(payload.get("title") or market_slug or "").strip()
    outcome = str(payload.get("outcome") or "").upper().strip()
    quantity = _safe_float(payload.get("quantity"))

    if not market_slug or outcome not in {"YES", "NO"} or quantity <= 0:
        raise HTTPException(
            status_code=400,
            detail="market_slug, outcome, and quantity are required",
        )

    force_resting = bool(payload.get("force_resting"))

    quote = await _get_close_quote(
        market_slug,
        outcome,
        quantity,
        _safe_float(payload.get("entry_price")),
    )
    if not quote.get("has_liquidity") and not force_resting:
        detail = (
            "No executable buyer is available right now. You can post a resting "
            "1c exit order, but it will only close if another trader buys it."
        )
        logger.info(f"Manual close no-fill for {market_slug}: no executable bid")
        raise HTTPException(status_code=409, detail=detail)

    intent = "ORDER_INTENT_BUY_LONG" if outcome == "YES" else "ORDER_INTENT_BUY_SHORT"
    # Executable closes use FAK. No-bid exits can only be posted as resting
    # orders and will remain open until another trader crosses them.
    exit_price = 0.01
    tif = "TIME_IN_FORCE_GOOD_TILL_CANCEL" if force_resting and not quote.get("has_liquidity") else "TIME_IN_FORCE_FILL_AND_KILL"
    strategy = "manual_resting_exit" if tif == "TIME_IN_FORCE_GOOD_TILL_CANCEL" else "manual_exit"

    pending_exit = _order_manager.get_pending_exit_order(market_slug, intent)
    if pending_exit:
        pending_id = str(pending_exit.get("order_id") or "")
        if pending_id:
            await _order_manager.cancel_order(pending_id)

    oid = await _order_manager.place_order(
        market_slug=market_slug,
        question=title or market_slug,
        intent=intent,
        price=exit_price,
        quantity=quantity,
        strategy=strategy,
        execution_side=SELL,
        tif=tif,
    )

    if oid:
        _account_cache["positions_ts"] = 0.0
        if tif == "TIME_IN_FORCE_GOOD_TILL_CANCEL":
            resting = quote.get("resting_exit") or {}
            msg = (
                f"Manual resting exit posted for {market_slug}: "
                f"{quantity:.1f}x {outcome} @ ${exit_price:.4f}; "
                f"if filled pnl=${_safe_float(resting.get('net_pnl_if_filled')):.2f}"
            )
        else:
            msg = (
                f"Manual close submitted for {market_slug}: "
                f"{quantity:.1f}x {outcome} @ ${exit_price:.4f}"
            )
        logger.info(msg)
        await db.log_to_db("INFO", msg)
        return {
            "ok": True,
            "order_id": oid,
            "price": exit_price,
            "resting": tif == "TIME_IN_FORCE_GOOD_TILL_CANCEL",
            "quote": quote,
        }

    if getattr(_order_manager, "last_order_status", "") == "no_match":
        detail = (
            "No matching exit liquidity at the current/minimum tick. "
            "The position remains open until a buyer appears or the market resolves."
        )
        logger.info(
            f"Manual close no-fill for {market_slug}: "
            f"best_bid=${_safe_float(quote.get('best_bid')):.4f}, attempted=${exit_price:.4f}"
        )
        raise HTTPException(status_code=409, detail=detail)

    raise HTTPException(status_code=400, detail="Manual close order failed")


@app.post("/api/close-position/{order_id}")
async def api_close_position(order_id: str, _=Depends(verify_password)):
    """Manually cancel/close an open position by order ID."""
    if _order_manager is None:
        raise HTTPException(status_code=503, detail="Bot not running")

    success = await _order_manager.cancel_order(order_id)
    if success:
        if _capital:
            # Try to find which strategy owned this and release capital
            # (best-effort — capital was already debited when the order was placed)
            pass
        return {"ok": True, "order_id": order_id}
    raise HTTPException(status_code=400, detail=f"Could not close order {order_id}")
