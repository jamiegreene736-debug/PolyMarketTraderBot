"""
End-to-end test suite for PolyMarket Trader
============================================
Tests every layer without placing real orders or requiring live API keys:

  Tier 1 — Imports          All source modules import cleanly
  Tier 2 — Capital Manager  Kelly sizing, allocation, limits
  Tier 3 — Circuit Breaker  All 4 triggers fire correctly
  Tier 4 — Database         init, write, read, stats query
  Tier 5 — Order Manager    Dedup, rate-limit, in-memory tracking
  Tier 6 — Strategy init    All 9 strategies instantiate without errors
  Tier 7 — Dashboard API    FastAPI endpoints respond correctly (bot stopped)

Run:
    python3 test_e2e.py
"""

import asyncio
import os
import sys
import time
import tempfile
import subprocess
import traceback
import httpx
from unittest.mock import AsyncMock, MagicMock

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
HEAD = "\033[94m{}\033[0m"

results = []

def check(name: str, passed: bool, detail: str = ""):
    tag = PASS if passed else FAIL
    msg = f"{tag}  {name}"
    if detail:
        msg += f"\n        {detail}"
    print(msg)
    results.append((name, passed, detail))

async def acheck(name: str, coro, expect=None):
    try:
        val = await coro
        if expect is not None:
            ok = val == expect
            check(name, ok, f"got {val!r}, expected {expect!r}" if not ok else "")
        else:
            check(name, True)
        return val
    except Exception as e:
        check(name, False, traceback.format_exc(limit=2).strip())
        return None

def section(title: str):
    print(f"\n{HEAD.format('── ' + title + ' ' + '─' * max(0, 50 - len(title)))}")


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Imports
# ══════════════════════════════════════════════════════════════════════════════
section("Tier 1: Imports")

modules = [
    ("src.capital_manager",    "CapitalManager"),
    ("src.circuit_breaker",    "CircuitBreaker"),
    ("src.database",           "init_db"),
    ("src.order_manager",      "OrderManager"),
    ("src.market_data",        "MarketData"),
    ("src.news_client",        "NewsClient"),
    ("src.strategies.base",              "BaseStrategy"),
    ("src.strategies.near_certainty",    "NearCertaintyStrategy"),
    ("src.strategies.inverted_near_certainty", "InvertedNearCertaintyStrategy"),
    ("src.strategies.market_making",     "MarketMakingStrategy"),
    ("src.strategies.logical_arb",       "LogicalArbStrategy"),
    ("src.strategies.ai_trader",         "AITradingStrategy"),
    ("src.strategies.position_monitor",  "PositionMonitorStrategy"),
    ("src.strategies.whale_tracker",     "WhaleTrackerStrategy"),
    ("src.strategies.cross_platform",    "CrossPlatformArbStrategy"),
    ("src.strategies.news_catalyst",     "NewsCatalystStrategy"),
    ("dashboard",                        "app"),
]

imported = {}
for mod, attr in modules:
    try:
        m = __import__(mod, fromlist=[attr])
        obj = getattr(m, attr)
        imported[attr] = obj
        check(f"import {mod}.{attr}", True)
    except Exception as e:
        check(f"import {mod}.{attr}", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Capital Manager
# ══════════════════════════════════════════════════════════════════════════════
section("Tier 2: Capital Manager")

from src.capital_manager import CapitalManager

cfg = {
    "near_certainty":          {"capital_pct": 20},
    "inverted_near_certainty": {"capital_pct": 20},
    "market_making":           {"capital_pct": 25},
    "logical_arb":             {"capital_pct": 15},
    "ai_trader":               {"capital_pct": 15},
    "whale_tracker":           {"capital_pct": 15},
}
cap = CapitalManager(total_usdc=1000.0, strategy_config=cfg, reserve_pct=10)

check("reserve_pct holds back 10%",
      abs(cap.available_usdc - 900.0) < 0.01,
      f"available={cap.available_usdc}")

check("strategy_limit near_certainty = $200",
      abs(cap.strategy_limit("near_certainty") - 200.0) < 0.01)

check("allocate $150 to near_certainty succeeds",
      cap.allocate("near_certainty", 150.0))

check("allocate another $60 to near_certainty fails (would exceed $200 limit)",
      not cap.allocate("near_certainty", 60.0))

cap.release("near_certainty", 150.0)
check("release restores availability",
      cap.strategy_available("near_certainty") >= 150.0)

# Kelly sizing
size = cap.kelly_size("near_certainty", win_prob=0.95, net_return_pct=0.06, kelly_fraction=0.25)
check("kelly_size returns float > 0", isinstance(size, float) and size > 0, f"size={size}")

min_kelly_size = 1.0
size_zero_edge = cap.kelly_size(
    "near_certainty",
    win_prob=0.50,
    net_return_pct=0.00,
    kelly_fraction=0.25,
    min_size=min_kelly_size,
)
check("kelly_size with zero edge returns min_size",
      size_zero_edge == min_kelly_size, f"size={size_zero_edge}")

cap2 = CapitalManager(total_usdc=10.0, strategy_config=cfg, reserve_pct=10)
tiny = cap2.kelly_size("near_certainty", win_prob=0.95, net_return_pct=0.06)
check("kelly_size capped by strategy_available when balance is tiny",
      tiny <= cap2.strategy_available("near_certainty") or tiny == 10.0,
      f"size={tiny}, available={cap2.strategy_available('near_certainty')}")

cap.update_balance(500.0)
check("update_balance changes total_usdc",
      cap.total_usdc == 500.0)


# ══════════════════════════════════════════════════════════════════════════════
# Tier 3 — Circuit Breaker
# ══════════════════════════════════════════════════════════════════════════════
section("Tier 3: Circuit Breaker")

from src.circuit_breaker import CircuitBreaker

CB_CFG = {
    "circuit_breaker": {
        "max_daily_loss_usdc": 50.0,
        "max_drawdown_pct": 0.20,
        "max_consecutive_losses": 3,
        "max_orders_per_minute": 5,
    }
}

async def test_circuit_breaker():
    # Patch db.log_to_db so the circuit breaker doesn't need a real DB
    import src.circuit_breaker as cb_mod
    import src.database as _db
    original_log = _db.log_to_db
    _db.log_to_db = AsyncMock()

    # Normal state — should pass
    cb = CircuitBreaker(CB_CFG, start_balance=1000.0)
    ok = await cb.check(current_balance=990.0, recent_pnl_list=[10, -5, 8])
    check("CB: no trigger when healthy", ok)

    # Trigger 1: daily loss
    cb2 = CircuitBreaker(CB_CFG, start_balance=1000.0)
    ok = await cb2.check(current_balance=940.0, recent_pnl_list=[])   # lost $60 > $50 limit
    check("CB trigger 1: daily loss limit ($60 > $50)", not ok)
    check("CB trip_reason mentions daily loss", "Daily loss" in cb2.trip_reason)

    # Trigger 2: portfolio drawdown — use a small max_daily_loss so only drawdown fires
    drawdown_cfg = {
        "circuit_breaker": {
            "max_daily_loss_usdc": 9999.0,   # disable daily-loss so drawdown fires first
            "max_drawdown_pct": 0.20,
            "max_consecutive_losses": 99,
            "max_orders_per_minute": 9999,
        }
    }
    cb3 = CircuitBreaker(drawdown_cfg, start_balance=1000.0)
    ok = await cb3.check(current_balance=750.0, recent_pnl_list=[])   # 25% down > 20% limit
    check("CB trigger 2: drawdown (25% > 20%)", not ok)
    check("CB trip_reason mentions drawdown", "drawdown" in cb3.trip_reason.lower())

    # Trigger 3: consecutive losses
    cb4 = CircuitBreaker(CB_CFG, start_balance=1000.0)
    ok = await cb4.check(current_balance=990.0, recent_pnl_list=[-5, -3, -8])   # 3 losses in a row
    check("CB trigger 3: consecutive losses (3 >= limit 3)", not ok)
    check("CB trip_reason mentions consecutive", "consecutive" in cb4.trip_reason.lower())

    # Trigger 4: rapid order rate
    cb5 = CircuitBreaker(CB_CFG, start_balance=1000.0)
    for _ in range(6):   # 6 orders > limit of 5
        cb5.record_order()
    ok = await cb5.check(current_balance=990.0, recent_pnl_list=[])
    check("CB trigger 4: rapid order rate (6 > 5/min)", not ok)
    check("CB trip_reason mentions order rate", "order" in cb5.trip_reason.lower())

    # Reset works
    cb5.reset(current_balance=990.0)
    check("CB reset clears tripped state", not cb5.tripped)
    ok = await cb5.check(current_balance=990.0, recent_pnl_list=[])
    check("CB safe to trade after reset", ok)

    # Restore original
    _db.log_to_db = original_log

asyncio.run(test_circuit_breaker())


# ══════════════════════════════════════════════════════════════════════════════
# Tier 4 — Database
# ══════════════════════════════════════════════════════════════════════════════
section("Tier 4: Database")

import src.database as db_module

async def test_database():
    # Use a temp DB so we don't pollute production data
    original_path = db_module.DB_PATH
    db_module.DB_PATH = tempfile.mktemp(suffix=".db")

    try:
        await acheck("init_db creates tables", db_module.init_db())

        await acheck("log_to_db writes without error",
                     db_module.log_to_db("INFO", "test message"))

        await acheck("insert_trade writes row",
                     db_module.insert_trade(
                         strategy="near_certainty",
                         market_slug="test-market",
                         question="Will this test pass?",
                         side="ORDER_INTENT_BUY_LONG",
                         price=0.95,
                         quantity=10.5,
                         order_id="test-order-001",
                     ))

        await acheck("snapshot_balance writes row",
                     db_module.snapshot_balance(950.0, 0.0, 0.0))

        stats = await db_module.get_dashboard_stats()
        check("get_dashboard_stats returns dict with expected keys",
              all(k in stats for k in ["total_pnl", "win_rate", "open_positions",
                                        "total_trades", "recent_trades", "recent_logs"]),
              str(list(stats.keys())))

        check("total_trades == 1 after one insert",
              stats["total_trades"] == 1, f"got {stats['total_trades']}")

        check("open_positions == 1",
              stats["open_positions"] == 1, f"got {stats['open_positions']}")

        meta = await db_module.get_open_trades_metadata()
        check("get_open_trades_metadata returns our order",
              "test-order-001" in meta,
              f"keys={list(meta.keys())}")

        await db_module.close_trade("test-order-001", pnl=2.50)
        pnls = await db_module.get_recent_closed_pnls(limit=5)
        check("get_recent_closed_pnls returns closed trade pnl",
              len(pnls) == 1 and abs(pnls[0] - 2.50) < 0.001,
              f"pnls={pnls}")

        await db_module.cancel_trade("test-order-001")  # idempotent test
        await acheck("cancel_trade runs without error",
                     db_module.cancel_trade("nonexistent-order"))

    finally:
        import os
        try:
            os.unlink(db_module.DB_PATH)
        except Exception:
            pass
        db_module.DB_PATH = original_path

asyncio.run(test_database())


# ══════════════════════════════════════════════════════════════════════════════
# Tier 5 — Order Manager (in-memory, no live client)
# ══════════════════════════════════════════════════════════════════════════════
section("Tier 5: Order Manager (in-memory logic)")

from src.order_manager import OrderManager

async def test_order_manager():
    # Build a mock client
    mock_client = MagicMock()
    mock_client.place_order = AsyncMock(return_value={"id": "order-abc-123"})
    mock_client.cancel_order = AsyncMock(return_value=True)

    om = OrderManager(mock_client, max_concurrent=5)

    # Use a temp DB
    original_path = db_module.DB_PATH
    db_module.DB_PATH = tempfile.mktemp(suffix=".db")
    await db_module.init_db()

    try:
        oid = await om.place_order(
            market_slug="will-btc-hit-100k",
            question="Will BTC hit $100k?",
            intent="ORDER_INTENT_BUY_LONG",
            price=0.72,
            quantity=50.0,
            strategy="near_certainty",
        )
        check("place_order returns order_id", oid == "order-abc-123", f"got {oid!r}")
        check("order tracked in _open_orders", "order-abc-123" in om._open_orders)
        check("market slug tracked", "will-btc-hit-100k" in om._market_orders)
        check("get_total_open_orders == 1", om.get_total_open_orders() == 1)

        # Duplicate detection
        oid2 = await om.place_order(
            market_slug="will-btc-hit-100k",
            question="Will BTC hit $100k?",
            intent="ORDER_INTENT_BUY_LONG",
            price=0.72,      # same price — should be deduplicated
            quantity=50.0,
            strategy="near_certainty",
        )
        check("duplicate order rejected (same market+intent+price)", oid2 is None)
        check("still only 1 order after dupe attempt", om.get_total_open_orders() == 1)

        # Different price is NOT a duplicate
        mock_client.place_order = AsyncMock(return_value={"id": "order-xyz-456"})
        oid3 = await om.place_order(
            market_slug="will-btc-hit-100k",
            question="Will BTC hit $100k?",
            intent="ORDER_INTENT_BUY_LONG",
            price=0.65,      # different price — allowed
            quantity=30.0,
            strategy="near_certainty",
        )
        check("non-duplicate different-price order accepted", oid3 == "order-xyz-456")
        check("now 2 open orders", om.get_total_open_orders() == 2)

        # Max concurrent limit
        om2 = OrderManager(mock_client, max_concurrent=2)
        om2._open_orders = {"a": {}, "b": {}}   # simulate full
        oid4 = await om2.place_order("m", "q", "ORDER_INTENT_BUY_LONG", 0.5, 10, "test")
        check("order rejected when max_concurrent reached", oid4 is None)

        # Cancel order
        success = await om.cancel_order("order-abc-123")
        check("cancel_order returns True", success is True)
        check("cancelled order removed from _open_orders", "order-abc-123" not in om._open_orders)

        # get_open_positions
        positions = om.get_open_positions()
        check("get_open_positions returns remaining order",
              len(positions) == 1 and positions[0]["order_id"] == "order-xyz-456",
              f"positions={positions}")

        # mark_filled
        await om.mark_filled("order-xyz-456", pnl=5.0)
        check("mark_filled removes order", om.get_total_open_orders() == 0)

        # Immediate exit attempts (FAK/FOK) should not become immortal open rows.
        mock_client.place_order = AsyncMock(return_value={"id": "exit-fak-789", "order_type": "FAK"})
        oid5 = await om.place_order(
            market_slug="will-btc-hit-100k",
            question="Will BTC hit $100k?",
            intent="ORDER_INTENT_BUY_LONG",
            price=0.01,
            quantity=50.0,
            strategy="position_monitor",
            execution_side="SELL",
            tif="TIME_IN_FORCE_FILL_AND_KILL",
        )
        check("FAK exit submit returns order_id", oid5 == "exit-fak-789", f"got {oid5!r}")
        check("FAK exit is not tracked as resting open order", om.get_total_open_orders() == 0)

        mock_client.place_order = AsyncMock(return_value={
            "id": "exit-no-fill-000",
            "order_type": "FAK",
            "status": "no_match",
        })
        oid6 = await om.place_order(
            market_slug="will-btc-hit-100k",
            question="Will BTC hit $100k?",
            intent="ORDER_INTENT_BUY_LONG",
            price=0.01,
            quantity=50.0,
            strategy="position_monitor",
            execution_side="SELL",
            tif="TIME_IN_FORCE_FILL_AND_KILL",
        )
        check("FAK no-liquidity exit returns None", oid6 is None, f"got {oid6!r}")
        check("FAK no-liquidity status is not an error", om.last_order_status == "no_match")
        check("FAK no-liquidity is not tracked", om.get_total_open_orders() == 0)

    finally:
        try:
            os.unlink(db_module.DB_PATH)
        except Exception:
            pass
        db_module.DB_PATH = original_path

asyncio.run(test_order_manager())


def test_order_type_mapping():
    from src.client import PolymarketClient
    from py_clob_client.clob_types import OrderType

    check("client maps GTC tif", PolymarketClient._order_type_from_tif("TIME_IN_FORCE_GOOD_TILL_CANCEL") == OrderType.GTC)
    check("client maps FAK tif", PolymarketClient._order_type_from_tif("TIME_IN_FORCE_FILL_AND_KILL") == OrderType.FAK)
    check("client maps FOK tif", PolymarketClient._order_type_from_tif("TIME_IN_FORCE_FILL_OR_KILL") == OrderType.FOK)

test_order_type_mapping()


async def test_open_order_token_hydration():
    from src.client import PolymarketClient

    client = PolymarketClient("", "", "", "", dry_run=True)
    client._client = MagicMock()
    client._client.get_orders = MagicMock(return_value=[{
        "id": "exit-order-1",
        "asset_id": "YES_TOKEN",
        "side": "SELL",
        "price": "0.01",
        "size": "1500",
    }])

    async def fake_get_markets():
        client._slug_tokens["will-test-market"] = {
            "condition_id": "condition-1",
            "yes_token_id": "YES_TOKEN",
            "no_token_id": "NO_TOKEN",
        }
        return []

    client.get_markets = fake_get_markets
    try:
        orders = await client.get_open_orders()
        check("open-order sync hydrates token mapping before classification",
              len(orders) == 1 and orders[0]["market_slug"] == "will-test-market")
        check("restored SELL YES order keeps BUY_LONG intent",
              orders[0]["execution_side"] == "SELL"
              and orders[0]["intent"] == "ORDER_INTENT_BUY_LONG")
    finally:
        await client.close()

asyncio.run(test_open_order_token_hydration())


# ══════════════════════════════════════════════════════════════════════════════
# Tier 6 — Strategy Initialization
# ══════════════════════════════════════════════════════════════════════════════
section("Tier 6: Strategy Initialization")

import yaml
from src.capital_manager import CapitalManager

with open("config.yaml") as f:
    full_config = yaml.safe_load(f)

strat_cfg = full_config.get("strategies", {})
mock_client = MagicMock()
mock_market_data = MagicMock()
mock_order_manager = MagicMock()
mock_capital = CapitalManager(total_usdc=500.0, strategy_config=strat_cfg, reserve_pct=10)

from src.strategies.near_certainty import NearCertaintyStrategy
from src.strategies.inverted_near_certainty import InvertedNearCertaintyStrategy
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.logical_arb import LogicalArbStrategy
from src.strategies.ai_trader import AITradingStrategy
from src.strategies.position_monitor import PositionMonitorStrategy
from src.strategies.whale_tracker import WhaleTrackerStrategy
from src.strategies.cross_platform import CrossPlatformArbStrategy
from src.strategies.news_catalyst import NewsCatalystStrategy
from src.news_client import NewsClient

strategy_classes = [
    ("NearCertaintyStrategy",          NearCertaintyStrategy,
     ["near_certainty", strat_cfg["near_certainty"], mock_client, mock_market_data, mock_order_manager, mock_capital]),
    ("InvertedNearCertaintyStrategy",  InvertedNearCertaintyStrategy,
     ["inverted_near_certainty", strat_cfg["inverted_near_certainty"], mock_client, mock_market_data, mock_order_manager, mock_capital]),
    ("MarketMakingStrategy",           MarketMakingStrategy,
     ["market_making", strat_cfg["market_making"], mock_client, mock_market_data, mock_order_manager, mock_capital]),
    ("LogicalArbStrategy",             LogicalArbStrategy,
     ["logical_arb", strat_cfg["logical_arb"], mock_client, mock_market_data, mock_order_manager, mock_capital]),
    ("PositionMonitorStrategy",        PositionMonitorStrategy,
     ["position_monitor", strat_cfg.get("position_monitor", {"enabled": True}), mock_client, mock_market_data, mock_order_manager, mock_capital]),
    ("WhaleTrackerStrategy",           WhaleTrackerStrategy,
     ["whale_tracker", strat_cfg.get("whale_tracker", {"enabled": True}), mock_client, mock_market_data, mock_order_manager, mock_capital]),
    ("CrossPlatformArbStrategy",       CrossPlatformArbStrategy,
     ["cross_platform_arb", strat_cfg["cross_platform_arb"], mock_client, mock_market_data, mock_order_manager, mock_capital]),
    ("NewsCatalystStrategy",           NewsCatalystStrategy,
     ["news_catalyst", strat_cfg["news_catalyst"], mock_client, mock_market_data, mock_order_manager, mock_capital]),
]

# AITradingStrategy takes an extra news_client kwarg
news_client = NewsClient(api_key="test")
ai_args = ["ai_trader", strat_cfg.get("ai_trader", {"enabled": False}),
           mock_client, mock_market_data, mock_order_manager, mock_capital]

for name, cls, args in strategy_classes:
    try:
        s = cls(*args)
        check(f"{name} instantiates", True)
        check(f"{name} has .enabled attribute", hasattr(s, "enabled"))
        check(f"{name} has .run() method", callable(getattr(s, "run", None)))
    except Exception as e:
        check(f"{name} instantiates", False, str(e))

try:
    ai = AITradingStrategy(*ai_args, news_client=news_client)
    check("AITradingStrategy instantiates (with news_client)", True)
    check("AITradingStrategy has .run()", callable(getattr(ai, "run", None)))
except Exception as e:
    check("AITradingStrategy instantiates", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Tier 7 — Dashboard API (live FastAPI, bot stopped)
# ══════════════════════════════════════════════════════════════════════════════
section("Tier 7: Dashboard API (FastAPI endpoints)")

import os
import signal

env = os.environ.copy()
env["DASHBOARD_PASSWORD"] = "testpass"
env["POLY_API_KEY"] = "fake-api-key"
env["POLY_API_SECRET"] = "fake-api-secret"
env["POLY_API_PASSPHRASE"] = "fake-passphrase"
env["POLY_PRIVATE_KEY"] = "0x" + "11" * 32
env["NEWS_API_KEY"] = "fake-news-key"
env["ANTHROPIC_API_KEY"] = "sk-ant-fake"
env["OPENAI_API_KEY"] = "sk-fake-openai"

server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "dashboard:app",
     "--host", "127.0.0.1", "--port", "18432", "--log-level", "warning"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for server to be ready
ready = False
for _ in range(30):
    time.sleep(0.5)
    try:
        r = httpx.get("http://127.0.0.1:18432/api/status",
                      auth=("admin", "testpass"), timeout=2)
        if r.status_code in (200, 401):
            ready = True
            break
    except Exception:
        pass

if not ready:
    stdout, stderr = server.communicate(timeout=3)
    check("FastAPI server started", False,
          f"stdout: {stdout.decode()[-500:]}\nstderr: {stderr.decode()[-500:]}")
else:
    async def test_api():
        auth = ("admin", "testpass")
        bad_auth = ("admin", "wrongpass")
        base = "http://127.0.0.1:18432"

        async with httpx.AsyncClient() as client:
            # Auth guard
            r = await client.get(f"{base}/api/status", auth=bad_auth)
            check("GET /api/status rejects bad password", r.status_code == 401)

            # /api/status
            r = await client.get(f"{base}/api/status", auth=auth)
            check("GET /api/status returns 200", r.status_code == 200)
            data = r.json()
            check("/api/status has 'status' field", "status" in data, str(data))
            check("/api/status.status == 'stopped'", data.get("status") == "stopped",
                  f"got {data.get('status')!r}")

            # /api/stats
            r = await client.get(f"{base}/api/stats", auth=auth)
            check("GET /api/stats returns 200", r.status_code == 200)
            data = r.json()
            check("/api/stats has expected keys",
                  all(k in data for k in ["total_pnl", "win_rate", "open_positions",
                                           "total_trades", "recent_trades", "recent_logs"]),
                  str(list(data.keys())))

            # /api/circuit-breaker
            r = await client.get(f"{base}/api/circuit-breaker", auth=auth)
            check("GET /api/circuit-breaker returns 200", r.status_code == 200)
            data = r.json()
            check("/api/circuit-breaker has 'tripped' field", "tripped" in data, str(data))
            check("circuit breaker not tripped at startup", data.get("tripped") is False)

            # /api/positions (bot not running)
            r = await client.get(f"{base}/api/positions", auth=auth)
            check("GET /api/positions returns 200", r.status_code == 200)
            data = r.json()
            check("/api/positions has 'positions' list", "positions" in data)
            check("/api/positions is empty when bot stopped",
                  data.get("positions") == [], f"got {data.get('positions')}")

            # /api/circuit-breaker/reset when bot not running
            r = await client.post(f"{base}/api/circuit-breaker/reset", auth=auth)
            check("POST /api/circuit-breaker/reset returns 503 when bot stopped",
                  r.status_code == 503)

            # Dashboard HTML
            r = await client.get(f"{base}/", auth=auth)
            check("GET / returns HTML", r.status_code == 200)
            check("Dashboard HTML contains 'PolyMarket'",
                  "PolyMarket" in r.text, f"first 200 chars: {r.text[:200]}")
            check("Dashboard HTML has circuit breaker banner",
                  "cb-banner" in r.text)
            check("Dashboard HTML has balance cards",
                  "cashBalance" in r.text)

    asyncio.run(test_api())

server.terminate()
try:
    server.wait(timeout=5)
except subprocess.TimeoutExpired:
    server.kill()


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
section("Results")

total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n  {passed}/{total} passed", end="")
if failed:
    print(f"  (\033[91m{failed} failed\033[0m)")
    print("\n  Failed tests:")
    for name, ok, detail in results:
        if not ok:
            print(f"    \033[91m✗\033[0m {name}")
            if detail:
                for line in detail.splitlines()[:4]:
                    print(f"      {line}")
else:
    print("  \033[92m✓ All good\033[0m")

sys.exit(0 if failed == 0 else 1)
