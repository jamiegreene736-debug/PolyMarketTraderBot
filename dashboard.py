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
from contextlib import asynccontextmanager
from datetime import datetime

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from jinja2 import Environment, FileSystemLoader
import secrets
from loguru import logger

from src.logger import setup_logger
from src.client import PolymarketClient
from src.market_data import MarketData
from src.order_manager import OrderManager
from src.capital_manager import CapitalManager
from src import database as db

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

load_dotenv()

# ── Shared in-memory bot state ────────────────────────────────────────────────
_bot_state = {
    "status": "stopped",        # stopped | running | error
    "last_heartbeat": None,
    "last_error": None,
    "task": None,               # asyncio.Task reference
}


def load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# ── Bot loop ──────────────────────────────────────────────────────────────────

async def run_bot_loop():
    config = load_config()
    bot_cfg = config.get("bot", {})

    key_id = os.getenv("POLYMARKET_KEY_ID")
    secret_key = os.getenv("POLYMARKET_SECRET_KEY")

    if not key_id or not secret_key:
        _bot_state["status"] = "error"
        _bot_state["last_error"] = "Missing POLYMARKET_KEY_ID or POLYMARKET_SECRET_KEY"
        logger.error(_bot_state["last_error"])
        return

    client = PolymarketClient(
        key_id=key_id,
        secret_key=secret_key,
        dry_run=bot_cfg.get("dry_run", False),
    )
    await client.connect()

    market_data = MarketData(client)
    order_manager = OrderManager(client, max_concurrent=bot_cfg.get("max_concurrent_orders", 20))
    capital = CapitalManager(
        total_usdc=1000.0,
        strategy_config=config.get("strategies", {}),
        reserve_pct=config.get("capital", {}).get("reserve_pct", 10),
    )

    news_client = NewsClient(api_key=os.getenv("NEWS_API_KEY", ""))

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
                             client, market_data, order_manager, capital),
    ]

    enabled = [s for s in strategies if s.enabled]
    poll_interval = bot_cfg.get("poll_interval_seconds", 30)
    balance_refresh_counter = 0

    msg = f"Bot started — active strategies: {[s.name for s in enabled]}"
    logger.info(msg)
    await db.log_to_db("INFO", msg)

    try:
        while _bot_state["status"] == "running":
            try:
                # Refresh balance every 10 ticks
                if balance_refresh_counter % 10 == 0:
                    try:
                        balance_data = await client.get_balance()
                        for key in ("availableBalance", "balance", "usdc", "availableUsdc", "cashBalance"):
                            val = balance_data.get(key)
                            if val is not None:
                                balance = float(val)
                                capital.update_balance(balance)
                                await db.snapshot_balance(balance, 0.0, 0.0)
                                msg = f"Balance refreshed: ${balance:.2f} USDC"
                                logger.info(msg)
                                await db.log_to_db("INFO", msg)
                                break
                    except Exception as e:
                        msg = f"Balance refresh failed: {e}"
                        logger.warning(msg)
                        await db.log_to_db("WARNING", msg)

                balance_refresh_counter += 1

                tick_msg = (
                    f"Tick #{balance_refresh_counter} | "
                    f"open_orders={order_manager.get_total_open_orders()} | "
                    f"strategies={[s.name for s in enabled]}"
                )
                logger.info(tick_msg)
                await db.log_to_db("INFO", tick_msg)

                for s in enabled:
                    try:
                        await s.run()
                    except Exception as e:
                        msg = f"[{s.name}] crashed: {e}"
                        logger.error(msg)
                        await db.log_to_db("ERROR", msg)

                _bot_state["last_heartbeat"] = datetime.utcnow().isoformat()
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
        await client.cancel_all_orders()
        await client.close()
        logger.info("Bot stopped cleanly.")

    _bot_state["status"] = "stopped"


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    await db.init_db()
    yield
    # Cancel bot task on shutdown
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(_=Depends(verify_password)):
    stats = await db.get_dashboard_stats()
    template = jinja_env.get_template("index.html")
    html = template.render(**stats)
    return HTMLResponse(content=html)


@app.get("/api/stats")
async def api_stats(_=Depends(verify_password)):
    return await db.get_dashboard_stats()


@app.get("/api/status")
async def api_status(_=Depends(verify_password)):
    state = dict(_bot_state)
    state.pop("task", None)

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
    task = _bot_state.get("task")
    if task and not task.done():
        return {"status": _bot_state["status"]}

    _bot_state["status"] = "running"
    _bot_state["last_error"] = None
    _bot_state["task"] = asyncio.create_task(run_bot_loop())
    logger.info("Bot started via dashboard")
    return {"status": "running"}


@app.post("/api/stop")
async def api_stop(_=Depends(verify_password)):
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
