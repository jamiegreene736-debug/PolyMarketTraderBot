"""
PolyMarket Trader Bot
=====================
Runs all enabled trading strategies concurrently against Polymarket.us.
Strategies are configured in config.yaml. API keys are loaded from .env.

Usage:
    python main.py

Deploy:
    Railway auto-runs this via the Procfile `worker` process.
"""

import asyncio
import os
import sys
import yaml
from dotenv import load_dotenv
from loguru import logger

from src.logger import setup_logger
from src.client import PolymarketClient
from src.market_data import MarketData
from src.order_manager import OrderManager
from src.capital_manager import CapitalManager
from src import database as db

from src.strategies.near_certainty import NearCertaintyStrategy
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.logical_arb import LogicalArbStrategy
from src.strategies.cross_platform import CrossPlatformArbStrategy
from src.strategies.news_catalyst import NewsCatalystStrategy
from src.strategies.position_monitor import PositionMonitorStrategy
from src.strategies.whale_tracker import WhaleTrackerStrategy


def load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


async def refresh_balance(client: PolymarketClient, capital: CapitalManager):
    try:
        balance_data = await client.get_balance()
        # Try common field names returned by the SDK
        balance = 0.0
        for key in ("availableBalance", "balance", "usdc", "availableUsdc", "cashBalance"):
            val = balance_data.get(key)
            if val is not None:
                balance = float(val)
                break
        if balance > 0:
            capital.update_balance(balance)
            await db.snapshot_balance(balance, 0.0, 0.0)
            logger.info(f"Balance refreshed: ${balance:.2f} USDC")
    except Exception as e:
        logger.warning(f"Could not refresh balance: {e}")


async def run_strategy_safely(strategy):
    try:
        await strategy.run()
    except Exception as e:
        logger.error(f"Strategy '{strategy.name}' crashed: {e}")


async def main():
    load_dotenv()

    config = load_config()
    bot_cfg = config.get("bot", {})
    setup_logger(bot_cfg.get("log_level", "INFO"))

    logger.info("=" * 50)
    logger.info("PolyMarket Trader Bot starting...")
    logger.info(f"Dry run mode: {bot_cfg.get('dry_run', True)}")
    logger.info("=" * 50)

    api_key = os.getenv("POLY_API_KEY", "").strip()
    api_secret = os.getenv("POLY_API_SECRET", "").strip()
    api_passphrase = os.getenv("POLY_API_PASSPHRASE", "").strip()
    private_key = os.getenv("POLY_PRIVATE_KEY", "").strip()
    funder_address = os.getenv("POLY_FUNDER_ADDRESS", "").strip()
    signature_type_raw = os.getenv("POLY_SIGNATURE_TYPE", "").strip()
    signature_type = None
    if signature_type_raw:
        try:
            signature_type = int(signature_type_raw)
        except ValueError:
            logger.warning(f"Invalid POLY_SIGNATURE_TYPE={signature_type_raw!r}; using auto-detect")

    if not api_key or not api_secret or not api_passphrase or not private_key:
        missing = [
            name
            for name, value in {
                "POLY_API_KEY": api_key,
                "POLY_API_SECRET": api_secret,
                "POLY_API_PASSPHRASE": api_passphrase,
                "POLY_PRIVATE_KEY": private_key,
            }.items()
            if not value
        ]
        logger.error(f"Missing Polymarket env vars: {', '.join(missing)}")
        sys.exit(1)

    await db.init_db()

    client = PolymarketClient(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
        private_key=private_key,
        funder_address=funder_address,
        signature_type=signature_type,
        dry_run=bot_cfg.get("dry_run", True),
    )
    await client.connect()
    if not bot_cfg.get("dry_run", True) and client.signature_type == 0:
        await client.setup_allowances()

    market_data = MarketData(client)
    order_manager = OrderManager(client, max_concurrent=bot_cfg.get("max_concurrent_orders", 20))
    capital = CapitalManager(
        total_usdc=1000.0,  # placeholder until first balance refresh
        strategy_config=config.get("strategies", {}),
        reserve_pct=config.get("capital", {}).get("reserve_pct", 10),
    )

    strategies = [
        NearCertaintyStrategy("near_certainty", config["strategies"]["near_certainty"],
                              client, market_data, order_manager, capital),
        MarketMakingStrategy("market_making", config["strategies"]["market_making"],
                             client, market_data, order_manager, capital),
        LogicalArbStrategy("logical_arb", config["strategies"]["logical_arb"],
                           client, market_data, order_manager, capital),
        CrossPlatformArbStrategy("cross_platform_arb", config["strategies"]["cross_platform_arb"],
                                 client, market_data, order_manager, capital),
        NewsCatalystStrategy("news_catalyst", config["strategies"]["news_catalyst"],
                             client, market_data, order_manager, capital),
    ]

    enabled = [s for s in strategies if s.enabled]
    logger.info(f"Active strategies: {[s.name for s in enabled]}")

    poll_interval = bot_cfg.get("poll_interval_seconds", 30)
    balance_refresh_counter = 0

    try:
        while True:
            bot_status = await db.get_bot_status()

            if bot_status["status"] == "stopped":
                logger.info("Bot is paused (status=stopped). Waiting for start signal...")
                await asyncio.sleep(5)
                continue

            if bot_status["status"] == "error":
                logger.warning("Bot is in error state. Waiting for restart...")
                await asyncio.sleep(5)
                continue

            # Refresh balance every 10 ticks
            if balance_refresh_counter % 10 == 0:
                await refresh_balance(client, capital)
            balance_refresh_counter += 1

            logger.info(f"--- Running {len(enabled)} strategies | "
                        f"open orders={order_manager.get_total_open_orders()} | "
                        f"capital={capital.summary()} ---")

            try:
                await asyncio.gather(*[run_strategy_safely(s) for s in enabled])
                await db.update_heartbeat()
            except Exception as e:
                logger.error(f"Strategy loop error: {e}")
                await db.update_heartbeat(error=str(e))

            await asyncio.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info("Shutting down bot...")
    finally:
        await client.cancel_all_orders()
        await client.close()
        await db.set_bot_status("stopped")
        logger.info("Bot stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
