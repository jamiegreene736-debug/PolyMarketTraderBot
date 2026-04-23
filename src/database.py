import aiosqlite
from datetime import datetime
from loguru import logger

DB_PATH = "bot_data.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                strategy    TEXT NOT NULL,
                market_slug TEXT NOT NULL,
                question    TEXT,
                side        TEXT NOT NULL,
                price       REAL NOT NULL,
                quantity    REAL NOT NULL,
                order_id    TEXT,
                status      TEXT DEFAULT 'open',
                pnl         REAL DEFAULT 0.0,
                resolved_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS balance_snapshots (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                balance_usdc    REAL NOT NULL,
                realized_pnl    REAL DEFAULT 0.0,
                unrealized_pnl  REAL DEFAULT 0.0
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bot_logs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level     TEXT NOT NULL,
                message   TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bot_control (
                id             INTEGER PRIMARY KEY CHECK (id = 1),
                status         TEXT DEFAULT 'stopped',
                last_heartbeat TEXT,
                last_error     TEXT,
                updated_at     TEXT
            )
        """)
        # Ensure exactly one control row exists
        await db.execute("""
            INSERT OR IGNORE INTO bot_control (id, status, updated_at)
            VALUES (1, 'stopped', ?)
        """, (datetime.utcnow().isoformat(),))
        await db.commit()
    logger.info("Database initialized")


async def get_bot_status() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM bot_control WHERE id = 1") as cur:
            row = await cur.fetchone()
            if not row:
                return {"status": "stopped", "last_heartbeat": None, "last_error": None}
            data = dict(row)

    # Determine if bot is healthy based on heartbeat age
    heartbeat = data.get("last_heartbeat")
    if data["status"] == "running" and heartbeat:
        from datetime import timezone
        last = datetime.fromisoformat(heartbeat)
        age = (datetime.utcnow() - last).total_seconds()
        if age > 120:  # No heartbeat in 2 minutes = error
            data["status"] = "error"
    return data


async def set_bot_status(status: str, error: str = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE bot_control SET status = ?, last_error = ?, updated_at = ? WHERE id = 1
        """, (status, error, datetime.utcnow().isoformat()))
        await db.commit()


async def update_heartbeat(error: str = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE bot_control SET last_heartbeat = ?, last_error = ? WHERE id = 1
        """, (datetime.utcnow().isoformat(), error))
        await db.commit()


async def insert_trade(strategy: str, market_slug: str, question: str,
                       side: str, price: float, quantity: float, order_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO trades (timestamp, strategy, market_slug, question, side, price, quantity, order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), strategy, market_slug, question,
              side, price, quantity, order_id))
        await db.commit()


async def close_trade(order_id: str, pnl: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE trades SET status = 'closed', pnl = ?, resolved_at = ?
            WHERE order_id = ?
        """, (pnl, datetime.utcnow().isoformat(), order_id))
        await db.commit()


async def cancel_trade(order_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE trades SET status = 'cancelled'
            WHERE order_id = ?
        """, (order_id,))
        await db.commit()


async def snapshot_balance(balance: float, realized_pnl: float, unrealized_pnl: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO balance_snapshots (timestamp, balance_usdc, realized_pnl, unrealized_pnl)
            VALUES (?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), balance, realized_pnl, unrealized_pnl))
        await db.commit()


async def log_to_db(level: str, message: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO bot_logs (timestamp, level, message) VALUES (?, ?, ?)
        """, (datetime.utcnow().isoformat(), level, message))
        await db.commit()


async def get_recent_closed_pnls(limit: int = 20) -> list[float]:
    """Return PnL values for the most recent closed trades (newest first)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT pnl FROM trades WHERE status = 'closed' ORDER BY resolved_at DESC LIMIT ?",
            (limit,)
        ) as cur:
            rows = await cur.fetchall()
    return [row["pnl"] for row in rows]


async def get_open_trades_metadata() -> dict[str, dict]:
    """
    Return a dict of order_id → {strategy, question, market_slug, side, price, quantity}
    for all trades currently marked 'open' in the DB.
    Used by OrderManager.sync_from_exchange() to restore strategy/question on restart.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT order_id, strategy, question, market_slug, side, price, quantity, timestamp "
            "FROM trades WHERE status = 'open' AND order_id IS NOT NULL"
        ) as cur:
            rows = await cur.fetchall()
    return {row["order_id"]: dict(row) for row in rows}


async def get_open_trade_rows() -> list[dict]:
    """
    Return all trades still marked open.
    Used for rebuilding live position metadata after fills/restarts, even when
    the original order is no longer present in the in-memory order tracker.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT order_id, strategy, question, market_slug, side, price, quantity, timestamp "
            "FROM trades WHERE status = 'open' ORDER BY timestamp ASC"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def count_trades_today(strategy: str | None = None) -> int:
    """
    Count trades inserted today in UTC, optionally filtered by strategy.
    """
    today = datetime.utcnow().date().isoformat()
    query = "SELECT COUNT(*) as count FROM trades WHERE timestamp LIKE ?"
    params: tuple = (f"{today}%",)
    if strategy:
        query += " AND strategy = ?"
        params = (f"{today}%", strategy)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cur:
            row = await cur.fetchone()
    return int(row["count"] if row else 0)


async def get_dashboard_stats() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Total realized P&L
        async with db.execute("SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE status = 'closed'") as cur:
            row = await cur.fetchone()
            total_pnl = row["total"]

        # Win rate
        async with db.execute("SELECT COUNT(*) as total FROM trades WHERE status = 'closed'") as cur:
            total_closed = (await cur.fetchone())["total"]
        async with db.execute("SELECT COUNT(*) as wins FROM trades WHERE status = 'closed' AND pnl > 0") as cur:
            wins = (await cur.fetchone())["wins"]
        win_rate = round((wins / total_closed * 100), 1) if total_closed > 0 else 0.0

        # Open positions
        async with db.execute("SELECT COUNT(*) as open FROM trades WHERE status = 'open'") as cur:
            open_positions = (await cur.fetchone())["open"]

        # Total trades
        async with db.execute("SELECT COUNT(*) as total FROM trades") as cur:
            total_trades = (await cur.fetchone())["total"]

        # Trades today
        today = datetime.utcnow().date().isoformat()
        async with db.execute("SELECT COUNT(*) as count FROM trades WHERE timestamp LIKE ?", (f"{today}%",)) as cur:
            trades_today = (await cur.fetchone())["count"]

        # Recent trades (last 50)
        async with db.execute("""
            SELECT * FROM trades ORDER BY timestamp DESC LIMIT 50
        """) as cur:
            recent_trades = [dict(row) for row in await cur.fetchall()]

        # P&L per strategy
        async with db.execute("""
            SELECT strategy, COALESCE(SUM(pnl), 0) as pnl,
                   COUNT(*) as total,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM trades WHERE status = 'closed'
            GROUP BY strategy
        """) as cur:
            strategy_stats = [dict(row) for row in await cur.fetchall()]

        # Balance history (last 100 snapshots)
        async with db.execute("""
            SELECT timestamp, balance_usdc, realized_pnl
            FROM balance_snapshots ORDER BY timestamp DESC LIMIT 100
        """) as cur:
            balance_history = [dict(row) for row in await cur.fetchall()]
            balance_history.reverse()

        # Recent logs (last 100)
        async with db.execute("""
            SELECT * FROM bot_logs ORDER BY timestamp DESC LIMIT 100
        """) as cur:
            recent_logs = [dict(row) for row in await cur.fetchall()]

    return {
        "total_pnl": round(total_pnl, 2),
        "win_rate": win_rate,
        "open_positions": open_positions,
        "total_trades": total_trades,
        "trades_today": trades_today,
        "wins": wins,
        "total_closed": total_closed,
        "recent_trades": recent_trades,
        "strategy_stats": strategy_stats,
        "balance_history": balance_history,
        "recent_logs": recent_logs,
    }
