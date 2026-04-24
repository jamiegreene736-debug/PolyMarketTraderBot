import aiosqlite
from datetime import datetime
from loguru import logger

DB_PATH = "bot_data.db"


async def _column_exists(db, table: str, column: str) -> bool:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        rows = await cur.fetchall()
    return any(row[1] == column for row in rows)


async def _ensure_column(db, table: str, column: str, ddl: str):
    if not await _column_exists(db, table, column):
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


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
        await _ensure_column(db, "trades", "execution_side", "TEXT")
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
            CREATE TABLE IF NOT EXISTS ai_observer_reports (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                category       TEXT NOT NULL,
                severity       TEXT NOT NULL,
                title          TEXT NOT NULL,
                summary        TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                recommended_action TEXT NOT NULL DEFAULT 'review',
                acknowledged   INTEGER NOT NULL DEFAULT 0,
                acknowledged_at TEXT,
                fingerprint    TEXT NOT NULL UNIQUE
            )
        """)
        await _ensure_column(db, "ai_observer_reports", "recommended_action", "TEXT NOT NULL DEFAULT 'review'")
        await _ensure_column(db, "ai_observer_reports", "acknowledged", "INTEGER NOT NULL DEFAULT 0")
        await _ensure_column(db, "ai_observer_reports", "acknowledged_at", "TEXT")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bot_control (
                id             INTEGER PRIMARY KEY CHECK (id = 1),
                status         TEXT DEFAULT 'stopped',
                last_heartbeat TEXT,
                last_error     TEXT,
                updated_at     TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS auto_close_overrides (
                condition_id TEXT NOT NULL,
                outcome      TEXT NOT NULL,
                active       INTEGER NOT NULL DEFAULT 1,
                updated_at   TEXT NOT NULL,
                PRIMARY KEY (condition_id, outcome)
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
                       side: str, price: float, quantity: float, order_id: str,
                       execution_side: str | None = None):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO trades (
                timestamp, strategy, market_slug, question, side, execution_side,
                price, quantity, order_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), strategy, market_slug, question,
              side, execution_side, price, quantity, order_id))
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


async def get_recent_logs(
    limit: int = 100,
    exclude_prefix: str | None = None,
    since_timestamp: str | None = None,
) -> list[dict]:
    query = "SELECT timestamp, level, message FROM bot_logs"
    clauses = []
    params: list = []
    if exclude_prefix:
        clauses.append("message NOT LIKE ?")
        params.append(f"{exclude_prefix}%")
    if since_timestamp:
        clauses.append("timestamp >= ?")
        params.append(since_timestamp)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, tuple(params)) as cur:
            rows = [dict(row) for row in await cur.fetchall()]
    rows.reverse()
    return rows


async def insert_ai_observer_report(
    category: str,
    severity: str,
    title: str,
    summary: str,
    recommendation: str,
    recommended_action: str,
    fingerprint: str,
 ) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT OR IGNORE INTO ai_observer_reports
                (
                    timestamp, category, severity, title, summary,
                    recommendation, recommended_action, fingerprint
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                category,
                severity,
                title,
                summary,
                recommendation,
                recommended_action,
                fingerprint,
            ),
        )
        await db.commit()
    return cur.rowcount > 0


async def get_recent_ai_observer_reports(limit: int = 6) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT
                timestamp, category, severity, title, summary, recommendation,
                recommended_action, acknowledged, acknowledged_at
            FROM ai_observer_reports
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ) as cur:
            return [dict(row) for row in await cur.fetchall()]


async def acknowledge_ai_observer_alerts() -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            UPDATE ai_observer_reports
            SET acknowledged = 1, acknowledged_at = ?
            WHERE acknowledged = 0
            """,
            (datetime.utcnow().isoformat(),),
        )
        await db.commit()
    return cur.rowcount


async def clear_ai_observer_reports() -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("DELETE FROM ai_observer_reports")
        await db.commit()
    return cur.rowcount


async def get_ai_observer_alert_state() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT
                timestamp, category, severity, title, summary, recommendation,
                recommended_action
            FROM ai_observer_reports
            WHERE acknowledged = 0
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 3
                    WHEN 'warning' THEN 2
                    ELSE 1
                END DESC,
                timestamp DESC
            LIMIT 1
            """
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {"active": False}

    data = dict(row)
    data["active"] = True
    data["pause_recommended"] = data.get("recommended_action") == "pause"
    return data


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
            "SELECT order_id, strategy, question, market_slug, side, execution_side, price, quantity, timestamp "
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
            "SELECT order_id, strategy, question, market_slug, side, execution_side, price, quantity, timestamp "
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


async def get_auto_close_overrides() -> set[tuple[str, str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT condition_id, outcome FROM auto_close_overrides WHERE active = 1"
        ) as cur:
            rows = await cur.fetchall()
    return {
        (str(row["condition_id"]).strip(), str(row["outcome"]).upper().strip())
        for row in rows
    }


async def set_auto_close_override(condition_id: str, outcome: str, active: bool):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO auto_close_overrides (condition_id, outcome, active, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(condition_id, outcome)
            DO UPDATE SET active = excluded.active, updated_at = excluded.updated_at
            """,
            (
                condition_id.strip(),
                outcome.upper().strip(),
                1 if active else 0,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()


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
            recent_logs.reverse()

        # Recent AI observer reports
        async with db.execute("""
            SELECT
                timestamp, category, severity, title, summary, recommendation,
                recommended_action, acknowledged, acknowledged_at
            FROM ai_observer_reports
            ORDER BY timestamp DESC
            LIMIT 6
        """) as cur:
            ai_reports = [dict(row) for row in await cur.fetchall()]

        async with db.execute(
            """
            SELECT
                timestamp, category, severity, title, summary, recommendation,
                recommended_action
            FROM ai_observer_reports
            WHERE acknowledged = 0
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 3
                    WHEN 'warning' THEN 2
                    ELSE 1
                END DESC,
                timestamp DESC
            LIMIT 1
            """
        ) as cur:
            ai_alert = dict(row) if (row := await cur.fetchone()) else {"active": False}
            if ai_alert.get("timestamp"):
                ai_alert["active"] = True
                ai_alert["pause_recommended"] = ai_alert.get("recommended_action") == "pause"

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
        "ai_reports": ai_reports,
        "ai_alert": ai_alert,
    }
