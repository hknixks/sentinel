"""
Persistence for alert lifecycle state.

SQLite was chosen as the simplest appropriate mechanism for the current
single-process VPS deployment: it's stdlib-only (no new dependency),
transactional, and durable across process restarts -- exactly what
restart-safe idempotency requires. It is isolated behind AlertStore's
method interface (get_active / create_active / resolve / record_decision
/ ...) so a future PostgreSQL or Redis backend can implement the same
interface without touching the alert engine or Telegram layers.

All writes are synchronous SQLite calls run off the event loop via
asyncio.to_thread, so they never block the WebSocket/market-data loop.

Idempotency is enforced at the database layer, not just in application
logic: `alert_id` is the primary key, so a duplicate INSERT (e.g. from a
race or a restart re-processing the same scan) is rejected by the
UNIQUE constraint and treated as a no-op rather than raising or
duplicating a row. `record_decision` only ever accepts the FIRST
decision for a given alert_id; repeated or conflicting button clicks are
no-ops after that.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

from sentinel.alerts.models import AlertRecord, STATUS_ACTIVE, STATUS_RESOLVED

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    status TEXT NOT NULL,
    setup_score REAL NOT NULL,
    entry_low REAL NOT NULL,
    entry_high REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit_1 REAL NOT NULL,
    take_profit_2 REAL,
    risk_reward_tp1 REAL NOT NULL,
    invalidation_level REAL NOT NULL,
    structure_context TEXT NOT NULL,
    confirmation_factors TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    created_at REAL NOT NULL,
    resolved_at REAL,
    telegram_message_id INTEGER,
    telegram_chat_id TEXT,
    decision TEXT,
    decision_at REAL,
    decision_reference_price REAL
);
CREATE INDEX IF NOT EXISTS idx_alerts_symbol_status ON alerts(symbol, status);
"""

_COLUMNS = (
    "alert_id", "symbol", "direction", "setup_type", "status", "setup_score",
    "entry_low", "entry_high", "stop_loss", "take_profit_1", "take_profit_2",
    "risk_reward_tp1", "invalidation_level", "structure_context",
    "confirmation_factors", "reasoning", "created_at", "resolved_at",
    "telegram_message_id", "telegram_chat_id", "decision", "decision_at",
    "decision_reference_price",
)


def _row_to_record(row: sqlite3.Row) -> AlertRecord:
    return AlertRecord(**{col: row[col] for col in _COLUMNS})


class AlertStore:
    def __init__(self, db_path: str) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- synchronous implementations --

    def get_active_sync(self, symbol: str) -> AlertRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM alerts WHERE symbol = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
                (symbol, STATUS_ACTIVE),
            ).fetchone()
            return _row_to_record(row) if row else None
        finally:
            conn.close()

    def get_by_id_sync(self, alert_id: str) -> AlertRecord | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
            return _row_to_record(row) if row else None
        finally:
            conn.close()

    def create_active_sync(self, record: AlertRecord) -> bool:
        """Insert a new ACTIVE alert. Returns False if this exact
        alert_id already exists (relies on the PRIMARY KEY constraint for
        true atomicity, not an application-level check-then-insert)."""
        conn = self._connect()
        try:
            values = tuple(getattr(record, col) for col in _COLUMNS)
            placeholders = ",".join("?" for _ in _COLUMNS)
            try:
                conn.execute(f"INSERT INTO alerts ({','.join(_COLUMNS)}) VALUES ({placeholders})", values)
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
        finally:
            conn.close()

    def resolve_sync(self, alert_id: str, resolved_at: float | None = None) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE alerts SET status = ?, resolved_at = ? WHERE alert_id = ? AND status = ?",
                (STATUS_RESOLVED, resolved_at if resolved_at is not None else time.time(), alert_id, STATUS_ACTIVE),
            )
            conn.commit()
        finally:
            conn.close()

    def set_telegram_message_id_sync(self, alert_id: str, chat_id: str, message_id: int) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE alerts SET telegram_message_id = ?, telegram_chat_id = ? WHERE alert_id = ?",
                (message_id, chat_id, alert_id),
            )
            conn.commit()
        finally:
            conn.close()

    def record_decision_sync(
        self, alert_id: str, decision: str, reference_price: float | None, decided_at: float | None = None
    ) -> bool:
        """Idempotent: only the FIRST decision for a given alert_id is
        ever recorded. A repeated click (same or opposite button) after
        that is a no-op and returns False."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE alerts SET decision = ?, decision_at = ?, decision_reference_price = ? "
                "WHERE alert_id = ? AND decision IS NULL",
                (decision, decided_at if decided_at is not None else time.time(), reference_price, alert_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # -- async wrappers, safe to call from the event loop --

    async def get_active(self, symbol: str) -> AlertRecord | None:
        return await asyncio.to_thread(self.get_active_sync, symbol)

    async def get_by_id(self, alert_id: str) -> AlertRecord | None:
        return await asyncio.to_thread(self.get_by_id_sync, alert_id)

    async def create_active(self, record: AlertRecord) -> bool:
        return await asyncio.to_thread(self.create_active_sync, record)

    async def resolve(self, alert_id: str, resolved_at: float | None = None) -> None:
        await asyncio.to_thread(self.resolve_sync, alert_id, resolved_at)

    async def set_telegram_message_id(self, alert_id: str, chat_id: str, message_id: int) -> None:
        await asyncio.to_thread(self.set_telegram_message_id_sync, alert_id, chat_id, message_id)

    async def record_decision(
        self, alert_id: str, decision: str, reference_price: float | None, decided_at: float | None = None
    ) -> bool:
        return await asyncio.to_thread(self.record_decision_sync, alert_id, decision, reference_price, decided_at)
