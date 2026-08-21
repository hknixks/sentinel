"""
Persistence for outcome tracking.

Follows the same pattern proven in sentinel.alerts.store.AlertStore --
SQLite (stdlib-only, no new dependency), synchronous calls wrapped in
asyncio.to_thread so they never block the event loop, and idempotency
enforced at the database layer via PRIMARY KEY constraints rather than
application-level check-then-insert races. Not literally shared code
with AlertStore (the schemas and query shapes differ enough that a
forced shared base class would add indirection without real benefit),
but deliberately the same proven approach, not a new one.

Two tables:
  signals          -- one immutable row per alert, plus evolving
                       outcome-tracking columns (see models.py).
  outcome_windows  -- one row per (signal, window_label), holding that
                       window's forward_return/MFE/MAE once its
                       wall-clock duration has elapsed. Upserted (not
                       insert-once) because a window's value is a pure
                       function of the signal + observed candles, so
                       recomputing and overwriting it is always safe and
                       always yields the same result once the window's
                       candles are fixed history.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from sentinel.outcomes.models import SignalRecord, WindowResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    alert_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    setup_score REAL NOT NULL,
    scanner_activity_score REAL,
    structure_pattern TEXT,
    structure_confidence REAL,
    timeframe_alignment TEXT,
    entry_zone_low REAL NOT NULL,
    entry_zone_high REAL NOT NULL,
    reference_entry_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    take_profit_1 REAL NOT NULL,
    take_profit_2 REAL,
    initial_rr_tp1 REAL NOT NULL,
    initial_rr_tp2 REAL,
    signal_timestamp REAL NOT NULL,
    outcome_state TEXT NOT NULL,
    tp1_hit_at REAL,
    tp2_hit_at REAL,
    sl_hit_at REAL,
    ambiguous_at REAL,
    resolved_at REAL,
    last_evaluated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);

CREATE TABLE IF NOT EXISTS outcome_windows (
    signal_id TEXT NOT NULL,
    window_label TEXT NOT NULL,
    forward_return REAL,
    mfe REAL,
    mae REAL,
    evaluated_at REAL NOT NULL,
    PRIMARY KEY (signal_id, window_label)
);
"""

_SIGNAL_COLUMNS = (
    "signal_id", "alert_id", "symbol", "direction", "setup_type", "setup_score",
    "scanner_activity_score", "structure_pattern", "structure_confidence",
    "timeframe_alignment", "entry_zone_low", "entry_zone_high",
    "reference_entry_price", "stop_price", "take_profit_1", "take_profit_2",
    "initial_rr_tp1", "initial_rr_tp2", "signal_timestamp", "outcome_state",
    "tp1_hit_at", "tp2_hit_at", "sl_hit_at", "ambiguous_at", "resolved_at",
    "last_evaluated_at",
)

_WINDOW_COLUMNS = ("signal_id", "window_label", "forward_return", "mfe", "mae", "evaluated_at")


def _row_to_signal(row: sqlite3.Row) -> SignalRecord:
    return SignalRecord(**{col: row[col] for col in _SIGNAL_COLUMNS})


def _row_to_window(row: sqlite3.Row) -> WindowResult:
    return WindowResult(**{col: row[col] for col in _WINDOW_COLUMNS})


class OutcomeStore:
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

    def create_signal_sync(self, signal: SignalRecord) -> bool:
        """Insert a new immutable signal snapshot. Returns False if this
        signal_id already exists (PRIMARY KEY constraint) -- restart-safe
        idempotency without an application-level race."""
        conn = self._connect()
        try:
            values = tuple(getattr(signal, col) for col in _SIGNAL_COLUMNS)
            placeholders = ",".join("?" for _ in _SIGNAL_COLUMNS)
            try:
                conn.execute(f"INSERT INTO signals ({','.join(_SIGNAL_COLUMNS)}) VALUES ({placeholders})", values)
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
        finally:
            conn.close()

    def get_signal_sync(self, signal_id: str) -> SignalRecord | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM signals WHERE signal_id = ?", (signal_id,)).fetchone()
            return _row_to_signal(row) if row else None
        finally:
            conn.close()

    def get_signals_needing_evaluation_sync(self, max_horizon_seconds: float, now: float) -> list[SignalRecord]:
        """A signal needs (re-)evaluation every cycle while its horizon
        hasn't ended yet, and exactly one more time just after the
        horizon ends (to lock in the final state/windows); never again
        after that."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM signals WHERE last_evaluated_at IS NULL "
                "OR last_evaluated_at < MIN(signal_timestamp + ?, ?)",
                (max_horizon_seconds, now),
            ).fetchall()
            return [_row_to_signal(r) for r in rows]
        finally:
            conn.close()

    def get_all_signals_sync(self) -> list[SignalRecord]:
        """Read-only listing of every stored signal, for offline analytics
        (see sentinel.analytics). Never used by the live tracking path."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM signals ORDER BY signal_timestamp").fetchall()
            return [_row_to_signal(r) for r in rows]
        finally:
            conn.close()

    def get_all_windows_sync(self) -> list[WindowResult]:
        """Read-only listing of every stored outcome window, for offline
        analytics (see sentinel.analytics). Never used by the live
        tracking path."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM outcome_windows").fetchall()
            return [_row_to_window(r) for r in rows]
        finally:
            conn.close()

    def get_signals_for_symbol_sync(self, symbol: str) -> list[SignalRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM signals WHERE symbol = ? ORDER BY signal_timestamp", (symbol,)
            ).fetchall()
            return [_row_to_signal(r) for r in rows]
        finally:
            conn.close()

    def update_evaluation_sync(
        self,
        signal_id: str,
        outcome_state: str,
        tp1_hit_at: float | None,
        tp2_hit_at: float | None,
        sl_hit_at: float | None,
        ambiguous_at: float | None,
        resolved_at: float | None,
        last_evaluated_at: float,
    ) -> None:
        """Updates ONLY the outcome-tracking columns -- never touches the
        immutable snapshot columns (entry, stop, TP, score, structure,
        setup_type, ...)."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE signals SET outcome_state=?, tp1_hit_at=?, tp2_hit_at=?, sl_hit_at=?, "
                "ambiguous_at=?, resolved_at=?, last_evaluated_at=? WHERE signal_id=?",
                (outcome_state, tp1_hit_at, tp2_hit_at, sl_hit_at, ambiguous_at, resolved_at, last_evaluated_at, signal_id),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_window_sync(self, window: WindowResult) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO outcome_windows (signal_id, window_label, forward_return, mfe, mae, evaluated_at) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(signal_id, window_label) DO UPDATE SET "
                "forward_return=excluded.forward_return, mfe=excluded.mfe, mae=excluded.mae, "
                "evaluated_at=excluded.evaluated_at",
                (window.signal_id, window.window_label, window.forward_return, window.mfe, window.mae, window.evaluated_at),
            )
            conn.commit()
        finally:
            conn.close()

    def get_windows_sync(self, signal_id: str) -> list[WindowResult]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM outcome_windows WHERE signal_id = ? ORDER BY window_label", (signal_id,)
            ).fetchall()
            return [_row_to_window(r) for r in rows]
        finally:
            conn.close()

    # -- async wrappers, safe to call from the event loop --

    async def create_signal(self, signal: SignalRecord) -> bool:
        return await asyncio.to_thread(self.create_signal_sync, signal)

    async def get_signal(self, signal_id: str) -> SignalRecord | None:
        return await asyncio.to_thread(self.get_signal_sync, signal_id)

    async def get_signals_needing_evaluation(self, max_horizon_seconds: float, now: float) -> list[SignalRecord]:
        return await asyncio.to_thread(self.get_signals_needing_evaluation_sync, max_horizon_seconds, now)

    async def get_signals_for_symbol(self, symbol: str) -> list[SignalRecord]:
        return await asyncio.to_thread(self.get_signals_for_symbol_sync, symbol)

    async def get_all_signals(self) -> list[SignalRecord]:
        return await asyncio.to_thread(self.get_all_signals_sync)

    async def get_all_windows(self) -> list[WindowResult]:
        return await asyncio.to_thread(self.get_all_windows_sync)

    async def update_evaluation(
        self,
        signal_id: str,
        outcome_state: str,
        tp1_hit_at: float | None,
        tp2_hit_at: float | None,
        sl_hit_at: float | None,
        ambiguous_at: float | None,
        resolved_at: float | None,
        last_evaluated_at: float,
    ) -> None:
        await asyncio.to_thread(
            self.update_evaluation_sync,
            signal_id, outcome_state, tp1_hit_at, tp2_hit_at, sl_hit_at, ambiguous_at, resolved_at, last_evaluated_at,
        )

    async def upsert_window(self, window: WindowResult) -> None:
        await asyncio.to_thread(self.upsert_window_sync, window)

    async def get_windows(self, signal_id: str) -> list[WindowResult]:
        return await asyncio.to_thread(self.get_windows_sync, signal_id)
