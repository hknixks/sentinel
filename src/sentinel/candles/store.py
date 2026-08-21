"""
Persistence for 1m candle history (Phase 8).

Closes the documented Phase 1/6 limitation: MarketStateStore.candle_history
lives only in memory, so a process restart loses every accumulated candle,
which can cause outcome-window gaps and forces structure/scanner history
to rebuild from zero. This module persists every real CLOSED 1m candle as
it arrives so it can be restored into MarketStateStore on the next
startup -- see sentinel.market_state.MarketStateStore.restore_candle_history
for the restore side, and sentinel.binance.ws_client for where live
candles get persisted as they close.

Follows the same proven pattern as sentinel.alerts.store.AlertStore and
sentinel.outcomes.store.OutcomeStore: SQLite (stdlib-only, no new
dependency), synchronous calls wrapped in asyncio.to_thread so they never
block the event loop, idempotency enforced at the database layer via a
(symbol, open_time) PRIMARY KEY rather than an application-level
check-then-insert race.

Pure I/O only. Validation (is_valid_closed_candle) is a deterministic,
side-effect-free predicate kept separate from the I/O methods so it is
independently testable and reusable on both the write path (refuse to
persist garbage) and the read path (refuse to restore garbage that
somehow ended up on disk).
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
from pathlib import Path

from sentinel.market_state import Candle

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, open_time)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_open_time ON candles(symbol, open_time);
"""


def _is_finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def is_valid_closed_candle(candle: Candle) -> bool:
    """Only real, sane closed candles are ever persisted or restored --
    never NaN/Inf, never negative, never high < low, never a non-positive
    timestamp. A candle failing this is logged and dropped -- never
    "corrected" or fabricated into something valid."""
    values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
    if not all(_is_finite(v) for v in values):
        return False
    if any(v < 0 for v in values):
        return False
    if candle.open_time <= 0:
        return False
    if candle.high < candle.low:
        return False
    return True


def _row_to_candle(row: sqlite3.Row) -> Candle:
    return Candle(
        open_time=row["open_time"],
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        volume=row["volume"],
        is_closed=True,
    )


class CandleStore:
    def __init__(self, db_path: str, max_history_minutes: int) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._max_history_minutes = max_history_minutes
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

    def append_candle_sync(self, symbol: str, candle: Candle) -> bool:
        """Idempotently persists one closed candle: inserting the same
        (symbol, open_time) twice is a safe no-op (PRIMARY KEY), never a
        duplicate row and never an error -- restart-safe by construction.
        Returns False (and persists nothing) if the candle fails
        validation. Also trims this symbol's history down to
        max_history_minutes rows on every write, so persisted state never
        grows past the same cap MarketStateStore enforces in memory."""
        if not is_valid_closed_candle(candle):
            logger.warning(
                "Refusing to persist invalid candle for %s at open_time=%s", symbol, candle.open_time
            )
            return False
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO candles (symbol, open_time, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?)",
                (symbol, candle.open_time, candle.open, candle.high, candle.low, candle.close, candle.volume),
            )
            conn.execute(
                "DELETE FROM candles WHERE symbol=? AND open_time NOT IN ("
                "SELECT open_time FROM candles WHERE symbol=? ORDER BY open_time DESC LIMIT ?)",
                (symbol, symbol, self._max_history_minutes),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def load_history_sync(self, symbol: str) -> list[Candle]:
        """Oldest-first, capped at max_history_minutes, with any
        corrupted row (shouldn't exist given append-time validation, but
        checked defensively) skipped rather than restored."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM candles WHERE symbol=? ORDER BY open_time DESC LIMIT ?",
                (symbol, self._max_history_minutes),
            ).fetchall()
            candles = [_row_to_candle(r) for r in reversed(rows)]
            return [c for c in candles if is_valid_closed_candle(c)]
        finally:
            conn.close()

    def load_all_sync(self) -> dict[str, list[Candle]]:
        """One query for every symbol's persisted history -- used once at
        startup so restoring the full universe doesn't cost one DB round
        trip per symbol. Each symbol's list is oldest-first; a symbol with
        no persisted candles simply has no key (never a fabricated empty
        placeholder list)."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM candles ORDER BY symbol, open_time").fetchall()
            by_symbol: dict[str, list[Candle]] = {}
            for row in rows:
                candle = _row_to_candle(row)
                if not is_valid_closed_candle(candle):
                    logger.warning(
                        "Skipping corrupted stored candle for %s at open_time=%s", row["symbol"], row["open_time"]
                    )
                    continue
                by_symbol.setdefault(row["symbol"], []).append(candle)
            return by_symbol
        finally:
            conn.close()

    def count_sync(self, symbol: str | None = None) -> int:
        conn = self._connect()
        try:
            if symbol is not None:
                row = conn.execute("SELECT COUNT(*) AS n FROM candles WHERE symbol=?", (symbol,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS n FROM candles").fetchone()
            return row["n"]
        finally:
            conn.close()

    # -- async wrappers, safe to call from the event loop --

    async def append_candle(self, symbol: str, candle: Candle) -> bool:
        return await asyncio.to_thread(self.append_candle_sync, symbol, candle)

    async def load_history(self, symbol: str) -> list[Candle]:
        return await asyncio.to_thread(self.load_history_sync, symbol)

    async def load_all(self) -> dict[str, list[Candle]]:
        return await asyncio.to_thread(self.load_all_sync)

    async def count(self, symbol: str | None = None) -> int:
        return await asyncio.to_thread(self.count_sync, symbol)
