"""
Core in-memory market state model.

This module is pure logic with no network I/O, so it can be unit tested
directly. It represents the live state of every discovered Binance USDT-M
perpetual futures market.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace

from sentinel.config import MAX_CANDLE_HISTORY_MINUTES


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


@dataclass(frozen=True)
class SymbolState:
    symbol: str
    price: float | None = None
    volume_24h: float | None = None
    last_update_ts: float | None = None
    last_candle_1m: Candle | None = None
    # Closed 1m candles only, oldest first, bounded to MAX_CANDLE_HISTORY_MINUTES.
    candle_history: tuple[Candle, ...] = field(default_factory=tuple)


class MarketStateStore:
    """Thread-safe (within one event loop) store of SymbolState per symbol."""

    def __init__(self) -> None:
        self._states: dict[str, SymbolState] = {}
        self._lock = asyncio.Lock()

    async def init_symbols(self, symbols: list[str]) -> None:
        async with self._lock:
            for symbol in symbols:
                self._states.setdefault(symbol, SymbolState(symbol=symbol))

    async def update_ticker(
        self,
        symbol: str,
        price: float,
        volume_24h: float,
        event_ts: float | None = None,
    ) -> None:
        async with self._lock:
            current = self._states.get(symbol, SymbolState(symbol=symbol))
            self._states[symbol] = replace(
                current,
                price=price,
                volume_24h=volume_24h,
                last_update_ts=event_ts if event_ts is not None else time.time(),
            )

    async def update_kline(
        self,
        symbol: str,
        candle: Candle,
        event_ts: float | None = None,
    ) -> None:
        async with self._lock:
            current = self._states.get(symbol, SymbolState(symbol=symbol))
            history = current.candle_history
            if candle.is_closed and (not history or history[-1].open_time != candle.open_time):
                history = (history + (candle,))[-MAX_CANDLE_HISTORY_MINUTES:]
            self._states[symbol] = replace(
                current,
                last_candle_1m=candle,
                candle_history=history,
                last_update_ts=event_ts if event_ts is not None else time.time(),
            )

    async def get(self, symbol: str) -> SymbolState | None:
        async with self._lock:
            return self._states.get(symbol)

    async def snapshot(self) -> dict[str, SymbolState]:
        async with self._lock:
            return dict(self._states)

    async def symbol_count(self) -> int:
        async with self._lock:
            return len(self._states)
