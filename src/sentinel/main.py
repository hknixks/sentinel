"""
SENTINEL Phase 1 entrypoint.

Discovers every Binance USDT-M perpetual futures symbol, streams live
market data over WebSocket into an in-memory MarketStateStore, and
periodically logs a snapshot so the engine's liveness is observable.

Alert-only market-data engine. No order execution, no wallet connection,
no fund transfers -- and none are planned.
"""

from __future__ import annotations

import asyncio
import logging

from sentinel.binance.symbols import discover_usdt_perpetual_symbols
from sentinel.binance.ws_client import run_symbol_group
from sentinel.config import MAX_SYMBOLS_PER_CONNECTION, STRUCTURE_TOP_N
from sentinel.logging_setup import setup_logging
from sentinel.market_state import MarketStateStore
from sentinel.scanner.scanner import MarketScanner
from sentinel.structure.structure import StructureEngine, analyze_top_markets

logger = logging.getLogger(__name__)


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _log_snapshot_periodically(store: MarketStateStore, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=15)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        snapshot = await store.snapshot()
        populated = [s for s in snapshot.values() if s.price is not None]
        logger.info(
            "State snapshot: %d/%d symbols have live data", len(populated), len(snapshot)
        )
        for state in sorted(populated, key=lambda s: s.symbol)[:5]:
            logger.info(
                "  %s price=%s vol24h=%s candle=%s",
                state.symbol,
                state.price,
                state.volume_24h,
                state.last_candle_1m,
            )


async def _log_scanner_periodically(store: MarketStateStore, stop_event: asyncio.Event) -> None:
    scanner = MarketScanner()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        snapshot = await store.snapshot()
        results = scanner.scan(snapshot, top_n=10)
        if not results:
            logger.info("Scanner: no markets with sufficient history yet")
            continue
        logger.info("Scanner: top %d of %d ranked markets", len(results), len(snapshot))
        for r in results:
            logger.info("  %s score=%.1f %s", r.symbol, r.score, r.direction)


async def _log_structure_periodically(store: MarketStateStore, stop_event: asyncio.Event) -> None:
    scanner = MarketScanner()
    engine = StructureEngine()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        snapshot = await store.snapshot()
        ranked = scanner.scan(snapshot, top_n=STRUCTURE_TOP_N)
        if not ranked:
            continue
        for scan_result, structure_result in analyze_top_markets(ranked, snapshot, engine):
            logger.info(
                "  %s activity=%.1f structure=%s bias=%s confidence=%.1f alignment=%s",
                scan_result.symbol,
                scan_result.score,
                structure_result.structure.pattern,
                structure_result.directional_bias,
                structure_result.confidence,
                structure_result.timeframe_alignment,
            )


async def run() -> None:
    setup_logging()
    logger.info("SENTINEL Phase 1 starting: real-time market-data engine")

    symbols = await discover_usdt_perpetual_symbols()
    if not symbols:
        logger.error("No USDT-perpetual symbols discovered; exiting")
        return

    store = MarketStateStore()
    await store.init_symbols(symbols)

    max_per_conn = int(MAX_SYMBOLS_PER_CONNECTION)
    groups = _chunk(symbols, max_per_conn)
    logger.info("Split %d symbols into %d WebSocket connection(s)", len(symbols), len(groups))

    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(run_symbol_group(group, store, stop_event)) for group in groups
    ]
    tasks.append(asyncio.create_task(_log_snapshot_periodically(store, stop_event)))
    tasks.append(asyncio.create_task(_log_scanner_periodically(store, stop_event)))
    tasks.append(asyncio.create_task(_log_structure_periodically(store, stop_event)))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        stop_event.set()
        raise


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")


if __name__ == "__main__":
    main()
