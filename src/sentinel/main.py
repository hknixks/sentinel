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

from sentinel.alerts.alert_engine import AlertEngine
from sentinel.alerts.store import AlertStore
from sentinel.alerts.telegram_client import TelegramClient
from sentinel.binance.symbols import discover_usdt_perpetual_symbols
from sentinel.binance.ws_client import run_symbol_group
from sentinel.config import (
    ALERT_CHECK_INTERVAL_SECONDS,
    ALERT_DB_PATH,
    MAX_SYMBOLS_PER_CONNECTION,
    OUTCOME_DB_PATH,
    OUTCOME_EVALUATION_INTERVAL_SECONDS,
    OUTCOME_MAX_HORIZON_SECONDS,
    STRUCTURE_TOP_N,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_DRY_RUN,
)
from sentinel.logging_setup import setup_logging
from sentinel.market_state import MarketStateStore
from sentinel.outcomes.outcome_tracker import OutcomeTracker
from sentinel.outcomes.store import OutcomeStore
from sentinel.scanner.features import FeatureEngine
from sentinel.scanner.scanner import MarketScanner
from sentinel.setups.setup_engine import SetupEngine
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


async def _run_alerts_periodically(
    store: MarketStateStore,
    alert_engine: AlertEngine,
    outcome_tracker: OutcomeTracker,
    stop_event: asyncio.Event,
) -> None:
    feature_engine = FeatureEngine()
    scanner = MarketScanner()
    structure_engine = StructureEngine()
    setup_engine = SetupEngine()

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=ALERT_CHECK_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break

        snapshot = await store.snapshot()
        # Phase 6 needs each symbol's scanner_activity_score for its own
        # signal snapshot (see outcome_tracker.create_signal below) --
        # ranked once per cycle over the full universe. This ranking is
        # NOT passed into setup_engine.evaluate(): Phase 5's alert-
        # generation behavior (scanner_result=None there) is unchanged.
        ranked_by_symbol = {r.symbol: r for r in scanner.scan(snapshot)}

        for symbol, state in snapshot.items():
            features = feature_engine.compute(state)
            structure_result = structure_engine.analyze(state, features=features)
            scanner_result = ranked_by_symbol.get(symbol)
            evaluation = setup_engine.evaluate(symbol, state, features, None, structure_result)
            best = max(evaluation.candidates, key=lambda c: c.setup_score, default=None)
            record = await alert_engine.process_candidate(symbol, best, structure_result, features)
            if record is not None:
                logger.info(
                    "New alert: %s %s %s score=%.1f",
                    record.symbol,
                    record.direction,
                    record.setup_type,
                    record.setup_score,
                )
                await outcome_tracker.create_signal(record, best, scanner_result, structure_result)


async def _run_outcome_evaluation_periodically(outcome_tracker: OutcomeTracker, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=OUTCOME_EVALUATION_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        updated = await outcome_tracker.evaluate_all_pending()
        if updated:
            logger.info("Outcome tracker: re-evaluated %d signal(s)", updated)


async def _poll_telegram_callbacks(
    telegram: TelegramClient, alert_engine: AlertEngine, stop_event: asyncio.Event
) -> None:
    offset: int | None = None
    while not stop_event.is_set():
        updates = await telegram.get_updates(offset=offset, timeout=25)
        for update in updates:
            offset = update["update_id"] + 1
            callback_query = update.get("callback_query")
            if callback_query:
                await alert_engine.handle_callback(callback_query)


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

    alert_store = AlertStore(ALERT_DB_PATH)
    telegram = TelegramClient(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
    dry_run = TELEGRAM_DRY_RUN or telegram is None or not TELEGRAM_CHAT_ID
    alert_engine = AlertEngine(alert_store, telegram, TELEGRAM_CHAT_ID, market_store=store, dry_run=dry_run)
    logger.info("Alert engine starting (dry_run=%s)", dry_run)

    outcome_store = OutcomeStore(OUTCOME_DB_PATH)
    outcome_tracker = OutcomeTracker(outcome_store, store, max_horizon_seconds=OUTCOME_MAX_HORIZON_SECONDS)

    tasks.append(asyncio.create_task(_run_alerts_periodically(store, alert_engine, outcome_tracker, stop_event)))
    tasks.append(asyncio.create_task(_run_outcome_evaluation_periodically(outcome_tracker, stop_event)))
    if not dry_run and telegram is not None:
        tasks.append(asyncio.create_task(_poll_telegram_callbacks(telegram, alert_engine, stop_event)))

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
