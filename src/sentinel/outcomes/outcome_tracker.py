"""
Outcome Tracker: records the forward market behavior of every newly
generated alert. Purely observational -- it never influences whether an
alert fires, never modifies AlertEngine/SetupEngine/StructureEngine
state, and reads only the 1m candle history MarketState already
maintains (no extra REST calls per signal).

LIMITATION (documented per Phase 6 instructions): MarketStateStore's
candle history lives only in memory. If the process restarts while a
signal is still being evaluated, candles from BEFORE the restart are
gone -- Phase 1 does not persist raw candles to disk, and Phase 6
deliberately does not add REST backfill to close that gap (out of
scope; "do not fabricate missing history"). Any window whose horizon
falls in that pre-restart gap will simply remain unpopulated (None)
rather than being backfilled or guessed.
"""

from __future__ import annotations

import logging
import time

from sentinel.alerts.models import AlertRecord
from sentinel.market_state import MarketStateStore
from sentinel.outcomes.models import STATE_EXPIRED, STATE_PENDING, WINDOW_LABELS, SignalRecord, WindowResult
from sentinel.outcomes.store import OutcomeStore
from sentinel.outcomes.windows import candles_since, compute_tp_sl_outcome, compute_window_metrics
from sentinel.scanner.scanner import ScannerResult
from sentinel.setups.models import SetupCandidate
from sentinel.structure.models import StructureResult

logger = logging.getLogger(__name__)


class OutcomeTracker:
    def __init__(
        self,
        store: OutcomeStore,
        market_store: MarketStateStore,
        max_horizon_seconds: float = 14400.0,
    ) -> None:
        self._store = store
        self._market_store = market_store
        self._max_horizon_seconds = max_horizon_seconds

    async def create_signal(
        self,
        alert_record: AlertRecord,
        candidate: SetupCandidate,
        scanner_result: ScannerResult | None,
        structure_result: StructureResult | None,
    ) -> SignalRecord | None:
        """Call once, immediately after AlertEngine.process_candidate
        returns a genuinely NEW alert record (never for a suppressed
        duplicate -- AlertEngine's own dedup is the primary guarantee
        there). Returns None if this signal_id already exists (belt-
        and-suspenders DB-level idempotency for restart safety)."""
        reference_entry = (candidate.entry_zone.low + candidate.entry_zone.high) / 2.0

        signal = SignalRecord(
            signal_id=alert_record.alert_id,
            alert_id=alert_record.alert_id,
            symbol=candidate.symbol,
            direction=candidate.direction,
            setup_type=candidate.setup_type,
            setup_score=candidate.setup_score,
            scanner_activity_score=scanner_result.score if scanner_result else None,
            structure_pattern=structure_result.structure.pattern if structure_result else None,
            structure_confidence=structure_result.confidence if structure_result else None,
            timeframe_alignment=structure_result.timeframe_alignment if structure_result else None,
            entry_zone_low=candidate.entry_zone.low,
            entry_zone_high=candidate.entry_zone.high,
            reference_entry_price=reference_entry,
            stop_price=candidate.risk.stop_loss,
            take_profit_1=candidate.risk.take_profit_1,
            take_profit_2=candidate.risk.take_profit_2,
            initial_rr_tp1=candidate.risk.risk_reward_tp1,
            initial_rr_tp2=candidate.risk.risk_reward_tp2,
            signal_timestamp=alert_record.created_at,
            outcome_state=STATE_PENDING,
        )

        created = await self._store.create_signal(signal)
        if not created:
            return None
        logger.info("Outcome tracking started: %s %s %s", signal.symbol, signal.direction, signal.setup_type)
        return signal

    async def evaluate_all_pending(self) -> int:
        """Called periodically. Re-evaluates every signal that still
        needs it against current candle history. Returns the count
        actually (re-)evaluated this pass."""
        now = time.time()
        candidates = await self._store.get_signals_needing_evaluation(self._max_horizon_seconds, now)
        updated = 0
        for signal in candidates:
            if await self._evaluate_one(signal, now):
                updated += 1
        return updated

    async def _evaluate_one(self, signal: SignalRecord, now: float) -> bool:
        # Even with zero candle history (e.g. a post-restart gap, see the
        # module docstring), we still run the evaluation with an empty
        # candle list: windows correctly stay None (nothing fabricated),
        # and a signal whose horizon has passed still correctly reaches
        # EXPIRED rather than remaining PENDING forever with no path to
        # resolution.
        state = await self._market_store.get(signal.symbol)
        candles = candles_since(state.candle_history, signal.signal_timestamp) if state else []

        for label in WINDOW_LABELS:
            forward_return, mfe, mae = compute_window_metrics(
                signal.direction, signal.reference_entry_price, signal.signal_timestamp, label, candles, now
            )
            if forward_return is None:
                continue
            await self._store.upsert_window(WindowResult(signal.signal_id, label, forward_return, mfe, mae, now))

        outcome = compute_tp_sl_outcome(signal.direction, signal.stop_price, signal.take_profit_1, signal.take_profit_2, candles)
        final_state = outcome["state"]
        resolved_at = None
        horizon_end = signal.signal_timestamp + self._max_horizon_seconds

        if final_state == STATE_PENDING and now >= horizon_end:
            # Only overwrite PENDING with EXPIRED -- a signal that already
            # reached TP1_HIT (etc.) keeps that real, observed outcome
            # rather than being relabeled EXPIRED just because the clock
            # ran out with nothing FURTHER happening.
            final_state = STATE_EXPIRED
            resolved_at = now
        elif final_state != STATE_PENDING:
            resolved_at = outcome["sl_hit_at"] or outcome["tp2_hit_at"] or outcome["tp1_hit_at"] or outcome["ambiguous_at"]

        await self._store.update_evaluation(
            signal.signal_id,
            final_state,
            outcome["tp1_hit_at"],
            outcome["tp2_hit_at"],
            outcome["sl_hit_at"],
            outcome["ambiguous_at"],
            resolved_at,
            now,
        )
        return True
