"""
Analytics Engine: builds a full AnalyticsReport from Phase 6's stored
outcome data.

    OutcomeStore -> Analytics Engine -> Aggregations -> AnalyticsReport

Reads ONLY through OutcomeStore's own read methods (get_all_signals_sync /
get_all_windows_sync) -- never opens a second connection to the database
file and never duplicates its contents into a separate store. Purely
observational: nothing here writes to OutcomeStore, and nothing here is
imported by the live scanner/structure/setup/alert/outcome-tracking path.
Deterministic: the same OutcomeStore contents always produce the same
AnalyticsReport.
"""

from __future__ import annotations

import dataclasses
import time

from sentinel.outcomes.models import STATE_PENDING, SignalRecord, WindowResult
from sentinel.outcomes.store import OutcomeStore

from sentinel.analytics import aggregations as agg
from sentinel.analytics.models import AnalyticsReport, MomentumVolumeVolatilityNote

MOMENTUM_VOLUME_VOLATILITY_UNAVAILABLE = MomentumVolumeVolatilityNote(
    available=False,
    reason=(
        "SignalRecord (OutcomeStore) does not persist momentum, relative-volume, "
        "volume-acceleration, or volatility-expansion values -- Phase 6 never "
        "captured them at signal-creation time, and Phase 7 must not modify "
        "SignalRecord or OutcomeStore's schema to add them (that would be "
        "modifying outcome tracking, which is out of scope). Reported honestly "
        "as a data gap rather than fabricated."
    ),
)


def generate_report(signals: list[SignalRecord], windows: list[WindowResult]) -> AnalyticsReport:
    """Pure aggregation over already-loaded data -- see
    generate_report_from_store for the OutcomeStore-backed entry point."""
    start = time.perf_counter()

    completed = [s for s in signals if s.outcome_state != STATE_PENDING]
    pending = [s for s in signals if s.outcome_state == STATE_PENDING]

    report = AnalyticsReport(
        generated_at=time.time(),
        execution_seconds=0.0,
        total_signals=len(signals),
        completed_signals=len(completed),
        pending_signals=len(pending),
        outcome_distribution=agg.outcome_rates(signals),
        by_setup_type=agg.group_by_setup_type(signals, windows),
        by_direction=agg.group_by_direction(signals, windows),
        by_setup_score_bucket=agg.bucket_by(signals, windows, lambda s: s.setup_score, agg.SCORE_BUCKETS),
        by_scanner_score_bucket=agg.bucket_by(signals, windows, lambda s: s.scanner_activity_score, agg.SCORE_BUCKETS),
        by_structure_confidence_bucket=agg.bucket_by(signals, windows, lambda s: s.structure_confidence, agg.SCORE_BUCKETS),
        by_rr_bucket=agg.bucket_by(signals, windows, lambda s: s.initial_rr_tp1, agg.RR_BUCKETS),
        by_timeframe_alignment=agg.group_by_timeframe_alignment(signals, windows),
        forward_returns_by_window=agg.forward_return_by_window(signals, windows),
        mfe_mae_by_direction_window=agg.mfe_mae_by_direction_and_window(signals, windows),
        time_to_outcome=agg.time_to_outcome(signals),
        historical_r=agg.historical_r_expectancy(signals),
        momentum_volume_volatility=MOMENTUM_VOLUME_VOLATILITY_UNAVAILABLE,
    )

    execution_seconds = time.perf_counter() - start
    return dataclasses.replace(report, execution_seconds=execution_seconds)


def generate_report_from_store(store: OutcomeStore) -> AnalyticsReport:
    """Read-only entry point: loads every signal/window through
    OutcomeStore's own accessors and runs the pure aggregation pipeline.
    Never bypasses OutcomeStore with raw SQL, never writes anything back."""
    signals = store.get_all_signals_sync()
    windows = store.get_all_windows_sync()
    return generate_report(signals, windows)
