"""
Typed structures for Phase 7 historical signal analytics.

Everything here describes what ALREADY HAPPENED to past alerts, as
recorded by Phase 6's OutcomeStore. None of it is a prediction, a
probability of future success, or a trading recommendation -- see each
field's docstring/label for the exact honest framing.

Every aggregation carries its own sample_size (and a low_confidence flag
below a fixed threshold) so a tiny sample can never be mistaken for a
reliable statistic -- see sentinel.analytics.aggregations.LOW_CONFIDENCE_SAMPLE_SIZE.
Fields are None (never 0, never fabricated) whenever the underlying
observations don't exist.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricStats:
    """Deterministic summary of a numeric sample. count=0 => every other
    field is None, never 0 -- "no data" and "value of zero" must never be
    confused."""

    count: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    positive_pct: float | None
    negative_pct: float | None


@dataclass(frozen=True)
class OutcomeRates:
    """TP1/TP2 rates are "did the signal ever reach this level" (not
    mutually exclusive with each other: reaching TP2 implies TP1 was also
    reached). SL/AMBIGUOUS/EXPIRED/PENDING rates are the terminal
    outcome_state distribution (mutually exclusive, sums to 100% with
    TP1-only-then-nothing-further folded into whichever terminal state
    actually applies)."""

    sample_size: int
    low_confidence: bool
    tp1_hit_rate: float | None
    tp2_hit_rate: float | None
    sl_hit_rate: float | None
    ambiguous_rate: float | None
    expired_rate: float | None
    pending_rate: float | None


@dataclass(frozen=True)
class GroupPerformance:
    """Historical performance for one group of signals (a setup type, a
    direction, a score bucket, ...). MFE/MAE/forward-return figures are
    pooled across every outcome_windows observation available for the
    group's signals (mixing window durations 1m..4h) -- for a
    window-separated breakdown see AnalyticsReport.forward_returns_by_window
    and .mfe_mae_by_direction_window."""

    group: str
    sample_size: int
    low_confidence: bool
    outcomes: OutcomeRates
    avg_mfe: float | None
    median_mfe: float | None
    avg_mae: float | None
    median_mae: float | None
    avg_forward_return: float | None
    median_forward_return: float | None
    median_time_to_resolution_seconds: float | None


@dataclass(frozen=True)
class ForwardReturnWindowStats:
    """Forward-return distribution for one window label (1m/5m/.../4h),
    built only from windows that were actually evaluated -- a signal too
    young for a given window simply contributes nothing to that window's
    stats, it is never assumed to be 0 or interpolated."""

    window_label: str
    sample_size: int
    low_confidence: bool
    avg_return: float | None
    median_return: float | None
    positive_pct: float | None
    negative_pct: float | None


@dataclass(frozen=True)
class MfeMaeStats:
    direction: str
    window_label: str
    mfe: MetricStats
    mae: MetricStats


@dataclass(frozen=True)
class TimeToOutcomeStats:
    sample_size_tp1: int
    median_seconds_to_tp1: float | None
    sample_size_tp2: int
    median_seconds_to_tp2: float | None
    sample_size_sl: int
    median_seconds_to_sl: float | None
    sample_size_resolved: int
    median_seconds_to_resolution: float | None


@dataclass(frozen=True)
class HistoricalRExpectancy:
    """Historical realized R-multiple performance for completed,
    unambiguous outcomes only (TP1_HIT/TP2_HIT/SL_HIT -- never AMBIGUOUS,
    which by definition has no determinable single outcome, and never
    EXPIRED, which never reached any defined exit level). This is NOT a
    probability and NOT a prediction of future performance -- it is a
    backward-looking summary of what actually happened to past signals."""

    sample_size: int
    low_confidence: bool
    avg_r: float | None
    median_r: float | None
    win_rate: float | None
    label: str


@dataclass(frozen=True)
class MomentumVolumeVolatilityNote:
    """Requirement 8 (momentum/relative-volume/volume-acceleration/
    volatility-expansion vs. outcome) cannot currently be computed: Phase
    6's SignalRecord snapshot never captured those raw feature values at
    signal-creation time, and Phase 7 is prohibited from modifying
    SignalRecord/OutcomeStore to add them. Reported honestly as a data
    gap rather than fabricated or silently omitted."""

    available: bool
    reason: str


@dataclass(frozen=True)
class AnalyticsReport:
    generated_at: float
    execution_seconds: float

    total_signals: int
    completed_signals: int
    pending_signals: int
    outcome_distribution: OutcomeRates

    by_setup_type: dict[str, GroupPerformance]
    by_direction: dict[str, GroupPerformance]
    by_setup_score_bucket: dict[str, GroupPerformance]
    by_scanner_score_bucket: dict[str, GroupPerformance]
    by_structure_confidence_bucket: dict[str, GroupPerformance]
    by_rr_bucket: dict[str, GroupPerformance]
    by_timeframe_alignment: dict[str, GroupPerformance]

    forward_returns_by_window: dict[str, ForwardReturnWindowStats]
    mfe_mae_by_direction_window: tuple[MfeMaeStats, ...]
    time_to_outcome: TimeToOutcomeStats
    historical_r: HistoricalRExpectancy
    momentum_volume_volatility: MomentumVolumeVolatilityNote
