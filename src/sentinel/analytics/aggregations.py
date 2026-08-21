"""
Pure, deterministic aggregation functions over Phase 6's stored
SignalRecord/WindowResult data. No I/O here -- every function takes
already-loaded lists and returns typed results, so this module is fully
unit-testable with synthetic fixtures and is trivially safe to call
repeatedly (same input, same output, always).

Nothing here reads live market state, nothing here is called from the
live scanner/structure/setup/alert/outcome-tracking path -- see
sentinel.analytics.analytics for the read-only entry point that loads
data from OutcomeStore.
"""

from __future__ import annotations

import statistics

from sentinel.outcomes.models import (
    STATE_AMBIGUOUS,
    STATE_EXPIRED,
    STATE_PENDING,
    STATE_SL_HIT,
    STATE_TP1_HIT,
    STATE_TP2_HIT,
    WINDOW_LABELS,
    SignalRecord,
    WindowResult,
)

from sentinel.analytics.models import (
    ForwardReturnWindowStats,
    GroupPerformance,
    HistoricalRExpectancy,
    MetricStats,
    MfeMaeStats,
    OutcomeRates,
    TimeToOutcomeStats,
)

# Below this many observations, a group's percentages/statistics are
# flagged low_confidence -- they are still reported in full (never
# hidden or deleted), just clearly labeled as statistically unreliable.
LOW_CONFIDENCE_SAMPLE_SIZE = 10

# Fixed bucket edges as specified for Phase 7. Half-open [low, high),
# except the final bucket of each set which is closed on the top so a
# value of exactly the maximum still lands somewhere.
SCORE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-49", 0.0, 50.0),
    ("50-59", 50.0, 60.0),
    ("60-69", 60.0, 70.0),
    ("70-79", 70.0, 80.0),
    ("80-89", 80.0, 90.0),
    ("90-100", 90.0, 100.0000001),
)

RR_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<1.5", float("-inf"), 1.5),
    ("1.5-1.99", 1.5, 2.0),
    ("2.0-2.99", 2.0, 3.0),
    ("3.0-4.99", 3.0, 5.0),
    ("5.0+", 5.0, float("inf")),
)

# Known setup_type/direction values (see sentinel.setups.models docstrings).
# Always shown even with zero signals, so an absent category is visibly
# "0 samples" rather than silently missing from the report. Any other
# value actually found in the data is appended too, never dropped.
KNOWN_SETUP_TYPES = ("breakout", "pullback", "trend_continuation")
KNOWN_DIRECTIONS = ("long", "short")

NOT_AVAILABLE = "not_available"
OUT_OF_RANGE = "out_of_range"


def _metric_stats(values: list[float]) -> MetricStats:
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return MetricStats(count=0, mean=None, median=None, minimum=None, maximum=None, positive_pct=None, negative_pct=None)
    return MetricStats(
        count=n,
        mean=statistics.fmean(clean),
        median=statistics.median(clean),
        minimum=min(clean),
        maximum=max(clean),
        positive_pct=sum(1 for v in clean if v > 0) / n * 100,
        negative_pct=sum(1 for v in clean if v < 0) / n * 100,
    )


def _index_windows(windows: list[WindowResult]) -> dict[str, list[WindowResult]]:
    by_signal: dict[str, list[WindowResult]] = {}
    for w in windows:
        by_signal.setdefault(w.signal_id, []).append(w)
    return by_signal


def outcome_rates(signals: list[SignalRecord]) -> OutcomeRates:
    n = len(signals)
    if n == 0:
        return OutcomeRates(0, True, None, None, None, None, None, None)

    def state_rate(state: str) -> float:
        return sum(1 for s in signals if s.outcome_state == state) / n * 100

    return OutcomeRates(
        sample_size=n,
        low_confidence=n < LOW_CONFIDENCE_SAMPLE_SIZE,
        tp1_hit_rate=sum(1 for s in signals if s.tp1_hit_at is not None) / n * 100,
        tp2_hit_rate=sum(1 for s in signals if s.tp2_hit_at is not None) / n * 100,
        sl_hit_rate=state_rate(STATE_SL_HIT),
        ambiguous_rate=state_rate(STATE_AMBIGUOUS),
        expired_rate=state_rate(STATE_EXPIRED),
        pending_rate=state_rate(STATE_PENDING),
    )


def _pooled_window_values(
    signals: list[SignalRecord], windows_by_signal: dict[str, list[WindowResult]], field: str
) -> list[float]:
    values: list[float] = []
    for s in signals:
        for w in windows_by_signal.get(s.signal_id, ()):
            v = getattr(w, field)
            if v is not None:
                values.append(v)
    return values


def _group_performance(
    group: str, signals: list[SignalRecord], windows_by_signal: dict[str, list[WindowResult]]
) -> GroupPerformance:
    mfe_stats = _metric_stats(_pooled_window_values(signals, windows_by_signal, "mfe"))
    mae_stats = _metric_stats(_pooled_window_values(signals, windows_by_signal, "mae"))
    fr_stats = _metric_stats(_pooled_window_values(signals, windows_by_signal, "forward_return"))
    resolution_times = [s.resolved_at - s.signal_timestamp for s in signals if s.resolved_at is not None]

    return GroupPerformance(
        group=group,
        sample_size=len(signals),
        low_confidence=len(signals) < LOW_CONFIDENCE_SAMPLE_SIZE,
        outcomes=outcome_rates(signals),
        avg_mfe=mfe_stats.mean,
        median_mfe=mfe_stats.median,
        avg_mae=mae_stats.mean,
        median_mae=mae_stats.median,
        avg_forward_return=fr_stats.mean,
        median_forward_return=fr_stats.median,
        median_time_to_resolution_seconds=statistics.median(resolution_times) if resolution_times else None,
    )


def _group_by_values(
    signals: list[SignalRecord],
    windows: list[WindowResult],
    key_fn,
    known_keys: tuple[str, ...] = (),
) -> dict[str, GroupPerformance]:
    windows_by_signal = _index_windows(windows)
    buckets: dict[str, list[SignalRecord]] = {k: [] for k in known_keys}
    for s in signals:
        key = key_fn(s)
        buckets.setdefault(key, []).append(s)
    return {key: _group_performance(key, sigs, windows_by_signal) for key, sigs in buckets.items()}


def group_by_setup_type(signals: list[SignalRecord], windows: list[WindowResult]) -> dict[str, GroupPerformance]:
    return _group_by_values(signals, windows, lambda s: s.setup_type, KNOWN_SETUP_TYPES)


def group_by_direction(signals: list[SignalRecord], windows: list[WindowResult]) -> dict[str, GroupPerformance]:
    return _group_by_values(signals, windows, lambda s: s.direction, KNOWN_DIRECTIONS)


def group_by_timeframe_alignment(signals: list[SignalRecord], windows: list[WindowResult]) -> dict[str, GroupPerformance]:
    """Only actual labels observed in the stored data (plus a NOT_AVAILABLE
    bucket for signals with no recorded alignment) -- never a pre-declared
    fixed set, per Phase 7 requirement 7."""
    windows_by_signal = _index_windows(windows)
    buckets: dict[str, list[SignalRecord]] = {}
    for s in signals:
        key = s.timeframe_alignment if s.timeframe_alignment is not None else NOT_AVAILABLE
        buckets.setdefault(key, []).append(s)
    return {key: _group_performance(key, sigs, windows_by_signal) for key, sigs in buckets.items()}


def _bucket_label(value: float, buckets: tuple[tuple[str, float, float], ...]) -> str | None:
    for label, low, high in buckets:
        if low <= value < high:
            return label
    return None


def bucket_by(
    signals: list[SignalRecord],
    windows: list[WindowResult],
    value_fn,
    buckets: tuple[tuple[str, float, float], ...],
) -> dict[str, GroupPerformance]:
    """Groups signals into the given fixed numeric buckets. value_fn may
    return None (field not recorded for that signal, e.g. scanner_activity_score
    on an older signal) -- those go into a NOT_AVAILABLE bucket rather than
    being silently dropped."""
    windows_by_signal = _index_windows(windows)
    grouped: dict[str, list[SignalRecord]] = {label: [] for label, _, _ in buckets}
    for s in signals:
        value = value_fn(s)
        if value is None:
            grouped.setdefault(NOT_AVAILABLE, []).append(s)
            continue
        label = _bucket_label(value, buckets)
        grouped.setdefault(label or OUT_OF_RANGE, []).append(s)
    return {label: _group_performance(label, sigs, windows_by_signal) for label, sigs in grouped.items()}


def forward_return_by_window(signals: list[SignalRecord], windows: list[WindowResult]) -> dict[str, ForwardReturnWindowStats]:
    """Global (ungrouped) forward-return distribution per window label.
    Only uses windows that were actually evaluated (WindowResult rows that
    exist) -- a signal too young for a window contributes nothing to it,
    it is never treated as 0 or interpolated."""
    signal_ids = {s.signal_id for s in signals}
    result: dict[str, ForwardReturnWindowStats] = {}
    for label in WINDOW_LABELS:
        values = [
            w.forward_return
            for w in windows
            if w.window_label == label and w.signal_id in signal_ids and w.forward_return is not None
        ]
        stats = _metric_stats(values)
        result[label] = ForwardReturnWindowStats(
            window_label=label,
            sample_size=stats.count,
            low_confidence=stats.count < LOW_CONFIDENCE_SAMPLE_SIZE,
            avg_return=stats.mean,
            median_return=stats.median,
            positive_pct=stats.positive_pct,
            negative_pct=stats.negative_pct,
        )
    return result


def mfe_mae_by_direction_and_window(signals: list[SignalRecord], windows: list[WindowResult]) -> tuple[MfeMaeStats, ...]:
    """MFE/MAE distributions per (direction, window_label), respecting
    LONG vs SHORT as required -- their MFE/MAE are computed with opposite
    sign conventions (see sentinel.outcomes.windows.compute_window_metrics)
    so must never be pooled together."""
    signal_ids_by_direction: dict[str, set[str]] = {direction: set() for direction in KNOWN_DIRECTIONS}
    for s in signals:
        signal_ids_by_direction.setdefault(s.direction, set()).add(s.signal_id)

    results: list[MfeMaeStats] = []
    for direction, ids in signal_ids_by_direction.items():
        for label in WINDOW_LABELS:
            mfe_values = [w.mfe for w in windows if w.window_label == label and w.signal_id in ids and w.mfe is not None]
            mae_values = [w.mae for w in windows if w.window_label == label and w.signal_id in ids and w.mae is not None]
            results.append(
                MfeMaeStats(direction=direction, window_label=label, mfe=_metric_stats(mfe_values), mae=_metric_stats(mae_values))
            )
    return tuple(results)


def time_to_outcome(signals: list[SignalRecord]) -> TimeToOutcomeStats:
    def deltas(hit_field: str) -> list[float]:
        return [getattr(s, hit_field) - s.signal_timestamp for s in signals if getattr(s, hit_field) is not None]

    tp1 = deltas("tp1_hit_at")
    tp2 = deltas("tp2_hit_at")
    sl = deltas("sl_hit_at")
    resolved = [s.resolved_at - s.signal_timestamp for s in signals if s.resolved_at is not None]

    return TimeToOutcomeStats(
        sample_size_tp1=len(tp1),
        median_seconds_to_tp1=statistics.median(tp1) if tp1 else None,
        sample_size_tp2=len(tp2),
        median_seconds_to_tp2=statistics.median(tp2) if tp2 else None,
        sample_size_sl=len(sl),
        median_seconds_to_sl=statistics.median(sl) if sl else None,
        sample_size_resolved=len(resolved),
        median_seconds_to_resolution=statistics.median(resolved) if resolved else None,
    )


HISTORICAL_R_LABEL = (
    "Historical realized performance only, computed from completed and "
    "unambiguous outcomes (TP1_HIT/TP2_HIT/SL_HIT). This is NOT a "
    "probability and NOT a prediction of future results."
)


def historical_r_expectancy(signals: list[SignalRecord]) -> HistoricalRExpectancy:
    """R = realized return in units of initial risk, derived only from the
    immutable stored risk/reward ratios (initial_rr_tp1/initial_rr_tp2) and
    the terminal outcome_state -- never from a re-derived exit price.
    SL_HIT is exactly -1R by construction (the stop is the exit level).
    AMBIGUOUS and EXPIRED signals have no determinable single exit level
    and are excluded rather than guessed."""
    r_values: list[float] = []
    wins = 0
    for s in signals:
        if s.outcome_state == STATE_SL_HIT:
            r_values.append(-1.0)
        elif s.outcome_state == STATE_TP1_HIT:
            r_values.append(s.initial_rr_tp1)
            wins += 1
        elif s.outcome_state == STATE_TP2_HIT and s.initial_rr_tp2 is not None:
            r_values.append(s.initial_rr_tp2)
            wins += 1

    stats = _metric_stats(r_values)
    return HistoricalRExpectancy(
        sample_size=stats.count,
        low_confidence=stats.count < LOW_CONFIDENCE_SAMPLE_SIZE,
        avg_r=stats.mean,
        median_r=stats.median,
        win_rate=(wins / stats.count * 100) if stats.count else None,
        label=HISTORICAL_R_LABEL,
    )
