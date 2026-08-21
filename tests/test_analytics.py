from __future__ import annotations

from sentinel.analytics import aggregations as agg
from sentinel.analytics.analytics import generate_report, generate_report_from_store
from sentinel.outcomes.models import (
    STATE_AMBIGUOUS,
    STATE_EXPIRED,
    STATE_PENDING,
    STATE_SL_HIT,
    STATE_TP1_HIT,
    STATE_TP2_HIT,
    SignalRecord,
    WindowResult,
)
from sentinel.outcomes.store import OutcomeStore


def _signal(
    signal_id="sig-1",
    symbol="BTCUSDT",
    direction="long",
    setup_type="breakout",
    setup_score=75.0,
    scanner_activity_score=60.0,
    structure_confidence=50.0,
    timeframe_alignment="strong_bullish_alignment",
    initial_rr_tp1=2.0,
    initial_rr_tp2=3.5,
    signal_timestamp=1_000_000.0,
    outcome_state=STATE_PENDING,
    tp1_hit_at=None,
    tp2_hit_at=None,
    sl_hit_at=None,
    ambiguous_at=None,
    resolved_at=None,
) -> SignalRecord:
    return SignalRecord(
        signal_id=signal_id,
        alert_id=signal_id,
        symbol=symbol,
        direction=direction,
        setup_type=setup_type,
        setup_score=setup_score,
        scanner_activity_score=scanner_activity_score,
        structure_pattern="bullish_breakout",
        structure_confidence=structure_confidence,
        timeframe_alignment=timeframe_alignment,
        entry_zone_low=100.0,
        entry_zone_high=102.0,
        reference_entry_price=101.0,
        stop_price=95.0,
        take_profit_1=110.0,
        take_profit_2=120.0,
        initial_rr_tp1=initial_rr_tp1,
        initial_rr_tp2=initial_rr_tp2,
        signal_timestamp=signal_timestamp,
        outcome_state=outcome_state,
        tp1_hit_at=tp1_hit_at,
        tp2_hit_at=tp2_hit_at,
        sl_hit_at=sl_hit_at,
        ambiguous_at=ambiguous_at,
        resolved_at=resolved_at,
        last_evaluated_at=signal_timestamp,
    )


def _window(signal_id, window_label, forward_return=None, mfe=None, mae=None, evaluated_at=1_000_100.0) -> WindowResult:
    return WindowResult(signal_id=signal_id, window_label=window_label, forward_return=forward_return, mfe=mfe, mae=mae, evaluated_at=evaluated_at)


# -- empty / single-record database -------------------------------------

def test_empty_database_produces_zeroed_report_without_crashing(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    report = generate_report_from_store(store)

    assert report.total_signals == 0
    assert report.completed_signals == 0
    assert report.pending_signals == 0
    assert report.outcome_distribution.sample_size == 0
    assert report.outcome_distribution.low_confidence is True
    assert report.outcome_distribution.tp1_hit_rate is None
    # known categories still present with 0 samples, never dropped
    assert report.by_setup_type["breakout"].sample_size == 0
    assert report.by_direction["long"].sample_size == 0
    assert report.historical_r.sample_size == 0
    assert report.historical_r.avg_r is None


def test_single_record_database_reports_full_confidence_warning(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    store.create_signal_sync(_signal(outcome_state=STATE_TP1_HIT, tp1_hit_at=1_000_060.0, resolved_at=1_000_060.0))

    report = generate_report_from_store(store)

    assert report.total_signals == 1
    breakout = report.by_setup_type["breakout"]
    assert breakout.sample_size == 1
    assert breakout.low_confidence is True
    assert breakout.outcomes.tp1_hit_rate == 100.0


# -- setup-type aggregation -----------------------------------------------

def test_setup_type_aggregation_groups_correctly_and_shows_absent_types():
    signals = [
        _signal(signal_id="a", setup_type="breakout", outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0),
        _signal(signal_id="b", setup_type="breakout", outcome_state=STATE_SL_HIT, sl_hit_at=10.0),
        _signal(signal_id="c", setup_type="pullback", outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0),
    ]
    groups = agg.group_by_setup_type(signals, [])

    assert groups["breakout"].sample_size == 2
    assert groups["pullback"].sample_size == 1
    assert groups["trend_continuation"].sample_size == 0  # absent type still shown


# -- LONG/SHORT aggregation -------------------------------------------------

def test_direction_aggregation_long_vs_short():
    signals = [
        _signal(signal_id="a", direction="long", outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0),
        _signal(signal_id="b", direction="short", outcome_state=STATE_SL_HIT, sl_hit_at=10.0),
        _signal(signal_id="c", direction="short", outcome_state=STATE_SL_HIT, sl_hit_at=10.0),
    ]
    groups = agg.group_by_direction(signals, [])

    assert groups["long"].sample_size == 1
    assert groups["short"].sample_size == 2
    assert groups["short"].outcomes.sl_hit_rate == 100.0


# -- score / confidence / R:R buckets ---------------------------------------

def test_setup_score_buckets_place_values_correctly():
    signals = [
        _signal(signal_id="a", setup_score=10.0),
        _signal(signal_id="b", setup_score=55.0),
        _signal(signal_id="c", setup_score=99.9),
        _signal(signal_id="d", setup_score=100.0),
    ]
    groups = agg.bucket_by(signals, [], lambda s: s.setup_score, agg.SCORE_BUCKETS)

    assert groups["0-49"].sample_size == 1
    assert groups["50-59"].sample_size == 1
    assert groups["90-100"].sample_size == 2  # both 99.9 and exactly 100.0


def test_score_buckets_do_not_assume_higher_is_better_low_bucket_can_outperform():
    signals = [
        _signal(signal_id="a", setup_score=10.0, outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0),
        _signal(signal_id="b", setup_score=95.0, outcome_state=STATE_SL_HIT, sl_hit_at=10.0),
    ]
    groups = agg.bucket_by(signals, [], lambda s: s.setup_score, agg.SCORE_BUCKETS)

    assert groups["0-49"].outcomes.tp1_hit_rate == 100.0
    assert groups["90-100"].outcomes.sl_hit_rate == 100.0


def test_scanner_and_structure_confidence_buckets_handle_missing_values():
    signals = [
        _signal(signal_id="a", scanner_activity_score=None, structure_confidence=None),
        _signal(signal_id="b", scanner_activity_score=65.0, structure_confidence=72.0),
    ]
    scanner_groups = agg.bucket_by(signals, [], lambda s: s.scanner_activity_score, agg.SCORE_BUCKETS)
    structure_groups = agg.bucket_by(signals, [], lambda s: s.structure_confidence, agg.SCORE_BUCKETS)

    assert scanner_groups[agg.NOT_AVAILABLE].sample_size == 1
    assert scanner_groups["60-69"].sample_size == 1
    assert structure_groups[agg.NOT_AVAILABLE].sample_size == 1
    assert structure_groups["70-79"].sample_size == 1


def test_rr_buckets_place_values_correctly():
    signals = [
        _signal(signal_id="a", initial_rr_tp1=1.2),
        _signal(signal_id="b", initial_rr_tp1=1.5),
        _signal(signal_id="c", initial_rr_tp1=6.0),
    ]
    groups = agg.bucket_by(signals, [], lambda s: s.initial_rr_tp1, agg.RR_BUCKETS)

    assert groups["<1.5"].sample_size == 1
    assert groups["1.5-1.99"].sample_size == 1
    assert groups["5.0+"].sample_size == 1


# -- alignment groups (only real labels, per requirement 7) -----------------

def test_alignment_groups_use_only_labels_present_in_data():
    signals = [
        _signal(signal_id="a", timeframe_alignment="strong_bullish_alignment"),
        _signal(signal_id="b", timeframe_alignment="mixed"),
        _signal(signal_id="c", timeframe_alignment=None),
    ]
    groups = agg.group_by_timeframe_alignment(signals, [])

    assert set(groups.keys()) == {"strong_bullish_alignment", "mixed", agg.NOT_AVAILABLE}
    assert "insufficient_data" not in groups  # never invented -- not present in this data
    assert groups["mixed"].sample_size == 1
    assert groups[agg.NOT_AVAILABLE].sample_size == 1


# -- forward-return aggregation, respects missing windows --------------------

def test_forward_return_aggregation_only_uses_existing_windows():
    signals = [_signal(signal_id="a"), _signal(signal_id="b")]
    windows = [
        _window("a", "1m", forward_return=2.0),
        _window("b", "1m", forward_return=-1.0),
        _window("a", "5m", forward_return=3.0),
        # "b" has no 5m window yet -- must not be substituted with 0 or dropped silently
    ]
    result = agg.forward_return_by_window(signals, windows)

    assert result["1m"].sample_size == 2
    assert result["1m"].avg_return == 0.5
    assert result["5m"].sample_size == 1
    assert result["5m"].avg_return == 3.0
    assert result["15m"].sample_size == 0
    assert result["15m"].avg_return is None


def test_forward_return_never_substitutes_missing_values_with_zero():
    signals = [_signal(signal_id="a")]
    windows = [_window("a", "1m", forward_return=None, mfe=None, mae=None)]  # not yet elapsed
    result = agg.forward_return_by_window(signals, windows)

    assert result["1m"].sample_size == 0
    assert result["1m"].avg_return is None


# -- MFE / MAE, direction-respecting ------------------------------------------

def test_mfe_mae_respects_long_vs_short_direction():
    signals = [
        _signal(signal_id="a", direction="long"),
        _signal(signal_id="b", direction="short"),
    ]
    windows = [
        _window("a", "1m", mfe=5.0, mae=1.0),
        _window("b", "1m", mfe=4.0, mae=2.0),
    ]
    results = {(m.direction, m.window_label): m for m in agg.mfe_mae_by_direction_and_window(signals, windows)}

    assert results[("long", "1m")].mfe.mean == 5.0
    assert results[("long", "1m")].mae.mean == 1.0
    assert results[("short", "1m")].mfe.mean == 4.0
    assert results[("short", "1m")].mae.mean == 2.0


# -- time-to-outcome -----------------------------------------------------------

def test_time_to_outcome_computes_medians_only_where_timestamps_exist():
    signals = [
        _signal(signal_id="a", signal_timestamp=1000.0, outcome_state=STATE_TP1_HIT, tp1_hit_at=1060.0, resolved_at=1060.0),
        _signal(signal_id="b", signal_timestamp=2000.0, outcome_state=STATE_TP1_HIT, tp1_hit_at=2120.0, resolved_at=2120.0),
        _signal(signal_id="c", signal_timestamp=3000.0, outcome_state=STATE_PENDING),  # no timestamps yet
    ]
    result = agg.time_to_outcome(signals)

    assert result.sample_size_tp1 == 2
    assert result.median_seconds_to_tp1 == 90.0
    assert result.sample_size_sl == 0
    assert result.median_seconds_to_sl is None
    assert result.sample_size_resolved == 2


# -- historical R expectancy ----------------------------------------------------

def test_historical_r_only_uses_completed_unambiguous_outcomes():
    signals = [
        _signal(signal_id="a", outcome_state=STATE_TP1_HIT, initial_rr_tp1=2.0, tp1_hit_at=10.0, resolved_at=10.0),
        _signal(signal_id="b", outcome_state=STATE_SL_HIT, sl_hit_at=10.0, resolved_at=10.0),
        _signal(signal_id="c", outcome_state=STATE_TP2_HIT, initial_rr_tp2=3.5, tp1_hit_at=5.0, tp2_hit_at=10.0, resolved_at=10.0),
        _signal(signal_id="d", outcome_state=STATE_AMBIGUOUS, ambiguous_at=10.0, resolved_at=10.0),
        _signal(signal_id="e", outcome_state=STATE_EXPIRED, resolved_at=10.0),
        _signal(signal_id="f", outcome_state=STATE_PENDING),
    ]
    result = agg.historical_r_expectancy(signals)

    assert result.sample_size == 3  # a, b, c only -- d/e/f excluded
    assert result.avg_r == (2.0 + -1.0 + 3.5) / 3
    assert result.win_rate == (2 / 3) * 100
    assert "not a probability" in result.label.lower() or "NOT a probability" in result.label


def test_historical_r_sl_hit_is_exactly_minus_one_r():
    signals = [_signal(signal_id="a", outcome_state=STATE_SL_HIT, sl_hit_at=10.0, resolved_at=10.0)]
    result = agg.historical_r_expectancy(signals)

    assert result.avg_r == -1.0


# -- sample-size protection -----------------------------------------------------

def test_sample_size_of_one_still_flagged_low_confidence_even_at_100_percent():
    signals = [_signal(signal_id="a", outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0)]
    rates = agg.outcome_rates(signals)

    assert rates.sample_size == 1
    assert rates.low_confidence is True
    assert rates.tp1_hit_rate == 100.0  # honest, not hidden -- just flagged


def test_sample_size_above_threshold_not_flagged_low_confidence():
    signals = [_signal(signal_id=str(i), outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0) for i in range(agg.LOW_CONFIDENCE_SAMPLE_SIZE)]
    rates = agg.outcome_rates(signals)

    assert rates.low_confidence is False


# -- mixed completed/pending, ambiguous, expired -------------------------------

def test_mixed_completed_and_pending_split_correctly():
    signals = [
        _signal(signal_id="a", outcome_state=STATE_PENDING),
        _signal(signal_id="b", outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0, resolved_at=10.0),
        _signal(signal_id="c", outcome_state=STATE_AMBIGUOUS, ambiguous_at=10.0, resolved_at=10.0),
        _signal(signal_id="d", outcome_state=STATE_EXPIRED, resolved_at=10.0),
    ]
    report = generate_report(signals, [])

    assert report.total_signals == 4
    assert report.pending_signals == 1
    assert report.completed_signals == 3
    assert report.outcome_distribution.ambiguous_rate == 25.0
    assert report.outcome_distribution.expired_rate == 25.0


# -- determinism ----------------------------------------------------------------

def test_repeated_calculation_is_deterministic():
    signals = [
        _signal(signal_id="a", outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0, resolved_at=10.0),
        _signal(signal_id="b", outcome_state=STATE_SL_HIT, sl_hit_at=20.0, resolved_at=20.0, direction="short"),
    ]
    windows = [_window("a", "1m", forward_return=1.5, mfe=2.0, mae=0.5)]

    report_1 = generate_report(signals, windows)
    report_2 = generate_report(signals, windows)

    assert report_1.outcome_distribution == report_2.outcome_distribution
    assert report_1.by_setup_type == report_2.by_setup_type
    assert report_1.forward_returns_by_window == report_2.forward_returns_by_window
    assert report_1.historical_r == report_2.historical_r


def test_generate_report_from_store_matches_generate_report(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    signal = _signal(signal_id="a", outcome_state=STATE_TP1_HIT, tp1_hit_at=10.0, resolved_at=10.0)
    store.create_signal_sync(signal)
    store.upsert_window_sync(_window("a", "1m", forward_return=2.0, mfe=3.0, mae=1.0))

    via_store = generate_report_from_store(store)
    via_direct = generate_report([signal], [_window("a", "1m", forward_return=2.0, mfe=3.0, mae=1.0)])

    assert via_store.total_signals == via_direct.total_signals
    assert via_store.by_setup_type == via_direct.by_setup_type
    assert via_store.forward_returns_by_window == via_direct.forward_returns_by_window


# -- immutability: analytics must never write to the store ---------------------

def test_generate_report_from_store_does_not_modify_stored_signal(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    original = _signal(signal_id="a", outcome_state=STATE_PENDING)
    store.create_signal_sync(original)

    generate_report_from_store(store)

    reloaded = store.get_signal_sync("a")
    assert reloaded == original


# -- momentum/volume/volatility gap is disclosed, never fabricated -------------

def test_momentum_volume_volatility_is_reported_as_unavailable_not_fabricated():
    report = generate_report([], [])

    assert report.momentum_volume_volatility.available is False
    assert report.momentum_volume_volatility.reason  # non-empty explanation
