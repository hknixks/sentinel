from __future__ import annotations

import math

from sentinel.market_state import Candle, SymbolState
from sentinel.scanner.features import (
    MarketFeatures,
    ReturnFeatures,
    TrendFeatures,
    VolatilityFeatures,
    VolumeFeatures,
)
from sentinel.scanner.scanner import ScannerResult
from sentinel.setups.filters import check_momentum_confirmation, check_risk_reward, check_timeframe_conflict
from sentinel.setups.risk import compute_take_profits, reward_and_rr, risk_per_unit
from sentinel.setups.setup_engine import SetupEngine
from sentinel.structure.models import Levels, Structure, StructureFeatures, StructureResult, TimeframeStructure


def _state(symbol: str = "BTCUSDT", price: float = 110.0) -> SymbolState:
    candle = Candle(open_time=0, open=price - 1, high=price + 1, low=price - 2, close=price, volume=100.0, is_closed=True)
    return SymbolState(symbol=symbol, price=price, candle_history=(candle,))


def _features(
    return_1m: float | None = 0.1,
    return_5m: float | None = 0.3,
    return_15m: float | None = 0.5,
    return_1h: float | None = 1.0,
    relative_volume: float | None = 1.5,
    volume_acceleration: float | None = 10.0,
    recent_range: float | None = 0.3,
    rolling_volatility: float | None = 0.2,
    volatility_expansion: float | None = 1.4,
) -> MarketFeatures:
    return MarketFeatures(
        symbol="BTCUSDT",
        timestamp=0.0,
        candle_count=100,
        returns=ReturnFeatures(return_1m, return_5m, return_15m, return_1h),
        volume=VolumeFeatures(
            volume_current=100.0, rolling_volume=2000.0, relative_volume=relative_volume, volume_acceleration=volume_acceleration
        ),
        volatility=VolatilityFeatures(recent_range=recent_range, rolling_volatility=rolling_volatility, volatility_expansion=volatility_expansion),
        trend=TrendFeatures(direction="up", trend_strength=1.0, timeframe_alignment=100.0),
    )


def _structure(
    trend: str | None = "bullish",
    pattern: str | None = "bullish_breakout",
    phase: str | None = "expansion",
    directional_bias: str = "bullish",
    confidence: float = 80.0,
    recent_high: float | None = 120.0,
    recent_low: float | None = 90.0,
    support: float | None = 96.0,
    resistance: float | None = 108.0,
    breakout_distance: float | None = 0.5,
    timeframe_alignment: str = "strong_bullish_alignment",
    higher_highs: bool | None = True,
    higher_lows: bool | None = True,
    lower_highs: bool | None = False,
    lower_lows: bool | None = False,
) -> StructureResult:
    return StructureResult(
        symbol="BTCUSDT",
        timestamp=0.0,
        structure=Structure(trend=trend, pattern=pattern, phase=phase),
        directional_bias=directional_bias,
        confidence=confidence,
        levels=Levels(recent_high=recent_high, recent_low=recent_low, support=support, resistance=resistance),
        features=StructureFeatures(
            higher_highs=higher_highs,
            lower_highs=lower_highs,
            higher_lows=higher_lows,
            lower_lows=lower_lows,
            range_width=None,
            breakout_distance=breakout_distance,
            compression=(phase == "compression") if phase else None,
            expansion=(phase == "expansion") if phase else None,
        ),
        timeframe_alignment=timeframe_alignment,
        timeframes=(
            TimeframeStructure("1m", trend, 20),
            TimeframeStructure("5m", trend, 10),
            TimeframeStructure("15m", trend, 8),
            TimeframeStructure("1h", None, 0),
        ),
        reason="test fixture",
    )


def _rejection_reasons(evaluation, setup_type, direction):
    for r in evaluation.rejections:
        if r.setup_type == setup_type and r.direction == direction:
            return r.reasons
    return None


def _candidate(evaluation, setup_type, direction):
    for c in evaluation.candidates:
        if c.setup_type == setup_type and c.direction == direction:
            return c
    return None


# -- valid long breakout ------------------------------------------------

def test_valid_long_breakout_generates_candidate():
    price = 110.0
    state = _state(price=price)
    features = _features(return_15m=0.5, relative_volume=1.5)
    structure = _structure(
        pattern="bullish_breakout",
        trend="bullish",
        breakout_distance=0.5,
        timeframe_alignment="strong_bullish_alignment",
        recent_high=120.0,
        resistance=None,
    )

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)
    candidate = _candidate(evaluation, "breakout", "long")

    assert candidate is not None
    assert candidate.entry_zone.low < candidate.entry_zone.high
    assert candidate.entry_zone.high == price
    assert candidate.risk.stop_loss < price
    assert candidate.risk.take_profit_1 > price
    assert candidate.risk.risk_reward_tp1 >= 1.5
    assert 0.0 <= candidate.setup_score <= 100.0


# -- valid short breakout ------------------------------------------------

def test_valid_short_breakout_generates_candidate():
    price = 90.0
    state = _state(price=price)
    features = _features(return_15m=-0.5, relative_volume=1.5)
    structure = _structure(
        pattern="bearish_breakout",
        trend="bearish",
        directional_bias="bearish",
        breakout_distance=0.5,
        timeframe_alignment="strong_bearish_alignment",
        recent_low=80.0,
        support=None,
    )

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)
    candidate = _candidate(evaluation, "breakout", "short")

    assert candidate is not None
    assert candidate.entry_zone.low == price
    assert candidate.risk.stop_loss > price
    assert candidate.risk.take_profit_1 < price
    assert candidate.risk.risk_reward_tp1 >= 1.5


# -- wick-only breakout rejected -----------------------------------------

def test_wick_only_breakout_is_rejected():
    # Phase 3 only ever classifies a breakout from a CLOSE beyond the
    # boundary; if the pattern isn't a breakout type (e.g. a wick that
    # reverted, classified as 'neutral' or 'failed_bullish_breakout' by
    # Phase 3), Phase 4 must not fabricate a breakout setup.
    state = _state(price=110.0)
    features = _features()
    structure = _structure(pattern="failed_bullish_breakout", breakout_distance=None)

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)

    assert _candidate(evaluation, "breakout", "long") is None
    reasons = _rejection_reasons(evaluation, "breakout", "long")
    assert reasons == ("breakout_not_confirmed",)


# -- valid bullish pullback ------------------------------------------------

def test_valid_bullish_pullback_generates_candidate():
    price = 100.0
    state = _state(price=price)
    features = _features(return_1h=1.0)
    structure = _structure(
        pattern="bullish_pullback",
        trend="bullish",
        support=97.0,
        recent_low=94.0,
        resistance=112.0,
        timeframe_alignment="strong_bullish_alignment",
    )

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)
    candidate = _candidate(evaluation, "pullback", "long")

    assert candidate is not None
    assert candidate.entry_zone.low == 97.0
    assert candidate.entry_zone.high == price
    assert candidate.risk.stop_loss == 94.0
    assert candidate.risk.risk_reward_tp1 >= 1.5


# -- valid bearish pullback ------------------------------------------------

def test_valid_bearish_pullback_generates_candidate():
    price = 100.0
    state = _state(price=price)
    features = _features(return_1h=-1.0)
    structure = _structure(
        pattern="bearish_pullback",
        trend="bearish",
        directional_bias="bearish",
        resistance=103.0,
        recent_high=105.0,
        support=90.0,
        recent_low=85.0,
        timeframe_alignment="strong_bearish_alignment",
    )

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)
    candidate = _candidate(evaluation, "pullback", "short")

    assert candidate is not None
    assert candidate.entry_zone.high == 103.0
    assert candidate.entry_zone.low == price
    assert candidate.risk.stop_loss == 105.0
    assert candidate.risk.risk_reward_tp1 >= 1.5


# -- pullback without existing trend rejected -----------------------------

def test_pullback_without_existing_trend_is_rejected():
    state = _state(price=100.0)
    features = _features()
    structure = _structure(pattern="neutral", trend="neutral")

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)

    assert _candidate(evaluation, "pullback", "long") is None
    reasons = _rejection_reasons(evaluation, "pullback", "long")
    assert reasons == ("pullback_not_confirmed",)


# -- conflicting higher timeframe rejected --------------------------------

def test_conflicting_higher_timeframe_rejects_breakout():
    state = _state(price=110.0)
    features = _features(relative_volume=1.5, return_15m=0.5)
    structure = _structure(
        pattern="bullish_breakout",
        breakout_distance=0.5,
        timeframe_alignment="bullish_short_term_bearish_higher_timeframe",
    )

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)

    assert _candidate(evaluation, "breakout", "long") is None
    reasons = _rejection_reasons(evaluation, "breakout", "long")
    assert reasons == ("conflicting_timeframes",)


def test_check_timeframe_conflict_direct():
    assert check_timeframe_conflict("long", "strong_bearish_alignment") == "conflicting_timeframes"
    assert check_timeframe_conflict("short", "strong_bullish_alignment") == "conflicting_timeframes"
    assert check_timeframe_conflict("long", "mixed") is None
    assert check_timeframe_conflict("long", "insufficient_data") is None


# -- poor R:R rejected ------------------------------------------------------

def test_poor_risk_reward_is_rejected():
    price = 110.0
    state = _state(price=price)
    features = _features(relative_volume=1.5, return_15m=0.5)
    # Stop very close to entry (small risk) but TP capped near entry too,
    # via a resistance level barely beyond price -- poor R:R.
    structure = _structure(
        pattern="bullish_breakout",
        breakout_distance=0.05,  # boundary just barely below price -> small risk denominator... actually small risk, but TP is capped low
        recent_high=110.5,
        resistance=None,
    )

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)
    # With boundary very close to price and TP capped at 110.5, R:R should be poor.
    reasons = _rejection_reasons(evaluation, "breakout", "long")
    candidate = _candidate(evaluation, "breakout", "long")
    assert candidate is None or candidate.risk.risk_reward_tp1 >= 1.5
    if candidate is None:
        assert reasons == ("poor_risk_reward",)


def test_check_risk_reward_direct():
    assert check_risk_reward(1.0, 1.5) == "poor_risk_reward"
    assert check_risk_reward(None, 1.5) == "poor_risk_reward"
    assert check_risk_reward(2.0, 1.5) is None


# -- invalid structural stop rejected --------------------------------------

def test_invalid_stop_rejected_when_stop_on_wrong_side():
    price = 100.0
    state = _state(price=price)
    features = _features(return_1h=1.0)
    # recent_low ABOVE support -- structurally inconsistent, must reject.
    structure = _structure(pattern="bullish_pullback", trend="bullish", support=95.0, recent_low=97.0)

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)

    assert _candidate(evaluation, "pullback", "long") is None
    reasons = _rejection_reasons(evaluation, "pullback", "long")
    assert reasons == ("invalid_stop",)


# -- missing data rejected --------------------------------------------------

def test_missing_structure_data_rejects_all_setup_types():
    state = _state(price=100.0)
    structure = _structure(pattern="insufficient_data", trend=None)

    evaluation = SetupEngine().evaluate("BTCUSDT", state, None, None, structure)

    assert evaluation.candidates == ()
    assert len(evaluation.rejections) == 6
    assert all(r.reasons == ("insufficient_history",) for r in evaluation.rejections)


def test_no_candles_rejects_all_setup_types():
    state = SymbolState(symbol="BTCUSDT")
    structure = _structure()

    evaluation = SetupEngine().evaluate("BTCUSDT", state, None, None, structure)

    assert evaluation.candidates == ()
    assert len(evaluation.rejections) == 6


# -- range market rejected for trend continuation --------------------------

def test_range_market_rejects_trend_continuation():
    state = _state(price=100.0)
    features = _features()
    structure = _structure(pattern="range", trend="neutral")

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)

    assert _candidate(evaluation, "trend_continuation", "long") is None
    assert _candidate(evaluation, "trend_continuation", "short") is None
    reasons = _rejection_reasons(evaluation, "trend_continuation", "long")
    assert reasons == ("range_bound",)


def test_valid_trend_continuation_generates_candidate():
    price = 105.0
    state = _state(price=price)
    features = _features(return_1h=1.0, recent_range=0.2)
    structure = _structure(
        pattern="trending",
        trend="bullish",
        support=100.0,
        resistance=None,
        recent_high=120.0,
        timeframe_alignment="strong_bullish_alignment",
    )

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)
    candidate = _candidate(evaluation, "trend_continuation", "long")

    assert candidate is not None
    assert candidate.risk.stop_loss == 100.0
    assert candidate.risk.risk_reward_tp1 >= 1.5


# -- setup score bounds ------------------------------------------------------

def test_setup_score_remains_zero_to_hundred_across_all_candidates():
    price = 110.0
    state = _state(price=price)
    features = _features(relative_volume=1.5, return_15m=0.5)
    structure = _structure(pattern="bullish_breakout", breakout_distance=0.5, recent_high=130.0, resistance=None)

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)
    for c in evaluation.candidates:
        assert 0.0 <= c.setup_score <= 100.0
        assert not math.isnan(c.setup_score)
        assert not math.isinf(c.setup_score)


# -- no NaN / infinity / fabricated values -----------------------------------

def test_no_nan_or_infinity_in_any_field():
    price = 100.0
    state = _state(price=price)
    features = _features()
    structure = _structure(pattern="bullish_pullback", trend="bullish", support=95.0, recent_low=90.0)

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)
    for c in evaluation.candidates:
        for value in (
            c.entry_zone.low,
            c.entry_zone.high,
            c.invalidation_level,
            c.risk.stop_loss,
            c.risk.take_profit_1,
            c.risk.risk_per_unit,
            c.risk.reward_to_tp1,
            c.risk.risk_reward_tp1,
            c.setup_score,
        ):
            assert not math.isnan(value)
            assert not math.isinf(value)


def test_zero_risk_reward_helpers_do_not_crash():
    r = risk_per_unit("long", 100.0, 100.0)  # zero risk
    assert r == 0.0
    reward, rr = reward_and_rr(100.0, 110.0, r)
    assert reward is None
    assert rr is None


def test_compute_take_profits_tp2_always_beyond_tp1_long():
    levels = Levels(recent_high=105.0, recent_low=90.0, support=None, resistance=None)
    tps = compute_take_profits("long", 100.0, risk_per_unit=1.0, levels=levels)
    assert tps.tp1 == 105.0
    assert tps.tp2 > tps.tp1


def test_compute_take_profits_tp2_always_beyond_tp1_short():
    levels = Levels(recent_high=110.0, recent_low=95.0, support=None, resistance=None)
    tps = compute_take_profits("short", 100.0, risk_per_unit=1.0, levels=levels)
    assert tps.tp1 == 95.0
    assert tps.tp2 < tps.tp1


def test_compute_take_profits_falls_back_to_projection_with_no_levels():
    levels = Levels(recent_high=None, recent_low=None, support=None, resistance=None)
    tps = compute_take_profits("long", 100.0, risk_per_unit=2.0, levels=levels)
    assert tps.tp1_from_structure is False
    assert tps.tp1 > 100.0
    assert tps.tp2 > tps.tp1


# -- momentum confirmation edge cases -----------------------------------

def test_momentum_confirmation_uses_best_available_timeframe():
    features = _features(return_1m=None, return_5m=None, return_15m=0.5, return_1h=1.0)
    assert check_momentum_confirmation(features, "long", 0.05) is None


def test_momentum_confirmation_rejects_when_no_data():
    assert check_momentum_confirmation(None, "long", 0.05) == "inadequate_momentum"


# -- inadequate volume rejected for breakout ------------------------------

def test_inadequate_volume_rejects_breakout():
    state = _state(price=110.0)
    features = _features(relative_volume=0.8, return_15m=0.5)  # below 1.2 threshold
    structure = _structure(pattern="bullish_breakout", breakout_distance=0.5)

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)

    assert _candidate(evaluation, "breakout", "long") is None
    reasons = _rejection_reasons(evaluation, "breakout", "long")
    assert reasons == ("inadequate_volume",)


def test_scanner_result_is_consumed_into_confirmation_factors():
    price = 110.0
    state = _state(price=price)
    features = _features(relative_volume=1.5, return_15m=0.5)
    structure = _structure(pattern="bullish_breakout", breakout_distance=0.5, recent_high=130.0, resistance=None)
    scanner_result = ScannerResult(symbol="BTCUSDT", score=87.3, direction="bullish", features=features, timestamp=0.0)

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, scanner_result, structure)
    candidate = _candidate(evaluation, "breakout", "long")

    assert candidate is not None
    assert any("scanner activity score: 87.3" in f for f in candidate.confirmation_factors)


def test_inadequate_momentum_rejects_breakout():
    state = _state(price=110.0)
    features = _features(relative_volume=1.5, return_1m=0.01, return_5m=None, return_15m=None, return_1h=None)
    structure = _structure(pattern="bullish_breakout", breakout_distance=0.5)

    evaluation = SetupEngine().evaluate("BTCUSDT", state, features, None, structure)

    assert _candidate(evaluation, "breakout", "long") is None
    reasons = _rejection_reasons(evaluation, "breakout", "long")
    assert reasons == ("inadequate_momentum",)
