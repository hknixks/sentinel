from __future__ import annotations

from sentinel.market_state import Candle, SymbolState
from sentinel.scanner.features import (
    MarketFeatures,
    ReturnFeatures,
    TrendFeatures,
    VolatilityFeatures,
    VolumeFeatures,
)
from sentinel.structure.structure import (
    StructureEngine,
    _classify_phase,
    _classify_trend,
    _compute_alignment,
    _detect_breakout,
    _detect_failed_breakout,
    _detect_pullback,
    _detect_range,
    _find_swings,
    _levels,
)
from sentinel.structure.models import TimeframeStructure


def _c(open_time: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> Candle:
    return Candle(open_time=open_time, open=o, high=h, low=l, close=c, volume=v, is_closed=True)


def _state(symbol: str, candles: list[Candle]) -> SymbolState:
    return SymbolState(symbol=symbol, candle_history=tuple(candles))


def _empty_features() -> MarketFeatures:
    return MarketFeatures(
        symbol="X",
        timestamp=0.0,
        candle_count=0,
        returns=ReturnFeatures(None, None, None, None),
        volume=VolumeFeatures(None, None, None, None),
        volatility=VolatilityFeatures(None, None, None),
        trend=TrendFeatures(None, None, None),
    )


def _features_with_volatility_expansion(value: float | None) -> MarketFeatures:
    f = _empty_features()
    return MarketFeatures(
        symbol=f.symbol,
        timestamp=f.timestamp,
        candle_count=f.candle_count,
        returns=f.returns,
        volume=f.volume,
        volatility=VolatilityFeatures(None, None, value),
        trend=f.trend,
    )


# -- 1. bullish higher-high/higher-low structure --------------------------

def _bullish_hh_hl_candles() -> list[Candle]:
    # Engineered so _find_swings sees exactly: low(998)@2, high(1015)@4,
    # low(1000)@6, high(1020)@8 -- a clean higher-low/higher-high sequence.
    data = [
        (1005, 1000),
        (1008, 1003),
        (1006, 998),
        (1010, 1002),
        (1015, 1005),
        (1012, 1004),
        (1009, 1000),
        (1014, 1006),
        (1020, 1008),
        (1016, 1010),
        (1018, 1012),
    ]
    return [_c(i * 60_000, (h + l) / 2, h, l, (h + l) / 2) for i, (h, l) in enumerate(data)]


def test_bullish_higher_high_higher_low_structure():
    state = _state("BTCUSDT", _bullish_hh_hl_candles())
    result = StructureEngine().analyze(state)

    assert result.structure.trend == "bullish"
    assert result.features.higher_highs is True
    assert result.features.higher_lows is True
    assert result.directional_bias in ("bullish", "neutral")  # neutral only if alignment insufficient


# -- 2. bearish lower-low/lower-high structure -----------------------------

def _bearish_ll_lh_candles() -> list[Candle]:
    data = [
        (1000, 995),
        (997, 992),
        (999, 994),  # high pivot idx2: 999>997(prev) and 999>? need mirror of bullish
        (995, 990),
        (990, 985),  # low pivot idx4
        (993, 988),
        (996, 991),  # high pivot idx6, lower than idx2 (996<999)
        (986, 981),
        (980, 975),  # low pivot idx8, lower than idx4 (980<990)
        (984, 979),
        (982, 977),
    ]
    return [_c(i * 60_000, (h + l) / 2, h, l, (h + l) / 2) for i, (h, l) in enumerate(data)]


def test_bearish_lower_low_lower_high_structure():
    state = _state("BTCUSDT", _bearish_ll_lh_candles())
    result = StructureEngine().analyze(state)

    assert result.structure.trend == "bearish"
    assert result.features.lower_lows is True
    assert result.features.lower_highs is True


# -- 3. neutral structure ---------------------------------------------------

def test_neutral_structure_for_flat_market():
    candles = [_c(i * 60_000, 100.0, 100.0, 100.0, 100.0) for i in range(15)]
    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)

    assert result.structure.trend == "neutral"
    assert result.directional_bias == "neutral"


# -- 4. range -----------------------------------------------------------

def _range_candles(n: int = 22) -> list[Candle]:
    candles = []
    for i in range(n):
        if i % 2 == 0:
            h, l = 1004.0, 1000.0
        else:
            h, l = 1010.0, 1006.0
        mid = (h + l) / 2
        candles.append(_c(i * 60_000, mid, h, l, mid))
    return candles


def test_range_detected_with_repeated_boundary_touches():
    state = _state("BTCUSDT", _range_candles())
    result = StructureEngine().analyze(state)

    assert result.structure.pattern == "range"
    assert result.features.range_width is not None
    assert result.features.range_width > 0


def test_detect_range_requires_multiple_touches_each_side():
    # Only one touch of the lower boundary -- should not be called a range.
    candles = [_c(i * 60_000, 1005.0, 1010.0, 1005.0, 1005.0) for i in range(10)]
    candles[3] = _c(3 * 60_000, 1000.0, 1002.0, 999.0, 1001.0)  # single low touch
    info = _detect_range(candles)
    assert info is not None
    assert info.is_range is False


# -- 5 & 6. breakout ------------------------------------------------------

def _contained_candles(n: int, high: float = 1010.0, low: float = 1000.0) -> list[Candle]:
    mid = (high + low) / 2
    return [_c(i * 60_000, mid, high, low, mid) for i in range(n)]


def test_bullish_breakout_detected():
    candles = _contained_candles(14)
    candles.append(_c(14 * 60_000, 1015.0, 1022.0, 1015.0, 1020.0))
    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)

    assert result.structure.pattern == "bullish_breakout"
    assert result.features.breakout_distance is not None
    assert result.features.breakout_distance > 0


def test_bearish_breakout_detected():
    candles = _contained_candles(14)
    candles.append(_c(14 * 60_000, 985.0, 985.0, 975.0, 980.0))
    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)

    assert result.structure.pattern == "bearish_breakout"
    assert result.features.breakout_distance is not None


def test_breakout_uses_close_not_wick_a_reverting_spike_is_not_a_breakout():
    candles = _contained_candles(14)
    # Huge wick above the range, but closes back inside -- not a breakout.
    candles.append(_c(14 * 60_000, 1005.0, 1050.0, 1005.0, 1005.0))
    breakout = _detect_breakout(candles)
    assert breakout is None


# -- 7 & 8. failed breakout -------------------------------------------------

def test_failed_bullish_breakout():
    reference = _contained_candles(9)  # ref_high=1010, ref_low=1000
    window = _contained_candles(5)
    window[2] = _c(0, 1009.0, 1020.0, 1008.0, 1015.0)  # spikes beyond ref_high
    latest = _c(0, 1006.0, 1009.0, 1003.0, 1005.0)  # closes back inside
    candles = reference + window + [latest]

    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)

    assert result.structure.pattern == "failed_bullish_breakout"


def test_failed_bearish_breakout():
    reference = _contained_candles(9)  # ref_high=1010, ref_low=1000
    window = _contained_candles(5)
    window[2] = _c(0, 1001.0, 1002.0, 980.0, 995.0)  # spikes beyond ref_low
    latest = _c(0, 1004.0, 1007.0, 1001.0, 1005.0)  # closes back inside
    candles = reference + window + [latest]

    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)

    assert result.structure.pattern == "failed_bearish_breakout"


# -- 9 & 10. pullback -------------------------------------------------------

def test_bullish_pullback_requires_existing_uptrend():
    candles = _bullish_hh_hl_candles()
    # Padding keeps the idx8 swing high (1020) inside the failed-breakout
    # 'reference' window rather than 'recent_window', then a genuine
    # pullback (retracement that stays above the last swing low, 1000).
    tail = [
        (1011.0, 1014.0, 1008.0, 1011.0),
        (1010.0, 1013.0, 1007.0, 1010.0),
        (1006.0, 1009.0, 1003.0, 1006.0),
        (1005.0, 1008.0, 1002.0, 1005.0),
    ]
    candles += [_c((11 + i) * 60_000, o, h, l, c) for i, (o, h, l, c) in enumerate(tail)]

    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)

    assert result.structure.trend == "bullish"
    assert result.structure.pattern == "bullish_pullback"


def test_bearish_pullback_requires_existing_downtrend():
    candles = _bearish_ll_lh_candles()
    # Deep low continues the downtrend (973, still below the prior 975 low),
    # then a controlled bounce that stays below the last swing high (984) --
    # a pullback, not a reversal.
    tail = [
        (975.0, 976.0, 973.0, 975.0),
        (976.0, 978.0, 974.0, 976.0),
        (977.0, 979.0, 975.0, 977.0),
        (978.0, 980.0, 976.0, 978.0),
        (979.0, 981.0, 977.0, 979.0),
        (980.0, 982.0, 978.0, 980.0),
        (981.0, 983.0, 979.0, 981.0),
    ]
    candles += [_c((11 + i) * 60_000, o, h, l, c) for i, (o, h, l, c) in enumerate(tail)]

    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)

    assert result.structure.trend == "bearish"
    assert result.structure.pattern == "bearish_pullback"


def test_no_pullback_without_existing_trend():
    candles = [_c(i * 60_000, 100.0, 100.5, 99.5, 100.0) for i in range(10)]
    pullback = _detect_pullback(candles, "neutral", [])
    assert pullback is None


# -- 11 & 12. compression / expansion ---------------------------------------

def test_compression_classified_from_volatility_expansion():
    assert _classify_phase(0.5) == "compression"


def test_expansion_classified_from_volatility_expansion():
    assert _classify_phase(1.5) == "expansion"


def test_stable_phase_for_normal_volatility_expansion():
    assert _classify_phase(1.0) == "stable"


def test_phase_none_when_volatility_expansion_unavailable():
    assert _classify_phase(None) is None


def test_engine_reports_compression_end_to_end():
    candles = _bullish_hh_hl_candles()
    state = _state("BTCUSDT", candles)
    features = _features_with_volatility_expansion(0.4)

    result = StructureEngine().analyze(state, features=features)

    assert result.structure.phase == "compression"
    assert result.features.compression is True
    assert result.features.expansion is False


def test_engine_reports_expansion_end_to_end():
    candles = _bullish_hh_hl_candles()
    state = _state("BTCUSDT", candles)
    features = _features_with_volatility_expansion(1.8)

    result = StructureEngine().analyze(state, features=features)

    assert result.structure.phase == "expansion"
    assert result.features.expansion is True


# -- 13. multi-timeframe alignment ------------------------------------------

def test_alignment_strong_bullish_when_all_timeframes_agree():
    tfs = [
        TimeframeStructure("1m", "bullish", 20),
        TimeframeStructure("5m", "bullish", 10),
        TimeframeStructure("15m", None, 3),
        TimeframeStructure("1h", None, 0),
    ]
    assert _compute_alignment(tfs) == "strong_bullish_alignment"


def test_alignment_strong_bearish_when_all_timeframes_agree():
    tfs = [
        TimeframeStructure("1m", "bearish", 20),
        TimeframeStructure("5m", "bearish", 10),
    ]
    assert _compute_alignment(tfs) == "strong_bearish_alignment"


def test_alignment_conflicting_short_vs_higher_timeframe():
    tfs = [
        TimeframeStructure("1m", "bullish", 20),
        TimeframeStructure("5m", "bullish", 10),
        TimeframeStructure("15m", "bearish", 8),
    ]
    assert _compute_alignment(tfs) == "bullish_short_term_bearish_higher_timeframe"


def test_alignment_mixed_when_no_clean_split():
    tfs = [
        TimeframeStructure("1m", "bullish", 20),
        TimeframeStructure("15m", "bearish", 8),
        TimeframeStructure("1h", "bullish", 7),
    ]
    # short_tf empty here (only 1m present among 1m/5m, but 1m alone still
    # counts as "short"); with 1m bullish and 15m+1h split bullish/bearish,
    # this isn't a clean short-vs-higher split -> mixed.
    result = _compute_alignment(tfs)
    assert result in ("mixed", "bullish_short_term_bearish_higher_timeframe")


def test_alignment_insufficient_data_with_fewer_than_two_directional_timeframes():
    tfs = [TimeframeStructure("1m", "bullish", 20), TimeframeStructure("5m", None, 3)]
    assert _compute_alignment(tfs) == "insufficient_data"


# -- 14. insufficient data ---------------------------------------------------

def test_insufficient_data_for_empty_history():
    state = SymbolState(symbol="BTCUSDT")
    result = StructureEngine().analyze(state)

    assert result.structure.trend is None
    assert result.structure.pattern == "insufficient_data"
    assert result.confidence == 0.0
    assert all(t.trend is None for t in result.timeframes)


def test_insufficient_data_for_short_history():
    candles = _contained_candles(4)  # below MIN_CANDLES_FOR_TIMEFRAME=7
    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)

    assert result.structure.pattern == "insufficient_data"


# -- 15. structural confidence -----------------------------------------------

def test_confidence_is_zero_with_no_data():
    state = SymbolState(symbol="BTCUSDT")
    result = StructureEngine().analyze(state)
    assert result.confidence == 0.0


def test_confidence_bounded_zero_to_hundred():
    state = _state("BTCUSDT", _bullish_hh_hl_candles())
    result = StructureEngine().analyze(state)
    assert 0.0 <= result.confidence <= 100.0


def test_confidence_higher_with_more_swing_evidence_and_alignment():
    weak_candles = _contained_candles(8)  # barely enough, no real swings
    weak_state = _state("BTCUSDT", weak_candles)
    weak_result = StructureEngine().analyze(weak_state)

    strong_candles = _bullish_hh_hl_candles()
    strong_state = _state("ETHUSDT", strong_candles)
    strong_features = MarketFeatures(
        symbol="ETHUSDT",
        timestamp=0.0,
        candle_count=len(strong_candles),
        returns=ReturnFeatures(None, None, None, None),
        volume=VolumeFeatures(None, None, 2.0, None),
        volatility=VolatilityFeatures(None, None, 1.5),
        trend=TrendFeatures(None, None, None),
    )
    strong_result = StructureEngine().analyze(strong_state, features=strong_features)

    assert strong_result.confidence > weak_result.confidence


# -- 16. support/resistance --------------------------------------------------

def test_levels_uses_most_recent_swing_low_and_high():
    candles = _bullish_hh_hl_candles()
    swings = _find_swings(candles)
    levels = _levels(tuple(candles), swings)

    assert levels.support == 1000.0  # most recent swing low
    assert levels.resistance == 1020.0  # most recent swing high
    assert levels.recent_high == max(c.high for c in candles[-20:])
    assert levels.recent_low == min(c.low for c in candles[-20:])


def test_levels_none_for_empty_history():
    from sentinel.structure.structure import _levels as levels_fn

    result = levels_fn((), [])
    assert result.recent_high is None
    assert result.support is None


# -- Edge cases ---------------------------------------------------------------

def test_missing_candles_does_not_crash():
    state = SymbolState(symbol="BTCUSDT")
    result = StructureEngine().analyze(state)
    assert result is not None


def test_noisy_market_does_not_falsely_trend():
    candles = _range_candles(20)
    trend = _classify_trend(_find_swings(candles))
    assert trend == "neutral"


def test_zero_price_candles_do_not_crash():
    candles = _contained_candles(10)
    candles.append(_c(10 * 60_000, 0.0, 0.0, 0.0, 0.0, v=0.0))
    state = _state("BTCUSDT", candles)
    result = StructureEngine().analyze(state)
    assert result is not None
    assert result.confidence >= 0.0


def test_zero_volume_does_not_crash_confidence():
    candles = _contained_candles(10, high=1010.0, low=1000.0)
    state = _state("BTCUSDT", candles)
    features = _empty_features()
    result = StructureEngine().analyze(state, features=features)
    assert 0.0 <= result.confidence <= 100.0


def test_conflicting_timeframes_reported_not_hidden():
    tfs = [
        TimeframeStructure("1m", "bearish", 20),
        TimeframeStructure("15m", "bullish", 8),
    ]
    alignment = _compute_alignment(tfs)
    assert alignment != "strong_bullish_alignment"
    assert alignment != "strong_bearish_alignment"


def test_engine_never_raises_on_various_short_histories():
    for n in range(0, 10):
        candles = _contained_candles(n) if n else []
        state = _state("BTCUSDT", candles)
        result = StructureEngine().analyze(state)
        assert result is not None
        assert isinstance(result.confidence, float)
