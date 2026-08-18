"""
Structure engine: describes the CURRENT market structure of a symbol --
swing-based trend, price-action pattern (range/breakout/pullback/etc.),
volatility phase (compression/expansion), key levels, and multi-timeframe
alignment. All derived locally from already-held 1m candle history -- no
extra API calls.

Descriptive only. This is not a trade signal and `confidence` is not a
probability of trade success -- see StructureResult.confidence.

Everything is computed from real candle data; anything without enough
history is None, never fabricated, padded, or substituted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sentinel.market_state import Candle, SymbolState
from sentinel.scanner.features import MarketFeatures
from sentinel.scanner.scanner import ScannerResult
from sentinel.structure.models import (
    Levels,
    Structure,
    StructureFeatures,
    StructureResult,
    TimeframeStructure,
)

# Timeframes analyzed, all derived locally from 1m candle history.
TIMEFRAME_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}

# Preferred timeframe for the primary structure read (trend/pattern/phase/
# levels/features): 15m first as a solid noise-filtered structural read,
# falling back to 5m then 1m as less history accumulates. 1h is still
# analyzed and included in the per-timeframe alignment breakdown, but is
# not used as primary -- it needs 7+ complete 1h candles (7+ hours of
# uptime) to be usable, which is rarely available.
PRIMARY_TIMEFRAME_PREFERENCE = ["15m", "5m", "1m"]

# Minimum resampled candles at a timeframe needed to attempt any structure
# read at all. Below this, the timeframe is None -- not guessed.
MIN_CANDLES_FOR_TIMEFRAME = 7

# Minimum alternating swing points (highs/lows) needed to call a trend
# bullish/bearish rather than "neutral" (no sufficiently clear structure).
MIN_SWINGS_FOR_DIRECTION = 4

RANGE_LOOKBACK = 20
RANGE_TOUCH_TOLERANCE_FRACTION = 0.15  # fraction of range width counted as a "touch"
RANGE_MIN_TOUCHES = 2

BREAKOUT_MIN_DISTANCE_PCT = 0.05
FAILED_BREAKOUT_LOOKBACK = 5
PULLBACK_LOOKBACK = 5
PULLBACK_MIN_RETRACEMENT_PCT = 0.05

COMPRESSION_VOLATILITY_EXPANSION_THRESHOLD = 0.7
EXPANSION_VOLATILITY_EXPANSION_THRESHOLD = 1.3

# Structural confidence weights (sum to 1.0). Trend clarity weighted
# highest since it's literally what's being classified -- more swing
# evidence means more confidence in the read itself. Alignment and
# volume/volatility confirmation are classic corroborating signals,
# weighted equally. History depth is included but lowest-weighted: a
# small amount of data can still show a valid pattern, but more history
# is genuinely more reliable.
CONFIDENCE_WEIGHTS = {
    "trend_clarity": 0.30,
    "alignment": 0.25,
    "volume_volatility": 0.25,
    "history_depth": 0.20,
}
FULL_TREND_CLARITY_SWING_COUNT = 6
FULL_HISTORY_DEPTH_CANDLES = 60

ALIGNMENT_SCORES: dict[str, float] = {
    "strong_bullish_alignment": 100.0,
    "strong_bearish_alignment": 100.0,
    "bullish_short_term_bearish_higher_timeframe": 40.0,
    "bearish_short_term_bullish_higher_timeframe": 40.0,
    "mixed": 30.0,
    "insufficient_data": 0.0,
}

Swing = tuple[int, str, float]  # (index, "high"|"low", price)


def _weighted_avg(pairs: list[tuple[float | None, float]]) -> float | None:
    available = [(v, w) for v, w in pairs if v is not None]
    if not available:
        return None
    total_weight = sum(w for _, w in available)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in available) / total_weight


def _resample(candles: tuple[Candle, ...], group_size: int) -> list[Candle]:
    """Group the most recent complete buckets of `group_size` 1m candles into
    higher-timeframe candles. Only complete buckets are used -- no partial
    or fabricated bars. Grouped from the most recent candle backward, so
    the result always reflects the latest complete window (not wall-clock
    aligned to Binance's own 5m/15m/1h boundaries, since Phase 1 only
    subscribes to the 1m kline stream)."""
    if group_size == 1:
        return list(candles)
    n = len(candles) // group_size
    if n == 0:
        return []
    usable = candles[-(n * group_size) :]
    out = []
    for i in range(n):
        bucket = usable[i * group_size : (i + 1) * group_size]
        out.append(
            Candle(
                open_time=bucket[0].open_time,
                open=bucket[0].open,
                high=max(c.high for c in bucket),
                low=min(c.low for c in bucket),
                close=bucket[-1].close,
                volume=sum(c.volume for c in bucket),
                is_closed=True,
            )
        )
    return out


def _find_swings(candles: list[Candle]) -> list[Swing]:
    """Local pivots using a 1-candle lookaround (simple fractal definition)."""
    swings: list[Swing] = []
    for i in range(1, len(candles) - 1):
        if candles[i].high > candles[i - 1].high and candles[i].high > candles[i + 1].high:
            swings.append((i, "high", candles[i].high))
        elif candles[i].low < candles[i - 1].low and candles[i].low < candles[i + 1].low:
            swings.append((i, "low", candles[i].low))
    return swings


def _alternate_swings(swings: list[Swing]) -> list[Swing]:
    """Collapse consecutive same-type swings to the most extreme, so the
    result strictly alternates high/low/high/low."""
    if not swings:
        return []
    cleaned = [swings[0]]
    for s in swings[1:]:
        if s[1] == cleaned[-1][1]:
            if s[1] == "high" and s[2] > cleaned[-1][2]:
                cleaned[-1] = s
            elif s[1] == "low" and s[2] < cleaned[-1][2]:
                cleaned[-1] = s
        else:
            cleaned.append(s)
    return cleaned


def _classify_trend(swings: list[Swing]) -> str:
    if len(swings) < MIN_SWINGS_FOR_DIRECTION:
        return "neutral"
    last4 = swings[-4:]
    types = [s[1] for s in last4]
    prices = [s[2] for s in last4]
    if types == ["low", "high", "low", "high"]:
        low1, high1, low2, high2 = prices
    elif types == ["high", "low", "high", "low"]:
        high1, low1, high2, low2 = prices
    else:
        return "neutral"
    if low2 > low1 and high2 > high1:
        return "bullish"
    if low2 < low1 and high2 < high1:
        return "bearish"
    return "neutral"


def _swing_feature_flags(
    swings: list[Swing],
) -> tuple[bool | None, bool | None, bool | None, bool | None]:
    """higher_highs, lower_highs, higher_lows, lower_lows -- comparing the
    two most recent swings of each type, when available."""
    highs = [s[2] for s in swings if s[1] == "high"]
    lows = [s[2] for s in swings if s[1] == "low"]
    higher_highs = highs[-1] > highs[-2] if len(highs) >= 2 else None
    lower_highs = highs[-1] < highs[-2] if len(highs) >= 2 else None
    higher_lows = lows[-1] > lows[-2] if len(lows) >= 2 else None
    lower_lows = lows[-1] < lows[-2] if len(lows) >= 2 else None
    return higher_highs, lower_highs, higher_lows, lower_lows


@dataclass(frozen=True)
class _RangeInfo:
    high: float
    low: float
    width_pct: float
    is_range: bool


def _detect_range(candles: list[Candle]) -> _RangeInfo | None:
    window = candles[-RANGE_LOOKBACK:]
    if len(window) < MIN_CANDLES_FOR_TIMEFRAME:
        return None
    high = max(c.high for c in window)
    low = min(c.low for c in window)
    if low <= 0:
        return None
    width_pct = (high - low) / low * 100
    tolerance = (high - low) * RANGE_TOUCH_TOLERANCE_FRACTION
    upper_touches = sum(1 for c in window if c.high >= high - tolerance)
    lower_touches = sum(1 for c in window if c.low <= low + tolerance)
    is_range = upper_touches >= RANGE_MIN_TOUCHES and lower_touches >= RANGE_MIN_TOUCHES
    return _RangeInfo(high=high, low=low, width_pct=width_pct, is_range=is_range)


def _candle_strength(candle: Candle) -> float:
    span = candle.high - candle.low
    if span <= 0:
        return 0.0
    return abs(candle.close - candle.open) / span


@dataclass(frozen=True)
class _BreakoutInfo:
    direction: str  # "bullish" | "bearish"
    boundary: float
    distance_pct: float
    candle_strength: float


def _detect_breakout(candles: list[Candle]) -> _BreakoutInfo | None:
    if len(candles) < MIN_CANDLES_FOR_TIMEFRAME + 1:
        return None
    prior = candles[:-1]
    latest = candles[-1]
    prior_high = max(c.high for c in prior)
    prior_low = min(c.low for c in prior)
    strength = _candle_strength(latest)
    if prior_high > 0 and latest.close > prior_high:
        distance = (latest.close - prior_high) / prior_high * 100
        if distance >= BREAKOUT_MIN_DISTANCE_PCT:
            return _BreakoutInfo("bullish", prior_high, distance, strength)
    if prior_low > 0 and latest.close < prior_low:
        distance = (prior_low - latest.close) / prior_low * 100
        if distance >= BREAKOUT_MIN_DISTANCE_PCT:
            return _BreakoutInfo("bearish", prior_low, distance, strength)
    return None


def _detect_failed_breakout(candles: list[Candle]) -> str | None:
    needed = MIN_CANDLES_FOR_TIMEFRAME + FAILED_BREAKOUT_LOOKBACK + 1
    if len(candles) < needed:
        return None
    latest = candles[-1]
    reference = candles[: -(FAILED_BREAKOUT_LOOKBACK + 1)]
    if len(reference) < MIN_CANDLES_FOR_TIMEFRAME:
        return None
    ref_high = max(c.high for c in reference)
    ref_low = min(c.low for c in reference)
    recent_window = candles[-(FAILED_BREAKOUT_LOOKBACK + 1) : -1]
    breached_high = any(c.high > ref_high for c in recent_window)
    breached_low = any(c.low < ref_low for c in recent_window)
    if breached_high and latest.close < ref_high:
        return "failed_bullish_breakout"
    if breached_low and latest.close > ref_low:
        return "failed_bearish_breakout"
    return None


def _detect_pullback(candles: list[Candle], trend: str, swings: list[Swing]) -> str | None:
    if trend not in ("bullish", "bearish"):
        return None
    window = candles[-PULLBACK_LOOKBACK:]
    if len(window) < 3:
        return None
    latest_close = candles[-1].close
    if trend == "bullish":
        swing_lows = [s[2] for s in swings if s[1] == "low"]
        if not swing_lows:
            return None
        floor = swing_lows[-1]
        recent_high = max(c.high for c in window)
        if recent_high <= 0:
            return None
        retracement = (recent_high - latest_close) / recent_high * 100
        if retracement > PULLBACK_MIN_RETRACEMENT_PCT and latest_close > floor:
            return "bullish_pullback"
        return None
    swing_highs = [s[2] for s in swings if s[1] == "high"]
    if not swing_highs:
        return None
    ceiling = swing_highs[-1]
    recent_low = min(c.low for c in window)
    if recent_low <= 0:
        return None
    retracement = (latest_close - recent_low) / recent_low * 100
    if retracement > PULLBACK_MIN_RETRACEMENT_PCT and latest_close < ceiling:
        return "bearish_pullback"
    return None


def _classify_pattern(
    candles: list[Candle], trend: str, swings: list[Swing]
) -> tuple[str, _BreakoutInfo | None, _RangeInfo | None]:
    """Checked most-specific-event-first: a failed breakout or an active
    breakout is more informative right now than a generic range/trend read."""
    failed = _detect_failed_breakout(candles)
    if failed:
        return failed, None, None

    breakout = _detect_breakout(candles)
    if breakout:
        return f"{breakout.direction}_breakout", breakout, None

    pullback = _detect_pullback(candles, trend, swings)
    if pullback:
        return pullback, None, None

    range_info = _detect_range(candles)
    if range_info and range_info.is_range:
        return "range", None, range_info

    if trend in ("bullish", "bearish"):
        return "trending", None, range_info

    return "neutral", None, range_info


def _classify_phase(volatility_expansion: float | None) -> str | None:
    if volatility_expansion is None:
        return None
    if volatility_expansion < COMPRESSION_VOLATILITY_EXPANSION_THRESHOLD:
        return "compression"
    if volatility_expansion > EXPANSION_VOLATILITY_EXPANSION_THRESHOLD:
        return "expansion"
    return "stable"


def _compute_alignment(timeframes: list[TimeframeStructure]) -> str:
    available = {t.timeframe: t.trend for t in timeframes if t.trend not in (None, "neutral")}
    if len(available) < 2:
        return "insufficient_data"
    bullish = sum(1 for d in available.values() if d == "bullish")
    bearish = sum(1 for d in available.values() if d == "bearish")
    total = len(available)
    if bullish == total:
        return "strong_bullish_alignment"
    if bearish == total:
        return "strong_bearish_alignment"
    short_tf = [available[tf] for tf in ("1m", "5m") if tf in available]
    long_tf = [available[tf] for tf in ("15m", "1h") if tf in available]
    if short_tf and long_tf:
        if all(d == "bullish" for d in short_tf) and all(d == "bearish" for d in long_tf):
            return "bullish_short_term_bearish_higher_timeframe"
        if all(d == "bearish" for d in short_tf) and all(d == "bullish" for d in long_tf):
            return "bearish_short_term_bullish_higher_timeframe"
    return "mixed"


def _directional_bias(alignment: str, primary_trend: str | None) -> str:
    if alignment == "strong_bullish_alignment":
        return "bullish"
    if alignment == "strong_bearish_alignment":
        return "bearish"
    if alignment in (
        "bullish_short_term_bearish_higher_timeframe",
        "bearish_short_term_bullish_higher_timeframe",
        "mixed",
    ):
        return "neutral"
    # insufficient_data: fall back to the primary timeframe alone, if it has
    # a directional read -- still honest, since timeframe_alignment stays
    # "insufficient_data" and makes the lack of corroboration visible.
    if primary_trend in ("bullish", "bearish"):
        return primary_trend
    return "neutral"


def _levels(candles_1m: tuple[Candle, ...], swings_1m: list[Swing]) -> Levels:
    if not candles_1m:
        return Levels(None, None, None, None)
    window = candles_1m[-RANGE_LOOKBACK:]
    recent_high = max(c.high for c in window)
    recent_low = min(c.low for c in window)
    support = None
    resistance = None
    for _, kind, price in reversed(swings_1m):
        if kind == "low" and support is None:
            support = price
        elif kind == "high" and resistance is None:
            resistance = price
        if support is not None and resistance is not None:
            break
    return Levels(recent_high=recent_high, recent_low=recent_low, support=support, resistance=resistance)


def _reason(structure: Structure, alignment: str, features: StructureFeatures) -> str:
    parts = []
    if structure.pattern:
        parts.append(structure.pattern.replace("_", " "))
    if features.higher_highs and features.higher_lows:
        parts.append("higher-high/higher-low structure")
    elif features.lower_highs and features.lower_lows:
        parts.append("lower-high/lower-low structure")
    if alignment in ("strong_bullish_alignment", "strong_bearish_alignment"):
        parts.append("aligned across timeframes")
    elif alignment == "mixed" or "higher_timeframe" in alignment:
        parts.append("conflicting timeframes")
    return ", ".join(parts) if parts else "insufficient structural evidence"


class StructureEngine:
    """Analyzes current market structure for a single symbol from its
    locally-held candle history. No API calls, no external requests."""

    def analyze(self, state: SymbolState, features: MarketFeatures | None = None) -> StructureResult:
        candles_1m = state.candle_history
        now = time.time()

        timeframe_reads: list[TimeframeStructure] = []
        per_timeframe: dict[str, tuple[list[Candle], list[Swing], str]] = {}
        for tf, minutes in TIMEFRAME_MINUTES.items():
            tf_candles = _resample(candles_1m, minutes)
            if len(tf_candles) < MIN_CANDLES_FOR_TIMEFRAME:
                timeframe_reads.append(TimeframeStructure(tf, None, len(tf_candles)))
                continue
            swings = _alternate_swings(_find_swings(tf_candles))
            trend = _classify_trend(swings)
            timeframe_reads.append(TimeframeStructure(tf, trend, len(tf_candles)))
            per_timeframe[tf] = (tf_candles, swings, trend)

        alignment = _compute_alignment(timeframe_reads)
        primary_tf = next((tf for tf in PRIMARY_TIMEFRAME_PREFERENCE if tf in per_timeframe), None)

        levels_swings = (
            _alternate_swings(_find_swings(list(candles_1m)))
            if len(candles_1m) >= MIN_CANDLES_FOR_TIMEFRAME
            else []
        )
        levels = _levels(candles_1m, levels_swings)

        if primary_tf is None:
            structure = Structure(trend=None, pattern="insufficient_data", phase=None)
            bias = _directional_bias(alignment, None)
            return StructureResult(
                symbol=state.symbol,
                timestamp=now,
                structure=structure,
                directional_bias=bias,
                confidence=0.0,
                levels=levels,
                features=StructureFeatures(None, None, None, None, None, None, None, None),
                timeframe_alignment=alignment,
                timeframes=tuple(timeframe_reads),
                reason="Not enough candle history yet to classify structure on any timeframe.",
            )

        primary_candles, primary_swings, primary_trend = per_timeframe[primary_tf]
        pattern, breakout_info, range_info = _classify_pattern(primary_candles, primary_trend, primary_swings)
        phase = _classify_phase(features.volatility.volatility_expansion if features else None)
        higher_highs, lower_highs, higher_lows, lower_lows = _swing_feature_flags(primary_swings)

        structure = Structure(trend=primary_trend, pattern=pattern, phase=phase)
        bias = _directional_bias(alignment, primary_trend)

        feature_flags = StructureFeatures(
            higher_highs=higher_highs,
            lower_highs=lower_highs,
            higher_lows=higher_lows,
            lower_lows=lower_lows,
            range_width=range_info.width_pct if range_info else None,
            breakout_distance=breakout_info.distance_pct if breakout_info else None,
            compression=(phase == "compression") if phase else None,
            expansion=(phase == "expansion") if phase else None,
        )

        confidence = self._confidence(
            primary_swings=primary_swings,
            alignment=alignment,
            features=features,
            breakout_info=breakout_info,
            primary_candle_count=len(primary_candles),
        )

        result_structure = structure
        result_features = feature_flags
        return StructureResult(
            symbol=state.symbol,
            timestamp=now,
            structure=result_structure,
            directional_bias=bias,
            confidence=confidence,
            levels=levels,
            features=result_features,
            timeframe_alignment=alignment,
            timeframes=tuple(timeframe_reads),
            reason=_reason(result_structure, alignment, result_features),
        )

    def _confidence(
        self,
        primary_swings: list[Swing],
        alignment: str,
        features: MarketFeatures | None,
        breakout_info: _BreakoutInfo | None,
        primary_candle_count: int,
    ) -> float:
        trend_clarity = min(100.0, len(primary_swings) / FULL_TREND_CLARITY_SWING_COUNT * 100)
        alignment_score = ALIGNMENT_SCORES.get(alignment, 0.0)

        vol_vol_inputs: list[tuple[float | None, float]] = []
        if features is not None:
            rel_vol = features.volume.relative_volume
            vol_exp = features.volatility.volatility_expansion
            vol_vol_inputs.append(
                (min(100.0, rel_vol / 2.0 * 100) if rel_vol is not None else None, 1.0)
            )
            vol_vol_inputs.append(
                (min(100.0, vol_exp / 2.0 * 100) if vol_exp is not None else None, 1.0)
            )
        if breakout_info is not None:
            vol_vol_inputs.append((breakout_info.candle_strength * 100, 1.0))
        volume_volatility = _weighted_avg(vol_vol_inputs)

        history_depth = min(100.0, primary_candle_count / FULL_HISTORY_DEPTH_CANDLES * 100)

        final = _weighted_avg(
            [
                (trend_clarity, CONFIDENCE_WEIGHTS["trend_clarity"]),
                (alignment_score, CONFIDENCE_WEIGHTS["alignment"]),
                (volume_volatility, CONFIDENCE_WEIGHTS["volume_volatility"]),
                (history_depth, CONFIDENCE_WEIGHTS["history_depth"]),
            ]
        )
        return final if final is not None else 0.0


def analyze_top_markets(
    scanner_results: list[ScannerResult],
    states: dict[str, SymbolState],
    engine: StructureEngine | None = None,
) -> list[tuple[ScannerResult, StructureResult]]:
    """Feeds each of the scanner's ranked results into the structure engine,
    reusing the scanner's already-computed MarketFeatures (no recomputation)."""
    engine = engine or StructureEngine()
    out = []
    for result in scanner_results:
        state = states.get(result.symbol)
        if state is None:
            continue
        out.append((result, engine.analyze(state, features=result.features)))
    return out
