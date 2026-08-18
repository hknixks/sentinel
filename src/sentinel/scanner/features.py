"""
Feature engine: turns raw per-symbol market state (candle history) into a
typed set of measurable features -- momentum, volume, volatility, and
trend. Pure calculation, no I/O, no scoring/ranking (that's the scanner's
job). Every value is either a real number computed from actual candle
data, or None when there isn't enough history yet -- nothing is
fabricated.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from sentinel.market_state import Candle, SymbolState

# Return lookback windows, in closed 1m candles.
RETURN_TIMEFRAMES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}

ROLLING_VOLUME_WINDOW = 20
RELATIVE_VOLUME_RECENT_WINDOW = 5
RELATIVE_VOLUME_BASELINE_WINDOW = 20
VOLUME_ACCEL_WINDOW = 5

ROLLING_VOLATILITY_WINDOW = 20
VOLATILITY_EXPANSION_RECENT_WINDOW = 5

TREND_SHORT_MA_WINDOW = 5
TREND_LONG_MA_WINDOW = 20
TREND_FLAT_EPSILON = 1e-9  # below this, short/long MA are considered equal ("flat")


@dataclass(frozen=True)
class ReturnFeatures:
    return_1m: float | None
    return_5m: float | None
    return_15m: float | None
    return_1h: float | None


@dataclass(frozen=True)
class VolumeFeatures:
    volume_current: float | None
    rolling_volume: float | None
    relative_volume: float | None
    volume_acceleration: float | None


@dataclass(frozen=True)
class VolatilityFeatures:
    recent_range: float | None
    rolling_volatility: float | None
    volatility_expansion: float | None


@dataclass(frozen=True)
class TrendFeatures:
    direction: str | None  # "up" | "down" | "flat" | None (not enough history)
    trend_strength: float | None
    timeframe_alignment: float | None


@dataclass(frozen=True)
class MarketFeatures:
    symbol: str
    timestamp: float
    candle_count: int
    returns: ReturnFeatures
    volume: VolumeFeatures
    volatility: VolatilityFeatures
    trend: TrendFeatures


def _calc_return(closes: list[float], periods_ago: int) -> float | None:
    if len(closes) < periods_ago + 1:
        return None
    base = closes[-(periods_ago + 1)]
    if base == 0:
        return None
    return (closes[-1] - base) / base * 100


def _rolling_volume(candles: tuple[Candle, ...], window: int) -> float | None:
    if not candles:
        return None
    return sum(c.volume for c in candles[-window:])


def _relative_volume(
    candles: tuple[Candle, ...], recent_window: int, baseline_window: int
) -> float | None:
    if len(candles) < baseline_window:
        return None
    recent_avg = statistics.fmean(c.volume for c in candles[-recent_window:])
    baseline_avg = statistics.fmean(c.volume for c in candles[-baseline_window:])
    if baseline_avg == 0:
        return None
    return recent_avg / baseline_avg


def _volume_acceleration(candles: tuple[Candle, ...], window: int) -> float | None:
    if len(candles) < window * 2:
        return None
    recent_avg = statistics.fmean(c.volume for c in candles[-window:])
    prior_avg = statistics.fmean(c.volume for c in candles[-window * 2 : -window])
    if prior_avg == 0:
        return None
    return (recent_avg - prior_avg) / prior_avg * 100


def _recent_range(candles: tuple[Candle, ...]) -> float | None:
    if not candles:
        return None
    last = candles[-1]
    if last.open == 0:
        return None
    return (last.high - last.low) / last.open * 100


def _stdev_pct_returns(closes: list[float]) -> float | None:
    """Population stdev of consecutive close-to-close % returns, as a percentage."""
    returns = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev == 0:
            continue
        returns.append((closes[i] - prev) / prev)
    if len(returns) < 2:
        return None
    return statistics.pstdev(returns) * 100


def _rolling_volatility(closes: list[float], window: int) -> float | None:
    if len(closes) < window + 1:
        return None
    return _stdev_pct_returns(closes[-(window + 1) :])


def _volatility_expansion(
    closes: list[float], recent_window: int, baseline_window: int
) -> float | None:
    if len(closes) < baseline_window + 1:
        return None
    baseline_vol = _stdev_pct_returns(closes[-(baseline_window + 1) :])
    recent_vol = _stdev_pct_returns(closes[-(recent_window + 1) :])
    if not baseline_vol or recent_vol is None:
        return None
    return recent_vol / baseline_vol


def _timeframe_alignment(returns: ReturnFeatures, direction: str | None) -> float | None:
    if direction not in ("up", "down"):
        return None
    expect_positive = direction == "up"
    values = [returns.return_1m, returns.return_5m, returns.return_15m, returns.return_1h]
    available = [r for r in values if r is not None and r != 0]
    if len(available) < 2:
        return None
    matches = sum(1 for r in available if (r > 0) == expect_positive)
    return matches / len(available) * 100


def _calc_trend(closes: list[float], returns: ReturnFeatures) -> TrendFeatures:
    if len(closes) < TREND_LONG_MA_WINDOW:
        return TrendFeatures(direction=None, trend_strength=None, timeframe_alignment=None)

    short_ma = statistics.fmean(closes[-TREND_SHORT_MA_WINDOW:])
    long_ma = statistics.fmean(closes[-TREND_LONG_MA_WINDOW:])
    if long_ma == 0:
        return TrendFeatures(direction=None, trend_strength=None, timeframe_alignment=None)

    diff_pct = (short_ma - long_ma) / long_ma * 100
    if diff_pct > TREND_FLAT_EPSILON:
        direction = "up"
    elif diff_pct < -TREND_FLAT_EPSILON:
        direction = "down"
    else:
        direction = "flat"

    return TrendFeatures(
        direction=direction,
        trend_strength=abs(diff_pct),
        timeframe_alignment=_timeframe_alignment(returns, direction),
    )


class FeatureEngine:
    """Computes MarketFeatures from a single symbol's live market state."""

    def compute(self, state: SymbolState) -> MarketFeatures | None:
        candles = state.candle_history
        if not candles:
            return None

        closes = [c.close for c in candles]

        returns = ReturnFeatures(
            **{
                f"return_{label}": _calc_return(closes, periods)
                for label, periods in RETURN_TIMEFRAMES.items()
            }
        )
        volume = VolumeFeatures(
            volume_current=candles[-1].volume,
            rolling_volume=_rolling_volume(candles, ROLLING_VOLUME_WINDOW),
            relative_volume=_relative_volume(
                candles, RELATIVE_VOLUME_RECENT_WINDOW, RELATIVE_VOLUME_BASELINE_WINDOW
            ),
            volume_acceleration=_volume_acceleration(candles, VOLUME_ACCEL_WINDOW),
        )
        volatility = VolatilityFeatures(
            recent_range=_recent_range(candles),
            rolling_volatility=_rolling_volatility(closes, ROLLING_VOLATILITY_WINDOW),
            volatility_expansion=_volatility_expansion(
                closes, VOLATILITY_EXPANSION_RECENT_WINDOW, ROLLING_VOLATILITY_WINDOW
            ),
        )
        trend = _calc_trend(closes, returns)

        return MarketFeatures(
            symbol=state.symbol,
            timestamp=time.time(),
            candle_count=len(candles),
            returns=returns,
            volume=volume,
            volatility=volatility,
            trend=trend,
        )
