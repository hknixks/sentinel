"""
Market scanner: ranks markets by transparent, measurable features -- not
a fabricated "prediction score". Raw feature values are converted to
percentile ranks (0-100) within the current scan batch, since a fixed
threshold (e.g. "1m return > 2%") means wildly different things for BTC
vs. a micro-cap altcoin. Percentiles self-calibrate across whatever
universe is being scanned. Percentile components are then combined with
fixed, explained weights into a single 0-100 score.

This module only describes current market conditions (bullish / bearish /
neutral). It does not generate trade signals, entries, stops, or targets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sentinel.market_state import SymbolState
from sentinel.scanner.features import FeatureEngine, MarketFeatures

# A symbol needs at least this many closed candles to be scanned at all --
# enough for a 15m return, the practical minimum for a meaningful momentum
# read. Symbols below this are excluded from ranking, never padded.
MIN_CANDLES_FOR_SCANNING = 16

# Momentum: percentile-rank |return| per timeframe, then blend. 15m weighted
# highest as the best balance between real signal and single-candle noise;
# 1h included for durability context but weighted lowest since the scanner
# cares about what's moving *now*.
MOMENTUM_TIMEFRAME_WEIGHTS: dict[str, float] = {"1m": 0.15, "5m": 0.30, "15m": 0.35, "1h": 0.20}

# Volume: relative volume (current vs. baseline) is the primary "is this
# unusual" signal; acceleration adds recency but shouldn't dominate.
VOLUME_METRIC_WEIGHTS: dict[str, float] = {"relative_volume": 0.6, "volume_acceleration": 0.4}

# Volatility: expansion (increasing vs. baseline) signals a *developing*
# opportunity; absolute level alone doesn't indicate change and already
# correlates with momentum, so it's weighted lower.
VOLATILITY_METRIC_WEIGHTS: dict[str, float] = {
    "volatility_expansion": 0.7,
    "rolling_volatility": 0.3,
}

# Trend: MA-spread strength is the primary structural signal; timeframe
# alignment confirms momentum agrees across horizons, reducing single-
# timeframe false positives.
TREND_METRIC_WEIGHTS: dict[str, float] = {"trend_strength": 0.6, "timeframe_alignment": 0.4}

# Final score = weighted blend of the 4 components. Momentum and volume
# dominate (they're the direct evidence of "unusual activity"); volatility
# and trend are supporting/confirming signals, weighted lower to avoid
# double-counting and because they need the most history to be reliable.
COMPONENT_WEIGHTS: dict[str, float] = {
    "momentum": 0.35,
    "volume": 0.30,
    "volatility": 0.20,
    "trend": 0.15,
}


@dataclass(frozen=True)
class ScannerResult:
    symbol: str
    score: float
    direction: str  # "bullish" | "bearish" | "neutral" -- a market-condition label, not a trade signal
    features: MarketFeatures
    timestamp: float


def _weighted_avg(pairs: list[tuple[float | None, float]]) -> float | None:
    """Weighted average over available (non-None) values, weights renormalized."""
    available = [(v, w) for v, w in pairs if v is not None]
    if not available:
        return None
    total_weight = sum(w for _, w in available)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in available) / total_weight


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    """Rank each symbol's value 0-100 within the given set. Ties share the average rank."""
    if not values:
        return {}
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 50.0}
    ranks: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg_rank = (i + j) / 2
        percentile = avg_rank / (n - 1) * 100
        for k in range(i, j + 1):
            ranks[items[k][0]] = percentile
        i = j + 1
    return ranks


def _map_direction(trend_direction: str | None) -> str:
    if trend_direction == "up":
        return "bullish"
    if trend_direction == "down":
        return "bearish"
    return "neutral"


class MarketScanner:
    """Ranks markets with sufficient data by transparent, measurable features."""

    def __init__(self, feature_engine: FeatureEngine | None = None) -> None:
        self._feature_engine = feature_engine or FeatureEngine()

    def scan(
        self, states: dict[str, SymbolState], top_n: int | None = None
    ) -> list[ScannerResult]:
        eligible: dict[str, MarketFeatures] = {}
        for symbol, state in states.items():
            features = self._feature_engine.compute(state)
            if features is None:
                continue
            if features.candle_count < MIN_CANDLES_FOR_SCANNING:
                continue
            if features.returns.return_15m is None:
                continue
            eligible[symbol] = features

        if not eligible:
            return []

        scores = self._score_all(eligible)
        now = time.time()

        results = [
            ScannerResult(
                symbol=symbol,
                score=scores[symbol],
                direction=_map_direction(features.trend.direction),
                features=features,
                timestamp=now,
            )
            for symbol, features in eligible.items()
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n] if top_n is not None else results

    def _score_all(self, eligible: dict[str, MarketFeatures]) -> dict[str, float]:
        momentum_percentiles = {
            tf: _percentile_ranks(
                {
                    s: abs(getattr(f.returns, f"return_{tf}"))
                    for s, f in eligible.items()
                    if getattr(f.returns, f"return_{tf}") is not None
                }
            )
            for tf in MOMENTUM_TIMEFRAME_WEIGHTS
        }
        volume_percentiles = {
            metric: _percentile_ranks(
                {
                    s: getattr(f.volume, metric)
                    for s, f in eligible.items()
                    if getattr(f.volume, metric) is not None
                }
            )
            for metric in VOLUME_METRIC_WEIGHTS
        }
        volatility_percentiles = {
            metric: _percentile_ranks(
                {
                    s: getattr(f.volatility, metric)
                    for s, f in eligible.items()
                    if getattr(f.volatility, metric) is not None
                }
            )
            for metric in VOLATILITY_METRIC_WEIGHTS
        }
        trend_percentiles = {
            metric: _percentile_ranks(
                {
                    s: getattr(f.trend, metric)
                    for s, f in eligible.items()
                    if getattr(f.trend, metric) is not None
                }
            )
            for metric in TREND_METRIC_WEIGHTS
        }

        scores: dict[str, float] = {}
        for symbol in eligible:
            momentum = _weighted_avg(
                [
                    (momentum_percentiles[tf].get(symbol), w)
                    for tf, w in MOMENTUM_TIMEFRAME_WEIGHTS.items()
                ]
            )
            volume = _weighted_avg(
                [
                    (volume_percentiles[m].get(symbol), w)
                    for m, w in VOLUME_METRIC_WEIGHTS.items()
                ]
            )
            volatility = _weighted_avg(
                [
                    (volatility_percentiles[m].get(symbol), w)
                    for m, w in VOLATILITY_METRIC_WEIGHTS.items()
                ]
            )
            trend = _weighted_avg(
                [(trend_percentiles[m].get(symbol), w) for m, w in TREND_METRIC_WEIGHTS.items()]
            )

            final = _weighted_avg(
                [
                    (momentum, COMPONENT_WEIGHTS["momentum"]),
                    (volume, COMPONENT_WEIGHTS["volume"]),
                    (volatility, COMPONENT_WEIGHTS["volatility"]),
                    (trend, COMPONENT_WEIGHTS["trend"]),
                ]
            )
            # momentum's 15m timeframe is guaranteed present by the scan() gate,
            # so final is never None for an eligible symbol; the fallback is
            # purely defensive.
            scores[symbol] = final if final is not None else 0.0

        return scores
