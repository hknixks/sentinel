"""
Typed structures describing current market structure. Phase 3 output is
descriptive, not predictive: it does not generate trade signals, entries,
stops, or targets, and `confidence` is structural evidence strength, not
a probability of trade success.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Structure:
    trend: str | None  # "bullish" | "bearish" | "neutral" | None (insufficient data)
    pattern: str | None  # e.g. "bullish_breakout", "range", "trending", "insufficient_data"
    phase: str | None  # "compression" | "expansion" | "stable" | None


@dataclass(frozen=True)
class Levels:
    recent_high: float | None
    recent_low: float | None
    support: float | None
    resistance: float | None


@dataclass(frozen=True)
class StructureFeatures:
    higher_highs: bool | None
    lower_highs: bool | None
    higher_lows: bool | None
    lower_lows: bool | None
    range_width: float | None
    breakout_distance: float | None
    compression: bool | None
    expansion: bool | None


@dataclass(frozen=True)
class TimeframeStructure:
    """Per-timeframe trend read, used to build multi-timeframe alignment."""

    timeframe: str
    trend: str | None  # "bullish" | "bearish" | "neutral" | None (insufficient data)
    candle_count: int


@dataclass(frozen=True)
class StructureResult:
    symbol: str
    timestamp: float

    structure: Structure
    directional_bias: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0-100 structural confidence -- NOT trade probability
    levels: Levels
    features: StructureFeatures
    timeframe_alignment: str
    timeframes: tuple[TimeframeStructure, ...]  # 1m/5m/15m/1h detail
    reason: str
