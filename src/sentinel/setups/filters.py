"""
Reusable rejection-reason checks shared across setup types. Each returns
a rejection-reason string (from the fixed vocabulary the engine reports)
or None if the check passes. Pure functions, no I/O.
"""

from __future__ import annotations

from sentinel.scanner.features import MarketFeatures
from sentinel.structure.models import StructureResult

# Alignment labels that STRONGLY oppose a setup's direction -- these hard-
# reject ("Reject immediately if higher-timeframe structure strongly
# conflicts"). Labels that merely lack full corroboration ("mixed",
# "insufficient_data") do not hard-reject; they only lower the setup
# score's alignment component.
CONFLICTING_ALIGNMENTS_FOR_LONG = {
    "strong_bearish_alignment",
    "bullish_short_term_bearish_higher_timeframe",
}
CONFLICTING_ALIGNMENTS_FOR_SHORT = {
    "strong_bullish_alignment",
    "bearish_short_term_bullish_higher_timeframe",
}


def check_sufficient_history(structure_result: StructureResult) -> str | None:
    if structure_result.structure.trend is None or structure_result.structure.pattern == "insufficient_data":
        return "insufficient_history"
    return None


def check_timeframe_conflict(direction: str, alignment: str) -> str | None:
    conflicting = CONFLICTING_ALIGNMENTS_FOR_LONG if direction == "long" else CONFLICTING_ALIGNMENTS_FOR_SHORT
    if alignment in conflicting:
        return "conflicting_timeframes"
    return None


def check_pattern_matches(structure_result: StructureResult, expected_patterns: set[str], not_confirmed_reason: str) -> str | None:
    if structure_result.structure.pattern not in expected_patterns:
        return not_confirmed_reason
    return None


def check_not_range_bound(structure_result: StructureResult) -> str | None:
    if structure_result.structure.pattern == "range":
        return "range_bound"
    return None


def check_volume_confirmation(features: MarketFeatures | None, min_relative_volume: float) -> str | None:
    rv = features.volume.relative_volume if features is not None else None
    if rv is None or rv < min_relative_volume:
        return "inadequate_volume"
    return None


def check_momentum_confirmation(features: MarketFeatures | None, direction: str, min_abs_return_pct: float) -> str | None:
    """Uses the best (longest) available return timeframe only -- 15m
    preferred, falling back to 5m then 1m. Does not cherry-pick among
    multiple timeframes until one passes."""
    if features is None:
        return "inadequate_momentum"
    for r in (features.returns.return_15m, features.returns.return_5m, features.returns.return_1m):
        if r is None:
            continue
        if direction == "long" and r >= min_abs_return_pct:
            return None
        if direction == "short" and r <= -min_abs_return_pct:
            return None
        return "inadequate_momentum"
    return "inadequate_momentum"


def check_structural_level(level: float | None) -> str | None:
    if level is None:
        return "no_structural_level"
    return None


def check_valid_stop(current_price: float, stop_loss: float | None, direction: str) -> str | None:
    if stop_loss is None:
        return "invalid_stop"
    if direction == "long" and stop_loss >= current_price:
        return "invalid_stop"
    if direction == "short" and stop_loss <= current_price:
        return "invalid_stop"
    return None


def check_valid_target(current_price: float, take_profit_1: float | None, direction: str) -> str | None:
    if take_profit_1 is None:
        return "invalid_target"
    if direction == "long" and take_profit_1 <= current_price:
        return "invalid_target"
    if direction == "short" and take_profit_1 >= current_price:
        return "invalid_target"
    return None


def check_risk_reward(risk_reward_tp1: float | None, min_rr: float) -> str | None:
    if risk_reward_tp1 is None or risk_reward_tp1 < min_rr:
        return "poor_risk_reward"
    return None
