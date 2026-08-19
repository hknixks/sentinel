"""
Trade Setup Engine: consumes MarketState, MarketFeatures, ScannerResult,
and StructureResult to produce structured, deterministic trade-setup
CANDIDATES. This is alert-only market analysis -- it never places,
sizes, times, or executes a trade, and never connects to a wallet.

`setup_score` is evidence strength, not a probability. A setup existing
here is not a recommendation and never a guarantee of profit.
"""

from __future__ import annotations

import dataclasses
import time

from sentinel.config import (
    SETUP_MIN_BREAKOUT_MOMENTUM_PCT,
    SETUP_MIN_BREAKOUT_RELATIVE_VOLUME,
    SETUP_MIN_RISK_REWARD,
)
from sentinel.market_state import SymbolState
from sentinel.scanner.features import MarketFeatures
from sentinel.scanner.scanner import ScannerResult
from sentinel.setups import entry, filters, risk
from sentinel.setups.models import EntryZone, EvaluationResult, RejectedCandidate, RiskLevels, SetupCandidate
from sentinel.structure.models import StructureResult

BREAKOUT_PATTERNS = {"long": {"bullish_breakout"}, "short": {"bearish_breakout"}}
PULLBACK_PATTERNS = {"long": {"bullish_pullback"}, "short": {"bearish_pullback"}}
DIRECTION_TO_TREND = {"long": "bullish", "short": "bearish"}

# Setup-quality score weights (sum to 1.0). Structure quality weighted
# highest -- it reuses Phase 3's own structural-confidence work directly,
# the foundation everything else builds on. Alignment, momentum, and
# volume are classic corroborating signals, weighted equally. Risk/reward
# quality matters almost as much (a technically clean setup with poor R:R
# is still a poor setup). Volatility state and entry tightness are
# secondary/supporting, weighted lowest.
SCORE_WEIGHTS = {
    "structure_quality": 0.20,
    "timeframe_alignment": 0.15,
    "momentum_confirmation": 0.15,
    "volume_confirmation": 0.15,
    "volatility_state": 0.10,
    "entry_quality": 0.10,
    "risk_reward_quality": 0.15,
}
ALIGNMENT_SCORE_MAP: dict[str, float] = {
    "strong_bullish_alignment": 100.0,
    "strong_bearish_alignment": 100.0,
    "bullish_short_term_bearish_higher_timeframe": 40.0,
    "bearish_short_term_bullish_higher_timeframe": 40.0,
    "mixed": 30.0,
    "insufficient_data": 0.0,
}
RR_FULL_SCORE_TARGET = 3.0


def _weighted_avg(pairs: list[tuple[float | None, float]]) -> float | None:
    available = [(v, w) for v, w in pairs if v is not None]
    if not available:
        return None
    total_weight = sum(w for _, w in available)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in available) / total_weight


def _current_price(state: SymbolState) -> float | None:
    if not state.candle_history:
        return None
    return state.candle_history[-1].close


def _momentum_score(features: MarketFeatures | None, direction: str) -> float | None:
    if features is None:
        return None
    values = [
        features.returns.return_1m,
        features.returns.return_5m,
        features.returns.return_15m,
        features.returns.return_1h,
    ]
    available = [v for v in values if v is not None]
    if not available:
        return None
    matches = sum(1 for v in available if (v > 0) == (direction == "long"))
    return matches / len(available) * 100


def _volume_score(features: MarketFeatures | None) -> float | None:
    if features is None or features.volume.relative_volume is None:
        return None
    return min(100.0, features.volume.relative_volume / 2.0 * 100)


def _volatility_score(features: MarketFeatures | None) -> float | None:
    if features is None or features.volatility.volatility_expansion is None:
        return None
    return min(100.0, features.volatility.volatility_expansion / 2.0 * 100)


def _entry_quality_score(entry_zone: EntryZone, current_price: float) -> float | None:
    if current_price <= 0:
        return None
    width_pct = abs(entry_zone.high - entry_zone.low) / current_price * 100
    return max(0.0, 100.0 - width_pct * 50.0)


def _rr_quality_score(risk_levels: RiskLevels) -> float | None:
    if risk_levels.risk_reward_tp1 is None:
        return None
    return min(100.0, risk_levels.risk_reward_tp1 / RR_FULL_SCORE_TARGET * 100)


def _compute_setup_score(
    structure_result: StructureResult,
    features: MarketFeatures | None,
    direction: str,
    entry_zone: EntryZone,
    current_price: float,
    risk_levels: RiskLevels,
) -> float:
    components = [
        (structure_result.confidence, SCORE_WEIGHTS["structure_quality"]),
        (ALIGNMENT_SCORE_MAP.get(structure_result.timeframe_alignment, 0.0), SCORE_WEIGHTS["timeframe_alignment"]),
        (_momentum_score(features, direction), SCORE_WEIGHTS["momentum_confirmation"]),
        (_volume_score(features), SCORE_WEIGHTS["volume_confirmation"]),
        (_volatility_score(features), SCORE_WEIGHTS["volatility_state"]),
        (_entry_quality_score(entry_zone, current_price), SCORE_WEIGHTS["entry_quality"]),
        (_rr_quality_score(risk_levels), SCORE_WEIGHTS["risk_reward_quality"]),
    ]
    result = _weighted_avg(components)
    return result if result is not None else 0.0


def _build_risk_levels(direction: str, current_price: float, stop_loss: float, levels) -> RiskLevels | None:
    risk_amount = risk.risk_per_unit(direction, current_price, stop_loss)
    if risk_amount <= 0:
        return None
    tps = risk.compute_take_profits(direction, current_price, risk_amount, levels)
    reward_tp1, rr_tp1 = risk.reward_and_rr(current_price, tps.tp1, risk_amount)
    reward_tp2, rr_tp2 = risk.reward_and_rr(current_price, tps.tp2, risk_amount)
    return RiskLevels(
        stop_loss=stop_loss,
        take_profit_1=tps.tp1,
        take_profit_2=tps.tp2,
        risk_per_unit=risk_amount,
        reward_to_tp1=reward_tp1,
        reward_to_tp2=reward_tp2,
        risk_reward_tp1=rr_tp1,
        risk_reward_tp2=rr_tp2,
    )


class SetupEngine:
    """Evaluates one symbol against all supported setup types."""

    def evaluate(
        self,
        symbol: str,
        state: SymbolState,
        features: MarketFeatures | None,
        scanner_result: ScannerResult | None,
        structure_result: StructureResult,
    ) -> EvaluationResult:
        now = time.time()
        candidates: list[SetupCandidate] = []
        rejections: list[RejectedCandidate] = []

        history_reason = filters.check_sufficient_history(structure_result)
        current_price = _current_price(state)
        if history_reason or current_price is None:
            reason = history_reason or "insufficient_history"
            for setup_type, direction in (
                ("breakout", "long"),
                ("breakout", "short"),
                ("pullback", "long"),
                ("pullback", "short"),
                ("trend_continuation", "long"),
                ("trend_continuation", "short"),
            ):
                rejections.append(RejectedCandidate(symbol, setup_type, direction, (reason,)))
            return EvaluationResult(symbol, now, tuple(candidates), tuple(rejections))

        for direction in ("long", "short"):
            for setup_type, evaluator in (
                ("breakout", self._evaluate_breakout),
                ("pullback", self._evaluate_pullback),
                ("trend_continuation", self._evaluate_trend_continuation),
            ):
                result = evaluator(symbol, direction, current_price, features, structure_result, now)
                if isinstance(result, SetupCandidate):
                    if scanner_result is not None:
                        result = dataclasses.replace(
                            result,
                            confirmation_factors=(
                                *result.confirmation_factors,
                                f"scanner activity score: {scanner_result.score:.1f} ({scanner_result.direction})",
                            ),
                        )
                    candidates.append(result)
                else:
                    rejections.append(result)

        return EvaluationResult(symbol, now, tuple(candidates), tuple(rejections))

    def _evaluate_breakout(self, symbol, direction, current_price, features, structure_result, now):
        setup_type = "breakout"
        reason = filters.check_pattern_matches(structure_result, BREAKOUT_PATTERNS[direction], "breakout_not_confirmed")
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        distance_pct = structure_result.features.breakout_distance
        if distance_pct is None or distance_pct <= 0:
            return RejectedCandidate(symbol, setup_type, direction, ("breakout_not_confirmed",))

        boundary = (
            current_price / (1 + distance_pct / 100)
            if direction == "long"
            else current_price / (1 - distance_pct / 100)
        )
        if boundary <= 0:
            return RejectedCandidate(symbol, setup_type, direction, ("breakout_not_confirmed",))

        reason = filters.check_timeframe_conflict(direction, structure_result.timeframe_alignment)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        reason = filters.check_volume_confirmation(features, SETUP_MIN_BREAKOUT_RELATIVE_VOLUME)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        reason = filters.check_momentum_confirmation(features, direction, SETUP_MIN_BREAKOUT_MOMENTUM_PCT)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        return self._finalize(
            symbol,
            setup_type,
            direction,
            current_price,
            stop_loss=boundary,
            entry_zone=entry.breakout_entry_zone(direction, current_price, boundary),
            structure_result=structure_result,
            features=features,
            now=now,
            structure_context=f"{direction}_breakout beyond {boundary:.6g} ({distance_pct:.2f}% close beyond boundary)",
            confirmation_factors=(
                f"confirmed close beyond structural boundary ({distance_pct:.2f}% distance)",
                f"relative volume {features.volume.relative_volume:.2f}x baseline",
                f"timeframe alignment: {structure_result.timeframe_alignment}",
            ),
        )

    def _evaluate_pullback(self, symbol, direction, current_price, features, structure_result, now):
        setup_type = "pullback"
        reason = filters.check_pattern_matches(structure_result, PULLBACK_PATTERNS[direction], "pullback_not_confirmed")
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        levels = structure_result.levels
        structural_level = levels.support if direction == "long" else levels.resistance
        reason = filters.check_structural_level(structural_level)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        stop_reference = levels.recent_low if direction == "long" else levels.recent_high
        reason = filters.check_structural_level(stop_reference)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))
        if direction == "long" and stop_reference >= structural_level:
            return RejectedCandidate(symbol, setup_type, direction, ("invalid_stop",))
        if direction == "short" and stop_reference <= structural_level:
            return RejectedCandidate(symbol, setup_type, direction, ("invalid_stop",))

        reason = filters.check_timeframe_conflict(direction, structure_result.timeframe_alignment)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        return self._finalize(
            symbol,
            setup_type,
            direction,
            current_price,
            stop_loss=stop_reference,
            entry_zone=entry.pullback_entry_zone(direction, current_price, structural_level),
            structure_result=structure_result,
            features=features,
            now=now,
            structure_context=f"{direction}_pullback toward structural level {structural_level:.6g}",
            confirmation_factors=(
                "existing directional structure confirmed by Phase 3 pattern classification",
                f"retracement toward {'support' if direction == 'long' else 'resistance'} {structural_level:.6g}",
                f"timeframe alignment: {structure_result.timeframe_alignment}",
            ),
        )

    def _evaluate_trend_continuation(self, symbol, direction, current_price, features, structure_result, now):
        setup_type = "trend_continuation"
        reason = filters.check_not_range_bound(structure_result)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))
        if structure_result.structure.pattern != "trending" or structure_result.structure.trend != DIRECTION_TO_TREND[direction]:
            return RejectedCandidate(symbol, setup_type, direction, ("trend_not_confirmed",))

        levels = structure_result.levels
        structural_level = levels.support if direction == "long" else levels.resistance
        reason = filters.check_structural_level(structural_level)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        reason = filters.check_timeframe_conflict(direction, structure_result.timeframe_alignment)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        recent_range_pct = features.volatility.recent_range if features else None
        return self._finalize(
            symbol,
            setup_type,
            direction,
            current_price,
            stop_loss=structural_level,
            entry_zone=entry.trend_continuation_entry_zone(direction, current_price, recent_range_pct),
            structure_result=structure_result,
            features=features,
            now=now,
            structure_context=f"{direction} trend continuation, structure trend={structure_result.structure.trend}",
            confirmation_factors=(
                "established directional structure (Phase 3 'trending' pattern, not range-bound)",
                f"timeframe alignment: {structure_result.timeframe_alignment}",
            ),
        )

    def _finalize(
        self,
        symbol,
        setup_type,
        direction,
        current_price,
        stop_loss,
        entry_zone,
        structure_result,
        features,
        now,
        structure_context,
        confirmation_factors,
    ):
        reason = filters.check_valid_stop(current_price, stop_loss, direction)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        risk_levels = _build_risk_levels(direction, current_price, stop_loss, structure_result.levels)
        if risk_levels is None:
            return RejectedCandidate(symbol, setup_type, direction, ("invalid_stop",))

        reason = filters.check_valid_target(current_price, risk_levels.take_profit_1, direction)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        reason = filters.check_risk_reward(risk_levels.risk_reward_tp1, SETUP_MIN_RISK_REWARD)
        if reason:
            return RejectedCandidate(symbol, setup_type, direction, (reason,))

        score = _compute_setup_score(structure_result, features, direction, entry_zone, current_price, risk_levels)

        return SetupCandidate(
            symbol=symbol,
            timestamp=now,
            direction=direction,
            setup_type=setup_type,
            entry_zone=entry_zone,
            invalidation_level=stop_loss,
            risk=risk_levels,
            structure_context=structure_context,
            confirmation_factors=confirmation_factors,
            setup_score=score,
        )
