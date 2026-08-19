"""
Typed structures describing candidate trade setups.

SENTINEL is alert-only: nothing here places, sizes, or executes a trade.
`setup_score` is a deterministic measure of how much structural/technical
evidence supports a candidate -- it is NOT a probability of winning, NOT
an expected return, and NOT a guarantee. Two setups scored 90 and 40 are
not "90% and 40% likely to win"; the score only says the 90 has more
corroborating evidence by this engine's transparent rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntryZone:
    low: float
    high: float


@dataclass(frozen=True)
class RiskLevels:
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    risk_per_unit: float
    reward_to_tp1: float
    reward_to_tp2: float | None
    risk_reward_tp1: float
    risk_reward_tp2: float | None


@dataclass(frozen=True)
class SetupCandidate:
    symbol: str
    timestamp: float
    direction: str  # "long" | "short"
    setup_type: str  # "breakout" | "pullback" | "trend_continuation"
    entry_zone: EntryZone
    invalidation_level: float
    risk: RiskLevels
    structure_context: str
    confirmation_factors: tuple[str, ...]
    setup_score: float  # 0-100 -- evidence strength, NOT a win probability


@dataclass(frozen=True)
class RejectedCandidate:
    symbol: str
    setup_type: str
    direction: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    symbol: str
    timestamp: float
    candidates: tuple[SetupCandidate, ...]
    rejections: tuple[RejectedCandidate, ...]
