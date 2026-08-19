"""
Stop-loss, take-profit, and risk/reward calculations. Stops always come
from real structural levels (never an arbitrary fixed percentage of
price). Take-profits prefer real opposing structural levels; when none
exist ahead of price, they fall back to a disclosed risk-multiple
projection -- a math derivation from the already-computed risk amount,
not fabricated market data.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.structure.models import Levels

# Risk-multiple used only when no real opposing structural level exists
# ahead of price. 1.5R / 3.0R are conventional, conservative defaults for
# a first/second target respectively.
TP1_PROJECTION_RR = 1.5
TP2_PROJECTION_RR = 3.0


@dataclass(frozen=True)
class TakeProfits:
    tp1: float
    tp1_from_structure: bool
    tp2: float | None
    tp2_from_structure: bool


def compute_take_profits(direction: str, current_price: float, risk_per_unit: float, levels: Levels) -> TakeProfits:
    if direction == "long":
        candidates = sorted(
            v for v in (levels.recent_high, levels.resistance) if v is not None and v > current_price
        )
    else:
        candidates = sorted(
            (v for v in (levels.recent_low, levels.support) if v is not None and v < current_price),
            reverse=True,
        )

    if candidates:
        tp1 = candidates[0]
        tp1_from_structure = True
    else:
        tp1 = (
            current_price + risk_per_unit * TP1_PROJECTION_RR
            if direction == "long"
            else current_price - risk_per_unit * TP1_PROJECTION_RR
        )
        tp1_from_structure = False

    # TP2 prefers a farther real structural level beyond TP1; otherwise it's
    # projected beyond TP1 (never behind it, regardless of which branch TP1
    # came from) using the risk-multiple fallback.
    farther = [c for c in candidates if (c > tp1 if direction == "long" else c < tp1)]
    if farther:
        tp2 = farther[-1] if direction == "long" else farther[-1]
        tp2_from_structure = True
    else:
        projected = (
            current_price + risk_per_unit * TP2_PROJECTION_RR
            if direction == "long"
            else current_price - risk_per_unit * TP2_PROJECTION_RR
        )
        tp2 = max(projected, tp1 + risk_per_unit) if direction == "long" else min(projected, tp1 - risk_per_unit)
        tp2_from_structure = False

    return TakeProfits(tp1=tp1, tp1_from_structure=tp1_from_structure, tp2=tp2, tp2_from_structure=tp2_from_structure)


def risk_per_unit(direction: str, current_price: float, stop_loss: float) -> float:
    if direction == "long":
        return current_price - stop_loss
    return stop_loss - current_price


def reward_and_rr(current_price: float, target: float | None, risk: float) -> tuple[float | None, float | None]:
    if target is None or risk <= 0:
        return None, None
    reward = abs(target - current_price)
    return reward, reward / risk
