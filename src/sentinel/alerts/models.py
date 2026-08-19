"""
Alert lifecycle models. An AlertRecord tracks ONE setup's alert lifecycle
end-to-end.

`status` tracks the underlying SETUP's own validity in the market:
ACTIVE while it is still the current setup for its symbol, RESOLVED once
it is superseded by a new setup identity or disappears entirely.

`decision` is an independent dimension -- the user's stated TAKEN/SKIPPED
choice via the Telegram buttons -- and persists regardless of what
`status` later becomes. A resolved setup's decision is not overwritten
or erased.

This is alert-only bookkeeping. SENTINEL never executes, sizes, or
manages a real trade or wallet; TAKEN/SKIPPED only records what the user
says they did.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUS_ACTIVE = "active"
STATUS_RESOLVED = "resolved"

DECISION_TAKEN = "taken"
DECISION_SKIPPED = "skipped"


@dataclass(frozen=True)
class AlertRecord:
    alert_id: str
    symbol: str
    direction: str
    setup_type: str
    status: str

    setup_score: float
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    risk_reward_tp1: float
    invalidation_level: float
    structure_context: str
    confirmation_factors: str  # "|"-joined for simple storage
    reasoning: str  # precomputed deterministic Deeper Reasoning report

    created_at: float
    resolved_at: float | None

    telegram_message_id: int | None
    telegram_chat_id: str | None

    decision: str | None
    decision_at: float | None
    decision_reference_price: float | None  # market price at moment of decision
