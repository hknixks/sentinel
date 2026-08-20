"""
Outcome tracking models.

A SignalRecord is an IMMUTABLE snapshot of a validated setup at the
moment its alert was created: entry, stop, targets, score, and structure
context exactly as they existed then. Nothing about the original signal
is ever recomputed or overwritten later from current market data.

Only the separate outcome-measurement fields (outcome_state, hit
timestamps, resolved_at, last_evaluated_at, and the per-window forward
metrics in outcome_windows) evolve as real market time passes -- that
evolution is the entire point of this module, and is not a violation of
"immutability": the ORIGINAL characterization of the setup never changes,
only the record of WHAT HAPPENED AFTERWARD.
"""

from __future__ import annotations

from dataclasses import dataclass

WINDOW_LABELS = ("1m", "5m", "15m", "30m", "1h", "4h")
WINDOW_SECONDS: dict[str, int] = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}

STATE_PENDING = "PENDING"
STATE_TP1_HIT = "TP1_HIT"
STATE_TP2_HIT = "TP2_HIT"
STATE_SL_HIT = "SL_HIT"
STATE_AMBIGUOUS = "AMBIGUOUS"
STATE_EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class SignalRecord:
    # -- immutable snapshot, set once at creation, never overwritten --
    signal_id: str
    alert_id: str
    symbol: str
    direction: str
    setup_type: str
    setup_score: float
    scanner_activity_score: float | None
    structure_pattern: str | None
    structure_confidence: float | None
    timeframe_alignment: str | None
    entry_zone_low: float
    entry_zone_high: float
    reference_entry_price: float
    stop_price: float
    take_profit_1: float
    take_profit_2: float | None
    initial_rr_tp1: float
    initial_rr_tp2: float | None
    signal_timestamp: float

    # -- outcome-tracking fields, evolve as time passes --
    outcome_state: str = STATE_PENDING
    tp1_hit_at: float | None = None
    tp2_hit_at: float | None = None
    sl_hit_at: float | None = None
    ambiguous_at: float | None = None
    resolved_at: float | None = None
    last_evaluated_at: float | None = None


@dataclass(frozen=True)
class WindowResult:
    signal_id: str
    window_label: str
    forward_return: float | None
    mfe: float | None
    mae: float | None
    evaluated_at: float
