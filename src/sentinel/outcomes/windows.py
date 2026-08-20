"""
Pure, deterministic outcome calculations: forward return, MFE/MAE per
window, and sequential TP1/TP2/SL detection from observed 1m candles.
No I/O -- fully testable with synthetic candle fixtures, no live data.
"""

from __future__ import annotations

from sentinel.market_state import Candle
from sentinel.outcomes.models import (
    STATE_AMBIGUOUS,
    STATE_PENDING,
    STATE_SL_HIT,
    STATE_TP1_HIT,
    STATE_TP2_HIT,
    WINDOW_SECONDS,
)


def _candle_time_s(candle: Candle) -> float:
    """Candle.open_time is Binance-style milliseconds; everything else in
    this module works in seconds."""
    return candle.open_time / 1000.0


def candles_since(candles: tuple[Candle, ...], signal_timestamp: float) -> list[Candle]:
    return [c for c in candles if _candle_time_s(c) >= signal_timestamp]


def compute_window_metrics(
    direction: str,
    reference_entry: float,
    signal_timestamp: float,
    window_label: str,
    candles: list[Candle],
    now: float,
) -> tuple[float | None, float | None, float | None]:
    """Returns (forward_return, mfe, mae) as percentages. Returns
    (None, None, None) until the window's full wall-clock duration has
    actually elapsed -- never fabricated early. `candles` must already be
    filtered to candles at/after signal_timestamp (see candles_since)."""
    window_seconds = WINDOW_SECONDS[window_label]
    horizon = signal_timestamp + window_seconds
    if now < horizon:
        return None, None, None

    in_window = [c for c in candles if _candle_time_s(c) <= horizon]
    if not in_window:
        return None, None, None

    price_future = in_window[-1].close
    highs = [c.high for c in in_window]
    lows = [c.low for c in in_window]

    if direction == "long":
        forward_return = (price_future - reference_entry) / reference_entry * 100
        mfe = (max(highs) - reference_entry) / reference_entry * 100
        mae = (reference_entry - min(lows)) / reference_entry * 100
    else:
        forward_return = (reference_entry - price_future) / reference_entry * 100
        mfe = (reference_entry - min(lows)) / reference_entry * 100
        mae = (max(highs) - reference_entry) / reference_entry * 100

    return forward_return, mfe, mae


def _touches(
    direction: str, stop: float, tp1: float, tp2: float | None, candle: Candle
) -> tuple[bool, bool, bool]:
    """SL/TP1/TP2 touches use real candle high/low, not just closes:
    LONG:  SL if low <= stop,  TP if high >= target.
    SHORT: SL if high >= stop, TP if low <= target."""
    if direction == "long":
        sl = candle.low <= stop
        t1 = candle.high >= tp1
        t2 = tp2 is not None and candle.high >= tp2
    else:
        sl = candle.high >= stop
        t1 = candle.low <= tp1
        t2 = tp2 is not None and candle.low <= tp2
    return sl, t1, t2


def compute_tp_sl_outcome(
    direction: str, stop: float, tp1: float, tp2: float | None, candles: list[Candle]
) -> dict:
    """Sequential, chronological walk through candles (oldest first).
    Returns {state, tp1_hit_at, tp2_hit_at, sl_hit_at, ambiguous_at} (hit
    times as candle open_time in SECONDS, or None).

    Never guesses intrabar ordering: if two levels are touched within the
    SAME candle, the result is AMBIGUOUS from that point on -- 1-minute
    OHLC cannot tell us which happened first within that minute. This
    applies both before TP1 (SL vs TP1 vs a huge move hitting TP2
    directly) and after TP1 (SL vs TP2)."""
    state = STATE_PENDING
    tp1_hit_at = tp2_hit_at = sl_hit_at = ambiguous_at = None

    for candle in candles:
        if state in (STATE_SL_HIT, STATE_TP2_HIT, STATE_AMBIGUOUS):
            break

        sl_touch, t1_touch, t2_touch = _touches(direction, stop, tp1, tp2, candle)
        candle_time = _candle_time_s(candle)

        if state == STATE_PENDING:
            touched = [name for name, hit in (("SL", sl_touch), ("TP1", t1_touch), ("TP2", t2_touch)) if hit]
            if len(touched) >= 2:
                state = STATE_AMBIGUOUS
                ambiguous_at = candle_time
            elif touched == ["SL"]:
                state = STATE_SL_HIT
                sl_hit_at = candle_time
            elif touched == ["TP1"]:
                state = STATE_TP1_HIT
                tp1_hit_at = candle_time
            elif touched == ["TP2"]:
                state = STATE_TP2_HIT
                tp1_hit_at = candle_time
                tp2_hit_at = candle_time

        elif state == STATE_TP1_HIT:
            touched = [name for name, hit in (("SL", sl_touch), ("TP2", t2_touch)) if hit]
            if len(touched) >= 2:
                state = STATE_AMBIGUOUS
                ambiguous_at = candle_time
            elif touched == ["SL"]:
                state = STATE_SL_HIT
                sl_hit_at = candle_time
            elif touched == ["TP2"]:
                state = STATE_TP2_HIT
                tp2_hit_at = candle_time

    return {
        "state": state,
        "tp1_hit_at": tp1_hit_at,
        "tp2_hit_at": tp2_hit_at,
        "sl_hit_at": sl_hit_at,
        "ambiguous_at": ambiguous_at,
    }
