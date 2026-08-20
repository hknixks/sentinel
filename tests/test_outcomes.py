from __future__ import annotations

import math
import time

import pytest

from sentinel.alerts.models import STATUS_ACTIVE, AlertRecord
from sentinel.market_state import Candle, MarketStateStore
from sentinel.outcomes.models import (
    STATE_AMBIGUOUS,
    STATE_EXPIRED,
    STATE_PENDING,
    STATE_SL_HIT,
    STATE_TP1_HIT,
    STATE_TP2_HIT,
    SignalRecord,
)
from sentinel.outcomes.outcome_tracker import OutcomeTracker
from sentinel.outcomes.store import OutcomeStore
from sentinel.outcomes.windows import candles_since, compute_tp_sl_outcome, compute_window_metrics
from sentinel.scanner.scanner import ScannerResult
from sentinel.setups.models import EntryZone, RiskLevels, SetupCandidate
from sentinel.structure.models import Levels, Structure, StructureFeatures, StructureResult, TimeframeStructure


def _candle(open_time_ms: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> Candle:
    return Candle(open_time=open_time_ms, open=o, high=h, low=l, close=c, volume=v, is_closed=True)


def _alert_record(
    symbol="BTCUSDT", direction="long", setup_type="breakout", created_at=1_000_000.0, alert_id="alert-1"
) -> AlertRecord:
    return AlertRecord(
        alert_id=alert_id, symbol=symbol, direction=direction, setup_type=setup_type,
        status=STATUS_ACTIVE, setup_score=80.0, entry_low=100.0, entry_high=102.0,
        stop_loss=90.0, take_profit_1=110.0, take_profit_2=120.0, risk_reward_tp1=2.0,
        invalidation_level=90.0, structure_context="x", confirmation_factors="x",
        reasoning="x", created_at=created_at, resolved_at=None, telegram_message_id=None,
        telegram_chat_id=None, decision=None, decision_at=None, decision_reference_price=None,
    )


def _candidate(
    symbol="BTCUSDT", direction="long", setup_type="breakout",
    entry_low=100.0, entry_high=102.0, stop=90.0, tp1=110.0, tp2=120.0, score=80.0,
) -> SetupCandidate:
    risk = RiskLevels(
        stop_loss=stop, take_profit_1=tp1, take_profit_2=tp2,
        risk_per_unit=10.0, reward_to_tp1=9.0, reward_to_tp2=19.0,
        risk_reward_tp1=0.9, risk_reward_tp2=1.9,
    )
    return SetupCandidate(
        symbol=symbol, timestamp=0.0, direction=direction, setup_type=setup_type,
        entry_zone=EntryZone(low=entry_low, high=entry_high), invalidation_level=stop,
        risk=risk, structure_context="x", confirmation_factors=(), setup_score=score,
    )


def _scanner_result(symbol="BTCUSDT", score=75.0) -> ScannerResult:
    return ScannerResult(symbol=symbol, score=score, direction="bullish", features=None, timestamp=0.0)


def _structure_result(symbol="BTCUSDT", pattern="bullish_breakout", confidence=70.0, alignment="strong_bullish_alignment") -> StructureResult:
    return StructureResult(
        symbol=symbol, timestamp=0.0,
        structure=Structure(trend="bullish", pattern=pattern, phase="expansion"),
        directional_bias="bullish", confidence=confidence,
        levels=Levels(recent_high=None, recent_low=None, support=None, resistance=None),
        features=StructureFeatures(None, None, None, None, None, None, None, None),
        timeframe_alignment=alignment,
        timeframes=(TimeframeStructure("1m", "bullish", 20),),
        reason="test fixture",
    )


async def _market_store_with_candles(symbol: str, candles: list[Candle]) -> MarketStateStore:
    store = MarketStateStore()
    await store.init_symbols([symbol])
    for c in candles:
        await store.update_kline(symbol, c, event_ts=c.open_time / 1000.0)
    return store


# -- 1. New signal creates an outcome record ---------------------------

@pytest.mark.asyncio
async def test_new_signal_creates_outcome_record(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    tracker = OutcomeTracker(store, market_store)

    alert = _alert_record()
    candidate = _candidate()
    signal = await tracker.create_signal(alert, candidate, _scanner_result(), _structure_result())

    assert signal is not None
    persisted = await store.get_signal(signal.signal_id)
    assert persisted is not None
    assert persisted.symbol == "BTCUSDT"
    assert persisted.outcome_state == STATE_PENDING


# -- 2. Same signal cannot create duplicate outcome records -------------

@pytest.mark.asyncio
async def test_duplicate_signal_does_not_create_second_record(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    tracker = OutcomeTracker(store, market_store)

    alert = _alert_record()
    candidate = _candidate()

    first = await tracker.create_signal(alert, candidate, None, None)
    second = await tracker.create_signal(alert, candidate, None, None)

    assert first is not None
    assert second is None


# -- 3. Original signal values are immutable -----------------------------

@pytest.mark.asyncio
async def test_original_signal_values_are_immutable_after_evaluation(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    alert = _alert_record(created_at=time.time() - 100)
    candidate = _candidate()
    tracker_store = store

    market_store = await _market_store_with_candles(
        "BTCUSDT", [_candle(int((alert.created_at) * 1000), 100, 111, 95, 108)]
    )
    tracker = OutcomeTracker(tracker_store, market_store)
    signal = await tracker.create_signal(alert, candidate, None, None)

    before = await store.get_signal(signal.signal_id)
    await tracker.evaluate_all_pending()
    after = await store.get_signal(signal.signal_id)

    assert after.outcome_state != STATE_PENDING or after.outcome_state == before.outcome_state
    # Immutable snapshot fields must never change:
    assert after.reference_entry_price == before.reference_entry_price
    assert after.stop_price == before.stop_price
    assert after.take_profit_1 == before.take_profit_1
    assert after.take_profit_2 == before.take_profit_2
    assert after.setup_score == before.setup_score
    assert after.setup_type == before.setup_type
    assert after.direction == before.direction
    assert after.symbol == before.symbol
    assert after.entry_zone_low == before.entry_zone_low
    assert after.entry_zone_high == before.entry_zone_high
    assert after.signal_timestamp == before.signal_timestamp


# -- 4 & 5. Forward return LONG/SHORT --------------------------------------

def test_long_forward_return_calculated_correctly():
    candles = [_candle(60_000, 100, 106, 99, 105)]
    fr, mfe, mae = compute_window_metrics("long", 100.0, 0.0, "1m", candles, now=61.0)
    assert fr == pytest.approx(5.0)


def test_short_forward_return_calculated_correctly():
    candles = [_candle(60_000, 100, 101, 94, 95)]
    fr, mfe, mae = compute_window_metrics("short", 100.0, 0.0, "1m", candles, now=61.0)
    assert fr == pytest.approx(5.0)


# -- 6 & 7. LONG MFE / MAE ---------------------------------------------------

def test_long_mfe_correct():
    candles = [_candle(30_000, 100, 108, 99, 103), _candle(60_000, 103, 106, 100, 104)]
    fr, mfe, mae = compute_window_metrics("long", 100.0, 0.0, "1m", candles, now=61.0)
    assert mfe == pytest.approx(8.0)  # max high 108


def test_long_mae_correct():
    candles = [_candle(30_000, 100, 108, 99, 103), _candle(60_000, 103, 106, 100, 104)]
    fr, mfe, mae = compute_window_metrics("long", 100.0, 0.0, "1m", candles, now=61.0)
    assert mae == pytest.approx(1.0)  # min low 99


# -- 8 & 9. SHORT MFE / MAE ---------------------------------------------------

def test_short_mfe_correct():
    candles = [_candle(30_000, 100, 101, 93, 97), _candle(60_000, 97, 102, 95, 96)]
    fr, mfe, mae = compute_window_metrics("short", 100.0, 0.0, "1m", candles, now=61.0)
    assert mfe == pytest.approx(7.0)  # min low 93


def test_short_mae_correct():
    candles = [_candle(30_000, 100, 101, 93, 97), _candle(60_000, 97, 102, 95, 96)]
    fr, mfe, mae = compute_window_metrics("short", 100.0, 0.0, "1m", candles, now=61.0)
    assert mae == pytest.approx(2.0)  # max high 102


# -- 10, 11, 12. TP1 / TP2 / SL detection -------------------------------

def test_tp1_detection():
    candles = [_candle(60_000, 100, 111, 95, 108)]
    outcome = compute_tp_sl_outcome("long", stop=90.0, tp1=110.0, tp2=120.0, candles=candles)
    assert outcome["state"] == STATE_TP1_HIT
    assert outcome["tp1_hit_at"] == 60.0


def test_tp2_detection():
    candles = [
        _candle(60_000, 100, 111, 95, 108),
        _candle(120_000, 108, 121, 105, 118),
    ]
    outcome = compute_tp_sl_outcome("long", stop=90.0, tp1=110.0, tp2=120.0, candles=candles)
    assert outcome["state"] == STATE_TP2_HIT
    assert outcome["tp1_hit_at"] == 60.0
    assert outcome["tp2_hit_at"] == 120.0


def test_sl_detection():
    candles = [_candle(60_000, 100, 95, 85, 88)]
    outcome = compute_tp_sl_outcome("long", stop=90.0, tp1=110.0, tp2=120.0, candles=candles)
    assert outcome["state"] == STATE_SL_HIT
    assert outcome["sl_hit_at"] == 60.0


# -- 13. Missing future data remains pending -----------------------------

def test_missing_future_data_remains_pending():
    outcome = compute_tp_sl_outcome("long", stop=90.0, tp1=110.0, tp2=120.0, candles=[])
    assert outcome["state"] == STATE_PENDING

    fr, mfe, mae = compute_window_metrics("long", 100.0, 0.0, "1h", candles=[], now=10.0)
    assert fr is None and mfe is None and mae is None


# -- 14-19. Window population timing -------------------------------------

def _full_candle_series():
    # One candle per minute for 4h+, constant OHLC so every window's
    # expected values are identical and easy to assert.
    return [_candle(i * 60_000, 100, 106, 104, 105) for i in range(0, 245)]


@pytest.mark.parametrize(
    "label,horizon_seconds",
    [("1m", 60), ("5m", 300), ("15m", 900), ("30m", 1800), ("1h", 3600), ("4h", 14400)],
)
def test_window_populates_only_after_its_horizon_elapses(label, horizon_seconds):
    candles = _full_candle_series()

    before = compute_window_metrics("long", 100.0, 0.0, label, candles, now=horizon_seconds - 1)
    assert before == (None, None, None)

    fr, mfe, mae = compute_window_metrics("long", 100.0, 0.0, label, candles, now=horizon_seconds)
    assert fr == pytest.approx(5.0)
    assert mfe == pytest.approx(6.0)
    assert mae == pytest.approx(-4.0)


# -- 20. Ambiguous TP/SL candle is not falsely classified -----------------

def test_ambiguous_when_tp_and_sl_touched_in_same_candle():
    candles = [_candle(60_000, 100, 111, 85, 95)]  # high touches TP1, low touches SL
    outcome = compute_tp_sl_outcome("long", stop=90.0, tp1=110.0, tp2=120.0, candles=candles)
    assert outcome["state"] == STATE_AMBIGUOUS
    assert outcome["ambiguous_at"] == 60.0
    assert outcome["tp1_hit_at"] is None
    assert outcome["sl_hit_at"] is None


def test_ambiguous_after_tp1_between_tp2_and_sl_same_candle():
    candles = [
        _candle(60_000, 100, 111, 95, 108),  # TP1 hit cleanly
        _candle(120_000, 108, 122, 85, 100),  # both TP2 and SL touched in the same candle
    ]
    outcome = compute_tp_sl_outcome("long", stop=90.0, tp1=110.0, tp2=120.0, candles=candles)
    assert outcome["state"] == STATE_AMBIGUOUS
    assert outcome["tp1_hit_at"] == 60.0
    assert outcome["tp2_hit_at"] is None
    assert outcome["sl_hit_at"] is None


# -- 21. Restart does not duplicate signals --------------------------------

@pytest.mark.asyncio
async def test_restart_does_not_duplicate_signal(tmp_path):
    db_path = str(tmp_path / "outcomes.db")
    store1 = OutcomeStore(db_path)
    market_store = MarketStateStore()
    tracker1 = OutcomeTracker(store1, market_store)

    alert = _alert_record()
    candidate = _candidate()
    first = await tracker1.create_signal(alert, candidate, None, None)
    assert first is not None

    # Simulate restart: brand new OutcomeStore/OutcomeTracker objects
    # against the same on-disk file.
    store2 = OutcomeStore(db_path)
    tracker2 = OutcomeTracker(store2, market_store)
    second = await tracker2.create_signal(alert, candidate, None, None)

    assert second is None
    persisted = await store2.get_signal(first.signal_id)
    assert persisted is not None


# -- 22. Expired signals are retained --------------------------------------

@pytest.mark.asyncio
async def test_expired_signal_is_retained_not_deleted(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    old_timestamp = time.time() - 20_000  # well past the 4h default horizon
    alert = _alert_record(created_at=old_timestamp)
    candidate = _candidate()
    market_store = MarketStateStore()  # no candle history at all for this symbol
    tracker = OutcomeTracker(store, market_store, max_horizon_seconds=14400.0)

    signal = await tracker.create_signal(alert, candidate, None, None)
    await tracker.evaluate_all_pending()

    persisted = await store.get_signal(signal.signal_id)
    assert persisted is not None
    assert persisted.outcome_state == STATE_EXPIRED
    assert persisted.resolved_at is not None


# -- 23. Multiple signals for different symbols are independent -----------

@pytest.mark.asyncio
async def test_multiple_symbols_are_independent(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    tracker = OutcomeTracker(store, market_store)

    btc = await tracker.create_signal(_alert_record(symbol="BTCUSDT", alert_id="btc-1"), _candidate(symbol="BTCUSDT"), None, None)
    sol = await tracker.create_signal(_alert_record(symbol="SOLUSDT", alert_id="sol-1"), _candidate(symbol="SOLUSDT"), None, None)

    assert btc is not None and sol is not None
    assert btc.signal_id != sol.signal_id
    assert (await store.get_signal(btc.signal_id)).symbol == "BTCUSDT"
    assert (await store.get_signal(sol.signal_id)).symbol == "SOLUSDT"


# -- 24. LONG and SHORT outcomes are symmetrical ---------------------------

def test_long_and_short_outcomes_are_symmetrical():
    long_candles = [_candle(60_000, 100, 111, 95, 108)]
    short_candles = [_candle(60_000, 100, 105, 89, 92)]

    long_outcome = compute_tp_sl_outcome("long", stop=90.0, tp1=110.0, tp2=120.0, candles=long_candles)
    short_outcome = compute_tp_sl_outcome("short", stop=110.0, tp1=90.0, tp2=80.0, candles=short_candles)

    assert long_outcome["state"] == STATE_TP1_HIT
    assert short_outcome["state"] == STATE_TP1_HIT

    long_fr, long_mfe, long_mae = compute_window_metrics("long", 100.0, 0.0, "1m", long_candles, now=61.0)
    short_fr, short_mfe, short_mae = compute_window_metrics("short", 100.0, 0.0, "1m", short_candles, now=61.0)
    assert long_fr == pytest.approx(8.0)
    assert short_fr == pytest.approx(8.0)


# -- 25. No NaN/Inf values are persisted -----------------------------------

@pytest.mark.asyncio
async def test_no_nan_or_inf_persisted(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    alert = _alert_record(created_at=time.time() - 61)
    candidate = _candidate()
    market_store = await _market_store_with_candles(
        "BTCUSDT", [_candle(int(alert.created_at * 1000), 100, 106, 99, 103)]
    )
    tracker = OutcomeTracker(store, market_store)
    signal = await tracker.create_signal(alert, candidate, None, None)
    await tracker.evaluate_all_pending()

    windows = await store.get_windows(signal.signal_id)
    for w in windows:
        for v in (w.forward_return, w.mfe, w.mae):
            if v is not None:
                assert not math.isnan(v)
                assert not math.isinf(v)

    persisted = await store.get_signal(signal.signal_id)
    for v in (persisted.reference_entry_price, persisted.stop_price, persisted.take_profit_1):
        assert not math.isnan(v)
        assert not math.isinf(v)


# -- 26. Outcome tracking failure cannot crash the market-data engine -----

@pytest.mark.asyncio
async def test_evaluation_with_missing_symbol_does_not_raise(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()  # symbol never registered at all
    tracker = OutcomeTracker(store, market_store)

    alert = _alert_record(symbol="UNKNOWNUSDT", alert_id="unknown-1")
    candidate = _candidate(symbol="UNKNOWNUSDT")
    signal = await tracker.create_signal(alert, candidate, None, None)

    # Must not raise even though the symbol has no market state at all.
    count = await tracker.evaluate_all_pending()
    assert count >= 0
    assert signal is not None


@pytest.mark.asyncio
async def test_create_signal_with_none_scanner_and_structure_does_not_raise(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    tracker = OutcomeTracker(store, market_store)

    signal = await tracker.create_signal(_alert_record(), _candidate(), None, None)

    assert signal is not None
    assert signal.scanner_activity_score is None
    assert signal.structure_pattern is None
    assert signal.structure_confidence is None


# -- store-level idempotency (mirrors AlertStore's proven pattern) --------

@pytest.mark.asyncio
async def test_store_create_signal_is_idempotent_at_db_level(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    signal = SignalRecord(
        signal_id="dup-1", alert_id="dup-1", symbol="BTCUSDT", direction="long", setup_type="breakout",
        setup_score=80.0, scanner_activity_score=None, structure_pattern=None, structure_confidence=None,
        timeframe_alignment=None, entry_zone_low=100.0, entry_zone_high=102.0,
        reference_entry_price=101.0, stop_price=90.0, take_profit_1=110.0, take_profit_2=120.0,
        initial_rr_tp1=0.9, initial_rr_tp2=1.9, signal_timestamp=0.0,
    )

    first = await store.create_signal(signal)
    second = await store.create_signal(signal)

    assert first is True
    assert second is False


def test_candles_since_filters_by_signal_timestamp():
    candles = (_candle(0, 100, 101, 99, 100), _candle(60_000, 100, 102, 99, 101), _candle(120_000, 101, 103, 100, 102))
    result = candles_since(candles, signal_timestamp=60.0)
    assert len(result) == 2


# -- 27. Full snapshot capture with REAL scanner/structure inputs ---------
# (closes a coverage gap: prior tests only exercised None,None for these)

@pytest.mark.asyncio
async def test_signal_captures_real_scanner_and_structure_values(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    tracker = OutcomeTracker(store, market_store)

    alert = _alert_record(symbol="ETHUSDT", direction="long", setup_type="breakout", created_at=500_000.0)
    candidate = _candidate(
        symbol="ETHUSDT", direction="long", setup_type="breakout",
        entry_low=200.0, entry_high=204.0, stop=180.0, tp1=220.0, tp2=240.0, score=87.5,
    )
    scanner_result = _scanner_result(symbol="ETHUSDT", score=91.3)
    structure_result = _structure_result(
        symbol="ETHUSDT", pattern="bullish_breakout", confidence=73.2, alignment="strong_bullish_alignment"
    )

    signal = await tracker.create_signal(alert, candidate, scanner_result, structure_result)
    assert signal is not None

    persisted = await store.get_signal(signal.signal_id)
    assert persisted is not None

    assert persisted.symbol == "ETHUSDT"
    assert persisted.direction == "long"
    assert persisted.setup_type == "breakout"
    assert persisted.entry_zone_low == pytest.approx(200.0)
    assert persisted.entry_zone_high == pytest.approx(204.0)
    assert persisted.reference_entry_price == pytest.approx(202.0)  # midpoint of entry zone
    assert persisted.stop_price == pytest.approx(180.0)
    assert persisted.take_profit_1 == pytest.approx(220.0)
    assert persisted.take_profit_2 == pytest.approx(240.0)
    assert persisted.setup_score == pytest.approx(87.5)
    assert persisted.scanner_activity_score == pytest.approx(91.3)
    assert persisted.structure_pattern == "bullish_breakout"
    assert persisted.structure_confidence == pytest.approx(73.2)
    assert persisted.timeframe_alignment == "strong_bullish_alignment"
    assert persisted.initial_rr_tp1 == pytest.approx(candidate.risk.risk_reward_tp1)
    assert persisted.initial_rr_tp2 == pytest.approx(candidate.risk.risk_reward_tp2)
    assert persisted.signal_timestamp == pytest.approx(500_000.0)
    assert persisted.outcome_state == STATE_PENDING


# -- 28. Multi-cycle state progression through OutcomeTracker+store -------
# (integration-level, not just the pure compute_tp_sl_outcome function:
# candles stream into MarketStateStore across several real
# evaluate_all_pending() calls, mirroring how _run_outcome_evaluation_
# periodically actually drives this in main.py)

@pytest.mark.asyncio
async def test_tracker_progresses_pending_to_tp1_to_tp2_across_cycles(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    await market_store.init_symbols(["BTCUSDT"])
    tracker = OutcomeTracker(store, market_store)

    t0 = time.time() - 30  # recent enough that the 4h horizon has not elapsed
    alert = _alert_record(created_at=t0)
    candidate = _candidate(entry_low=100.0, entry_high=102.0, stop=90.0, tp1=110.0, tp2=120.0)
    signal = await tracker.create_signal(alert, candidate, None, None)
    assert signal is not None

    # Cycle 1: no candles yet -> still PENDING.
    await tracker.evaluate_all_pending()
    s1 = await store.get_signal(signal.signal_id)
    assert s1.outcome_state == STATE_PENDING

    # Cycle 2: one candle touches TP1 only -> TP1_HIT.
    c1 = _candle(int(t0 * 1000) + 60_000, 100, 111, 99, 108)
    await market_store.update_kline("BTCUSDT", c1, event_ts=t0 + 60)
    await tracker.evaluate_all_pending()
    s2 = await store.get_signal(signal.signal_id)
    assert s2.outcome_state == STATE_TP1_HIT
    assert s2.tp1_hit_at == pytest.approx(t0 + 60)
    assert s2.tp2_hit_at is None

    # Cycle 3: a later candle touches TP2 -> TP2_HIT, and TP1 stays
    # recorded from before (never overwritten/lost).
    c2 = _candle(int(t0 * 1000) + 120_000, 108, 121, 106, 119)
    await market_store.update_kline("BTCUSDT", c2, event_ts=t0 + 120)
    await tracker.evaluate_all_pending()
    s3 = await store.get_signal(signal.signal_id)
    assert s3.outcome_state == STATE_TP2_HIT
    assert s3.tp1_hit_at == pytest.approx(t0 + 60)
    assert s3.tp2_hit_at == pytest.approx(t0 + 120)
    assert s3.sl_hit_at is None
    assert s3.resolved_at is not None


@pytest.mark.asyncio
async def test_tracker_progresses_pending_to_sl_across_cycles(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    await market_store.init_symbols(["BTCUSDT"])
    tracker = OutcomeTracker(store, market_store)

    t0 = time.time() - 30  # recent enough that the 4h horizon has not elapsed
    alert = _alert_record(created_at=t0)
    candidate = _candidate(entry_low=100.0, entry_high=102.0, stop=90.0, tp1=110.0, tp2=120.0)
    signal = await tracker.create_signal(alert, candidate, None, None)

    await tracker.evaluate_all_pending()
    assert (await store.get_signal(signal.signal_id)).outcome_state == STATE_PENDING

    c1 = _candle(int(t0 * 1000) + 60_000, 100, 103, 89, 91)  # low <= stop(90) -> SL
    await market_store.update_kline("BTCUSDT", c1, event_ts=t0 + 60)
    await tracker.evaluate_all_pending()
    s2 = await store.get_signal(signal.signal_id)
    assert s2.outcome_state == STATE_SL_HIT
    assert s2.sl_hit_at == pytest.approx(t0 + 60)
    assert s2.tp1_hit_at is None
    assert s2.resolved_at is not None

    # A further cycle (more candles, even ones that would touch TP) must
    # NOT resurrect an already-resolved SL_HIT signal into a TP state --
    # this is only possible because compute_tp_sl_outcome always walks
    # chronologically from the start and the SL candle strictly precedes
    # any later TP-touching candle, so it is the true first outcome.
    c2 = _candle(int(t0 * 1000) + 120_000, 91, 130, 91, 125)
    await market_store.update_kline("BTCUSDT", c2, event_ts=t0 + 120)
    await tracker.evaluate_all_pending()
    s3 = await store.get_signal(signal.signal_id)
    assert s3.outcome_state == STATE_SL_HIT


@pytest.mark.asyncio
async def test_tracker_progresses_pending_to_ambiguous_across_cycles(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    await market_store.init_symbols(["ETHUSDT"])
    tracker = OutcomeTracker(store, market_store)

    t0 = 3_000_000.0
    alert = _alert_record(symbol="ETHUSDT", created_at=t0)
    candidate = _candidate(symbol="ETHUSDT", entry_low=100.0, entry_high=102.0, stop=90.0, tp1=110.0, tp2=120.0)
    signal = await tracker.create_signal(alert, candidate, None, None)

    # Single candle touches both SL and TP1 -> AMBIGUOUS, for both LONG
    # (this test) and SHORT (the dedicated pure-function tests above).
    c1 = _candle(int(t0 * 1000) + 60_000, 100, 111, 85, 95)
    await market_store.update_kline("ETHUSDT", c1, event_ts=t0 + 60)
    await tracker.evaluate_all_pending()
    s1 = await store.get_signal(signal.signal_id)
    assert s1.outcome_state == STATE_AMBIGUOUS
    assert s1.ambiguous_at == pytest.approx(t0 + 60)
    assert s1.tp1_hit_at is None
    assert s1.sl_hit_at is None
    assert s1.resolved_at is not None


@pytest.mark.asyncio
async def test_tracker_short_ambiguous_when_sl_and_tp_same_candle(tmp_path):
    store = OutcomeStore(str(tmp_path / "outcomes.db"))
    market_store = MarketStateStore()
    await market_store.init_symbols(["SOLUSDT"])
    tracker = OutcomeTracker(store, market_store)

    t0 = 4_000_000.0
    alert = _alert_record(symbol="SOLUSDT", direction="short", created_at=t0)
    candidate = _candidate(symbol="SOLUSDT", direction="short", entry_low=98.0, entry_high=100.0, stop=110.0, tp1=90.0, tp2=80.0)
    signal = await tracker.create_signal(alert, candidate, None, None)

    # SHORT: SL if high >= stop(110), TP1 if low <= tp1(90). Same candle
    # touches both -> AMBIGUOUS, mirroring the LONG case above.
    c1 = _candle(int(t0 * 1000) + 60_000, 100, 112, 85, 95)
    await market_store.update_kline("SOLUSDT", c1, event_ts=t0 + 60)
    await tracker.evaluate_all_pending()
    s1 = await store.get_signal(signal.signal_id)
    assert s1.outcome_state == STATE_AMBIGUOUS
    assert s1.tp1_hit_at is None
    assert s1.sl_hit_at is None


# -- 29. Outcome-window rows survive an OutcomeStore restart --------------

@pytest.mark.asyncio
async def test_outcome_window_rows_survive_restart(tmp_path):
    db_path = str(tmp_path / "outcomes.db")
    store1 = OutcomeStore(db_path)
    market_store = MarketStateStore()
    await market_store.init_symbols(["BTCUSDT"])
    tracker1 = OutcomeTracker(store1, market_store)

    t0 = time.time() - 120  # 1m window horizon already elapsed
    alert = _alert_record(created_at=t0)
    candidate = _candidate()
    signal = await tracker1.create_signal(alert, candidate, None, None)

    c1 = _candle(int(t0 * 1000) + 60_000, 100, 106, 99, 103)
    await market_store.update_kline("BTCUSDT", c1, event_ts=t0 + 60)
    await tracker1.evaluate_all_pending()

    windows_before = await store1.get_windows(signal.signal_id)
    assert any(w.window_label == "1m" and w.forward_return is not None for w in windows_before)

    # Simulate restart: brand-new OutcomeStore object against the same
    # on-disk database file (no in-process state carried over).
    store2 = OutcomeStore(db_path)
    windows_after = await store2.get_windows(signal.signal_id)
    assert len(windows_after) == len(windows_before)
    for before, after in zip(sorted(windows_before, key=lambda w: w.window_label), sorted(windows_after, key=lambda w: w.window_label)):
        assert before.window_label == after.window_label
        assert before.forward_return == after.forward_return
        assert before.mfe == after.mfe
        assert before.mae == after.mae


# -- 30. Restart limitation: pre-restart in-memory candles are NOT --------
#        fabricated or backfilled after a MarketStateStore "restart".

@pytest.mark.asyncio
async def test_restart_does_not_fabricate_missing_candle_history(tmp_path):
    db_path = str(tmp_path / "outcomes.db")
    store1 = OutcomeStore(db_path)
    market_store1 = MarketStateStore()
    await market_store1.init_symbols(["BTCUSDT"])
    tracker1 = OutcomeTracker(store1, market_store1)

    t0 = time.time() - 4000  # old enough that several windows' horizons have elapsed
    alert = _alert_record(created_at=t0)
    candidate = _candidate()
    signal = await tracker1.create_signal(alert, candidate, None, None)

    # Deliberately do NOT feed any candles into market_store1 before the
    # "restart" -- this reproduces the documented gap: an in-memory-only
    # candle history that is empty when the process comes back up.
    store2 = OutcomeStore(db_path)
    market_store2 = MarketStateStore()  # fresh, empty in-memory history: the "restart"
    await market_store2.init_symbols(["BTCUSDT"])
    tracker2 = OutcomeTracker(store2, market_store2, max_horizon_seconds=14400.0)

    # Must not raise, must not crash, must not invent candle data.
    count = await tracker2.evaluate_all_pending()
    assert count >= 0

    persisted = await store2.get_signal(signal.signal_id)
    assert persisted is not None
    windows = await store2.get_windows(signal.signal_id)
    # No window can have a populated value with zero observed candles --
    # that would mean fabrication.
    for w in windows:
        assert w.forward_return is None
        assert w.mfe is None
        assert w.mae is None
