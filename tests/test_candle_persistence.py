from __future__ import annotations

import asyncio
import math

import pytest

from sentinel.binance.ws_client import _process_message
from sentinel.candles.store import CandleStore, is_valid_closed_candle
from sentinel.market_state import Candle, MarketStateStore
from sentinel.scanner.features import FeatureEngine
from sentinel.structure.structure import StructureEngine
from sentinel.outcomes.windows import candles_since, compute_tp_sl_outcome, compute_window_metrics


def _candle(open_time_ms: int, o=100.0, h=101.0, l=99.0, c=100.5, v=10.0, is_closed=True) -> Candle:
    return Candle(open_time=open_time_ms, open=o, high=h, low=l, close=c, volume=v, is_closed=is_closed)


MIN_MS = 60_000


def _sequence(symbol_start_ms: int, count: int, base_price: float = 100.0) -> list[Candle]:
    return [
        _candle(symbol_start_ms + i * MIN_MS, o=base_price + i, h=base_price + i + 1, l=base_price + i - 1, c=base_price + i + 0.5)
        for i in range(count)
    ]


# -- 1. candle persistence -------------------------------------------------

def test_append_candle_persists_row(tmp_path):
    store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    ok = store.append_candle_sync("BTCUSDT", _candle(1_000 * MIN_MS))

    assert ok is True
    assert store.count_sync("BTCUSDT") == 1


# -- 2. candle restoration ---------------------------------------------------

@pytest.mark.asyncio
async def test_restore_candle_history_repopulates_market_state(tmp_path):
    candle_store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    candles = _sequence(1_000 * MIN_MS, 5)
    for c in candles:
        candle_store.append_candle_sync("BTCUSDT", c)

    restored = await candle_store.load_history("BTCUSDT")
    market_store = MarketStateStore()
    await market_store.init_symbols(["BTCUSDT"])
    await market_store.restore_candle_history("BTCUSDT", restored)

    state = await market_store.get("BTCUSDT")
    assert state.candle_history == tuple(candles)
    assert state.last_candle_1m == candles[-1]


# -- 3. ordering --------------------------------------------------------------

@pytest.mark.asyncio
async def test_restore_orders_chronologically_even_if_input_is_shuffled():
    candles = _sequence(1_000 * MIN_MS, 5)
    shuffled = [candles[3], candles[0], candles[4], candles[1], candles[2]]

    market_store = MarketStateStore()
    await market_store.restore_candle_history("BTCUSDT", shuffled)

    state = await market_store.get("BTCUSDT")
    open_times = [c.open_time for c in state.candle_history]
    assert open_times == sorted(open_times)
    assert state.candle_history == tuple(candles)


# -- 4. duplicate prevention ---------------------------------------------------

def test_append_candle_same_open_time_twice_is_idempotent(tmp_path):
    store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    c = _candle(1_000 * MIN_MS)

    store.append_candle_sync("BTCUSDT", c)
    store.append_candle_sync("BTCUSDT", c)

    assert store.count_sync("BTCUSDT") == 1


@pytest.mark.asyncio
async def test_restore_dedupes_duplicate_open_time_in_input():
    c = _candle(1_000 * MIN_MS)
    market_store = MarketStateStore()
    await market_store.restore_candle_history("BTCUSDT", [c, c, c])

    state = await market_store.get("BTCUSDT")
    assert len(state.candle_history) == 1


# -- 5. cap enforcement ---------------------------------------------------------

def test_candle_store_enforces_cap_on_write(tmp_path):
    store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=5)
    for c in _sequence(1_000 * MIN_MS, 10):
        store.append_candle_sync("BTCUSDT", c)

    assert store.count_sync("BTCUSDT") == 5


@pytest.mark.asyncio
async def test_candle_store_cap_keeps_most_recent(tmp_path):
    store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=5)
    seq = _sequence(1_000 * MIN_MS, 10)
    for c in seq:
        store.append_candle_sync("BTCUSDT", c)

    remaining = await store.load_history("BTCUSDT")
    assert [c.open_time for c in remaining] == [c.open_time for c in seq[-5:]]


@pytest.mark.asyncio
async def test_restore_candle_history_respects_market_state_cap(monkeypatch):
    import sentinel.market_state as market_state_module

    monkeypatch.setattr(market_state_module, "MAX_CANDLE_HISTORY_MINUTES", 3)
    market_store = MarketStateStore()
    await market_store.restore_candle_history("BTCUSDT", _sequence(1_000 * MIN_MS, 10))

    state = await market_store.get("BTCUSDT")
    assert len(state.candle_history) == 3


# -- 6. restart continuation -----------------------------------------------------

@pytest.mark.asyncio
async def test_restart_continuation_appends_new_candles_after_restore(tmp_path):
    candle_store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    pre_restart = _sequence(1_000 * MIN_MS, 5)
    for c in pre_restart:
        candle_store.append_candle_sync("BTCUSDT", c)

    # simulate restart: fresh MarketStateStore, restore, then continue live
    market_store = MarketStateStore()
    restored = await candle_store.load_history("BTCUSDT")
    await market_store.restore_candle_history("BTCUSDT", restored)

    new_candle = _candle(1_005 * MIN_MS, o=200.0, h=201.0, l=199.0, c=200.5)
    await market_store.update_kline("BTCUSDT", new_candle, event_ts=1.0)

    state = await market_store.get("BTCUSDT")
    assert state.candle_history == tuple(pre_restart + [new_candle])


# -- 7. restart at an exact candle boundary --------------------------------------

@pytest.mark.asyncio
async def test_restart_exactly_on_candle_boundary_no_duplicate():
    """The last persisted candle's open_time is exactly the most recent
    closed candle -- the next live close must not duplicate it."""
    pre_restart = _sequence(1_000 * MIN_MS, 3)
    market_store = MarketStateStore()
    await market_store.restore_candle_history("BTCUSDT", pre_restart)

    # A live re-delivery of the SAME last candle (Binance can resend the
    # most recent closed kline right after reconnect) must not duplicate.
    same_last = pre_restart[-1]
    await market_store.update_kline("BTCUSDT", same_last, event_ts=1.0)

    state = await market_store.get("BTCUSDT")
    assert state.candle_history == tuple(pre_restart)


# -- 8. restart between candles ---------------------------------------------------

@pytest.mark.asyncio
async def test_restart_between_candles_leaves_gap_until_next_close():
    """Process restarts mid-minute (between candle closes): restored
    history ends at the last real close, and the in-progress (unclosed)
    candle at restart time is correctly NOT part of history until it
    actually closes."""
    pre_restart = _sequence(1_000 * MIN_MS, 3)
    market_store = MarketStateStore()
    await market_store.restore_candle_history("BTCUSDT", pre_restart)

    in_progress = _candle(1_003 * MIN_MS, is_closed=False)
    await market_store.update_kline("BTCUSDT", in_progress, event_ts=1.0)

    state = await market_store.get("BTCUSDT")
    assert state.candle_history == tuple(pre_restart)  # unclosed candle not appended
    assert state.last_candle_1m == in_progress  # but still visible as the live tick

    closed = _candle(1_003 * MIN_MS, c=105.0, is_closed=True)
    await market_store.update_kline("BTCUSDT", closed, event_ts=2.0)
    state = await market_store.get("BTCUSDT")
    assert state.candle_history == tuple(pre_restart + [closed])


# -- 9. missing historical gap remains a gap ---------------------------------------

@pytest.mark.asyncio
async def test_missing_gap_is_preserved_not_fabricated(tmp_path):
    candle_store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    early = _sequence(1_000 * MIN_MS, 3)          # minutes 1000-1002
    late = _sequence(1_010 * MIN_MS, 3)            # minutes 1010-1012 -- gap 1003-1009
    for c in early + late:
        candle_store.append_candle_sync("BTCUSDT", c)

    restored = await candle_store.load_history("BTCUSDT")
    assert len(restored) == 6  # exactly what was persisted, nothing interpolated

    open_times = [c.open_time for c in restored]
    gap = open_times[3] - open_times[2]
    assert gap == 8 * MIN_MS  # the real gap, untouched


# -- 10. corrupted/invalid candle handling ------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        _candle(1_000 * MIN_MS, c=math.nan),
        _candle(1_000 * MIN_MS, h=math.inf),
        _candle(1_000 * MIN_MS, v=-1.0),
        _candle(1_000 * MIN_MS, h=90.0, l=99.0),  # high < low
        _candle(0, o=1.0, h=1.0, l=1.0, c=1.0, v=1.0),  # non-positive open_time
    ],
)
def test_invalid_candles_are_rejected_not_persisted(tmp_path, bad):
    store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)

    assert is_valid_closed_candle(bad) is False
    ok = store.append_candle_sync("BTCUSDT", bad)

    assert ok is False
    assert store.count_sync("BTCUSDT") == 0


def test_load_history_skips_corrupted_row_written_directly_to_db(tmp_path):
    import sqlite3

    db_path = str(tmp_path / "candles.db")
    store = CandleStore(db_path, max_history_minutes=720)
    store.append_candle_sync("BTCUSDT", _candle(1_000 * MIN_MS))

    # Simulate external corruption bypassing append_candle's validation
    # (a negative volume -- a finite value SQLite will happily store, but
    # is_valid_closed_candle correctly rejects on read).
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO candles (symbol, open_time, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        ("BTCUSDT", 2_000 * MIN_MS, 1.0, 1.0, 1.0, 1.0, -5.0),
    )
    conn.commit()
    conn.close()

    restored = store.load_history_sync("BTCUSDT")
    assert len(restored) == 1
    assert restored[0].open_time == 1_000 * MIN_MS


# -- 11. concurrent/asynchronous persistence safety ----------------------------------

@pytest.mark.asyncio
async def test_concurrent_appends_across_symbols_are_all_persisted(tmp_path):
    store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    symbols = [f"SYM{i}USDT" for i in range(20)]

    await asyncio.gather(*[store.append_candle(sym, _candle(1_000 * MIN_MS)) for sym in symbols])

    for sym in symbols:
        assert await store.count(sym) == 1


@pytest.mark.asyncio
async def test_concurrent_appends_same_symbol_no_lost_writes(tmp_path):
    store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    candles = _sequence(1_000 * MIN_MS, 30)

    await asyncio.gather(*[store.append_candle("BTCUSDT", c) for c in candles])

    assert await store.count("BTCUSDT") == 30


# -- 12. restored history produces identical resampled timeframes ---------------------

@pytest.mark.asyncio
async def test_restored_history_produces_identical_feature_engine_output(tmp_path):
    candle_store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    seq = _sequence(1_000 * MIN_MS, 25, base_price=50.0)
    for c in seq:
        candle_store.append_candle_sync("BTCUSDT", c)

    live_store = MarketStateStore()
    await live_store.init_symbols(["BTCUSDT"])
    for c in seq:
        await live_store.update_kline("BTCUSDT", c, event_ts=c.open_time / 1000.0)

    restarted_store = MarketStateStore()
    await restarted_store.init_symbols(["BTCUSDT"])
    restored = await candle_store.load_history("BTCUSDT")
    await restarted_store.restore_candle_history("BTCUSDT", restored)

    live_state = await live_store.get("BTCUSDT")
    restarted_state = await restarted_store.get("BTCUSDT")
    assert live_state.candle_history == restarted_state.candle_history

    engine = FeatureEngine()
    live_features = engine.compute(live_state)
    restarted_features = engine.compute(restarted_state)
    assert live_features.returns == restarted_features.returns
    assert live_features.volume == restarted_features.volume
    assert live_features.volatility == restarted_features.volatility
    assert live_features.trend == restarted_features.trend


# -- 13. restored history produces identical structure inputs -------------------------

@pytest.mark.asyncio
async def test_restored_history_produces_identical_structure_result(tmp_path):
    candle_store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    seq = _sequence(1_000 * MIN_MS, 30, base_price=50.0)
    for c in seq:
        candle_store.append_candle_sync("BTCUSDT", c)

    live_store = MarketStateStore()
    await live_store.init_symbols(["BTCUSDT"])
    for c in seq:
        await live_store.update_kline("BTCUSDT", c, event_ts=c.open_time / 1000.0)

    restarted_store = MarketStateStore()
    await restarted_store.init_symbols(["BTCUSDT"])
    restored = await candle_store.load_history("BTCUSDT")
    await restarted_store.restore_candle_history("BTCUSDT", restored)

    engine = StructureEngine()
    live_result = engine.analyze(await live_store.get("BTCUSDT"))
    restarted_result = engine.analyze(await restarted_store.get("BTCUSDT"))

    assert live_result.structure == restarted_result.structure
    assert live_result.confidence == restarted_result.confidence
    assert live_result.timeframe_alignment == restarted_result.timeframe_alignment
    assert live_result.levels == restarted_result.levels


# -- 14. restored history allows outcome evaluation to continue -----------------------

@pytest.mark.asyncio
async def test_outcome_evaluation_works_against_restored_history(tmp_path):
    candle_store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    signal_ts = 1_000 * MIN_MS / 1000.0
    seq = [
        _candle(1_000 * MIN_MS, o=100.0, h=101.0, l=99.0, c=100.0),
        _candle(1_001 * MIN_MS, o=100.0, h=112.0, l=99.5, c=111.0),  # hits TP1=110
    ]
    for c in seq:
        candle_store.append_candle_sync("BTCUSDT", c)

    restarted_store = MarketStateStore()
    await restarted_store.init_symbols(["BTCUSDT"])
    restored = await candle_store.load_history("BTCUSDT")
    await restarted_store.restore_candle_history("BTCUSDT", restored)

    state = await restarted_store.get("BTCUSDT")
    relevant = candles_since(state.candle_history, signal_ts)
    outcome = compute_tp_sl_outcome("long", stop=90.0, tp1=110.0, tp2=120.0, candles=relevant)

    assert outcome["state"] == "TP1_HIT"
    assert outcome["tp1_hit_at"] == 1_001 * MIN_MS / 1000.0


# -- 15. restart does not fabricate candles --------------------------------------------

@pytest.mark.asyncio
async def test_restart_never_fabricates_extra_candles(tmp_path):
    candle_store = CandleStore(str(tmp_path / "candles.db"), max_history_minutes=720)
    seq = _sequence(1_000 * MIN_MS, 7)
    for c in seq:
        candle_store.append_candle_sync("BTCUSDT", c)

    restored = await candle_store.load_history("BTCUSDT")
    assert len(restored) == 7
    assert restored == seq  # exact values, exact order, no additions


# -- 16. persistence failure does not corrupt in-memory state --------------------------

class _RaisingCandleStore:
    async def append_candle(self, symbol, candle):
        raise sqlite3_error()


def sqlite3_error():
    import sqlite3
    return sqlite3.OperationalError("simulated disk failure")


@pytest.mark.asyncio
async def test_persist_failure_does_not_prevent_in_memory_update():
    market_store = MarketStateStore()
    failing_candle_store = _RaisingCandleStore()

    message = (
        '{"data": {"e": "kline", "s": "BTCUSDT", "k": '
        '{"t": %d, "o": "100.0", "h": "101.0", "l": "99.0", "c": "100.5", "v": "10.0", "x": true}}}'
    ) % (1_000 * MIN_MS)

    # Must not raise, even though the candle_store.append_candle call inside will.
    await _process_message(message, market_store, failing_candle_store)

    state = await market_store.get("BTCUSDT")
    assert state.last_candle_1m is not None
    assert state.last_candle_1m.open == 100.0
    assert len(state.candle_history) == 1
