from __future__ import annotations

import pytest

from sentinel.market_state import Candle, MarketStateStore


@pytest.mark.asyncio
async def test_init_symbols_creates_empty_states():
    store = MarketStateStore()
    await store.init_symbols(["BTCUSDT", "ETHUSDT"])

    snapshot = await store.snapshot()

    assert set(snapshot.keys()) == {"BTCUSDT", "ETHUSDT"}
    assert snapshot["BTCUSDT"].price is None
    assert snapshot["BTCUSDT"].symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_update_ticker_sets_price_and_volume():
    store = MarketStateStore()

    await store.update_ticker("BTCUSDT", price=65000.5, volume_24h=12345.6, event_ts=1000.0)

    state = await store.get("BTCUSDT")
    assert state is not None
    assert state.price == 65000.5
    assert state.volume_24h == 12345.6
    assert state.last_update_ts == 1000.0


@pytest.mark.asyncio
async def test_update_ticker_is_idempotent_per_symbol_and_does_not_affect_others():
    store = MarketStateStore()
    await store.init_symbols(["BTCUSDT", "ETHUSDT"])

    await store.update_ticker("BTCUSDT", price=100.0, volume_24h=1.0, event_ts=1.0)
    await store.update_ticker("BTCUSDT", price=200.0, volume_24h=2.0, event_ts=2.0)

    btc = await store.get("BTCUSDT")
    eth = await store.get("ETHUSDT")

    assert btc.price == 200.0
    assert btc.volume_24h == 2.0
    assert eth.price is None


@pytest.mark.asyncio
async def test_update_kline_sets_candle_without_clobbering_price():
    store = MarketStateStore()
    await store.update_ticker("BTCUSDT", price=100.0, volume_24h=1.0, event_ts=1.0)

    candle = Candle(
        open_time=1_700_000_000_000,
        open=99.0,
        high=101.0,
        low=98.5,
        close=100.5,
        volume=42.0,
        is_closed=False,
    )
    await store.update_kline("BTCUSDT", candle, event_ts=5.0)

    state = await store.get("BTCUSDT")
    assert state.last_candle_1m == candle
    assert state.price == 100.0  # untouched by kline update
    assert state.last_update_ts == 5.0


@pytest.mark.asyncio
async def test_update_kline_before_ticker_still_creates_state():
    store = MarketStateStore()

    candle = Candle(
        open_time=1,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
        is_closed=True,
    )
    await store.update_kline("SOLUSDT", candle, event_ts=3.0)

    state = await store.get("SOLUSDT")
    assert state is not None
    assert state.last_candle_1m == candle
    assert state.price is None


@pytest.mark.asyncio
async def test_get_unknown_symbol_returns_none():
    store = MarketStateStore()
    assert await store.get("DOESNOTEXIST") is None


@pytest.mark.asyncio
async def test_symbol_count():
    store = MarketStateStore()
    await store.init_symbols(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert await store.symbol_count() == 3

    await store.update_ticker("XRPUSDT", price=1.0, volume_24h=1.0)
    assert await store.symbol_count() == 4


@pytest.mark.asyncio
async def test_snapshot_returns_independent_copy():
    store = MarketStateStore()
    await store.init_symbols(["BTCUSDT"])

    snapshot = await store.snapshot()
    snapshot["BTCUSDT"] = "mutated"

    fresh = await store.snapshot()
    assert fresh["BTCUSDT"] != "mutated"
