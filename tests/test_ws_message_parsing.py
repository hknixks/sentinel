from __future__ import annotations

import pytest

from sentinel.binance.ws_client import _process_message
from sentinel.market_state import MarketStateStore

MINI_TICKER_MESSAGE = """
{
  "stream": "btcusdt@miniTicker",
  "data": {
    "e": "24hrMiniTicker",
    "E": 1700000000000,
    "s": "BTCUSDT",
    "c": "65000.10",
    "o": "64000.00",
    "h": "66000.00",
    "l": "63000.00",
    "v": "12345.678",
    "q": "800000000.00"
  }
}
"""

KLINE_MESSAGE = """
{
  "stream": "btcusdt@kline_1m",
  "data": {
    "e": "kline",
    "E": 1700000000000,
    "s": "BTCUSDT",
    "k": {
      "t": 1699999940000,
      "T": 1699999999999,
      "s": "BTCUSDT",
      "i": "1m",
      "o": "64990.00",
      "c": "65000.10",
      "h": "65010.00",
      "l": "64980.00",
      "v": "10.5",
      "x": false
    }
  }
}
"""


@pytest.mark.asyncio
async def test_process_message_updates_ticker():
    store = MarketStateStore()

    await _process_message(MINI_TICKER_MESSAGE, store)

    state = await store.get("BTCUSDT")
    assert state is not None
    assert state.price == 65000.10
    assert state.volume_24h == 12345.678


@pytest.mark.asyncio
async def test_process_message_updates_kline():
    store = MarketStateStore()

    await _process_message(KLINE_MESSAGE, store)

    state = await store.get("BTCUSDT")
    assert state is not None
    assert state.last_candle_1m is not None
    assert state.last_candle_1m.open == 64990.00
    assert state.last_candle_1m.close == 65000.10
    assert state.last_candle_1m.is_closed is False


@pytest.mark.asyncio
async def test_process_message_ignores_unknown_event_type():
    store = MarketStateStore()

    await _process_message('{"data": {"e": "somethingElse"}}', store)

    assert await store.symbol_count() == 0


@pytest.mark.asyncio
async def test_process_message_does_not_raise_on_invalid_json():
    store = MarketStateStore()

    # Must not raise -- a single malformed message should never crash the consumer.
    await _process_message("{not valid json", store)

    assert await store.symbol_count() == 0


@pytest.mark.asyncio
async def test_process_message_does_not_raise_on_missing_fields():
    store = MarketStateStore()

    await _process_message('{"data": {"e": "24hrMiniTicker", "s": "BTCUSDT"}}', store)

    # 'c' and 'v' are missing -- should be logged and skipped, not raised.
    assert await store.symbol_count() == 0
