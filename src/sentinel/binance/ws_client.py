"""
Binance USDT-M futures combined WebSocket stream client.

Connects to the public market-data stream (no API key required), consumes
miniTicker (price + 24h volume) and 1m kline events for every discovered
symbol, and feeds them into a MarketStateStore. Handles disconnects with
automatic reconnect + exponential backoff, and never lets a single
malformed message crash the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets
from websockets.exceptions import WebSocketException

from sentinel.config import (
    BINANCE_FSTREAM_BASE_URL,
    WS_PING_TIMEOUT_SECONDS,
    WS_RECONNECT_BASE_DELAY_SECONDS,
    WS_RECONNECT_MAX_DELAY_SECONDS,
)
from sentinel.market_state import Candle, MarketStateStore

logger = logging.getLogger(__name__)


def _build_stream_url(symbols: list[str]) -> str:
    streams = []
    for symbol in symbols:
        lower = symbol.lower()
        streams.append(f"{lower}@miniTicker")
        streams.append(f"{lower}@kline_1m")
    # miniTicker and kline are /market-category streams; Binance decommissioned
    # the unrouted legacy path for this category on 2026-04-23.
    return f"{BINANCE_FSTREAM_BASE_URL}/market/stream?streams={'/'.join(streams)}"


def _handle_mini_ticker(data: dict) -> tuple[str, float, float]:
    symbol = data["s"]
    price = float(data["c"])
    volume_24h = float(data["v"])
    return symbol, price, volume_24h


def _handle_kline(data: dict) -> tuple[str, Candle]:
    symbol = data["s"]
    k = data["k"]
    candle = Candle(
        open_time=int(k["t"]),
        open=float(k["o"]),
        high=float(k["h"]),
        low=float(k["l"]),
        close=float(k["c"]),
        volume=float(k["v"]),
        is_closed=bool(k["x"]),
    )
    return symbol, candle


async def _process_message(raw_message: str, store: MarketStateStore) -> None:
    try:
        envelope = json.loads(raw_message)
        data = envelope.get("data", envelope)
        event_type = data.get("e")

        if event_type == "24hrMiniTicker":
            symbol, price, volume_24h = _handle_mini_ticker(data)
            await store.update_ticker(symbol, price, volume_24h, event_ts=time.time())
        elif event_type == "kline":
            symbol, candle = _handle_kline(data)
            await store.update_kline(symbol, candle, event_ts=time.time())
        else:
            logger.debug("Ignoring unrecognized event type: %s", event_type)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Skipping malformed WS message: %s (%s)", exc, raw_message[:200])


async def run_symbol_group(
    symbols: list[str],
    store: MarketStateStore,
    stop_event: asyncio.Event,
) -> None:
    """Connect to one combined stream for a group of symbols, reconnecting forever."""
    url = _build_stream_url(symbols)
    delay = WS_RECONNECT_BASE_DELAY_SECONDS

    while not stop_event.is_set():
        try:
            logger.info("Connecting to Binance stream for %d symbols", len(symbols))
            logger.debug("Stream URL: %s", url)
            async with websockets.connect(
                url, ping_timeout=WS_PING_TIMEOUT_SECONDS
            ) as ws:
                logger.info("Connected: %d symbols", len(symbols))
                delay = WS_RECONNECT_BASE_DELAY_SECONDS  # reset backoff on success
                async for raw_message in ws:
                    await _process_message(raw_message, store)
                    if stop_event.is_set():
                        break
        except (WebSocketException, OSError, asyncio.TimeoutError) as exc:
            logger.error("WebSocket error (%d symbols): %s", len(symbols), exc)
        except Exception:
            logger.exception("Unexpected error in WS consumer for %d symbols", len(symbols))

        if stop_event.is_set():
            break

        logger.info("Reconnecting in %.1fs", delay)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass
        delay = min(delay * 2, WS_RECONNECT_MAX_DELAY_SECONDS)
