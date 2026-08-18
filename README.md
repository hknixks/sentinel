# SENTINEL

Real-time market intelligence and trade signal system for crypto perpetual
futures (FX support planned later).

**SENTINEL is alert-only.** It never places trades, never withdraws or
transfers funds, and never connects to a wallet. It has no order-execution
functionality, and none is planned.

## Phase 1: Real-time market-data engine

This phase builds only the project foundation and a real-time Binance
USDT-M perpetual futures market-data engine:

- Dynamically discovers every USDT-margined perpetual symbol from Binance
  (no hard-coded coin list).
- Streams live price, 24h volume, and 1-minute candle updates over
  WebSocket (push-based, not a polling cron).
- Maintains an in-memory live state per symbol.
- Automatically reconnects with exponential backoff on disconnect.
- Never crashes on a malformed message -- it's logged and skipped.

No trade prediction, scoring, news, Telegram alerts, position tracking, or
order execution exist yet -- those are later phases.

## Requirements

- Python 3.11+
- No Binance API key/secret needed: Phase 1 only uses public market data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # includes runtime deps + pytest
cp .env.example .env                  # optional; defaults work out of the box
```

## Running the data engine

```bash
PYTHONPATH=src python3 -m sentinel.main
```

This will:

1. Call the public Binance Futures `exchangeInfo` REST endpoint to discover
   every actively-trading USDT-perpetual symbol.
2. Open one or more combined WebSocket connections (`miniTicker` +
   `kline_1m` streams per symbol) to `fstream.binance.com`.
3. Update an in-memory state store as data arrives.
4. Log a snapshot of live state every ~15 seconds so you can see it's
   working.

Stop it with `Ctrl+C`.

## Running tests

```bash
PYTHONPATH=src pytest
```

Tests cover the core market-state store and WebSocket message-parsing
logic directly, with no network calls.

## Project structure

```
src/sentinel/
├── config.py           # env-based settings (no secrets required in Phase 1)
├── logging_setup.py     # logging configuration
├── main.py              # entrypoint wiring discovery -> WS -> state
├── market_state.py      # in-memory state model (pure logic, unit tested)
└── binance/
    ├── symbols.py        # REST-based USDT-perpetual symbol discovery
    └── ws_client.py       # WebSocket client with reconnect + malformed-message handling
tests/
├── test_market_state.py
└── test_ws_message_parsing.py
```

## Configuration

All configuration is via environment variables (see `.env.example`).
Phase 1 requires no secrets. When authenticated endpoints are needed in a
later phase, credentials will be loaded from environment variables only
and must never be committed to Git (`.env` is git-ignored).
