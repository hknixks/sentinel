"""
Phase 1 configuration.

No secrets are required for public Binance USDT-perpetual market data.
Values are read from the environment (with sensible defaults) so the
module never hard-codes deployment-specific settings, and so that adding
authenticated endpoints later is a config change, not a code change.
"""

from __future__ import annotations

import os


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


BINANCE_FAPI_BASE_URL = _env_str("BINANCE_FAPI_BASE_URL", "https://fapi.binance.com")
BINANCE_FSTREAM_BASE_URL = _env_str("BINANCE_FSTREAM_BASE_URL", "wss://fstream.binance.com")

# Max symbols per combined-stream WebSocket connection. Binance's documented
# limit is 1024 streams per connection; we use 2 streams (ticker + kline) per
# symbol, so cap symbols per connection well under that.
MAX_SYMBOLS_PER_CONNECTION = _env_str("MAX_SYMBOLS_PER_CONNECTION", "400")

WS_RECONNECT_BASE_DELAY_SECONDS = _env_float("WS_RECONNECT_BASE_DELAY_SECONDS", 1.0)
WS_RECONNECT_MAX_DELAY_SECONDS = _env_float("WS_RECONNECT_MAX_DELAY_SECONDS", 30.0)
WS_PING_TIMEOUT_SECONDS = _env_float("WS_PING_TIMEOUT_SECONDS", 20.0)

LOG_LEVEL = _env_str("LOG_LEVEL", "INFO")
