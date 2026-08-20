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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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

# How many closed 1-minute candles to retain per symbol. Must cover the
# longest lookback used by the feature engine (1h return = 60 candles back)
# plus headroom for rolling-window baselines and the structure engine's
# resampled higher-timeframe swing analysis. Raised from Phase 2's 120 to
# Phase 3's 240, then to 720 (12 hours) so the structure engine can
# accumulate the 7+ complete 1h candles genuine 1h structure requires --
# 240 minutes can never provide more than 4. Bounded, never unbounded.
MAX_CANDLE_HISTORY_MINUTES = _env_int("MAX_CANDLE_HISTORY_MINUTES", 720)

# How many of the scanner's top-ranked markets the structure engine
# analyzes per pass. Configurable rather than hard-coded so it can be
# tuned per deployment without a code change.
STRUCTURE_TOP_N = _env_int("STRUCTURE_TOP_N", 20)

# Trade setup engine thresholds (Phase 4). A setup whose resulting
# risk/reward to TP1 falls below this is rejected outright.
SETUP_MIN_RISK_REWARD = _env_float("SETUP_MIN_RISK_REWARD", 1.5)

# Breakout setups require relative volume at least this many times the
# baseline to be considered volume-confirmed.
SETUP_MIN_BREAKOUT_RELATIVE_VOLUME = _env_float("SETUP_MIN_BREAKOUT_RELATIVE_VOLUME", 1.2)

# Breakout setups require at least this much momentum (% move, best
# available timeframe) in the setup's direction.
SETUP_MIN_BREAKOUT_MOMENTUM_PCT = _env_float("SETUP_MIN_BREAKOUT_MOMENTUM_PCT", 0.05)

# --- Alert engine (Phase 5) ---
# Telegram bot token / chat id. Left empty by default -- never commit
# real values here or anywhere else; set them in a local, git-ignored
# .env. The alert engine never logs these.
TELEGRAM_BOT_TOKEN = _env_str("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = _env_str("TELEGRAM_CHAT_ID", "")

# Dry-run defaults to true: alerts are logged, never sent to Telegram,
# until this is explicitly set to false with a real bot token configured.
TELEGRAM_DRY_RUN = _env_bool("TELEGRAM_DRY_RUN", True)

# SQLite file backing the alert store -- see sentinel.alerts.store for
# why SQLite was chosen. Isolated behind AlertStore so it can be swapped
# for PostgreSQL/Redis later without touching the alert engine.
ALERT_DB_PATH = _env_str("ALERT_DB_PATH", "data/alerts.db")

# How often the alert engine re-evaluates the full market universe for
# new/changed setups.
ALERT_CHECK_INTERVAL_SECONDS = _env_float("ALERT_CHECK_INTERVAL_SECONDS", 30.0)

# --- Outcome tracking (Phase 6) ---
# SQLite file backing the outcome store -- see sentinel.outcomes.store.
# Separate database file from ALERT_DB_PATH: alerts and outcomes are
# different lifecycles/retention concerns, kept isolated.
OUTCOME_DB_PATH = _env_str("OUTCOME_DB_PATH", "data/outcomes.db")

# Maximum forward evaluation horizon per signal. A signal still PENDING
# (no TP/SL/AMBIGUOUS resolution) after this is marked EXPIRED -- the
# historical record is kept, never deleted.
OUTCOME_MAX_HORIZON_SECONDS = _env_float("OUTCOME_MAX_HORIZON_SECONDS", 14400.0)

# How often the outcome tracker re-evaluates pending signals against
# current candle history.
OUTCOME_EVALUATION_INTERVAL_SECONDS = _env_float("OUTCOME_EVALUATION_INTERVAL_SECONDS", 60.0)
