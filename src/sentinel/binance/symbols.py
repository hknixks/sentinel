"""
Dynamic discovery of tradable Binance USDT-M perpetual futures symbols.

Uses the public exchangeInfo REST endpoint -- no API key required.
"""

from __future__ import annotations

import logging

import httpx

from sentinel.config import BINANCE_FAPI_BASE_URL

logger = logging.getLogger(__name__)

EXCHANGE_INFO_PATH = "/fapi/v1/exchangeInfo"


async def discover_usdt_perpetual_symbols() -> list[str]:
    """Return every actively-trading USDT-margined perpetual symbol."""
    url = f"{BINANCE_FAPI_BASE_URL}{EXCHANGE_INFO_PATH}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()

    symbols: list[str] = []
    for entry in payload.get("symbols", []):
        if (
            entry.get("contractType") == "PERPETUAL"
            and entry.get("quoteAsset") == "USDT"
            and entry.get("status") == "TRADING"
        ):
            symbols.append(entry["symbol"])

    logger.info("Discovered %d USDT-perpetual symbols", len(symbols))
    return sorted(symbols)
