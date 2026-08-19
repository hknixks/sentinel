"""
Minimal async Telegram Bot API client.

Long-polls getUpdates for callback queries rather than requiring a
public HTTPS webhook -- the simplest fit for a single-process VPS with
no reverse proxy assumed. All network calls are async (httpx), so they
never block the WebSocket/market-data loop, and every call swallows its
own exceptions and returns None/[] rather than propagating -- a Telegram
outage must never crash the market-data engine.

The bot token is never logged. Any URL that gets logged has the token
redacted first.
"""

from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

_TOKEN_IN_URL_RE = re.compile(r"/bot[^/]+/")


def _redact(url: str) -> str:
    return _TOKEN_IN_URL_RE.sub("/bot***REDACTED***/", url)


class TelegramClient:
    def __init__(self, token: str, timeout: float = 10.0) -> None:
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._timeout = timeout

    async def send_message(self, chat_id: str, text: str, reply_markup: dict | None = None) -> dict | None:
        payload: dict = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return await self._post("sendMessage", payload)

    async def edit_message_reply_markup(self, chat_id: str, message_id: int, reply_markup: dict | None) -> dict | None:
        payload: dict = {"chat_id": chat_id, "message_id": message_id}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return await self._post("editMessageReplyMarkup", payload)

    async def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> dict | None:
        payload: dict = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        return await self._post("answerCallbackQuery", payload)

    async def get_updates(self, offset: int | None = None, timeout: int = 25) -> list[dict]:
        params: dict = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        url = f"{self._base_url}/getUpdates"
        try:
            async with httpx.AsyncClient(timeout=timeout + 5) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                return data.get("result", [])
        except Exception:
            logger.exception("Telegram getUpdates failed: %s", _redact(url))
            return []

    async def _post(self, method: str, payload: dict) -> dict | None:
        url = f"{self._base_url}/{method}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Telegram request failed: %s", _redact(url))
            return None
