"""
Alert Engine: converts validated SetupCandidate objects into compact
Telegram alerts, enforcing exactly one active alert per setup identity
per symbol, with restart-safe persistence and idempotent callback
handling.

Alert-only. Nothing here executes, sizes, or places a trade, and nothing
connects to a wallet. TAKEN/SKIPPED only records the user's stated
decision for later review.
"""

from __future__ import annotations

import logging
import time

from sentinel.alerts.identity import compute_setup_identity
from sentinel.alerts.models import AlertRecord, DECISION_SKIPPED, DECISION_TAKEN, STATUS_ACTIVE
from sentinel.alerts.store import AlertStore
from sentinel.alerts.telegram_client import TelegramClient
from sentinel.market_state import MarketStateStore
from sentinel.scanner.features import MarketFeatures
from sentinel.setups.models import SetupCandidate
from sentinel.structure.models import StructureResult

logger = logging.getLogger(__name__)


def build_reasoning_report(
    candidate: SetupCandidate,
    structure_result: StructureResult | None,
    features: MarketFeatures | None,
) -> str:
    """Deterministic, structured explanation built once at alert-creation
    time from data the existing system already computed -- no LLM, no
    re-interpretation, nothing invented. This exact string is what a
    later 'Deeper Reasoning' click returns, unchanged."""
    lines = [
        f"{candidate.symbol} {candidate.direction.upper()} {candidate.setup_type}",
        f"Setup score: {candidate.setup_score:.1f} (evidence strength, NOT a win probability)",
        "",
    ]
    if structure_result is not None:
        lines += [
            f"Structure pattern: {structure_result.structure.pattern}",
            f"Structure confidence: {structure_result.confidence:.1f}",
            f"Timeframe alignment: {structure_result.timeframe_alignment}",
            f"Directional bias: {structure_result.directional_bias}",
            "",
        ]
    if features is not None:
        lines += [
            f"Momentum: 1m={features.returns.return_1m} 5m={features.returns.return_5m} "
            f"15m={features.returns.return_15m} 1h={features.returns.return_1h}",
            f"Volume: relative={features.volume.relative_volume} "
            f"acceleration={features.volume.volume_acceleration}",
            f"Volatility: expansion={features.volatility.volatility_expansion} "
            f"rolling={features.volatility.rolling_volatility}",
            "",
        ]
    lines += [
        f"Entry zone: {candidate.entry_zone.low:.6g} - {candidate.entry_zone.high:.6g}",
        f"Stop loss: {candidate.risk.stop_loss:.6g}",
        f"TP1: {candidate.risk.take_profit_1:.6g} (R:R {candidate.risk.risk_reward_tp1:.2f})",
    ]
    if candidate.risk.take_profit_2 is not None:
        lines.append(f"TP2: {candidate.risk.take_profit_2:.6g} (R:R {candidate.risk.risk_reward_tp2:.2f})")
    lines += ["", "Confirmation factors:"]
    lines += [f"- {f}" for f in candidate.confirmation_factors]
    lines += ["", f"Context: {candidate.structure_context}"]
    return "\n".join(lines)


def _build_alert_text(record: AlertRecord) -> str:
    # Compact, one line -- the full report lives behind "Deeper Reasoning".
    # The example format in the brief ("+25% from entry") describes
    # unrealized PnL, which doesn't exist yet for a freshly generated
    # setup (no entry has happened). We show the setup's own score/R:R
    # instead, kept to the same one-line compactness.
    return (
        f"\U0001F6A8 {record.symbol}: {record.direction.upper()} {record.setup_type} "
        f"(score {record.setup_score:.0f}, R:R {record.risk_reward_tp1:.1f})"
    )


def _build_keyboard(alert_id: str) -> dict:
    return {
        "inline_keyboard": [
            [{"text": "\U0001F9E0 Deeper Reasoning", "callback_data": f"reason:{alert_id}"}],
            [
                {"text": "✅ Took Trade", "callback_data": f"taken:{alert_id}"},
                {"text": "❌ Skipped", "callback_data": f"skipped:{alert_id}"},
            ],
        ]
    }


class AlertEngine:
    def __init__(
        self,
        store: AlertStore,
        telegram: TelegramClient | None,
        chat_id: str | None,
        market_store: MarketStateStore | None = None,
        dry_run: bool = True,
    ) -> None:
        self._store = store
        self._telegram = telegram
        self._chat_id = chat_id
        self._market_store = market_store
        self._dry_run = dry_run

    async def process_candidate(
        self,
        symbol: str,
        candidate: SetupCandidate | None,
        structure_result: StructureResult | None = None,
        features: MarketFeatures | None = None,
    ) -> AlertRecord | None:
        """Called once per symbol per scan with that symbol's current
        setup candidate (or None if no valid setup exists right now).
        Returns the new AlertRecord if a fresh alert was created this
        call, else None (no candidate, or a duplicate of the already-
        active setup)."""
        current = await self._store.get_active(symbol)

        if candidate is None:
            if current is not None:
                await self._store.resolve(current.alert_id)
                logger.info("Alert resolved (setup no longer valid): %s %s", symbol, current.alert_id)
            return None

        identity = compute_setup_identity(candidate)

        if current is not None and current.alert_id == identity:
            return None  # same setup already alerted -- no duplicate

        if current is not None and current.alert_id != identity:
            await self._store.resolve(current.alert_id)
            logger.info("Alert resolved (setup changed): %s %s -> %s", symbol, current.alert_id, identity)

        reasoning = build_reasoning_report(candidate, structure_result, features)
        now = time.time()
        record = AlertRecord(
            alert_id=identity,
            symbol=candidate.symbol,
            direction=candidate.direction,
            setup_type=candidate.setup_type,
            status=STATUS_ACTIVE,
            setup_score=candidate.setup_score,
            entry_low=candidate.entry_zone.low,
            entry_high=candidate.entry_zone.high,
            stop_loss=candidate.risk.stop_loss,
            take_profit_1=candidate.risk.take_profit_1,
            take_profit_2=candidate.risk.take_profit_2,
            risk_reward_tp1=candidate.risk.risk_reward_tp1,
            invalidation_level=candidate.invalidation_level,
            structure_context=candidate.structure_context,
            confirmation_factors="|".join(candidate.confirmation_factors),
            reasoning=reasoning,
            created_at=now,
            resolved_at=None,
            telegram_message_id=None,
            telegram_chat_id=None,
            decision=None,
            decision_at=None,
            decision_reference_price=None,
        )

        created = await self._store.create_active(record)
        if not created:
            # Another call already created this exact identity (race or
            # restart replay) -- idempotent no-op, no duplicate send.
            return None

        await self._send_alert(record)
        return record

    async def _send_alert(self, record: AlertRecord) -> None:
        text = _build_alert_text(record)
        keyboard = _build_keyboard(record.alert_id)

        if self._dry_run or self._telegram is None or not self._chat_id:
            logger.info("[DRY RUN] Telegram alert: %s", text)
            return

        response = await self._telegram.send_message(self._chat_id, text, keyboard)
        if response and response.get("ok"):
            message_id = response["result"]["message_id"]
            await self._store.set_telegram_message_id(record.alert_id, self._chat_id, message_id)
        else:
            logger.warning("Telegram alert send failed or unacknowledged for %s", record.symbol)

    async def handle_callback(self, callback_query: dict) -> None:
        """Idempotent: repeated clicks (even from a stale/duplicated
        Telegram update) never corrupt state -- record_decision only
        ever accepts the first decision, and re-sending Deeper Reasoning
        is always safe since it's read-only."""
        callback_id = callback_query.get("id")
        data = callback_query.get("data", "")

        if ":" not in data:
            await self._ack(callback_id)
            return

        action, alert_id = data.split(":", 1)
        record = await self._store.get_by_id(alert_id)
        if record is None:
            await self._ack(callback_id, "Alert not found")
            return

        if action == "reason":
            await self._ack(callback_id)
            if self._telegram is not None and record.telegram_chat_id:
                await self._telegram.send_message(record.telegram_chat_id, record.reasoning)
            return

        if action in ("taken", "skipped"):
            decision = DECISION_TAKEN if action == "taken" else DECISION_SKIPPED
            reference_price = await self._current_price(record.symbol)
            recorded = await self._store.record_decision(alert_id, decision, reference_price)
            if recorded:
                ack_text = "Recorded: Took Trade" if action == "taken" else "Recorded: Skipped"
                logger.info("Decision recorded: %s %s -> %s", record.symbol, alert_id, decision)
            else:
                ack_text = "Already recorded"
            await self._ack(callback_id, ack_text)
            return

        await self._ack(callback_id)

    async def _current_price(self, symbol: str) -> float | None:
        if self._market_store is None:
            return None
        state = await self._market_store.get(symbol)
        return state.price if state else None

    async def _ack(self, callback_id: str | None, text: str | None = None) -> None:
        if self._telegram is not None and callback_id:
            await self._telegram.answer_callback_query(callback_id, text)
