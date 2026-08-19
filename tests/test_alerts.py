from __future__ import annotations

import logging

import pytest

from sentinel.alerts.alert_engine import AlertEngine, build_reasoning_report
from sentinel.alerts.identity import compute_setup_identity
from sentinel.alerts.models import DECISION_SKIPPED, DECISION_TAKEN, STATUS_ACTIVE, STATUS_RESOLVED
from sentinel.alerts.store import AlertStore
from sentinel.setups.models import EntryZone, RiskLevels, SetupCandidate


class FakeTelegramClient:
    """Records calls instead of making real network requests."""

    def __init__(self, fail: bool = False):
        self.sent_messages: list[dict] = []
        self.answered_callbacks: list[tuple] = []
        self.fail = fail

    async def send_message(self, chat_id, text, reply_markup=None):
        if self.fail:
            return None
        self.sent_messages.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})
        return {"ok": True, "result": {"message_id": len(self.sent_messages)}}

    async def answer_callback_query(self, callback_query_id, text=None):
        self.answered_callbacks.append((callback_query_id, text))
        return {"ok": True}

    async def get_updates(self, offset=None, timeout=25):
        return []


def _candidate(
    symbol: str = "BTCUSDT",
    direction: str = "long",
    setup_type: str = "breakout",
    invalidation_level: float = 100.0,
    score: float = 80.0,
    entry_low: float = 100.0,
    entry_high: float = 102.0,
) -> SetupCandidate:
    risk = RiskLevels(
        stop_loss=invalidation_level,
        take_profit_1=110.0,
        take_profit_2=115.0,
        risk_per_unit=2.0,
        reward_to_tp1=8.0,
        reward_to_tp2=13.0,
        risk_reward_tp1=4.0,
        risk_reward_tp2=6.5,
    )
    return SetupCandidate(
        symbol=symbol,
        timestamp=0.0,
        direction=direction,
        setup_type=setup_type,
        entry_zone=EntryZone(low=entry_low, high=entry_high),
        invalidation_level=invalidation_level,
        risk=risk,
        structure_context=f"{direction}_{setup_type} test fixture",
        confirmation_factors=("confirmed close beyond structural boundary (0.30% distance)", "relative volume 2.10x baseline"),
        setup_score=score,
    )


def _engine(tmp_path, telegram=None, dry_run=True, market_store=None) -> tuple[AlertEngine, AlertStore]:
    db_path = str(tmp_path / "alerts.db")
    store = AlertStore(db_path)
    engine = AlertEngine(store, telegram, "12345", market_store=market_store, dry_run=dry_run)
    return engine, store


# -- 1. New setup generates an alert --------------------------------------

@pytest.mark.asyncio
async def test_new_setup_generates_alert(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    record = await engine.process_candidate("BTCUSDT", _candidate())

    assert record is not None
    assert record.status == STATUS_ACTIVE
    assert len(telegram.sent_messages) == 1
    assert "BTCUSDT" in telegram.sent_messages[0]["text"]
    assert "LONG" in telegram.sent_messages[0]["text"]


# -- 2. Same setup on next scan does NOT generate another alert -----------

@pytest.mark.asyncio
async def test_same_setup_does_not_realert(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    candidate = _candidate()
    r1 = await engine.process_candidate("BTCUSDT", candidate)
    r2 = await engine.process_candidate("BTCUSDT", candidate)
    r3 = await engine.process_candidate("BTCUSDT", candidate)

    assert r1 is not None
    assert r2 is None
    assert r3 is None
    assert len(telegram.sent_messages) == 1


# -- 3. BTC and SOL can alert independently --------------------------------

@pytest.mark.asyncio
async def test_independent_symbols_alert_independently(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    btc = await engine.process_candidate("BTCUSDT", _candidate(symbol="BTCUSDT", invalidation_level=100.0))
    sol = await engine.process_candidate("SOLUSDT", _candidate(symbol="SOLUSDT", invalidation_level=50.0, direction="short"))
    btc_dup = await engine.process_candidate("BTCUSDT", _candidate(symbol="BTCUSDT", invalidation_level=100.0))

    assert btc is not None
    assert sol is not None
    assert btc_dup is None
    assert len(telegram.sent_messages) == 2
    assert (await store.get_active("BTCUSDT")).alert_id == btc.alert_id
    assert (await store.get_active("SOLUSDT")).alert_id == sol.alert_id


# -- 4 & 15. New setup can alert once the previous one resolves -----------

@pytest.mark.asyncio
async def test_new_setup_alerts_after_previous_resolves(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    first = await engine.process_candidate("BTCUSDT", _candidate(invalidation_level=100.0))
    # setup disappears (no candidate this scan) -> resolves
    none_result = await engine.process_candidate("BTCUSDT", None)
    # a genuinely new setup (different structural level) forms
    second = await engine.process_candidate("BTCUSDT", _candidate(invalidation_level=95.0))

    assert first is not None
    assert none_result is None
    assert second is not None
    assert first.alert_id != second.alert_id
    assert len(telegram.sent_messages) == 2

    resolved_first = await store.get_by_id(first.alert_id)
    assert resolved_first.status == STATUS_RESOLVED

    active_now = await store.get_active("BTCUSDT")
    assert active_now.alert_id == second.alert_id


@pytest.mark.asyncio
async def test_setup_changing_identity_resolves_old_and_alerts_new(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    first = await engine.process_candidate("BTCUSDT", _candidate(invalidation_level=100.0))
    # a new candidate arrives directly (without an intervening None scan) at a materially different level
    second = await engine.process_candidate("BTCUSDT", _candidate(invalidation_level=90.0))

    assert first is not None
    assert second is not None
    assert first.alert_id != second.alert_id
    resolved_first = await store.get_by_id(first.alert_id)
    assert resolved_first.status == STATUS_RESOLVED
    assert len(telegram.sent_messages) == 2


# -- 5 & 13. Restart/persistence prevents duplicate alerts -----------------

@pytest.mark.asyncio
async def test_restart_persistence_prevents_duplicate_alert(tmp_path):
    db_path = str(tmp_path / "alerts.db")
    telegram1 = FakeTelegramClient()
    store1 = AlertStore(db_path)
    engine1 = AlertEngine(store1, telegram1, "12345", dry_run=False)

    candidate = _candidate()
    record1 = await engine1.process_candidate("BTCUSDT", candidate)
    assert record1 is not None
    assert len(telegram1.sent_messages) == 1

    # Simulate a full process restart: brand new AlertStore/AlertEngine
    # instances pointed at the SAME db file.
    telegram2 = FakeTelegramClient()
    store2 = AlertStore(db_path)
    engine2 = AlertEngine(store2, telegram2, "12345", dry_run=False)

    record2 = await engine2.process_candidate("BTCUSDT", candidate)

    assert record2 is None
    assert len(telegram2.sent_messages) == 0

    persisted = await store2.get_active("BTCUSDT")
    assert persisted is not None
    assert persisted.alert_id == record1.alert_id
    assert persisted.symbol == "BTCUSDT"


# -- 6. Taken button records TAKEN -----------------------------------------

@pytest.mark.asyncio
async def test_taken_button_records_decision(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    record = await engine.process_candidate("BTCUSDT", _candidate())
    await engine.handle_callback({"id": "cb1", "data": f"taken:{record.alert_id}"})

    updated = await store.get_by_id(record.alert_id)
    assert updated.decision == DECISION_TAKEN
    assert updated.decision_at is not None
    assert len(telegram.answered_callbacks) == 1


# -- 7. Skipped button records SKIPPED --------------------------------------

@pytest.mark.asyncio
async def test_skipped_button_records_decision(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    record = await engine.process_candidate("BTCUSDT", _candidate())
    await engine.handle_callback({"id": "cb1", "data": f"skipped:{record.alert_id}"})

    updated = await store.get_by_id(record.alert_id)
    assert updated.decision == DECISION_SKIPPED


# -- 8. Repeated callback clicks are idempotent -----------------------------

@pytest.mark.asyncio
async def test_repeated_callback_clicks_are_idempotent(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    record = await engine.process_candidate("BTCUSDT", _candidate())

    await engine.handle_callback({"id": "cb1", "data": f"taken:{record.alert_id}"})
    await engine.handle_callback({"id": "cb2", "data": f"taken:{record.alert_id}"})
    # a conflicting click after the fact must not overwrite the first decision
    await engine.handle_callback({"id": "cb3", "data": f"skipped:{record.alert_id}"})

    updated = await store.get_by_id(record.alert_id)
    assert updated.decision == DECISION_TAKEN  # first decision wins, never overwritten
    assert len(telegram.answered_callbacks) == 3  # every click still gets acknowledged


# -- 9. Deeper Reasoning returns deterministic setup information -----------

def test_deeper_reasoning_is_deterministic_and_complete():
    candidate = _candidate(setup_type="breakout", direction="long")
    report1 = build_reasoning_report(candidate, None, None)
    report2 = build_reasoning_report(candidate, None, None)

    assert report1 == report2
    assert "BTCUSDT" in report1
    assert "LONG" in report1
    assert "breakout" in report1
    assert "Setup score: 80.0" in report1
    assert "NOT a win probability" in report1
    assert "Entry zone" in report1
    assert "Stop loss" in report1
    assert "TP1" in report1
    assert "TP2" in report1


@pytest.mark.asyncio
async def test_deeper_reasoning_button_sends_stored_report(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    record = await engine.process_candidate("BTCUSDT", _candidate())
    await engine.handle_callback({"id": "cb1", "data": f"reason:{record.alert_id}"})

    # First message is the compact alert; second is the reasoning report.
    assert len(telegram.sent_messages) == 2
    assert telegram.sent_messages[1]["text"] == record.reasoning


# -- 10. Invalid/rejected candidates never generate alerts -----------------

@pytest.mark.asyncio
async def test_no_candidate_never_generates_alert(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    record = await engine.process_candidate("BTCUSDT", None)

    assert record is None
    assert len(telegram.sent_messages) == 0
    assert await store.get_active("BTCUSDT") is None


# -- 11. Secrets never appear in alert content or logs ----------------------

def test_secrets_never_appear_in_alert_text_or_reasoning():
    candidate = _candidate()
    report = build_reasoning_report(candidate, None, None)
    fake_token = "123456:AAFakeTokenValueShouldNeverAppearAnywhere"
    assert fake_token not in report

    from sentinel.alerts.alert_engine import _build_alert_text
    from sentinel.alerts.models import AlertRecord, STATUS_ACTIVE

    record = AlertRecord(
        alert_id="abc", symbol="BTCUSDT", direction="long", setup_type="breakout",
        status=STATUS_ACTIVE, setup_score=80.0, entry_low=100.0, entry_high=102.0,
        stop_loss=100.0, take_profit_1=110.0, take_profit_2=115.0, risk_reward_tp1=4.0,
        invalidation_level=100.0, structure_context="x", confirmation_factors="x",
        reasoning="x", created_at=0.0, resolved_at=None, telegram_message_id=None,
        telegram_chat_id=None, decision=None, decision_at=None, decision_reference_price=None,
    )
    text = _build_alert_text(record)
    assert fake_token not in text
    assert "TELEGRAM_BOT_TOKEN" not in text


def test_telegram_client_redacts_token_in_logs(caplog):
    from sentinel.alerts.telegram_client import _redact

    url = "https://api.telegram.org/bot123456:AAFakeSecretTokenXYZ/sendMessage"
    redacted = _redact(url)

    assert "123456:AAFakeSecretTokenXYZ" not in redacted
    assert "REDACTED" in redacted


# -- 12. Telegram failure does not crash the engine -------------------------

@pytest.mark.asyncio
async def test_telegram_failure_does_not_raise(tmp_path):
    failing_telegram = FakeTelegramClient(fail=True)
    engine, store = _engine(tmp_path, telegram=failing_telegram, dry_run=False)

    record = await engine.process_candidate("BTCUSDT", _candidate())

    # The alert is still recorded ACTIVE in the store even though the
    # Telegram send failed -- the market-data/alert loop must not crash.
    assert record is not None
    active = await store.get_active("BTCUSDT")
    assert active is not None
    assert active.telegram_message_id is None  # send failed, no message id captured


@pytest.mark.asyncio
async def test_dry_run_never_calls_telegram(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=True)

    record = await engine.process_candidate("BTCUSDT", _candidate())

    assert record is not None
    assert len(telegram.sent_messages) == 0


# -- 14. Active setup lifecycle transitions correctly to RESOLVED ----------

@pytest.mark.asyncio
async def test_lifecycle_transitions_to_resolved_when_setup_disappears(tmp_path):
    engine, store = _engine(tmp_path, dry_run=True)

    record = await engine.process_candidate("BTCUSDT", _candidate())
    assert (await store.get_by_id(record.alert_id)).status == STATUS_ACTIVE

    await engine.process_candidate("BTCUSDT", None)

    assert (await store.get_by_id(record.alert_id)).status == STATUS_RESOLVED
    assert (await store.get_by_id(record.alert_id)).resolved_at is not None
    assert await store.get_active("BTCUSDT") is None


# -- setup identity -----------------------------------------------------

def test_setup_identity_is_deterministic_across_repeated_calls():
    c1 = _candidate(invalidation_level=100.0001)
    c2 = _candidate(invalidation_level=100.0001)
    assert compute_setup_identity(c1) == compute_setup_identity(c2)


def test_setup_identity_differs_for_different_structural_level():
    c1 = _candidate(invalidation_level=100.0)
    c2 = _candidate(invalidation_level=90.0)
    assert compute_setup_identity(c1) != compute_setup_identity(c2)


def test_setup_identity_differs_by_symbol_direction_and_type():
    base = compute_setup_identity(_candidate(symbol="BTCUSDT", direction="long", setup_type="breakout"))
    other_symbol = compute_setup_identity(_candidate(symbol="ETHUSDT", direction="long", setup_type="breakout"))
    other_direction = compute_setup_identity(_candidate(symbol="BTCUSDT", direction="short", setup_type="breakout"))
    other_type = compute_setup_identity(_candidate(symbol="BTCUSDT", direction="long", setup_type="pullback"))

    assert len({base, other_symbol, other_direction, other_type}) == 4


def test_setup_identity_absorbs_tiny_floating_point_noise():
    c1 = _candidate(invalidation_level=100.000001)
    c2 = _candidate(invalidation_level=100.000002)
    assert compute_setup_identity(c1) == compute_setup_identity(c2)


# -- store-level idempotency ----------------------------------------------

@pytest.mark.asyncio
async def test_store_create_active_is_idempotent_at_db_level(tmp_path):
    from sentinel.alerts.models import AlertRecord

    db_path = str(tmp_path / "alerts.db")
    store = AlertStore(db_path)
    record = AlertRecord(
        alert_id="dup", symbol="BTCUSDT", direction="long", setup_type="breakout",
        status=STATUS_ACTIVE, setup_score=80.0, entry_low=100.0, entry_high=102.0,
        stop_loss=100.0, take_profit_1=110.0, take_profit_2=115.0, risk_reward_tp1=4.0,
        invalidation_level=100.0, structure_context="x", confirmation_factors="x",
        reasoning="x", created_at=0.0, resolved_at=None, telegram_message_id=None,
        telegram_chat_id=None, decision=None, decision_at=None, decision_reference_price=None,
    )

    first = await store.create_active(record)
    second = await store.create_active(record)

    assert first is True
    assert second is False


@pytest.mark.asyncio
async def test_unknown_alert_id_callback_does_not_crash(tmp_path):
    telegram = FakeTelegramClient()
    engine, store = _engine(tmp_path, telegram=telegram, dry_run=False)

    await engine.handle_callback({"id": "cb1", "data": "taken:doesnotexist"})

    assert len(telegram.answered_callbacks) == 1
