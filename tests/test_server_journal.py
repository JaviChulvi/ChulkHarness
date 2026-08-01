"""Tests for versioned server inputs and the durable public event journal."""

from __future__ import annotations

import pytest

from chulk.events import AgentEvent, ModelDeltaPayload
from chulk.server import (
    ConversationMessageRequest,
    PermissionDecisionRequest,
    PublicEventCursorExpiredError,
    PublicEventJournal,
)


def _event(conversation_id: str, text: str, *, profile_id: str = "work") -> AgentEvent:
    return AgentEvent(
        name="model.delta",
        profile_id=profile_id,
        conversation_id=conversation_id,
        turn_id="turn-1",
        payload=ModelDeltaPayload(text),
    )


def test_public_event_journal_assigns_sequence_and_resumes_by_event_id(tmp_path) -> None:
    journal = PublicEventJournal(tmp_path / "store.sqlite", profile_id="work")

    first = journal.append(_event("conversation-1", "one"))
    second = journal.append(_event("conversation-1", "two"))
    other = journal.append(_event("conversation-2", "other"))

    assert (first.sequence, second.sequence, other.sequence) == (1, 2, 1)
    resumed = journal.list("conversation-1", after_id=first.event_id)
    assert [record.event.to_dict()["payload"]["text"] for record in resumed] == ["two"]
    assert resumed[0].event.profile_id == "work"


def test_public_event_journal_enforces_profile_ownership_and_retention(tmp_path) -> None:
    journal = PublicEventJournal(
        tmp_path / "store.sqlite",
        profile_id="work",
        retention=2,
    )
    first = journal.append(_event("conversation-1", "one"))
    journal.append(_event("conversation-1", "two"))
    journal.append(_event("conversation-1", "three"))

    assert [record.sequence for record in journal.list("conversation-1")] == [2, 3]
    with pytest.raises(PublicEventCursorExpiredError):
        journal.list("conversation-1", after_id=first.event_id)
    with pytest.raises(ValueError, match="ownership"):
        journal.append(_event("conversation-1", "wrong", profile_id="other"))


def test_public_event_journal_rejects_oversized_events(tmp_path) -> None:
    journal = PublicEventJournal(
        tmp_path / "store.sqlite",
        profile_id="work",
        max_event_bytes=250,
    )

    with pytest.raises(ValueError, match="payload limit"):
        journal.append(_event("conversation-1", "x" * 1_000))


def test_server_request_models_reject_invalid_or_unbounded_inputs() -> None:
    request = ConversationMessageRequest.from_dict(
        {"message": " hello ", "mode": "plan", "idempotency_key": "key-1"}
    )
    decision = PermissionDecisionRequest.from_dict(
        {"decision": "allow", "idempotency_key": "decision-1"}
    )

    assert request.message == "hello"
    assert request.mode == "plan"
    assert decision.decision == "allow"
    with pytest.raises(ValueError, match="mode"):
        ConversationMessageRequest.from_dict({"message": "hello", "mode": "raw"})
    with pytest.raises(ValueError, match="idempotency_key"):
        PermissionDecisionRequest.from_dict({"decision": "deny"})
