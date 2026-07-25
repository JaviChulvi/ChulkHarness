"""Tests for durable permission rendezvous and decision races."""

from __future__ import annotations

import threading
import time

import pytest

from chulk.server import (
    PermissionBroker,
    PermissionDecisionConflictError,
    PublicEventJournal,
)
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    ToolPermissionLevel,
)


def _request(arguments=None) -> PermissionRequest:
    return PermissionRequest(
        tool_name="shell",
        permission_level=ToolPermissionLevel.SHELL,
        arguments=arguments or {"command": "printf hello", "api_key": "secret"},
        requires_confirmation=True,
        policy_name="workspace-write",
        reason="shell tool requires approval",
    )


def _record() -> PermissionDecisionRecord:
    return PermissionDecisionRecord(
        tool_name="shell",
        permission_level=ToolPermissionLevel.SHELL,
        decision=PermissionDecision.ASK,
        reason="tool call requires user approval",
        policy_name="workspace-write",
        requires_confirmation=True,
    )


def _broker(tmp_path, **kwargs) -> PermissionBroker:
    path = tmp_path / "store.sqlite"
    journal = PublicEventJournal(path, profile_id="work")
    return PermissionBroker(
        path,
        profile_id="work",
        conversation_id="conversation-1",
        turn_id=lambda: "turn-1",
        journal=journal,
        **kwargs,
    )


def test_permission_broker_waits_for_one_idempotent_api_decision(tmp_path) -> None:
    broker = _broker(tmp_path)
    answer: list[PermissionDecision] = []

    worker = threading.Thread(
        target=lambda: answer.append(broker.callback(_request(), _record())),
    )
    worker.start()
    for _ in range(100):
        pending = broker.list(status="pending")
        if pending:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("permission request was not persisted")

    resolved = broker.decide(
        pending[0].id,
        "allow",
        idempotency_key="decision-1",
        reason="owner approved",
    )
    duplicate = broker.decide(
        pending[0].id,
        "allow",
        idempotency_key="decision-1",
        reason="ignored duplicate",
    )
    worker.join(timeout=2)

    assert answer == [PermissionDecision.ALLOW]
    assert resolved.status == duplicate.status == "allowed"
    assert resolved.argument_preview["api_key"] == "[redacted]"
    assert len(resolved.argument_sha256) == 64
    events = broker.journal.list("conversation-1") if broker.journal else ()
    assert [item.event.name for item in events] == [
        "permission.requested",
        "permission.resolved",
    ]


def test_permission_broker_rejects_conflicting_and_late_decisions(tmp_path) -> None:
    broker = _broker(tmp_path)
    pending = broker.create(_request())
    broker.decide(pending.id, "deny", idempotency_key="decision-1")

    with pytest.raises(PermissionDecisionConflictError):
        broker.decide(pending.id, "allow", idempotency_key="decision-2")

    expiring = _broker(tmp_path, ttl_seconds=1)
    late = expiring.create(_request({"command": "slow"}))
    time.sleep(1.05)
    with pytest.raises(PermissionDecisionConflictError, match="expired"):
        expiring.decide(late.id, "allow", idempotency_key="decision-late")


def test_permission_broker_marks_restart_requests_uncertain_without_replay(tmp_path) -> None:
    first = _broker(tmp_path)
    pending = first.create(_request())

    restarted = _broker(tmp_path)

    assert restarted.get(pending.id).status == "uncertain"
    with pytest.raises(PermissionDecisionConflictError, match="uncertain"):
        restarted.decide(pending.id, "allow", idempotency_key="decision-1")


def test_permission_broker_preserves_immediate_callback_compatibility(tmp_path) -> None:
    broker = _broker(tmp_path, immediate_callback=lambda _request, _record: True)

    assert broker.callback(_request(), _record()) is PermissionDecision.ALLOW
    assert broker.list()[0].status == "allowed"
