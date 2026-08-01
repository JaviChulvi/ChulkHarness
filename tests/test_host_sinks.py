"""Host sink safety and application-stream adapter contracts."""

from __future__ import annotations

import pytest

from chulk import (
    AgentEvent,
    CallbackEventSink,
    ExecutionScope,
    InMemoryAuditSink,
    RunLifecyclePayload,
    SinkDeliveryError,
)


def _scope() -> ExecutionScope:
    return ExecutionScope(
        tenant_id="tenant",
        workspace_id="workspace",
        actor_id="operator",
        agent_id="agent",
        agent_version="1.0.0",
        run_id="run",
    )


def _event() -> AgentEvent:
    return AgentEvent(
        name="run.queued",
        conversation_id="conversation",
        execution_scope=_scope(),
        payload=RunLifecyclePayload(
            status="queued",
            action="queued",
        ),
    )


def test_callback_event_sink_is_fail_closed_by_default() -> None:
    def reject(event: AgentEvent) -> None:
        raise RuntimeError("application stream unavailable")

    sink = CallbackEventSink(reject)

    with pytest.raises(SinkDeliveryError, match="rejected"):
        sink.emit(_event())
    assert sink.failures == ["RuntimeError"]


def test_callback_event_sink_can_explicitly_drop_delivery() -> None:
    sink = CallbackEventSink(
        lambda event: (_ for _ in ()).throw(RuntimeError("offline")),
        fail_closed=False,
    )

    sink.emit(_event())

    assert sink.failures == ["RuntimeError"]


def test_audit_sink_rejects_raw_secrets_and_accepts_digests() -> None:
    scope = _scope()
    sink = InMemoryAuditSink(scope)
    sink.record(
        "effect.intended",
        {
            "effect_id": "effect-1",
            "arguments_digest": "sha256:arguments",
        },
        scope=scope,
    )

    assert sink.events[0]["payload"]["arguments_digest"] == "sha256:arguments"
    with pytest.raises(ValueError, match="cannot contain"):
        sink.record(
            "tool.called",
            {"raw_arguments": {"token": "must-not-leak"}},
            scope=scope,
        )
