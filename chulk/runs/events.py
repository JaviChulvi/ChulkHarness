"""Projection and publication of durable transitions as schema-v3 events."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from chulk.events import (
    AgentEvent,
    ApprovalLifecyclePayload,
    EffectLifecyclePayload,
    ReconciliationPayload,
    RunLifecyclePayload,
    SerializedEventPayload,
    StepLifecyclePayload,
)
from chulk.hosting.scope import ExecutionScope
from chulk.hosting.services import EventSink
from chulk.runs.models import RunEvent
from chulk.runs.protocols import AsyncRunStore, RunStore


_RUN_STATUS = {
    "run.queued": "queued",
    "run.started": "running",
    "run.paused": "waiting_for_approval",
    "run.resumed": "queued",
    "run.retry_scheduled": "waiting_for_retry",
    "run.requeued": "queued",
    "run.steered": "unchanged",
    "run.completed": "completed",
    "run.failed": "failed",
    "run.cancellation_requested": "running",
    "run.cancelled": "cancelled",
    "run.unknown": "unknown",
    "run.dead_lettered": "dead_letter",
}
_STEP_STATUS = {
    "step.started": "running",
    "step.checkpointed": "running",
    "step.completed": "completed",
    "step.failed": "failed",
}
_EFFECT_STATUS = {
    "effect.intended": "intended",
    "effect.started": "executing",
    "effect.completed": "completed",
    "effect.failed": "failed",
    "effect.unknown": "unknown",
    "effect.retried": "intended",
}
_APPROVAL_STATUS = {
    "approval.requested": "pending",
    "approval.decided": "decided",
    "approval.consumed": "consumed",
    "approval.invalidated": "invalidated",
    "approval.expired": "expired",
    "approval.cancelled": "cancelled",
}


def project_run_event(
    event: RunEvent,
    scope: ExecutionScope,
    *,
    previous_event_id: str | None = None,
) -> AgentEvent:
    """Create one public envelope without exposing raw tool arguments."""
    payload = _payload(event)
    return AgentEvent(
        event_id=event.id,
        name=event.name,
        conversation_id=scope.conversation_id or scope.run_id,
        execution_scope=scope,
        run_id=event.run_id,
        step_id=event.step_id,
        correlation_id=event.correlation_id or event.run_id,
        causation_id=event.causation_id or previous_event_id,
        source_event_id=_text(event.payload.get("source_event_id")),
        idempotency_key=event.idempotency_key or event.id,
        timestamp=event.created_at.isoformat(),
        payload=payload,
        extensions={
            "durable_sequence": event.sequence,
            "actor": event.actor,
        },
    )


def project_run_events(
    events: Iterable[RunEvent],
    scope: ExecutionScope,
    *,
    previous_event_id: str | None = None,
) -> tuple[AgentEvent, ...]:
    projected: list[AgentEvent] = []
    causation = previous_event_id
    for event in events:
        public = project_run_event(
            event,
            scope,
            previous_event_id=causation,
        )
        projected.append(public)
        causation = public.event_id
    return tuple(projected)


class RunEventPublisher:
    """Publish new durable transitions from their append-only source."""

    def __init__(
        self,
        runs: RunStore,
        sink: EventSink,
        *,
        scope: ExecutionScope,
    ) -> None:
        self.runs = runs
        self.sink = sink
        self.scope = scope
        self.sequence = 0
        self.last_event_id: str | None = None

    def publish(self) -> tuple[AgentEvent, ...]:
        durable = self.runs.events(
            self.scope,
            self.scope.run_id,
            after_sequence=self.sequence,
        )
        public = project_run_events(
            durable,
            self.scope,
            previous_event_id=self.last_event_id,
        )
        for event in public:
            self.sink.emit(event)
            self.sequence = int(event.extensions["durable_sequence"])
            self.last_event_id = event.event_id
        return public


class AsyncRunEventPublisher:
    """Async equivalent that keeps store I/O off the event loop."""

    def __init__(
        self,
        runs: AsyncRunStore,
        sink: EventSink,
        *,
        scope: ExecutionScope,
    ) -> None:
        self.runs = runs
        self.sink = sink
        self.scope = scope
        self.sequence = 0
        self.last_event_id: str | None = None

    async def publish(self) -> tuple[AgentEvent, ...]:
        durable = await self.runs.events(
            self.scope,
            self.scope.run_id,
            after_sequence=self.sequence,
        )
        public = project_run_events(
            durable,
            self.scope,
            previous_event_id=self.last_event_id,
        )
        for event in public:
            self.sink.emit(event)
            self.sequence = int(event.extensions["durable_sequence"])
            self.last_event_id = event.event_id
        return public


def _payload(event: RunEvent):
    data = dict(event.payload)
    action = event.name.rsplit(".", 1)[-1]
    reason = _text(data.get("reason"))
    if event.name in _RUN_STATUS:
        return RunLifecyclePayload(
            status=_text(data.get("status")) or _RUN_STATUS[event.name],
            action=action,
            reason=reason,
            extensions=data,
        )
    if event.name in _STEP_STATUS:
        return StepLifecyclePayload(
            step_id=event.step_id or _text(data.get("step_id")) or "unknown",
            status=_STEP_STATUS[event.name],
            action=action,
            attempt_id=_text(data.get("attempt_id")),
            reason=reason,
            extensions=data,
        )
    if event.name == "effect.reconciled":
        return ReconciliationPayload(
            target_kind="effect",
            target_id=_text(data.get("effect_id")) or "unknown",
            decision=_text(data.get("decision")) or "unknown",
            reason=reason or "effect reconciled",
            extensions=data,
        )
    if event.name in _EFFECT_STATUS:
        return EffectLifecyclePayload(
            effect_id=_text(data.get("effect_id")) or "unknown",
            step_id=event.step_id or "unknown",
            status=_EFFECT_STATUS[event.name],
            action=action,
            tool_name=_text(data.get("tool_name")),
            arguments_digest=_text(data.get("arguments_digest")),
            extensions=data,
        )
    if event.name in _APPROVAL_STATUS:
        return ApprovalLifecyclePayload(
            approval_id=_text(data.get("approval_id")) or "unknown",
            step_id=event.step_id or "unknown",
            status=_APPROVAL_STATUS[event.name],
            action=action,
            decision=_text(data.get("decision")),
            arguments_digest=_text(data.get("arguments_digest")),
            extensions=data,
        )
    return SerializedEventPayload(data=data)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "AsyncRunEventPublisher",
    "RunEventPublisher",
    "project_run_event",
    "project_run_events",
]
