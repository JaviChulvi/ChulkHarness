"""Stable, versioned public event contract for SDK embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, TypeAlias
from uuid import uuid4

from chulk.hosting.scope import ExecutionScope
from chulk.redaction import redact_data
from chulk.results import Cost, Plan, RunResult, Usage, freeze_mapping, plain_data


EVENT_SCHEMA_VERSION = 3
SUPPORTED_EVENT_SCHEMA_VERSIONS = (1, 2, 3)


class EventName(str, Enum):
    """Compatibility-stable event names exposed by the SDK."""

    RUN_QUEUED = "run.queued"
    RUN_STARTED = "run.started"
    RUN_PAUSED = "run.paused"
    RUN_RESUMED = "run.resumed"
    RUN_RETRY_SCHEDULED = "run.retry_scheduled"
    RUN_REQUEUED = "run.requeued"
    RUN_STEERED = "run.steered"
    RUN_CANCELLATION_REQUESTED = "run.cancellation_requested"
    RUN_CANCELLED = "run.cancelled"
    RUN_UNKNOWN = "run.unknown"
    RUN_DEAD_LETTERED = "run.dead_lettered"
    STEP_STARTED = "step.started"
    STEP_CHECKPOINTED = "step.checkpointed"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    EFFECT_INTENDED = "effect.intended"
    EFFECT_STARTED = "effect.started"
    EFFECT_COMPLETED = "effect.completed"
    EFFECT_FAILED = "effect.failed"
    EFFECT_UNKNOWN = "effect.unknown"
    EFFECT_RECONCILED = "effect.reconciled"
    EFFECT_RETRIED = "effect.retried"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"
    APPROVAL_CONSUMED = "approval.consumed"
    APPROVAL_INVALIDATED = "approval.invalidated"
    APPROVAL_EXPIRED = "approval.expired"
    APPROVAL_CANCELLED = "approval.cancelled"
    DELIVERY_STARTED = "delivery.started"
    DELIVERY_COMPLETED = "delivery.completed"
    DELIVERY_FAILED = "delivery.failed"
    DELIVERY_UNKNOWN = "delivery.unknown"
    DELIVERY_RECONCILED = "delivery.reconciled"
    DELIVERY_DEAD_LETTERED = "delivery.dead_lettered"
    MODEL_REQUEST_STARTED = "model.request.started"
    MODEL_DELTA = "model.delta"
    MODEL_RESPONSE_COMPLETED = "model.response.completed"
    BUDGET_RESERVED = "budget.reserved"
    BUDGET_COMMITTED = "budget.committed"
    BUDGET_RELEASED = "budget.released"
    BUDGET_EXHAUSTED = "budget.exhausted"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_RESOLVED = "permission.resolved"
    MEMORY_LOADED = "memory.loaded"
    SKILL_LOADED = "skill.loaded"
    LEARNING_PROPOSAL_CHANGED = "learning.proposal.changed"
    PLAN_CREATED = "plan.created"
    PLAN_APPROVED = "plan.approved"
    GOAL_CREATED = "goal.created"
    GOAL_STATE_CHANGED = "goal.state.changed"
    GOAL_STEERED = "goal.steered"
    GOAL_EVIDENCE_RECORDED = "goal.evidence.recorded"
    GOAL_CANCELLATION_REQUESTED = "goal.cancellation.requested"
    CHILD_TASK_CREATED = "child_task.created"
    CHILD_TASK_STATE_CHANGED = "child_task.state.changed"
    CHILD_TASK_DELIVERY_CHANGED = "child_task.delivery.changed"
    AUTOMATION_JOB_CREATED = "automation.job.created"
    AUTOMATION_JOB_STATE_CHANGED = "automation.job.state.changed"
    AUTOMATION_RUN_STATE_CHANGED = "automation.run.state.changed"
    AUTOMATION_TRIGGER_RECEIVED = "automation.trigger.received"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


@dataclass(frozen=True, kw_only=True)
class ExtensiblePayload:
    """Base for typed payloads with read-only forward-compatible fields."""

    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extensions", freeze_mapping(redact_data(dict(self.extensions))))


@dataclass(frozen=True)
class RunStartedPayload(ExtensiblePayload):
    message: str


@dataclass(frozen=True)
class ModelRequestPayload(ExtensiblePayload):
    request_index: int | None = None
    purpose: str | None = None


@dataclass(frozen=True)
class ModelDeltaPayload(ExtensiblePayload):
    text: str


@dataclass(frozen=True)
class ModelResponsePayload(ExtensiblePayload):
    request_index: int | None = None
    content: str | None = None
    usage: Usage | None = None
    cost: Cost | None = None


@dataclass(frozen=True)
class BudgetPayload(ExtensiblePayload):
    resource_kind: str
    scope: str | None = None
    reservation_id: str | None = None
    dimension: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class ToolCallPayload(ExtensiblePayload):
    tool_name: str
    success: bool | None = None
    failure_kind: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class PermissionPayload(ExtensiblePayload):
    tool_name: str
    decision: str | None = None
    reason: str | None = None
    policy_name: str | None = None


@dataclass(frozen=True)
class ResourcesLoadedPayload(ExtensiblePayload):
    items: tuple[str, ...]


@dataclass(frozen=True)
class LearningProposalChangedPayload(ExtensiblePayload):
    proposal_id: str
    kind: str
    status: str
    action: str
    target_name: str | None = None


@dataclass(frozen=True)
class PlanPayload(ExtensiblePayload):
    plan: Plan


@dataclass(frozen=True)
class GoalChangedPayload(ExtensiblePayload):
    goal_id: str
    status: str
    revision: int
    action: str
    step_id: str | None = None


@dataclass(frozen=True)
class ChildTaskChangedPayload(ExtensiblePayload):
    task_id: str
    status: str
    revision: int
    action: str
    parent_task_id: str | None = None
    attempt_id: str | None = None
    delivery_id: str | None = None


@dataclass(frozen=True)
class AutomationChangedPayload(ExtensiblePayload):
    job_id: str
    status: str
    revision: int
    action: str
    run_id: str | None = None
    trigger_id: str | None = None


@dataclass(frozen=True)
class RunLifecyclePayload(ExtensiblePayload):
    """Typed durable-run transition metadata."""

    status: str
    action: str
    revision: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class StepLifecyclePayload(ExtensiblePayload):
    """Typed durable-step transition metadata."""

    step_id: str
    status: str
    action: str
    attempt_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class EffectLifecyclePayload(ExtensiblePayload):
    """Typed external-effect transition metadata."""

    effect_id: str
    step_id: str
    status: str
    action: str
    tool_name: str | None = None
    arguments_digest: str | None = None


@dataclass(frozen=True)
class ApprovalLifecyclePayload(ExtensiblePayload):
    """Typed durable-approval transition metadata."""

    approval_id: str
    step_id: str
    status: str
    action: str
    decision: str | None = None
    arguments_digest: str | None = None


@dataclass(frozen=True)
class ReconciliationPayload(ExtensiblePayload):
    """Typed operator reconciliation metadata for uncertain work."""

    target_kind: str
    target_id: str
    decision: str
    reason: str


@dataclass(frozen=True)
class DeliveryPayload(ExtensiblePayload):
    """Typed host delivery lifecycle metadata."""

    delivery_id: str
    status: str
    action: str
    target: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RunCompletedPayload(ExtensiblePayload):
    result: RunResult


@dataclass(frozen=True)
class RunFailedPayload(ExtensiblePayload):
    error: Mapping[str, Any]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "error", freeze_mapping(redact_data(dict(self.error))))


@dataclass(frozen=True)
class SerializedEventPayload(ExtensiblePayload):
    """Typed wrapper used when reading a serialized event envelope."""

    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "data", freeze_mapping(redact_data(dict(self.data))))


EventPayload: TypeAlias = (
    RunStartedPayload
    | ModelRequestPayload
    | ModelDeltaPayload
    | ModelResponsePayload
    | BudgetPayload
    | ToolCallPayload
    | PermissionPayload
    | ResourcesLoadedPayload
    | LearningProposalChangedPayload
    | PlanPayload
    | GoalChangedPayload
    | ChildTaskChangedPayload
    | AutomationChangedPayload
    | RunLifecyclePayload
    | StepLifecyclePayload
    | EffectLifecyclePayload
    | ApprovalLifecyclePayload
    | ReconciliationPayload
    | DeliveryPayload
    | RunCompletedPayload
    | RunFailedPayload
    | SerializedEventPayload
)


@dataclass(frozen=True)
class AgentEvent:
    """One public event with stable identity and a typed payload."""

    name: str
    conversation_id: str
    payload: EventPayload
    turn_id: str | None = None
    profile_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = EVENT_SCHEMA_VERSION
    event_id: str = field(default_factory=lambda: uuid4().hex)
    execution_scope: ExecutionScope | None = None
    run_id: str | None = None
    step_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    source_event_id: str | None = None
    idempotency_key: str | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_EVENT_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported event schema_version: {self.schema_version}")
        object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))
        for name in (
            "run_id",
            "step_id",
            "correlation_id",
            "causation_id",
            "source_event_id",
            "idempotency_key",
        ):
            object.__setattr__(self, name, _optional(getattr(self, name)))
        if self.execution_scope is not None:
            if self.run_id is None:
                object.__setattr__(
                    self,
                    "run_id",
                    self.execution_scope.run_id,
                )
            elif self.run_id != self.execution_scope.run_id:
                raise ValueError("event run_id must match execution_scope")
        object.__setattr__(self, "extensions", freeze_mapping(redact_data(dict(self.extensions))))

    @property
    def type(self) -> str:
        """Compatibility alias for callers that previously read ``event.type``."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        payload = (
            plain_data(self.payload.data)
            if isinstance(self.payload, SerializedEventPayload)
            else redact_data(plain_data(self.payload))
        )
        return {
            "event_id": self.event_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "profile_id": self.profile_id,
            "execution_scope": (
                self.execution_scope.to_dict()
                if self.execution_scope is not None
                else None
            ),
            "run_id": self.run_id,
            "step_id": self.step_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "source_event_id": self.source_event_id,
            "idempotency_key": self.idempotency_key,
            "payload": payload,
            "extensions": plain_data(self.extensions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentEvent:
        """Read schema-v1 through schema-v3 envelopes without runtime code."""
        schema_version = value.get("schema_version", 1)
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError("event schema_version must be an integer")
        if schema_version not in SUPPORTED_EVENT_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported event schema_version: {schema_version}")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("event payload must be an object")
        extensions = value.get("extensions", {})
        if not isinstance(extensions, Mapping):
            raise ValueError("event extensions must be an object")
        raw_profile_id = value.get("profile_id")
        profile_id = raw_profile_id if isinstance(raw_profile_id, str) and raw_profile_id else "default"
        scope_value = value.get("execution_scope")
        if scope_value is None:
            execution_scope = None
        elif isinstance(scope_value, Mapping):
            execution_scope = ExecutionScope.from_dict(dict(scope_value))
        else:
            raise ValueError("event execution_scope must be an object")
        return cls(
            event_id=(
                value["event_id"]
                if isinstance(value.get("event_id"), str)
                else _legacy_event_id(value)
            ),
            name=str(value.get("name") or value.get("type") or ""),
            conversation_id=str(value.get("conversation_id") or ""),
            turn_id=value.get("turn_id") if isinstance(value.get("turn_id"), str) else None,
            profile_id=profile_id,
            timestamp=str(value.get("timestamp") or value.get("created_at") or ""),
            schema_version=schema_version,
            execution_scope=execution_scope,
            run_id=_string(value.get("run_id")),
            step_id=_string(value.get("step_id")),
            correlation_id=_string(value.get("correlation_id")),
            causation_id=_string(value.get("causation_id")),
            source_event_id=_string(value.get("source_event_id")),
            idempotency_key=_string(value.get("idempotency_key")),
            payload=SerializedEventPayload(data=payload),
            extensions=extensions,
        )


def _legacy_event_id(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_data(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"legacy_{hashlib.sha256(encoded).hexdigest()}"


def _required(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"event {name} cannot be empty")
    return value.strip()


def _optional(value: object) -> str | None:
    if value is None:
        return None
    return _required(value, "field")


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "AutomationChangedPayload",
    "EVENT_SCHEMA_VERSION",
    "SUPPORTED_EVENT_SCHEMA_VERSIONS",
    "AgentEvent",
    "ApprovalLifecyclePayload",
    "BudgetPayload",
    "ChildTaskChangedPayload",
    "DeliveryPayload",
    "EffectLifecyclePayload",
    "EventName",
    "EventPayload",
    "GoalChangedPayload",
    "LearningProposalChangedPayload",
    "ModelDeltaPayload",
    "ModelRequestPayload",
    "ModelResponsePayload",
    "PermissionPayload",
    "PlanPayload",
    "ResourcesLoadedPayload",
    "ReconciliationPayload",
    "RunCompletedPayload",
    "RunFailedPayload",
    "RunLifecyclePayload",
    "RunStartedPayload",
    "SerializedEventPayload",
    "StepLifecyclePayload",
    "ToolCallPayload",
]
