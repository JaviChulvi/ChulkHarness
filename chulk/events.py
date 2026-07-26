"""Stable, versioned public event contract for SDK embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeAlias

from chulk.redaction import redact_data
from chulk.results import Cost, Plan, RunResult, Usage, freeze_mapping, plain_data


EVENT_SCHEMA_VERSION = 2
SUPPORTED_EVENT_SCHEMA_VERSIONS = (1, 2)


class EventName(str, Enum):
    """Compatibility-stable event names exposed by the SDK."""

    RUN_STARTED = "run.started"
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
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            "name": self.name,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "profile_id": self.profile_id,
            "payload": payload,
            "extensions": plain_data(self.extensions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentEvent:
        """Read schema-v1 or schema-v2 envelopes without executing runtime code."""
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
        return cls(
            name=str(value.get("name") or value.get("type") or ""),
            conversation_id=str(value.get("conversation_id") or ""),
            turn_id=value.get("turn_id") if isinstance(value.get("turn_id"), str) else None,
            profile_id=profile_id,
            timestamp=str(value.get("timestamp") or value.get("created_at") or ""),
            schema_version=schema_version,
            payload=SerializedEventPayload(data=payload),
            extensions=extensions,
        )


__all__ = [
    "AutomationChangedPayload",
    "EVENT_SCHEMA_VERSION",
    "SUPPORTED_EVENT_SCHEMA_VERSIONS",
    "AgentEvent",
    "BudgetPayload",
    "ChildTaskChangedPayload",
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
    "RunCompletedPayload",
    "RunFailedPayload",
    "RunStartedPayload",
    "SerializedEventPayload",
    "ToolCallPayload",
]
