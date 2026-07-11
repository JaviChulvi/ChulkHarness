"""Stable, versioned public event contract for SDK embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

from chulk.redaction import redact_data


EVENT_SCHEMA_VERSION = 1


class EventName(str, Enum):
    """Compatibility-stable event names exposed by the SDK."""

    RUN_STARTED = "run.started"
    MODEL_REQUEST_STARTED = "model.request.started"
    MODEL_DELTA = "model.delta"
    MODEL_RESPONSE_COMPLETED = "model.response.completed"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_RESOLVED = "permission.resolved"
    MEMORY_LOADED = "memory.loaded"
    SKILL_LOADED = "skill.loaded"
    PLAN_CREATED = "plan.created"
    PLAN_APPROVED = "plan.approved"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


@dataclass(frozen=True, kw_only=True)
class ExtensiblePayload:
    """Base for typed payloads with read-only forward-compatible fields."""

    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extensions", MappingProxyType(redact_data(dict(self.extensions))))


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
    usage: Mapping[str, Any] | None = None
    cost: Mapping[str, Any] | None = None


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
class PlanPayload(ExtensiblePayload):
    plan: Mapping[str, Any]


@dataclass(frozen=True)
class RunCompletedPayload(ExtensiblePayload):
    result: Any


@dataclass(frozen=True)
class RunFailedPayload(ExtensiblePayload):
    error: Mapping[str, Any]


EventPayload: TypeAlias = (
    RunStartedPayload
    | ModelRequestPayload
    | ModelDeltaPayload
    | ModelResponsePayload
    | ToolCallPayload
    | PermissionPayload
    | ResourcesLoadedPayload
    | PlanPayload
    | RunCompletedPayload
    | RunFailedPayload
)


@dataclass(frozen=True)
class AgentEvent:
    """One public event with stable identity and a typed payload."""

    name: str
    conversation_id: str
    payload: EventPayload
    turn_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = EVENT_SCHEMA_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extensions", MappingProxyType(redact_data(dict(self.extensions))))

    @property
    def type(self) -> str:
        """Compatibility alias for callers that previously read ``event.type``."""
        return self.name

    def to_dict(self) -> dict[str, Any]:
        payload = (
            {item.name: getattr(self.payload, item.name) for item in fields(self.payload)}
            if is_dataclass(self.payload)
            else self.payload
        )
        if isinstance(self.payload, RunCompletedPayload):
            result = self.payload.result
            payload = {"result": result.to_dict() if hasattr(result, "to_dict") else redact_data(result)}
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "payload": redact_data(payload),
            "extensions": dict(self.extensions),
        }


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "AgentEvent",
    "EventName",
    "EventPayload",
    "ModelDeltaPayload",
    "ModelRequestPayload",
    "ModelResponsePayload",
    "PermissionPayload",
    "PlanPayload",
    "ResourcesLoadedPayload",
    "RunCompletedPayload",
    "RunFailedPayload",
    "RunStartedPayload",
    "ToolCallPayload",
]
