"""Immutable contracts for governed child-task orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Mapping

from chulk._serialization import _freeze_mapping, _plain
from chulk.capabilities import Capabilities, FileAccess, MemoryMode
from chulk.execution import WorkspaceMode
from chulk.usage import (
    BudgetScope,
    ExactCost,
    RunBudget,
    UnknownCostPolicy,
)


class ChildTaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNKNOWN = "unknown"


class ChildTaskRole(StrEnum):
    LEAF = "leaf"
    ORCHESTRATOR = "orchestrator"


class ChildDeliveryStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DELIVERED = "delivered"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ChildEvidenceRef:
    """Evidence returned by a child without granting completion authority."""

    id: str
    kind: str
    summary: str
    reference: str | None = None
    sha256: str | None = None
    criterion_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "evidence id"))
        object.__setattr__(self, "kind", _required(self.kind, "evidence kind"))
        object.__setattr__(self, "summary", _required(self.summary, "evidence summary"))
        object.__setattr__(self, "reference", _optional(self.reference))
        object.__setattr__(self, "sha256", _optional(self.sha256))
        object.__setattr__(
            self,
            "criterion_ids",
            _unique_strings(self.criterion_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "reference": self.reference,
            "sha256": self.sha256,
            "criterion_ids": list(self.criterion_ids),
        }


@dataclass(frozen=True, slots=True)
class ChildTaskLineage:
    """Parentage and bounded delegation depth for one child."""

    parent_task_id: str | None = None
    root_task_id: str | None = None
    depth: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_task_id", _optional(self.parent_task_id))
        object.__setattr__(self, "root_task_id", _optional(self.root_task_id))
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 1:
            raise ValueError("child task depth must be a positive integer")
        if self.parent_task_id is None and self.depth != 1:
            raise ValueError("root child tasks must have depth one")
        if self.parent_task_id is not None and self.root_task_id is None:
            raise ValueError("nested child tasks require root_task_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_task_id": self.parent_task_id,
            "root_task_id": self.root_task_id,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class ChildTaskSpec:
    """Host-validated package provided to a fresh child context."""

    instruction: str
    context: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Capabilities = field(default_factory=Capabilities.read_only)
    tool_names: tuple[str, ...] = ()
    skill_names: tuple[str, ...] = ()
    mcp_server_labels: tuple[str, ...] = ()
    model_profile_id: str | None = None
    backend_name: str = "host"
    workspace_mode: WorkspaceMode = WorkspaceMode.HOST
    role: ChildTaskRole = ChildTaskRole.LEAF
    mutable: bool = False
    max_depth: int = 1
    max_parallelism: int = 1
    budget: RunBudget = field(
        default_factory=lambda: RunBudget(scope=BudgetScope.CHILD_TASK)
    )
    result_schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instruction",
            _required(self.instruction, "child instruction"),
        )
        object.__setattr__(self, "context", _freeze_mapping(self.context))
        object.__setattr__(self, "tool_names", _unique_strings(self.tool_names))
        object.__setattr__(self, "skill_names", _unique_strings(self.skill_names))
        object.__setattr__(
            self,
            "mcp_server_labels",
            _unique_strings(self.mcp_server_labels),
        )
        object.__setattr__(
            self,
            "model_profile_id",
            _optional(self.model_profile_id),
        )
        object.__setattr__(
            self,
            "backend_name",
            _required(self.backend_name, "child backend name"),
        )
        object.__setattr__(self, "workspace_mode", WorkspaceMode(self.workspace_mode))
        object.__setattr__(self, "role", ChildTaskRole(self.role))
        for name in ("max_depth", "max_parallelism"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.budget.scope is not BudgetScope.CHILD_TASK:
            raise ValueError("child task budget scope must be child_task")
        if self.capabilities.memory not in {
            MemoryMode.OFF,
            MemoryMode.READ_ONLY,
        }:
            raise ValueError("child tasks cannot receive memory mutation authority")
        if self.mutable and self.capabilities.files is not FileAccess.WRITE:
            raise ValueError("mutable child tasks require file write capability")
        if self.mutable and self.workspace_mode not in {
            WorkspaceMode.TEMPORARY,
            WorkspaceMode.GIT_WORKTREE,
            WorkspaceMode.CONTAINER,
        }:
            raise ValueError("mutable child tasks require an isolated workspace")
        object.__setattr__(
            self,
            "result_schema",
            _freeze_mapping(self.result_schema),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "context": _plain(self.context),
            "capabilities": self.capabilities.to_dict(),
            "tool_names": list(self.tool_names),
            "skill_names": list(self.skill_names),
            "mcp_server_labels": list(self.mcp_server_labels),
            "model_profile_id": self.model_profile_id,
            "backend_name": self.backend_name,
            "workspace_mode": self.workspace_mode.value,
            "role": self.role.value,
            "mutable": self.mutable,
            "max_depth": self.max_depth,
            "max_parallelism": self.max_parallelism,
            "budget": self.budget.to_dict(),
            "result_schema": _plain(self.result_schema),
        }


@dataclass(frozen=True, slots=True)
class ChildTaskResult:
    """Non-authoritative child output awaiting parent-side validation."""

    summary: str
    structured_output: Mapping[str, Any]
    evidence: tuple[ChildEvidenceRef, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    completion_claims: tuple[str, ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    change_set_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _required(self.summary, "child summary"))
        object.__setattr__(
            self,
            "structured_output",
            _freeze_mapping(self.structured_output),
        )
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(
            self,
            "artifact_refs",
            _unique_strings(self.artifact_refs),
        )
        object.__setattr__(
            self,
            "changed_files",
            _unique_strings(self.changed_files),
        )
        object.__setattr__(
            self,
            "completion_claims",
            _unique_strings(self.completion_claims),
        )
        object.__setattr__(self, "usage", _freeze_mapping(self.usage))
        object.__setattr__(self, "trace_id", _optional(self.trace_id))
        object.__setattr__(self, "change_set_id", _optional(self.change_set_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "structured_output": _plain(self.structured_output),
            "evidence": [item.to_dict() for item in self.evidence],
            "artifact_refs": list(self.artifact_refs),
            "changed_files": list(self.changed_files),
            "completion_claims": list(self.completion_claims),
            "usage": _plain(self.usage),
            "trace_id": self.trace_id,
            "change_set_id": self.change_set_id,
        }


@dataclass(frozen=True, slots=True)
class ChildTask:
    """Revisioned durable child-task snapshot."""

    id: str
    profile_id: str
    spec: ChildTaskSpec
    lineage: ChildTaskLineage
    dependency_ids: tuple[str, ...] = ()
    goal_id: str | None = None
    goal_step_id: str | None = None
    parent_conversation_id: str | None = None
    parent_turn_id: str | None = None
    parent_trace_id: str | None = None
    status: ChildTaskStatus = ChildTaskStatus.PENDING
    revision: int = 0
    attempt_count: int = 0
    cancellation_requested: bool = False
    result: ChildTaskResult | None = None
    terminal_reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "child task id"))
        object.__setattr__(self, "profile_id", _required(self.profile_id, "profile id"))
        object.__setattr__(
            self,
            "dependency_ids",
            _unique_strings(self.dependency_ids),
        )
        if self.id in self.dependency_ids:
            raise ValueError("child task cannot depend on itself")
        for field_name in (
            "goal_id",
            "goal_step_id",
            "parent_conversation_id",
            "parent_turn_id",
            "parent_trace_id",
            "terminal_reason",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional(getattr(self, field_name)),
            )
        object.__setattr__(self, "status", ChildTaskStatus(self.status))
        for field_name in ("revision", "attempt_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"child task {field_name} must be non-negative")
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        if self.completed_at is not None:
            object.__setattr__(
                self,
                "completed_at",
                _utc(self.completed_at, "completed_at"),
            )
        if self.status is ChildTaskStatus.COMPLETED and self.result is None:
            raise ValueError("completed child task requires a result")

    @property
    def terminal(self) -> bool:
        return self.status in {
            ChildTaskStatus.COMPLETED,
            ChildTaskStatus.FAILED,
            ChildTaskStatus.CANCELLED,
            ChildTaskStatus.BUDGET_EXHAUSTED,
            ChildTaskStatus.UNKNOWN,
        }

    def with_revision(self, revision: int, *, now: datetime) -> ChildTask:
        return replace(self, revision=revision, updated_at=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "spec": self.spec.to_dict(),
            "lineage": self.lineage.to_dict(),
            "dependency_ids": list(self.dependency_ids),
            "goal_id": self.goal_id,
            "goal_step_id": self.goal_step_id,
            "parent_conversation_id": self.parent_conversation_id,
            "parent_turn_id": self.parent_turn_id,
            "parent_trace_id": self.parent_trace_id,
            "status": self.status.value,
            "revision": self.revision,
            "attempt_count": self.attempt_count,
            "cancellation_requested": self.cancellation_requested,
            "result": self.result.to_dict() if self.result is not None else None,
            "terminal_reason": self.terminal_reason,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ChildTaskClaim:
    task_id: str
    profile_id: str
    attempt_id: str
    attempt_number: int
    worker_id: str
    claim_token: str
    lease_until: datetime

    def __post_init__(self) -> None:
        for field_name, label in (
            ("task_id", "child task id"),
            ("profile_id", "profile id"),
            ("attempt_id", "attempt id"),
            ("worker_id", "worker id"),
            ("claim_token", "claim token"),
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), label),
            )
        if (
            isinstance(self.attempt_number, bool)
            or not isinstance(self.attempt_number, int)
            or self.attempt_number < 1
        ):
            raise ValueError("attempt number must be positive")
        object.__setattr__(self, "lease_until", _utc(self.lease_until, "lease_until"))


@dataclass(frozen=True, slots=True)
class ChildTaskEvent:
    id: str
    task_id: str
    profile_id: str
    revision: int
    kind: str
    actor: str
    payload: Mapping[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        for field_name, label in (
            ("id", "child event id"),
            ("task_id", "child task id"),
            ("profile_id", "profile id"),
            ("kind", "child event kind"),
            ("actor", "child event actor"),
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), label),
            )
        if isinstance(self.revision, bool) or self.revision < 0:
            raise ValueError("child event revision must be non-negative")
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))


@dataclass(frozen=True, slots=True)
class ChildCompletionDelivery:
    id: str
    task_id: str
    profile_id: str
    task_revision: int
    status: ChildDeliveryStatus
    idempotency_key: str
    created_at: datetime
    updated_at: datetime
    claim_token: str | None = None
    worker_id: str | None = None
    lease_until: datetime | None = None
    attempts: int = 0
    delivered_at: datetime | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        for field_name, label in (
            ("id", "delivery id"),
            ("task_id", "child task id"),
            ("profile_id", "profile id"),
            ("idempotency_key", "delivery idempotency key"),
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), label),
            )
        for field_name in ("task_revision", "attempts"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"delivery {field_name} must be non-negative")
        object.__setattr__(self, "status", ChildDeliveryStatus(self.status))
        object.__setattr__(self, "claim_token", _optional(self.claim_token))
        object.__setattr__(self, "worker_id", _optional(self.worker_id))
        object.__setattr__(self, "error", _optional(self.error))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        if self.lease_until is not None:
            object.__setattr__(
                self,
                "lease_until",
                _utc(self.lease_until, "lease_until"),
            )
        if self.delivered_at is not None:
            object.__setattr__(
                self,
                "delivered_at",
                _utc(self.delivered_at, "delivered_at"),
            )
        claimed_fields = (
            self.claim_token is not None,
            self.worker_id is not None,
            self.lease_until is not None,
        )
        if any(claimed_fields) and not all(claimed_fields):
            raise ValueError("delivery claim fields must be present together")
        if self.status is ChildDeliveryStatus.CLAIMED and not all(claimed_fields):
            raise ValueError("claimed delivery requires claim ownership")
        if self.status is not ChildDeliveryStatus.CLAIMED and any(claimed_fields):
            raise ValueError("unclaimed delivery cannot retain claim ownership")


def child_task_from_dict(value: Mapping[str, Any]) -> ChildTask:
    raw_result = value.get("result")
    return ChildTask(
        id=str(value["id"]),
        profile_id=str(value["profile_id"]),
        spec=_spec_from_dict(_mapping(value.get("spec"), "spec")),
        lineage=_lineage_from_dict(_mapping(value.get("lineage"), "lineage")),
        dependency_ids=_strings(value.get("dependency_ids", ())),
        goal_id=_optional_value(value.get("goal_id")),
        goal_step_id=_optional_value(value.get("goal_step_id")),
        parent_conversation_id=_optional_value(
            value.get("parent_conversation_id")
        ),
        parent_turn_id=_optional_value(value.get("parent_turn_id")),
        parent_trace_id=_optional_value(value.get("parent_trace_id")),
        status=ChildTaskStatus(str(value.get("status", "pending"))),
        revision=_integer(value.get("revision", 0), "revision"),
        attempt_count=_integer(value.get("attempt_count", 0), "attempt_count"),
        cancellation_requested=bool(value.get("cancellation_requested", False)),
        result=(
            child_result_from_dict(_mapping(raw_result, "result"))
            if raw_result is not None
            else None
        ),
        terminal_reason=_optional_value(value.get("terminal_reason")),
        created_at=_datetime(value["created_at"], "created_at"),
        updated_at=_datetime(value["updated_at"], "updated_at"),
        completed_at=_optional_datetime(value.get("completed_at"), "completed_at"),
    )


def _spec_from_dict(value: Mapping[str, Any]) -> ChildTaskSpec:
    capabilities = _mapping(value.get("capabilities"), "capabilities")
    return ChildTaskSpec(
        instruction=str(value["instruction"]),
        context=_mapping(value.get("context", {}), "context"),
        capabilities=Capabilities(**dict(capabilities)),
        tool_names=_strings(value.get("tool_names", ())),
        skill_names=_strings(value.get("skill_names", ())),
        mcp_server_labels=_strings(value.get("mcp_server_labels", ())),
        model_profile_id=_optional_value(value.get("model_profile_id")),
        backend_name=str(value.get("backend_name", "host")),
        workspace_mode=WorkspaceMode(str(value.get("workspace_mode", "host"))),
        role=ChildTaskRole(str(value.get("role", "leaf"))),
        mutable=bool(value.get("mutable", False)),
        max_depth=_integer(value.get("max_depth", 1), "max_depth"),
        max_parallelism=_integer(
            value.get("max_parallelism", 1),
            "max_parallelism",
        ),
        budget=_budget_from_dict(_mapping(value.get("budget"), "budget")),
        result_schema=_mapping(value.get("result_schema", {}), "result_schema"),
    )


def _lineage_from_dict(value: Mapping[str, Any]) -> ChildTaskLineage:
    return ChildTaskLineage(
        parent_task_id=_optional_value(value.get("parent_task_id")),
        root_task_id=_optional_value(value.get("root_task_id")),
        depth=_integer(value.get("depth", 1), "depth"),
    )


def child_result_from_dict(value: Mapping[str, Any]) -> ChildTaskResult:
    evidence_values = value.get("evidence", ())
    if not isinstance(evidence_values, (list, tuple)):
        raise ValueError("evidence must be an array")
    return ChildTaskResult(
        summary=str(value["summary"]),
        structured_output=_mapping(
            value.get("structured_output", {}),
            "structured_output",
        ),
        evidence=tuple(
            ChildEvidenceRef(
                id=str(item["id"]),
                kind=str(item["kind"]),
                summary=str(item["summary"]),
                reference=_optional_value(item.get("reference")),
                sha256=_optional_value(item.get("sha256")),
                criterion_ids=_strings(item.get("criterion_ids", ())),
            )
            for raw in evidence_values
            if (item := _mapping(raw, "evidence"))
        ),
        artifact_refs=_strings(value.get("artifact_refs", ())),
        changed_files=_strings(value.get("changed_files", ())),
        completion_claims=_strings(value.get("completion_claims", ())),
        usage=_mapping(value.get("usage", {}), "usage"),
        trace_id=_optional_value(value.get("trace_id")),
        change_set_id=_optional_value(value.get("change_set_id")),
    )


def _budget_from_dict(value: Mapping[str, Any]) -> RunBudget:
    raw_cost = value.get("max_cost")
    cost = None
    if raw_cost is not None:
        cost_value = _mapping(raw_cost, "max_cost")
        amount = cost_value.get("amount")
        cost = ExactCost(
            Decimal(str(amount)) if amount is not None else None,
            currency=str(cost_value.get("currency", "USD")),
            pricing_known=bool(cost_value.get("pricing_known", False)),
            estimated=bool(cost_value.get("estimated", False)),
            reported=bool(cost_value.get("reported", False)),
        )
    return RunBudget(
        scope=BudgetScope(str(value.get("scope", "child_task"))),
        max_model_calls=_optional_integer(value.get("max_model_calls"), "max_model_calls"),
        max_tool_calls=_optional_integer(value.get("max_tool_calls"), "max_tool_calls"),
        max_tokens=_optional_integer(value.get("max_tokens"), "max_tokens"),
        max_cost=cost,
        deadline=_optional_datetime(value.get("deadline"), "deadline"),
        unknown_cost_policy=UnknownCostPolicy(
            str(value.get("unknown_cost_policy", "fail_closed"))
        ),
    )


def _required(value: Any, label: str) -> str:
    clean = str(value).strip()
    if not clean:
        raise ValueError(f"{label} cannot be empty")
    return clean


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _optional_value(value: Any) -> str | None:
    return _optional(value)


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_required(value, "list item") for value in values))


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected an array of strings")
    return tuple(str(item) for item in value)


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    return _utc(parsed, label)


def _optional_datetime(value: Any, label: str) -> datetime | None:
    return None if value is None else _datetime(value, label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


__all__ = [
    "ChildCompletionDelivery",
    "ChildDeliveryStatus",
    "ChildEvidenceRef",
    "ChildTask",
    "ChildTaskClaim",
    "ChildTaskEvent",
    "ChildTaskLineage",
    "ChildTaskResult",
    "ChildTaskRole",
    "ChildTaskSpec",
    "ChildTaskStatus",
    "child_result_from_dict",
    "child_task_from_dict",
]
