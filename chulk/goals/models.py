"""Immutable domain models for durable, operator-owned goals."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from chulk.usage import BudgetScope, ExactCost, RunBudget, UnknownCostPolicy


class GoalStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class GoalStepStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class GoalRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GoalActionState(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class GoalCriterion:
    """One goal-level result that must be evidenced before completion."""

    id: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "criterion id"))
        object.__setattr__(
            self,
            "description",
            _required(self.description, "criterion description"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "description": self.description}


@dataclass(frozen=True, slots=True)
class GoalEvidence:
    """Evidence tied to one step and one or more acceptance criteria."""

    id: str
    summary: str
    criterion_ids: tuple[str, ...]
    step_id: str | None = None
    kind: str = "observation"
    reference: str | None = None
    recorded_by: str = "runner"
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "evidence id"))
        object.__setattr__(self, "summary", _required(self.summary, "evidence summary"))
        object.__setattr__(
            self,
            "criterion_ids",
            _unique_required(self.criterion_ids, "criterion id"),
        )
        object.__setattr__(self, "step_id", _optional(self.step_id))
        object.__setattr__(self, "kind", _required(self.kind, "evidence kind"))
        object.__setattr__(self, "reference", _optional(self.reference))
        object.__setattr__(
            self,
            "recorded_by",
            _required(self.recorded_by, "evidence actor"),
        )
        object.__setattr__(self, "recorded_at", _utc(self.recorded_at, "recorded_at"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "criterion_ids": list(self.criterion_ids),
            "step_id": self.step_id,
            "kind": self.kind,
            "reference": self.reference,
            "recorded_by": self.recorded_by,
            "recorded_at": self.recorded_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GoalApproval:
    """Auditable operator approval for a goal or selected high-risk step."""

    id: str
    approved_by: str
    scope: str = "goal"
    step_id: str | None = None
    reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "approval id"))
        object.__setattr__(
            self,
            "approved_by",
            _required(self.approved_by, "approval actor"),
        )
        if self.scope not in {"goal", "step", "skip"}:
            raise ValueError("approval scope must be goal, step, or skip")
        object.__setattr__(self, "step_id", _optional(self.step_id))
        if self.scope != "goal" and self.step_id is None:
            raise ValueError(f"{self.scope} approval requires step_id")
        object.__setattr__(self, "reason", _optional(self.reason))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "approved_by": self.approved_by,
            "scope": self.scope,
            "step_id": self.step_id,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class GoalSteering:
    """Append-only operator guidance observed by a runner at boundaries."""

    id: str
    instruction: str
    created_by: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "steering id"))
        object.__setattr__(
            self,
            "instruction",
            _required(self.instruction, "steering instruction"),
        )
        object.__setattr__(
            self,
            "created_by",
            _required(self.created_by, "steering actor"),
        )
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class GoalStep:
    """One durable step in a dependency-ordered goal."""

    id: str
    title: str
    description: str
    acceptance_criterion_ids: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    risk: GoalRisk = GoalRisk.LOW
    expected_tools: tuple[str, ...] = ()
    status: GoalStepStatus = GoalStepStatus.PENDING
    attempt: int = 0
    max_attempts: int = 1
    evidence_ids: tuple[str, ...] = ()
    blocked_reason: str | None = None
    skip_reason: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "step id"))
        object.__setattr__(self, "title", _required(self.title, "step title"))
        object.__setattr__(
            self,
            "description",
            _required(self.description, "step description"),
        )
        object.__setattr__(
            self,
            "acceptance_criterion_ids",
            _unique_required(
                self.acceptance_criterion_ids,
                "acceptance criterion id",
            ),
        )
        object.__setattr__(
            self,
            "depends_on",
            _unique_optional(self.depends_on),
        )
        object.__setattr__(self, "risk", GoalRisk(self.risk))
        object.__setattr__(
            self,
            "expected_tools",
            _unique_optional(self.expected_tools),
        )
        object.__setattr__(self, "status", GoalStepStatus(self.status))
        if isinstance(self.attempt, bool) or self.attempt < 0:
            raise ValueError("step attempt must be a non-negative integer")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("step max_attempts must be a positive integer")
        if self.attempt > self.max_attempts:
            raise ValueError("step attempt cannot exceed max_attempts")
        object.__setattr__(
            self,
            "evidence_ids",
            _unique_optional(self.evidence_ids),
        )
        object.__setattr__(self, "blocked_reason", _optional(self.blocked_reason))
        object.__setattr__(self, "skip_reason", _optional(self.skip_reason))
        if self.started_at is not None:
            object.__setattr__(
                self,
                "started_at",
                _utc(self.started_at, "started_at"),
            )
        if self.completed_at is not None:
            object.__setattr__(
                self,
                "completed_at",
                _utc(self.completed_at, "completed_at"),
            )

    @property
    def terminal(self) -> bool:
        return self.status in {
            GoalStepStatus.COMPLETED,
            GoalStepStatus.SKIPPED,
            GoalStepStatus.FAILED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "acceptance_criterion_ids": list(self.acceptance_criterion_ids),
            "depends_on": list(self.depends_on),
            "risk": self.risk.value,
            "expected_tools": list(self.expected_tools),
            "status": self.status.value,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "evidence_ids": list(self.evidence_ids),
            "blocked_reason": self.blocked_reason,
            "skip_reason": self.skip_reason,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
        }


@dataclass(frozen=True, slots=True)
class Goal:
    """Complete revisioned goal snapshot stored by the host."""

    id: str
    profile_id: str
    title: str
    acceptance_criteria: tuple[GoalCriterion, ...]
    steps: tuple[GoalStep, ...]
    budget: RunBudget
    status: GoalStatus = GoalStatus.DRAFT
    revision: int = 0
    evidence: tuple[GoalEvidence, ...] = ()
    approvals: tuple[GoalApproval, ...] = ()
    steering: tuple[GoalSteering, ...] = ()
    source_conversation_id: str | None = None
    source_turn_id: str | None = None
    source_plan: Mapping[str, Any] | None = None
    child_task_ids: tuple[str, ...] = ()
    schedule_ids: tuple[str, ...] = ()
    process_ids: tuple[str, ...] = ()
    trace_ids: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    cancellation_requested: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "goal id"))
        object.__setattr__(self, "profile_id", _required(self.profile_id, "profile id"))
        object.__setattr__(self, "title", _required(self.title, "goal title"))
        object.__setattr__(self, "acceptance_criteria", tuple(self.acceptance_criteria))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "status", GoalStatus(self.status))
        if self.budget.scope is not BudgetScope.GOAL:
            raise ValueError("durable goal budget scope must be goal")
        if isinstance(self.revision, bool) or self.revision < 0:
            raise ValueError("goal revision must be a non-negative integer")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "approvals", tuple(self.approvals))
        object.__setattr__(self, "steering", tuple(self.steering))
        object.__setattr__(
            self,
            "source_conversation_id",
            _optional(self.source_conversation_id),
        )
        object.__setattr__(self, "source_turn_id", _optional(self.source_turn_id))
        if self.source_plan is not None:
            object.__setattr__(
                self,
                "source_plan",
                MappingProxyType(dict(self.source_plan)),
            )
        for field_name in (
            "child_task_ids",
            "schedule_ids",
            "process_ids",
            "trace_ids",
            "artifact_refs",
        ):
            object.__setattr__(
                self,
                field_name,
                _unique_optional(getattr(self, field_name)),
            )
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        for field_name in ("approved_at", "started_at", "completed_at"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _utc(value, field_name))
        object.__setattr__(self, "last_error", _optional(self.last_error))
        _validate_goal_graph(self.acceptance_criteria, self.steps, self.evidence)

    @property
    def terminal(self) -> bool:
        return self.status in {
            GoalStatus.COMPLETED,
            GoalStatus.CANCELLED,
            GoalStatus.FAILED,
        }

    @property
    def evidenced_criterion_ids(self) -> frozenset[str]:
        return frozenset(
            criterion_id
            for item in self.evidence
            for criterion_id in item.criterion_ids
        )

    @property
    def missing_criterion_ids(self) -> tuple[str, ...]:
        evidenced = self.evidenced_criterion_ids
        return tuple(
            item.id
            for item in self.acceptance_criteria
            if item.id not in evidenced
        )

    def step(self, step_id: str) -> GoalStep:
        for item in self.steps:
            if item.id == step_id:
                return item
        raise KeyError(step_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "title": self.title,
            "acceptance_criteria": [
                criterion.to_dict() for criterion in self.acceptance_criteria
            ],
            "steps": [step.to_dict() for step in self.steps],
            "budget": self.budget.to_dict(),
            "status": self.status.value,
            "revision": self.revision,
            "evidence": [item.to_dict() for item in self.evidence],
            "approvals": [item.to_dict() for item in self.approvals],
            "steering": [item.to_dict() for item in self.steering],
            "source_conversation_id": self.source_conversation_id,
            "source_turn_id": self.source_turn_id,
            "source_plan": dict(self.source_plan) if self.source_plan is not None else None,
            "child_task_ids": list(self.child_task_ids),
            "schedule_ids": list(self.schedule_ids),
            "process_ids": list(self.process_ids),
            "trace_ids": list(self.trace_ids),
            "artifact_refs": list(self.artifact_refs),
            "cancellation_requested": self.cancellation_requested,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "approved_at": _iso(self.approved_at),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "last_error": self.last_error,
        }

    def with_revision(self, revision: int, *, now: datetime) -> Goal:
        return replace(self, revision=revision, updated_at=now)


@dataclass(frozen=True, slots=True)
class GoalEvent:
    id: str
    goal_id: str
    profile_id: str
    revision: int
    kind: str
    actor: str
    payload: Mapping[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "profile_id": self.profile_id,
            "revision": self.revision,
            "kind": self.kind,
            "actor": self.actor,
            "payload": dict(self.payload),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class GoalClaim:
    goal_id: str
    profile_id: str
    runner_id: str
    claim_token: str
    lease_until: datetime


@dataclass(frozen=True, slots=True)
class GoalActionCheckpoint:
    id: str
    goal_id: str
    step_id: str
    profile_id: str
    idempotency_key: str
    action_kind: str
    action_ref: str | None
    state: GoalActionState
    claim_token: str
    created_at: datetime
    updated_at: datetime
    result: Mapping[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", GoalActionState(self.state))
        if self.result is not None:
            object.__setattr__(self, "result", MappingProxyType(dict(self.result)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "step_id": self.step_id,
            "profile_id": self.profile_id,
            "idempotency_key": self.idempotency_key,
            "action_kind": self.action_kind,
            "action_ref": self.action_ref,
            "state": self.state.value,
            "claim_token": self.claim_token,
            "result": dict(self.result) if self.result is not None else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


def goal_from_dict(value: Mapping[str, Any]) -> Goal:
    budget = _budget_from_dict(_mapping(value.get("budget"), "budget"))
    return Goal(
        id=str(value["id"]),
        profile_id=str(value["profile_id"]),
        title=str(value["title"]),
        acceptance_criteria=tuple(
            GoalCriterion(
                id=str(item["id"]),
                description=str(item["description"]),
            )
            for item in _mapping_items(
                value.get("acceptance_criteria"),
                "acceptance_criteria",
            )
        ),
        steps=tuple(_step_from_dict(item) for item in _mapping_items(value.get("steps"), "steps")),
        budget=budget,
        status=GoalStatus(str(value.get("status", GoalStatus.DRAFT.value))),
        revision=int(value.get("revision", 0)),
        evidence=tuple(
            _evidence_from_dict(item)
            for item in _mapping_items(value.get("evidence", ()), "evidence")
        ),
        approvals=tuple(
            _approval_from_dict(item)
            for item in _mapping_items(value.get("approvals", ()), "approvals")
        ),
        steering=tuple(
            _steering_from_dict(item)
            for item in _mapping_items(value.get("steering", ()), "steering")
        ),
        source_conversation_id=_optional_text_value(value.get("source_conversation_id")),
        source_turn_id=_optional_text_value(value.get("source_turn_id")),
        source_plan=(
            _mapping(value.get("source_plan"), "source_plan")
            if value.get("source_plan") is not None
            else None
        ),
        child_task_ids=_strings(value.get("child_task_ids", ())),
        schedule_ids=_strings(value.get("schedule_ids", ())),
        process_ids=_strings(value.get("process_ids", ())),
        trace_ids=_strings(value.get("trace_ids", ())),
        artifact_refs=_strings(value.get("artifact_refs", ())),
        cancellation_requested=bool(value.get("cancellation_requested", False)),
        created_at=_datetime(value["created_at"], "created_at"),
        updated_at=_datetime(value["updated_at"], "updated_at"),
        approved_at=_optional_datetime(value.get("approved_at"), "approved_at"),
        started_at=_optional_datetime(value.get("started_at"), "started_at"),
        completed_at=_optional_datetime(value.get("completed_at"), "completed_at"),
        last_error=_optional_text_value(value.get("last_error")),
    )


def _step_from_dict(value: Mapping[str, Any]) -> GoalStep:
    return GoalStep(
        id=str(value["id"]),
        title=str(value["title"]),
        description=str(value["description"]),
        acceptance_criterion_ids=_strings(value.get("acceptance_criterion_ids", ())),
        depends_on=_strings(value.get("depends_on", ())),
        risk=GoalRisk(str(value.get("risk", GoalRisk.LOW.value))),
        expected_tools=_strings(value.get("expected_tools", ())),
        status=GoalStepStatus(str(value.get("status", GoalStepStatus.PENDING.value))),
        attempt=int(value.get("attempt", 0)),
        max_attempts=int(value.get("max_attempts", 1)),
        evidence_ids=_strings(value.get("evidence_ids", ())),
        blocked_reason=_optional_text_value(value.get("blocked_reason")),
        skip_reason=_optional_text_value(value.get("skip_reason")),
        started_at=_optional_datetime(value.get("started_at"), "started_at"),
        completed_at=_optional_datetime(value.get("completed_at"), "completed_at"),
    )


def _evidence_from_dict(value: Mapping[str, Any]) -> GoalEvidence:
    return GoalEvidence(
        id=str(value["id"]),
        summary=str(value["summary"]),
        criterion_ids=_strings(value.get("criterion_ids", ())),
        step_id=_optional_text_value(value.get("step_id")),
        kind=str(value.get("kind", "observation")),
        reference=_optional_text_value(value.get("reference")),
        recorded_by=str(value.get("recorded_by", "runner")),
        recorded_at=_datetime(value["recorded_at"], "recorded_at"),
        metadata=_mapping(value.get("metadata", {}), "metadata"),
    )


def _approval_from_dict(value: Mapping[str, Any]) -> GoalApproval:
    return GoalApproval(
        id=str(value["id"]),
        approved_by=str(value["approved_by"]),
        scope=str(value.get("scope", "goal")),
        step_id=_optional_text_value(value.get("step_id")),
        reason=_optional_text_value(value.get("reason")),
        created_at=_datetime(value["created_at"], "created_at"),
    )


def _steering_from_dict(value: Mapping[str, Any]) -> GoalSteering:
    return GoalSteering(
        id=str(value["id"]),
        instruction=str(value["instruction"]),
        created_by=str(value["created_by"]),
        created_at=_datetime(value["created_at"], "created_at"),
    )


def _budget_from_dict(value: Mapping[str, Any]) -> RunBudget:
    raw_cost = value.get("max_cost")
    cost = None
    if raw_cost is not None:
        cost_values = _mapping(raw_cost, "max_cost")
        raw_amount = cost_values.get("amount")
        cost = ExactCost(
            Decimal(str(raw_amount)) if raw_amount is not None else None,
            currency=str(cost_values.get("currency", "USD")),
            pricing_known=bool(cost_values.get("pricing_known", False)),
            estimated=bool(cost_values.get("estimated", False)),
            reported=bool(cost_values.get("reported", False)),
        )
    return RunBudget(
        scope=BudgetScope(str(value.get("scope", BudgetScope.GOAL.value))),
        max_model_calls=_optional_int(value.get("max_model_calls"), "max_model_calls"),
        max_tool_calls=_optional_int(value.get("max_tool_calls"), "max_tool_calls"),
        max_tokens=_optional_int(value.get("max_tokens"), "max_tokens"),
        max_cost=cost,
        deadline=_optional_datetime(value.get("deadline"), "deadline"),
        unknown_cost_policy=UnknownCostPolicy(
            str(
                value.get(
                    "unknown_cost_policy",
                    UnknownCostPolicy.FAIL_CLOSED.value,
                )
            )
        ),
    )


def _validate_goal_graph(
    criteria: tuple[GoalCriterion, ...],
    steps: tuple[GoalStep, ...],
    evidence: tuple[GoalEvidence, ...],
) -> None:
    criterion_ids = [item.id for item in criteria]
    if not criterion_ids:
        raise ValueError("goal requires at least one acceptance criterion")
    if len(set(criterion_ids)) != len(criterion_ids):
        raise ValueError("goal acceptance criterion ids must be unique")
    step_ids = [item.id for item in steps]
    if not step_ids:
        raise ValueError("goal requires at least one step")
    if len(set(step_ids)) != len(step_ids):
        raise ValueError("goal step ids must be unique")
    known_criteria = set(criterion_ids)
    known_steps = set(step_ids)
    for step in steps:
        unknown_criteria = set(step.acceptance_criterion_ids) - known_criteria
        if unknown_criteria:
            raise ValueError(
                f"goal step {step.id} references unknown criteria: "
                f"{', '.join(sorted(unknown_criteria))}"
            )
        unknown_steps = set(step.depends_on) - known_steps
        if unknown_steps:
            raise ValueError(
                f"goal step {step.id} references unknown dependencies: "
                f"{', '.join(sorted(unknown_steps))}"
            )
        if step.id in step.depends_on:
            raise ValueError(f"goal step {step.id} cannot depend on itself")
    _reject_cycles(steps)
    evidence_ids: set[str] = set()
    for item in evidence:
        if item.id in evidence_ids:
            raise ValueError(f"duplicate goal evidence id: {item.id}")
        evidence_ids.add(item.id)
        if item.step_id is not None and item.step_id not in known_steps:
            raise ValueError(f"goal evidence references unknown step: {item.step_id}")
        unknown_criteria = set(item.criterion_ids) - known_criteria
        if unknown_criteria:
            raise ValueError(
                "goal evidence references unknown criteria: "
                f"{', '.join(sorted(unknown_criteria))}"
            )
    for step in steps:
        unknown_evidence = set(step.evidence_ids) - evidence_ids
        if unknown_evidence:
            raise ValueError(
                f"goal step {step.id} references unknown evidence: "
                f"{', '.join(sorted(unknown_evidence))}"
            )


def _reject_cycles(steps: tuple[GoalStep, ...]) -> None:
    dependencies = {item.id: set(item.depends_on) for item in steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visited:
            return
        if step_id in visiting:
            raise ValueError("goal step dependencies must be acyclic")
        visiting.add(step_id)
        for dependency in dependencies[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in dependencies:
        visit(step_id)


def _required(value: str, label: str) -> str:
    clean = str(value).strip()
    if not clean:
        raise ValueError(f"{label} cannot be empty")
    return clean


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _unique_required(values: tuple[str, ...], label: str) -> tuple[str, ...]:
    clean = _unique_optional(values)
    if not clean:
        raise ValueError(f"{label} list cannot be empty")
    return clean


def _unique_optional(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_required(item, "list item") for item in values))


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


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


def _mapping_items(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return tuple(_mapping(item, label) for item in value)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected an array of strings")
    return tuple(str(item) for item in value)


def _optional_text_value(value: Any) -> str | None:
    return _optional(str(value)) if value is not None else None


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


__all__ = [
    "Goal",
    "GoalActionCheckpoint",
    "GoalActionState",
    "GoalApproval",
    "GoalClaim",
    "GoalCriterion",
    "GoalEvent",
    "GoalEvidence",
    "GoalRisk",
    "GoalStatus",
    "GoalSteering",
    "GoalStep",
    "GoalStepStatus",
    "goal_from_dict",
]
