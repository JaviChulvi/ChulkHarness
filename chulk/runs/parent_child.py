"""Immutable contracts for durable parent/child run orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Mapping

from chulk.results import freeze_mapping, plain_data
from chulk.runs.models import RunRecord
from chulk.usage import BudgetScope, ExactCost, RunBudget, UnknownCostPolicy


class ParentAggregationStatus(StrEnum):
    """Durable state of a parent run's child aggregation."""

    OPEN = "open"
    COMPLETED = "completed"


class ParentCompletionStatus(StrEnum):
    """Delivery state for one terminal parent aggregation."""

    PENDING = "pending"
    CLAIMED = "claimed"
    DELIVERED = "delivered"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ParentRunPolicy:
    """Immutable fan-out and aggregate budget limits for one parent."""

    required_children: int
    max_children: int
    budget: RunBudget

    def __post_init__(self) -> None:
        for name in ("required_children", "max_children"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.required_children > self.max_children:
            raise ValueError("required_children cannot exceed max_children")
        if self.budget.scope is not BudgetScope.CHILD_TASK:
            raise ValueError("parent child budget scope must be child_task")
        if not self.budget.limited:
            raise ValueError("parent child budget must define at least one limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_children": self.required_children,
            "max_children": self.max_children,
            "budget": self.budget.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParentRunPolicy":
        _require_keys(
            value,
            {"required_children", "max_children", "budget"},
            "parent run policy",
        )
        budget = value.get("budget")
        if not isinstance(budget, Mapping):
            raise ValueError("parent run policy budget must be an object")
        return cls(
            required_children=_integer(
                value.get("required_children"),
                "required_children",
            ),
            max_children=_integer(value.get("max_children"), "max_children"),
            budget=run_budget_from_dict(budget),
        )


@dataclass(frozen=True, slots=True)
class ChildRunProgress:
    """One ordered progress update committed by a live child lease."""

    child_run_id: str
    sequence: int
    payload: Mapping[str, Any]
    actor: str
    idempotency_key: str
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "child_run_id",
            _required(self.child_run_id, "child run id"),
        )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("child progress sequence must be a positive integer")
        object.__setattr__(self, "payload", freeze_mapping(self.payload))
        object.__setattr__(self, "actor", _required(self.actor, "progress actor"))
        object.__setattr__(
            self,
            "idempotency_key",
            _required(self.idempotency_key, "progress idempotency key"),
        )
        object.__setattr__(
            self,
            "created_at",
            _utc(self.created_at, "progress created_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_run_id": self.child_run_id,
            "sequence": self.sequence,
            "payload": plain_data(self.payload),
            "actor": self.actor,
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ChildRunRecord:
    """A durable child run plus its immutable parent-owned metadata."""

    parent_run_id: str
    run: RunRecord
    ordinal: int
    definition_revision: str
    progress: tuple[ChildRunProgress, ...] = ()
    terminal_evidence: Mapping[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_run_id",
            _required(self.parent_run_id, "parent run id"),
        )
        if self.run.scope.parent_run_id != self.parent_run_id:
            raise ValueError("child run scope does not match its durable parent")
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 1
        ):
            raise ValueError("child ordinal must be a positive integer")
        object.__setattr__(
            self,
            "definition_revision",
            _required(self.definition_revision, "definition revision"),
        )
        object.__setattr__(self, "progress", tuple(self.progress))
        if self.terminal_evidence is not None:
            object.__setattr__(
                self,
                "terminal_evidence",
                freeze_mapping(self.terminal_evidence),
            )
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "run": self.run.to_dict(),
            "ordinal": self.ordinal,
            "definition_revision": self.definition_revision,
            "progress": [item.to_dict() for item in self.progress],
            "terminal_evidence": (
                plain_data(self.terminal_evidence)
                if self.terminal_evidence is not None
                else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ParentRunRecord:
    """Recoverable aggregate view for one configured parent run."""

    run: RunRecord
    policy: ParentRunPolicy
    children: tuple[ChildRunRecord, ...]
    aggregation_status: ParentAggregationStatus
    aggregation_revision: int
    aggregate_result: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.run.scope.parent_run_id is not None:
            raise ValueError("a configured parent cannot itself be a child run")
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(
            self,
            "aggregation_status",
            ParentAggregationStatus(self.aggregation_status),
        )
        if (
            isinstance(self.aggregation_revision, bool)
            or not isinstance(self.aggregation_revision, int)
            or self.aggregation_revision < 0
        ):
            raise ValueError("aggregation revision cannot be negative")
        if self.aggregate_result is not None:
            object.__setattr__(
                self,
                "aggregate_result",
                freeze_mapping(self.aggregate_result),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "policy": self.policy.to_dict(),
            "children": [item.to_dict() for item in self.children],
            "aggregation_status": self.aggregation_status.value,
            "aggregation_revision": self.aggregation_revision,
            "aggregate_result": (
                plain_data(self.aggregate_result)
                if self.aggregate_result is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ParentCompletion:
    """One exactly-once terminal parent delivery record."""

    id: str
    parent_run_id: str
    status: ParentCompletionStatus
    payload: Mapping[str, Any]
    attempt_count: int
    worker_id: str | None
    lease_token: str | None
    lease_until: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime
    delivered_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "completion id"))
        object.__setattr__(
            self,
            "parent_run_id",
            _required(self.parent_run_id, "parent run id"),
        )
        object.__setattr__(self, "status", ParentCompletionStatus(self.status))
        object.__setattr__(self, "payload", freeze_mapping(self.payload))
        if (
            isinstance(self.attempt_count, bool)
            or not isinstance(self.attempt_count, int)
            or self.attempt_count < 0
        ):
            raise ValueError("completion attempt_count cannot be negative")
        object.__setattr__(self, "worker_id", _optional(self.worker_id))
        object.__setattr__(self, "lease_token", _optional(self.lease_token))
        object.__setattr__(self, "lease_until", _utc_optional(self.lease_until))
        object.__setattr__(self, "last_error", _optional(self.last_error))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        object.__setattr__(self, "delivered_at", _utc_optional(self.delivered_at))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_run_id": self.parent_run_id,
            "status": self.status.value,
            "payload": plain_data(self.payload),
            "attempt_count": self.attempt_count,
            "worker_id": self.worker_id,
            "lease_until": (
                self.lease_until.isoformat() if self.lease_until is not None else None
            ),
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "delivered_at": (
                self.delivered_at.isoformat()
                if self.delivered_at is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ParentCompletionClaim:
    """Lease token required to acknowledge one parent completion."""

    completion: ParentCompletion
    worker_id: str
    lease_token: str
    lease_until: datetime

    def __post_init__(self) -> None:
        if self.completion.status is not ParentCompletionStatus.CLAIMED:
            raise ValueError("parent completion claim must reference claimed delivery")
        object.__setattr__(
            self,
            "worker_id",
            _required(self.worker_id, "completion worker id"),
        )
        object.__setattr__(
            self,
            "lease_token",
            _required(self.lease_token, "completion lease token"),
        )
        object.__setattr__(
            self,
            "lease_until",
            _utc(self.lease_until, "completion lease_until"),
        )


def run_budget_from_dict(value: Mapping[str, Any]) -> RunBudget:
    """Parse the strict serialized form emitted by :class:`RunBudget`."""

    _require_keys(
        value,
        {
            "scope",
            "max_model_calls",
            "max_tool_calls",
            "max_tokens",
            "max_cost",
            "deadline",
            "unknown_cost_policy",
        },
        "run budget",
    )
    max_cost_value = value.get("max_cost")
    max_cost: ExactCost | None = None
    if max_cost_value is not None:
        if not isinstance(max_cost_value, Mapping):
            raise ValueError("run budget max_cost must be an object")
        _require_keys(
            max_cost_value,
            {"amount", "currency", "pricing_known", "estimated", "reported"},
            "run budget max_cost",
        )
        amount = max_cost_value.get("amount")
        max_cost = ExactCost(
            Decimal(str(amount)) if amount is not None else None,
            currency=str(max_cost_value.get("currency") or ""),
            pricing_known=_boolean(
                max_cost_value.get("pricing_known"),
                "max_cost.pricing_known",
            ),
            estimated=_boolean(
                max_cost_value.get("estimated"),
                "max_cost.estimated",
            ),
            reported=_boolean(
                max_cost_value.get("reported"),
                "max_cost.reported",
            ),
        )
    deadline_value = value.get("deadline")
    deadline = (
        datetime.fromisoformat(str(deadline_value))
        if deadline_value is not None
        else None
    )
    return RunBudget(
        scope=BudgetScope(str(value.get("scope") or "")),
        max_model_calls=_optional_integer(
            value.get("max_model_calls"),
            "max_model_calls",
        ),
        max_tool_calls=_optional_integer(
            value.get("max_tool_calls"),
            "max_tool_calls",
        ),
        max_tokens=_optional_integer(value.get("max_tokens"), "max_tokens"),
        max_cost=max_cost,
        deadline=deadline,
        unknown_cost_policy=UnknownCostPolicy(
            str(value.get("unknown_cost_policy") or "")
        ),
    )


def validate_child_budget_allocation(
    parent: RunBudget,
    child: RunBudget,
    existing: tuple[RunBudget, ...],
) -> None:
    """Fail closed when a child allocation exceeds the parent envelope."""

    if parent.scope is not BudgetScope.CHILD_TASK:
        raise ValueError("parent child budget scope must be child_task")
    if child.scope is not BudgetScope.CHILD_TASK:
        raise ValueError("child run budget scope must be child_task")
    for name in ("max_model_calls", "max_tool_calls", "max_tokens"):
        parent_limit = getattr(parent, name)
        child_limit = getattr(child, name)
        if parent_limit is None:
            continue
        if child_limit is None:
            raise ValueError(f"child {name} must be bounded by the parent")
        allocated = sum(getattr(item, name) or 0 for item in existing)
        if allocated + child_limit > parent_limit:
            raise ValueError(f"child {name} allocation exceeds the parent budget")
    if parent.max_cost is not None:
        if child.max_cost is None:
            raise ValueError("child max_cost must be bounded by the parent")
        if child.max_cost.currency != parent.max_cost.currency:
            raise ValueError("child max_cost currency must match the parent")
        allocated_cost = sum(
            (
                (
                    item.max_cost.amount
                    if item.max_cost is not None
                    and item.max_cost.amount is not None
                    else Decimal(0)
                )
                for item in existing
            ),
            Decimal(0),
        )
        child_amount = child.max_cost.amount or Decimal(0)
        parent_amount = parent.max_cost.amount or Decimal(0)
        if allocated_cost + child_amount > parent_amount:
            raise ValueError("child max_cost allocation exceeds the parent budget")
    if parent.deadline is not None and (
        child.deadline is None or child.deadline > parent.deadline
    ):
        raise ValueError("child deadline exceeds the parent budget deadline")
    if (
        parent.unknown_cost_policy is UnknownCostPolicy.FAIL_CLOSED
        and child.unknown_cost_policy is not UnknownCostPolicy.FAIL_CLOSED
    ):
        raise ValueError("child unknown-cost policy is broader than the parent")


def _require_keys(
    value: Mapping[str, Any],
    known: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - known)
    missing = sorted(known - set(value))
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")


def _required(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value.strip()


def _optional(value: object) -> str | None:
    if value is None:
        return None
    return _required(value, "optional value")


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return value.astimezone(timezone.utc)


def _utc_optional(value: datetime | None) -> datetime | None:
    return _utc(value, "datetime") if value is not None else None


__all__ = [
    "ChildRunProgress",
    "ChildRunRecord",
    "ParentAggregationStatus",
    "ParentCompletion",
    "ParentCompletionClaim",
    "ParentCompletionStatus",
    "ParentRunPolicy",
    "ParentRunRecord",
]
