"""Typed durable usage, cost, and budget models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from chulk.errors import ChulkError, ErrorDetails


class ResourceKind(StrEnum):
    """Metered resource families supported by the shared ledger."""

    MODEL = "model"
    TOOL = "tool"
    MEDIA = "media"
    EXTERNAL_SERVICE = "external_service"


class BudgetScope(StrEnum):
    """Durable ownership boundary used when evaluating a run budget."""

    PROFILE = "profile"
    CONVERSATION = "conversation"
    TURN = "turn"
    GOAL = "goal"
    JOB = "job"
    CHILD_TASK = "child_task"


class UnknownCostPolicy(StrEnum):
    """How a cost-limited run treats a request without catalogued pricing."""

    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"


class ReservationState(StrEnum):
    ACTIVE = "active"
    COMMITTED = "committed"
    RELEASED = "released"


class UsageGroupBy(StrEnum):
    RESOURCE_KIND = "resource_kind"
    MODEL = "model"
    TOOL_SERVICE = "tool_service"
    PROFILE = "profile"
    CHANNEL = "channel"
    CONVERSATION = "conversation"
    GOAL = "goal"
    JOB = "job"
    CHILD_TASK = "child_task"


@dataclass(frozen=True, slots=True)
class ExactCost:
    """An exact decimal cost amount, or an explicit unknown-price marker."""

    amount: Decimal | None
    currency: str = "USD"
    pricing_known: bool = False
    estimated: bool = False
    reported: bool = False

    def __post_init__(self) -> None:
        if self.amount is not None:
            if not isinstance(self.amount, Decimal):
                raise TypeError("cost amount must be a Decimal or None")
            if not self.amount.is_finite() or self.amount < 0:
                raise ValueError("cost amount must be a finite non-negative Decimal")
        currency = self.currency.strip().upper()
        if not currency:
            raise ValueError("cost currency cannot be empty")
        if self.pricing_known and self.amount is None:
            raise ValueError("known pricing requires an amount")
        object.__setattr__(self, "currency", currency)

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": str(self.amount) if self.amount is not None else None,
            "currency": self.currency,
            "pricing_known": self.pricing_known,
            "estimated": self.estimated,
            "reported": self.reported,
        }


@dataclass(frozen=True, slots=True)
class UsageDimensions:
    """Nullable ownership dimensions attached to one metered event."""

    profile_id: str = "default"
    channel: str | None = None
    conversation_id: str | None = None
    turn_id: str | None = None
    goal_id: str | None = None
    job_id: str | None = None
    child_task_id: str | None = None

    def __post_init__(self) -> None:
        profile_id = self.profile_id.strip()
        if not profile_id:
            raise ValueError("usage profile_id cannot be empty")
        object.__setattr__(self, "profile_id", profile_id)
        for field_name in (
            "channel",
            "conversation_id",
            "turn_id",
            "goal_id",
            "job_id",
            "child_task_id",
        ):
            value = getattr(self, field_name)
            if value is not None:
                clean = value.strip()
                object.__setattr__(self, field_name, clean or None)

    def scope_values(self, scope: BudgetScope) -> dict[str, str]:
        values = {"profile_id": self.profile_id}
        required: tuple[str, ...]
        if scope is BudgetScope.PROFILE:
            required = ()
        elif scope is BudgetScope.CONVERSATION:
            required = ("conversation_id",)
        elif scope is BudgetScope.TURN:
            required = ("conversation_id", "turn_id")
        elif scope is BudgetScope.GOAL:
            required = ("goal_id",)
        elif scope is BudgetScope.JOB:
            required = ("job_id",)
        else:
            required = ("child_task_id",)
        for field_name in required:
            value = getattr(self, field_name)
            if value is None:
                raise ValueError(f"{scope.value} budget scope requires {field_name}")
            values[field_name] = value
        return values

    def to_dict(self) -> dict[str, str | None]:
        return {
            "profile_id": self.profile_id,
            "channel": self.channel,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "goal_id": self.goal_id,
            "job_id": self.job_id,
            "child_task_id": self.child_task_id,
        }


@dataclass(frozen=True, slots=True)
class RunBudget:
    """Model, tool, token, time, and exact-cost ceilings for one run scope."""

    scope: BudgetScope = BudgetScope.TURN
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_cost: ExactCost | None = None
    deadline: datetime | None = None
    unknown_cost_policy: UnknownCostPolicy = UnknownCostPolicy.FAIL_CLOSED

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", BudgetScope(self.scope))
        object.__setattr__(
            self,
            "unknown_cost_policy",
            UnknownCostPolicy(self.unknown_cost_policy),
        )
        for field_name in ("max_model_calls", "max_tool_calls", "max_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_cost is not None:
            if self.max_cost.amount is None:
                raise ValueError("a cost ceiling requires an exact amount")
            if not self.max_cost.pricing_known:
                raise ValueError("a cost ceiling must be marked as known")
        if self.deadline is not None and self.deadline.tzinfo is None:
            raise ValueError("budget deadline must be timezone-aware")

    @property
    def limited(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_model_calls,
                self.max_tool_calls,
                self.max_tokens,
                self.max_cost,
                self.deadline,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost.to_dict()
            if self.max_cost is not None
            else None,
            "deadline": self.deadline.isoformat()
            if self.deadline is not None
            else None,
            "unknown_cost_policy": self.unknown_cost_policy.value,
        }


@dataclass(frozen=True, slots=True)
class UsageEntry:
    """One immutable, idempotently ingested metered resource event."""

    id: str
    resource_kind: ResourceKind
    source_event_id: str
    dimensions: UsageDimensions
    occurred_at: datetime
    billing_period: str
    purpose: str
    units: Mapping[str, Decimal] = field(default_factory=dict)
    cost: ExactCost = field(default_factory=lambda: ExactCost(None))
    provider: str | None = None
    model: str | None = None
    tool_or_service: str | None = None
    credential_ref: str | None = None
    model_profile_id: str | None = None
    usage_estimated: bool = False
    trace_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_kind", ResourceKind(self.resource_kind))
        if not self.id.strip() or not self.source_event_id.strip():
            raise ValueError("usage entry ids cannot be empty")
        if self.occurred_at.tzinfo is None:
            raise ValueError("usage entry timestamp must be timezone-aware")
        object.__setattr__(
            self,
            "occurred_at",
            self.occurred_at.astimezone(timezone.utc),
        )
        if not self.billing_period.strip():
            raise ValueError("billing_period cannot be empty")
        if not self.purpose.strip():
            raise ValueError("usage purpose cannot be empty")
        clean_units: dict[str, Decimal] = {}
        for name, quantity in self.units.items():
            clean_name = str(name).strip()
            if not clean_name:
                raise ValueError("usage unit names cannot be empty")
            decimal_quantity = (
                quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
            )
            if not decimal_quantity.is_finite() or decimal_quantity < 0:
                raise ValueError("usage unit quantities must be finite and non-negative")
            clean_units[clean_name] = decimal_quantity
        object.__setattr__(self, "units", MappingProxyType(clean_units))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "resource_kind": self.resource_kind.value,
            "source_event_id": self.source_event_id,
            **self.dimensions.to_dict(),
            "occurred_at": self.occurred_at.isoformat(),
            "billing_period": self.billing_period,
            "purpose": self.purpose,
            "units": {key: str(value) for key, value in self.units.items()},
            "cost": self.cost.to_dict(),
            "provider": self.provider,
            "model": self.model,
            "tool_or_service": self.tool_or_service,
            "credential_ref": self.credential_ref,
            "model_profile_id": self.model_profile_id,
            "usage_estimated": self.usage_estimated,
            "trace_path": self.trace_path,
            "metadata": dict(self.metadata),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Return safe query/export data without credential references."""
        payload = self.to_dict()
        payload.pop("credential_ref", None)
        return payload


@dataclass(frozen=True, slots=True)
class UsageQuery:
    """Bounded, profile-owned ledger query."""

    profile_id: str
    start: datetime | None = None
    end: datetime | None = None
    resource_kind: ResourceKind | None = None
    channel: str | None = None
    conversation_id: str | None = None
    goal_id: str | None = None
    job_id: str | None = None
    child_task_id: str | None = None
    limit: int = 100
    cursor: str | None = None

    def __post_init__(self) -> None:
        profile_id = self.profile_id.strip()
        if not profile_id:
            raise ValueError("usage query profile_id cannot be empty")
        object.__setattr__(self, "profile_id", profile_id)
        if self.resource_kind is not None:
            object.__setattr__(
                self,
                "resource_kind",
                ResourceKind(self.resource_kind),
            )
        for field_name in ("start", "end"):
            value = getattr(self, field_name)
            if value is not None:
                if value.tzinfo is None:
                    raise ValueError(f"usage query {field_name} must be timezone-aware")
                object.__setattr__(self, field_name, value.astimezone(timezone.utc))
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("usage query start must be earlier than end")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError("usage query limit must be an integer")
        if self.limit < 1 or self.limit > 10_000:
            raise ValueError("usage query limit must be between 1 and 10000")


@dataclass(frozen=True, slots=True)
class UsagePage:
    """One deterministic page of immutable usage entries."""

    entries: tuple[UsageEntry, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_public_dict() for entry in self.entries],
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True, slots=True)
class UsageAggregate:
    """Exact totals for one query group."""

    key: str
    entry_count: int
    model_calls: int
    tool_calls: int
    total_tokens: int
    cost: ExactCost
    unknown_cost_entries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "entry_count": self.entry_count,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "cost": self.cost.to_dict(),
            "unknown_cost_entries": self.unknown_cost_entries,
        }


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """One durable allowance held before a metered operation starts."""

    id: str
    idempotency_key: str
    source_event_id: str
    resource_kind: ResourceKind
    dimensions: UsageDimensions
    budget: RunBudget
    state: ReservationState
    reserved_model_calls: int = 0
    reserved_tool_calls: int = 0
    reserved_tokens: int = 0
    reserved_cost: ExactCost = field(default_factory=lambda: ExactCost(None))
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_kind", ResourceKind(self.resource_kind))
        object.__setattr__(self, "state", ReservationState(self.state))
        if not self.id.strip() or not self.idempotency_key.strip():
            raise ValueError("reservation ids cannot be empty")
        for field_name in (
            "reserved_model_calls",
            "reserved_tool_calls",
            "reserved_tokens",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "idempotency_key": self.idempotency_key,
            "source_event_id": self.source_event_id,
            "resource_kind": self.resource_kind.value,
            "dimensions": self.dimensions.to_dict(),
            "budget": self.budget.to_dict(),
            "state": self.state.value,
            "reserved_model_calls": self.reserved_model_calls,
            "reserved_tool_calls": self.reserved_tool_calls,
            "reserved_tokens": self.reserved_tokens,
            "reserved_cost": self.reserved_cost.to_dict(),
            "created_at": self.created_at.isoformat()
            if self.created_at is not None
            else None,
            "updated_at": self.updated_at.isoformat()
            if self.updated_at is not None
            else None,
            "expires_at": self.expires_at.isoformat()
            if self.expires_at is not None
            else None,
        }


@dataclass(frozen=True, slots=True)
class PersistedModelUsage:
    """Private provider-result checkpoint used to reconcile an active hold."""

    reservation: BudgetReservation
    request_index: int
    purpose: str
    occurred_at: datetime
    usage: Mapping[str, Any] | None = None
    cost: Mapping[str, Any] | None = None
    fallback_attempts: tuple[Mapping[str, Any], ...] = ()
    provider: str | None = None
    model: str | None = None
    model_profile_id: str | None = None
    credential_ref: str | None = None
    trace_path: str | None = None

    def __post_init__(self) -> None:
        if self.request_index < 1:
            raise ValueError("persisted model request index must be positive")
        if not self.purpose.strip():
            raise ValueError("persisted model usage purpose cannot be empty")
        if self.occurred_at.tzinfo is None:
            raise ValueError("persisted model usage timestamp must be timezone-aware")
        object.__setattr__(
            self,
            "occurred_at",
            self.occurred_at.astimezone(timezone.utc),
        )
        if self.usage is not None:
            object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))
        if self.cost is not None:
            object.__setattr__(self, "cost", MappingProxyType(dict(self.cost)))
        object.__setattr__(
            self,
            "fallback_attempts",
            tuple(MappingProxyType(dict(value)) for value in self.fallback_attempts),
        )


class BudgetExceededError(ChulkError):
    """Raised before work starts when a shared budget has no remaining allowance."""

    def __init__(
        self,
        *,
        scope: BudgetScope,
        dimension: str,
        limit: str,
        committed: str,
        reserved: str,
        requested: str,
    ) -> None:
        self.scope = scope
        self.dimension = dimension
        self.limit = limit
        self.committed = committed
        self.reserved = reserved
        self.requested = requested
        super().__init__(
            f"{scope.value} budget exhausted for {dimension}: limit {limit}, "
            f"committed {committed}, reserved {reserved}, requested {requested}",
            details=ErrorDetails(
                failure_kind="budget_exhausted",
                extensions={
                    "scope": scope.value,
                    "dimension": dimension,
                    "limit": limit,
                    "committed": committed,
                    "reserved": reserved,
                    "requested": requested,
                },
            ),
        )

    category = "budget_exhausted"


__all__ = [
    "BudgetExceededError",
    "BudgetReservation",
    "BudgetScope",
    "ExactCost",
    "ReservationState",
    "ResourceKind",
    "RunBudget",
    "UnknownCostPolicy",
    "UsageDimensions",
    "UsageEntry",
    "UsageAggregate",
    "UsageGroupBy",
    "UsagePage",
    "UsageQuery",
]
