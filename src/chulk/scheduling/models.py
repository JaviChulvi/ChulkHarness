"""Immutable contracts for profile-owned automation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from chulk.gateway import DeliveryTarget
from chulk.redaction import redact_data
from chulk.usage import BudgetScope, RunBudget


MAX_SCHEDULED_PROMPT_CHARS = 8_000
MAX_TRIGGER_PAYLOAD_BYTES = 64 * 1024


class RecurrenceKind(StrEnum):
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    RRULE = "rrule"


class MisfirePolicy(StrEnum):
    SKIP = "skip"
    RUN_ONCE = "run_once"
    CATCH_UP = "catch_up"


class NonexistentTimePolicy(StrEnum):
    SHIFT_FORWARD = "shift_forward"
    SKIP = "skip"


class AmbiguousTimePolicy(StrEnum):
    EARLIEST = "earliest"
    LATEST = "latest"


class AutomationJobStatus(StrEnum):
    PENDING_APPROVAL = "pending_approval"
    ACTIVE = "active"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class AutomationRunStatus(StrEnum):
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class AutomationRunReason(StrEnum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    TRIGGER = "trigger"
    RETRY = "retry"


class AutomationDeliveryState(StrEnum):
    NONE = "none"
    PENDING = "pending"
    DELIVERED = "delivered"
    RETRYABLE = "retryable"
    FAILED = "failed"


class TriggerKind(StrEnum):
    WEBHOOK = "webhook"
    JOB_COMPLETION = "job_completion"
    GOAL_COMPLETION = "goal_completion"
    CHILD_COMPLETION = "child_completion"


class TriggerTrust(StrEnum):
    OWNER = "owner"
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True, slots=True)
class RecurrenceSpec:
    """Normalized schedule independent from persistence and wall-clock access."""

    kind: RecurrenceKind = RecurrenceKind.ONCE
    timezone_name: str = "UTC"
    interval_seconds: int | None = None
    cron: str | None = None
    rrule: str | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    misfire_policy: MisfirePolicy = MisfirePolicy.RUN_ONCE
    nonexistent_time_policy: NonexistentTimePolicy = NonexistentTimePolicy.SHIFT_FORWARD
    ambiguous_time_policy: AmbiguousTimePolicy = AmbiguousTimePolicy.EARLIEST
    misfire_grace_seconds: int = 60
    max_catch_up: int = 1
    jitter_seconds: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RecurrenceKind(self.kind))
        object.__setattr__(self, "misfire_policy", MisfirePolicy(self.misfire_policy))
        object.__setattr__(
            self,
            "nonexistent_time_policy",
            NonexistentTimePolicy(self.nonexistent_time_policy),
        )
        object.__setattr__(
            self,
            "ambiguous_time_policy",
            AmbiguousTimePolicy(self.ambiguous_time_policy),
        )
        timezone_name = self.timezone_name.strip()
        if not timezone_name:
            raise ValueError("recurrence timezone cannot be empty")
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown recurrence timezone: {timezone_name}") from exc
        object.__setattr__(self, "timezone_name", timezone_name)
        for field_name in ("starts_at", "ends_at"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _utc(value, field_name))
        if (
            self.starts_at is not None
            and self.ends_at is not None
            and self.ends_at <= self.starts_at
        ):
            raise ValueError("recurrence ends_at must be after starts_at")
        if (
            isinstance(self.misfire_grace_seconds, bool)
            or not isinstance(self.misfire_grace_seconds, int)
            or self.misfire_grace_seconds < 0
            or self.misfire_grace_seconds > 86_400
        ):
            raise ValueError("misfire_grace_seconds must be between zero and 86400")
        if (
            isinstance(self.max_catch_up, bool)
            or not isinstance(self.max_catch_up, int)
            or self.max_catch_up < 1
            or self.max_catch_up > 100
        ):
            raise ValueError("max_catch_up must be between 1 and 100")
        if (
            isinstance(self.jitter_seconds, bool)
            or not isinstance(self.jitter_seconds, int)
            or self.jitter_seconds < 0
            or self.jitter_seconds > 86_400
        ):
            raise ValueError("jitter_seconds must be between zero and 86400")
        expected = {
            RecurrenceKind.ONCE: (None, None, None),
            RecurrenceKind.INTERVAL: (self.interval_seconds, None, None),
            RecurrenceKind.CRON: (None, self.cron, None),
            RecurrenceKind.RRULE: (None, None, self.rrule),
        }[self.kind]
        if self.kind is RecurrenceKind.INTERVAL:
            if (
                isinstance(self.interval_seconds, bool)
                or not isinstance(self.interval_seconds, int)
                or self.interval_seconds < 1
            ):
                raise ValueError(
                    "interval recurrence requires positive interval_seconds"
                )
        elif self.interval_seconds is not None:
            raise ValueError("interval_seconds is only valid for interval recurrence")
        if self.kind is RecurrenceKind.CRON:
            object.__setattr__(self, "cron", _required(self.cron, "cron expression"))
        elif self.cron is not None:
            raise ValueError("cron is only valid for cron recurrence")
        if self.kind is RecurrenceKind.RRULE:
            object.__setattr__(self, "rrule", _required(self.rrule, "RRULE"))
        elif self.rrule is not None:
            raise ValueError("rrule is only valid for RRULE recurrence")
        del expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "timezone": self.timezone_name,
            "interval_seconds": self.interval_seconds,
            "cron": self.cron,
            "rrule": self.rrule,
            "starts_at": self.starts_at.isoformat() if self.starts_at else None,
            "ends_at": self.ends_at.isoformat() if self.ends_at else None,
            "misfire_policy": self.misfire_policy.value,
            "nonexistent_time_policy": self.nonexistent_time_policy.value,
            "ambiguous_time_policy": self.ambiguous_time_policy.value,
            "misfire_grace_seconds": self.misfire_grace_seconds,
            "max_catch_up": self.max_catch_up,
            "jitter_seconds": self.jitter_seconds,
        }


@dataclass(frozen=True, slots=True)
class AutomationRetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: int = 60
    max_backoff_seconds: int = 3_600
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        for name in ("max_attempts", "initial_backoff_seconds", "max_backoff_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_attempts > 100:
            raise ValueError("max_attempts cannot exceed 100")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds cannot be below initial backoff")
        if self.multiplier < 1:
            raise ValueError("retry multiplier must be at least one")

    def delay_for(self, attempt: int) -> int:
        if attempt < 1:
            raise ValueError("attempt must be positive")
        return min(
            self.max_backoff_seconds,
            int(self.initial_backoff_seconds * self.multiplier ** (attempt - 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "max_backoff_seconds": self.max_backoff_seconds,
            "multiplier": self.multiplier,
        }


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    """One durable automation definition; legacy scheduling fields stay ergonomic."""

    id: str
    profile_id: str
    target: DeliveryTarget
    prompt: str
    recurrence: RecurrenceSpec
    next_run_at: datetime
    status: AutomationJobStatus
    scheduled_for: datetime
    budget: RunBudget
    retry_policy: AutomationRetryPolicy
    revision: int = 0
    run_count: int = 0
    max_runs: int | None = None
    requires_approval: bool = False
    approved_at: datetime | None = None
    run_now_requested_at: datetime | None = None
    claim_token: str | None = None
    lease_until: datetime | None = None
    active_run_id: str | None = None
    last_run_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "job id"))
        object.__setattr__(self, "profile_id", _required(self.profile_id, "profile id"))
        prompt = _required(self.prompt, "prompt")
        if len(prompt) > MAX_SCHEDULED_PROMPT_CHARS:
            raise ValueError(
                f"prompt exceeds the {MAX_SCHEDULED_PROMPT_CHARS}-character limit"
            )
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "status", AutomationJobStatus(self.status))
        for field_name in (
            "next_run_at",
            "scheduled_for",
            "approved_at",
            "run_now_requested_at",
            "lease_until",
            "last_run_at",
            "created_at",
            "updated_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _utc(value, field_name))
        for field_name in ("revision", "run_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.max_runs is not None and (
            isinstance(self.max_runs, bool)
            or not isinstance(self.max_runs, int)
            or self.max_runs < 1
        ):
            raise ValueError("max_runs must be a positive integer")
        if self.budget.scope is not BudgetScope.JOB:
            raise ValueError("automation budget scope must be job")
        claim_values = (self.claim_token, self.lease_until, self.active_run_id)
        if any(value is not None for value in claim_values) and not all(
            value is not None for value in claim_values
        ):
            raise ValueError(
                "job claim token, lease, and active run must be set together"
            )

    @property
    def adapter(self) -> str:
        return self.target.adapter

    @property
    def destination_id(self) -> str:
        return self.target.destination_id

    @property
    def interval_seconds(self) -> int | None:
        return self.recurrence.interval_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "target": {
                "adapter": self.target.adapter,
                "account_id": self.target.account_id,
                "destination_id": self.target.destination_id,
                "thread_id": self.target.thread_id,
            },
            "prompt": self.prompt,
            "recurrence": self.recurrence.to_dict(),
            "next_run_at": self.next_run_at.isoformat(),
            "scheduled_for": self.scheduled_for.isoformat(),
            "status": self.status.value,
            "budget": self.budget.to_dict(),
            "retry_policy": self.retry_policy.to_dict(),
            "revision": self.revision,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "requires_approval": self.requires_approval,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "run_now_requested_at": (
                self.run_now_requested_at.isoformat()
                if self.run_now_requested_at
                else None
            ),
            "lease_until": self.lease_until.isoformat() if self.lease_until else None,
            "active_run_id": self.active_run_id,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class AutomationRun:
    id: str
    job_id: str
    profile_id: str
    occurrence_at: datetime
    reason: AutomationRunReason
    status: AutomationRunStatus
    attempt: int
    claim_token: str | None = None
    worker_id: str | None = None
    lease_until: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Mapping[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    artifact_refs: tuple[str, ...] = ()
    delivery_state: AutomationDeliveryState = AutomationDeliveryState.NONE
    delivery_error: str | None = None
    trigger_event_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        for field_name in ("id", "job_id", "profile_id"):
            object.__setattr__(
                self, field_name, _required(getattr(self, field_name), field_name)
            )
        object.__setattr__(self, "reason", AutomationRunReason(self.reason))
        object.__setattr__(self, "status", AutomationRunStatus(self.status))
        object.__setattr__(
            self,
            "delivery_state",
            AutomationDeliveryState(self.delivery_state),
        )
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("automation run attempt must be positive")
        for field_name in (
            "occurrence_at",
            "lease_until",
            "started_at",
            "finished_at",
            "created_at",
            "updated_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _utc(value, field_name))
        object.__setattr__(self, "result", _freeze(self.result))
        object.__setattr__(self, "usage", _freeze(self.usage))
        object.__setattr__(self, "cost", _freeze(self.cost))
        object.__setattr__(
            self,
            "artifact_refs",
            tuple(
                dict.fromkeys(
                    _required(item, "artifact ref") for item in self.artifact_refs
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "profile_id": self.profile_id,
            "occurrence_at": self.occurrence_at.isoformat(),
            "reason": self.reason.value,
            "status": self.status.value,
            "attempt": self.attempt,
            "worker_id": self.worker_id,
            "lease_until": self.lease_until.isoformat() if self.lease_until else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
            "result": _plain(self.result),
            "trace_id": self.trace_id,
            "usage": _plain(self.usage),
            "cost": _plain(self.cost),
            "error": self.error,
            "artifact_refs": list(self.artifact_refs),
            "delivery_state": self.delivery_state.value,
            "delivery_error": self.delivery_error,
            "trigger_event_id": self.trigger_event_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @property
    def duration_ms(self) -> int | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return max(0, int((self.finished_at - self.started_at).total_seconds() * 1_000))


@dataclass(frozen=True, slots=True)
class AutomationDeliveryAttempt:
    id: str
    run_id: str
    profile_id: str
    state: AutomationDeliveryState
    created_at: datetime
    error: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("id", "run_id", "profile_id"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "state", AutomationDeliveryState(self.state))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "state": self.state.value,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class AutomationTrigger:
    id: str
    profile_id: str
    job_id: str
    kind: TriggerKind
    source_resource_id: str | None = None
    secret_digest: str | None = None
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        for field_name in ("id", "profile_id", "job_id"):
            object.__setattr__(
                self, field_name, _required(getattr(self, field_name), field_name)
            )
        object.__setattr__(self, "kind", TriggerKind(self.kind))
        if self.kind is TriggerKind.WEBHOOK and not self.secret_digest:
            raise ValueError("webhook trigger requires a secret digest")
        if self.kind is not TriggerKind.WEBHOOK and not self.source_resource_id:
            raise ValueError("completion trigger requires source_resource_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "job_id": self.job_id,
            "kind": self.kind.value,
            "source_resource_id": self.source_resource_id,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TriggerEnvelope:
    id: str
    profile_id: str
    trigger_id: str
    trust: TriggerTrust
    payload: Mapping[str, Any]
    occurred_at: datetime
    source_event_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("id", "profile_id", "trigger_id"):
            object.__setattr__(
                self, field_name, _required(getattr(self, field_name), field_name)
            )
        object.__setattr__(self, "trust", TriggerTrust(self.trust))
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at, "occurred_at"))
        frozen = _freeze(redact_data(dict(self.payload)))
        if len(repr(_plain(frozen)).encode("utf-8")) > MAX_TRIGGER_PAYLOAD_BYTES:
            raise ValueError("trigger payload exceeds the 65536-byte limit")
        object.__setattr__(self, "payload", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "trigger_id": self.trigger_id,
            "trust": self.trust.value,
            "payload": _plain(self.payload),
            "occurred_at": self.occurred_at.isoformat(),
            "source_event_id": self.source_event_id,
        }


@dataclass(frozen=True, slots=True)
class AutomationJobEvent:
    id: str
    job_id: str
    profile_id: str
    action: str
    revision: int
    actor: str
    created_at: datetime
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("id", "job_id", "profile_id", "action", "actor"):
            object.__setattr__(
                self, field_name, _required(getattr(self, field_name), field_name)
            )
        if self.revision < 0:
            raise ValueError("event revision cannot be negative")
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _freeze(self.metadata))


def _required(value: str | None, label: str) -> str:
    clean = value.strip() if value is not None else ""
    if not clean:
        raise ValueError(f"{label} cannot be empty")
    return clean


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in value.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "AutomationDeliveryState",
    "AutomationDeliveryAttempt",
    "AutomationJobEvent",
    "AutomationJobStatus",
    "AutomationRetryPolicy",
    "AutomationRun",
    "AutomationRunReason",
    "AutomationRunStatus",
    "AutomationTrigger",
    "AmbiguousTimePolicy",
    "MAX_SCHEDULED_PROMPT_CHARS",
    "MAX_TRIGGER_PAYLOAD_BYTES",
    "MisfirePolicy",
    "NonexistentTimePolicy",
    "RecurrenceKind",
    "RecurrenceSpec",
    "ScheduledJob",
    "TriggerEnvelope",
    "TriggerKind",
    "TriggerTrust",
]
