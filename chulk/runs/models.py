"""Typed contracts for durable hosted execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Mapping

from chulk.hosting.scope import ExecutionScope
from chulk.redaction import redact_data
from chulk.results import freeze_mapping, plain_data


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_RETRY = "waiting_for_retry"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    DEAD_LETTER = "dead_letter"


class StepStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_RETRY = "waiting_for_retry"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    DEAD_LETTER = "dead_letter"


class AttemptStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class EffectStatus(StrEnum):
    INTENDED = "intended"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ReconciliationDecision(StrEnum):
    CONFIRMED = "confirmed"
    RETRY = "retry"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.DEAD_LETTER,
    }
)
TERMINAL_STEP_STATUSES = frozenset(
    {
        StepStatus.COMPLETED,
        StepStatus.FAILED,
        StepStatus.CANCELLED,
        StepStatus.DEAD_LETTER,
    }
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Deterministic retry policy evaluated only at durable boundaries."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        for name in ("initial_delay_seconds", "max_delay_seconds", "multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError(
                "max_delay_seconds cannot be less than initial_delay_seconds"
            )
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least one")

    def delay_for_attempt(self, attempt: int) -> timedelta:
        if attempt < 1:
            raise ValueError("attempt must be positive")
        seconds = min(
            self.max_delay_seconds,
            self.initial_delay_seconds * (self.multiplier ** (attempt - 1)),
        )
        return timedelta(seconds=seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "initial_delay_seconds": self.initial_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "multiplier": self.multiplier,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetryPolicy":
        return cls(
            max_attempts=int(value.get("max_attempts", 3)),
            initial_delay_seconds=float(
                value.get("initial_delay_seconds", 1.0)
            ),
            max_delay_seconds=float(value.get("max_delay_seconds", 60.0)),
            multiplier=float(value.get("multiplier", 2.0)),
        )


@dataclass(frozen=True, slots=True)
class StepDefinition:
    """One immutable step declared when a durable run is submitted."""

    id: str
    name: str
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "step id"))
        object.__setattr__(self, "name", _required(self.name, "step name"))
        object.__setattr__(self, "metadata", _safe_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "retry_policy": self.retry_policy.to_dict(),
            "metadata": plain_data(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StepDefinition":
        retry = value.get("retry_policy", {})
        metadata = value.get("metadata", {})
        if not isinstance(retry, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("step retry_policy and metadata must be objects")
        return cls(
            id=str(value.get("id") or ""),
            name=str(value.get("name") or ""),
            retry_policy=RetryPolicy.from_dict(retry),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class RunSubmission:
    """Idempotent input used to create one durable run."""

    idempotency_key: str
    input_digest: str
    definition_digest: str
    steps: tuple[StepDefinition, ...]
    budget: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_event_id: str | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "idempotency_key",
            _required(self.idempotency_key, "idempotency key"),
        )
        object.__setattr__(
            self,
            "input_digest",
            _required(self.input_digest, "input digest"),
        )
        object.__setattr__(
            self,
            "definition_digest",
            _required(self.definition_digest, "definition digest"),
        )
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("run submission requires at least one step")
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("run step ids must be unique")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "budget", _safe_mapping(self.budget))
        object.__setattr__(self, "metadata", _safe_mapping(self.metadata))
        object.__setattr__(
            self,
            "source_event_id",
            _optional(self.source_event_id),
        )
        object.__setattr__(
            self,
            "correlation_id",
            _optional(self.correlation_id),
        )


@dataclass(frozen=True, slots=True)
class StepRecord:
    id: str
    run_id: str
    name: str
    status: StepStatus
    revision: int
    attempt_count: int
    retry_policy: RetryPolicy
    next_retry_at: datetime | None = None
    last_checkpoint_id: str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "step id"))
        object.__setattr__(self, "run_id", _required(self.run_id, "run id"))
        object.__setattr__(self, "name", _required(self.name, "step name"))
        object.__setattr__(self, "status", StepStatus(self.status))
        _revision(self.revision, "step revision")
        _revision(self.attempt_count, "step attempt_count")
        object.__setattr__(self, "next_retry_at", _utc_optional(self.next_retry_at))
        object.__setattr__(
            self,
            "last_checkpoint_id",
            _optional(self.last_checkpoint_id),
        )
        object.__setattr__(self, "error", _optional(self.error))
        object.__setattr__(self, "metadata", _safe_mapping(self.metadata))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        object.__setattr__(self, "started_at", _utc_optional(self.started_at))
        object.__setattr__(self, "completed_at", _utc_optional(self.completed_at))

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STEP_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status.value,
            "revision": self.revision,
            "attempt_count": self.attempt_count,
            "retry_policy": self.retry_policy.to_dict(),
            "next_retry_at": _iso(self.next_retry_at),
            "last_checkpoint_id": self.last_checkpoint_id,
            "error": self.error,
            "metadata": plain_data(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
        }


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    scope: ExecutionScope
    idempotency_key: str
    input_digest: str
    definition_digest: str
    status: RunStatus
    revision: int
    steps: tuple[StepRecord, ...]
    cancellation_requested: bool = False
    waiting_reason: str | None = None
    next_retry_at: datetime | None = None
    budget: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required(self.id, "run id"))
        if self.id != self.scope.run_id:
            raise ValueError("run id must match execution scope run_id")
        object.__setattr__(
            self,
            "idempotency_key",
            _required(self.idempotency_key, "idempotency key"),
        )
        object.__setattr__(
            self,
            "input_digest",
            _required(self.input_digest, "input digest"),
        )
        object.__setattr__(
            self,
            "definition_digest",
            _required(self.definition_digest, "definition digest"),
        )
        object.__setattr__(self, "status", RunStatus(self.status))
        _revision(self.revision, "run revision")
        steps = tuple(self.steps)
        if not steps or any(step.run_id != self.id for step in steps):
            raise ValueError("run steps must be non-empty and owned by the run")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "waiting_reason", _optional(self.waiting_reason))
        object.__setattr__(self, "next_retry_at", _utc_optional(self.next_retry_at))
        object.__setattr__(self, "budget", _safe_mapping(self.budget))
        object.__setattr__(self, "metadata", _safe_mapping(self.metadata))
        object.__setattr__(
            self,
            "result",
            _safe_mapping(self.result) if self.result is not None else None,
        )
        object.__setattr__(self, "error", _optional(self.error))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))
        object.__setattr__(self, "completed_at", _utc_optional(self.completed_at))

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES

    def step(self, step_id: str) -> StepRecord:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"run step {step_id!r} does not exist")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope.to_dict(),
            "idempotency_key": self.idempotency_key,
            "input_digest": self.input_digest,
            "definition_digest": self.definition_digest,
            "status": self.status.value,
            "revision": self.revision,
            "steps": [step.to_dict() for step in self.steps],
            "cancellation_requested": self.cancellation_requested,
            "waiting_reason": self.waiting_reason,
            "next_retry_at": _iso(self.next_retry_at),
            "budget": plain_data(self.budget),
            "metadata": plain_data(self.metadata),
            "result": plain_data(self.result) if self.result is not None else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": _iso(self.completed_at),
        }


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    id: str
    run_id: str
    step_id: str
    number: int
    status: AttemptStatus
    worker_id: str
    lease_token: str
    started_at: datetime
    completed_at: datetime | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("id", "run_id", "step_id", "worker_id", "lease_token"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )
        if self.number < 1:
            raise ValueError("attempt number must be positive")
        object.__setattr__(self, "status", AttemptStatus(self.status))
        object.__setattr__(self, "started_at", _utc(self.started_at, "started_at"))
        object.__setattr__(self, "completed_at", _utc_optional(self.completed_at))
        object.__setattr__(self, "error", _optional(self.error))


@dataclass(frozen=True, slots=True)
class Checkpoint:
    id: str
    run_id: str
    step_id: str
    attempt_id: str
    sequence: int
    kind: str
    payload: Mapping[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        for field_name in ("id", "run_id", "step_id", "attempt_id", "kind"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )
        if self.sequence < 1:
            raise ValueError("checkpoint sequence must be positive")
        object.__setattr__(self, "payload", _safe_mapping(self.payload))
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))


@dataclass(frozen=True, slots=True)
class EffectRecord:
    id: str
    run_id: str
    step_id: str
    attempt_id: str
    logical_key: str
    tool_name: str
    tool_version: str
    schema_version: str
    arguments_digest: str
    status: EffectStatus
    result_digest: str | None = None
    reconciliation: ReconciliationDecision | None = None
    reconciled_by: str | None = None
    reconciliation_reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        for field_name in (
            "id",
            "run_id",
            "step_id",
            "attempt_id",
            "logical_key",
            "tool_name",
            "tool_version",
            "schema_version",
            "arguments_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "status", EffectStatus(self.status))
        object.__setattr__(self, "result_digest", _optional(self.result_digest))
        if self.reconciliation is not None:
            object.__setattr__(
                self,
                "reconciliation",
                ReconciliationDecision(self.reconciliation),
            )
        object.__setattr__(self, "reconciled_by", _optional(self.reconciled_by))
        object.__setattr__(
            self,
            "reconciliation_reason",
            _optional(self.reconciliation_reason),
        )
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _utc(self.updated_at, "updated_at"))


@dataclass(frozen=True, slots=True)
class RunEvent:
    id: str
    run_id: str
    sequence: int
    name: str
    actor: str
    payload: Mapping[str, Any]
    step_id: str | None = None
    causation_id: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        for field_name in ("id", "run_id", "name", "actor"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )
        if self.sequence < 1:
            raise ValueError("run event sequence must be positive")
        object.__setattr__(self, "payload", _safe_mapping(self.payload))
        object.__setattr__(self, "step_id", _optional(self.step_id))
        object.__setattr__(self, "causation_id", _optional(self.causation_id))
        object.__setattr__(self, "correlation_id", _optional(self.correlation_id))
        object.__setattr__(
            self,
            "idempotency_key",
            _optional(self.idempotency_key),
        )
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))


@dataclass(frozen=True, slots=True)
class RunClaim:
    run_id: str
    scope_key: str
    worker_id: str
    lease_token: str
    lease_until: datetime
    revision: int

    def __post_init__(self) -> None:
        for field_name in ("run_id", "scope_key", "worker_id", "lease_token"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "lease_until",
            _utc(self.lease_until, "lease_until"),
        )
        _revision(self.revision, "claim revision")


@dataclass(frozen=True, slots=True)
class ReconciliationRecord:
    effect: EffectRecord
    run: RunRecord
    decision: ReconciliationDecision


def _required(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    clean = value.strip()
    if len(clean) > 1_000 or "\x00" in clean:
        raise ValueError(f"{name} is invalid")
    return clean


def _optional(value: object) -> str | None:
    if value is None:
        return None
    return _required(value, "value")


def _revision(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc)


def _utc_optional(value: datetime | None) -> datetime | None:
    return _utc(value, "datetime") if value is not None else None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _safe_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    redacted = redact_data(dict(value))
    if not isinstance(redacted, dict):
        raise ValueError("mapping payload must remain an object after redaction")
    return freeze_mapping(redacted)


__all__ = [
    "AttemptRecord",
    "AttemptStatus",
    "Checkpoint",
    "EffectRecord",
    "EffectStatus",
    "ReconciliationDecision",
    "ReconciliationRecord",
    "RetryPolicy",
    "RunClaim",
    "RunEvent",
    "RunRecord",
    "RunStatus",
    "RunSubmission",
    "StepDefinition",
    "StepRecord",
    "StepStatus",
    "TERMINAL_RUN_STATUSES",
    "TERMINAL_STEP_STATUSES",
]
