"""Immutable contracts for restart-safe hosted approvals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping

from chulk.core.signals import DurableApprovalPaused
from chulk.hosting.scope import ExecutionScope
from chulk.redaction import redact_data
from chulk.results import freeze_mapping, plain_data
from chulk.runs.models import RunRecord


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"


class ApprovalOutcomeKind(StrEnum):
    PAUSED = "paused"
    RESUMED = "resumed"
    DENIED = "denied"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    CANCELLED = "cancelled"
    REVOKED_AUTHORITY = "revoked_authority"
    UNAVAILABLE_INTEGRATION = "unavailable_integration"


@dataclass(frozen=True, slots=True)
class ApprovalSubmission:
    step_id: str
    tool_name: str
    tool_version: str
    schema_version: str
    arguments_digest: str
    policy_version: str
    preview: Mapping[str, Any]
    expires_at: datetime
    effect_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "step_id",
            "tool_name",
            "tool_version",
            "schema_version",
            "arguments_digest",
            "policy_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "effect_id", _optional(self.effect_id))
        object.__setattr__(self, "preview", _safe_mapping(self.preview))
        object.__setattr__(
            self,
            "expires_at",
            _utc(self.expires_at, "expires_at"),
        )


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    id: str
    scope: ExecutionScope
    run_id: str
    step_id: str
    tool_name: str
    tool_version: str
    schema_version: str
    arguments_digest: str
    policy_version: str
    preview: Mapping[str, Any]
    status: ApprovalStatus
    revision: int
    created_at: datetime
    expires_at: datetime
    effect_id: str | None = None
    decision: ApprovalDecision | None = None
    decided_by: str | None = None
    decision_reason: str | None = None
    decided_at: datetime | None = None
    consumed_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "id",
            "run_id",
            "step_id",
            "tool_name",
            "tool_version",
            "schema_version",
            "arguments_digest",
            "policy_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )
        if self.run_id != self.scope.run_id:
            raise ValueError("approval run_id must match execution scope")
        object.__setattr__(self, "effect_id", _optional(self.effect_id))
        object.__setattr__(self, "preview", _safe_mapping(self.preview))
        object.__setattr__(self, "status", ApprovalStatus(self.status))
        if isinstance(self.revision, bool) or self.revision < 0:
            raise ValueError("approval revision must be non-negative")
        if self.decision is not None:
            object.__setattr__(
                self,
                "decision",
                ApprovalDecision(self.decision),
            )
        object.__setattr__(self, "decided_by", _optional(self.decided_by))
        object.__setattr__(
            self,
            "decision_reason",
            _optional(self.decision_reason),
        )
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        object.__setattr__(self, "expires_at", _utc(self.expires_at, "expires_at"))
        object.__setattr__(self, "decided_at", _utc_optional(self.decided_at))
        object.__setattr__(self, "consumed_at", _utc_optional(self.consumed_at))
        object.__setattr__(
            self,
            "updated_at",
            _utc_optional(self.updated_at) or self.created_at,
        )

    @property
    def terminal(self) -> bool:
        return self.status in {
            ApprovalStatus.DENIED,
            ApprovalStatus.EXPIRED,
            ApprovalStatus.CONSUMED,
            ApprovalStatus.INVALIDATED,
            ApprovalStatus.CANCELLED,
        }

    def to_dict(self) -> dict[str, Any]:
        updated_at = self.updated_at or self.created_at
        return {
            "id": self.id,
            "scope": self.scope.to_dict(),
            "run_id": self.run_id,
            "step_id": self.step_id,
            "effect_id": self.effect_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "schema_version": self.schema_version,
            "arguments_digest": self.arguments_digest,
            "policy_version": self.policy_version,
            "preview": plain_data(self.preview),
            "status": self.status.value,
            "revision": self.revision,
            "decision": self.decision.value if self.decision is not None else None,
            "decided_by": self.decided_by,
            "decision_reason": self.decision_reason,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "decided_at": _iso(self.decided_at),
            "consumed_at": _iso(self.consumed_at),
            "updated_at": updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ApprovalValidation:
    """Fresh host facts that must still match before an approval is consumed."""

    scope: ExecutionScope
    tool_name: str
    tool_version: str
    schema_version: str
    arguments_digest: str
    policy_version: str
    authority_valid: bool = True
    credentials_available: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "tool_name",
            "tool_version",
            "schema_version",
            "arguments_digest",
            "policy_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, slots=True)
class PausedRunOutcome:
    kind: ApprovalOutcomeKind
    run: RunRecord
    approval: ApprovalRequest
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ApprovalOutcomeKind(self.kind))
        object.__setattr__(self, "reason", _required(self.reason, "outcome reason"))


@dataclass(frozen=True, slots=True)
class ApprovalResumeOutcome:
    kind: ApprovalOutcomeKind
    approval: ApprovalRequest
    run: RunRecord
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ApprovalOutcomeKind(self.kind))
        object.__setattr__(self, "reason", _required(self.reason, "outcome reason"))

    @property
    def resumed(self) -> bool:
        return self.kind is ApprovalOutcomeKind.RESUMED


def _required(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


def _optional(value: object) -> str | None:
    if value is None:
        return None
    return _required(value, "value")


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
        raise ValueError("approval preview must remain an object after redaction")
    return freeze_mapping(redacted)


__all__ = [
    "ApprovalDecision",
    "ApprovalOutcomeKind",
    "ApprovalRequest",
    "ApprovalResumeOutcome",
    "ApprovalStatus",
    "ApprovalSubmission",
    "ApprovalValidation",
    "DurableApprovalPaused",
    "PausedRunOutcome",
]
