"""Immutable models for governed skill lifecycle and learning proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any


class LearningProposalKind(StrEnum):
    MEMORY_CREATE = "memory_create"
    MEMORY_UPDATE = "memory_update"
    SKILL_CREATE = "skill_create"
    SKILL_PATCH = "skill_patch"
    SKILL_ARCHIVE = "skill_archive"


class LearningProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class SkillLifecycleStatus(StrEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    EVALUATING = "evaluating"
    AWAITING_REVIEW = "awaiting_review"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"
    PINNED = "pinned"


class SkillUsageKind(StrEnum):
    VIEW = "view"
    USE = "use"
    SUCCESS = "success"
    PATCH = "patch"


@dataclass(frozen=True, slots=True)
class LearningReviewUsage:
    """Daily reviewer usage reserved or consumed for quota enforcement."""

    proposal_count: int = 0
    token_count: int = 0
    cost_amount: Decimal = Decimal(0)


@dataclass(frozen=True, slots=True)
class LearningProposalRecord:
    """One reviewable memory or skill change with bounded evidence."""

    id: str
    profile_id: str
    kind: LearningProposalKind
    target_name: str | None
    rationale: str
    evidence_turn_ids: tuple[str, ...]
    source_trace: str | None
    content: str | None
    diff: str | None
    required_capabilities: tuple[str, ...]
    confidence: float
    verification_steps: tuple[str, ...]
    reviewer_model: str | None
    cost: str | None
    status: LearningProposalStatus
    created_at: str
    reviewed_at: str | None = None
    reviewed_by: str | None = None
    applied_revision_id: str | None = None
    accepted_memory_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "kind": self.kind.value,
            "target_name": self.target_name,
            "rationale": self.rationale,
            "evidence_turn_ids": list(self.evidence_turn_ids),
            "source_trace": self.source_trace,
            "content": self.content,
            "diff": self.diff,
            "required_capabilities": list(self.required_capabilities),
            "confidence": self.confidence,
            "verification_steps": list(self.verification_steps),
            "reviewer_model": self.reviewer_model,
            "cost": self.cost,
            "status": self.status.value,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "reviewed_by": self.reviewed_by,
            "applied_revision_id": self.applied_revision_id,
            "accepted_memory_id": self.accepted_memory_id,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SkillRevisionRecord:
    """Immutable snapshot of one validated skill package version."""

    id: str
    profile_id: str
    scope: str
    name: str
    version: str
    digest: str
    source: str
    trust: str
    package_files: dict[str, bytes]
    manifest: dict[str, Any]
    created_at: str
    proposal_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "scope": self.scope,
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "source": self.source,
            "trust": self.trust,
            "manifest": dict(self.manifest),
            "created_at": self.created_at,
            "proposal_id": self.proposal_id,
        }


@dataclass(frozen=True, slots=True)
class SkillLifecycleRecord:
    """Current governed state and counters for one profile-owned skill."""

    profile_id: str
    scope: str
    name: str
    version: str
    digest: str
    source: str
    trust: str
    status: SkillLifecycleStatus
    active_revision_id: str
    view_count: int
    use_count: int
    success_count: int
    patch_count: int
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "scope": self.scope,
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "source": self.source,
            "trust": self.trust,
            "status": self.status.value,
            "active_revision_id": self.active_revision_id,
            "view_count": self.view_count,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "patch_count": self.patch_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


__all__ = [
    "LearningProposalKind",
    "LearningProposalRecord",
    "LearningProposalStatus",
    "LearningReviewUsage",
    "SkillLifecycleRecord",
    "SkillLifecycleStatus",
    "SkillRevisionRecord",
    "SkillUsageKind",
]
