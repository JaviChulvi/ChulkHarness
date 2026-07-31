"""Skill registry primitives and public skill references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chulk._lazy import public_dir, resolve_export

if TYPE_CHECKING:
    from chulk.skills.manifest import (
        LEGACY_SKILL_VERSION,
        SKILL_MANIFEST_SCHEMA_VERSION,
        SkillManifest,
        SkillManifestError,
        SkillPackage,
        load_skill_package,
        resolve_skill_resource,
        skill_package_digest,
    )
    from chulk.skills.lifecycle_models import (
        LearningProposalKind,
        LearningProposalRecord,
        LearningProposalStatus,
        LearningReviewUsage,
        SkillLifecycleRecord,
        SkillLifecycleStatus,
        SkillRevisionRecord,
        SkillUsageKind,
    )
    from chulk.skills.lifecycle import (
        SkillApprovalError,
        SkillConflictError,
        SkillLifecycleError,
        SkillLifecycleManager,
        SkillScope,
        proposal_diff,
    )
    from chulk.skills.lifecycle_store import SQLiteSkillLifecycleStore
    from chulk.skills.locks import (
        SKILL_LOCK_SCHEMA_VERSION,
        SkillLockEntry,
        SkillLockFile,
    )
    from chulk.skills.proposals import (
        AutomaticLearningBlocked,
        LearningProposalDraft,
        LearningProposalService,
    )
    from chulk.skills.reviewer import (
        LearningReviewContext,
        LearningReviewCoordinator,
        LearningReviewError,
        LearningReviewOutcome,
        LearningReviewPolicy,
        LearningReviewQuota,
        LearningReviewQuotaExceeded,
        LearningReviewResult,
        LearningReviewTrigger,
        RestrictedLearningReviewer,
    )
    from chulk.skills.registry import (
        Skill,
        SkillRegistry,
        SkillRouteDecision,
        SkillReranker,
        SkillRoutingResult,
        SkillSelection,
        explicit_skill_names,
    )
    from chulk.skills.publication import (
        AsyncInMemorySkillPublicationStore,
        AsyncSkillPublicationManager,
        AsyncSkillPublicationStore,
        InMemorySkillPublicationStore,
        PortableSkill,
        SkillActivationRecord,
        SkillPublicationManager,
        SkillPublicationRecord,
        SkillPublicationStore,
        skill_reference_map,
    )


def bundled_skills_dir() -> Path:
    """Return the installed directory containing Chulk's bundled skill playbooks."""
    return Path(__file__).resolve().parent / "bundled"


@dataclass(frozen=True)
class SkillRef:
    """Reference to a skill that should be available and pinned for an agent."""

    name: str | None = None
    skill_path: Path | None = None

    def register(self, registry: SkillRegistry) -> str | None:
        if self.skill_path is not None:
            skill = registry.register_path(self.skill_path)
            return skill.name
        if self.name is None:
            raise ValueError("SkillRef must include name or skill_path")
        if registry.get_skill(self.name) is None:
            return None
        return self.name


@dataclass(frozen=True)
class SkillDirectoryRef:
    """Reference to an additional directory of selectable skill playbooks."""

    skills_dir: Path

    def register(self, registry: SkillRegistry) -> str | None:
        registry.register_directory(self.skills_dir)
        return None


@dataclass(frozen=True)
class SkillAllowlistRef:
    """Names of catalog skills that may be selected for one agent."""

    names: tuple[str, ...]


@dataclass(frozen=True)
class SkillPinRef:
    """Names of catalog skills that should always be loaded for one agent."""

    names: tuple[str, ...]


def path(skill_path: str | Path) -> SkillRef:
    """Pin one skill from a SKILL.md path or a directory containing SKILL.md."""
    return SkillRef(skill_path=Path(skill_path))


def from_dir(skills_dir: str | Path) -> SkillDirectoryRef:
    """Register an additional directory of skill folders for selection."""
    return SkillDirectoryRef(Path(skills_dir))


def only(*names: str) -> SkillAllowlistRef:
    """Allow automatic selection only from these catalog skill names."""
    return SkillAllowlistRef(tuple(names))


def pin(*names: str) -> SkillPinRef:
    """Always load these catalog skill names for the agent."""
    return SkillPinRef(tuple(names))


files = SkillRef(name="files")
shell = SkillRef(name="shell")
memory = SkillRef(name="memory")


__all__ = [
    "Skill",
    "SkillAllowlistRef",
    "SkillDirectoryRef",
    "SkillPinRef",
    "SkillRef",
    "SkillRegistry",
    "SkillRouteDecision",
    "SkillReranker",
    "SkillRoutingResult",
    "SkillSelection",
    "SkillManifest",
    "SkillManifestError",
    "SkillPackage",
    "LEGACY_SKILL_VERSION",
    "AutomaticLearningBlocked",
    "LearningProposalKind",
    "LearningProposalRecord",
    "LearningProposalStatus",
    "LearningProposalDraft",
    "LearningProposalService",
    "LearningReviewContext",
    "LearningReviewCoordinator",
    "LearningReviewError",
    "LearningReviewOutcome",
    "LearningReviewPolicy",
    "LearningReviewQuota",
    "LearningReviewQuotaExceeded",
    "LearningReviewResult",
    "LearningReviewTrigger",
    "LearningReviewUsage",
    "SKILL_MANIFEST_SCHEMA_VERSION",
    "SQLiteSkillLifecycleStore",
    "SKILL_LOCK_SCHEMA_VERSION",
    "SkillApprovalError",
    "SkillConflictError",
    "SkillLifecycleError",
    "SkillLifecycleManager",
    "SkillLifecycleRecord",
    "SkillLifecycleStatus",
    "SkillLockEntry",
    "SkillLockFile",
    "SkillRevisionRecord",
    "SkillScope",
    "SkillUsageKind",
    "AsyncInMemorySkillPublicationStore",
    "AsyncSkillPublicationManager",
    "AsyncSkillPublicationStore",
    "InMemorySkillPublicationStore",
    "PortableSkill",
    "SkillActivationRecord",
    "SkillPublicationManager",
    "SkillPublicationRecord",
    "SkillPublicationStore",
    "RestrictedLearningReviewer",
    "bundled_skills_dir",
    "explicit_skill_names",
    "files",
    "from_dir",
    "memory",
    "load_skill_package",
    "only",
    "path",
    "pin",
    "proposal_diff",
    "resolve_skill_resource",
    "shell",
    "skill_package_digest",
    "skill_reference_map",
]


_EXPORT_MODULES = (
    "chulk.skills.manifest",
    "chulk.skills.lifecycle_models",
    "chulk.skills.registry",
    "chulk.skills.locks",
    "chulk.skills.lifecycle",
    "chulk.skills.lifecycle_store",
    "chulk.skills.proposals",
    "chulk.skills.reviewer",
    "chulk.skills.publication",
)


if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        return resolve_export(
            name,
            public_names=__all__,
            owner_modules=_EXPORT_MODULES,
            namespace=globals(),
        )

    def __dir__() -> list[str]:
        return public_dir(__all__, globals())
