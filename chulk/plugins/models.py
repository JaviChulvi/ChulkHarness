"""Immutable models for statically inspected, operator-trusted plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


PLUGIN_MANIFEST_SCHEMA_VERSION = 1
PLUGIN_LOCK_SCHEMA_VERSION = 2
PLUGIN_MANIFEST_FILENAME = "chulk-plugin.yaml"


class PluginCategory(StrEnum):
    """Supported host extension contracts."""

    PROVIDER = "provider"
    TOOL = "tool"
    TOOLSET = "toolset"
    CHANNEL_ADAPTER = "channel_adapter"
    EXECUTION_BACKEND = "execution_backend"
    MEMORY_BACKEND = "memory_backend"
    RETRIEVAL_BACKEND = "retrieval_backend"
    SKILL_REGISTRY = "skill_registry"
    MEDIA_PROCESSOR = "media_processor"
    SCHEDULER = "scheduler"
    TRIGGER = "trigger"
    MCP_CONFIGURATION = "mcp_configuration"


class PluginTrustState(StrEnum):
    """Declared or host-reviewed trust state."""

    UNTRUSTED = "untrusted"
    LOCAL = "local"
    OPERATOR_TRUSTED = "operator-trusted"
    REVOKED = "revoked"


class PluginRegistrationStatus(StrEnum):
    """Whether a reviewed registration may be loaded."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    REVOKED = "revoked"


class PluginSourceKind(StrEnum):
    """Supported package origins for this lifecycle stage."""

    LOCAL_DIRECTORY = "local-directory"
    PREBUILT_WHEEL = "prebuilt-wheel"
    TRUSTED_GIT = "trusted-git"


class PluginLifecycleAction(StrEnum):
    """Host-only reviewed lifecycle mutations."""

    INSTALL = "install"
    UPDATE = "update"
    UNINSTALL = "uninstall"
    ROLLBACK = "rollback"
    REVOKE = "revoke"


class FilesystemAccess(StrEnum):
    """Declared host filesystem need."""

    READ = "read"
    WRITE = "write"


class AuditSeverity(StrEnum):
    """Stable audit finding severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PluginEntryPoint:
    """One named Python object exposed through a supported category."""

    name: str
    category: PluginCategory
    target: str
    required_capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "target": self.target,
            "required_capabilities": list(self.required_capabilities),
        }


@dataclass(frozen=True, slots=True)
class PluginDependency:
    """An exact plugin or installed Python distribution requirement."""

    name: str
    version_spec: str
    optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version_spec": self.version_spec,
            "optional": self.optional,
        }


@dataclass(frozen=True, slots=True)
class PluginFilesystemRequirement:
    """One disclosed filesystem path and access mode."""

    path: str
    access: FilesystemAccess

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "access": self.access.value}


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Validated static metadata that is safe to inspect before import."""

    name: str
    version: str
    description: str
    source: str
    trust: PluginTrustState
    requires_chulk: str
    requires_python: str
    entry_points: tuple[PluginEntryPoint, ...]
    capabilities: tuple[str, ...] = ()
    secret_refs: tuple[str, ...] = ()
    network_domains: tuple[str, ...] = ()
    filesystem: tuple[PluginFilesystemRequirement, ...] = ()
    dependencies: tuple[PluginDependency, ...] = ()
    python_dependencies: tuple[PluginDependency, ...] = ()
    migrations: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    schema_version: int = PLUGIN_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        entry_points: dict[str, dict[str, dict[str, Any]]] = {}
        for item in self.entry_points:
            entry_points.setdefault(item.category.value, {})[item.name] = {
                "target": item.target,
                "required_capabilities": list(
                    item.required_capabilities
                ),
            }
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "source": self.source,
            "trust": self.trust.value,
            "requires_chulk": self.requires_chulk,
            "requires_python": self.requires_python,
            "entry_points": entry_points,
            "capabilities": list(self.capabilities),
            "secret_refs": list(self.secret_refs),
            "network_domains": list(self.network_domains),
            "filesystem": [item.to_dict() for item in self.filesystem],
            "dependencies": {
                item.name: {
                    "version": item.version_spec,
                    "optional": item.optional,
                }
                for item in self.dependencies
            },
            "python_dependencies": {
                item.name: {
                    "version": item.version_spec,
                    "optional": item.optional,
                }
                for item in self.python_dependencies
            },
            "migrations": list(self.migrations),
            "instructions": list(self.instructions),
        }


@dataclass(frozen=True, slots=True)
class PluginPackage:
    """A validated local package and its exact static identity."""

    root: Path
    manifest_path: Path
    manifest: PluginManifest
    digest: str
    file_count: int
    total_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "manifest": self.manifest.to_dict(),
            "digest": self.digest,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True, slots=True)
class PluginInspection:
    """Static package inspection result with compatibility diagnostics."""

    package: PluginPackage
    chulk_compatible: bool
    python_compatible: bool
    missing_python_dependencies: tuple[str, ...] = ()
    incompatible_python_dependencies: tuple[str, ...] = ()

    @property
    def compatible(self) -> bool:
        return (
            self.chulk_compatible
            and self.python_compatible
            and not self.missing_python_dependencies
            and not self.incompatible_python_dependencies
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.package.to_dict(),
            "compatible": self.compatible,
            "chulk_compatible": self.chulk_compatible,
            "python_compatible": self.python_compatible,
            "missing_python_dependencies": list(
                self.missing_python_dependencies
            ),
            "incompatible_python_dependencies": list(
                self.incompatible_python_dependencies
            ),
        }


@dataclass(frozen=True, slots=True)
class PluginReview:
    """Host approval captured separately from plugin-authored metadata."""

    approved_by: str
    approved_at: str
    acknowledged_host_authority: bool
    granted_capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "acknowledged_host_authority": self.acknowledged_host_authority,
            "granted_capabilities": list(self.granted_capabilities),
        }


@dataclass(frozen=True, slots=True)
class PluginLockEntry:
    """Exact reviewed identity for one profile-local plugin registration."""

    name: str
    version: str
    digest: str
    source_kind: PluginSourceKind
    source_path: Path
    status: PluginRegistrationStatus
    manifest: PluginManifest
    review: PluginReview
    installed_at: str
    source_reference: str = ""
    artifact_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "source_kind": self.source_kind.value,
            "source_path": str(self.source_path),
            "status": self.status.value,
            "manifest": self.manifest.to_dict(),
            "review": self.review.to_dict(),
            "installed_at": self.installed_at,
            "source_reference": self.source_reference,
            "artifact_digest": self.artifact_digest,
        }


@dataclass(frozen=True, slots=True)
class PluginAuthorityDiff:
    """Review-relevant authority and behavior changes between packages."""

    added_capabilities: tuple[str, ...] = ()
    removed_capabilities: tuple[str, ...] = ()
    added_secret_refs: tuple[str, ...] = ()
    removed_secret_refs: tuple[str, ...] = ()
    added_network_domains: tuple[str, ...] = ()
    removed_network_domains: tuple[str, ...] = ()
    added_filesystem: tuple[str, ...] = ()
    removed_filesystem: tuple[str, ...] = ()
    added_dependencies: tuple[str, ...] = ()
    removed_dependencies: tuple[str, ...] = ()
    added_entry_points: tuple[str, ...] = ()
    removed_entry_points: tuple[str, ...] = ()
    added_instructions: tuple[str, ...] = ()
    removed_instructions: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return any(
            (
                self.added_capabilities,
                self.removed_capabilities,
                self.added_secret_refs,
                self.removed_secret_refs,
                self.added_network_domains,
                self.removed_network_domains,
                self.added_filesystem,
                self.removed_filesystem,
                self.added_dependencies,
                self.removed_dependencies,
                self.added_entry_points,
                self.removed_entry_points,
                self.added_instructions,
                self.removed_instructions,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: list(getattr(self, field_name))
            for field_name in (
                "added_capabilities",
                "removed_capabilities",
                "added_secret_refs",
                "removed_secret_refs",
                "added_network_domains",
                "removed_network_domains",
                "added_filesystem",
                "removed_filesystem",
                "added_dependencies",
                "removed_dependencies",
                "added_entry_points",
                "removed_entry_points",
                "added_instructions",
                "removed_instructions",
            )
        } | {"changed": self.changed}


@dataclass(frozen=True, slots=True)
class PluginUpdatePlan:
    """Static update diff produced before any package is enabled."""

    plugin_name: str
    current_version: str
    candidate_version: str
    current_digest: str
    candidate_digest: str
    artifact_digest: str
    source_kind: PluginSourceKind
    compatible: bool
    authority_diff: PluginAuthorityDiff
    migration_paths: tuple[str, ...] = ()

    @property
    def requires_reapproval(self) -> bool:
        return self.authority_diff.changed

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "current_version": self.current_version,
            "candidate_version": self.candidate_version,
            "current_digest": self.current_digest,
            "candidate_digest": self.candidate_digest,
            "artifact_digest": self.artifact_digest,
            "source_kind": self.source_kind.value,
            "compatible": self.compatible,
            "requires_reapproval": self.requires_reapproval,
            "authority_diff": self.authority_diff.to_dict(),
            "migration_paths": list(self.migration_paths),
        }


@dataclass(frozen=True, slots=True)
class PluginLifecycleReceipt:
    """Auditable result from one explicit operator lifecycle action."""

    action: PluginLifecycleAction
    plugin_name: str
    version: str
    digest: str
    performed_by: str
    performed_at: str
    previous_version: str | None = None
    recovery_id: str | None = None
    migration_backup: Path | None = None
    authority_diff: PluginAuthorityDiff | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "plugin_name": self.plugin_name,
            "version": self.version,
            "digest": self.digest,
            "performed_by": self.performed_by,
            "performed_at": self.performed_at,
            "previous_version": self.previous_version,
            "recovery_id": self.recovery_id,
            "migration_backup": (
                str(self.migration_backup)
                if self.migration_backup is not None
                else None
            ),
            "authority_diff": (
                self.authority_diff.to_dict()
                if self.authority_diff is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class PluginRevocation:
    """Operator-issued fail-closed revocation metadata."""

    digest: str
    plugin_name: str
    reason: str
    revoked_by: str
    revoked_at: str
    source_reference: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "digest": self.digest,
            "plugin_name": self.plugin_name,
            "reason": self.reason,
            "revoked_by": self.revoked_by,
            "revoked_at": self.revoked_at,
            "source_reference": self.source_reference,
        }


@dataclass(frozen=True, slots=True)
class PluginCatalogSource:
    """Pinned trusted-Git provenance for a metadata-only catalog."""

    repository_url: str
    commit_sha: str
    digest: str
    reviewed_by: str

    def to_dict(self) -> dict[str, str]:
        return {
            "repository_url": self.repository_url,
            "commit_sha": self.commit_sha,
            "digest": self.digest,
            "reviewed_by": self.reviewed_by,
        }


@dataclass(frozen=True, slots=True)
class PluginCatalogEntry:
    """Searchable static catalog metadata that grants no authority."""

    name: str
    version: str
    description: str
    package_source: str
    package_digest: str
    source_kind: PluginSourceKind
    trust: PluginTrustState
    capabilities: tuple[str, ...] = ()
    secret_refs: tuple[str, ...] = ()
    network_domains: tuple[str, ...] = ()
    categories: tuple[PluginCategory, ...] = ()
    audit_state: str = "reviewed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "package_source": self.package_source,
            "package_digest": self.package_digest,
            "source_kind": self.source_kind.value,
            "trust": self.trust.value,
            "capabilities": list(self.capabilities),
            "secret_refs": list(self.secret_refs),
            "network_domains": list(self.network_domains),
            "categories": [item.value for item in self.categories],
            "audit_state": self.audit_state,
        }


@dataclass(frozen=True, slots=True)
class PluginCatalogSnapshot:
    """One statically loaded reviewed catalog."""

    catalog_id: str
    source: PluginCatalogSource
    entries: tuple[PluginCatalogEntry, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "source": self.source.to_dict(),
            "entries": [item.to_dict() for item in self.entries],
        }


@dataclass(frozen=True, slots=True)
class PluginAuditFinding:
    """One credential-free verification result."""

    plugin_name: str
    severity: AuditSeverity
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "plugin_name": self.plugin_name,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class PluginAuditReport:
    """Deterministic startup audit across one profile lock."""

    profile_id: str
    findings: tuple[PluginAuditFinding, ...] = ()
    verified_plugins: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(
            finding.severity is not AuditSeverity.ERROR
            for finding in self.findings
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "ok": self.ok,
            "verified_plugins": list(self.verified_plugins),
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True, slots=True)
class LoadedPluginEntryPoint:
    """A reviewed object imported through an exact locked entry point."""

    plugin_name: str
    entry_point: PluginEntryPoint
    value: object = field(repr=False, compare=False)


__all__ = [
    "AuditSeverity",
    "FilesystemAccess",
    "LoadedPluginEntryPoint",
    "PLUGIN_LOCK_SCHEMA_VERSION",
    "PLUGIN_MANIFEST_FILENAME",
    "PLUGIN_MANIFEST_SCHEMA_VERSION",
    "PluginAuditFinding",
    "PluginAuditReport",
    "PluginCategory",
    "PluginCatalogEntry",
    "PluginCatalogSnapshot",
    "PluginCatalogSource",
    "PluginDependency",
    "PluginEntryPoint",
    "PluginFilesystemRequirement",
    "PluginInspection",
    "PluginAuthorityDiff",
    "PluginLifecycleAction",
    "PluginLifecycleReceipt",
    "PluginLockEntry",
    "PluginManifest",
    "PluginPackage",
    "PluginRegistrationStatus",
    "PluginRevocation",
    "PluginReview",
    "PluginSourceKind",
    "PluginTrustState",
    "PluginUpdatePlan",
]
