"""Immutable models for statically inspected, operator-trusted plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


PLUGIN_MANIFEST_SCHEMA_VERSION = 1
PLUGIN_LOCK_SCHEMA_VERSION = 1
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
    schema_version: int = PLUGIN_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "source": self.source,
            "trust": self.trust.value,
            "requires_chulk": self.requires_chulk,
            "requires_python": self.requires_python,
            "entry_points": [item.to_dict() for item in self.entry_points],
            "capabilities": list(self.capabilities),
            "secret_refs": list(self.secret_refs),
            "network_domains": list(self.network_domains),
            "filesystem": [item.to_dict() for item in self.filesystem],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "python_dependencies": [
                item.to_dict() for item in self.python_dependencies
            ],
            "migrations": list(self.migrations),
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
    "PluginDependency",
    "PluginEntryPoint",
    "PluginFilesystemRequirement",
    "PluginInspection",
    "PluginLockEntry",
    "PluginManifest",
    "PluginPackage",
    "PluginRegistrationStatus",
    "PluginReview",
    "PluginSourceKind",
    "PluginTrustState",
]
