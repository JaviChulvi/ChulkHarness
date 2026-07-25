"""Static plugin inspection and reviewed local registration."""

from chulk.plugins.inspection import (
    PluginInspectionError,
    inspect_plugin_directory,
)
from chulk.plugins.manifest import (
    PluginManifestError,
    load_plugin_manifest,
    resolve_plugin_resource,
)
from chulk.plugins.models import (
    AuditSeverity,
    FilesystemAccess,
    LoadedPluginEntryPoint,
    PLUGIN_LOCK_SCHEMA_VERSION,
    PLUGIN_MANIFEST_FILENAME,
    PLUGIN_MANIFEST_SCHEMA_VERSION,
    PluginAuditFinding,
    PluginAuditReport,
    PluginCategory,
    PluginDependency,
    PluginEntryPoint,
    PluginFilesystemRequirement,
    PluginInspection,
    PluginLockEntry,
    PluginManifest,
    PluginPackage,
    PluginRegistrationStatus,
    PluginReview,
    PluginSourceKind,
    PluginTrustState,
)


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
    "PluginInspectionError",
    "PluginLockEntry",
    "PluginManifest",
    "PluginManifestError",
    "PluginPackage",
    "PluginRegistrationStatus",
    "PluginReview",
    "PluginSourceKind",
    "PluginTrustState",
    "inspect_plugin_directory",
    "load_plugin_manifest",
    "resolve_plugin_resource",
]
