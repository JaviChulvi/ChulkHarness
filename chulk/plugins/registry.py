"""Reviewed local plugin registration, startup verification, and loading."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys
from types import ModuleType

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from chulk.plugins.inspection import (
    PluginInspectionError,
    inspect_plugin_directory,
)
from chulk.plugins.locks import PluginLockError, PluginLockFile
from chulk.plugins.models import (
    AuditSeverity,
    LoadedPluginEntryPoint,
    PluginAuditFinding,
    PluginAuditReport,
    PluginCategory,
    PluginEntryPoint,
    PluginInspection,
    PluginLockEntry,
    PluginRegistrationStatus,
    PluginReview,
    PluginSourceKind,
    PluginTrustState,
)


class PluginRegistrationError(RuntimeError):
    """A local package cannot be safely registered."""


class PluginVerificationError(RuntimeError):
    """A locked plugin failed exact startup verification."""


class PluginLoadError(RuntimeError):
    """A trusted plugin entry point could not be imported safely."""


class LocalPluginRegistry:
    """Profile-local registry for explicitly trusted Python packages."""

    def __init__(
        self,
        runtime_dir: Path | str,
        *,
        profile_id: str,
        lock_path: Path | str | None = None,
    ) -> None:
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        self.profile_id = profile_id
        self.lock = PluginLockFile(
            lock_path or self.runtime_dir / "plugins.lock",
            profile_id=profile_id,
        )

    def inspect(self, path: Path | str) -> PluginInspection:
        """Return static authority and compatibility metadata."""
        return inspect_plugin_directory(path)

    def register_local(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLockEntry:
        """Register an exact local directory after explicit host review."""
        identity = approved_by.strip()
        if not identity:
            raise PluginRegistrationError(
                "approved_by cannot be empty"
            )
        if not acknowledge_host_authority:
            raise PluginRegistrationError(
                "in-process plugins require explicit host-authority "
                "acknowledgement"
            )
        try:
            inspection = self.inspect(path)
        except PluginInspectionError as exc:
            raise PluginRegistrationError(str(exc)) from exc
        if not inspection.compatible:
            raise PluginRegistrationError(
                _compatibility_error(inspection)
            )
        package = inspection.package
        manifest = package.manifest
        if manifest.trust is PluginTrustState.REVOKED:
            raise PluginRegistrationError(
                "a revoked plugin manifest cannot be registered"
            )
        granted = tuple(sorted(set(granted_capabilities)))
        undeclared = sorted(set(granted) - set(manifest.capabilities))
        if undeclared:
            raise PluginRegistrationError(
                "cannot grant undeclared plugin capabilities: "
                + ", ".join(undeclared)
            )
        entries = self.lock.read()
        existing = entries.get(manifest.name)
        if existing is not None:
            if (
                existing.digest == package.digest
                and existing.source_path == package.root
                and existing.review.granted_capabilities == granted
                and existing.status is PluginRegistrationStatus.ENABLED
            ):
                return existing
            raise PluginRegistrationError(
                f"plugin {manifest.name!r} is already registered; "
                "updates require the reviewed distribution lifecycle"
            )
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = PluginLockEntry(
            name=manifest.name,
            version=manifest.version,
            digest=package.digest,
            source_kind=PluginSourceKind.LOCAL_DIRECTORY,
            source_path=package.root,
            status=PluginRegistrationStatus.ENABLED,
            manifest=manifest,
            review=PluginReview(
                approved_by=identity,
                approved_at=timestamp,
                acknowledged_host_authority=True,
                granted_capabilities=granted,
            ),
            installed_at=timestamp,
        )
        proposed = {**entries, entry.name: entry}
        findings = _dependency_findings(proposed)
        errors = [
            finding for finding in findings
            if finding.severity is AuditSeverity.ERROR
        ]
        if errors:
            raise PluginRegistrationError(
                "; ".join(finding.message for finding in errors)
            )
        self.lock.write(proposed)
        return entry

    def list(self) -> tuple[PluginLockEntry, ...]:
        """List exact registrations without importing them."""
        return tuple(self.lock.read().values())

    def audit(self) -> PluginAuditReport:
        """Verify every lock and package without importing plugin code."""
        entries = self.lock.read()
        findings: list[PluginAuditFinding] = list(
            _dependency_findings(entries)
        )
        verified: list[str] = []
        for name, entry in sorted(entries.items()):
            if entry.status is PluginRegistrationStatus.REVOKED:
                findings.append(
                    PluginAuditFinding(
                        plugin_name=name,
                        severity=AuditSeverity.WARNING,
                        code="revoked",
                        message=f"plugin {name!r} is revoked and will not load",
                    )
                )
                continue
            if entry.status is PluginRegistrationStatus.DISABLED:
                findings.append(
                    PluginAuditFinding(
                        plugin_name=name,
                        severity=AuditSeverity.INFO,
                        code="disabled",
                        message=f"plugin {name!r} is disabled",
                    )
                )
                continue
            try:
                inspection = self.inspect(entry.source_path)
            except (OSError, PluginInspectionError) as exc:
                findings.append(
                    PluginAuditFinding(
                        plugin_name=name,
                        severity=AuditSeverity.ERROR,
                        code="package_invalid",
                        message=f"plugin {name!r} cannot be inspected: {exc}",
                    )
                )
                continue
            package = inspection.package
            mismatches: list[str] = []
            if package.manifest.name != entry.name:
                mismatches.append("name")
            if package.manifest.version != entry.version:
                mismatches.append("version")
            if package.digest != entry.digest:
                mismatches.append("digest")
            if package.manifest != entry.manifest:
                mismatches.append("manifest")
            if mismatches:
                findings.append(
                    PluginAuditFinding(
                        plugin_name=name,
                        severity=AuditSeverity.ERROR,
                        code="lock_mismatch",
                        message=(
                            f"plugin {name!r} differs from its reviewed lock: "
                            f"{', '.join(mismatches)}"
                        ),
                    )
                )
                continue
            if not inspection.compatible:
                findings.append(
                    PluginAuditFinding(
                        plugin_name=name,
                        severity=AuditSeverity.ERROR,
                        code="incompatible",
                        message=_compatibility_error(inspection),
                    )
                )
                continue
            verified.append(name)
        return PluginAuditReport(
            profile_id=self.profile_id,
            findings=tuple(
                sorted(
                    findings,
                    key=lambda item: (
                        item.plugin_name,
                        item.severity.value,
                        item.code,
                    ),
                )
            ),
            verified_plugins=tuple(sorted(verified)),
        )

    def verify_startup(self) -> PluginAuditReport:
        """Fail closed before runtime assembly if an enabled plugin changed."""
        try:
            report = self.audit()
        except PluginLockError as exc:
            raise PluginVerificationError(str(exc)) from exc
        if not report.ok:
            errors = [
                finding.message
                for finding in report.findings
                if finding.severity is AuditSeverity.ERROR
            ]
            raise PluginVerificationError("; ".join(errors))
        return report

    def load_entry_point(
        self,
        plugin_name: str,
        category: PluginCategory | str,
        entry_name: str,
        *,
        available_capabilities: tuple[str, ...] = (),
    ) -> LoadedPluginEntryPoint:
        """Import one exact trusted factory after startup verification."""
        self.verify_startup()
        entry = self.lock.get(plugin_name)
        if entry is None:
            raise PluginLoadError(
                f"plugin {plugin_name!r} is not registered"
            )
        if entry.status is not PluginRegistrationStatus.ENABLED:
            raise PluginLoadError(
                f"plugin {plugin_name!r} is not enabled"
            )
        selected_category = PluginCategory(category)
        selected = next(
            (
                candidate
                for candidate in entry.manifest.entry_points
                if candidate.category is selected_category
                and candidate.name == entry_name
            ),
            None,
        )
        if selected is None:
            raise PluginLoadError(
                f"plugin entry point does not exist: "
                f"{selected_category.value}:{entry_name}"
            )
        available = set(available_capabilities)
        granted = set(entry.review.granted_capabilities)
        if not granted.issubset(available):
            missing = sorted(granted - available)
            raise PluginLoadError(
                "runtime does not grant reviewed plugin capabilities: "
                + ", ".join(missing)
            )
        if not set(selected.required_capabilities).issubset(granted):
            raise PluginLoadError(
                "entry point requires capabilities that were not reviewed"
            )
        value = _import_target(entry, selected)
        if not callable(value):
            raise PluginLoadError(
                "plugin entry points must resolve to callable factories"
            )
        return LoadedPluginEntryPoint(
            plugin_name=entry.name,
            entry_point=selected,
            value=value,
        )


def _dependency_findings(
    entries: dict[str, PluginLockEntry],
) -> tuple[PluginAuditFinding, ...]:
    findings: list[PluginAuditFinding] = []
    enabled = {
        name: entry
        for name, entry in entries.items()
        if entry.status is PluginRegistrationStatus.ENABLED
    }
    graph: dict[str, set[str]] = {name: set() for name in enabled}
    for name, entry in enabled.items():
        for dependency in entry.manifest.dependencies:
            matched = enabled.get(dependency.name)
            if matched is None:
                if not dependency.optional:
                    findings.append(
                        PluginAuditFinding(
                            plugin_name=name,
                            severity=AuditSeverity.ERROR,
                            code="missing_dependency",
                            message=(
                                f"plugin {name!r} requires registered plugin "
                                f"{dependency.name!r} {dependency.version_spec}"
                            ),
                        )
                    )
                continue
            if Version(matched.version) not in SpecifierSet(
                dependency.version_spec
            ):
                findings.append(
                    PluginAuditFinding(
                        plugin_name=name,
                        severity=AuditSeverity.ERROR,
                        code="incompatible_dependency",
                        message=(
                            f"plugin {name!r} requires {dependency.name!r} "
                            f"{dependency.version_spec}, found "
                            f"{matched.version}"
                        ),
                    )
                )
            graph[name].add(dependency.name)
    cycle = _dependency_cycle(graph)
    if cycle:
        cycle_text = " -> ".join((*cycle, cycle[0]))
        for name in cycle:
            findings.append(
                PluginAuditFinding(
                    plugin_name=name,
                    severity=AuditSeverity.ERROR,
                    code="dependency_cycle",
                    message=f"plugin dependency cycle: {cycle_text}",
                )
            )
    return tuple(findings)


def _dependency_cycle(
    graph: dict[str, set[str]],
) -> tuple[str, ...]:
    visiting: list[str] = []
    active: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> tuple[str, ...]:
        if name in active:
            return tuple(visiting[visiting.index(name) :])
        if name in visited:
            return ()
        active.add(name)
        visiting.append(name)
        for dependency in sorted(graph.get(name, ())):
            cycle = visit(dependency)
            if cycle:
                return cycle
        visiting.pop()
        active.remove(name)
        visited.add(name)
        return ()

    for name in sorted(graph):
        cycle = visit(name)
        if cycle:
            return cycle
    return ()


def _import_target(
    lock_entry: PluginLockEntry,
    entry_point: PluginEntryPoint,
) -> object:
    module_name, _, attribute_path = entry_point.target.partition(":")
    top_level = module_name.partition(".")[0]
    conflicting = next(
        (
            name
            for name, module in sys.modules.items()
            if (
                name == top_level or name.startswith(f"{top_level}.")
            )
            and module is not None
            and not _module_belongs_to(
                module,
                lock_entry.source_path,
            )
        ),
        None,
    )
    if conflicting is not None:
        raise PluginLoadError(
            f"plugin module namespace {conflicting!r} is already owned "
            "outside the reviewed package"
        )
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    before_modules = set(sys.modules)
    try:
        sys.path.insert(0, str(lock_entry.source_path))
        sys.dont_write_bytecode = True
        module = importlib.import_module(module_name)
        if not _module_belongs_to(module, lock_entry.source_path):
            raise PluginLoadError(
                "imported plugin module does not belong to the reviewed package"
            )
        value: object = module
        for attribute in attribute_path.split("."):
            value = getattr(value, attribute)
        return value
    except PluginLoadError:
        _discard_failed_plugin_modules(
            before_modules,
            lock_entry.source_path,
        )
        raise
    except Exception as exc:
        _discard_failed_plugin_modules(
            before_modules,
            lock_entry.source_path,
        )
        raise PluginLoadError(
            f"failed to load plugin entry point "
            f"{entry_point.category.value}:{entry_point.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode


def _module_belongs_to(module: ModuleType, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return False
    try:
        Path(module_file).resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _discard_failed_plugin_modules(
    before_modules: set[str],
    root: Path,
) -> None:
    for name in set(sys.modules) - before_modules:
        module = sys.modules.get(name)
        if module is not None and _module_belongs_to(module, root):
            sys.modules.pop(name, None)


def _compatibility_error(inspection: PluginInspection) -> str:
    parts: list[str] = []
    if not inspection.chulk_compatible:
        parts.append(
            "the installed Chulk version does not satisfy "
            f"{inspection.package.manifest.requires_chulk}"
        )
    if not inspection.python_compatible:
        parts.append(
            "the active Python version does not satisfy "
            f"{inspection.package.manifest.requires_python}"
        )
    if inspection.missing_python_dependencies:
        parts.append(
            "missing Python dependencies: "
            + ", ".join(inspection.missing_python_dependencies)
        )
    if inspection.incompatible_python_dependencies:
        parts.append(
            "incompatible Python dependencies: "
            + ", ".join(
                inspection.incompatible_python_dependencies
            )
        )
    return "; ".join(parts) or "plugin compatibility check failed"


__all__ = [
    "LocalPluginRegistry",
    "PluginLoadError",
    "PluginRegistrationError",
    "PluginVerificationError",
]
