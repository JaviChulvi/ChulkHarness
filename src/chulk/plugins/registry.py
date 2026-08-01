"""Reviewed local plugin registration, startup verification, and loading."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib
from pathlib import Path
import sys
import threading
from types import ModuleType

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from chulk.plugins.inspection import (
    PluginInspectionError,
    inspect_plugin_directory,
)
from chulk.plugins.artifacts import (
    PluginArtifactError,
    PluginPackageStore,
    PreparedPluginPackage,
)
from chulk.plugins.catalog import ReviewedPluginCatalog
from chulk.plugins.lifecycle import (
    PluginHistoryStore,
    PluginLifecycleError,
    PluginRevocationStore,
    disabled_entry,
    plugin_authority_diff,
    validate_trusted_git_checkout,
)
from chulk.plugins.locks import PluginLockError, PluginLockFile
from chulk.plugins.manifest import resolve_plugin_resource
from chulk.plugins.migrations import (
    PluginMigrationError,
    PluginMigrationManager,
    validate_forward_only_migrations,
)
from chulk.plugins.models import (
    AuditSeverity,
    LoadedPluginEntryPoint,
    PluginAuditFinding,
    PluginAuditReport,
    PluginCategory,
    PluginEntryPoint,
    PluginInspection,
    PluginLifecycleAction,
    PluginLifecycleReceipt,
    PluginLockEntry,
    PluginRegistrationStatus,
    PluginReview,
    PluginSourceKind,
    PluginTrustState,
    PluginUpdatePlan,
)


class PluginRegistrationError(RuntimeError):
    """A local package cannot be safely registered."""


class PluginVerificationError(RuntimeError):
    """A locked plugin failed exact startup verification."""


class PluginLoadError(RuntimeError):
    """A trusted plugin entry point could not be imported safely."""


_IMPORT_LOCK = threading.RLock()


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
        self.packages = PluginPackageStore(self.runtime_dir)
        self.migrations = PluginMigrationManager(self.runtime_dir)
        self.history = PluginHistoryStore(
            self.runtime_dir,
            profile_id=profile_id,
        )
        self.revocations = PluginRevocationStore(self.runtime_dir)

    def inspect(self, path: Path | str) -> PluginInspection:
        """Return static authority and compatibility metadata."""
        source = Path(path).expanduser()
        if source.is_dir():
            return inspect_plugin_directory(source)
        return self.packages.prepare(source).inspection

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
            source_reference=str(package.root),
            artifact_digest=package.digest,
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

    def install(
        self,
        source: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLifecycleReceipt:
        """Quarantine and install one exact package after explicit review."""
        return self._install_prepared(
            self._prepare(source),
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=granted_capabilities,
        )

    def install_trusted_git(
        self,
        path: Path | str,
        *,
        repository_url: str,
        commit_sha: str,
        allowed_hosts: tuple[str, ...],
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLifecycleReceipt:
        """Install an already checked-out clean, pinned trusted-Git source."""
        try:
            root = validate_trusted_git_checkout(
                path,
                repository_url=repository_url,
                commit_sha=commit_sha,
                allowed_hosts=allowed_hosts,
            )
        except PluginLifecycleError as exc:
            raise PluginRegistrationError(str(exc)) from exc
        prepared = self.packages.prepare(root)
        prepared = replace(
            prepared,
            source_kind=PluginSourceKind.TRUSTED_GIT,
            source_reference=f"{repository_url}@{commit_sha.lower()}",
        )
        return self._install_prepared(
            prepared,
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=granted_capabilities,
        )

    def plan_update(
        self,
        source: Path | str,
    ) -> PluginUpdatePlan:
        """Quarantine and statically diff a candidate without enabling it."""
        prepared = self._prepare(source)
        return self._plan_prepared_update(prepared)

    def update(
        self,
        source: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] | None = None,
        approve_authority_changes: bool = False,
    ) -> PluginLifecycleReceipt:
        """Install a reviewed update with migration and lock rollback."""
        prepared = self._prepare(source)
        return self._update_prepared(
            prepared,
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=granted_capabilities,
            approve_authority_changes=approve_authority_changes,
        )

    def update_trusted_git(
        self,
        path: Path | str,
        *,
        repository_url: str,
        commit_sha: str,
        allowed_hosts: tuple[str, ...],
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] | None = None,
        approve_authority_changes: bool = False,
    ) -> PluginLifecycleReceipt:
        """Update from an exact clean trusted-Git checkout."""
        try:
            root = validate_trusted_git_checkout(
                path,
                repository_url=repository_url,
                commit_sha=commit_sha,
                allowed_hosts=allowed_hosts,
            )
        except PluginLifecycleError as exc:
            raise PluginRegistrationError(str(exc)) from exc
        prepared = replace(
            self.packages.prepare(root),
            source_kind=PluginSourceKind.TRUSTED_GIT,
            source_reference=f"{repository_url}@{commit_sha.lower()}",
        )
        return self._update_prepared(
            prepared,
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=granted_capabilities,
            approve_authority_changes=approve_authority_changes,
        )

    def uninstall(
        self,
        plugin_name: str,
        *,
        approved_by: str,
    ) -> PluginLifecycleReceipt:
        """Disable loading atomically while retaining recoverable metadata."""
        identity = _operator_identity(approved_by)
        entries = self.lock.read()
        entry = entries.get(plugin_name)
        if entry is None:
            raise PluginRegistrationError(
                f"plugin {plugin_name!r} is not installed"
            )
        dependents = sorted(
            name
            for name, candidate in entries.items()
            if candidate.status is PluginRegistrationStatus.ENABLED
            and any(
                dependency.name == plugin_name
                and not dependency.optional
                for dependency in candidate.manifest.dependencies
            )
        )
        if dependents:
            raise PluginRegistrationError(
                f"cannot uninstall {plugin_name!r}; enabled dependents: "
                + ", ".join(dependents)
            )
        point = self.history.save(entry, migration_backup=None)
        disabled = disabled_entry(entry)
        self.lock.write({**entries, plugin_name: disabled})
        timestamp = datetime.now(timezone.utc).isoformat()
        return PluginLifecycleReceipt(
            action=PluginLifecycleAction.UNINSTALL,
            plugin_name=entry.name,
            version=entry.version,
            digest=entry.digest,
            performed_by=identity,
            performed_at=timestamp,
            previous_version=entry.version,
            recovery_id=point.recovery_id,
        )

    def rollback(
        self,
        plugin_name: str,
        *,
        approved_by: str,
    ) -> PluginLifecycleReceipt:
        """Restore the newest exact recoverable package and database state."""
        identity = _operator_identity(approved_by)
        entries = self.lock.read()
        current = entries.get(plugin_name)
        if current is None:
            raise PluginRegistrationError(
                f"plugin {plugin_name!r} is not installed"
            )
        point = self.history.latest(
            plugin_name,
            different_from=current,
        )
        if point is None:
            raise PluginRegistrationError(
                f"plugin {plugin_name!r} has no recovery point"
            )
        recovered = point.entry
        if self.revocations.match(recovered) is not None:
            raise PluginRegistrationError(
                "cannot roll back to a revoked plugin digest"
            )
        try:
            inspected = inspect_plugin_directory(recovered.source_path)
        except PluginInspectionError as exc:
            raise PluginRegistrationError(
                f"recovery package is unavailable: {exc}"
            ) from exc
        if inspected.package.digest != recovered.digest:
            raise PluginRegistrationError(
                "recovery package no longer matches its exact lock"
            )
        proposed = {**entries, plugin_name: recovered}
        _raise_dependency_errors(proposed)
        current_backup = self.migrations.backup_current(
            current.name,
            current.version,
        )
        current_point = self.history.save(
            current,
            migration_backup=current_backup,
        )
        database = self.migrations.database_path(plugin_name)
        try:
            if point.migration_backup is not None:
                self.migrations.restore(
                    database,
                    point.migration_backup,
                )
            self.lock.write(proposed)
        except (PluginLockError, PluginMigrationError, OSError) as exc:
            if current_backup is not None:
                self.migrations.restore(database, current_backup)
            raise PluginRegistrationError(
                f"plugin rollback transaction failed: {exc}"
            ) from exc
        timestamp = datetime.now(timezone.utc).isoformat()
        return PluginLifecycleReceipt(
            action=PluginLifecycleAction.ROLLBACK,
            plugin_name=recovered.name,
            version=recovered.version,
            digest=recovered.digest,
            performed_by=identity,
            performed_at=timestamp,
            previous_version=current.version,
            recovery_id=current_point.recovery_id,
            migration_backup=point.migration_backup,
        )

    def revoke(
        self,
        plugin_name: str,
        *,
        reason: str,
        revoked_by: str,
    ) -> PluginLifecycleReceipt:
        """Persist a digest revocation and fail closed at startup."""
        entries = self.lock.read()
        entry = entries.get(plugin_name)
        if entry is None:
            raise PluginRegistrationError(
                f"plugin {plugin_name!r} is not installed"
            )
        record = self.revocations.revoke(
            entry,
            reason=reason,
            revoked_by=revoked_by,
        )
        point = self.history.save(entry, migration_backup=None)
        revoked = replace(
            entry,
            status=PluginRegistrationStatus.REVOKED,
        )
        self.lock.write({**entries, plugin_name: revoked})
        return PluginLifecycleReceipt(
            action=PluginLifecycleAction.REVOKE,
            plugin_name=entry.name,
            version=entry.version,
            digest=entry.digest,
            performed_by=record.revoked_by,
            performed_at=record.revoked_at,
            previous_version=entry.version,
            recovery_id=point.recovery_id,
        )

    @staticmethod
    def load_catalog(
        path: Path | str,
        *,
        allowed_git_hosts: tuple[str, ...],
    ) -> ReviewedPluginCatalog:
        """Load explicit metadata-only catalog state."""
        return ReviewedPluginCatalog.load(
            path,
            allowed_git_hosts=allowed_git_hosts,
        )

    def _prepare(
        self,
        source: Path | str,
    ) -> PreparedPluginPackage:
        try:
            return self.packages.prepare(source)
        except (PluginArtifactError, PluginInspectionError) as exc:
            raise PluginRegistrationError(str(exc)) from exc

    def _install_prepared(
        self,
        prepared: PreparedPluginPackage,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...],
    ) -> PluginLifecycleReceipt:
        identity, granted = _review_inputs(
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=granted_capabilities,
        )
        package = prepared.inspection.package
        manifest = package.manifest
        self._validate_candidate(prepared, granted=granted)
        entries = self.lock.read()
        if manifest.name in entries:
            raise PluginRegistrationError(
                f"plugin {manifest.name!r} is already installed; "
                "use the reviewed update lifecycle"
            )
        try:
            installed = self.packages.promote(prepared)
        except PluginArtifactError as exc:
            raise PluginRegistrationError(str(exc)) from exc
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = PluginLockEntry(
            name=manifest.name,
            version=manifest.version,
            digest=installed.package.digest,
            source_kind=prepared.source_kind,
            source_path=installed.package.root,
            status=PluginRegistrationStatus.ENABLED,
            manifest=installed.package.manifest,
            review=PluginReview(
                approved_by=identity,
                approved_at=timestamp,
                acknowledged_host_authority=True,
                granted_capabilities=granted,
            ),
            installed_at=timestamp,
            source_reference=prepared.source_reference,
            artifact_digest=prepared.artifact_digest,
        )
        proposed = {**entries, entry.name: entry}
        _raise_dependency_errors(proposed)
        database = self.migrations.database_path(entry.name)
        database_existed = database.exists()
        backup: Path | None = None
        try:
            backup = self.migrations.apply(
                manifest=entry.manifest,
                package_root=entry.source_path,
            )
            self.lock.write(proposed)
        except (PluginLockError, PluginMigrationError, OSError) as exc:
            self.migrations.restore(
                database,
                backup,
                remove_without_backup=not database_existed,
            )
            raise PluginRegistrationError(
                f"plugin install transaction failed: {exc}"
            ) from exc
        return PluginLifecycleReceipt(
            action=PluginLifecycleAction.INSTALL,
            plugin_name=entry.name,
            version=entry.version,
            digest=entry.digest,
            performed_by=identity,
            performed_at=timestamp,
            migration_backup=backup,
        )

    def _plan_prepared_update(
        self,
        prepared: PreparedPluginPackage,
    ) -> PluginUpdatePlan:
        package = prepared.inspection.package
        current = self.lock.get(package.manifest.name)
        if current is None:
            raise PluginRegistrationError(
                f"plugin {package.manifest.name!r} is not installed"
            )
        try:
            validate_forward_only_migrations(
                previous_manifest=current.manifest,
                previous_root=current.source_path,
                candidate_manifest=package.manifest,
                candidate_root=package.root,
            )
        except PluginMigrationError as exc:
            raise PluginRegistrationError(str(exc)) from exc
        authority_diff = plugin_authority_diff(
            current.manifest,
            package.manifest,
        )
        old_instructions = _instruction_contracts(
            current.manifest,
            current.source_path,
        )
        new_instructions = _instruction_contracts(
            package.manifest,
            package.root,
        )
        authority_diff = replace(
            authority_diff,
            added_instructions=tuple(
                sorted(new_instructions - old_instructions)
            ),
            removed_instructions=tuple(
                sorted(old_instructions - new_instructions)
            ),
        )
        return PluginUpdatePlan(
            plugin_name=current.name,
            current_version=current.version,
            candidate_version=package.manifest.version,
            current_digest=current.digest,
            candidate_digest=package.digest,
            artifact_digest=prepared.artifact_digest,
            source_kind=prepared.source_kind,
            compatible=prepared.inspection.compatible,
            authority_diff=authority_diff,
            migration_paths=package.manifest.migrations,
        )

    def _update_prepared(
        self,
        prepared: PreparedPluginPackage,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] | None,
        approve_authority_changes: bool,
    ) -> PluginLifecycleReceipt:
        plan = self._plan_prepared_update(prepared)
        entries = self.lock.read()
        current = entries[plan.plugin_name]
        requested = (
            current.review.granted_capabilities
            if granted_capabilities is None
            else granted_capabilities
        )
        identity, granted = _review_inputs(
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=requested,
        )
        if Version(plan.candidate_version) <= Version(plan.current_version):
            raise PluginRegistrationError(
                "reviewed updates must advance the plugin version; "
                "use rollback for older versions"
            )
        if plan.requires_reapproval and not approve_authority_changes:
            raise PluginRegistrationError(
                "plugin authority changed; explicit authority-change "
                "approval is required"
            )
        self._validate_candidate(prepared, granted=granted)
        try:
            installed = self.packages.promote(prepared)
        except PluginArtifactError as exc:
            raise PluginRegistrationError(str(exc)) from exc
        timestamp = datetime.now(timezone.utc).isoformat()
        updated = PluginLockEntry(
            name=current.name,
            version=installed.package.manifest.version,
            digest=installed.package.digest,
            source_kind=prepared.source_kind,
            source_path=installed.package.root,
            status=PluginRegistrationStatus.ENABLED,
            manifest=installed.package.manifest,
            review=PluginReview(
                approved_by=identity,
                approved_at=timestamp,
                acknowledged_host_authority=True,
                granted_capabilities=granted,
            ),
            installed_at=timestamp,
            source_reference=prepared.source_reference,
            artifact_digest=prepared.artifact_digest,
        )
        proposed = {**entries, current.name: updated}
        _raise_dependency_errors(proposed)
        database = self.migrations.database_path(current.name)
        database_existed = database.exists()
        backup: Path | None = None
        point = None
        try:
            backup = self.migrations.apply(
                manifest=updated.manifest,
                package_root=updated.source_path,
                previous_manifest=current.manifest,
                previous_root=current.source_path,
            )
            point = self.history.save(
                current,
                migration_backup=backup,
            )
            self.lock.write(proposed)
        except (
            PluginLifecycleError,
            PluginLockError,
            PluginMigrationError,
            OSError,
        ) as exc:
            self.migrations.restore(
                database,
                backup,
                remove_without_backup=not database_existed,
            )
            raise PluginRegistrationError(
                f"plugin update transaction failed: {exc}"
            ) from exc
        return PluginLifecycleReceipt(
            action=PluginLifecycleAction.UPDATE,
            plugin_name=updated.name,
            version=updated.version,
            digest=updated.digest,
            performed_by=identity,
            performed_at=timestamp,
            previous_version=current.version,
            recovery_id=point.recovery_id if point is not None else None,
            migration_backup=backup,
            authority_diff=plan.authority_diff,
        )

    def _validate_candidate(
        self,
        prepared: PreparedPluginPackage,
        *,
        granted: tuple[str, ...],
    ) -> None:
        inspection = prepared.inspection
        if not inspection.compatible:
            raise PluginRegistrationError(
                _compatibility_error(inspection)
            )
        manifest = inspection.package.manifest
        if manifest.trust is PluginTrustState.REVOKED:
            raise PluginRegistrationError(
                "a revoked plugin manifest cannot be installed"
            )
        undeclared = sorted(set(granted) - set(manifest.capabilities))
        if undeclared:
            raise PluginRegistrationError(
                "cannot grant undeclared plugin capabilities: "
                + ", ".join(undeclared)
            )
        for record in self.revocations.list():
            if (
                record.digest == inspection.package.digest
                or (
                    record.source_reference
                    and record.source_reference
                    == prepared.source_reference
                    and record.plugin_name == manifest.name
                )
            ):
                raise PluginRegistrationError(
                    "plugin package or source has been revoked"
                )

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
            revoked_record = self.revocations.match(entry)
            if revoked_record is not None:
                findings.append(
                    PluginAuditFinding(
                        plugin_name=name,
                        severity=AuditSeverity.ERROR,
                        code="revoked_digest",
                        message=(
                            f"plugin {name!r} matches revoked digest "
                            f"{revoked_record.digest}: "
                            f"{revoked_record.reason}"
                        ),
                    )
                )
                continue
            if entry.status is PluginRegistrationStatus.REVOKED:
                findings.append(
                    PluginAuditFinding(
                        plugin_name=name,
                        severity=AuditSeverity.ERROR,
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
                self.packages.verify_artifact(entry)
                inspection = self.inspect(entry.source_path)
            except (
                OSError,
                PluginArtifactError,
                PluginInspectionError,
            ) as exc:
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
        except (PluginLifecycleError, PluginLockError) as exc:
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
        try:
            selected_category = PluginCategory(category)
        except ValueError as exc:
            raise PluginLoadError(
                f"unsupported plugin category: {category!r}"
            ) from exc
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
        return LoadedPluginEntryPoint(
            plugin_name=entry.name,
            entry_point=selected,
            value=value,
        )


def _operator_identity(value: str) -> str:
    identity = value.strip()
    if (
        not identity
        or identity != value
        or "\x00" in identity
        or len(identity) > 200
    ):
        raise PluginRegistrationError("operator identity is invalid")
    return identity


def _review_inputs(
    *,
    approved_by: str,
    acknowledge_host_authority: bool,
    granted_capabilities: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    identity = _operator_identity(approved_by)
    if not acknowledge_host_authority:
        raise PluginRegistrationError(
            "in-process plugins require explicit host-authority "
            "acknowledgement"
        )
    granted = tuple(sorted(set(granted_capabilities)))
    if any(
        not isinstance(item, str)
        or not item
        or "\x00" in item
        for item in granted
    ):
        raise PluginRegistrationError(
            "granted plugin capabilities are invalid"
        )
    return identity, granted


def _raise_dependency_errors(
    entries: dict[str, PluginLockEntry],
) -> None:
    errors = [
        finding
        for finding in _dependency_findings(entries)
        if finding.severity is AuditSeverity.ERROR
    ]
    if errors:
        raise PluginRegistrationError(
            "; ".join(finding.message for finding in errors)
        )


def _instruction_contracts(
    manifest: object,
    root: Path,
) -> set[str]:
    contracts: set[str] = set()
    for relative in getattr(manifest, "instructions"):
        content = resolve_plugin_resource(root, relative).read_bytes()
        contracts.add(
            f"{relative}@sha256:{sha256(content).hexdigest()}"
        )
    return contracts


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
    with _IMPORT_LOCK:
        return _import_target_locked(lock_entry, entry_point)


def _import_target_locked(
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
        if not callable(value):
            raise PluginLoadError(
                "plugin entry points must resolve to callable factories"
            )
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
