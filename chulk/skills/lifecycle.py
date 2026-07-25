"""Host-owned skill proposal application, rollback, and lock synchronization."""

from __future__ import annotations

from dataclasses import replace
from difflib import unified_diff
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Literal
from uuid import uuid4

from chulk.memory.security import ensure_memory_payload_safe
from chulk.skills.lifecycle_models import (
    LearningProposalKind,
    LearningProposalRecord,
    LearningProposalStatus,
    SkillLifecycleRecord,
    SkillLifecycleStatus,
)
from chulk.skills.lifecycle_store import SQLiteSkillLifecycleStore
from chulk.skills.locks import SkillLockEntry, SkillLockFile
from chulk.skills.manifest import (
    SkillPackage,
    load_skill_package,
    split_skill_front_matter,
)
from chulk.skills.registry import SkillRegistry


SkillScope = Literal["project", "profile"]
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?$")


class SkillLifecycleError(RuntimeError):
    """Base failure for a governed skill mutation."""


class SkillConflictError(SkillLifecycleError):
    """The reviewed base no longer matches the package on disk."""


class SkillApprovalError(SkillLifecycleError):
    """A proposal cannot be safely applied."""


class SkillLifecycleManager:
    """Apply only host-approved proposals through recoverable package swaps."""

    def __init__(
        self,
        store: SQLiteSkillLifecycleStore,
        *,
        project_skills_dir: Path | str,
        profile_skills_dir: Path | str,
        project_lock_path: Path | str | None = None,
        profile_lock_path: Path | str | None = None,
        registry: SkillRegistry | None = None,
    ) -> None:
        self.store = store
        self.project_skills_dir = Path(project_skills_dir).resolve()
        self.profile_skills_dir = Path(profile_skills_dir).resolve()
        self.project_lock = SkillLockFile(
            project_lock_path
            or self.project_skills_dir.parent / "skills.lock",
            scope="project",
            private=False,
        )
        self.profile_lock = SkillLockFile(
            profile_lock_path
            or self.profile_skills_dir.parent / "profile-skills.lock",
            scope="profile",
            private=True,
        )
        if self.project_lock.path.resolve() == self.profile_lock.path.resolve():
            raise ValueError("project and profile skill locks must be distinct")
        self.registry = registry

    def register_existing(
        self,
        *,
        scope: SkillScope,
    ) -> tuple[SkillLifecycleRecord, ...]:
        """Snapshot all validated packages currently owned by one scope."""
        root = self._skills_dir(scope)
        if not root.exists():
            self._verify_scope_inventory(scope=scope, found_names=set())
            return ()
        owned_root = root.resolve(strict=True)
        lock = self._lock(scope)
        initial_lock_entries = lock.read()
        found_names: set[str] = set()
        records: list[SkillLifecycleRecord] = []
        for path in sorted(root.glob("*/SKILL.md")):
            if path.is_symlink() or path.parent.is_symlink():
                raise SkillApprovalError(
                    f"skill package cannot be a symlink: {path.parent}"
                )
            try:
                path.resolve(strict=True).relative_to(owned_root)
            except ValueError as exc:
                raise SkillApprovalError(
                    f"skill package escapes its owned directory: {path}"
                ) from exc
            package = load_skill_package(path)
            if package.root.name != package.manifest.name:
                raise SkillApprovalError(
                    "skill manifest name must match its package directory"
                )
            _ensure_package_content_safe(package)
            found_names.add(package.manifest.name)
            lock_entry = initial_lock_entries.get(package.manifest.name)
            try:
                current = self.store.get_skill(
                    package.manifest.name,
                    scope=scope,
                )
            except KeyError:
                if lock_entry is not None or self._has_pending_change(
                    package.manifest.name,
                    scope=scope,
                ):
                    raise SkillConflictError(
                        f"uncommitted lifecycle state for skill "
                        f"{package.manifest.name!r}"
                    )
                revision = self.store.save_revision(
                    manifest=package.manifest,
                    digest=package.digest,
                    package_files=_snapshot_package(package.root),
                    scope=scope,
                    status=SkillLifecycleStatus.ACTIVE,
                )
                record = self.store.get_skill(
                    package.manifest.name,
                    scope=scope,
                )
            else:
                if lock_entry is None:
                    raise SkillConflictError(
                        f"skill lock for {package.manifest.name!r} is missing"
                    )
                if current.status in {
                    SkillLifecycleStatus.ARCHIVED,
                    SkillLifecycleStatus.STALE,
                }:
                    raise SkillConflictError(
                        f"governed skill {package.manifest.name!r} "
                        f"is {current.status.value}"
                    )
                if current.digest != package.digest:
                    raise SkillConflictError(
                        f"governed skill {package.manifest.name!r} changed "
                        "outside the approval lifecycle"
                    )
                revision = self.store.get_revision(
                    current.active_revision_id
                )
                record = current
                if not _lock_entry_matches(
                    lock_entry,
                    package=package,
                    revision_id=current.active_revision_id,
                    status=current.status,
                ):
                    raise SkillConflictError(
                        f"skill lock for {package.manifest.name!r} "
                        "does not match governed state"
                    )
            lock.update(
                SkillLockEntry.from_package(
                    package,
                    revision_id=revision.id,
                    status=record.status,
                    installed_at=record.updated_at,
                )
            )
            records.append(record)
        self._verify_scope_inventory(scope=scope, found_names=found_names)
        return tuple(records)

    def _has_pending_change(
        self,
        name: str,
        *,
        scope: SkillScope,
    ) -> bool:
        return any(
            proposal.target_name == name
            and proposal.kind.value.startswith("skill_")
            and str(proposal.metadata.get("scope", "project")) == scope
            for proposal in self.store.list_proposals(limit=1_000)
        )

    def _verify_scope_inventory(
        self,
        *,
        scope: SkillScope,
        found_names: set[str],
    ) -> None:
        lock_entries = self._lock(scope).read()
        records = {
            record.name: record
            for record in self.store.list_skills(scope=scope)
        }
        for name in sorted(set(lock_entries) | set(records)):
            entry = lock_entries.get(name)
            record = records.get(name)
            if entry is None:
                raise SkillConflictError(
                    f"skill lock for governed skill {name!r} is missing"
                )
            if record is None:
                raise SkillConflictError(
                    f"skill lock for {name!r} has no governed state"
                )
            if (
                entry.version != record.version
                or entry.digest != record.digest
                or entry.source != record.source
                or entry.trust != record.trust
                or entry.revision_id != record.active_revision_id
                or entry.status is not record.status
            ):
                raise SkillConflictError(
                    f"skill lock for {name!r} does not match governed state"
                )
            may_be_absent = record.status in {
                SkillLifecycleStatus.ARCHIVED,
                SkillLifecycleStatus.STALE,
            }
            if name not in found_names and not may_be_absent:
                raise SkillConflictError(
                    f"governed skill package {name!r} is missing"
                )

    def approve(
        self,
        proposal_id: str,
        *,
        approved_by: str,
    ) -> LearningProposalRecord:
        """Apply one pending skill proposal as a separate host operation."""
        if not approved_by.strip():
            raise ValueError("approved_by cannot be empty")
        proposal = self.store.get_proposal(proposal_id)
        if proposal.status is not LearningProposalStatus.PENDING:
            return proposal
        if not proposal.kind.value.startswith("skill_"):
            raise SkillApprovalError("proposal is not a skill change")
        scope = _proposal_scope(proposal)
        if proposal.kind is LearningProposalKind.SKILL_ARCHIVE:
            return self._approve_archive(
                proposal,
                scope=scope,
                approved_by=approved_by,
            )
        return self._approve_content_change(
            proposal,
            scope=scope,
            approved_by=approved_by,
        )

    def reject(
        self,
        proposal_id: str,
        *,
        rejected_by: str,
    ) -> LearningProposalRecord:
        if not rejected_by.strip():
            raise ValueError("rejected_by cannot be empty")
        return self.store.transition_proposal(
            proposal_id,
            status=LearningProposalStatus.REJECTED,
            reviewed_by=rejected_by,
        )

    def rollback(
        self,
        revision_id: str,
        *,
        scope: SkillScope,
        approved_by: str,
    ) -> SkillLifecycleRecord:
        """Restore an immutable package snapshot through a recoverable swap."""
        if not approved_by.strip():
            raise ValueError("approved_by cannot be empty")
        revision = self.store.get_revision(revision_id)
        if revision.scope != scope:
            raise SkillConflictError(
                "skill revision belongs to a different lifecycle scope"
            )
        target = self._skill_root(scope, revision.name)
        current = _load_existing_package(target)
        staged_parent, staged = _stage_snapshot(
            self._skills_dir(scope),
            revision.name,
            revision.package_files,
        )
        staged_package = load_skill_package(
            staged / "SKILL.md",
            expected_digest=revision.digest,
        )
        previous_lock = self._lock(scope).snapshot()
        backup = staged_parent / "previous"
        swapped = False
        try:
            _swap_package(target, staged, backup)
            swapped = True
            self._lock(scope).update(
                SkillLockEntry.from_package(
                    staged_package,
                    revision_id=revision.id,
                    status=SkillLifecycleStatus.ACTIVE,
                )
            )
            record = self.store.activate_revision(revision.id)
        except BaseException:
            if swapped:
                _restore_package(target, backup)
            self._lock(scope).restore(previous_lock)
            raise
        finally:
            shutil.rmtree(staged_parent, ignore_errors=True)
        if current is not None and current.digest == revision.digest:
            return record
        self._refresh_registry()
        return record

    def verify_locks(self) -> tuple[str, ...]:
        """Return deterministic lock/package mismatches without changing state."""
        failures: list[str] = []
        for scope in ("project", "profile"):
            lock = self._lock(scope)
            root = self._skills_dir(scope)
            for name, entry in lock.read().items():
                try:
                    record = self.store.get_skill(name, scope=scope)
                except KeyError:
                    failures.append(f"{scope}:{name}:missing_governed_state")
                    continue
                if record.digest != entry.digest:
                    failures.append(f"{scope}:{name}:database_digest_mismatch")
                    continue
                if record.active_revision_id != entry.revision_id:
                    failures.append(f"{scope}:{name}:revision_mismatch")
                    continue
                if record.status is not entry.status:
                    failures.append(f"{scope}:{name}:status_mismatch")
                    continue
                target = root / name / "SKILL.md"
                if (
                    record.status
                    in {
                        SkillLifecycleStatus.ARCHIVED,
                        SkillLifecycleStatus.STALE,
                    }
                    and not target.exists()
                ):
                    continue
                try:
                    package = load_skill_package(
                        target,
                        expected_digest=entry.digest,
                    )
                except (OSError, ValueError) as exc:
                    failures.append(f"{scope}:{name}:{type(exc).__name__}")
                    continue
                if package.manifest.version != entry.version:
                    failures.append(f"{scope}:{name}:version_mismatch")
                    continue
                if not _lock_entry_matches(
                    entry,
                    package=package,
                    revision_id=record.active_revision_id,
                    status=record.status,
                ):
                    failures.append(f"{scope}:{name}:lock_metadata_mismatch")
        return tuple(sorted(failures))

    def _approve_content_change(
        self,
        proposal: LearningProposalRecord,
        *,
        scope: SkillScope,
        approved_by: str,
    ) -> LearningProposalRecord:
        assert proposal.target_name is not None
        assert proposal.content is not None
        target = self._skill_root(scope, proposal.target_name)
        current = _load_existing_package(target)
        expected_base = proposal.metadata.get("base_digest")
        if proposal.kind is LearningProposalKind.SKILL_CREATE:
            if current is not None:
                raise SkillConflictError(
                    f"skill {proposal.target_name!r} already exists"
                )
            increment_patch = False
        else:
            if current is None:
                raise SkillConflictError(
                    f"skill {proposal.target_name!r} no longer exists"
                )
            if not isinstance(expected_base, str) or expected_base != current.digest:
                raise SkillConflictError(
                    "skill package changed after the proposal was reviewed"
                )
            increment_patch = True

        ensure_memory_payload_safe(
            rationale=proposal.rationale,
            metadata=proposal.metadata,
        )
        staged_parent, staged = _stage_content_change(
            self._skills_dir(scope),
            proposal.target_name,
            proposal.content,
            current=current,
        )
        package = load_skill_package(staged / "SKILL.md")
        if package.manifest.name != proposal.target_name:
            shutil.rmtree(staged_parent, ignore_errors=True)
            raise SkillApprovalError(
                "proposal target does not match the skill manifest name"
            )
        if (
            tuple(proposal.required_capabilities)
            != package.manifest.required_capabilities
        ):
            shutil.rmtree(staged_parent, ignore_errors=True)
            raise SkillApprovalError(
                "proposal capabilities do not match the validated manifest"
            )
        _ensure_package_content_safe(package)
        existing_revision = self.store.find_revision(
            name=package.manifest.name,
            digest=package.digest,
            scope=scope,
        )
        revision_id = (
            existing_revision.id if existing_revision is not None else str(uuid4())
        )
        lock = self._lock(scope)
        previous_lock = lock.snapshot()
        backup = staged_parent / "previous"
        swapped = False
        try:
            _swap_package(target, staged, backup)
            swapped = True
            installed_package = load_skill_package(
                target / "SKILL.md",
                expected_digest=package.digest,
            )
            lock.update(
                SkillLockEntry.from_package(
                    installed_package,
                    revision_id=revision_id,
                )
            )
            approved = self.store.approve_skill_revision(
                proposal.id,
                manifest=installed_package.manifest,
                digest=installed_package.digest,
                package_files=_snapshot_package(installed_package.root),
                increment_patch=increment_patch,
                revision_id=revision_id,
                reviewed_by=approved_by,
                scope=scope,
            )
        except BaseException:
            if swapped:
                _restore_package(target, backup)
            lock.restore(previous_lock)
            raise
        finally:
            shutil.rmtree(staged_parent, ignore_errors=True)
        self._refresh_registry()
        return approved

    def _approve_archive(
        self,
        proposal: LearningProposalRecord,
        *,
        scope: SkillScope,
        approved_by: str,
    ) -> LearningProposalRecord:
        assert proposal.target_name is not None
        target = self._skill_root(scope, proposal.target_name)
        current = _load_existing_package(target)
        if current is None:
            raise SkillConflictError(
                f"skill {proposal.target_name!r} no longer exists"
            )
        expected_base = proposal.metadata.get("base_digest")
        if not isinstance(expected_base, str) or expected_base != current.digest:
            raise SkillConflictError(
                "skill package changed after the proposal was reviewed"
            )
        lock = self._lock(scope)
        previous_lock = lock.snapshot()
        entries = lock.read()
        entry = entries.get(proposal.target_name)
        if entry is None:
            raise SkillApprovalError("skill archive requires a matching lock entry")
        entries[proposal.target_name] = replace(
            entry,
            status=SkillLifecycleStatus.ARCHIVED,
        )
        staged_parent = Path(
            tempfile.mkdtemp(
                prefix=".chulk-skill-archive-",
                dir=self._skills_dir(scope).parent,
            )
        )
        backup = staged_parent / "previous"
        swapped = False
        try:
            os.replace(target, backup)
            swapped = True
            lock.write(entries)
            approved = self.store.approve_skill_archive(
                proposal.id,
                name=proposal.target_name,
                reviewed_by=approved_by,
                scope=scope,
            )
        except BaseException:
            if swapped and backup.exists():
                os.replace(backup, target)
            lock.restore(previous_lock)
            raise
        finally:
            shutil.rmtree(staged_parent, ignore_errors=True)
        self._refresh_registry()
        return approved

    def _skills_dir(self, scope: SkillScope) -> Path:
        return (
            self.project_skills_dir
            if scope == "project"
            else self.profile_skills_dir
        )

    def _lock(self, scope: SkillScope) -> SkillLockFile:
        return self.project_lock if scope == "project" else self.profile_lock

    def _skill_root(self, scope: SkillScope, name: str) -> Path:
        if _SKILL_NAME_PATTERN.fullmatch(name) is None:
            raise SkillApprovalError("proposal target is not a valid skill name")
        skills_dir = self._skills_dir(scope)
        target = skills_dir / name
        resolved_parent = target.parent.resolve()
        if resolved_parent != skills_dir:
            raise SkillApprovalError("skill target escapes its owned directory")
        if target.is_symlink():
            raise SkillApprovalError("skill target cannot be a symlink")
        return target

    def _refresh_registry(self) -> None:
        if self.registry is not None:
            self.registry.load_metadata()


def proposal_diff(
    *,
    before: str | None,
    after: str | None,
    name: str,
) -> str:
    """Return a deterministic review diff for full SKILL.md content."""
    return "".join(
        unified_diff(
            (before or "").splitlines(keepends=True),
            (after or "").splitlines(keepends=True),
            fromfile=f"{name}/SKILL.md:before",
            tofile=f"{name}/SKILL.md:after",
        )
    )


def _lock_entry_matches(
    entry: SkillLockEntry,
    *,
    package: SkillPackage,
    revision_id: str,
    status: SkillLifecycleStatus,
) -> bool:
    manifest = package.manifest
    return (
        entry.name == manifest.name
        and entry.version == manifest.version
        and entry.digest == package.digest
        and entry.source == manifest.source
        and entry.trust == manifest.trust
        and entry.status is status
        and entry.revision_id == revision_id
        and entry.required_tools == manifest.required_tools
        and entry.required_capabilities == manifest.required_capabilities
        and entry.review_state == "approved"
    )


def _proposal_scope(proposal: LearningProposalRecord) -> SkillScope:
    scope = str(proposal.metadata.get("scope", "project"))
    if scope not in {"project", "profile"}:
        raise SkillApprovalError("proposal scope must be project or profile")
    return scope  # type: ignore[return-value]


def _load_existing_package(root: Path) -> SkillPackage | None:
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise SkillApprovalError(f"skill package is not a directory: {root}")
    return load_skill_package(root / "SKILL.md")


def _stage_content_change(
    skills_dir: Path,
    name: str,
    content: str,
    *,
    current: SkillPackage | None,
) -> tuple[Path, Path]:
    skills_dir.mkdir(parents=True, exist_ok=True)
    staged_parent = Path(
        tempfile.mkdtemp(prefix=".chulk-skill-apply-", dir=skills_dir.parent)
    )
    staged = staged_parent / name
    if current is None:
        staged.mkdir()
    else:
        shutil.copytree(current.root, staged, symlinks=True)
    (staged / "SKILL.md").write_text(content, encoding="utf-8")
    return staged_parent, staged


def _stage_snapshot(
    skills_dir: Path,
    name: str,
    package_files: dict[str, bytes],
) -> tuple[Path, Path]:
    skills_dir.mkdir(parents=True, exist_ok=True)
    staged_parent = Path(
        tempfile.mkdtemp(prefix=".chulk-skill-rollback-", dir=skills_dir.parent)
    )
    staged = staged_parent / name
    staged.mkdir()
    for relative, content in package_files.items():
        destination = staged.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return staged_parent, staged


def _swap_package(target: Path, staged: Path, backup: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(staged, target)
    except BaseException:
        if backup.exists():
            os.replace(backup, target)
        raise


def _restore_package(target: Path, backup: Path) -> None:
    if target.exists():
        failed = backup.parent / "failed"
        os.replace(target, failed)
    if backup.exists():
        os.replace(backup, target)


def _snapshot_package(root: Path) -> dict[str, bytes]:
    package_root = root.resolve(strict=True)
    files: dict[str, bytes] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(package_root)
        except ValueError as exc:
            raise SkillApprovalError(
                f"skill package resource escapes its root: {path}"
            ) from exc
        files[path.relative_to(package_root).as_posix()] = resolved.read_bytes()
    return files


def _ensure_package_content_safe(package: SkillPackage) -> None:
    skill_text = package.manifest_path.read_text(encoding="utf-8")
    _metadata, body = split_skill_front_matter(skill_text)
    ensure_memory_payload_safe(skill_body=body)
    for relative in package.manifest.declared_resources:
        if relative == "SKILL.md":
            continue
        resource = package.root.joinpath(*relative.split("/"))
        try:
            text = resource.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        ensure_memory_payload_safe(skill_resource=text)


__all__ = [
    "SkillApprovalError",
    "SkillConflictError",
    "SkillLifecycleError",
    "SkillLifecycleManager",
    "SkillScope",
    "proposal_diff",
]
