"""Reviewable temporary-copy and Git-worktree execution backends."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePath
import shutil
import stat
import subprocess
import tempfile
import threading
from typing import Any
from uuid import uuid4

from chulk.execution.base import ExecutionSession
from chulk.execution.host import (
    HostExecutionBackend,
    HostExecutionSession,
    has_child_task_scope,
)
from chulk.execution.models import (
    ChangeApplicationResult,
    ChangeDisposition,
    ChangeRecord,
    ChangeSet,
    ChangeSetApproval,
    CommandExecutionRequest,
    ContainmentStatus,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSessionRequest,
    ExecutionWorkspace,
    FileWriteRequest,
    NetworkPolicy,
    PatchApplyRequest,
    ProcessStartRequest,
    WorkspaceMode,
    WorkspacePersistence,
)
from chulk.execution.policy import (
    EnvironmentPolicy,
    GitWorktreePolicy,
    ProcessPolicy,
    ResourcePolicy,
    SecretPolicy,
    TransferPolicy,
    UnsafePathAction,
    WorkspaceMaterializationPolicy,
)
from chulk.execution.processes import owner_key
from chulk.tools.files import FileReadPolicy, safe_read_error, safe_write_error
from chulk.tools.registry import ToolFailureKind, ToolResult
from chulk.tools.shell import (
    DirectShellExecutionPolicy,
    ShellExecutionDecision,
    ShellExecutionPolicy,
    ShellExecutionRequest,
)


_SECRET_NAME_MARKERS = (
    "access_key",
    "api_key",
    "apikey",
    "auth",
    "bearer",
    "cookie",
    "credential",
    "passwd",
    "password",
    "private_key",
    "secret",
    "token",
)


class WorkspacePolicyError(RuntimeError):
    """A workspace could not be materialized or reviewed safely."""

    def __init__(self, message: str, *, code: str, path: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


@dataclass(frozen=True)
class _FileSnapshot:
    data: bytes
    sha256: str
    text: str | None
    mode: int


@dataclass
class _ApplyBackup:
    host_root: Path
    path: Path
    relative_path: str
    existed: bool
    data: bytes | None
    mode: int | None


class TemporaryWorkspaceBackend(HostExecutionBackend):
    """Execute in an allowlisted copy and retain bounded changes for review."""

    name = "temporary"

    def __init__(
        self,
        project_root: Path,
        *,
        materialization_policy: WorkspaceMaterializationPolicy | None = None,
        environment_policy: EnvironmentPolicy | None = None,
        secret_policy: SecretPolicy | None = None,
        resource_policy: ResourcePolicy | None = None,
        transfer_policy: TransferPolicy | None = None,
        network_policy: NetworkPolicy = NetworkPolicy.HOST_INHERITED,
        persistence: WorkspacePersistence = WorkspacePersistence.EPHEMERAL,
        change_disposition: ChangeDisposition = ChangeDisposition.RETURN_CHANGE_SET,
        shell_execution_policy: ShellExecutionPolicy | None = None,
        require_shell_containment: bool = True,
        temporary_root: Path | None = None,
        process_policy: ProcessPolicy | None = None,
    ) -> None:
        resources = resource_policy or ResourcePolicy()
        selected_network_policy = NetworkPolicy(network_policy)
        selected_persistence = WorkspacePersistence(persistence)
        selected_change_disposition = ChangeDisposition(change_disposition)
        if selected_change_disposition is ChangeDisposition.APPLY_DIRECTLY:
            raise ValueError(
                "Temporary workspaces cannot apply changes directly; use apply_change_set with host approval"
            )
        super().__init__(
            project_root,
            shell_timeout_seconds=resources.command_timeout_seconds,
            max_stdout_bytes=resources.max_stdout_bytes,
            max_stderr_bytes=resources.max_stderr_bytes,
            shell_execution_policy=shell_execution_policy,
            require_shell_containment=require_shell_containment,
            process_policy=process_policy,
        )
        self.materialization_policy = (
            materialization_policy or WorkspaceMaterializationPolicy()
        )
        self.environment_policy = environment_policy or EnvironmentPolicy()
        self.secret_policy = secret_policy or SecretPolicy()
        self.resource_policy = resources
        self.transfer_policy = transfer_policy or TransferPolicy()
        self.network_policy = selected_network_policy
        self.persistence = selected_persistence
        self.change_disposition = selected_change_disposition
        self.temporary_root = temporary_root.resolve() if temporary_root else None
        if (
            self.temporary_root is not None
            and (
                self.temporary_root == self.project_root
                or self.project_root in self.temporary_root.parents
            )
        ):
            raise ValueError("temporary_root cannot be inside the source project")
        self._change_sets: dict[str, ChangeSet] = {}
        self._applied_change_sets: set[str] = set()
        self._sessions: dict[str, TemporaryWorkspaceSession] = {}
        self._application_lock = threading.RLock()

    def open_session(self, request: ExecutionSessionRequest) -> ExecutionSession:
        if self._closed:
            raise RuntimeError("Execution backend is closed")
        suffix = _safe_identifier(request.turn_id or uuid4().hex)
        workspace_root = self._materialize_workspace(suffix)
        workspace = ExecutionWorkspace(
            workspace_id=f"{self.name}-{suffix}-{uuid4().hex[:8]}",
            backend_name=self.name,
            mode=self._workspace_mode,
            containment=self._workspace_containment,
        )
        policy = ExecutionPolicy(
            name=f"{self.name}-review",
            network=self.network_policy,
            persistence=self.persistence,
            change_disposition=self.change_disposition,
            require_containment=self.require_shell_containment,
        )
        try:
            base_snapshot = _snapshot_workspace(
                workspace_root,
                materialization_policy=self.materialization_policy,
                resource_policy=self.resource_policy,
                unsafe_secret_action=UnsafePathAction.REJECT,
            )
        except BaseException:
            self._cleanup_materialized_workspace(workspace_root, workspace.workspace_id)
            raise
        shell_policy = _WorkspaceShellPolicy(
            legacy=self.shell_execution_policy,
            environment=self.environment_policy,
            secrets=self.secret_policy,
        )
        try:
            session = self._create_session(
                request=request,
                workspace=workspace,
                policy=policy,
                workspace_root=workspace_root,
                shell_policy=shell_policy,
                base_snapshot=base_snapshot,
            )
        except BaseException:
            self._cleanup_materialized_workspace(
                workspace_root,
                workspace.workspace_id,
            )
            raise
        self._sessions[workspace.workspace_id] = session
        return session

    async def open_session_async(self, request: ExecutionSessionRequest) -> ExecutionSession:
        return await asyncio.to_thread(self.open_session, request)

    def get_change_set(self, change_set_id: str) -> ChangeSet:
        """Return a retained immutable change set by opaque id."""
        try:
            return self._change_sets[change_set_id]
        except KeyError as exc:
            raise KeyError(f"Unknown change set: {change_set_id}") from exc

    def apply_change_set(
        self,
        change_set_id: str,
        *,
        approval: ChangeSetApproval,
    ) -> ChangeApplicationResult:
        """Apply one approved change set to the host with conflicts and rollback."""
        if self._closed:
            raise RuntimeError("Execution backend is closed")
        if approval.change_set_id != change_set_id:
            raise ValueError("Approval does not match the requested change set")
        with self._application_lock:
            if change_set_id in self._applied_change_sets:
                return ChangeApplicationResult(
                    success=False,
                    change_set_id=change_set_id,
                    approved_by=approval.approved_by,
                    reason=approval.reason,
                    error="change_set_already_applied",
                )
            change_set = self.get_change_set(change_set_id)
            conflicts = _preflight_change_set(change_set, self.project_root)
            if conflicts:
                return ChangeApplicationResult(
                    success=False,
                    change_set_id=change_set_id,
                    approved_by=approval.approved_by,
                    reason=approval.reason,
                    conflicts=tuple(conflicts),
                    error="change_conflict",
                )

            backups: list[_ApplyBackup] = []
            created_directories: set[Path] = set()
            applied_paths: list[str] = []
            try:
                for change in change_set.changes:
                    _assert_expected_host_state(change, self.project_root)
                    backup = _capture_backup(change, self.project_root)
                    backups.append(backup)
                    directories = _prepare_change_parent(change, self.project_root)
                    created_directories.update(directories)
                    _apply_change(change, backup)
                    applied_paths.append(change.path)
            except BaseException as exc:
                rollback_errors = _rollback_changes(
                    backups,
                    created_directories,
                    self.project_root,
                )
                if not isinstance(exc, Exception):
                    raise
                return ChangeApplicationResult(
                    success=False,
                    change_set_id=change_set_id,
                    approved_by=approval.approved_by,
                    reason=approval.reason,
                    applied_paths=tuple(applied_paths),
                    rollback_errors=tuple(rollback_errors),
                    error=(
                        "apply_failed_with_rollback_errors"
                        if rollback_errors
                        else f"apply_failed:{type(exc).__name__}"
                    ),
                )

            self._applied_change_sets.add(change_set_id)
            return ChangeApplicationResult(
                success=True,
                change_set_id=change_set_id,
                approved_by=approval.approved_by,
                reason=approval.reason,
                applied_paths=tuple(applied_paths),
            )

    async def apply_change_set_async(
        self,
        change_set_id: str,
        *,
        approval: ChangeSetApproval,
    ) -> ChangeApplicationResult:
        return await asyncio.to_thread(
            self.apply_change_set,
            change_set_id,
            approval=approval,
        )

    def cleanup_workspace(self, workspace_id: str) -> None:
        """Explicitly remove a retained workspace."""
        session = self._sessions.get(workspace_id)
        if session is None:
            raise KeyError(f"Unknown workspace: {workspace_id}")
        session.close()
        session.cleanup()

    def close(self) -> None:
        if self._closed:
            return
        failures: list[Exception] = []
        for session in tuple(self._sessions.values()):
            try:
                session.close()
                session.cleanup()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError(
                f"Failed to clean {len(failures)} temporary workspace(s)"
            ) from failures[0]
        super().close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    @property
    def _workspace_mode(self) -> WorkspaceMode:
        return WorkspaceMode.TEMPORARY

    @property
    def _workspace_containment(self) -> ContainmentStatus:
        return ContainmentStatus.UNCONTAINED

    def _create_session(
        self,
        *,
        request: ExecutionSessionRequest,
        workspace: ExecutionWorkspace,
        policy: ExecutionPolicy,
        workspace_root: Path,
        shell_policy: ShellExecutionPolicy,
        base_snapshot: dict[str, _FileSnapshot],
    ) -> TemporaryWorkspaceSession:
        return TemporaryWorkspaceSession(
            self,
            workspace=workspace,
            policy=policy,
            project_root=workspace_root,
            shell_execution_policy=shell_policy,
            base_snapshot=base_snapshot,
            process_owner_key=owner_key(
                conversation_id=request.conversation_id,
                turn_id=request.turn_id,
                metadata=request.metadata,
                workspace_id=workspace.workspace_id,
            ),
            cleanup_process_owner_on_close=has_child_task_scope(request),
        )

    def _materialize_workspace(self, suffix: str) -> Path:
        root = Path(
            tempfile.mkdtemp(
                prefix=f"chulk-{suffix}-",
                dir=self.temporary_root,
            )
        ).resolve()
        try:
            _copy_allowlisted_workspace(
                self.project_root,
                root,
                materialization_policy=self.materialization_policy,
                resource_policy=self.resource_policy,
            )
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return root

    def _cleanup_materialized_workspace(
        self,
        workspace_root: Path,
        workspace_id: str,
    ) -> None:
        del workspace_id
        shutil.rmtree(workspace_root, ignore_errors=False)

    def _retain_change_set(self, change_set: ChangeSet) -> None:
        if self.change_disposition is ChangeDisposition.RETURN_CHANGE_SET:
            self._change_sets[change_set.change_set_id] = change_set
            while len(self._change_sets) > self.resource_policy.max_retained_change_sets:
                oldest_change_set_id = next(iter(self._change_sets))
                self._change_sets.pop(oldest_change_set_id, None)

    def _session_cleaned(self, workspace_id: str) -> None:
        self._sessions.pop(workspace_id, None)


class GitWorktreeBackend(TemporaryWorkspaceBackend):
    """Materialize reviewable sessions with native Git worktrees."""

    name = "git-worktree"

    def __init__(
        self,
        project_root: Path,
        *,
        git_policy: GitWorktreePolicy | None = None,
        **kwargs: Any,
    ) -> None:
        selected_git_policy = git_policy or GitWorktreePolicy()
        if not selected_git_policy.remove_on_close and "persistence" not in kwargs:
            kwargs["persistence"] = WorkspacePersistence.PERSISTENT
        super().__init__(project_root, **kwargs)
        expected_persistence = (
            WorkspacePersistence.EPHEMERAL
            if selected_git_policy.remove_on_close
            else WorkspacePersistence.PERSISTENT
        )
        if self.persistence is not expected_persistence:
            raise ValueError(
                "Git worktree remove_on_close conflicts with workspace persistence"
            )
        if any(
            _safe_relative_path(path) != Path(".")
            for path in self.materialization_policy.allowed_paths
        ):
            raise WorkspacePolicyError(
                "Git worktrees currently require an allowlist containing the repository root",
                code="git_narrow_allowlist_unsupported",
            )
        self.git_policy = selected_git_policy
        self._worktree_branches: dict[Path, str | None] = {}
        self._worktree_parents: dict[Path, Path] = {}

    @property
    def _workspace_mode(self) -> WorkspaceMode:
        return WorkspaceMode.GIT_WORKTREE

    def _materialize_workspace(self, suffix: str) -> Path:
        repository_root = _git_output(
            self.project_root,
            "rev-parse",
            "--show-toplevel",
        )
        if Path(repository_root).resolve() != self.project_root:
            raise WorkspacePolicyError(
                "Git-worktree execution requires project_root to be the repository root",
                code="git_repository_root_required",
            )
        _assert_repository_owned(self.project_root)
        if self.git_policy.require_clean:
            dirty = _git_output(
                self.project_root,
                "status",
                "--porcelain",
                "--untracked-files=all",
            )
            if dirty:
                raise WorkspacePolicyError(
                    "Git-worktree execution requires a clean source repository",
                    code="dirty_git_workspace",
                )

        parent = Path(
            tempfile.mkdtemp(
                prefix=f"chulk-git-{suffix}-",
                dir=self.temporary_root,
            )
        ).resolve()
        workspace_root = parent / "workspace"
        branch_name: str | None = None
        command = ["worktree", "add"]
        if self.git_policy.detached:
            command.append("--detach")
        else:
            branch_name = (
                f"{self.git_policy.branch_prefix}/"
                f"{suffix}-{uuid4().hex[:8]}"
            )
            _run_git(
                self.project_root,
                "check-ref-format",
                "--branch",
                branch_name,
            )
            command.extend(["-b", branch_name])
        command.extend([str(workspace_root), "HEAD"])
        try:
            _run_git(self.project_root, *command)
        except BaseException:
            shutil.rmtree(parent, ignore_errors=True)
            raise
        self._worktree_branches[workspace_root.resolve()] = branch_name
        self._worktree_parents[workspace_root.resolve()] = parent
        return workspace_root.resolve()

    def _cleanup_materialized_workspace(
        self,
        workspace_root: Path,
        workspace_id: str,
    ) -> None:
        del workspace_id
        root = workspace_root.resolve()
        branch_name = self._worktree_branches.get(root)
        parent = self._worktree_parents.get(root, root.parent)
        failures: list[WorkspacePolicyError] = []
        if root.exists():
            try:
                _run_git(
                    self.project_root,
                    "worktree",
                    "remove",
                    "--force",
                    str(root),
                )
            except WorkspacePolicyError:
                shutil.rmtree(root, ignore_errors=True)
                try:
                    _run_git(self.project_root, "worktree", "prune")
                except WorkspacePolicyError as exc:
                    failures.append(exc)
        else:
            try:
                _run_git(self.project_root, "worktree", "prune")
            except WorkspacePolicyError as exc:
                failures.append(exc)
        try:
            if branch_name and self.git_policy.delete_branch_on_close:
                _run_git(self.project_root, "branch", "-D", branch_name)
        except WorkspacePolicyError as exc:
            failures.append(exc)
        finally:
            shutil.rmtree(parent, ignore_errors=True)
        if failures:
            raise WorkspacePolicyError(
                f"Failed to clean Git worktree: {failures[0]}",
                code="git_worktree_cleanup_failed",
            ) from failures[0]
        self._worktree_branches.pop(root, None)
        self._worktree_parents.pop(root, None)


class TemporaryWorkspaceSession(HostExecutionSession):
    """Host-compatible execution session rooted in a reviewable workspace."""

    def __init__(
        self,
        backend: TemporaryWorkspaceBackend,
        *,
        workspace: ExecutionWorkspace,
        policy: ExecutionPolicy,
        project_root: Path,
        shell_execution_policy: ShellExecutionPolicy,
        base_snapshot: dict[str, _FileSnapshot],
        process_owner_key: str,
        cleanup_process_owner_on_close: bool = False,
    ) -> None:
        super().__init__(
            backend,
            workspace=workspace,
            policy=policy,
            project_root=project_root,
            shell_execution_policy=shell_execution_policy,
            process_owner_key=process_owner_key,
            cleanup_process_owner_on_close=cleanup_process_owner_on_close,
        )
        self._temporary_backend = backend
        self._base_snapshot = base_snapshot
        self._latest_change_set: ChangeSet | None = None
        self._cleaned = False

    @property
    def latest_change_set(self) -> ChangeSet | None:
        return self._latest_change_set

    def write_file(self, request: FileWriteRequest) -> ExecutionResult:
        return self._with_change_set(super().write_file(request))

    def apply_patch(self, request: PatchApplyRequest) -> ExecutionResult:
        return self._with_change_set(super().apply_patch(request))

    def run_command(self, request: CommandExecutionRequest) -> ExecutionResult:
        if self.policy.network is NetworkPolicy.DENY:
            return self._normalize(
                ToolResult(
                    tool_name="run_cmd",
                    success=False,
                    observation=(
                        "The temporary backend cannot enforce network denial for host processes; "
                        "select a contained backend instead."
                    ),
                    error="network_policy_unsupported",
                    failure_kind=ToolFailureKind.FATAL_SAFETY,
                    metadata={"child_process_started": False},
                )
            )
        return self._with_change_set(super().run_command(request))

    def start_process(self, request: ProcessStartRequest) -> ExecutionResult:
        if self.policy.network is NetworkPolicy.DENY:
            return self._normalize(
                ToolResult(
                    tool_name="process.start",
                    success=False,
                    observation=(
                        "The temporary backend cannot enforce network denial for host "
                        "processes; select a contained backend instead."
                    ),
                    error="network_policy_unsupported",
                    failure_kind=ToolFailureKind.FATAL_SAFETY,
                    metadata={"child_process_started": False},
                )
            )
        return super().start_process(request)

    def _normalize(
        self,
        result: ToolResult,
        *,
        include_changes: bool = False,
    ) -> ExecutionResult:
        normalized = super()._normalize(
            result,
            include_changes=include_changes,
        )
        backend = self._temporary_backend
        return replace(
            normalized,
            metadata={
                **normalized.metadata,
                "workspace_policy": {
                    "allowed_paths": list(
                        backend.materialization_policy.allowed_paths
                    ),
                    "ignored_action": (
                        backend.materialization_policy.ignored_action.value
                    ),
                    "secret_action": (
                        backend.materialization_policy.secret_action.value
                    ),
                },
                "environment_policy": {
                    "inherit_all": backend.environment_policy.inherit_all,
                    "allowed_names": list(
                        backend.environment_policy.allowed_names
                    ),
                },
                "secret_policy": {
                    "allowed_environment_names": list(
                        backend.secret_policy.allowed_environment_names
                    ),
                },
                "resource_policy": asdict(backend.resource_policy),
                "transfer_policy": asdict(backend.transfer_policy),
            },
        )

    def close(self) -> None:
        if self.closed:
            return
        try:
            try:
                self._publish_change_set()
            except WorkspacePolicyError:
                pass
        finally:
            super().close()
            if self.policy.persistence is WorkspacePersistence.EPHEMERAL:
                self.cleanup()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._temporary_backend.process_registry.cleanup_workspace(
            self.workspace.workspace_id
        )
        self._temporary_backend._cleanup_materialized_workspace(
            self.project_root,
            self.workspace.workspace_id,
        )
        self._temporary_backend._session_cleaned(self.workspace.workspace_id)
        self._cleaned = True

    def _with_change_set(self, result: ExecutionResult) -> ExecutionResult:
        try:
            change_set = self._publish_change_set()
        except WorkspacePolicyError as exc:
            metadata = {
                **result.metadata,
                "change_set_error": exc.code,
                "change_set_error_path": exc.path,
            }
            return replace(
                result,
                success=False,
                observation=(
                    f"{result.observation}\nWorkspace changes cannot be reviewed safely: {exc}"
                ),
                error=exc.code,
                failure_kind=ToolFailureKind.FATAL_SAFETY,
                metadata=metadata,
                change_set=None,
            )
        if change_set is None:
            return replace(result, change_set=None)
        return replace(
            result,
            metadata={
                **result.metadata,
                "change_set": _change_set_metadata(change_set),
            },
            change_set=change_set,
        )

    def _publish_change_set(self) -> ChangeSet | None:
        change_set = _build_change_set(
            workspace_id=self.workspace.workspace_id,
            backend_name=self.workspace.backend_name,
            workspace_root=self.project_root,
            base_snapshot=self._base_snapshot,
            materialization_policy=self._temporary_backend.materialization_policy,
            resource_policy=self._temporary_backend.resource_policy,
        )
        if (
            change_set is not None
            and self.policy.change_disposition is ChangeDisposition.DISCARD
        ):
            self._latest_change_set = None
            return None
        self._latest_change_set = change_set
        if change_set is not None:
            self._temporary_backend._retain_change_set(change_set)
        return change_set


class _WorkspaceShellPolicy:
    """Preserve legacy command decisions while enforcing environment allowlists."""

    def __init__(
        self,
        *,
        legacy: ShellExecutionPolicy | None,
        environment: EnvironmentPolicy,
        secrets: SecretPolicy,
    ) -> None:
        self.legacy = legacy or DirectShellExecutionPolicy()
        self.environment = environment
        self.secrets = secrets

    def prepare(self, request: ShellExecutionRequest) -> ShellExecutionDecision:
        decision = self.legacy.prepare(request)
        if decision.command is None:
            return decision
        source_environment: Mapping[str, str] = (
            decision.environment if decision.environment is not None else os.environ
        )
        environment = self.environment.build(source_environment)
        allowed_secrets = set(self.secrets.allowed_environment_names)
        environment = {
            name: value
            for name, value in environment.items()
            if not _looks_secret_like_environment_name(name)
            or name in allowed_secrets
        }
        return replace(decision, environment=environment)


def _copy_allowlisted_workspace(
    source_root: Path,
    destination_root: Path,
    *,
    materialization_policy: WorkspaceMaterializationPolicy,
    resource_policy: ResourcePolicy,
) -> None:
    counters = {"files": 0, "bytes": 0}
    copied: set[str] = set()
    for raw_path in materialization_policy.allowed_paths:
        relative = _safe_relative_path(raw_path)
        source = _safe_source_path(source_root, relative)
        destination = (
            destination_root
            if relative == Path(".")
            else destination_root / relative
        )
        if not source.exists() and not source.is_symlink():
            raise WorkspacePolicyError(
                f"Allowlisted workspace path does not exist: {raw_path}",
                code="allowlisted_path_missing",
                path=raw_path,
            )
        _copy_entry(
            source,
            destination,
            relative=Path(".") if relative == Path(".") else relative,
            source_root=source_root,
            materialization_policy=materialization_policy,
            resource_policy=resource_policy,
            counters=counters,
            copied=copied,
        )


def _copy_entry(
    source: Path,
    destination: Path,
    *,
    relative: Path,
    source_root: Path,
    materialization_policy: WorkspaceMaterializationPolicy,
    resource_policy: ResourcePolicy,
    counters: dict[str, int],
    copied: set[str],
) -> None:
    relative_text = relative.as_posix()
    if relative_text in copied:
        return
    if _is_ignored(relative, materialization_policy):
        _handle_unsafe(
            materialization_policy.ignored_action,
            code="ignored_workspace_path",
            path=relative_text,
            message=f"Ignored runtime path is not allowed in the workspace: {relative_text}",
        )
        return
    info = source.lstat()
    if stat.S_ISLNK(info.st_mode):
        _handle_unsafe(
            materialization_policy.symlink_action,
            code="workspace_symlink",
            path=relative_text,
            message=f"Workspace symlink is not allowed: {relative_text}",
        )
        return
    if stat.S_ISDIR(info.st_mode):
        destination.mkdir(parents=True, exist_ok=True)
        copied.add(relative_text)
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            child_relative = (
                Path(child.name)
                if relative == Path(".")
                else relative / child.name
            )
            _copy_entry(
                child,
                destination / child.name,
                relative=child_relative,
                source_root=source_root,
                materialization_policy=materialization_policy,
                resource_policy=resource_policy,
                counters=counters,
                copied=copied,
            )
        return
    if not stat.S_ISREG(info.st_mode):
        _handle_unsafe(
            materialization_policy.special_file_action,
            code="workspace_special_file",
            path=relative_text,
            message=f"Special workspace file is not allowed: {relative_text}",
        )
        return
    if info.st_nlink > 1:
        _handle_unsafe(
            materialization_policy.hardlink_action,
            code="workspace_hardlink",
            path=relative_text,
            message=f"Hard-linked workspace file is not allowed: {relative_text}",
        )
        return
    secret_error = safe_read_error(
        source,
        source_root,
        FileReadPolicy(),
        requested_path=relative_text,
    )
    if secret_error:
        _handle_unsafe(
            materialization_policy.secret_action,
            code="workspace_secret",
            path=relative_text,
            message=f"Sensitive workspace file is not allowed: {relative_text}",
        )
        return
    try:
        data, mode = _read_regular_file(source_root, relative)
    except OSError as exc:
        raise WorkspacePolicyError(
            f"Workspace file cannot be copied safely: {relative_text}",
            code="workspace_copy_race",
            path=relative_text,
        ) from exc
    if len(data) > resource_policy.max_file_bytes:
        _handle_unsafe(
            materialization_policy.large_file_action,
            code="workspace_file_too_large",
            path=relative_text,
            message=f"Workspace file exceeds the configured limit: {relative_text}",
        )
        return
    counters["files"] += 1
    counters["bytes"] += len(data)
    if counters["files"] > resource_policy.max_files:
        raise WorkspacePolicyError(
            "Workspace exceeds the configured file-count limit",
            code="workspace_file_limit",
        )
    if counters["bytes"] > resource_policy.max_total_bytes:
        raise WorkspacePolicyError(
            "Workspace exceeds the configured total-byte limit",
            code="workspace_byte_limit",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data)
    destination.chmod(mode)
    copied.add(relative_text)


def _snapshot_workspace(
    workspace_root: Path,
    *,
    materialization_policy: WorkspaceMaterializationPolicy,
    resource_policy: ResourcePolicy,
    unsafe_secret_action: UnsafePathAction,
) -> dict[str, _FileSnapshot]:
    snapshots: dict[str, _FileSnapshot] = {}
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(
        workspace_root,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        safe_directories: list[str] = []
        for name in sorted(directory_names):
            path = directory_path / name
            relative = path.relative_to(workspace_root)
            if _is_ignored(relative, materialization_policy):
                _handle_unsafe(
                    materialization_policy.ignored_action,
                    code="ignored_workspace_path",
                    path=relative.as_posix(),
                    message=(
                        "Ignored runtime path is not allowed in the workspace: "
                        f"{relative.as_posix()}"
                    ),
                )
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                _handle_unsafe(
                    materialization_policy.symlink_action,
                    code="workspace_symlink",
                    path=relative.as_posix(),
                    message=f"Workspace symlink is not allowed: {relative.as_posix()}",
                )
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise WorkspacePolicyError(
                    f"Special workspace entry is not allowed: {relative.as_posix()}",
                    code="workspace_special_file",
                    path=relative.as_posix(),
                )
            safe_directories.append(name)
        directory_names[:] = safe_directories

        for name in sorted(file_names):
            path = directory_path / name
            relative = path.relative_to(workspace_root)
            relative_text = relative.as_posix()
            if _is_ignored(relative, materialization_policy):
                _handle_unsafe(
                    materialization_policy.ignored_action,
                    code="ignored_workspace_path",
                    path=relative_text,
                    message=(
                        "Ignored runtime path is not allowed in the workspace: "
                        f"{relative_text}"
                    ),
                )
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                _handle_unsafe(
                    materialization_policy.symlink_action,
                    code="workspace_symlink",
                    path=relative_text,
                    message=f"Workspace symlink is not allowed: {relative_text}",
                )
                continue
            if not stat.S_ISREG(info.st_mode):
                raise WorkspacePolicyError(
                    f"Special workspace file is not allowed: {relative_text}",
                    code="workspace_special_file",
                    path=relative_text,
                )
            if info.st_nlink > 1:
                raise WorkspacePolicyError(
                    f"Hard-linked workspace file is not allowed: {relative_text}",
                    code="workspace_hardlink",
                    path=relative_text,
                )
            if safe_read_error(
                path,
                workspace_root,
                FileReadPolicy(),
                requested_path=relative_text,
            ):
                _handle_unsafe(
                    unsafe_secret_action,
                    code="workspace_secret",
                    path=relative_text,
                    message=f"Sensitive workspace file is not allowed: {relative_text}",
                )
                continue
            if info.st_size > resource_policy.max_file_bytes:
                raise WorkspacePolicyError(
                    f"Workspace file exceeds the configured limit: {relative_text}",
                    code="workspace_file_too_large",
                    path=relative_text,
                )
            try:
                data, mode = _read_regular_file(
                    workspace_root,
                    relative,
                )
            except OSError as exc:
                raise WorkspacePolicyError(
                    f"Workspace file changed during review: {relative_text}",
                    code="workspace_snapshot_race",
                    path=relative_text,
                ) from exc
            total_bytes += len(data)
            if len(snapshots) + 1 > resource_policy.max_files:
                raise WorkspacePolicyError(
                    "Workspace exceeds the configured file-count limit",
                    code="workspace_file_limit",
                )
            if total_bytes > resource_policy.max_total_bytes:
                raise WorkspacePolicyError(
                    "Workspace exceeds the configured total-byte limit",
                    code="workspace_byte_limit",
                )
            snapshots[relative_text] = _snapshot(data, mode)
    return snapshots


def _build_change_set(
    *,
    workspace_id: str,
    backend_name: str,
    workspace_root: Path,
    base_snapshot: dict[str, _FileSnapshot],
    materialization_policy: WorkspaceMaterializationPolicy,
    resource_policy: ResourcePolicy,
) -> ChangeSet | None:
    current = _snapshot_workspace(
        workspace_root,
        materialization_policy=materialization_policy,
        resource_policy=resource_policy,
        unsafe_secret_action=UnsafePathAction.REJECT,
    )
    changes: list[ChangeRecord] = []
    patch_parts: list[str] = []
    total_bytes = 0
    for path in sorted(set(base_snapshot) | set(current)):
        before = base_snapshot.get(path)
        after = current.get(path)
        if (
            before is not None
            and after is not None
            and before.sha256 == after.sha256
            and before.mode == after.mode
        ):
            continue
        if before is None:
            status = "created"
        elif after is None:
            status = "deleted"
        else:
            status = "modified"
        content_after = after.text if after is not None else None
        content_after_base64 = (
            base64.b64encode(after.data).decode("ascii")
            if after is not None and after.text is None
            else None
        )
        patch_parts.append(_file_patch(path, before, after))
        total_bytes += len(after.data) if after is not None else 0
        changes.append(
            ChangeRecord(
                path=path,
                status=status,
                sha256_before=before.sha256 if before is not None else None,
                sha256_after=after.sha256 if after is not None else None,
                mode_before=before.mode if before is not None else None,
                mode_after=after.mode if after is not None else None,
                content_after=content_after,
                content_after_base64=content_after_base64,
            )
        )
    if not changes:
        return None
    if len(changes) > resource_policy.max_change_files:
        raise WorkspacePolicyError(
            "Change set exceeds the configured changed-file limit",
            code="change_set_file_limit",
        )
    full_patch = "".join(patch_parts)
    patch, truncated = _bounded_patch(full_patch, resource_policy.max_patch_bytes)
    identity = {
        "workspace_id": workspace_id,
        "changes": [
            {
                "path": change.path,
                "status": change.status,
                "before": change.sha256_before,
                "after": change.sha256_after,
                "mode_before": change.mode_before,
                "mode_after": change.mode_after,
            }
            for change in changes
        ],
    }
    change_set_id = "changes-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return ChangeSet(
        changes=tuple(changes),
        change_set_id=change_set_id,
        workspace_id=workspace_id,
        backend_name=backend_name,
        patch=patch,
        patch_truncated=truncated,
        total_bytes=total_bytes,
    )


def _change_set_metadata(change_set: ChangeSet) -> dict[str, Any]:
    return {
        "change_set_id": change_set.change_set_id,
        "workspace_id": change_set.workspace_id,
        "backend_name": change_set.backend_name,
        "changes": [
            {
                "path": change.path,
                "status": change.status,
                "sha256_before": change.sha256_before,
                "sha256_after": change.sha256_after,
                "mode_before": change.mode_before,
                "mode_after": change.mode_after,
            }
            for change in change_set.changes
        ],
        "changed_count": len(change_set.changes),
        "patch": change_set.patch,
        "patch_truncated": change_set.patch_truncated,
        "total_bytes": change_set.total_bytes,
        "requires_host_approval": True,
    }


def _preflight_change_set(change_set: ChangeSet, host_root: Path) -> list[str]:
    conflicts: list[str] = []
    seen: set[str] = set()
    for change in change_set.changes:
        if change.status not in {"created", "modified", "deleted"}:
            conflicts.append(f"{change.path}:invalid_status")
            continue
        if change.status == "created" and change.sha256_before is not None:
            conflicts.append(f"{change.path}:invalid_created_hash")
            continue
        if change.status == "created" and change.mode_before is not None:
            conflicts.append(f"{change.path}:invalid_created_mode")
            continue
        if change.status in {"modified", "deleted"} and change.sha256_before is None:
            conflicts.append(f"{change.path}:missing_base_hash")
            continue
        if change.status in {"modified", "deleted"} and change.mode_before is None:
            conflicts.append(f"{change.path}:missing_base_mode")
            continue
        if change.status == "deleted":
            if change.sha256_after is not None:
                conflicts.append(f"{change.path}:invalid_deleted_hash")
                continue
            if change.mode_after is not None:
                conflicts.append(f"{change.path}:invalid_deleted_mode")
                continue
        else:
            if not _valid_file_mode(change.mode_after):
                conflicts.append(f"{change.path}:invalid_mode")
                continue
            try:
                content = _change_content(change)
            except (WorkspacePolicyError, ValueError):
                conflicts.append(f"{change.path}:invalid_content")
                continue
            if hashlib.sha256(content).hexdigest() != change.sha256_after:
                conflicts.append(f"{change.path}:content_hash_mismatch")
                continue
        try:
            relative = _safe_relative_path(change.path)
        except WorkspacePolicyError:
            conflicts.append(f"{change.path}:unsafe_path")
            continue
        normalized = relative.as_posix()
        if normalized in seen:
            conflicts.append(f"{change.path}:duplicate_path")
            continue
        seen.add(normalized)
        target = host_root / relative
        safety_error = safe_write_error(target, host_root)
        if safety_error:
            conflicts.append(f"{change.path}:unsafe_target")
            continue
        try:
            _assert_expected_host_state(change, host_root)
        except WorkspacePolicyError as exc:
            conflicts.append(f"{change.path}:{exc.code}")
    return conflicts


def _assert_expected_host_state(change: ChangeRecord, host_root: Path) -> None:
    relative = _safe_relative_path(change.path)
    try:
        current_data, current_mode = _read_regular_file(host_root, relative)
    except FileNotFoundError:
        if change.sha256_before is not None:
            raise WorkspacePolicyError(
                f"Host file disappeared before apply: {change.path}",
                code="missing_host_file",
                path=change.path,
            )
        return
    except OSError as exc:
        raise WorkspacePolicyError(
            f"Host path cannot be opened safely: {change.path}",
            code="unsafe_host_path",
            path=change.path,
        ) from exc
    if change.sha256_before is None:
        raise WorkspacePolicyError(
            f"Host path now exists: {change.path}",
            code="host_path_exists",
            path=change.path,
        )
    current_hash = hashlib.sha256(current_data).hexdigest()
    if current_hash != change.sha256_before:
        raise WorkspacePolicyError(
            f"Host file changed since workspace creation: {change.path}",
            code="host_content_changed",
            path=change.path,
        )
    if current_mode != change.mode_before:
        raise WorkspacePolicyError(
            f"Host file mode changed since workspace creation: {change.path}",
            code="host_mode_changed",
            path=change.path,
        )


def _capture_backup(
    change: ChangeRecord,
    host_root: Path,
) -> _ApplyBackup:
    relative = _safe_relative_path(change.path)
    target = host_root / relative
    try:
        data, mode = _read_regular_file(host_root, relative)
        existed = True
    except FileNotFoundError:
        data = None
        mode = None
        existed = False
    return _ApplyBackup(
        host_root=host_root,
        path=target,
        relative_path=change.path,
        existed=existed,
        data=data,
        mode=mode,
    )


def _prepare_change_parent(
    change: ChangeRecord,
    host_root: Path,
) -> set[Path]:
    if change.status == "deleted":
        return set()
    target = host_root / _safe_relative_path(change.path)
    return _create_safe_directories(target.parent, host_root)


def _apply_change(change: ChangeRecord, backup: _ApplyBackup) -> None:
    relative = _safe_relative_path(change.path)
    if change.status == "deleted":
        _unlink_regular_file(backup.host_root, relative)
        return
    _atomic_write_relative(
        backup.host_root,
        relative,
        _change_content(change),
        mode=change.mode_after,
    )


def _rollback_changes(
    backups: list[_ApplyBackup],
    created_directories: set[Path],
    host_root: Path,
) -> list[str]:
    errors: list[str] = []
    for backup in reversed(backups):
        try:
            relative = _safe_relative_path(backup.relative_path)
            if backup.existed:
                _atomic_write_relative(
                    host_root,
                    relative,
                    backup.data or b"",
                    mode=backup.mode,
                )
            else:
                try:
                    _unlink_regular_file(host_root, relative)
                except FileNotFoundError:
                    pass
        except BaseException as exc:
            errors.append(f"{backup.relative_path}:{type(exc).__name__}")
    for directory in sorted(created_directories, key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            continue
    return errors


def _create_safe_directories(parent: Path, host_root: Path) -> set[Path]:
    relative = parent.relative_to(host_root)
    current = host_root
    created: set[Path] = set()
    for part in relative.parts:
        current = current / part
        if current.exists():
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise WorkspacePolicyError(
                    f"Unsafe host directory component: {current}",
                    code="unsafe_host_parent",
                )
            continue
        current.mkdir()
        created.add(current)
    return created


def _assert_no_symlink_parents(host_root: Path, relative_parent: Path) -> None:
    current = host_root
    for part in relative_parent.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if not current.exists() and not current.is_symlink():
            continue
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise WorkspacePolicyError(
                f"Unsafe host directory component: {current}",
                code="unsafe_host_parent",
            )


def _atomic_write_relative(
    host_root: Path,
    relative: Path,
    data: bytes,
    *,
    mode: int | None,
) -> None:
    path = host_root / relative
    if _supports_secure_dir_fd():
        with _open_directory_fd(host_root, relative.parent) as directory_fd:
            temporary_name = f".chulk-apply-{uuid4().hex}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                temporary_name,
                flags,
                mode if mode is not None else 0o666,
                dir_fd=directory_fd,
            )
            try:
                if mode is not None:
                    os.fchmod(descriptor, mode)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.rename(
                    temporary_name,
                    relative.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            except BaseException:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
                raise
        return

    _assert_no_symlink_parents(host_root, relative.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".chulk-apply-",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary_path.chmod(mode)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_regular_file(host_root: Path, relative: Path) -> tuple[bytes, int]:
    if _supports_secure_dir_fd():
        with _open_directory_fd(host_root, relative.parent) as directory_fd:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(relative.name, flags, dir_fd=directory_fd)
            try:
                info = os.fstat(descriptor)
                _assert_regular_host_info(info, relative.as_posix())
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 64 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks), stat.S_IMODE(info.st_mode)
            finally:
                os.close(descriptor)

    path = host_root / relative
    _assert_no_symlink_parents(host_root, relative.parent)
    info = path.lstat()
    _assert_regular_host_info(info, relative.as_posix())
    return path.read_bytes(), stat.S_IMODE(info.st_mode)


def _unlink_regular_file(host_root: Path, relative: Path) -> None:
    if _supports_secure_dir_fd():
        with _open_directory_fd(host_root, relative.parent) as directory_fd:
            info = os.stat(relative.name, dir_fd=directory_fd, follow_symlinks=False)
            _assert_regular_host_info(info, relative.as_posix())
            os.unlink(relative.name, dir_fd=directory_fd)
        return

    path = host_root / relative
    _assert_no_symlink_parents(host_root, relative.parent)
    info = path.lstat()
    _assert_regular_host_info(info, relative.as_posix())
    path.unlink()


def _assert_regular_host_info(info: os.stat_result, path: str) -> None:
    if stat.S_ISLNK(info.st_mode):
        raise WorkspacePolicyError(
            f"Host target became a symlink: {path}",
            code="host_symlink",
            path=path,
        )
    if not stat.S_ISREG(info.st_mode):
        raise WorkspacePolicyError(
            f"Host target is not a regular file: {path}",
            code="host_special_file",
            path=path,
        )
    if info.st_nlink > 1:
        raise WorkspacePolicyError(
            f"Host target is hard-linked: {path}",
            code="host_hardlink",
            path=path,
        )


def _supports_secure_dir_fd() -> bool:
    return (
        os.name == "posix"
        and os.open in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and hasattr(os, "O_NOFOLLOW")
    )


@contextmanager
def _open_directory_fd(host_root: Path, relative: Path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = [os.open(host_root, flags)]
    try:
        for part in relative.parts:
            if part in {"", "."}:
                continue
            descriptors.append(
                os.open(part, flags, dir_fd=descriptors[-1])
            )
        yield descriptors[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _change_content(change: ChangeRecord) -> bytes:
    if change.content_after is not None:
        return change.content_after.encode("utf-8")
    if change.content_after_base64 is not None:
        return base64.b64decode(change.content_after_base64, validate=True)
    raise WorkspacePolicyError(
        f"Change set has no content for {change.path}",
        code="change_content_missing",
        path=change.path,
    )


def _snapshot(data: bytes, mode: int) -> _FileSnapshot:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    return _FileSnapshot(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        text=text,
        mode=mode,
    )


def _file_patch(
    path: str,
    before: _FileSnapshot | None,
    after: _FileSnapshot | None,
) -> str:
    if (before is not None and before.text is None) or (
        after is not None and after.text is None
    ):
        return f"Binary files a/{path} and b/{path} differ\n"
    before_text = before.text or "" if before is not None else ""
    after_text = after.text or "" if after is not None else ""
    fromfile = f"a/{path}" if before is not None else "/dev/null"
    tofile = f"b/{path}" if after is not None else "/dev/null"
    return "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def _bounded_patch(patch: str, max_bytes: int) -> tuple[str, bool]:
    encoded = patch.encode("utf-8")
    if len(encoded) <= max_bytes:
        return patch, False
    marker = b"\n... change-set patch truncated ...\n"
    if max_bytes <= len(marker):
        return marker[:max_bytes].decode("ascii"), True
    available = max(0, max_bytes - len(marker))
    head_size = (available + 1) // 2
    tail_size = available - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = (
        encoded[-tail_size:].decode("utf-8", errors="ignore")
        if tail_size
        else ""
    )
    return head + marker.decode("ascii") + tail, True


def _valid_file_mode(mode: int | None) -> bool:
    return (
        mode is not None
        and not isinstance(mode, bool)
        and isinstance(mode, int)
        and 0 <= mode <= 0o7777
    )


def _safe_relative_path(raw_path: str) -> Path:
    if "\x00" in raw_path:
        raise WorkspacePolicyError(
            "Workspace path cannot contain NUL bytes",
            code="unsafe_workspace_path",
            path=raw_path,
        )
    candidate = PurePath(raw_path)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise WorkspacePolicyError(
            f"Unsafe workspace path: {raw_path}",
            code="unsafe_workspace_path",
            path=raw_path,
        )
    normalized_parts = tuple(part for part in candidate.parts if part not in {"", "."})
    return Path(*normalized_parts) if normalized_parts else Path(".")


def _safe_source_path(source_root: Path, relative: Path) -> Path:
    current = source_root
    if relative == Path("."):
        return current
    for index, part in enumerate(relative.parts):
        current = current / part
        if not current.exists() and not current.is_symlink():
            return current
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise WorkspacePolicyError(
                f"Allowlisted workspace path crosses a symlink: {relative.as_posix()}",
                code="workspace_symlink",
                path=relative.as_posix(),
            )
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise WorkspacePolicyError(
                f"Allowlisted workspace path crosses a non-directory: {relative.as_posix()}",
                code="workspace_path_component",
                path=relative.as_posix(),
            )
    return current


def _is_ignored(
    relative: Path,
    policy: WorkspaceMaterializationPolicy,
) -> bool:
    ignored = {name.lower() for name in policy.ignored_names}
    return any(part.lower() in ignored for part in relative.parts)


def _handle_unsafe(
    action: UnsafePathAction,
    *,
    code: str,
    path: str,
    message: str,
) -> None:
    if action is UnsafePathAction.REJECT:
        raise WorkspacePolicyError(message, code=code, path=path)


def _looks_secret_like_environment_name(name: str) -> bool:
    normalized = name.lower()
    return any(marker in normalized for marker in _SECRET_NAME_MARKERS)


def _safe_identifier(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value
    ).strip("-")
    return normalized[:80] or uuid4().hex


def _git_output(repository: Path, *arguments: str) -> str:
    completed = _run_git(repository, *arguments)
    return completed.stdout.strip()


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspacePolicyError(
            f"Git worktree operation failed: {type(exc).__name__}",
            code="git_worktree_unavailable",
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Git error"
        raise WorkspacePolicyError(
            f"Git worktree operation failed: {detail}",
            code="git_worktree_failed",
        )
    return completed


def _assert_repository_owned(repository: Path) -> None:
    if os.name != "posix" or not hasattr(os, "getuid"):
        return
    if repository.stat().st_uid != os.getuid():
        raise WorkspacePolicyError(
            "Git repository is not owned by the current process user",
            code="git_repository_ownership",
        )
