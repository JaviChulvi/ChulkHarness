"""Direct host execution backend preserving the existing tool behavior."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from chulk.execution.base import ExecutionBackend, ExecutionSession
from chulk.execution.models import (
    ChangeRecord,
    ChangeSet,
    CommandExecutionRequest,
    ContainmentStatus,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSessionRequest,
    ExecutionWorkspace,
    FileListRequest,
    FileReadRequest,
    FileSearchRequest,
    FileWriteRequest,
    NetworkPolicy,
    PatchApplyRequest,
    ProcessLogsRequest,
    ProcessPollRequest,
    ProcessStartRequest,
    ProcessTerminateRequest,
    ProcessWriteRequest,
    WorkspaceMode,
)
from chulk.execution.policy import ProcessPolicy
from chulk.execution.processes import (
    ManagedProcessRegistry,
    owner_key,
    prepare_host_process,
)
from chulk.tools.registry import ToolExecutionContext, ToolResult
from chulk.tools.shell import (
    DEFAULT_SHELL_STDERR_LIMIT_BYTES,
    DEFAULT_SHELL_STDOUT_LIMIT_BYTES,
    ShellExecutionPolicy,
)


class HostExecutionBackend:
    """Execute directly in the configured project root without containment."""

    name = "host"

    def __init__(
        self,
        project_root: Path,
        *,
        shell_timeout_seconds: int = 10,
        max_stdout_bytes: int = DEFAULT_SHELL_STDOUT_LIMIT_BYTES,
        max_stderr_bytes: int = DEFAULT_SHELL_STDERR_LIMIT_BYTES,
        shell_execution_policy: ShellExecutionPolicy | None = None,
        require_shell_containment: bool = False,
        allow_sensitive_reads: bool = False,
        process_policy: ProcessPolicy | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.shell_timeout_seconds = shell_timeout_seconds
        self.max_stdout_bytes = max_stdout_bytes
        self.max_stderr_bytes = max_stderr_bytes
        self.shell_execution_policy = shell_execution_policy
        self.require_shell_containment = require_shell_containment
        self.allow_sensitive_reads = allow_sensitive_reads
        self.process_registry = ManagedProcessRegistry(process_policy)
        self._closed = False

    def open_session(self, request: ExecutionSessionRequest) -> ExecutionSession:
        if self._closed:
            raise RuntimeError("Execution backend is closed")
        suffix = request.turn_id or uuid4().hex
        workspace = ExecutionWorkspace(
            workspace_id=f"host-{suffix}",
            backend_name=self.name,
            mode=WorkspaceMode.HOST,
            containment=ContainmentStatus.UNCONTAINED,
        )
        policy = ExecutionPolicy(
            name="host-direct",
            network=NetworkPolicy.HOST_INHERITED,
            require_containment=self.require_shell_containment,
        )
        return HostExecutionSession(
            self,
            workspace=workspace,
            policy=policy,
            process_owner_key=owner_key(
                conversation_id=request.conversation_id,
                turn_id=request.turn_id,
                metadata=request.metadata,
                workspace_id=workspace.workspace_id,
            ),
            cleanup_process_owner_on_close=has_child_task_scope(request),
        )

    async def open_session_async(self, request: ExecutionSessionRequest) -> ExecutionSession:
        return self.open_session(request)

    def close(self) -> None:
        if self._closed:
            return
        self.process_registry.close()
        self._closed = True

    async def aclose(self) -> None:
        self.close()


class HostExecutionSession:
    """Turn-scoped view of direct host file and command operations."""

    def __init__(
        self,
        backend: HostExecutionBackend,
        *,
        workspace: ExecutionWorkspace,
        policy: ExecutionPolicy,
        project_root: Path | None = None,
        shell_execution_policy: ShellExecutionPolicy | None = None,
        process_owner_key: str | None = None,
        cleanup_process_owner_on_close: bool = False,
    ) -> None:
        self._backend = backend
        self.workspace = workspace
        self.policy = policy
        self.project_root = (project_root or backend.project_root).resolve()
        self.shell_execution_policy = (
            shell_execution_policy
            if shell_execution_policy is not None
            else backend.shell_execution_policy
        )
        self.process_owner_key = process_owner_key or f"workspace:{workspace.workspace_id}"
        self.cleanup_process_owner_on_close = cleanup_process_owner_on_close
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def read_file(self, request: FileReadRequest) -> ExecutionResult:
        from chulk.tools.files import FileReadPolicy, read_file

        self._ensure_open()
        result = read_file(
            request.to_arguments(),
            self.project_root,
            read_policy=FileReadPolicy(
                allow_sensitive_paths=self._backend.allow_sensitive_reads
            ),
        )
        return self._normalize(result)

    def write_file(self, request: FileWriteRequest) -> ExecutionResult:
        from chulk.tools.files import write_file

        self._ensure_open()
        return self._normalize(
            write_file(request.to_arguments(), self.project_root),
            include_changes=True,
        )

    def apply_patch(self, request: PatchApplyRequest) -> ExecutionResult:
        from chulk.tools.files import apply_patch

        self._ensure_open()
        return self._normalize(
            apply_patch(request.to_arguments(), self.project_root),
            include_changes=True,
        )

    def list_files(self, request: FileListRequest) -> ExecutionResult:
        from chulk.tools.files import FileReadPolicy, list_files

        self._ensure_open()
        return self._normalize(
            list_files(
                request.to_arguments(),
                self.project_root,
                read_policy=FileReadPolicy(
                    allow_sensitive_paths=self._backend.allow_sensitive_reads
                ),
            )
        )

    def search_files(self, request: FileSearchRequest) -> ExecutionResult:
        from chulk.tools.files import FileReadPolicy, search_files

        self._ensure_open()
        return self._normalize(
            search_files(
                request.to_arguments(),
                self.project_root,
                read_policy=FileReadPolicy(
                    allow_sensitive_paths=self._backend.allow_sensitive_reads
                ),
            )
        )

    def run_command(self, request: CommandExecutionRequest) -> ExecutionResult:
        from chulk.tools.shell import run_shell_command

        self._ensure_open()
        result = run_shell_command(
            request.to_arguments(),
            self.project_root,
            self._backend.shell_timeout_seconds,
            stdout_limit_bytes=self._backend.max_stdout_bytes,
            stderr_limit_bytes=self._backend.max_stderr_bytes,
            execution_policy=self.shell_execution_policy,
            require_containment=self._backend.require_shell_containment,
        )
        return self._normalize(result)

    def start_process(self, request: ProcessStartRequest) -> ExecutionResult:
        self._ensure_open()
        registry = self._backend.process_registry
        result = registry.start(
            owner_key=self.process_owner_key,
            workspace_id=self.workspace.workspace_id,
            backend_name=self.workspace.backend_name,
            request=request,
            prepare=lambda _process_id: prepare_host_process(
                request=request,
                cwd=self.project_root,
                default_timeout_seconds=registry.policy.max_runtime_seconds,
                output_limit_bytes=registry.policy.max_log_bytes,
                execution_policy=self.shell_execution_policy,
                require_containment=self._backend.require_shell_containment,
            ),
        )
        return self._normalize(result)

    def poll_process(self, request: ProcessPollRequest) -> ExecutionResult:
        self._ensure_open()
        return self._normalize(
            self._backend.process_registry.poll(self.process_owner_key, request)
        )

    def read_process_logs(self, request: ProcessLogsRequest) -> ExecutionResult:
        self._ensure_open()
        return self._normalize(
            self._backend.process_registry.logs(self.process_owner_key, request)
        )

    def write_process(self, request: ProcessWriteRequest) -> ExecutionResult:
        self._ensure_open()
        return self._normalize(
            self._backend.process_registry.write(self.process_owner_key, request)
        )

    def terminate_process(self, request: ProcessTerminateRequest) -> ExecutionResult:
        self._ensure_open()
        return self._normalize(
            self._backend.process_registry.terminate(self.process_owner_key, request)
        )

    async def read_file_async(self, request: FileReadRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.read_file, request)

    async def write_file_async(self, request: FileWriteRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.write_file, request)

    async def apply_patch_async(self, request: PatchApplyRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.apply_patch, request)

    async def list_files_async(self, request: FileListRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.list_files, request)

    async def search_files_async(self, request: FileSearchRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.search_files, request)

    async def run_command_async(self, request: CommandExecutionRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.run_command, request)

    async def start_process_async(self, request: ProcessStartRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.start_process, request)

    async def poll_process_async(self, request: ProcessPollRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.poll_process, request)

    async def read_process_logs_async(
        self,
        request: ProcessLogsRequest,
    ) -> ExecutionResult:
        return await asyncio.to_thread(self.read_process_logs, request)

    async def write_process_async(self, request: ProcessWriteRequest) -> ExecutionResult:
        return await asyncio.to_thread(self.write_process, request)

    async def terminate_process_async(
        self,
        request: ProcessTerminateRequest,
    ) -> ExecutionResult:
        return await asyncio.to_thread(self.terminate_process, request)

    def close(self) -> None:
        if self._closed:
            return
        if self.cleanup_process_owner_on_close:
            self._backend.process_registry.cleanup_owner(self.process_owner_key)
        self._closed = True

    async def aclose(self) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Execution session is closed")

    def _normalize(
        self,
        result: ToolResult,
        *,
        include_changes: bool = False,
    ) -> ExecutionResult:
        metadata = dict(result.metadata)
        shell_policy = metadata.get("execution_policy")
        effective_policy = self.policy.name
        containment = self.workspace.containment.value
        if isinstance(shell_policy, dict):
            effective_policy = str(shell_policy.get("name") or effective_policy)
            if shell_policy.get("containment_applied"):
                containment = ContainmentStatus.CONTAINED.value
        metadata.update(
            {
                "execution_backend": self.workspace.backend_name,
                "execution_workspace_id": self.workspace.workspace_id,
                "workspace_mode": self.workspace.mode.value,
                "containment": containment,
                "effective_policy": effective_policy,
                "network_policy": self.policy.network.value,
                "workspace_persistence": self.policy.persistence.value,
                "change_disposition": self.policy.change_disposition.value,
            }
        )
        change_set = _change_set_from_metadata(metadata) if include_changes else None
        return ExecutionResult(
            success=result.success,
            observation=result.observation,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            error=result.error,
            failure_kind=result.failure_kind,
            metadata=metadata,
            value=result.value,
            change_set=change_set,
        )


class ExecutionContextLifecycle:
    """Attach one backend session to each agent turn's tool context."""

    def __init__(self, backend: ExecutionBackend) -> None:
        self.backend = backend

    def open(self, context: ToolExecutionContext[Any]) -> ToolExecutionContext[Any]:
        request = ExecutionSessionRequest(
            conversation_id=_metadata_text(context.metadata, "conversation_id"),
            turn_id=_metadata_text(context.metadata, "turn_id"),
            metadata=context.metadata,
        )
        return replace(context, execution_session=self.backend.open_session(request))

    def close(self, context: ToolExecutionContext[Any]) -> None:
        session = context.execution_session
        close = getattr(session, "close", None)
        if callable(close):
            close()

    async def aclose(self, context: ToolExecutionContext[Any]) -> None:
        session = context.execution_session
        aclose = getattr(session, "aclose", None)
        if callable(aclose):
            await aclose()
            return
        self.close(context)


def _metadata_text(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def has_child_task_scope(request: ExecutionSessionRequest) -> bool:
    child_id = request.metadata.get("child_task_id")
    return isinstance(child_id, str) and bool(child_id.strip())


def _change_set_from_metadata(metadata: dict[str, Any]) -> ChangeSet | None:
    raw_changes = metadata.get("changes")
    if not isinstance(raw_changes, list):
        path = metadata.get("path")
        status = metadata.get("status")
        if isinstance(path, str) and isinstance(status, str):
            raw_changes = [metadata]
        else:
            return None
    changes = tuple(
        ChangeRecord(
            path=str(item["path"]),
            status=str(item["status"]),
            sha256_before=_optional_text(item.get("sha256_before")),
            sha256_after=_optional_text(item.get("sha256_after")),
            mode_before=_optional_int(item.get("mode_before")),
            mode_after=_optional_int(item.get("mode_after")),
        )
        for item in raw_changes
        if isinstance(item, dict) and "path" in item and "status" in item
    )
    return ChangeSet(changes) if changes else None


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
