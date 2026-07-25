"""Backend-owned managed processes with scoped handles and bounded logs."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, BinaryIO, cast
from uuid import uuid4

from chulk.execution.models import (
    ProcessHandle,
    ProcessLogChunk,
    ProcessLogEntry,
    ProcessLogsRequest,
    ProcessPollRequest,
    ProcessSnapshot,
    ProcessStartRequest,
    ProcessState,
    ProcessTerminateRequest,
    ProcessWriteRequest,
)
from chulk.execution.policy import ProcessPolicy
from chulk.tools.registry import ToolFailureKind, ToolResult
from chulk.tools.shell import (
    ShellExecutionPolicy,
    ShellExecutionRequest,
    _blocked_reason,
    _execution_decision,
    _kill_process_tree,
    _policy_metadata,
    _validate_execution_decision,
)


@dataclass(frozen=True)
class PreparedProcess:
    """Validated host transport used to create one managed process."""

    command: str | tuple[str, ...]
    shell: bool
    cwd: Path
    environment: Mapping[str, str] | None
    timeout_seconds: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    signal_callback: Callable[[str], None] | None = None


@dataclass(frozen=True)
class _BufferedLog:
    start: int
    end: int
    stream: str
    data: bytes


@dataclass
class _ProcessRecord:
    handle: ProcessHandle
    owner_key: str
    process: subprocess.Popen[bytes]
    started_at: float
    deadline: float
    interactive: bool
    log_limit: int
    signal_callback: Callable[[str], None] | None
    metadata: dict[str, Any]
    logs: deque[_BufferedLog] = field(default_factory=deque)
    log_start: int = 0
    log_end: int = 0
    termination_reason: str | None = None
    termination_method: str | None = None
    ended_at: float | None = None
    read_errors: dict[str, str] = field(default_factory=dict)
    reader_threads: list[threading.Thread] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    stdin_lock: threading.Lock = field(default_factory=threading.Lock)
    termination_lock: threading.Lock = field(default_factory=threading.Lock)

    def append_log(self, stream: str, data: bytes) -> None:
        if not data:
            return
        with self.lock:
            start = self.log_end
            self.log_end += len(data)
            self.logs.append(
                _BufferedLog(
                    start=start,
                    end=self.log_end,
                    stream=stream,
                    data=data,
                )
            )
            while self.logs and self.log_end - self.logs[0].start > self.log_limit:
                excess = self.log_end - self.logs[0].start - self.log_limit
                first = self.logs[0]
                if excess >= len(first.data):
                    self.logs.popleft()
                    self.log_start = first.end
                    continue
                trimmed = first.data[excess:]
                self.logs[0] = _BufferedLog(
                    start=first.start + excess,
                    end=first.end,
                    stream=first.stream,
                    data=trimmed,
                )
                self.log_start = first.start + excess
                break
            if self.logs:
                self.log_start = self.logs[0].start
            else:
                self.log_start = self.log_end


class ManagedProcessRegistry:
    """Own subprocesses for one execution backend and enforce handle scope."""

    def __init__(self, policy: ProcessPolicy | None = None) -> None:
        self.policy = policy or ProcessPolicy()
        self._records: dict[str, _ProcessRecord] = {}
        self._starting_by_owner: dict[str, int] = {}
        self._lock = threading.RLock()
        self._closed = False

    def start(
        self,
        *,
        owner_key: str,
        workspace_id: str,
        backend_name: str,
        request: ProcessStartRequest,
        prepare: Callable[[str], PreparedProcess | ToolResult],
    ) -> ToolResult:
        if not isinstance(request.command, str) or not request.command.strip():
            return _failure(
                "process.start",
                "Process command must not be empty.",
                error="invalid_process_command",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        if request.timeout_seconds is not None and (
            isinstance(request.timeout_seconds, bool)
            or not isinstance(request.timeout_seconds, int)
            or request.timeout_seconds < 1
        ):
            return _failure(
                "process.start",
                "Process timeout must be a positive integer.",
                error="invalid_process_timeout",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        if not isinstance(request.interactive, bool):
            return _failure(
                "process.start",
                "Process interactive flag must be a boolean.",
                error="invalid_process_interactive",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        with self._lock:
            if self._closed:
                return _failure(
                    "process.start",
                    "Managed process registry is closed.",
                    error="process_registry_closed",
                    failure_kind=ToolFailureKind.ENVIRONMENT,
                )
            self._evict_finished(owner_key)
            owner_records = [
                record
                for record in self._records.values()
                if record.owner_key == owner_key
            ]
            starting = self._starting_by_owner.get(owner_key, 0)
            if (
                len(owner_records) + starting
                >= self.policy.max_processes_per_owner
            ):
                return _failure(
                    "process.start",
                    "The owner has reached the managed-process limit.",
                    error="process_limit_reached",
                    failure_kind=ToolFailureKind.FATAL_SAFETY,
                )
            self._starting_by_owner[owner_key] = starting + 1
            process_id = "process-" + uuid4().hex

        try:
            prepared = prepare(process_id)
        except Exception as exc:
            self._release_start(owner_key)
            return _failure(
                "process.start",
                "Managed process preparation failed before execution.",
                error="process_preparation_failed",
                failure_kind=ToolFailureKind.ENVIRONMENT,
                metadata={"exception_type": type(exc).__name__},
            )
        if isinstance(prepared, ToolResult):
            self._release_start(owner_key)
            return replace(prepared, tool_name="process.start")
        if (
            isinstance(prepared.timeout_seconds, bool)
            or not isinstance(prepared.timeout_seconds, int)
            or prepared.timeout_seconds < 1
        ):
            self._release_start(owner_key)
            return _failure(
                "process.start",
                "Prepared process timeout must be a positive integer.",
                error="invalid_prepared_process",
                failure_kind=ToolFailureKind.ENVIRONMENT,
            )
        timeout_seconds = min(
            prepared.timeout_seconds,
            self.policy.max_runtime_seconds,
        )
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif os.name == "nt":  # pragma: no cover - exercised on Windows
            popen_kwargs["creationflags"] = getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
            )
        try:
            process = subprocess.Popen(
                prepared.command,
                shell=prepared.shell,
                cwd=prepared.cwd,
                env=(
                    dict(prepared.environment)
                    if prepared.environment is not None
                    else None
                ),
                text=False,
                bufsize=0,
                stdin=subprocess.PIPE if request.interactive else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **popen_kwargs,
            )
        except OSError as exc:
            self._release_start(owner_key)
            return _failure(
                "process.start",
                "Managed process failed before the child process started.",
                error="process_start_failed",
                failure_kind=ToolFailureKind.ENVIRONMENT,
                metadata={
                    **dict(prepared.metadata),
                    "child_process_started": False,
                    "exception_type": type(exc).__name__,
                },
            )

        handle = ProcessHandle(
            process_id=process_id,
            backend_name=backend_name,
            workspace_id=workspace_id,
        )
        started_at = time.monotonic()
        record = _ProcessRecord(
            handle=handle,
            owner_key=owner_key,
            process=process,
            started_at=started_at,
            deadline=started_at + timeout_seconds,
            interactive=request.interactive,
            log_limit=self.policy.max_log_bytes,
            signal_callback=prepared.signal_callback,
            metadata={
                **dict(prepared.metadata),
                "timeout_seconds": timeout_seconds,
                "child_process_started": True,
            },
        )
        record.reader_threads = [
            _start_reader(record, "stdout", cast(BinaryIO | None, process.stdout)),
            _start_reader(record, "stderr", cast(BinaryIO | None, process.stderr)),
        ]
        threading.Thread(
            target=self._monitor,
            args=(record,),
            name=f"chulk-process-monitor-{process_id[-8:]}",
            daemon=True,
        ).start()
        with self._lock:
            registry_closed = self._closed
            if not registry_closed:
                self._records[process_id] = record
            self._release_start(owner_key)
        if registry_closed:
            _terminate_record(
                record,
                reason="backend_cleanup",
                grace_seconds=self.policy.termination_grace_seconds,
            )
            return _failure(
                "process.start",
                "Managed process registry closed while starting the process.",
                error="process_registry_closed",
                failure_kind=ToolFailureKind.ENVIRONMENT,
            )
        snapshot = _snapshot_record(record)
        return ToolResult(
            tool_name="process.start",
            success=True,
            observation=(
                "Managed process started. "
                f"Handle: process_id={process_id}, backend_name={backend_name}, "
                f"workspace_id={workspace_id}."
            ),
            metadata={
                **record.metadata,
                "process_id": process_id,
                "backend_name": backend_name,
                "workspace_id": workspace_id,
                "process_state": snapshot.state.value,
            },
            value=handle,
        )

    def poll(self, owner_key: str, request: ProcessPollRequest) -> ToolResult:
        record_or_error = self._record_for(
            owner_key,
            request.handle,
            tool_name="process.poll",
        )
        if isinstance(record_or_error, ToolResult):
            return record_or_error
        snapshot = _snapshot_record(record_or_error)
        return ToolResult(
            tool_name="process.poll",
            success=True,
            observation=f"Managed process is {snapshot.state.value}.",
            exit_code=snapshot.exit_code,
            metadata=_snapshot_metadata(record_or_error, snapshot),
            value=snapshot,
        )

    def logs(self, owner_key: str, request: ProcessLogsRequest) -> ToolResult:
        record_or_error = self._record_for(
            owner_key,
            request.handle,
            tool_name="process.logs",
        )
        if isinstance(record_or_error, ToolResult):
            return record_or_error
        if (
            isinstance(request.cursor, bool)
            or not isinstance(request.cursor, int)
            or request.cursor < 0
        ):
            return _failure(
                "process.logs",
                "Process log cursor must be a non-negative integer.",
                error="invalid_process_cursor",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        max_bytes = (
            self.policy.default_log_read_bytes
            if request.max_bytes is None
            else request.max_bytes
        )
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
            or max_bytes > self.policy.max_log_read_bytes
        ):
            return _failure(
                "process.logs",
                "Process log byte limit is outside the host-configured range.",
                error="invalid_process_log_limit",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        chunk = _read_logs(record_or_error, request.cursor, max_bytes)
        stdout = "".join(
            entry.text for entry in chunk.entries if entry.stream == "stdout"
        )
        stderr = "".join(
            entry.text for entry in chunk.entries if entry.stream == "stderr"
        )
        return ToolResult(
            tool_name="process.logs",
            success=True,
            observation=(
                f"Managed process logs read from cursor {chunk.cursor}; "
                f"next_cursor={chunk.next_cursor}; truncated={chunk.truncated}."
            ),
            stdout=stdout or None,
            stderr=stderr or None,
            metadata={
                "process_id": request.handle.process_id,
                "cursor": chunk.cursor,
                "next_cursor": chunk.next_cursor,
                "truncated": chunk.truncated,
                "entry_count": len(chunk.entries),
            },
            value=chunk,
        )

    def write(self, owner_key: str, request: ProcessWriteRequest) -> ToolResult:
        record_or_error = self._record_for(
            owner_key,
            request.handle,
            tool_name="process.write",
        )
        if isinstance(record_or_error, ToolResult):
            return record_or_error
        record = record_or_error
        if not isinstance(request.data, str) or not isinstance(
            request.close_stdin,
            bool,
        ):
            return _failure(
                "process.write",
                "Process stdin data and close flag are invalid.",
                error="invalid_process_write",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        data = request.data.encode("utf-8")
        if len(data) > self.policy.max_write_bytes:
            return _failure(
                "process.write",
                "Process stdin write exceeds the host-configured byte limit.",
                error="process_write_too_large",
                failure_kind=ToolFailureKind.FATAL_SAFETY,
            )
        if not record.interactive or record.process.stdin is None:
            return _failure(
                "process.write",
                "Managed process was not started with interactive stdin.",
                error="process_not_interactive",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        if record.process.poll() is not None:
            return _failure(
                "process.write",
                "Managed process has already exited.",
                error="process_not_running",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        try:
            with record.stdin_lock:
                if data:
                    record.process.stdin.write(data)
                    record.process.stdin.flush()
                if request.close_stdin:
                    record.process.stdin.close()
        except (BrokenPipeError, OSError, ValueError) as exc:
            return _failure(
                "process.write",
                "Managed process stdin is unavailable.",
                error="process_stdin_unavailable",
                failure_kind=ToolFailureKind.ENVIRONMENT,
                metadata={"exception_type": type(exc).__name__},
            )
        return ToolResult(
            tool_name="process.write",
            success=True,
            observation="Managed process stdin updated.",
            metadata={
                "process_id": request.handle.process_id,
                "written_bytes": len(data),
                "stdin_closed": request.close_stdin,
            },
        )

    def terminate(
        self,
        owner_key: str,
        request: ProcessTerminateRequest,
    ) -> ToolResult:
        record_or_error = self._record_for(
            owner_key,
            request.handle,
            tool_name="process.terminate",
        )
        if isinstance(record_or_error, ToolResult):
            return record_or_error
        grace_seconds = (
            self.policy.termination_grace_seconds
            if request.grace_seconds is None
            else request.grace_seconds
        )
        if (
            isinstance(grace_seconds, bool)
            or not isinstance(grace_seconds, int)
            or grace_seconds < 1
            or grace_seconds > self.policy.termination_grace_seconds
        ):
            return _failure(
                "process.terminate",
                "Termination grace period is outside the host-configured range.",
                error="invalid_termination_grace",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        _terminate_record(
            record_or_error,
            reason="requested",
            grace_seconds=grace_seconds,
        )
        snapshot = _snapshot_record(record_or_error)
        return ToolResult(
            tool_name="process.terminate",
            success=True,
            observation=f"Managed process is {snapshot.state.value}.",
            exit_code=snapshot.exit_code,
            metadata=_snapshot_metadata(record_or_error, snapshot),
            value=snapshot,
        )

    def cleanup_workspace(self, workspace_id: str) -> None:
        self._cleanup_matching(
            lambda record: record.handle.workspace_id == workspace_id,
            reason="workspace_cleanup",
        )

    def cleanup_owner(self, owner_key: str) -> None:
        self._cleanup_matching(
            lambda record: record.owner_key == owner_key,
            reason="owner_cleanup",
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._cleanup_matching(lambda record: True, reason="backend_cleanup")
        with self._lock:
            self._records.clear()

    def _cleanup_matching(
        self,
        predicate: Callable[[_ProcessRecord], bool],
        *,
        reason: str,
    ) -> None:
        with self._lock:
            records = [
                record
                for record in self._records.values()
                if predicate(record)
            ]
        for record in records:
            _terminate_record(
                record,
                reason=reason,
                grace_seconds=self.policy.termination_grace_seconds,
            )
        with self._lock:
            for record in records:
                self._records.pop(record.handle.process_id, None)

    def _monitor(self, record: _ProcessRecord) -> None:
        remaining = max(0.0, record.deadline - time.monotonic())
        try:
            record.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _terminate_record(
                record,
                reason="timeout",
                grace_seconds=self.policy.termination_grace_seconds,
            )
        finally:
            for reader in record.reader_threads:
                reader.join(timeout=self.policy.termination_grace_seconds)
            with record.lock:
                if record.ended_at is None and record.process.poll() is not None:
                    record.ended_at = time.monotonic()

    def _record_for(
        self,
        owner_key: str,
        handle: ProcessHandle,
        *,
        tool_name: str,
    ) -> _ProcessRecord | ToolResult:
        if not isinstance(handle, ProcessHandle):
            return _failure(
                tool_name,
                "Managed process handle is invalid.",
                error="invalid_process_handle",
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
            )
        with self._lock:
            record = self._records.get(handle.process_id)
        if (
            record is None
            or record.owner_key != owner_key
            or record.handle != handle
        ):
            return _failure(
                tool_name,
                "Managed process handle is unavailable for this owner.",
                error="process_handle_unavailable",
                failure_kind=ToolFailureKind.FATAL_SAFETY,
            )
        return record

    def _evict_finished(self, owner_key: str) -> None:
        finished = sorted(
            (
                record
                for record in self._records.values()
                if record.owner_key == owner_key
                and record.process.poll() is not None
            ),
            key=lambda record: record.started_at,
        )
        while (
            len(
                [
                    record
                    for record in self._records.values()
                    if record.owner_key == owner_key
                ]
            )
            >= self.policy.max_processes_per_owner
            and finished
        ):
            record = finished.pop(0)
            self._records.pop(record.handle.process_id, None)

    def _release_start(self, owner_key: str) -> None:
        with self._lock:
            remaining = self._starting_by_owner.get(owner_key, 0) - 1
            if remaining > 0:
                self._starting_by_owner[owner_key] = remaining
            else:
                self._starting_by_owner.pop(owner_key, None)


def prepare_host_process(
    *,
    request: ProcessStartRequest,
    cwd: Path,
    default_timeout_seconds: int,
    output_limit_bytes: int,
    execution_policy: ShellExecutionPolicy | None,
    require_containment: bool,
) -> PreparedProcess | ToolResult:
    """Apply the existing shell safety and policy seam before a managed start."""
    timeout_seconds = min(
        request.timeout_seconds or default_timeout_seconds,
        default_timeout_seconds,
    )
    blocked_reason = _blocked_reason(request.command, cwd)
    if blocked_reason:
        return _failure(
            "process.start",
            f"Blocked command: {blocked_reason}",
            error="blocked_command",
            failure_kind=ToolFailureKind.FATAL_SAFETY,
            metadata={
                "command": request.command,
                "cwd": str(cwd),
                "child_process_started": False,
            },
        )
    shell_request = ShellExecutionRequest(
        command=request.command,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=output_limit_bytes,
        stderr_limit_bytes=output_limit_bytes,
    )
    decision_result = _execution_decision(execution_policy, shell_request)
    if isinstance(decision_result, ToolResult):
        return replace(decision_result, tool_name="process.start")
    decision = decision_result
    decision_error = _validate_execution_decision(decision)
    policy_metadata = _policy_metadata(
        decision,
        require_containment=require_containment,
    )
    base_metadata = {
        "command": request.command,
        "cwd": str(cwd),
        "execution_policy": policy_metadata,
    }
    if decision_error:
        return _failure(
            "process.start",
            f"Shell execution policy returned an invalid decision: {decision_error}",
            error="invalid_execution_policy",
            failure_kind=ToolFailureKind.ENVIRONMENT,
            metadata={**base_metadata, "child_process_started": False},
        )
    if decision.command is None:
        return _failure(
            "process.start",
            (
                "Process execution denied by host policy: "
                f"{decision.denial_reason or 'no reason provided'}"
            ),
            error="execution_policy_denied",
            failure_kind=(
                ToolFailureKind.FATAL_SAFETY
                if decision.fatal
                else ToolFailureKind.USER_BLOCKED
            ),
            metadata={**base_metadata, "child_process_started": False},
        )
    if require_containment and not decision.containment_applied:
        return _failure(
            "process.start",
            (
                "Process execution requires host-provided containment, but the "
                "selected policy did not assert containment."
            ),
            error="containment_required",
            failure_kind=ToolFailureKind.FATAL_SAFETY,
            metadata={**base_metadata, "child_process_started": False},
        )
    return PreparedProcess(
        command=decision.command,
        shell=decision.shell,
        cwd=cwd,
        environment=decision.environment,
        timeout_seconds=timeout_seconds,
        metadata=base_metadata,
    )


def owner_key(
    *,
    conversation_id: str | None,
    turn_id: str | None,
    metadata: Mapping[str, Any],
    workspace_id: str,
) -> str:
    """Resolve the narrowest stable host-owned process authority scope."""
    child_id = metadata.get("child_task_id")
    if isinstance(child_id, str) and child_id.strip():
        parent_scope = (
            f"conversation:{conversation_id}"
            if conversation_id and conversation_id.strip()
            else (
                f"turn:{turn_id}"
                if turn_id and turn_id.strip()
                else f"workspace:{workspace_id}"
            )
        )
        return f"{parent_scope}:child:{child_id}"
    if conversation_id and conversation_id.strip():
        return f"conversation:{conversation_id}"
    if turn_id and turn_id.strip():
        return f"turn:{turn_id}"
    return f"workspace:{workspace_id}"


def _start_reader(
    record: _ProcessRecord,
    stream_name: str,
    pipe: BinaryIO | None,
) -> threading.Thread:
    def read_stream() -> None:
        if pipe is None:
            return
        try:
            while chunk := pipe.read(4096):
                record.append_log(stream_name, chunk)
        except OSError as exc:
            with record.lock:
                record.read_errors[stream_name] = type(exc).__name__
        finally:
            pipe.close()

    reader = threading.Thread(
        target=read_stream,
        name=f"chulk-process-{stream_name}-{record.handle.process_id[-8:]}",
        daemon=True,
    )
    reader.start()
    return reader


def _read_logs(
    record: _ProcessRecord,
    requested_cursor: int,
    max_bytes: int,
) -> ProcessLogChunk:
    entries: list[ProcessLogEntry] = []
    with record.lock:
        truncated = requested_cursor < record.log_start
        cursor = min(max(requested_cursor, record.log_start), record.log_end)
        remaining = max_bytes
        next_cursor = cursor
        for chunk in record.logs:
            if chunk.end <= cursor:
                continue
            offset = max(0, cursor - chunk.start)
            data = chunk.data[offset : offset + remaining]
            if not data:
                continue
            entries.append(
                ProcessLogEntry(
                    stream=chunk.stream,
                    text=(
                        data.decode("utf-8", errors="replace")
                        .replace("\r\n", "\n")
                        .replace("\r", "\n")
                    ),
                )
            )
            consumed = len(data)
            remaining -= consumed
            next_cursor = max(next_cursor, chunk.start + offset + consumed)
            if remaining == 0:
                break
        return ProcessLogChunk(
            cursor=cursor,
            next_cursor=next_cursor,
            entries=tuple(entries),
            truncated=truncated,
        )


def _snapshot_record(record: _ProcessRecord) -> ProcessSnapshot:
    exit_code = record.process.poll()
    with record.lock:
        if exit_code is None:
            state = ProcessState.RUNNING
            ended_at = time.monotonic()
        else:
            if record.ended_at is None:
                record.ended_at = time.monotonic()
            ended_at = record.ended_at
            if record.termination_reason == "timeout":
                state = ProcessState.TIMED_OUT
            elif record.termination_reason is not None:
                state = ProcessState.TERMINATED
            else:
                state = ProcessState.EXITED
        return ProcessSnapshot(
            handle=record.handle,
            state=state,
            exit_code=exit_code,
            duration_seconds=max(0.0, ended_at - record.started_at),
            termination_reason=record.termination_reason,
        )


def _snapshot_metadata(
    record: _ProcessRecord,
    snapshot: ProcessSnapshot,
) -> dict[str, Any]:
    with record.lock:
        return {
            **record.metadata,
            "process_id": record.handle.process_id,
            "process_state": snapshot.state.value,
            "duration_seconds": snapshot.duration_seconds,
            "termination_reason": snapshot.termination_reason,
            "termination_method": record.termination_method,
            "log_start_cursor": record.log_start,
            "log_end_cursor": record.log_end,
            "log_truncated": record.log_start > 0,
            "stream_read_errors": dict(record.read_errors),
            "reader_threads_stopped": all(
                not reader.is_alive()
                for reader in record.reader_threads
            ),
        }


def _terminate_record(
    record: _ProcessRecord,
    *,
    reason: str,
    grace_seconds: int,
) -> None:
    with record.termination_lock:
        with record.lock:
            if record.process.poll() is not None:
                if record.ended_at is None:
                    record.ended_at = time.monotonic()
                return
            if record.termination_reason is None:
                record.termination_reason = reason
        methods: list[str] = []
        if record.signal_callback is not None:
            try:
                record.signal_callback("TERM")
                methods.append("backend_sigterm")
            except Exception as exc:
                methods.append(f"backend_sigterm_failed:{type(exc).__name__}")
        elif os.name == "nt":  # pragma: no cover - exercised on Windows
            methods.append(_kill_process_tree(record.process))
        else:
            methods.append(_signal_process_tree(record.process))
        try:
            record.process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            if record.signal_callback is not None:
                try:
                    record.signal_callback("KILL")
                    methods.append("backend_sigkill")
                except Exception as exc:
                    methods.append(f"backend_sigkill_failed:{type(exc).__name__}")
            methods.append(_kill_process_tree(record.process))
            try:
                record.process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                methods.append("process_wait_timeout")
        with record.lock:
            record.termination_method = ",".join(methods)
            record.ended_at = time.monotonic()
        for reader in record.reader_threads:
            reader.join(timeout=grace_seconds)


def _signal_process_tree(process: subprocess.Popen[bytes]) -> str:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return "posix_process_group_sigterm"
        except ProcessLookupError:
            return "already_exited"
        except OSError:
            pass
    try:
        process.terminate()
        return "process_terminate"
    except ProcessLookupError:
        return "already_exited"


def _failure(
    tool_name: str,
    observation: str,
    *,
    error: str,
    failure_kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        success=False,
        observation=observation,
        error=error,
        failure_kind=failure_kind,
        metadata=dict(metadata or {}),
    )
