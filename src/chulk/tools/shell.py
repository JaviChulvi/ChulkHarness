"""Bounded shell execution with explicit host-owned containment policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import threading
import time
from typing import Any, BinaryIO, Protocol, cast

from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.registry import Tool, ToolFailureKind, ToolResult


DEFAULT_SHELL_STDOUT_LIMIT_BYTES = 8000
DEFAULT_SHELL_STDERR_LIMIT_BYTES = 4000
SHELL_CLEANUP_GRACE_SECONDS = 1.0
_READ_CHUNK_BYTES = 4096
_TRUNCATION_MARKER = b"\n... shell output truncated ...\n"
_SHELL_INTERPRETERS = {"ash", "bash", "dash", "fish", "ksh", "sh", "zsh"}
_SHELL_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")"}
_SHELL_COMMAND_WRAPPERS = {"builtin", "command", "env", "exec", "nohup"}
_SHELL_VARIABLE = re.compile(r"^\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))$")
_SHELL_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)

DESTRUCTIVE_PATTERNS = [
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:"),
    re.compile(r">\s*/(?:etc|bin|sbin|usr|System|Library)\b"),
]


@dataclass(frozen=True)
class ShellExecutionRequest:
    """Normalized shell request passed to a host execution policy."""

    command: str
    cwd: Path
    timeout_seconds: int
    stdout_limit_bytes: int
    stderr_limit_bytes: int


@dataclass(frozen=True)
class ShellExecutionDecision:
    """Host-selected command transport and its containment assertion.

    ``containment_applied`` is an assertion by the embedding host. Chulk cannot
    infer that a wrapper actually provides filesystem, network, or process
    isolation.
    """

    command: str | tuple[str, ...] | None
    policy_name: str
    shell: bool = True
    environment: Mapping[str, str] | None = None
    containment_applied: bool = False
    termination_callback: Callable[[str], None] | None = None
    denial_reason: str | None = None
    fatal: bool = False

    @classmethod
    def allow(
        cls,
        command: str | Sequence[str],
        *,
        policy_name: str,
        shell: bool,
        environment: Mapping[str, str] | None = None,
        containment_applied: bool = False,
        termination_callback: Callable[[str], None] | None = None,
    ) -> ShellExecutionDecision:
        """Allow execution through a host-selected command transport."""
        normalized = command if isinstance(command, str) else tuple(command)
        return cls(
            command=normalized,
            policy_name=policy_name,
            shell=shell,
            environment=environment,
            containment_applied=containment_applied,
            termination_callback=termination_callback,
        )

    @classmethod
    def deny(
        cls,
        reason: str,
        *,
        policy_name: str,
        fatal: bool = False,
    ) -> ShellExecutionDecision:
        """Deny execution before a child process is created."""
        return cls(
            command=None,
            policy_name=policy_name,
            denial_reason=reason,
            fatal=fatal,
        )


class ShellExecutionPolicy(Protocol):
    """Host boundary for denying, wrapping, or containing shell execution."""

    def prepare(self, request: ShellExecutionRequest) -> ShellExecutionDecision:
        """Return the host-owned execution decision for ``request``."""


class DirectShellExecutionPolicy:
    """Default local execution policy; deliberately makes no sandbox claim."""

    def prepare(self, request: ShellExecutionRequest) -> ShellExecutionDecision:
        return ShellExecutionDecision.allow(
            request.command,
            policy_name="direct-local",
            shell=True,
            containment_applied=False,
        )


def shell_tool(
    project_root: Path,
    timeout_seconds: int = 10,
    *,
    stdout_limit_bytes: int = DEFAULT_SHELL_STDOUT_LIMIT_BYTES,
    stderr_limit_bytes: int = DEFAULT_SHELL_STDERR_LIMIT_BYTES,
    execution_policy: ShellExecutionPolicy | None = None,
    require_containment: bool = False,
) -> Tool:
    """Create the shell command tool."""
    _validate_output_limit("stdout_limit_bytes", stdout_limit_bytes)
    _validate_output_limit("stderr_limit_bytes", stderr_limit_bytes)

    def invoke(arguments: dict[str, Any], context=None) -> ToolResult:
        session = getattr(context, "execution_session", None)
        if session is None:
            return run_shell_command(
                arguments,
                project_root,
                timeout_seconds,
                stdout_limit_bytes=stdout_limit_bytes,
                stderr_limit_bytes=stderr_limit_bytes,
                execution_policy=execution_policy,
                require_containment=require_containment,
            )
        from chulk.execution.models import CommandExecutionRequest

        return session.run_command(
            CommandExecutionRequest(
                command=arguments["command"],
                timeout_seconds=arguments.get("timeout_seconds"),
            )
        ).to_tool_result("run_cmd")

    return Tool(
        name="run_cmd",
        description=(
            "Run a shell command in the project directory with timeout, bounded output capture, "
            "and safety blocking."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run from the project root.",
                    "minLength": 1,
                    "maxLength": 4000,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout in seconds.",
                    "minimum": 1,
                    "maximum": timeout_seconds,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        callable=invoke,
        accepts_context=True,
        # The command enforces its own deadline and needs a small window to kill
        # descendants and drain the now-closed pipes before the executor returns.
        timeout_seconds=timeout_seconds + SHELL_CLEANUP_GRACE_SECONDS,
        run_in_executor=True,
        requires_confirmation=True,
        permission_level=ToolPermissionLevel.SHELL,
    )


def run_shell_command(
    arguments: dict[str, Any],
    project_root: Path | None = None,
    default_timeout_seconds: int = 10,
    *,
    stdout_limit_bytes: int = DEFAULT_SHELL_STDOUT_LIMIT_BYTES,
    stderr_limit_bytes: int = DEFAULT_SHELL_STDERR_LIMIT_BYTES,
    execution_policy: ShellExecutionPolicy | None = None,
    require_containment: bool = False,
) -> ToolResult:
    """Run a command with bounded live capture and host-owned containment."""
    _validate_output_limit("stdout_limit_bytes", stdout_limit_bytes)
    _validate_output_limit("stderr_limit_bytes", stderr_limit_bytes)
    root = (project_root or Path.cwd()).resolve()
    command = arguments["command"]
    timeout_seconds = min(arguments.get("timeout_seconds", default_timeout_seconds), default_timeout_seconds)

    blocked_reason = _blocked_reason(command, root)
    if blocked_reason:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation=f"Blocked command: {blocked_reason}",
            error="blocked_command",
            failure_kind=ToolFailureKind.FATAL_SAFETY,
            metadata={"command": command, "cwd": str(root), "child_process_started": False},
        )

    request = ShellExecutionRequest(
        command=command,
        cwd=root,
        timeout_seconds=timeout_seconds,
        stdout_limit_bytes=stdout_limit_bytes,
        stderr_limit_bytes=stderr_limit_bytes,
    )
    decision_result = _execution_decision(execution_policy, request)
    if isinstance(decision_result, ToolResult):
        return decision_result
    decision = decision_result
    decision_error = _validate_execution_decision(decision)
    if decision_error:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation=f"Shell execution policy returned an invalid decision: {decision_error}",
            error="invalid_execution_policy",
            failure_kind=ToolFailureKind.ENVIRONMENT,
            metadata={
                "command": command,
                "cwd": str(root),
                "child_process_started": False,
                "decision_type": type(decision).__name__,
            },
        )
    policy_metadata = _policy_metadata(decision, require_containment=require_containment)
    base_metadata = {
        "command": command,
        "cwd": str(root),
        "timeout_seconds": timeout_seconds,
        "stdout_limit_bytes": stdout_limit_bytes,
        "stderr_limit_bytes": stderr_limit_bytes,
        "execution_policy": policy_metadata,
    }
    if decision.command is None:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation=f"Shell execution denied by host policy: {decision.denial_reason or 'no reason provided'}",
            error="execution_policy_denied",
            failure_kind=(
                ToolFailureKind.FATAL_SAFETY
                if decision.fatal
                else ToolFailureKind.USER_BLOCKED
            ),
            metadata={**base_metadata, "child_process_started": False},
        )
    if require_containment and not decision.containment_applied:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation=(
                "Shell execution requires host-provided containment, but the selected execution policy "
                "did not assert that containment was applied."
            ),
            error="containment_required",
            failure_kind=ToolFailureKind.FATAL_SAFETY,
            metadata={**base_metadata, "child_process_started": False},
        )
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":  # pragma: no cover - exercised on Windows
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP")

    started_at = time.monotonic()
    try:
        process = subprocess.Popen(
            decision.command,
            shell=decision.shell,
            cwd=root,
            env=dict(decision.environment) if decision.environment is not None else None,
            text=False,
            bufsize=0,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
    except OSError as exc:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation="Shell execution failed before the child process started.",
            error="process_start_failed",
            failure_kind=ToolFailureKind.ENVIRONMENT,
            metadata={
                **base_metadata,
                "child_process_started": False,
                "exception_type": type(exc).__name__,
            },
        )

    stdout_capture = _BoundedStreamCapture(stdout_limit_bytes)
    stderr_capture = _BoundedStreamCapture(stderr_limit_bytes)
    overflow_event = threading.Event()
    readers = [
        _start_reader("stdout", cast(BinaryIO | None, process.stdout), stdout_capture, overflow_event),
        _start_reader("stderr", cast(BinaryIO | None, process.stderr), stderr_capture, overflow_event),
    ]

    termination_reason: str | None = None
    termination_method: str | None = None
    deadline = started_at + timeout_seconds
    while process.poll() is None or any(reader.is_alive() for reader in readers):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            termination_reason = "timeout"
            break
        if overflow_event.wait(timeout=min(0.02, remaining)):
            termination_reason = "output_limit_exceeded"
            break

    if termination_reason is not None:
        termination_method = _terminate_execution(process, decision)
    _wait_for_terminated_process(process)
    for reader in readers:
        reader.join(timeout=SHELL_CLEANUP_GRACE_SECONDS)

    if termination_reason is None and (stdout_capture.overflowed or stderr_capture.overflowed):
        # Fast commands may exit between the reader detecting overflow and the
        # polling loop observing it. Still record the required kill attempt.
        termination_reason = "output_limit_exceeded"
        termination_method = _terminate_execution(process, decision)

    stdout = stdout_capture.preview_text()
    stderr = stderr_capture.preview_text()
    duration_seconds = max(0.0, time.monotonic() - started_at)
    metadata = {
        **base_metadata,
        "child_process_started": True,
        "pid": process.pid,
        "exit_code": process.returncode,
        "duration_seconds": duration_seconds,
        "stdout_length": len(stdout),
        "stderr_length": len(stderr),
        **stdout_capture.metadata("stdout"),
        **stderr_capture.metadata("stderr"),
        "termination_reason": termination_reason,
        "termination_method": termination_method,
        "reader_threads_stopped": all(not reader.is_alive() for reader in readers),
    }
    read_errors = {
        name: capture.read_error
        for name, capture in (("stdout", stdout_capture), ("stderr", stderr_capture))
        if capture.read_error is not None
    }
    if read_errors:
        metadata["stream_read_errors"] = read_errors

    if termination_reason == "timeout":
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation=f"Command timed out after {timeout_seconds} seconds and its process group was terminated.",
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
            error="timeout",
            failure_kind=ToolFailureKind.TIMEOUT,
            metadata=metadata,
        )
    if termination_reason == "output_limit_exceeded":
        exceeded = [
            name
            for name, capture in (("stdout", stdout_capture), ("stderr", stderr_capture))
            if capture.overflowed
        ]
        metadata["output_limit_streams"] = exceeded
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation=(
                f"Command exceeded the configured {' and '.join(exceeded)} byte limit and its process group "
                "was terminated. Increase the host-configured limit only for trusted commands."
            ),
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
            error="output_limit_exceeded",
            failure_kind="output_limit_exceeded",
            metadata=metadata,
        )
    if read_errors:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation="Command output could not be captured reliably.",
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
            error="output_capture_failed",
            failure_kind=ToolFailureKind.ENVIRONMENT,
            metadata=metadata,
        )

    return ToolResult(
        tool_name="run_cmd",
        success=process.returncode == 0,
        observation="Command completed." if process.returncode == 0 else "Command failed.",
        stdout=stdout,
        stderr=stderr,
        exit_code=process.returncode,
        error=None if process.returncode == 0 else "nonzero_exit",
        metadata=metadata,
    )


@dataclass
class _BoundedStreamCapture:
    limit_bytes: int
    head: bytearray = field(default_factory=bytearray)
    tail: bytearray = field(default_factory=bytearray)
    total_bytes: int = 0
    overflowed: bool = False
    read_error: str | None = None

    @property
    def _head_limit(self) -> int:
        return (self.limit_bytes + 1) // 2

    @property
    def _tail_limit(self) -> int:
        return self.limit_bytes - self._head_limit

    def append(self, data: bytes, overflow_event: threading.Event) -> None:
        self.total_bytes += len(data)
        head_needed = max(0, self._head_limit - len(self.head))
        if head_needed:
            self.head.extend(data[:head_needed])
        remainder = data[head_needed:]
        if self._tail_limit and remainder:
            self.tail.extend(remainder)
            if len(self.tail) > self._tail_limit:
                del self.tail[: len(self.tail) - self._tail_limit]
        if self.total_bytes > self.limit_bytes:
            self.overflowed = True
            overflow_event.set()

    def preview_bytes(self) -> bytes:
        if not self.overflowed:
            return bytes(self.head + self.tail)
        if self.limit_bytes <= len(_TRUNCATION_MARKER):
            return _TRUNCATION_MARKER[: self.limit_bytes]
        available = self.limit_bytes - len(_TRUNCATION_MARKER)
        head_length = (available + 1) // 2
        tail_length = available - head_length
        tail = self.tail[-tail_length:] if tail_length else b""
        return bytes(self.head[:head_length]) + _TRUNCATION_MARKER + bytes(tail)

    def preview_text(self) -> str:
        return (
            self.preview_bytes()
            .decode("utf-8", errors="replace")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

    def metadata(self, stream_name: str) -> dict[str, Any]:
        preview = self.preview_bytes()
        return {
            f"{stream_name}_total_bytes": self.total_bytes,
            f"{stream_name}_preview_bytes": len(preview),
            f"{stream_name}_truncated": self.overflowed,
            f"{stream_name}_discarded_bytes": max(0, self.total_bytes - len(preview)),
        }


def _start_reader(
    stream_name: str,
    pipe: BinaryIO | None,
    capture: _BoundedStreamCapture,
    overflow_event: threading.Event,
) -> threading.Thread:
    def read_stream() -> None:
        if pipe is None:
            return
        try:
            while chunk := pipe.read(_READ_CHUNK_BYTES):
                capture.append(chunk, overflow_event)
        except OSError as exc:
            capture.read_error = type(exc).__name__
        finally:
            pipe.close()

    reader = threading.Thread(target=read_stream, name=f"chulk-shell-{stream_name}", daemon=True)
    reader.start()
    return reader


def _execution_decision(
    execution_policy: ShellExecutionPolicy | None,
    request: ShellExecutionRequest,
) -> ShellExecutionDecision | ToolResult:
    policy = execution_policy or DirectShellExecutionPolicy()
    try:
        decision = policy.prepare(request)
    except Exception as exc:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation="Shell execution policy failed before the child process started.",
            error="execution_policy_failed",
            failure_kind=ToolFailureKind.ENVIRONMENT,
            metadata={
                "command": request.command,
                "cwd": str(request.cwd),
                "child_process_started": False,
                "exception_type": type(exc).__name__,
            },
        )
    if not isinstance(decision, ShellExecutionDecision):
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation="Shell execution policy returned an unsupported decision type.",
            error="invalid_execution_policy",
            failure_kind=ToolFailureKind.ENVIRONMENT,
            metadata={
                "command": request.command,
                "cwd": str(request.cwd),
                "child_process_started": False,
                "decision_type": type(decision).__name__,
            },
        )
    return decision


def _validate_execution_decision(decision: ShellExecutionDecision) -> str | None:
    if not isinstance(decision.policy_name, str) or not decision.policy_name.strip():
        return "policy_name must not be empty"
    if decision.command is None:
        return None
    if decision.fatal:
        return "fatal=True is only valid for a denied decision"
    if decision.shell and not isinstance(decision.command, str):
        return "shell=True requires a string command"
    if not decision.shell and (
        isinstance(decision.command, str)
        or not decision.command
        or any(not isinstance(item, str) or not item for item in decision.command)
    ):
        return "shell=False requires a non-empty sequence of string arguments"
    if decision.environment is not None and any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in decision.environment.items()
    ):
        return "environment keys and values must be strings"
    if (
        decision.termination_callback is not None
        and not callable(decision.termination_callback)
    ):
        return "termination_callback must be callable"
    return None


def _policy_metadata(decision: ShellExecutionDecision, *, require_containment: bool) -> dict[str, Any]:
    metadata = {
        "name": decision.policy_name,
        "containment_required": require_containment,
        "containment_applied": decision.containment_applied,
        "uses_shell": decision.shell,
    }
    if decision.fatal:
        metadata["fatal"] = True
    return metadata


def _validate_output_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _blocked_reason(command: str, root: Path) -> str | None:
    lowered = command.strip().lower()
    if _has_recursive_force_rm(command):
        return "command matches a destructive pattern"
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.search(lowered):
            return "command matches a destructive pattern"
    if _redirects_outside_root(command, root):
        return "command redirects output outside the project root"
    return None


def _has_recursive_force_rm(command: str, *, _depth: int = 0) -> bool:
    if _depth > 3:
        return True
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return False

    assignments = {
        match.group(1): match.group(2)
        for token in tokens
        if (match := _SHELL_ASSIGNMENT.fullmatch(token)) is not None
    }

    for index, token in enumerate(tokens):
        if Path(token).name != "rm":
            continue
        has_recursive = False
        has_force = False
        for argument in tokens[index + 1 :]:
            if argument in _SHELL_SEPARATORS:
                break
            if argument == "--":
                break
            if argument == "--recursive":
                has_recursive = True
            elif argument == "--force":
                has_force = True
            elif argument.startswith("--"):
                continue
            elif argument.startswith("-") and argument != "-":
                flags = argument.lstrip("-")
                has_recursive = has_recursive or "r" in flags or "R" in flags
                has_force = has_force or "f" in flags
            if has_recursive and has_force:
                return True

    for index, token in enumerate(tokens):
        if Path(token).name not in _SHELL_INTERPRETERS:
            continue
        found_inline_command = False
        for option_index in range(index + 1, min(index + 4, len(tokens))):
            option = tokens[option_index]
            if option == "--":
                continue
            if option.startswith("-") and "c" in option.lstrip("-"):
                found_inline_command = True
                command_index = option_index + 1
                if command_index < len(tokens) and _has_recursive_force_rm(tokens[command_index], _depth=_depth + 1):
                    return True
                break
        if _is_shell_command_position(tokens, index) and not found_inline_command:
            return True

    for index, token in enumerate(tokens):
        command_position = _is_shell_command_position(tokens, index)
        if command_position and Path(token).name == "eval":
            nested = _expanded_shell_command(tokens[index + 1 :], assignments)
            if nested is None or _has_recursive_force_rm(nested, _depth=_depth + 1):
                return True
        if not command_position:
            continue
        if token in {"source", "."} or token == "$" or token.startswith("`"):
            return True
        variable = _SHELL_VARIABLE.fullmatch(token)
        if token.startswith("$") and variable is None:
            return True
        if variable is None:
            continue
        nested = _expanded_shell_command(tokens[index:], assignments)
        if nested is None or _has_recursive_force_rm(nested, _depth=_depth + 1):
            return True
    return False


def _expanded_shell_command(
    tokens: list[str],
    assignments: Mapping[str, str],
) -> str | None:
    expanded: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            break
        match = _SHELL_VARIABLE.fullmatch(token)
        if match is None:
            expanded.append(token)
            continue
        value = assignments.get(match.group(1) or match.group(2))
        if value is None:
            return None
        expanded.append(value)
    return " ".join(expanded)


def _is_shell_command_position(tokens: list[str], index: int) -> bool:
    if index == 0 or tokens[index - 1] in _SHELL_SEPARATORS:
        return True
    segment_start = max(
        (position for position in range(index) if tokens[position] in _SHELL_SEPARATORS),
        default=-1,
    ) + 1
    prefix = tokens[segment_start:index]
    wrapper_seen = False
    for token in prefix:
        if _SHELL_ASSIGNMENT.fullmatch(token) is not None:
            continue
        if wrapper_seen and token.startswith("-"):
            continue
        if Path(token).name in _SHELL_COMMAND_WRAPPERS:
            wrapper_seen = True
            continue
        return False
    return True


def _kill_process_tree(process: subprocess.Popen[bytes]) -> str:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return "posix_process_group_sigkill"
        except ProcessLookupError:
            return "already_exited"
        except OSError:
            pass
    elif os.name == "nt":  # pragma: no cover - exercised on Windows
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=SHELL_CLEANUP_GRACE_SECONDS,
            )
            if completed.returncode == 0:
                return "windows_process_tree_taskkill"
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        process.kill()
        return "process_kill_fallback"
    except ProcessLookupError:
        return "already_exited"


def _terminate_execution(
    process: subprocess.Popen[bytes],
    decision: ShellExecutionDecision,
) -> str:
    methods: list[str] = []
    if decision.termination_callback is not None:
        try:
            decision.termination_callback("KILL")
            methods.append("backend_kill")
        except Exception as exc:
            methods.append(f"backend_kill_failed:{type(exc).__name__}")
    methods.append(_kill_process_tree(process))
    return ",".join(methods)


def _wait_for_terminated_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(timeout=SHELL_CLEANUP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        try:
            process.wait(timeout=SHELL_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return


def _redirects_outside_root(command: str, root: Path) -> bool:
    for match in re.finditer(r"(?:\d?>{1,2}|&>)\s*([^\s;&|]+)", command):
        raw_path = match.group(1).strip("'\"")
        if raw_path.startswith("$"):
            return True
        candidate = (root / raw_path).resolve() if not raw_path.startswith("/") else Path(raw_path).resolve()
        if candidate != root and root not in candidate.parents:
            return True
    return False
