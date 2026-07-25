"""Managed process lifecycle, isolation, and bounded I/O."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from chulk import (
    ExecutionSessionRequest,
    EnvironmentPolicy,
    HostExecutionBackend,
    ProcessHandle,
    ProcessLogsRequest,
    ProcessPolicy,
    ProcessPollRequest,
    ProcessStartRequest,
    ProcessState,
    ProcessTerminateRequest,
    ProcessWriteRequest,
    SecretPolicy,
    TemporaryWorkspaceBackend,
    ToolContext,
    WorkspacePersistence,
)
from chulk.tools import create_default_tool_registry, process_tools


def _python_command(code: str) -> str:
    arguments = [sys.executable, "-c", code]
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def _wait_for_exit(session, handle: ProcessHandle, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = session.poll_process(ProcessPollRequest(handle))
        if result.value.state is not ProcessState.RUNNING:
            return result
        time.sleep(0.01)
    raise AssertionError("managed process did not exit")


def _wait_for_log(session, handle: ProcessHandle, expected: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = session.read_process_logs(ProcessLogsRequest(handle))
        if expected in (result.stdout or ""):
            return result
        time.sleep(0.01)
    raise AssertionError(f"managed process did not emit {expected!r}")


def test_host_process_lifecycle_and_cursor_logs(tmp_path: Path) -> None:
    backend = HostExecutionBackend(tmp_path)
    session = backend.open_session(
        ExecutionSessionRequest(conversation_id="conversation", turn_id="turn")
    )

    started = session.start_process(
        ProcessStartRequest(
            _python_command(
                "import sys,time;"
                "print('out', flush=True);"
                "print('err', file=sys.stderr, flush=True);"
                "time.sleep(.05)"
            )
        )
    )

    assert started.success is True
    assert isinstance(started.value, ProcessHandle)
    handle = started.value
    finished = _wait_for_exit(session, handle)
    logs = _wait_for_log(session, handle, "out")
    next_logs = session.read_process_logs(
        ProcessLogsRequest(handle, cursor=logs.value.next_cursor)
    )

    assert finished.value.state is ProcessState.EXITED
    assert finished.exit_code == 0
    assert {(entry.stream, entry.text) for entry in logs.value.entries} == {
        ("stdout", "out\n"),
        ("stderr", "err\n"),
    }
    assert next_logs.value.entries == ()
    assert started.metadata["execution_backend"] == "host"
    backend.close()


def test_interactive_process_accepts_bounded_stdin(tmp_path: Path) -> None:
    policy = ProcessPolicy(max_write_bytes=16)
    backend = HostExecutionBackend(tmp_path, process_policy=policy)
    session = backend.open_session(ExecutionSessionRequest(turn_id="interactive"))
    started = session.start_process(
        ProcessStartRequest(
            _python_command(
                "import sys; line=sys.stdin.readline(); print(line.upper(), end='')"
            ),
            interactive=True,
        )
    )
    handle = started.value

    too_large = session.write_process(ProcessWriteRequest(handle, "x" * 17))
    written = session.write_process(
        ProcessWriteRequest(handle, "hello\n", close_stdin=True)
    )
    _wait_for_exit(session, handle)
    logs = _wait_for_log(session, handle, "HELLO")

    assert too_large.success is False
    assert too_large.error == "process_write_too_large"
    assert written.success is True
    assert logs.stdout == "HELLO\n"
    backend.close()


def test_process_handles_are_scoped_to_conversation(tmp_path: Path) -> None:
    backend = HostExecutionBackend(tmp_path)
    owner = backend.open_session(
        ExecutionSessionRequest(conversation_id="owner", turn_id="first")
    )
    other = backend.open_session(
        ExecutionSessionRequest(conversation_id="other", turn_id="second")
    )
    resumed = backend.open_session(
        ExecutionSessionRequest(conversation_id="owner", turn_id="third")
    )
    started = owner.start_process(
        ProcessStartRequest(_python_command("import time; time.sleep(5)"))
    )
    handle = started.value

    denied = other.poll_process(ProcessPollRequest(handle))
    allowed = resumed.poll_process(ProcessPollRequest(handle))
    terminated = resumed.terminate_process(ProcessTerminateRequest(handle))

    assert denied.success is False
    assert denied.error == "process_handle_unavailable"
    assert allowed.success is True
    assert terminated.value.state is ProcessState.TERMINATED
    backend.close()


def test_child_process_scope_includes_parent_conversation(tmp_path: Path) -> None:
    backend = HostExecutionBackend(tmp_path)
    first = backend.open_session(
        ExecutionSessionRequest(
            conversation_id="first",
            turn_id="one",
            metadata={"child_task_id": "shared-child-id"},
        )
    )
    second = backend.open_session(
        ExecutionSessionRequest(
            conversation_id="second",
            turn_id="two",
            metadata={"child_task_id": "shared-child-id"},
        )
    )
    started = first.start_process(
        ProcessStartRequest(_python_command("import time; time.sleep(5)"))
    )

    denied = second.poll_process(ProcessPollRequest(started.value))
    first.close()
    resumed = backend.open_session(
        ExecutionSessionRequest(
            conversation_id="first",
            turn_id="three",
            metadata={"child_task_id": "shared-child-id"},
        )
    )
    cleaned = resumed.poll_process(ProcessPollRequest(started.value))

    assert denied.success is False
    assert denied.error == "process_handle_unavailable"
    assert cleaned.error == "process_handle_unavailable"
    backend.close()


def test_process_limit_is_enforced_per_owner(tmp_path: Path) -> None:
    backend = HostExecutionBackend(
        tmp_path,
        process_policy=ProcessPolicy(max_processes_per_owner=1),
    )
    session = backend.open_session(
        ExecutionSessionRequest(conversation_id="limited", turn_id="first")
    )
    first = session.start_process(
        ProcessStartRequest(_python_command("import time; time.sleep(5)"))
    )
    second = session.start_process(
        ProcessStartRequest(_python_command("print('must not start')"))
    )

    assert first.success is True
    assert second.success is False
    assert second.error == "process_limit_reached"
    session.terminate_process(ProcessTerminateRequest(first.value))
    backend.close()


def test_process_timeout_and_log_truncation(tmp_path: Path) -> None:
    backend = HostExecutionBackend(
        tmp_path,
        process_policy=ProcessPolicy(
            max_runtime_seconds=1,
            max_log_bytes=8,
            default_log_read_bytes=8,
            max_log_read_bytes=8,
        ),
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="bounded"))
    started = session.start_process(
        ProcessStartRequest(
            _python_command(
                "import time; print('0123456789', flush=True); time.sleep(5)"
            )
        )
    )
    handle = started.value

    finished = _wait_for_exit(session, handle)
    logs = _wait_for_log(session, handle, "456789")

    assert finished.value.state is ProcessState.TIMED_OUT
    assert finished.value.termination_reason == "timeout"
    assert logs.value.truncated is True
    assert logs.value.cursor > 0
    assert logs.stdout is not None
    assert logs.stdout.endswith("456789\n")
    assert len(logs.stdout.encode("utf-8")) <= 8
    backend.close()


def test_temporary_workspace_cleanup_terminates_process(tmp_path: Path) -> None:
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        persistence=WorkspacePersistence.PERSISTENT,
        require_shell_containment=False,
    )
    session = backend.open_session(
        ExecutionSessionRequest(conversation_id="owner", turn_id="temporary")
    )
    workspace_id = session.workspace.workspace_id
    workspace_root = session.project_root
    started = session.start_process(
        ProcessStartRequest(_python_command("import time; time.sleep(5)"))
    )
    handle = started.value

    session.close()
    assert workspace_root.exists()
    backend.cleanup_workspace(workspace_id)

    assert workspace_root.exists() is False
    replacement = backend.open_session(
        ExecutionSessionRequest(conversation_id="owner", turn_id="replacement")
    )
    missing = replacement.poll_process(ProcessPollRequest(handle))
    assert missing.error == "process_handle_unavailable"
    backend.close()


def test_network_denial_never_falls_back_to_host_process(tmp_path: Path) -> None:
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        network_policy="deny",
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="network"))

    result = session.start_process(
        ProcessStartRequest(_python_command("print('must not run')"))
    )

    assert result.success is False
    assert result.error == "network_policy_unsupported"
    assert result.metadata["child_process_started"] is False
    backend.close()


def test_temporary_process_environment_does_not_leak_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VISIBLE_VALUE", "visible")
    monkeypatch.setenv("PRIVATE_TOKEN", "hidden")
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        environment_policy=EnvironmentPolicy(
            allowed_names=(
                "PATH",
                "SYSTEMROOT",
                "COMSPEC",
                "PATHEXT",
                "WINDIR",
                "TMP",
                "TEMP",
                "VISIBLE_VALUE",
                "PRIVATE_TOKEN",
            )
        ),
        secret_policy=SecretPolicy(),
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="environment"))
    started = session.start_process(
        ProcessStartRequest(
            _python_command(
                "import os;"
                "print(os.environ.get('VISIBLE_VALUE'));"
                "print(os.environ.get('PRIVATE_TOKEN'))"
            )
        )
    )
    _wait_for_exit(session, started.value)
    logs = _wait_for_log(session, started.value, "visible")

    assert logs.stdout == "visible\nNone\n"
    backend.close()


def test_backend_close_kills_process_tree(tmp_path: Path) -> None:
    backend = HostExecutionBackend(
        tmp_path,
        process_policy=ProcessPolicy(termination_grace_seconds=1),
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="cleanup"))
    marker = tmp_path / "late.txt"
    started = session.start_process(
        ProcessStartRequest(
            _python_command(
                "from pathlib import Path; import time;"
                "time.sleep(1.5); Path('late.txt').write_text('late')"
            )
        )
    )

    backend.close()
    time.sleep(1.6)

    assert started.success is True
    assert marker.exists() is False


def test_managed_process_tools_route_through_active_session(tmp_path: Path) -> None:
    backend = HostExecutionBackend(tmp_path)
    session = backend.open_session(
        ExecutionSessionRequest(conversation_id="tools", turn_id="start")
    )
    registry = create_default_tool_registry(tmp_path)
    context = ToolContext(execution_session=session)

    started = registry.run(
        "process_start",
        {"command": _python_command("print('tool-output')")},
        context=context,
    )
    handle_arguments = {
        "process_id": started.metadata["process_id"],
        "backend_name": started.metadata["backend_name"],
        "workspace_id": started.metadata["workspace_id"],
    }
    deadline = time.monotonic() + 5
    while True:
        polled = registry.run("process_poll", handle_arguments, context=context)
        if polled.metadata["process_state"] != "running":
            break
        if time.monotonic() >= deadline:
            raise AssertionError("managed process tool did not finish")
        time.sleep(0.01)
    logs = registry.run("process_logs", handle_arguments, context=context)

    assert started.success is True
    assert polled.exit_code == 0
    assert logs.stdout == "tool-output\n"
    assert {tool.name for tool in process_tools()} == {
        "process_start",
        "process_poll",
        "process_logs",
        "process_write",
        "process_terminate",
    }
    backend.close()


def test_managed_process_tools_require_execution_session() -> None:
    start_tool = next(tool for tool in process_tools() if tool.name == "process_start")

    result = start_tool.callable({"command": "echo unavailable"}, None)

    assert result.error == "execution_session_required"
