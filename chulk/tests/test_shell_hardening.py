"""Focused regression tests for bounded and policy-controlled shell execution."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import sys
import time

import pytest

from chulk.capabilities import Capabilities, FileAccess, MemoryMode
from chulk.tools import create_default_tool_registry
from chulk.tools.shell import (
    ShellExecutionDecision,
    ShellExecutionRequest,
    run_shell_command,
)


def test_stdout_is_bounded_while_child_is_running(tmp_path: Path) -> None:
    command = _python_command(
        "import os, time; os.write(1, b'HEAD-' + (b'x' * 2000) + b'-TAIL'); time.sleep(30)"
    )

    started_at = time.monotonic()
    result = run_shell_command(
        {"command": command},
        tmp_path,
        stdout_limit_bytes=128,
        stderr_limit_bytes=64,
    )

    assert time.monotonic() - started_at < 5
    assert not result.success
    assert result.error == "output_limit_exceeded"
    assert result.failure_kind == "output_limit_exceeded"
    assert result.stdout is not None
    assert len(result.stdout) <= 128
    assert result.stdout.startswith("HEAD-")
    assert result.stdout.endswith("-TAIL")
    assert "shell output truncated" in result.stdout
    assert result.metadata["stdout_total_bytes"] > 128
    assert result.metadata["stdout_preview_bytes"] <= 128
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stdout_discarded_bytes"] > 0
    assert result.metadata["output_limit_streams"] == ["stdout"]
    assert result.metadata["termination_reason"] == "output_limit_exceeded"
    assert result.metadata["reader_threads_stopped"] is True
    if os.name == "posix":
        assert result.metadata["termination_method"] == "posix_process_group_sigkill"


def test_stderr_has_an_independent_live_byte_limit(tmp_path: Path) -> None:
    command = _python_command("import os, time; os.write(2, b'e' * 2000); time.sleep(30)")

    result = run_shell_command(
        {"command": command},
        tmp_path,
        stdout_limit_bytes=512,
        stderr_limit_bytes=73,
    )

    assert result.error == "output_limit_exceeded"
    assert result.stderr is not None
    assert len(result.stderr) <= 73
    assert result.metadata["stderr_total_bytes"] > 73
    assert result.metadata["stderr_preview_bytes"] <= 73
    assert result.metadata["stderr_truncated"] is True
    assert result.metadata["stdout_truncated"] is False
    assert result.metadata["output_limit_streams"] == ["stderr"]


def test_timeout_kills_group_and_preserves_bounded_partial_output(tmp_path: Path) -> None:
    command = _python_command("import time; print('partial', flush=True); time.sleep(30)")

    result = run_shell_command(
        {"command": command, "timeout_seconds": 1},
        tmp_path,
        default_timeout_seconds=1,
        stdout_limit_bytes=64,
        stderr_limit_bytes=64,
    )

    assert not result.success
    assert result.error == "timeout"
    assert result.stdout == "partial\n"
    assert result.metadata["termination_reason"] == "timeout"
    assert result.metadata["stdout_preview_bytes"] <= 64
    assert result.metadata["reader_threads_stopped"] is True
    if os.name == "posix":
        assert result.metadata["termination_method"] == "posix_process_group_sigkill"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_output_overflow_kills_descendant_processes(tmp_path: Path) -> None:
    overflow_command = _python_command("import os, time; os.write(1, b'x' * 2000); time.sleep(30)")
    command = (
        "sleep 30 & echo $! > child.pid; "
        f"{overflow_command}; wait"
    )

    result = run_shell_command(
        {"command": command},
        tmp_path,
        stdout_limit_bytes=64,
        stderr_limit_bytes=64,
    )

    child_pid = int((tmp_path / "child.pid").read_text(encoding="utf-8").strip())
    child_alive = _wait_for_process_exit(child_pid)
    if child_alive:
        os.kill(child_pid, signal.SIGKILL)
    assert result.error == "output_limit_exceeded"
    assert result.metadata["termination_method"] == "posix_process_group_sigkill"
    assert not child_alive


def test_required_containment_fails_closed_before_starting_child(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"

    result = run_shell_command(
        {"command": f"printf unsafe > {shlex.quote(str(marker))}"},
        tmp_path,
        require_containment=True,
    )

    assert not result.success
    assert result.error == "containment_required"
    assert result.metadata["child_process_started"] is False
    assert result.metadata["execution_policy"] == {
        "name": "direct-local",
        "containment_required": True,
        "containment_applied": False,
        "uses_shell": True,
    }
    assert not marker.exists()


def test_host_policy_can_supply_a_contained_non_shell_transport(tmp_path: Path) -> None:
    class RecordingPolicy:
        def __init__(self) -> None:
            self.requests: list[ShellExecutionRequest] = []

        def prepare(self, request: ShellExecutionRequest) -> ShellExecutionDecision:
            self.requests.append(request)
            return ShellExecutionDecision.allow(
                (sys.executable, "-c", "print('contained')"),
                policy_name="test-sandbox",
                shell=False,
                environment=os.environ,
                containment_applied=True,
            )

    policy = RecordingPolicy()

    result = run_shell_command(
        {"command": "ignored by host wrapper"},
        tmp_path,
        stdout_limit_bytes=91,
        stderr_limit_bytes=47,
        execution_policy=policy,
        require_containment=True,
    )

    assert result.success
    assert result.stdout == "contained\n"
    assert len(policy.requests) == 1
    assert policy.requests[0].command == "ignored by host wrapper"
    assert policy.requests[0].cwd == tmp_path.resolve()
    assert policy.requests[0].stdout_limit_bytes == 91
    assert policy.requests[0].stderr_limit_bytes == 47
    assert result.metadata["execution_policy"] == {
        "name": "test-sandbox",
        "containment_required": True,
        "containment_applied": True,
        "uses_shell": False,
    }


def test_host_policy_can_deny_without_starting_child(tmp_path: Path) -> None:
    class DenyingPolicy:
        def prepare(self, request: ShellExecutionRequest) -> ShellExecutionDecision:
            return ShellExecutionDecision.deny("tenant has no shell allocation", policy_name="tenant-policy")

    result = run_shell_command(
        {"command": "printf denied"},
        tmp_path,
        execution_policy=DenyingPolicy(),
    )

    assert not result.success
    assert result.error == "execution_policy_denied"
    assert "tenant has no shell allocation" in result.observation
    assert result.metadata["child_process_started"] is False


def test_default_registry_wires_host_configured_byte_limits(tmp_path: Path) -> None:
    capabilities = Capabilities(
        files=FileAccess.OFF,
        shell=True,
        memory=MemoryMode.OFF,
        utilities=False,
    )
    registry = create_default_tool_registry(
        tmp_path,
        capabilities=capabilities,
        max_tool_stdout_bytes=83,
        max_tool_stderr_bytes=41,
    )
    command = _python_command("import os, time; os.write(1, b'x' * 2000); time.sleep(30)")

    result = registry.run("run_cmd", {"command": command})

    assert result.error == "output_limit_exceeded"
    assert result.metadata["stdout_limit_bytes"] == 83
    assert result.metadata["stderr_limit_bytes"] == 41
    assert result.metadata["stdout_preview_bytes"] <= 83


@pytest.mark.parametrize(
    "command",
    [
        "true;rm -rf target",
        "sh -c 'rm -rf target'",
        'env FLAG=1 bash -lc "rm --recursive --force target"',
        "find . -exec rm -rf {} +",
    ],
)
def test_nested_recursive_force_rm_is_blocked(command: str, tmp_path: Path) -> None:
    result = run_shell_command({"command": command}, tmp_path)

    assert not result.success
    assert result.error == "blocked_command"
    assert result.metadata["child_process_started"] is False


def test_quoted_rm_text_remains_benign(tmp_path: Path) -> None:
    result = run_shell_command({"command": "printf 'rm -rf target'"}, tmp_path)

    assert result.success
    assert result.stdout == "rm -rf target"


def _python_command(source: str) -> str:
    if os.name == "nt":  # pragma: no cover - exercised on Windows
        import subprocess

        return subprocess.list2cmdline([sys.executable, "-c", source])
    return shlex.join((sys.executable, "-c", source))


def _wait_for_process_exit(pid: int) -> bool:
    for _ in range(30):
        if not _process_exists(pid):
            return False
        time.sleep(0.05)
    return True


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
