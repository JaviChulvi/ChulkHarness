"""Docker containment policy and optional integration coverage."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from chulk import (
    CommandExecutionRequest,
    DockerExecutionBackend,
    DockerPolicy,
    DockerUnavailableError,
    EnvironmentPolicy,
    ExecutionSessionRequest,
    NetworkPolicy,
    ProcessHandle,
    ProcessStartRequest,
    SecretPolicy,
    WorkspacePolicyError,
)
from chulk.execution.processes import PreparedProcess
from chulk.tools.registry import ToolResult
from chulk.tools.shell import ShellExecutionRequest


def _completed(
    arguments: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(arguments, returncode, stdout, stderr)


def _fake_cli(backend, captured):
    def run(arguments, *, timeout_seconds):
        command = list(arguments)
        captured.append((command, timeout_seconds))
        if command[0] == "run":
            if "--env-file" in command:
                path = Path(command[command.index("--env-file") + 1])
                captured.append((["env-content", path.read_text(encoding="utf-8")], 0))
            return _completed(command, stdout="container-id\n")
        if command[0] == "inspect":
            return _completed(command, stdout="true\n")
        return _completed(command, stdout="available\n")

    backend._run_cli = run


def test_docker_container_uses_locked_down_flags_and_cleans_up(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "seed.txt").write_text("seed", encoding="utf-8")
    backend = DockerExecutionBackend(
        tmp_path,
        docker_policy=DockerPolicy(image="local/test:latest", user="1000:1000"),
    )
    captured = []
    monkeypatch.setattr("chulk.execution.docker.shutil.which", lambda _name: "/docker")
    _fake_cli(backend, captured)

    session = backend.open_session(ExecutionSessionRequest(turn_id="docker"))
    run_command = next(command for command, _timeout in captured if command[0] == "run")

    assert session.workspace.containment.value == "contained"
    assert session.workspace.mode.value == "container"
    assert session.policy.network is NetworkPolicy.DENY
    assert ["--cap-drop", "ALL"] == run_command[
        run_command.index("--cap-drop") : run_command.index("--cap-drop") + 2
    ]
    assert "--read-only" in run_command
    assert "no-new-privileges=true" in run_command
    assert ["--network", "none"] == run_command[
        run_command.index("--network") : run_command.index("--network") + 2
    ]
    assert ["--user", "1000:1000"] == run_command[
        run_command.index("--user") : run_command.index("--user") + 2
    ]
    assert "--pids-limit" in run_command
    assert "--cpus" in run_command
    assert "--memory" in run_command
    assert "--memory-swap" in run_command
    assert "--tmpfs" in run_command
    assert "--mount" in run_command
    assert ["--entrypoint", "/bin/sh"] == run_command[
        run_command.index("--entrypoint") : run_command.index("--entrypoint") + 2
    ]
    assert "--pull" in run_command
    assert run_command[run_command.index("--pull") + 1] == "never"

    session.close()
    assert any(command[:2] == ["rm", "--force"] for command, _ in captured)
    backend.close()


def test_docker_environment_requires_explicit_secret_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SAFE_VALUE", "safe")
    monkeypatch.setenv("SERVICE_TOKEN", "secret")
    monkeypatch.setattr("chulk.execution.docker.shutil.which", lambda _name: "/docker")
    backend = DockerExecutionBackend(
        tmp_path,
        docker_policy=DockerPolicy(image="local/test", user="1000"),
        environment_policy=EnvironmentPolicy(
            allowed_names=("SAFE_VALUE", "SERVICE_TOKEN")
        ),
    )
    captured = []
    _fake_cli(backend, captured)

    session = backend.open_session(ExecutionSessionRequest(turn_id="environment"))
    environment = next(
        command[1]
        for command, _timeout in captured
        if command[0] == "env-content"
    )

    assert environment == "SAFE_VALUE=safe\n"
    run_command = next(command for command, _ in captured if command[0] == "run")
    environment_path = Path(run_command[run_command.index("--env-file") + 1])
    assert environment_path.exists() is False
    session.close()
    backend.close()

    approved = DockerExecutionBackend(
        tmp_path,
        docker_policy=DockerPolicy(image="local/test", user="1000"),
        environment_policy=EnvironmentPolicy(
            allowed_names=("SAFE_VALUE", "SERVICE_TOKEN")
        ),
        secret_policy=SecretPolicy(
            allowed_environment_names=("SERVICE_TOKEN",)
        ),
    )
    approved_captured = []
    _fake_cli(approved, approved_captured)
    approved_session = approved.open_session(
        ExecutionSessionRequest(turn_id="approved")
    )
    approved_environment = next(
        command[1]
        for command, _timeout in approved_captured
        if command[0] == "env-content"
    )
    assert approved_environment == "SAFE_VALUE=safe\nSERVICE_TOKEN=secret\n"
    approved_session.close()
    approved.close()


def test_docker_unavailability_is_diagnostic_and_never_materializes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("chulk.execution.docker.shutil.which", lambda _name: None)
    backend = DockerExecutionBackend(
        tmp_path,
        docker_policy=DockerPolicy(image="local/test"),
    )

    with pytest.raises(DockerUnavailableError) as error:
        backend.open_session(ExecutionSessionRequest(turn_id="unavailable"))

    assert error.value.code == "docker_cli_unavailable"
    assert backend._sessions == {}
    assert backend._containers == {}
    backend.close()


def test_docker_reports_daemon_and_image_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("chulk.execution.docker.shutil.which", lambda _name: "/docker")
    backend = DockerExecutionBackend(
        tmp_path,
        docker_policy=DockerPolicy(image="local/test"),
    )

    backend._run_cli = lambda arguments, timeout_seconds: _completed(
        list(arguments),
        returncode=1,
    )
    with pytest.raises(DockerUnavailableError) as daemon_error:
        backend.open_session(ExecutionSessionRequest(turn_id="daemon"))
    assert daemon_error.value.code == "docker_daemon_unavailable"

    def image_missing(arguments, *, timeout_seconds):
        command = list(arguments)
        return _completed(
            command,
            returncode=1 if command[:2] == ["image", "inspect"] else 0,
        )

    backend._run_cli = image_missing
    with pytest.raises(DockerUnavailableError) as image_error:
        backend.open_session(ExecutionSessionRequest(turn_id="image"))
    assert image_error.value.code == "docker_image_unavailable"
    backend.close()


def test_docker_removes_container_when_startup_probe_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("chulk.execution.docker.shutil.which", lambda _name: "/docker")
    backend = DockerExecutionBackend(
        tmp_path,
        docker_policy=DockerPolicy(image="local/test"),
    )
    captured = []

    def stopped_container(arguments, *, timeout_seconds):
        command = list(arguments)
        captured.append(command)
        if command[0] == "run":
            return _completed(command, stdout="stopped-id\n")
        if command[0] == "inspect":
            return _completed(command, stdout="false\n")
        return _completed(command, stdout="available\n")

    backend._run_cli = stopped_container

    with pytest.raises(WorkspacePolicyError) as error:
        backend.open_session(ExecutionSessionRequest(turn_id="stopped"))

    assert error.value.code == "docker_container_start_failed"
    assert ["rm", "--force", "stopped-id"] in captured
    assert backend._containers == {}
    backend.close()


@pytest.mark.parametrize("user", ["root", "0", "0:1000", "000"])
def test_docker_rejects_root_container_user(tmp_path: Path, user: str) -> None:
    with pytest.raises(ValueError, match="non-root"):
        DockerExecutionBackend(
            tmp_path,
            docker_policy=DockerPolicy(image="local/test", user=user),
        )


def test_docker_rejects_implicit_environment_forwarding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit allowlist"):
        DockerExecutionBackend(
            tmp_path,
            docker_policy=DockerPolicy(image="local/test"),
            environment_policy=EnvironmentPolicy(inherit_all=True),
        )


def test_docker_command_and_managed_process_transports_are_contained(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("chulk.execution.docker.shutil.which", lambda _name: "/docker")
    backend = DockerExecutionBackend(
        tmp_path,
        docker_policy=DockerPolicy(image="local/test", user="1000"),
    )
    captured_cli = []
    _fake_cli(backend, captured_cli)
    session = backend.open_session(ExecutionSessionRequest(turn_id="transport"))
    decision = session.shell_execution_policy.prepare(
        ShellExecutionRequest(
            command="printf contained",
            cwd=session.project_root,
            timeout_seconds=1,
            stdout_limit_bytes=100,
            stderr_limit_bytes=100,
        )
    )
    captured_prepared: list[PreparedProcess] = []

    def capture_start(**kwargs):
        prepared = kwargs["prepare"]("process-test")
        assert isinstance(prepared, PreparedProcess)
        captured_prepared.append(prepared)
        return ToolResult(
            tool_name="process.start",
            success=True,
            observation="captured",
            value=ProcessHandle(
                "process-test",
                backend.name,
                session.workspace.workspace_id,
            ),
        )

    monkeypatch.setattr(backend.process_registry, "start", capture_start)
    result = session.start_process(
        ProcessStartRequest("printf managed", interactive=True)
    )

    assert result.success is True
    assert decision.containment_applied is True
    assert decision.termination_callback is not None
    assert decision.command[:3] == ("docker", "exec", "container-id")
    decision.termination_callback("KILL")
    prepared = captured_prepared[0]
    assert prepared.shell is False
    assert prepared.command[:4] == (
        "docker",
        "exec",
        "--interactive",
        "container-id",
    )
    assert prepared.signal_callback is not None
    prepared.signal_callback("TERM")
    assert any(
        command[:3] == ["exec", "container-id", "/bin/sh"]
        for command, _timeout in captured_cli
    )
    session.close()
    backend.close()


_DOCKER_TEST_IMAGE = os.environ.get("CHULK_DOCKER_TEST_IMAGE")


@pytest.mark.skipif(
    os.environ.get("CHULK_RUN_DOCKER_TESTS") != "1" or not _DOCKER_TEST_IMAGE,
    reason="Set CHULK_RUN_DOCKER_TESTS=1 and CHULK_DOCKER_TEST_IMAGE to run",
)
def test_docker_backend_integration(tmp_path: Path) -> None:
    backend = DockerExecutionBackend(
        tmp_path,
        docker_policy=DockerPolicy(image=_DOCKER_TEST_IMAGE or ""),
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="integration"))

    result = session.run_command(CommandExecutionRequest("printf docker-ok"))

    assert result.success is True
    assert result.stdout == "docker-ok"
    assert result.metadata["containment"] == "contained"
    session.close()
    backend.close()
