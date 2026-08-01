"""Optional Docker execution with explicit containment and no host fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
from uuid import uuid4

from chulk.execution.base import ExecutionSession
from chulk.execution.host import HostExecutionSession, has_child_task_scope
from chulk.execution.models import (
    CommandExecutionRequest,
    ContainmentStatus,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSessionRequest,
    ExecutionWorkspace,
    NetworkPolicy,
    ProcessStartRequest,
    WorkspaceMode,
)
from chulk.execution.policy import (
    DockerPolicy,
    EnvironmentPolicy,
    ProcessPolicy,
    ResourcePolicy,
    SecretPolicy,
)
from chulk.execution.processes import PreparedProcess, owner_key, prepare_host_process
from chulk.execution.temporary import (
    TemporaryWorkspaceBackend,
    TemporaryWorkspaceSession,
    WorkspacePolicyError,
    _FileSnapshot,
)
from chulk.tools.registry import ToolResult
from chulk.tools.shell import (
    ShellExecutionDecision,
    ShellExecutionPolicy,
    ShellExecutionRequest,
)


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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


class DockerUnavailableError(WorkspacePolicyError):
    """Docker CLI, daemon, or configured image is unavailable."""


class DockerExecutionBackend(TemporaryWorkspaceBackend):
    """Run commands in a locked-down container over a reviewable workspace."""

    name = "docker"

    def __init__(
        self,
        project_root: Path,
        *,
        docker_policy: DockerPolicy,
        environment_policy: EnvironmentPolicy | None = None,
        secret_policy: SecretPolicy | None = None,
        resource_policy: ResourcePolicy | None = None,
        process_policy: ProcessPolicy | None = None,
        network_policy: NetworkPolicy = NetworkPolicy.DENY,
        **kwargs: Any,
    ) -> None:
        _validate_container_user(docker_policy.user)
        selected_environment_policy = environment_policy or EnvironmentPolicy()
        if selected_environment_policy.inherit_all:
            raise ValueError(
                "Docker environment forwarding requires an explicit allowlist"
            )
        requested_containment = kwargs.pop("require_shell_containment", True)
        if requested_containment is not True:
            raise ValueError("Docker execution always requires containment")
        super().__init__(
            project_root,
            environment_policy=selected_environment_policy,
            secret_policy=secret_policy,
            resource_policy=resource_policy,
            process_policy=process_policy,
            network_policy=network_policy,
            require_shell_containment=True,
            **kwargs,
        )
        self.docker_policy = docker_policy
        self._containers: dict[str, str] = {}

    def open_session(self, request: ExecutionSessionRequest) -> ExecutionSession:
        self._assert_available()
        return super().open_session(request)

    @property
    def _workspace_mode(self) -> WorkspaceMode:
        return WorkspaceMode.CONTAINER

    @property
    def _workspace_containment(self) -> ContainmentStatus:
        return ContainmentStatus.CONTAINED

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
        del shell_policy
        container_id = self._start_container(workspace_root)
        self._containers[workspace.workspace_id] = container_id
        return DockerExecutionSession(
            self,
            workspace=workspace,
            policy=policy,
            project_root=workspace_root,
            shell_execution_policy=_DockerExecPolicy(self, container_id),
            base_snapshot=base_snapshot,
            process_owner_key=owner_key(
                conversation_id=request.conversation_id,
                turn_id=request.turn_id,
                metadata=request.metadata,
                workspace_id=workspace.workspace_id,
            ),
            cleanup_process_owner_on_close=has_child_task_scope(request),
            container_id=container_id,
        )

    def _cleanup_materialized_workspace(
        self,
        workspace_root: Path,
        workspace_id: str,
    ) -> None:
        container_id = self._containers.get(workspace_id)
        failure: WorkspacePolicyError | None = None
        if container_id is not None:
            try:
                result = self._run_cli(
                    ("rm", "--force", container_id),
                    timeout_seconds=self.docker_policy.stop_timeout_seconds,
                )
                if result.returncode != 0:
                    failure = WorkspacePolicyError(
                        "Docker container cleanup failed.",
                        code="docker_cleanup_failed",
                    )
                else:
                    self._containers.pop(workspace_id, None)
            except (OSError, subprocess.TimeoutExpired) as exc:
                failure = WorkspacePolicyError(
                    f"Docker container cleanup failed: {type(exc).__name__}",
                    code="docker_cleanup_failed",
                )
        try:
            super()._cleanup_materialized_workspace(workspace_root, workspace_id)
        except Exception as exc:
            if failure is None:
                raise
            raise failure from exc
        if failure is not None:
            raise failure

    def _assert_available(self) -> None:
        binary = self.docker_policy.binary
        if shutil.which(binary) is None:
            raise DockerUnavailableError(
                f"Docker CLI is unavailable: {binary}",
                code="docker_cli_unavailable",
            )
        checks = (
            (
                ("version", "--format", "{{.Server.Version}}"),
                "docker_daemon_unavailable",
                "Docker daemon is unavailable.",
            ),
            (
                ("image", "inspect", self.docker_policy.image, "--format", "{{.Id}}"),
                "docker_image_unavailable",
                (
                    "Configured Docker image is unavailable locally; Chulk does not "
                    "pull images implicitly."
                ),
            ),
        )
        for arguments, code, message in checks:
            try:
                result = self._run_cli(
                    arguments,
                    timeout_seconds=self.docker_policy.availability_timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DockerUnavailableError(
                    f"{message} ({type(exc).__name__})",
                    code=code,
                ) from exc
            if result.returncode != 0:
                raise DockerUnavailableError(message, code=code)

    def _start_container(self, workspace_root: Path) -> str:
        mount_source = str(workspace_root)
        if "," in mount_source:
            raise WorkspacePolicyError(
                "Docker workspace paths containing commas are unsupported.",
                code="docker_mount_path_unsupported",
                path=mount_source,
            )
        environment = _container_environment(
            self.environment_policy,
            self.secret_policy,
        )
        container_user = _container_user(self.docker_policy.user)
        _prepare_workspace_access(workspace_root, container_user)
        environment_file = _write_environment_file(environment)
        container_name = "chulk-" + uuid4().hex
        command = [
            "run",
            "--detach",
            "--name",
            container_name,
            "--pull",
            "never",
            "--init",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges=true",
            "--pids-limit",
            str(self.docker_policy.pids_limit),
            "--cpus",
            str(self.docker_policy.cpus),
            "--memory",
            str(self.docker_policy.memory_bytes),
            "--memory-swap",
            str(self.docker_policy.memory_bytes),
            "--tmpfs",
            (
                "/tmp:rw,noexec,nosuid,nodev,size="
                f"{self.docker_policy.tmpfs_bytes}"
            ),
            "--mount",
            f"type=bind,source={mount_source},target=/workspace",
            "--workdir",
            "/workspace",
            "--user",
            container_user,
            "--entrypoint",
            "/bin/sh",
        ]
        if self.network_policy is NetworkPolicy.DENY:
            command.extend(("--network", "none"))
        else:
            command.extend(("--network", "host"))
        if environment_file is not None:
            command.extend(("--env-file", str(environment_file)))
        command.extend(
            (
                self.docker_policy.image,
                "-c",
                "while :; do sleep 3600; done",
            )
        )
        try:
            try:
                result = self._run_cli(
                    command,
                    timeout_seconds=self.docker_policy.availability_timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._remove_container_best_effort(container_name)
                raise WorkspacePolicyError(
                    f"Docker container failed to start: {type(exc).__name__}",
                    code="docker_container_start_failed",
                ) from exc
        finally:
            if environment_file is not None:
                environment_file.unlink(missing_ok=True)
        container_id = result.stdout.strip()
        if result.returncode != 0 or not container_id:
            self._remove_container_best_effort(container_name)
            raise WorkspacePolicyError(
                "Docker container failed to start.",
                code="docker_container_start_failed",
            )
        try:
            probe = self._run_cli(
                ("inspect", "--format", "{{.State.Running}}", container_id),
                timeout_seconds=self.docker_policy.availability_timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._remove_container_best_effort(container_id)
            raise WorkspacePolicyError(
                f"Docker container startup could not be verified: {type(exc).__name__}",
                code="docker_container_start_failed",
            ) from exc
        if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
            self._remove_container_best_effort(container_id)
            raise WorkspacePolicyError(
                "Docker container exited during startup.",
                code="docker_container_start_failed",
            )
        return container_id

    def _remove_container_best_effort(self, container: str) -> None:
        try:
            self._run_cli(
                ("rm", "--force", container),
                timeout_seconds=self.docker_policy.stop_timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _run_cli(
        self,
        arguments: Sequence[str],
        *,
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.docker_policy.binary, *arguments],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )


class DockerExecutionSession(TemporaryWorkspaceSession):
    """Temporary session whose command and process transports use Docker."""

    def __init__(
        self,
        backend: DockerExecutionBackend,
        *,
        container_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(backend, **kwargs)
        self._docker_backend = backend
        self.container_id = container_id

    def run_command(self, request: CommandExecutionRequest) -> ExecutionResult:
        return self._with_change_set(HostExecutionSession.run_command(self, request))

    def start_process(self, request: ProcessStartRequest) -> ExecutionResult:
        self._ensure_open()
        registry = self._docker_backend.process_registry

        def prepare(process_id: str) -> PreparedProcess | ToolResult:
            prepared = prepare_host_process(
                request=request,
                cwd=self.project_root,
                default_timeout_seconds=registry.policy.max_runtime_seconds,
                output_limit_bytes=registry.policy.max_log_bytes,
                execution_policy=_DockerManagedProcessPolicy(
                    self._docker_backend,
                    self.container_id,
                    process_id,
                    interactive=request.interactive,
                ),
                require_containment=True,
            )
            if isinstance(prepared, ToolResult):
                return prepared
            return replace(
                prepared,
                signal_callback=lambda signal_name: self._signal_process(
                    process_id,
                    signal_name,
                ),
            )

        result = registry.start(
            owner_key=self.process_owner_key,
            workspace_id=self.workspace.workspace_id,
            backend_name=self.workspace.backend_name,
            request=request,
            prepare=prepare,
        )
        return self._normalize(result)

    def _signal_process(self, process_id: str, signal_name: str) -> None:
        _signal_container_process(
            self._docker_backend,
            self.container_id,
            f"/tmp/{process_id}.pid",
            signal_name,
        )


class _DockerExecPolicy:
    def __init__(
        self,
        backend: DockerExecutionBackend,
        container_id: str,
    ) -> None:
        self.backend = backend
        self.container_id = container_id

    def prepare(self, request: ShellExecutionRequest) -> ShellExecutionDecision:
        process_id = "command-" + uuid4().hex
        pid_file = f"/tmp/{process_id}.pid"
        return ShellExecutionDecision.allow(
            (
                self.backend.docker_policy.binary,
                "exec",
                self.container_id,
                "/bin/sh",
                "-c",
                'echo $$ > "$1"; exec /bin/sh -lc "$2"',
                "chulk-command",
                pid_file,
                request.command,
            ),
            policy_name="docker-contained",
            shell=False,
            containment_applied=True,
            termination_callback=lambda signal_name: _signal_container_process(
                self.backend,
                self.container_id,
                pid_file,
                signal_name,
            ),
        )


class _DockerManagedProcessPolicy(_DockerExecPolicy):
    def __init__(
        self,
        backend: DockerExecutionBackend,
        container_id: str,
        process_id: str,
        *,
        interactive: bool,
    ) -> None:
        super().__init__(backend, container_id)
        self.process_id = process_id
        self.interactive = interactive

    def prepare(self, request: ShellExecutionRequest) -> ShellExecutionDecision:
        command = [self.backend.docker_policy.binary, "exec"]
        if self.interactive:
            command.append("--interactive")
        command.extend(
            (
                self.container_id,
                "/bin/sh",
                "-c",
                'echo $$ > "$1"; exec /bin/sh -lc "$2"',
                "chulk-process",
                f"/tmp/{self.process_id}.pid",
                request.command,
            )
        )
        return ShellExecutionDecision.allow(
            command,
            policy_name="docker-contained-process",
            shell=False,
            containment_applied=True,
        )


def _container_environment(
    environment_policy: EnvironmentPolicy,
    secret_policy: SecretPolicy,
) -> dict[str, str]:
    environment = environment_policy.build()
    allowed_secrets = set(secret_policy.allowed_environment_names)
    result: dict[str, str] = {}
    for name, value in environment.items():
        lowered = name.lower()
        looks_secret = any(marker in lowered for marker in _SECRET_NAME_MARKERS)
        if looks_secret and name not in allowed_secrets:
            continue
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise WorkspacePolicyError(
                f"Invalid Docker environment name: {name}",
                code="docker_environment_invalid",
            )
        if "\x00" in value or "\n" in value or "\r" in value:
            raise WorkspacePolicyError(
                f"Docker environment value cannot be represented safely: {name}",
                code="docker_environment_invalid",
            )
        result[name] = value
    return result


def _write_environment_file(environment: Mapping[str, str]) -> Path | None:
    if not environment:
        return None
    descriptor, raw_path = tempfile.mkstemp(prefix="chulk-docker-env-")
    path = Path(raw_path)
    try:
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for name, value in environment.items():
                handle.write(f"{name}={value}\n")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        path.unlink(missing_ok=True)
        raise
    return path


def _container_user(configured_user: str | None) -> str:
    if configured_user is not None:
        return configured_user.strip()
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if callable(getuid) and callable(getgid):
        uid = getuid()
        gid = getgid()
        if uid != 0:
            return f"{uid}:{gid}"
    return "65532:65532"


def _validate_container_user(configured_user: str | None) -> None:
    if configured_user is None:
        return
    user = configured_user.split(":", 1)[0].strip().lower()
    if user == "root" or (user.isdigit() and int(user) == 0):
        raise ValueError("Docker execution requires a non-root user")


def _prepare_workspace_access(workspace_root: Path, container_user: str) -> None:
    getuid = getattr(os, "getuid", None)
    if not callable(getuid) or getuid() != 0:
        return
    user, separator, group = container_user.partition(":")
    if not user.isdigit() or (separator and not group.isdigit()):
        raise WorkspacePolicyError(
            "Root hosts require a numeric Docker user for writable workspace access.",
            code="docker_numeric_user_required",
        )
    uid = int(user)
    gid = int(group) if separator else uid
    for path in (workspace_root, *workspace_root.rglob("*")):
        os.chown(path, uid, gid, follow_symlinks=False)


def _signal_container_process(
    backend: DockerExecutionBackend,
    container_id: str,
    pid_file: str,
    signal_name: str,
) -> None:
    script = 'test -f "$1" || exit 42; kill -"$2" "$(cat "$1")"'
    deadline = time.monotonic() + 0.5
    while True:
        result = backend._run_cli(
            (
                "exec",
                container_id,
                "/bin/sh",
                "-c",
                script,
                "chulk-signal",
                pid_file,
                signal_name,
            ),
            timeout_seconds=backend.docker_policy.stop_timeout_seconds,
        )
        if result.returncode == 0:
            return
        if result.returncode != 42 or time.monotonic() >= deadline:
            raise RuntimeError("Docker process signal failed")
        time.sleep(0.02)
