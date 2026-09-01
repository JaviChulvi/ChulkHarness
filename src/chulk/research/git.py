"""Structured, repository-root-bound Git and test-runner tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
import os
import re
import shlex
import shutil
import subprocess

from chulk.tools.permissions import ToolPermissionLevel
from chulk.redaction import redact_text
from chulk.tools.registry import Tool, ToolFailureKind, ToolResult


_SAFE_REMOTE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_SECRET_NAMES = frozenset(
    {
        ".env",
        ".envrc",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "service-account.json",
    }
)
_SECRET_MARKERS = frozenset(
    {
        "api_key",
        "apikey",
        "credential",
        "password",
        "private_key",
        "secret",
        "token",
    }
)


class GitPolicyError(PermissionError):
    """A structured Git operation was denied before Git was invoked."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class GitOperationError(RuntimeError):
    """Git returned a sanitized, bounded error."""

    def __init__(self, message: str, *, code: str, stderr: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.stderr = stderr


@dataclass(frozen=True, slots=True)
class GitPolicy:
    """Host-owned policy for bounded Git reads, writes, pushes, and tests."""

    command_timeout_seconds: int = 30
    test_timeout_seconds: int = 300
    max_output_chars: int = 40_000
    max_diff_chars: int = 100_000
    max_patch_chars: int = 250_000
    max_log_entries: int = 100
    allowed_test_commands: tuple[tuple[str, ...], ...] = (
        ("python", "-m", "pytest"),
        ("python", "-m", "compileall"),
        ("python", "-m", "ruff"),
        ("python", "-m", "mypy"),
    )
    allow_apply_patch: bool = True
    allow_branch_creation: bool = True
    allow_commit: bool = True
    allow_push: bool = False
    allowed_remotes: frozenset[str] = frozenset({"origin"})
    allow_submodule_repository: bool = False
    run_commit_hooks: bool = False

    def __post_init__(self) -> None:
        for name in (
            "command_timeout_seconds",
            "test_timeout_seconds",
            "max_output_chars",
            "max_diff_chars",
            "max_patch_chars",
            "max_log_entries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        commands = tuple(tuple(str(part) for part in command) for command in self.allowed_test_commands)
        if any(not command or any(not part for part in command) for command in commands):
            raise ValueError("allowed_test_commands cannot contain empty commands")
        object.__setattr__(self, "allowed_test_commands", commands)
        remotes = frozenset(remote.strip() for remote in self.allowed_remotes)
        if any(not _SAFE_REMOTE_NAME.fullmatch(remote) for remote in remotes):
            raise ValueError("allowed_remotes contains an invalid remote name")
        object.__setattr__(self, "allowed_remotes", remotes)


@dataclass(frozen=True, slots=True)
class GitRepository:
    root: str
    git_dir: str
    branch: str | None
    detached: bool
    is_linked_worktree: bool
    is_submodule: bool
    execution_workspace_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "git_dir": self.git_dir,
            "branch": self.branch,
            "detached": self.detached,
            "is_linked_worktree": self.is_linked_worktree,
            "is_submodule": self.is_submodule,
            "execution_workspace_id": self.execution_workspace_id,
        }


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    arguments: tuple[str, ...]
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool = False


class GitService:
    """Run only typed Git argv from one verified repository root."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        policy: GitPolicy | None = None,
        git_executable: str | None = None,
        execution_workspace_id: str | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.policy = policy or GitPolicy()
        self.git_executable = git_executable or shutil.which("git") or "git"
        self.execution_workspace_id = execution_workspace_id
        if not self.project_root.is_dir():
            raise ValueError("project_root must be an existing directory")
        root = self._run_raw(("rev-parse", "--show-toplevel")).stdout.strip()
        if Path(root).resolve() != self.project_root:
            raise GitPolicyError(
                "Git tools require project_root to be the repository root",
                code="repository_root_required",
            )
        superproject = self._run_raw(
            ("rev-parse", "--show-superproject-working-tree"),
            check=False,
        ).stdout.strip()
        if superproject and not self.policy.allow_submodule_repository:
            raise GitPolicyError(
                "Git tools deny submodule repositories by default",
                code="submodule_repository_denied",
            )

    @classmethod
    def from_execution_session(
        cls,
        session: object,
        *,
        policy: GitPolicy | None = None,
        git_executable: str | None = None,
    ) -> GitService:
        """Bind structured Git operations to an EP-01 execution workspace."""
        if bool(getattr(session, "closed", False)):
            raise GitPolicyError(
                "Cannot bind Git tools to a closed execution session",
                code="execution_session_closed",
            )
        root = getattr(session, "project_root", None)
        workspace = getattr(session, "workspace", None)
        workspace_id = getattr(workspace, "workspace_id", None)
        if root is None or not isinstance(workspace_id, str):
            raise TypeError(
                "execution session must expose project_root and workspace.workspace_id"
            )
        return cls(
            Path(root),
            policy=policy,
            git_executable=git_executable,
            execution_workspace_id=workspace_id,
        )

    def repository(self) -> GitRepository:
        branch_result = self._run_raw(
            ("symbolic-ref", "--quiet", "--short", "HEAD"),
            check=False,
        )
        branch = branch_result.stdout.strip() or None
        git_dir = Path(
            self._run_raw(("rev-parse", "--absolute-git-dir")).stdout.strip()
        ).resolve()
        common_dir = Path(
            self._run_raw(("rev-parse", "--path-format=absolute", "--git-common-dir")).stdout.strip()
        ).resolve()
        superproject = self._run_raw(
            ("rev-parse", "--show-superproject-working-tree"),
            check=False,
        ).stdout.strip()
        return GitRepository(
            root=str(self.project_root),
            git_dir=str(git_dir),
            branch=branch,
            detached=branch is None,
            is_linked_worktree=git_dir != common_dir,
            is_submodule=bool(superproject),
            execution_workspace_id=self.execution_workspace_id,
        )

    def status(self, *, include_untracked: bool = True) -> dict[str, object]:
        arguments = ["status", "--porcelain=v1", "-z", "--branch"]
        arguments.append("--untracked-files=all" if include_untracked else "--untracked-files=no")
        raw = self._run_raw(tuple(arguments)).stdout
        entries = _parse_status(raw)
        return {
            "repository": self.repository().to_dict(),
            "entries": entries,
            "clean": not entries,
            "ignored_files_included": False,
            "secret_contents_included": False,
        }

    def diff(
        self,
        *,
        staged: bool = False,
        base: str | None = None,
        path: str | None = None,
    ) -> dict[str, object]:
        common_arguments = ["diff", "--no-ext-diff", "--no-textconv"]
        selectors: list[str] = []
        if staged:
            selectors.append("--cached")
        if base is not None:
            self._verify_revision(base)
            selectors.append(base)
        excluded_sensitive_paths = 0
        if path is not None:
            safe_path = self._validate_path(path)
            arguments = [
                *common_arguments,
                "--unified=3",
                *selectors,
                "--",
                safe_path,
            ]
            result = self._run_raw(
                tuple(arguments),
                max_chars=self.policy.max_diff_chars,
            )
        else:
            names_result = self._run_raw(
                tuple(
                    [
                        *common_arguments,
                        "--name-only",
                        "-z",
                        *selectors,
                    ]
                )
            )
            safe_paths: list[str] = []
            for changed_path in names_result.stdout.split("\x00"):
                if not changed_path:
                    continue
                try:
                    safe_paths.append(self._validate_path(changed_path))
                except GitPolicyError:
                    excluded_sensitive_paths += 1
            if safe_paths:
                result = self._run_raw(
                    tuple(
                        [
                            *common_arguments,
                            "--unified=3",
                            *selectors,
                            "--",
                            *safe_paths,
                        ]
                    ),
                    max_chars=self.policy.max_diff_chars,
                )
            else:
                result = GitCommandResult(
                    arguments=(),
                    stdout="",
                    stderr="",
                    exit_code=0,
                )
        return {
            "repository": self.repository().to_dict(),
            "diff": result.stdout,
            "truncated": result.truncated,
            "staged": staged,
            "base": base,
            "path": path,
            "excluded_sensitive_path_count": excluded_sensitive_paths,
        }

    def log(self, *, limit: int = 20, revision: str = "HEAD") -> dict[str, object]:
        if not 1 <= limit <= self.policy.max_log_entries:
            raise ValueError(
                f"limit must be between 1 and {self.policy.max_log_entries}"
            )
        self._verify_revision(revision)
        result = self._run_raw(
            (
                "log",
                f"--max-count={limit}",
                "--date=iso-strict",
                "--format=%H%x1f%h%x1f%an%x1f%aI%x1f%s%x1e",
                revision,
            )
        )
        entries: list[dict[str, str]] = []
        for record in result.stdout.split("\x1e"):
            fields = record.strip().split("\x1f")
            if len(fields) == 5:
                entries.append(
                    {
                        "commit": fields[0],
                        "short_commit": fields[1],
                        "author": fields[2],
                        "authored_at": fields[3],
                        "subject": fields[4],
                    }
                )
        return {"repository": self.repository().to_dict(), "entries": entries}

    def branches(self) -> dict[str, object]:
        result = self._run_raw(
            (
                "for-each-ref",
                "--sort=refname",
                "--format=%(refname)%00%(objectname)%00%(HEAD)%00%(upstream:short)%00",
                "refs/heads",
                "refs/remotes",
            )
        )
        fields = result.stdout.split("\x00")
        entries: list[dict[str, object]] = []
        for index in range(0, len(fields) - 3, 4):
            ref, commit, head, upstream = fields[index : index + 4]
            ref = ref.strip()
            if not ref:
                continue
            entries.append(
                {
                    "ref": ref,
                    "commit": commit,
                    "current": head.strip() == "*",
                    "upstream": upstream or None,
                }
            )
        return {"repository": self.repository().to_dict(), "entries": entries}

    def worktrees(self) -> dict[str, object]:
        result = self._run_raw(("worktree", "list", "--porcelain"))
        entries: list[dict[str, object]] = []
        current: dict[str, object] = {}
        for line in result.stdout.splitlines():
            if not line:
                if current:
                    entries.append(current)
                    current = {}
                continue
            key, _, value = line.partition(" ")
            if key == "worktree":
                current["path"] = str(Path(value).resolve())
            elif key == "HEAD":
                current["commit"] = value
            elif key == "branch":
                current["branch"] = value
            elif key in {"bare", "detached", "locked", "prunable"}:
                current[key] = value or True
        if current:
            entries.append(current)
        return {"repository": self.repository().to_dict(), "entries": entries}

    def apply_patch(self, patch: str) -> dict[str, object]:
        if not self.policy.allow_apply_patch:
            raise GitPolicyError("Git patch application is disabled", code="git_patch_denied")
        if not patch.strip() or len(patch) > self.policy.max_patch_chars:
            raise ValueError(
                f"patch must contain between 1 and {self.policy.max_patch_chars} characters"
            )
        paths = self._validate_patch(patch)
        self._run_raw(
            ("apply", "--check", "--whitespace=error-all", "-"),
            input_text=patch,
        )
        self._run_raw(("apply", "--whitespace=error-all", "-"), input_text=patch)
        return {
            "repository": self.repository().to_dict(),
            "applied": True,
            "paths": sorted(paths),
        }

    def create_branch(self, name: str, *, start_point: str = "HEAD") -> dict[str, object]:
        if not self.policy.allow_branch_creation:
            raise GitPolicyError(
                "Git branch creation is disabled",
                code="git_branch_creation_denied",
            )
        clean_name = _bounded_text_field(name, "branch name", 200)
        self._run_raw(("check-ref-format", "--branch", clean_name))
        self._verify_revision(start_point)
        self._run_raw(("switch", "--no-guess", "-c", clean_name, start_point))
        return {
            "repository": self.repository().to_dict(),
            "created_branch": clean_name,
            "start_point": start_point,
        }

    def commit(
        self,
        *,
        message: str,
        paths: Sequence[str],
    ) -> dict[str, object]:
        if not self.policy.allow_commit:
            raise GitPolicyError("Git commits are disabled", code="git_commit_denied")
        clean_message = _bounded_text_field(message, "commit message", 500)
        if not paths:
            raise ValueError("commit paths cannot be empty")
        safe_paths = tuple(dict.fromkeys(self._validate_path(path) for path in paths))
        self._run_raw(("add", "--", *safe_paths))
        staged = self._run_raw(("diff", "--cached", "--quiet"), check=False)
        if staged.exit_code == 0:
            raise GitOperationError("No staged changes to commit", code="nothing_to_commit")
        arguments = ["commit"]
        if not self.policy.run_commit_hooks:
            arguments.append("--no-verify")
        arguments.extend(["-m", clean_message, "--", *safe_paths])
        self._run_raw(tuple(arguments))
        commit = self._run_raw(("rev-parse", "HEAD")).stdout.strip()
        return {
            "repository": self.repository().to_dict(),
            "commit": commit,
            "paths": list(safe_paths),
            "hooks_run": self.policy.run_commit_hooks,
        }

    def push(self, *, remote: str, branch: str) -> dict[str, object]:
        if not self.policy.allow_push:
            raise GitPolicyError(
                "Git push is disabled until the host explicitly enables it",
                code="git_push_denied",
            )
        clean_remote = remote.strip()
        if clean_remote not in self.policy.allowed_remotes:
            raise GitPolicyError(
                "Git remote is outside the host allowlist",
                code="git_remote_denied",
            )
        clean_branch = _bounded_text_field(branch, "branch", 200)
        self._run_raw(("check-ref-format", "--branch", clean_branch))
        current = self.repository()
        if current.branch is None:
            raise GitPolicyError(
                "Git push from detached HEAD is denied",
                code="detached_push_denied",
            )
        result = self._run_raw(
            (
                "push",
                "--porcelain",
                clean_remote,
                f"HEAD:refs/heads/{clean_branch}",
            ),
            timeout_seconds=self.policy.command_timeout_seconds,
        )
        return {
            "repository": self.repository().to_dict(),
            "remote": clean_remote,
            "branch": clean_branch,
            "output": result.stdout,
            "force": False,
        }

    def run_tests(self, command: str, *, timeout_seconds: int | None = None) -> dict[str, object]:
        arguments = _split_test_command(command)
        if not arguments:
            raise ValueError("test command cannot be empty")
        allowed = any(_starts_with(arguments, prefix) for prefix in self.policy.allowed_test_commands)
        if not allowed:
            raise GitPolicyError(
                "Test command is outside the configured argv allowlist",
                code="test_command_denied",
            )
        _validate_test_arguments(arguments)
        timeout = timeout_seconds or self.policy.test_timeout_seconds
        if not 1 <= timeout <= self.policy.test_timeout_seconds:
            raise ValueError(
                f"timeout_seconds must be between 1 and {self.policy.test_timeout_seconds}"
            )
        result = self._run_process(
            arguments,
            timeout_seconds=timeout,
            max_chars=self.policy.max_output_chars,
            check=False,
        )
        return {
            "repository": self.repository().to_dict(),
            "command": list(arguments),
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "truncated": result.truncated,
            "success": result.exit_code == 0,
        }

    def _verify_revision(self, revision: str) -> None:
        clean = _bounded_text_field(revision, "revision", 200)
        if clean.startswith("-"):
            raise ValueError("revision cannot begin with '-'")
        self._run_raw(("rev-parse", "--verify", f"{clean}^{{commit}}"))

    def _validate_path(self, path: str) -> str:
        clean = path.replace("\\", "/").strip()
        pure = PurePosixPath(clean)
        if (
            not clean
            or clean.startswith("/")
            or pure.is_absolute()
            or ".." in pure.parts
            or "\x00" in clean
        ):
            raise GitPolicyError(
                "Git path must remain inside the repository root",
                code="git_path_denied",
            )
        lower_parts = tuple(part.lower() for part in pure.parts)
        if ".git" in lower_parts or _looks_secret(lower_parts):
            raise GitPolicyError(
                "Git path is sensitive and cannot be used by structured writes",
                code="git_sensitive_path_denied",
            )
        candidate = (self.project_root / Path(*pure.parts)).resolve()
        if candidate != self.project_root and self.project_root not in candidate.parents:
            raise GitPolicyError(
                "Git path escapes the repository root",
                code="git_path_denied",
            )
        submodules = self._submodule_paths()
        if any(clean == submodule or clean.startswith(f"{submodule}/") for submodule in submodules):
            raise GitPolicyError(
                "Structured writes into submodules are denied",
                code="git_submodule_path_denied",
            )
        return clean

    def _validate_patch(self, patch: str) -> set[str]:
        if "GIT binary patch" in patch or "\nBinary files " in patch:
            raise GitPolicyError(
                "Binary Git patches are denied",
                code="git_binary_patch_denied",
            )
        parsed = self._run_raw(
            ("apply", "--numstat", "-z", "-"),
            input_text=patch,
            max_chars=self.policy.max_patch_chars,
        )
        if parsed.truncated:
            raise GitPolicyError(
                "Decoded Git patch path list exceeds the validation limit",
                code="git_patch_paths_too_large",
            )
        paths = self._validate_patch_paths(parsed.stdout)
        if not paths:
            raise ValueError("patch does not contain a recognized file path")
        return paths

    def _validate_patch_paths(self, numstat: str) -> set[str]:
        entries = numstat.split("\x00")
        paths: set[str] = set()
        index = 0
        while index < len(entries):
            entry = entries[index]
            if not entry:
                index += 1
                continue
            fields = entry.split("\t", 2)
            if len(fields) != 3:
                raise GitPolicyError(
                    "Git returned an invalid decoded patch path list",
                    code="git_patch_paths_invalid",
                )
            path = fields[2]
            if path:
                paths.add(self._validate_path(path))
                index += 1
                continue
            if index + 2 >= len(entries):
                raise GitPolicyError(
                    "Git returned an incomplete decoded rename path list",
                    code="git_patch_paths_invalid",
                )
            paths.add(self._validate_path(entries[index + 1]))
            paths.add(self._validate_path(entries[index + 2]))
            index += 3
        return paths

    def _submodule_paths(self) -> tuple[str, ...]:
        result = self._run_raw(("ls-files", "--stage", "-z"))
        paths: list[str] = []
        for entry in result.stdout.split("\x00"):
            metadata, separator, path = entry.partition("\t")
            if separator and metadata.startswith("160000 "):
                paths.append(path)
        return tuple(paths)

    def _run_raw(
        self,
        arguments: tuple[str, ...],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout_seconds: int | None = None,
        max_chars: int | None = None,
    ) -> GitCommandResult:
        return self._run_process(
            (
                self.git_executable,
                "-C",
                str(self.project_root),
                "--literal-pathspecs",
                *arguments,
            ),
            input_text=input_text,
            check=check,
            timeout_seconds=timeout_seconds or self.policy.command_timeout_seconds,
            max_chars=max_chars or self.policy.max_output_chars,
            public_arguments=arguments,
        )

    def _run_process(
        self,
        arguments: tuple[str, ...],
        *,
        input_text: str | None = None,
        check: bool = True,
        timeout_seconds: int,
        max_chars: int,
        public_arguments: tuple[str, ...] | None = None,
    ) -> GitCommandResult:
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "never",
            "LC_ALL": "C",
            "LANG": "C",
        }
        try:
            completed = subprocess.run(
                arguments,
                cwd=self.project_root,
                env=environment,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitOperationError(
                "Git or test operation timed out",
                code="git_timeout",
            ) from exc
        stdout, stdout_truncated = _bounded_output(completed.stdout, max_chars)
        stderr, stderr_truncated = _bounded_output(completed.stderr, max_chars)
        result = GitCommandResult(
            arguments=public_arguments or arguments,
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
            truncated=stdout_truncated or stderr_truncated,
        )
        if check and completed.returncode != 0:
            raise GitOperationError(
                "Git operation failed",
                code="git_command_failed",
                stderr=stderr,
            )
        return result


def git_read_tool(service: GitService) -> Tool:
    """Expose typed, bounded Git inspection without arbitrary Git argv."""

    def invoke(arguments: dict[str, object]) -> ToolResult:
        action = str(arguments["action"])
        try:
            if action == "status":
                value = service.status(
                    include_untracked=bool(arguments.get("include_untracked", True))
                )
            elif action == "diff":
                value = service.diff(
                    staged=bool(arguments.get("staged", False)),
                    base=_optional_string(arguments.get("base")),
                    path=_optional_string(arguments.get("path")),
                )
            elif action == "log":
                value = service.log(
                    limit=_integer(arguments.get("limit", 20), "limit"),
                    revision=str(arguments.get("revision", "HEAD")),
                )
            elif action == "branches":
                value = service.branches()
            elif action == "worktrees":
                value = service.worktrees()
            else:
                raise ValueError(f"unsupported Git read action: {action}")
        except Exception as exc:
            return _git_failure("git_read", exc)
        return _git_success("git_read", action, value)

    return Tool(
        name="git_read",
        description=(
            "Inspect structured Git status, bounded diff/log, branches, or worktrees in the "
            "configured repository root. Does not expose arbitrary Git arguments."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "branches", "worktrees"],
                },
                "include_untracked": {"type": "boolean"},
                "staged": {"type": "boolean"},
                "base": {"type": "string", "minLength": 1, "maxLength": 200},
                "path": {"type": "string", "minLength": 1, "maxLength": 1000},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": service.policy.max_log_entries,
                },
                "revision": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        callable=invoke,
        permission_level=ToolPermissionLevel.READ,
        run_in_executor=True,
        timeout_seconds=service.policy.command_timeout_seconds + 2,
        idempotent=True,
        metadata={
            "structured_git": True,
            "denied_operations": [
                "force_push",
                "history_rewrite",
                "destructive_cleanup",
                "arbitrary_git_argv",
            ],
        },
    )


def git_write_tool(service: GitService) -> Tool:
    """Expose confirmation-gated patch, branch, commit, and non-force push."""

    def invoke(arguments: dict[str, object]) -> ToolResult:
        action = str(arguments["action"])
        try:
            if action == "apply_patch":
                value = service.apply_patch(_required_string(arguments, "patch"))
            elif action == "create_branch":
                value = service.create_branch(
                    _required_string(arguments, "branch"),
                    start_point=str(arguments.get("start_point", "HEAD")),
                )
            elif action == "commit":
                paths = arguments.get("paths")
                if not isinstance(paths, list) or not all(
                    isinstance(path, str) for path in paths
                ):
                    raise ValueError("paths must be an array of repository-relative paths")
                value = service.commit(
                    message=_required_string(arguments, "message"),
                    paths=paths,
                )
            elif action == "push":
                value = service.push(
                    remote=_required_string(arguments, "remote"),
                    branch=_required_string(arguments, "branch"),
                )
            else:
                raise ValueError(f"unsupported Git write action: {action}")
        except Exception as exc:
            return _git_failure("git_write", exc)
        return _git_success("git_write", action, value)

    return Tool(
        name="git_write",
        description=(
            "Apply an approved text patch, create a branch, commit explicit safe paths, or "
            "perform a policy-enabled non-force push. Force, rewrite, cleanup, secret, and "
            "submodule writes are denied."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["apply_patch", "create_branch", "commit", "push"],
                },
                "patch": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": service.policy.max_patch_chars,
                },
                "branch": {"type": "string", "minLength": 1, "maxLength": 200},
                "start_point": {"type": "string", "minLength": 1, "maxLength": 200},
                "message": {"type": "string", "minLength": 1, "maxLength": 500},
                "paths": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 500,
                    "items": {"type": "string", "minLength": 1, "maxLength": 1000},
                },
                "remote": {"type": "string", "minLength": 1, "maxLength": 128},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        callable=invoke,
        permission_level=ToolPermissionLevel.DESTRUCTIVE,
        requires_confirmation=True,
        run_in_executor=True,
        timeout_seconds=service.policy.command_timeout_seconds + 2,
        metadata={
            "structured_git": True,
            "force": False,
            "history_rewrite": False,
            "destructive_cleanup": False,
        },
    )


def git_test_tool(service: GitService) -> Tool:
    """Expose confirmation-gated, argv-allowlisted, bounded test execution."""

    def invoke(arguments: dict[str, object]) -> ToolResult:
        try:
            value = service.run_tests(
                _required_string(arguments, "command"),
                timeout_seconds=(
                    _integer(arguments["timeout_seconds"], "timeout_seconds")
                    if "timeout_seconds" in arguments
                    else None
                ),
            )
        except Exception as exc:
            return _git_failure("git_test", exc)
        return ToolResult(
            tool_name="git_test",
            success=bool(value["success"]),
            observation=json.dumps(value, ensure_ascii=False, sort_keys=True),
            value=value,
            stdout=str(value["stdout"]),
            stderr=str(value["stderr"]),
            exit_code=_integer(value["exit_code"], "exit_code"),
            failure_kind=(
                None if value["success"] else ToolFailureKind.ENVIRONMENT
            ),
            metadata={"structured_git": True, "shell": False},
        )

    commands = [" ".join(command) for command in service.policy.allowed_test_commands]
    return Tool(
        name="git_test",
        description=(
            "Run one approved test command without a shell, through an exact argv-prefix "
            f"allowlist: {', '.join(commands)}."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1, "maxLength": 4000},
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": service.policy.test_timeout_seconds,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        callable=invoke,
        permission_level=ToolPermissionLevel.SHELL,
        requires_confirmation=True,
        run_in_executor=True,
        timeout_seconds=service.policy.test_timeout_seconds + 2,
        metadata={"structured_git": True, "shell": False, "allowed_commands": commands},
    )


def git_tools(service: GitService) -> tuple[Tool, Tool, Tool]:
    return git_read_tool(service), git_write_tool(service), git_test_tool(service)


def _parse_status(value: str) -> list[dict[str, object]]:
    records = value.split("\x00")
    entries: list[dict[str, object]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record or record.startswith("## "):
            continue
        if len(record) < 4:
            continue
        status = record[:2]
        path = record[3:]
        entry: dict[str, object] = {
            "index_status": status[0],
            "worktree_status": status[1],
            "path": path,
        }
        if status[0] in {"R", "C"} and index < len(records):
            entry["original_path"] = records[index]
            index += 1
        entries.append(entry)
    return entries


def _git_success(tool_name: str, action: str, value: object) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        success=True,
        observation=json.dumps(value, ensure_ascii=False, sort_keys=True),
        value=value,
        metadata={"structured_git": True, "action": action},
    )


def _git_failure(tool_name: str, exc: Exception) -> ToolResult:
    if isinstance(exc, GitPolicyError):
        kind = ToolFailureKind.FATAL_SAFETY
        code = exc.code
    elif isinstance(exc, GitOperationError):
        kind = ToolFailureKind.ENVIRONMENT
        code = exc.code
    elif isinstance(exc, (ValueError, KeyError)):
        kind = ToolFailureKind.INVALID_ARGUMENTS
        code = "invalid_git_action"
    else:
        kind = ToolFailureKind.ENVIRONMENT
        code = type(exc).__name__
    details = redact_text(exc.stderr) if isinstance(exc, GitOperationError) else ""
    observation = f"Structured Git operation failed: {redact_text(str(exc))}"
    if details:
        observation += f"\nstderr:\n{details}"
    return ToolResult(
        tool_name=tool_name,
        success=False,
        observation=observation,
        error=code,
        failure_kind=kind,
        metadata={"policy_code": code, "structured_git": True},
    )


def _looks_secret(parts: tuple[str, ...]) -> bool:
    for part in parts:
        if part in _SECRET_NAMES:
            return True
        stem = part.rsplit(".", 1)[0].lower()
        words = frozenset(filter(None, re.split(r"[^a-z0-9]+", stem)))
        if words & _SECRET_MARKERS or stem in _SECRET_MARKERS:
            return True
    return False


def _bounded_text_field(value: str, label: str, max_chars: int) -> str:
    clean = value.strip()
    if (
        not clean
        or len(clean) > max_chars
        or _CONTROL_CHARACTERS.search(clean.replace("\n", ""))
    ):
        raise ValueError(f"{label} must be a non-empty bounded text value")
    return clean


def _bounded_output(value: str, max_chars: int) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    marker = "\n...[structured command output truncated]"
    return value[: max(0, max_chars - len(marker))] + marker, True


def _starts_with(arguments: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    if len(arguments) < len(prefix):
        return False
    if arguments[: len(prefix)] == prefix:
        return True
    if prefix and prefix[0] == "python":
        executable_name = Path(arguments[0]).name.lower()
        return executable_name.startswith("python") and arguments[1 : len(prefix)] == prefix[1:]
    return False


def _split_test_command(command: str, *, windows: bool | None = None) -> tuple[str, ...]:
    use_windows_rules = os.name == "nt" if windows is None else windows
    try:
        raw_arguments = shlex.split(command, posix=not use_windows_rules)
    except ValueError as exc:
        raise ValueError("test command contains unbalanced quoting") from exc
    if not use_windows_rules:
        return tuple(raw_arguments)
    arguments: list[str] = []
    for argument in raw_arguments:
        if argument.startswith('"') or argument.endswith('"'):
            if len(argument) < 2 or not (
                argument.startswith('"') and argument.endswith('"')
            ):
                raise ValueError("test command contains unbalanced Windows quoting")
            argument = argument[1:-1]
        if '"' in argument:
            raise ValueError("test command contains unsupported embedded Windows quoting")
        arguments.append(argument)
    return tuple(arguments)


def _validate_test_arguments(arguments: tuple[str, ...]) -> None:
    """Keep allowlisted test argv scoped to the configured repository."""
    for argument in arguments[1:]:
        candidate = argument.split("=", 1)[-1] if "=" in argument else argument
        normalized = candidate.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise GitPolicyError(
                "Test command arguments cannot reference paths outside the repository",
                code="test_path_denied",
            )


def _required_string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required for this Git action")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optional Git string fields cannot be empty")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


__all__ = [
    "GitCommandResult",
    "GitOperationError",
    "GitPolicy",
    "GitPolicyError",
    "GitRepository",
    "GitService",
    "git_read_tool",
    "git_test_tool",
    "git_tools",
    "git_write_tool",
]
