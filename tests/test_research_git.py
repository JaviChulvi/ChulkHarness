from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from chulk.execution import ExecutionSessionRequest, HostExecutionBackend
from chulk.research import (
    GitPolicy,
    GitPolicyError,
    GitService,
    git_tools,
)
from chulk.research.git import _split_test_command


def _git(root: Path, *arguments: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Research Tests")
    _git(root, "config", "user.email", "research@example.com")
    (root / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", "initial")
    return root


def test_git_service_returns_structured_status_diff_log_branches_and_worktrees(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    (root / "tracked.txt").write_text("one\ntwo\n", encoding="utf-8")
    (root / "new.txt").write_text("new\n", encoding="utf-8")
    service = GitService(root)

    status = service.status()
    diff = service.diff(path="tracked.txt")
    log = service.log(limit=1)
    branches = service.branches()
    worktrees = service.worktrees()

    assert status["clean"] is False
    assert {entry["path"] for entry in status["entries"]} == {
        "tracked.txt",
        "new.txt",
    }
    assert "+two" in diff["diff"]
    assert log["entries"][0]["subject"] == "initial"
    assert any(entry["current"] for entry in branches["entries"])
    assert worktrees["entries"][0]["path"] == str(root)
    assert status["ignored_files_included"] is False
    assert status["secret_contents_included"] is False


def test_git_service_applies_patch_creates_branch_and_commits_explicit_paths(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    service = GitService(root)
    patch = """diff --git a/tracked.txt b/tracked.txt
--- a/tracked.txt
+++ b/tracked.txt
@@ -1 +1,2 @@
 one
+two
"""

    applied = service.apply_patch(patch)
    branch = service.create_branch("feature/research")
    committed = service.commit(message="Apply safe research change", paths=["tracked.txt"])

    assert applied["paths"] == ["tracked.txt"]
    assert (root / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
    assert branch["created_branch"] == "feature/research"
    assert committed["commit"] == _git(root, "rev-parse", "HEAD").strip()
    assert service.status()["clean"] is True


def test_git_service_denies_sensitive_submodule_binary_push_and_repository_escape(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    service = GitService(root)

    with pytest.raises(GitPolicyError, match="sensitive"):
        service.commit(message="bad", paths=[".env"])
    with pytest.raises(GitPolicyError, match="inside"):
        service.diff(path="../outside")
    with pytest.raises(GitPolicyError, match="Binary"):
        service.apply_patch(
            "diff --git a/image.png b/image.png\nGIT binary patch\nliteral 0\n"
        )
    with pytest.raises(GitPolicyError, match="disabled"):
        service.push(remote="origin", branch="main")

    (root / ".env").write_text("API_KEY=top-secret\n", encoding="utf-8")
    _git(root, "add", ".env")
    _git(root, "commit", "-m", "seed ignored secret fixture")
    (root / ".env").write_text("API_KEY=changed-secret\n", encoding="utf-8")
    filtered = service.diff()
    assert filtered["diff"] == ""
    assert filtered["excluded_sensitive_path_count"] == 1
    assert "changed-secret" not in str(filtered)

    nested = root / "nested"
    nested.mkdir()
    with pytest.raises(GitPolicyError, match="repository root"):
        GitService(nested)


def test_git_service_runs_only_allowlisted_tests_without_a_shell(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    service = GitService(
        root,
        policy=GitPolicy(
            allowed_test_commands=((sys.executable, "-c"),),
            test_timeout_seconds=5,
        ),
    )

    passed = service.run_tests(
        f"{shlex_quote(sys.executable)} -c \"print('bounded-test')\""
    )

    assert passed["success"] is True
    assert passed["stdout"] == "bounded-test\n"
    with pytest.raises(GitPolicyError, match="allowlist"):
        service.run_tests("git status")
    with pytest.raises(ValueError, match="timeout"):
        service.run_tests(
            f"{shlex_quote(sys.executable)} -c \"print('x')\"",
            timeout_seconds=6,
        )
    with pytest.raises(GitPolicyError, match="outside"):
        service.run_tests(
            f"{shlex_quote(sys.executable)} -c pass /tmp/outside"
        )


def test_windows_test_command_splitting_removes_only_balanced_outer_quotes() -> None:
    assert _split_test_command(
        '"C:\\Program Files\\Python\\python.exe" -c "print(\'bounded\')"',
        windows=True,
    ) == (
        "C:\\Program Files\\Python\\python.exe",
        "-c",
        "print('bounded')",
    )
    with pytest.raises(ValueError, match="unbalanced"):
        _split_test_command('"python.exe -m pytest', windows=True)


def test_git_tools_require_confirmation_for_tests_and_every_material_action(
    tmp_path: Path,
) -> None:
    service = GitService(_repository(tmp_path))
    read_tool, write_tool, test_tool = git_tools(service)

    result = read_tool.callable({"action": "status"})
    denied = write_tool.callable(
        {"action": "push", "remote": "origin", "branch": "main"}
    )

    assert result.success is True
    assert read_tool.requires_confirmation is False
    assert write_tool.requires_confirmation is True
    assert test_tool.requires_confirmation is True
    assert write_tool.metadata["force"] is False
    assert denied.failure_kind == "fatal_safety"
    assert denied.error == "git_push_denied"


def test_git_commit_does_not_absorb_unrelated_staged_paths(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (root / "other.txt").write_text("other\n", encoding="utf-8")
    _git(root, "add", "other.txt")
    service = GitService(root)

    service.commit(message="Only tracked", paths=["tracked.txt"])

    staged = _git(root, "diff", "--cached", "--name-only").splitlines()
    assert staged == ["other.txt"]
    assert _git(root, "show", "--format=", "--name-only", "HEAD").splitlines() == [
        "tracked.txt"
    ]


def test_git_service_binds_to_execution_workspace_identity(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    backend = HostExecutionBackend(root)
    session = backend.open_session(ExecutionSessionRequest(turn_id="git-turn"))

    service = GitService.from_execution_session(session)

    assert service.repository().execution_workspace_id == session.workspace.workspace_id
    session.close()
    with pytest.raises(GitPolicyError, match="closed"):
        GitService.from_execution_session(session)
    backend.close()


def shlex_quote(value: str) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline([value])
    import shlex

    return shlex.quote(value)
