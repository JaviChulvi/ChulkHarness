"""Reviewable temporary workspace, change application, and Git strategy tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from chulk import (
    Agent,
    AgentConfig,
    Capabilities,
    ChangeDisposition,
    ChangeSetApproval,
    CommandExecutionRequest,
    EnvironmentPolicy,
    ExecutionSessionRequest,
    FileReadRequest,
    FileWriteRequest,
    GitWorktreeBackend,
    GitWorktreePolicy,
    NetworkPolicy,
    ResourcePolicy,
    SecretPolicy,
    ShellExecutionDecision,
    TemporaryWorkspaceBackend,
    UnsafePathAction,
    WorkspaceMaterializationPolicy,
    WorkspacePersistence,
    WorkspacePolicyError,
)
from chulk.llm import LLMClient
import chulk.execution.temporary as temporary_module


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, messages, *, max_output_tokens=None) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _tool_call(name: str, **arguments) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": name,
            "arguments_json": json.dumps(arguments),
        }
    )


def _final(content: str = "done") -> str:
    return json.dumps({"type": "final_answer", "content": content})


def _approval(change_set_id: str) -> ChangeSetApproval:
    return ChangeSetApproval(change_set_id, approved_by="test-host")


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )


def _git_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.com")
    _git(repository, "config", "user.name", "Chulk Tests")
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-m", "initial")
    return repository


def test_temporary_session_shares_workspace_but_does_not_mutate_host(tmp_path):
    (tmp_path / "input.txt").write_text("host\n", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="shared"))

    write = session.write_file(
        FileWriteRequest("input.txt", "workspace\n", overwrite=True)
    )
    command = session.run_command(
        CommandExecutionRequest(
            "python -c \"from pathlib import Path;"
            "print(Path('input.txt').read_text().strip());"
            "Path('from-command.txt').write_text('command')\""
        )
    )

    assert write.success is True
    assert command.success is True
    assert command.stdout == "workspace\n"
    assert (tmp_path / "input.txt").read_text(encoding="utf-8") == "host\n"
    assert not (tmp_path / "from-command.txt").exists()
    assert command.change_set is not None
    assert {change.path for change in command.change_set.changes} == {
        "from-command.txt",
        "input.txt",
    }
    assert command.metadata["workspace_mode"] == "temporary"
    assert command.metadata["change_disposition"] == "return_change_set"
    assert command.metadata["environment_policy"]["inherit_all"] is False
    assert command.metadata["resource_policy"]["max_change_files"] == 500
    assert command.metadata["change_set"]["requires_host_approval"] is True


def test_temporary_shell_fails_closed_without_host_containment_opt_in(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    backend = TemporaryWorkspaceBackend(tmp_path)
    session = backend.open_session(
        ExecutionSessionRequest(turn_id="containment-required")
    )

    result = session.run_command(
        CommandExecutionRequest(
            "python -c \"from pathlib import Path;"
            f"Path({str(outside)!r}).write_text('escaped')\""
        )
    )

    assert result.success is False
    assert result.error == "containment_required"
    assert result.metadata["child_process_started"] is False
    assert not outside.exists()


def test_ephemeral_close_cleans_workspace_but_retains_change_set(tmp_path):
    backend = TemporaryWorkspaceBackend(tmp_path)
    session = backend.open_session(ExecutionSessionRequest(turn_id="cleanup"))
    result = session.write_file(FileWriteRequest("created.txt", "review me"))
    assert result.change_set is not None
    workspace_root = session.project_root
    change_set_id = result.change_set.change_set_id

    session.close()

    assert session.closed is True
    assert not workspace_root.exists()
    assert backend.get_change_set(change_set_id).changes[0].path == "created.txt"


def test_persistent_workspace_requires_explicit_cleanup(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        persistence=WorkspacePersistence.PERSISTENT,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="persistent"))
    workspace_id = session.workspace.workspace_id
    workspace_root = session.project_root
    session.write_file(FileWriteRequest("created.txt", "retained"))

    session.close()

    assert workspace_root.exists()
    backend.cleanup_workspace(workspace_id)
    assert not workspace_root.exists()


def test_approved_apply_handles_modify_create_delete_and_binary(tmp_path):
    (tmp_path / "modified.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "deleted.txt").write_text("delete\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00old")
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="apply"))
    session.write_file(
        FileWriteRequest("modified.txt", "new\n", overwrite=True)
    )
    session.run_command(
        CommandExecutionRequest(
            "python -c \"from pathlib import Path;"
            "Path('created.txt').write_text('created');"
            "Path('deleted.txt').unlink();"
            "Path('binary.bin').write_bytes(bytes.fromhex('006e6577'))\""
        )
    )
    change_set = session.latest_change_set
    assert change_set is not None
    session.close()

    applied = backend.apply_change_set(
        change_set.change_set_id,
        approval=_approval(change_set.change_set_id),
    )

    assert applied.success is True
    assert (tmp_path / "modified.txt").read_text(encoding="utf-8") == "new\n"
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created"
    assert not (tmp_path / "deleted.txt").exists()
    assert (tmp_path / "binary.bin").read_bytes() == b"\x00new"
    repeated = backend.apply_change_set(
        change_set.change_set_id,
        approval=_approval(change_set.change_set_id),
    )
    assert repeated.error == "change_set_already_applied"


def test_apply_rejects_mismatched_approval(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="approval"))
    result = session.write_file(FileWriteRequest("created.txt", "content"))
    assert result.change_set is not None

    with pytest.raises(ValueError, match="does not match"):
        backend.apply_change_set(
            result.change_set.change_set_id,
            approval=ChangeSetApproval("different", approved_by="test-host"),
        )
    assert not (tmp_path / "created.txt").exists()


def test_closed_backend_rejects_change_application(tmp_path):
    backend = TemporaryWorkspaceBackend(tmp_path)
    session = backend.open_session(ExecutionSessionRequest(turn_id="closed-apply"))
    result = session.write_file(FileWriteRequest("created.txt", "content"))
    assert result.change_set is not None
    session.close()
    backend.close()

    with pytest.raises(RuntimeError, match="backend is closed"):
        backend.apply_change_set(
            result.change_set.change_set_id,
            approval=_approval(result.change_set.change_set_id),
        )


def test_conflict_detection_prevents_partial_application(tmp_path):
    (tmp_path / "a.txt").write_text("a-old", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b-old", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="conflict"))
    session.write_file(FileWriteRequest("a.txt", "a-new", overwrite=True))
    result = session.write_file(
        FileWriteRequest("b.txt", "b-new", overwrite=True)
    )
    assert result.change_set is not None
    (tmp_path / "b.txt").write_text("external", encoding="utf-8")

    applied = backend.apply_change_set(
        result.change_set.change_set_id,
        approval=_approval(result.change_set.change_set_id),
    )

    assert applied.success is False
    assert applied.error == "change_conflict"
    assert any(conflict.startswith("b.txt:") for conflict in applied.conflicts)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a-old"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "external"


def test_apply_revalidates_each_file_and_rolls_back_on_race(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a-old", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b-old", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(tmp_path)
    session = backend.open_session(ExecutionSessionRequest(turn_id="race"))
    session.write_file(FileWriteRequest("a.txt", "a-new", overwrite=True))
    result = session.write_file(
        FileWriteRequest("b.txt", "b-new", overwrite=True)
    )
    assert result.change_set is not None
    original_assert = temporary_module._assert_expected_host_state
    b_checks = 0

    def race_before_second_apply(change, host_root):
        nonlocal b_checks
        if change.path == "b.txt":
            b_checks += 1
            if b_checks == 2:
                (host_root / "b.txt").write_text("raced", encoding="utf-8")
        original_assert(change, host_root)

    monkeypatch.setattr(
        temporary_module,
        "_assert_expected_host_state",
        race_before_second_apply,
    )

    applied = backend.apply_change_set(
        result.change_set.change_set_id,
        approval=_approval(result.change_set.change_set_id),
    )

    assert applied.success is False
    assert applied.rollback_errors == ()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a-old"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "raced"


def test_apply_rejects_symlinked_host_parent_without_touching_target(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source_directory = project / "nested"
    source_directory.mkdir()
    (source_directory / "file.txt").write_text("base", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("outside", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(project)
    session = backend.open_session(ExecutionSessionRequest(turn_id="parent-link"))
    result = session.write_file(
        FileWriteRequest("nested/file.txt", "changed", overwrite=True)
    )
    assert result.change_set is not None
    shutil.rmtree(source_directory)
    source_directory.symlink_to(outside, target_is_directory=True)

    applied = backend.apply_change_set(
        result.change_set.change_set_id,
        approval=_approval(result.change_set.change_set_id),
    )

    assert applied.success is False
    assert applied.error == "change_conflict"
    assert (outside / "file.txt").read_text(encoding="utf-8") == "outside"


def test_apply_rolls_back_prior_files_when_later_write_fails(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a-old", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b-old", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(tmp_path)
    session = backend.open_session(ExecutionSessionRequest(turn_id="rollback"))
    session.write_file(FileWriteRequest("a.txt", "a-new", overwrite=True))
    result = session.write_file(
        FileWriteRequest("b.txt", "b-new", overwrite=True)
    )
    assert result.change_set is not None
    original_apply = temporary_module._apply_change

    def fail_second(change, backup):
        if change.path == "b.txt":
            raise OSError("simulated apply failure")
        original_apply(change, backup)

    monkeypatch.setattr(temporary_module, "_apply_change", fail_second)

    applied = backend.apply_change_set(
        result.change_set.change_set_id,
        approval=_approval(result.change_set.change_set_id),
    )

    assert applied.success is False
    assert applied.rollback_errors == ()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a-old"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "b-old"


def test_apply_rolls_back_before_propagating_interruption(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a-old", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b-old", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(tmp_path)
    session = backend.open_session(ExecutionSessionRequest(turn_id="interrupt"))
    session.write_file(FileWriteRequest("a.txt", "a-new", overwrite=True))
    result = session.write_file(
        FileWriteRequest("b.txt", "b-new", overwrite=True)
    )
    assert result.change_set is not None
    original_apply = temporary_module._apply_change

    def interrupt_second(change, backup):
        if change.path == "b.txt":
            raise KeyboardInterrupt()
        original_apply(change, backup)

    monkeypatch.setattr(temporary_module, "_apply_change", interrupt_second)

    with pytest.raises(KeyboardInterrupt):
        backend.apply_change_set(
            result.change_set.change_set_id,
            approval=_approval(result.change_set.change_set_id),
        )
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a-old"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "b-old"


def test_materialization_skips_runtime_state_and_secrets(tmp_path):
    (tmp_path / "safe.txt").write_text("safe", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("private", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(tmp_path)

    session = backend.open_session(ExecutionSessionRequest(turn_id="filtered"))

    assert session.read_file(FileReadRequest("safe.txt")).success is True
    assert not (session.project_root / ".env").exists()
    assert not (session.project_root / ".git").exists()


def test_materialization_copies_only_explicitly_allowlisted_paths(tmp_path):
    (tmp_path / "allowed").mkdir()
    (tmp_path / "allowed" / "file.txt").write_text("allowed", encoding="utf-8")
    (tmp_path / "excluded.txt").write_text("excluded", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        materialization_policy=WorkspaceMaterializationPolicy(
            allowed_paths=("allowed",)
        ),
    )

    session = backend.open_session(ExecutionSessionRequest(turn_id="allowlist"))

    assert (session.project_root / "allowed" / "file.txt").exists()
    assert not (session.project_root / "excluded.txt").exists()


def test_materialization_rejects_path_traversal_symlinks_and_large_files(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(WorkspacePolicyError, match="symlink"):
        TemporaryWorkspaceBackend(tmp_path).open_session(
            ExecutionSessionRequest(turn_id="symlink")
        )

    (tmp_path / "link.txt").unlink()
    traversal_policy = WorkspaceMaterializationPolicy(
        allowed_paths=("../outside.txt",)
    )
    with pytest.raises(WorkspacePolicyError, match="Unsafe workspace path"):
        TemporaryWorkspaceBackend(
            tmp_path,
            materialization_policy=traversal_policy,
        ).open_session(ExecutionSessionRequest(turn_id="traversal"))

    (tmp_path / "large.txt").write_text("large", encoding="utf-8")
    with pytest.raises(WorkspacePolicyError, match="configured limit"):
        TemporaryWorkspaceBackend(
            tmp_path,
            resource_policy=ResourcePolicy(max_file_bytes=4),
        ).open_session(ExecutionSessionRequest(turn_id="large"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX special files and hardlinks")
def test_materialization_rejects_special_files_and_hardlinks(tmp_path):
    fifo = tmp_path / "events.fifo"
    os.mkfifo(fifo)
    with pytest.raises(WorkspacePolicyError, match="Special workspace file"):
        TemporaryWorkspaceBackend(tmp_path).open_session(
            ExecutionSessionRequest(turn_id="fifo")
        )
    fifo.unlink()

    original = tmp_path / "original.txt"
    original.write_text("linked", encoding="utf-8")
    os.link(original, tmp_path / "alias.txt")
    with pytest.raises(WorkspacePolicyError, match="Hard-linked"):
        TemporaryWorkspaceBackend(tmp_path).open_session(
            ExecutionSessionRequest(turn_id="hardlink")
        )


def test_shell_created_unsafe_change_fails_closed(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="unsafe-change"))

    result = session.run_command(
        CommandExecutionRequest(
            "python -c \"from pathlib import Path;"
            "Path('.env').write_text('SECRET=value')\""
        )
    )

    assert result.success is False
    assert result.error == "workspace_secret"
    assert result.change_set is None
    assert not (tmp_path / ".env").exists()


def test_default_environment_does_not_forward_secret_like_values(tmp_path, monkeypatch):
    monkeypatch.setenv("CHULK_TEST_SECRET_TOKEN", "do-not-forward")
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="environment"))

    result = session.run_command(
        CommandExecutionRequest(
            "python -c \"import os;"
            "print(os.environ.get('CHULK_TEST_SECRET_TOKEN', 'missing'))\""
        )
    )

    assert result.success is True
    assert result.stdout == "missing\n"


def test_explicit_secret_forwarding_requires_both_allowlists(tmp_path):
    environment = EnvironmentPolicy(
        allowed_names=("PATH", "CHULK_TEST_SECRET_TOKEN"),
        overrides=(("CHULK_TEST_SECRET_TOKEN", "forwarded"),),
    )
    without_secret_approval = TemporaryWorkspaceBackend(
        tmp_path,
        environment_policy=environment,
        require_shell_containment=False,
    )
    session = without_secret_approval.open_session(
        ExecutionSessionRequest(turn_id="secret-denied")
    )
    denied = session.run_command(
        CommandExecutionRequest(
            "python -c \"import os;"
            "print(os.environ.get('CHULK_TEST_SECRET_TOKEN', 'missing'))\""
        )
    )
    assert denied.stdout == "missing\n"

    approved_backend = TemporaryWorkspaceBackend(
        tmp_path,
        environment_policy=environment,
        secret_policy=SecretPolicy(
            allowed_environment_names=("CHULK_TEST_SECRET_TOKEN",)
        ),
        require_shell_containment=False,
    )
    approved_session = approved_backend.open_session(
        ExecutionSessionRequest(turn_id="secret-approved")
    )
    approved = approved_session.run_command(
        CommandExecutionRequest(
            "python -c \"import os;"
            "print(os.environ.get('CHULK_TEST_SECRET_TOKEN', 'missing'))\""
        )
    )
    assert approved.stdout == "forwarded\n"


def test_legacy_shell_policy_rewrite_and_containment_are_preserved(tmp_path):
    class RewritingPolicy:
        def prepare(self, request):
            return ShellExecutionDecision.allow(
                "python -c \"from pathlib import Path;"
                "Path('rewritten.txt').write_text('rewritten')\"",
                policy_name="rewritten-contained",
                shell=True,
                containment_applied=True,
            )

    backend = TemporaryWorkspaceBackend(
        tmp_path,
        shell_execution_policy=RewritingPolicy(),
        require_shell_containment=True,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="rewrite"))

    result = session.run_command(
        CommandExecutionRequest("python -c \"raise SystemExit(99)\"")
    )

    assert result.success is True
    assert result.metadata["effective_policy"] == "rewritten-contained"
    assert result.metadata["containment"] == "contained"
    assert result.change_set is not None
    assert result.change_set.changes[0].path == "rewritten.txt"


def test_network_deny_fails_without_unsafe_host_fallback(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        network_policy=NetworkPolicy.DENY,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="network"))

    result = session.run_command(
        CommandExecutionRequest("python -c \"print('must not run')\"")
    )

    assert result.success is False
    assert result.error == "network_policy_unsupported"
    assert result.metadata["child_process_started"] is False


def test_patch_preview_is_bounded_and_marked(tmp_path):
    (tmp_path / "content.txt").write_text("old\n", encoding="utf-8")
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        resource_policy=ResourcePolicy(
            max_file_bytes=10_000,
            max_patch_bytes=80,
        ),
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="patch-limit"))

    result = session.write_file(
        FileWriteRequest("content.txt", "n€w\n" * 100, overwrite=True)
    )

    assert result.change_set is not None
    assert result.change_set.patch_truncated is True
    assert len(result.change_set.patch.encode("utf-8")) <= 80
    assert all(not hasattr(change, "patch") for change in result.change_set.changes)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_apply_preserves_created_and_modified_file_modes(tmp_path):
    existing = tmp_path / "existing.sh"
    existing.write_text("#!/bin/sh\n", encoding="utf-8")
    existing.chmod(0o644)
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="modes"))
    result = session.run_command(
        CommandExecutionRequest(
            "python -c \"import os;"
            "from pathlib import Path;"
            "os.chmod('existing.sh', 0o755);"
            "Path('created.sh').write_text('#!/bin/sh\\\\n');"
            "os.chmod('created.sh', 0o700)\""
        )
    )
    assert result.change_set is not None
    by_path = {change.path: change for change in result.change_set.changes}
    assert by_path["existing.sh"].mode_before == 0o644
    assert by_path["existing.sh"].mode_after == 0o755
    assert by_path["created.sh"].mode_before is None
    assert by_path["created.sh"].mode_after == 0o700
    session.close()

    applied = backend.apply_change_set(
        result.change_set.change_set_id,
        approval=_approval(result.change_set.change_set_id),
    )

    assert applied.success is True
    assert existing.stat().st_mode & 0o7777 == 0o755
    assert (tmp_path / "created.sh").stat().st_mode & 0o7777 == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_apply_detects_host_mode_conflict(tmp_path):
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o644)
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="mode-conflict"))
    result = session.run_command(
        CommandExecutionRequest("python -c \"import os;os.chmod('script.sh', 0o755)\"")
    )
    assert result.change_set is not None
    target.chmod(0o600)

    applied = backend.apply_change_set(
        result.change_set.change_set_id,
        approval=_approval(result.change_set.change_set_id),
    )

    assert applied.success is False
    assert applied.error == "change_conflict"
    assert applied.conflicts == ("script.sh:host_mode_changed",)
    assert target.stat().st_mode & 0o7777 == 0o600


def test_snapshot_honors_rejected_ignored_paths(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        materialization_policy=WorkspaceMaterializationPolicy(
            ignored_action=UnsafePathAction.REJECT,
        ),
        require_shell_containment=False,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="ignored-change"))

    result = session.run_command(
        CommandExecutionRequest(
            "python -c \"from pathlib import Path;"
            "Path('node_modules').mkdir();"
            "Path('node_modules/generated.js').write_text('generated')\""
        )
    )

    assert result.success is False
    assert result.error == "ignored_workspace_path"
    assert result.change_set is None


def test_retained_change_sets_are_count_bounded(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        resource_policy=ResourcePolicy(max_retained_change_sets=1),
    )
    first_session = backend.open_session(ExecutionSessionRequest(turn_id="first"))
    first = first_session.write_file(FileWriteRequest("first.txt", "first"))
    assert first.change_set is not None
    first_session.close()
    second_session = backend.open_session(ExecutionSessionRequest(turn_id="second"))
    second = second_session.write_file(FileWriteRequest("second.txt", "second"))
    assert second.change_set is not None
    second_session.close()

    with pytest.raises(KeyError, match="Unknown change set"):
        backend.get_change_set(first.change_set.change_set_id)
    assert backend.get_change_set(second.change_set.change_set_id) == second.change_set


@pytest.mark.asyncio
async def test_async_temporary_session_and_apply(tmp_path):
    backend = TemporaryWorkspaceBackend(tmp_path)
    session = await backend.open_session_async(
        ExecutionSessionRequest(turn_id="async")
    )
    result = await session.write_file_async(
        FileWriteRequest("async.txt", "async content")
    )
    assert result.change_set is not None
    await session.aclose()

    applied = await backend.apply_change_set_async(
        result.change_set.change_set_id,
        approval=_approval(result.change_set.change_set_id),
    )

    assert applied.success is True
    assert (tmp_path / "async.txt").read_text(encoding="utf-8") == "async content"


def test_runtime_returns_reviewable_changes_without_touching_host(tmp_path):
    backend = TemporaryWorkspaceBackend(tmp_path)
    facade = Agent(
        config=AgentConfig(
            project_root=tmp_path,
            permission_profile="workspace-write",
        ),
        capabilities=Capabilities.full(),
        execution_backend=backend,
        llm=FakeLLM(
            [
                _tool_call("write_file", path="runtime.txt", content="review"),
                _final(),
            ]
        ),
        skills=[],
        permission_callback=lambda request, record: True,
    )

    result = facade.run_result("create a reviewable file")

    assert result.tool_calls[0].success is True
    change_set_metadata = result.tool_calls[0].metadata["change_set"]
    assert change_set_metadata["requires_host_approval"] is True
    assert not (tmp_path / "runtime.txt").exists()
    change_set = backend.get_change_set(change_set_metadata["change_set_id"])
    assert change_set.changes[0].content_after == "review"


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")
def test_git_worktree_strategy_isolated_cleanup_and_apply(tmp_path):
    repository = _git_repository(tmp_path)
    backend = GitWorktreeBackend(repository)
    session = backend.open_session(ExecutionSessionRequest(turn_id="git"))
    workspace_root = session.project_root
    assert session.workspace.mode.value == "git_worktree"
    assert workspace_root.exists()

    result = session.write_file(
        FileWriteRequest("tracked.txt", "changed\n", overwrite=True)
    )
    assert result.change_set is not None
    session.close()
    assert not workspace_root.exists()
    assert (repository / "tracked.txt").read_text(encoding="utf-8") == "base\n"

    applied = backend.apply_change_set(
        result.change_set.change_set_id,
        approval=_approval(result.change_set.change_set_id),
    )
    assert applied.success is True
    assert (repository / "tracked.txt").read_text(encoding="utf-8") == "changed\n"
    assert _git(repository, "worktree", "list", "--porcelain").stdout.count(
        "worktree "
    ) == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")
def test_git_worktree_dirty_policy_and_branch_cleanup(tmp_path):
    repository = _git_repository(tmp_path)
    (repository / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(WorkspacePolicyError, match="clean source"):
        GitWorktreeBackend(repository).open_session(
            ExecutionSessionRequest(turn_id="dirty")
        )

    (repository / "dirty.txt").unlink()
    backend = GitWorktreeBackend(
        repository,
        git_policy=GitWorktreePolicy(detached=False),
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="branch"))
    branches_during = _git(repository, "branch", "--format=%(refname:short)").stdout
    assert "chulk/workspace/branch-" in branches_during
    session.close()
    branches_after = _git(repository, "branch", "--format=%(refname:short)").stdout
    assert "chulk/workspace/branch-" not in branches_after


def test_git_worktree_rejects_narrow_allowlist_and_conflicting_lifecycle(tmp_path):
    with pytest.raises(WorkspacePolicyError, match="repository root"):
        GitWorktreeBackend(
            tmp_path,
            materialization_policy=WorkspaceMaterializationPolicy(
                allowed_paths=("tracked.txt",)
            ),
        )
    with pytest.raises(ValueError, match="conflicts"):
        GitWorktreeBackend(
            tmp_path,
            git_policy=GitWorktreePolicy(remove_on_close=False),
            persistence=WorkspacePersistence.EPHEMERAL,
        )


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")
def test_git_cleanup_prunes_metadata_when_workspace_was_removed(tmp_path):
    repository = _git_repository(tmp_path)
    backend = GitWorktreeBackend(repository)
    session = backend.open_session(ExecutionSessionRequest(turn_id="missing-root"))
    workspace_root = session.project_root
    shutil.rmtree(workspace_root)

    session.close()

    worktrees = _git(repository, "worktree", "list", "--porcelain").stdout
    assert worktrees.count("worktree ") == 1


def test_temporary_backend_rejects_direct_apply_policy(tmp_path):
    with pytest.raises(ValueError, match="cannot apply changes directly"):
        TemporaryWorkspaceBackend(
            tmp_path,
            change_disposition=ChangeDisposition.APPLY_DIRECTLY,
        )


def test_temporary_root_cannot_be_nested_inside_source(tmp_path):
    with pytest.raises(ValueError, match="cannot be inside"):
        TemporaryWorkspaceBackend(
            tmp_path,
            temporary_root=tmp_path / ".temporary",
        )


def test_discard_policy_does_not_publish_change_sets(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        change_disposition=ChangeDisposition.DISCARD,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="discard"))

    result = session.write_file(FileWriteRequest("discarded.txt", "discarded"))
    session.close()

    assert result.success is True
    assert result.change_set is None
    assert "change_set" not in result.metadata
    assert not (tmp_path / "discarded.txt").exists()


def test_backend_close_cleans_persistent_workspace_owned_by_backend(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        persistence=WorkspacePersistence.PERSISTENT,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="backend-close"))
    workspace_root = session.project_root
    session.close()
    assert workspace_root.exists()

    backend.close()

    assert not workspace_root.exists()


def test_backend_close_can_retry_failed_cleanup(tmp_path, monkeypatch):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        persistence=WorkspacePersistence.PERSISTENT,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="cleanup-retry"))
    workspace_root = session.project_root
    session.close()
    original_cleanup = backend._cleanup_materialized_workspace
    attempts = 0

    def fail_once(root, workspace_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated cleanup failure")
        original_cleanup(root, workspace_id)

    monkeypatch.setattr(backend, "_cleanup_materialized_workspace", fail_once)
    with pytest.raises(RuntimeError, match="Failed to clean"):
        backend.close()
    assert workspace_root.exists()

    backend.close()

    assert attempts == 2
    assert not workspace_root.exists()


@pytest.mark.asyncio
async def test_async_backend_close_cleans_retained_workspaces(tmp_path):
    backend = TemporaryWorkspaceBackend(
        tmp_path,
        persistence=WorkspacePersistence.PERSISTENT,
    )
    session = await backend.open_session_async(
        ExecutionSessionRequest(turn_id="async-backend-close")
    )
    workspace_root = session.project_root
    await session.aclose()

    await backend.aclose()

    assert not workspace_root.exists()
