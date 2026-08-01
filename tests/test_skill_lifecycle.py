"""Tests for host-owned skill application, locks, and rollback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat

import pytest

from chulk.skills import (
    LearningProposalKind,
    LearningProposalStatus,
    SQLiteSkillLifecycleStore,
    SkillApprovalError,
    SkillConflictError,
    SkillLifecycleManager,
    SkillLifecycleStatus,
    SkillLockFile,
    load_skill_package,
    proposal_diff,
)


def skill_content(
    *,
    name: str = "review",
    version: str = "1.0.0",
    body: str = "# Review\n\nReview carefully.\n",
    references: tuple[str, ...] = (),
) -> str:
    reference_line = (
        f"references: [{', '.join(references)}]\n" if references else ""
    )
    return f"""\
---
schema_version: 1
name: {name}
version: {version}
description: Review workflow.
required_capabilities: []
{reference_line}source: project
trust: reviewed
---
{body}"""


def write_skill(
    skills_dir: Path,
    content: str,
    *,
    name: str = "review",
) -> Path:
    root = skills_dir / name
    root.mkdir(parents=True)
    path = root / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


def manager(tmp_path):
    store = SQLiteSkillLifecycleStore(
        tmp_path / "store.sqlite",
        profile_id="default",
    )
    project_skills = tmp_path / "project" / ".chulk" / "skills"
    profile_skills = tmp_path / "profile-runtime" / "skills"
    lifecycle = SkillLifecycleManager(
        store,
        project_skills_dir=project_skills,
        profile_skills_dir=profile_skills,
    )
    return store, lifecycle, project_skills, profile_skills


def test_register_existing_writes_distinct_credential_free_locks(tmp_path):
    store, lifecycle, project_skills, profile_skills = manager(tmp_path)
    write_skill(project_skills, skill_content())
    write_skill(
        profile_skills,
        skill_content(name="personal", body="# Personal\n"),
        name="personal",
    )

    project_records = lifecycle.register_existing(scope="project")
    profile_records = lifecycle.register_existing(scope="profile")

    assert [record.name for record in project_records] == ["review"]
    assert [record.name for record in profile_records] == ["personal"]
    assert lifecycle.project_lock.path != lifecycle.profile_lock.path
    assert lifecycle.project_lock.get("review").digest.startswith("sha256:")
    assert lifecycle.profile_lock.get("personal").version == "1.0.0"
    assert "credential" not in lifecycle.project_lock.path.read_text(
        encoding="utf-8"
    )
    if os.name == "posix":
        assert stat.S_IMODE(lifecycle.project_lock.path.stat().st_mode) == 0o644
        assert stat.S_IMODE(lifecycle.profile_lock.path.stat().st_mode) == 0o600
    assert {record.name for record in store.list_skills()} == {
        "personal",
        "review",
    }


def test_project_and_profile_can_govern_the_same_skill_name(tmp_path):
    store, lifecycle, project_skills, profile_skills = manager(tmp_path)
    write_skill(project_skills, skill_content(version="1.0.0"))
    write_skill(profile_skills, skill_content(version="2.0.0"))

    project_record = lifecycle.register_existing(scope="project")[0]
    profile_record = lifecycle.register_existing(scope="profile")[0]

    assert project_record.scope == "project"
    assert project_record.version == "1.0.0"
    assert profile_record.scope == "profile"
    assert profile_record.version == "2.0.0"
    assert store.get_skill("review", scope="project") == project_record
    assert store.get_skill("review", scope="profile") == profile_record
    assert lifecycle.project_lock.get("review").version == "1.0.0"
    assert lifecycle.profile_lock.get("review").version == "2.0.0"


def test_registering_unchanged_skills_keeps_lock_content_stable(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    write_skill(project_skills, skill_content())

    first = lifecycle.register_existing(scope="project")[0]
    first_lock = lifecycle.project_lock.path.read_bytes()
    second = lifecycle.register_existing(scope="project")[0]

    assert second.updated_at == first.updated_at
    assert lifecycle.project_lock.path.read_bytes() == first_lock
    assert len(store.list_revisions("review")) == 1


def test_host_approval_creates_validated_skill_and_updates_lock(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    content = skill_content()
    proposal = store.create_proposal(
        kind=LearningProposalKind.SKILL_CREATE,
        target_name="review",
        rationale="The workflow succeeded repeatedly.",
        content=content,
        required_capabilities=(),
        verification_steps=("validate the manifest",),
        metadata={"scope": "project"},
    )

    approved = lifecycle.approve(proposal.id, approved_by="operator")
    package = load_skill_package(project_skills / "review")
    record = store.get_skill("review")
    lock = lifecycle.project_lock.get("review")

    assert approved.status is LearningProposalStatus.APPROVED
    assert approved.reviewed_by == "operator"
    assert approved.applied_revision_id == record.active_revision_id
    assert package.digest == record.digest
    assert lock is not None
    assert lock.digest == package.digest
    assert lifecycle.verify_locks() == ()


def test_patch_requires_reviewed_base_and_preserves_package_resources(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    original = skill_content(references=("guide.md",))
    path = write_skill(project_skills, original)
    (path.parent / "guide.md").write_text("guide", encoding="utf-8")
    lifecycle.register_existing(scope="project")
    base = load_skill_package(path)
    changed = skill_content(
        version="1.1.0",
        body="# Review\n\nReview with tests.\n",
        references=("guide.md",),
    )
    proposal = store.create_proposal(
        kind="skill_patch",
        target_name="review",
        rationale="Verified improvement.",
        content=changed,
        diff=proposal_diff(before=original, after=changed, name="review"),
        metadata={"scope": "project", "base_digest": base.digest},
    )

    approved = lifecycle.approve(proposal.id, approved_by="operator")

    assert approved.status is LearningProposalStatus.APPROVED
    assert (path.parent / "guide.md").read_text(encoding="utf-8") == "guide"
    assert load_skill_package(path).manifest.version == "1.1.0"
    assert store.get_skill("review").patch_count == 1


def test_patch_conflict_leaves_package_and_proposal_unchanged(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    original = skill_content()
    path = write_skill(project_skills, original)
    lifecycle.register_existing(scope="project")
    base = load_skill_package(path)
    changed = skill_content(version="1.1.0")
    proposal = store.create_proposal(
        kind="skill_patch",
        target_name="review",
        rationale="Reviewed against the first version.",
        content=changed,
        diff=proposal_diff(before=original, after=changed, name="review"),
        metadata={"scope": "project", "base_digest": base.digest},
    )
    path.write_text(skill_content(version="1.0.1"), encoding="utf-8")

    with pytest.raises(SkillConflictError, match="changed after"):
        lifecycle.approve(proposal.id, approved_by="operator")

    assert path.read_text(encoding="utf-8") == skill_content(version="1.0.1")
    assert (
        store.get_proposal(proposal.id).status
        is LearningProposalStatus.PENDING
    )


def test_failed_database_approval_restores_package_and_lock(
    tmp_path,
    monkeypatch,
):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    original = skill_content()
    path = write_skill(project_skills, original)
    lifecycle.register_existing(scope="project")
    base = load_skill_package(path)
    original_lock = lifecycle.project_lock.path.read_bytes()
    changed = skill_content(version="1.1.0")
    proposal = store.create_proposal(
        kind="skill_patch",
        target_name="review",
        rationale="Reviewed update.",
        content=changed,
        diff=proposal_diff(before=original, after=changed, name="review"),
        metadata={"scope": "project", "base_digest": base.digest},
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "approve_skill_revision", fail)

    with pytest.raises(RuntimeError, match="database unavailable"):
        lifecycle.approve(proposal.id, approved_by="operator")

    assert path.read_text(encoding="utf-8") == original
    assert lifecycle.project_lock.path.read_bytes() == original_lock
    assert (
        store.get_proposal(proposal.id).status
        is LearningProposalStatus.PENDING
    )


def test_archive_and_rollback_restore_exact_package_snapshot(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    content = skill_content(references=("guide.md",))
    path = write_skill(project_skills, content)
    (path.parent / "guide.md").write_bytes(b"\x00guide")
    lifecycle.register_existing(scope="project")
    current = store.get_skill("review")
    proposal = store.create_proposal(
        kind="skill_archive",
        target_name="review",
        rationale="Workflow is obsolete.",
        diff=proposal_diff(before=content, after=None, name="review"),
        metadata={"scope": "project", "base_digest": current.digest},
    )

    approved = lifecycle.approve(proposal.id, approved_by="operator")

    assert approved.status is LearningProposalStatus.APPROVED
    assert approved.applied_revision_id == current.active_revision_id
    assert not path.parent.exists()
    assert (
        lifecycle.project_lock.get("review").status
        is SkillLifecycleStatus.ARCHIVED
    )
    assert lifecycle.verify_locks() == ()
    restored = lifecycle.rollback(
        current.active_revision_id,
        scope="project",
        approved_by="operator",
    )

    assert restored.status is SkillLifecycleStatus.ACTIVE
    assert path.read_text(encoding="utf-8") == content
    assert (path.parent / "guide.md").read_bytes() == b"\x00guide"
    assert lifecycle.verify_locks() == ()


def test_rollback_rejects_revision_from_another_scope(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    write_skill(project_skills, skill_content())
    revision_id = lifecycle.register_existing(
        scope="project"
    )[0].active_revision_id

    with pytest.raises(SkillConflictError, match="different lifecycle scope"):
        lifecycle.rollback(
            revision_id,
            scope="profile",
            approved_by="operator",
        )


def test_proposal_with_embedded_secret_is_never_applied(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    content = skill_content(
        body="# Review\n\nOPENAI_API_KEY=sk-abcdefghijklmnop\n"
    )
    proposal = store.create_proposal(
        kind="skill_create",
        target_name="review",
        rationale="Proposed workflow.",
        content=content,
        metadata={"scope": "project"},
    )

    with pytest.raises(ValueError, match="Credential-like data"):
        lifecycle.approve(proposal.id, approved_by="operator")

    assert not (project_skills / "review").exists()
    assert (
        store.get_proposal(proposal.id).status
        is LearningProposalStatus.PENDING
    )


def test_lock_rejects_symlink_target(tmp_path):
    target = tmp_path / "actual.lock"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "skills.lock"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    lock = SkillLockFile(link, scope="project", private=False)

    with pytest.raises(ValueError, match="regular file"):
        lock.read()


def test_registration_rejects_symlinked_skill_package(tmp_path):
    _store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    outside = tmp_path / "outside"
    write_skill(outside, skill_content())
    project_skills.mkdir(parents=True)
    try:
        (project_skills / "review").symlink_to(
            outside / "review",
            target_is_directory=True,
        )
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(SkillApprovalError, match="cannot be a symlink"):
        lifecycle.register_existing(scope="project")


def test_registration_rejects_unreviewed_package_change(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    path = write_skill(project_skills, skill_content())
    original = lifecycle.register_existing(scope="project")[0]
    path.write_text(skill_content(version="1.0.1"), encoding="utf-8")

    with pytest.raises(SkillConflictError, match="outside the approval"):
        lifecycle.register_existing(scope="project")

    assert store.get_skill("review").digest == original.digest
    assert lifecycle.project_lock.get("review").digest == original.digest


def test_registration_rejects_partial_pending_create(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    content = skill_content()
    store.create_proposal(
        kind="skill_create",
        target_name="review",
        rationale="Pending host review.",
        content=content,
        metadata={"scope": "project"},
    )
    write_skill(project_skills, content)

    with pytest.raises(SkillConflictError, match="uncommitted lifecycle state"):
        lifecycle.register_existing(scope="project")

    with pytest.raises(KeyError):
        store.get_skill("review")
    assert lifecycle.project_lock.get("review") is None


def test_registration_rejects_missing_active_package(tmp_path):
    _store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    path = write_skill(project_skills, skill_content())
    lifecycle.register_existing(scope="project")
    shutil.rmtree(path.parent)

    with pytest.raises(SkillConflictError, match="package 'review' is missing"):
        lifecycle.register_existing(scope="project")


def test_registration_rejects_package_directory_name_mismatch(tmp_path):
    _store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    write_skill(project_skills, skill_content(name="review"), name="other")

    with pytest.raises(SkillApprovalError, match="package directory"):
        lifecycle.register_existing(scope="project")


def test_registration_rejects_reappearing_archived_package(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    content = skill_content()
    path = write_skill(project_skills, content)
    lifecycle.register_existing(scope="project")
    current = store.get_skill("review")
    proposal = store.create_proposal(
        kind="skill_archive",
        target_name="review",
        rationale="No longer needed.",
        diff=proposal_diff(before=content, after=None, name="review"),
        metadata={"scope": "project", "base_digest": current.digest},
    )
    lifecycle.approve(proposal.id, approved_by="operator")
    write_skill(project_skills, content)

    with pytest.raises(SkillConflictError, match="is archived"):
        lifecycle.register_existing(scope="project")

    assert path.exists()
    assert store.get_skill("review").status is SkillLifecycleStatus.ARCHIVED


def test_verify_locks_reports_database_mismatch(tmp_path):
    store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    write_skill(project_skills, skill_content())
    lifecycle.register_existing(scope="project")
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            """
            UPDATE skill_packages
            SET digest = ?
            WHERE profile_id = ? AND scope = ? AND name = ?
            """,
            (f"sha256:{'f' * 64}", "default", "project", "review"),
        )

    assert lifecycle.verify_locks() == (
        "project:review:database_digest_mismatch",
    )


def test_registration_rejects_tampered_lock_metadata(tmp_path):
    _store, lifecycle, project_skills, _profile_skills = manager(tmp_path)
    write_skill(project_skills, skill_content())
    lifecycle.register_existing(scope="project")
    lock_path = lifecycle.project_lock.path
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["skills"]["review"]["required_capabilities"] = ["shell"]
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SkillConflictError, match="governed state"):
        lifecycle.register_existing(scope="project")

    assert lifecycle.verify_locks() == (
        "project:review:lock_metadata_mismatch",
    )


def test_lock_rejects_path_traversal_entry(tmp_path):
    path = tmp_path / "skills.lock"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "project",
                "skills": {
                    "../escape": {
                        "name": "../escape",
                        "version": "1.0.0",
                        "digest": f"sha256:{'a' * 64}",
                        "source": "project",
                        "trust": "reviewed",
                        "status": "active",
                        "revision_id": "revision",
                        "required_tools": [],
                        "required_capabilities": [],
                        "installed_at": "2026-01-01T00:00:00+00:00",
                        "review_state": "approved",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid skill name"):
        SkillLockFile(path, scope="project", private=False).read()
