"""Tests for durable governed skill lifecycle state."""

from __future__ import annotations

import sqlite3

import pytest

from chulk.skills import (
    LearningProposalKind,
    LearningProposalStatus,
    SQLiteSkillLifecycleStore,
    SkillLifecycleStatus,
    SkillManifest,
    SkillUsageKind,
)


def manifest(*, version: str = "1.0.0") -> SkillManifest:
    return SkillManifest(
        name="review",
        version=version,
        description="Review workflow.",
        source="project",
        trust="reviewed",
    )


def test_skill_lifecycle_migration_creates_governed_tables(tmp_path):
    path = tmp_path / "store.sqlite"
    SQLiteSkillLifecycleStore(path, profile_id="default")

    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }

    assert {
        "skill_packages",
        "skill_package_revisions",
        "skill_usage_events",
        "learning_proposals",
    } <= tables


def test_learning_proposals_are_profile_scoped_and_transition_once(tmp_path):
    path = tmp_path / "store.sqlite"
    alpha = SQLiteSkillLifecycleStore(path, profile_id="alpha")
    beta = SQLiteSkillLifecycleStore(path, profile_id="beta")

    proposal = alpha.create_proposal(
        kind=LearningProposalKind.SKILL_CREATE,
        target_name="review",
        rationale="Repeated review workflow.",
        evidence_turn_ids=("turn-1",),
        content="# Review\n",
        required_capabilities=("files:read",),
        verification_steps=("run manifest validation",),
        reviewer_model="reviewer",
        confidence=0.9,
    )

    assert proposal.status is LearningProposalStatus.PENDING
    assert alpha.list_proposals() == (proposal,)
    assert beta.list_proposals() == ()
    with pytest.raises(KeyError):
        beta.get_proposal(proposal.id)

    rejected = alpha.transition_proposal(
        proposal.id,
        status=LearningProposalStatus.REJECTED,
    )
    repeated = alpha.transition_proposal(
        proposal.id,
        status=LearningProposalStatus.APPROVED,
        applied_revision_id="should-not-apply",
    )

    assert rejected.status is LearningProposalStatus.REJECTED
    assert repeated.status is LearningProposalStatus.REJECTED
    assert repeated.applied_revision_id is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "kind": "skill_create",
            "rationale": "missing target",
            "content": "# Skill",
        },
        {
            "kind": "skill_patch",
            "target_name": "review",
            "rationale": "missing content",
        },
        {
            "kind": "skill_archive",
            "target_name": "review",
            "rationale": "",
        },
    ],
)
def test_learning_proposal_validation_rejects_incomplete_changes(tmp_path, kwargs):
    store = SQLiteSkillLifecycleStore(
        tmp_path / "store.sqlite",
        profile_id="default",
    )

    with pytest.raises(ValueError):
        store.create_proposal(**kwargs)


def test_skill_revisions_preserve_bytes_and_update_current_state(tmp_path):
    store = SQLiteSkillLifecycleStore(
        tmp_path / "store.sqlite",
        profile_id="default",
    )
    first = store.save_revision(
        manifest=manifest(),
        digest="sha256:first",
        package_files={
            "SKILL.md": b"# Review\n",
            "references/guide.md": b"\x00guide",
        },
    )
    second = store.save_revision(
        manifest=manifest(version="1.1.0"),
        digest="sha256:second",
        package_files={"SKILL.md": b"# Better review\n"},
        proposal_id="proposal-1",
        increment_patch=True,
    )

    current = store.get_skill("review")

    assert first.package_files["references/guide.md"] == b"\x00guide"
    assert second.proposal_id == "proposal-1"
    assert current.version == "1.1.0"
    assert current.digest == "sha256:second"
    assert current.active_revision_id == second.id
    assert current.patch_count == 1
    assert [item.id for item in store.list_revisions("review")] == [
        second.id,
        first.id,
    ]


def test_skill_usage_is_idempotent_and_success_requires_host_confirmation(tmp_path):
    store = SQLiteSkillLifecycleStore(
        tmp_path / "store.sqlite",
        profile_id="default",
    )
    store.save_revision(
        manifest=manifest(),
        digest="sha256:first",
        package_files={"SKILL.md": b"# Review\n"},
    )

    used = store.record_usage(
        name="review",
        version="1.0.0",
        digest="sha256:first",
        kind=SkillUsageKind.USE,
        source_event_id="turn-1",
    )
    repeated = store.record_usage(
        name="review",
        version="1.0.0",
        digest="sha256:first",
        kind=SkillUsageKind.USE,
        source_event_id="turn-1",
    )

    assert used.use_count == 1
    assert repeated.use_count == 1
    with pytest.raises(ValueError, match="host confirmation"):
        store.record_usage(
            name="review",
            version="1.0.0",
            digest="sha256:first",
            kind=SkillUsageKind.SUCCESS,
            source_event_id="turn-1-success",
        )

    successful = store.record_usage(
        name="review",
        version="1.0.0",
        digest="sha256:first",
        kind=SkillUsageKind.SUCCESS,
        source_event_id="turn-1-success",
        host_confirmed=True,
    )
    assert successful.success_count == 1

    with pytest.raises(ValueError, match="current version and digest"):
        store.record_usage(
            name="review",
            version="0.9.0",
            digest="sha256:old",
            kind=SkillUsageKind.VIEW,
            source_event_id="old-view",
        )


def test_skill_status_and_revision_activation_support_rollback(tmp_path):
    store = SQLiteSkillLifecycleStore(
        tmp_path / "store.sqlite",
        profile_id="default",
    )
    first = store.save_revision(
        manifest=manifest(),
        digest="sha256:first",
        package_files={"SKILL.md": b"# Review\n"},
    )
    store.save_revision(
        manifest=manifest(version="2.0.0"),
        digest="sha256:second",
        package_files={"SKILL.md": b"# Changed\n"},
        increment_patch=True,
    )

    archived = store.set_skill_status("review", SkillLifecycleStatus.ARCHIVED)
    restored = store.activate_revision(first.id)

    assert archived.status is SkillLifecycleStatus.ARCHIVED
    assert restored.status is SkillLifecycleStatus.ACTIVE
    assert restored.version == "1.0.0"
    assert restored.digest == "sha256:first"
    assert restored.patch_count == 2
