"""Tests for the unified proposal queue and restricted learning reviewer."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from chulk.memory import SQLiteMemoryStore
from chulk.skills import (
    AutomaticLearningBlocked,
    LearningProposalDraft,
    LearningProposalKind,
    LearningProposalService,
    LearningProposalStatus,
    LearningReviewContext,
    LearningReviewCoordinator,
    LearningReviewError,
    LearningReviewPolicy,
    LearningReviewQuota,
    LearningReviewQuotaExceeded,
    LearningReviewTrigger,
    RestrictedLearningReviewer,
    SQLiteSkillLifecycleStore,
    SkillLifecycleManager,
)
from chulk.testing import ScriptedLLMClient


def services(
    tmp_path: Path,
    *,
    automatic_approval_enabled: bool = False,
):
    db_path = tmp_path / "store.sqlite"
    memory_store = SQLiteMemoryStore(db_path)
    lifecycle_store = SQLiteSkillLifecycleStore(
        db_path,
        profile_id="default",
    )
    project_skills = tmp_path / "project" / ".chulk" / "skills"
    lifecycle = SkillLifecycleManager(
        lifecycle_store,
        project_skills_dir=project_skills,
        profile_skills_dir=tmp_path / "profile" / "skills",
    )
    proposals = LearningProposalService(
        memory_store=memory_store,
        lifecycle_store=lifecycle_store,
        lifecycle_manager=lifecycle,
        automatic_approval_enabled=automatic_approval_enabled,
    )
    return memory_store, lifecycle_store, lifecycle, proposals, project_skills


def context(
    trigger: LearningReviewTrigger = LearningReviewTrigger.MANUAL,
    *,
    successful: bool = False,
    tool_calls: int = 0,
) -> LearningReviewContext:
    return LearningReviewContext(
        trigger=trigger,
        user_message="Please remember the validated workflow.",
        assistant_response="The workflow completed successfully.",
        turn_id="turn-1",
        source_trace="trace-1",
        tool_call_count=tool_calls,
        host_confirmed_success=successful,
    )


def skill_content() -> str:
    return """\
---
schema_version: 1
name: review
version: 1.0.0
description: Review workflow.
required_capabilities: []
source: agent_proposed
trust: reviewed
---
# Review

Run the verified review workflow.
"""


def test_unified_queue_preserves_legacy_memory_proposal_identity(tmp_path):
    memory, lifecycle_store, _lifecycle, proposals, _skills = services(tmp_path)
    legacy_id = memory.create_memory_proposal(
        "User prefers concise answers.",
        evidence="Explicit correction.",
        conversation_id="conversation-1",
        turn_id="turn-1",
    )
    governed = lifecycle_store.create_proposal(
        kind="memory_create",
        rationale="Validated preference.",
        content="User prefers direct summaries.",
    )

    pending = proposals.list()
    legacy = proposals.get(legacy_id)

    assert {item.id for item in pending} == {legacy_id, governed.id}
    assert legacy.id == legacy_id
    assert legacy.evidence_turn_ids == ("turn-1",)
    assert legacy.source_trace == "conversation-1"
    approved = proposals.approve(legacy_id, approved_by="operator")
    assert approved.status is LearningProposalStatus.APPROVED
    assert approved.accepted_memory_id is not None


def test_memory_create_and_update_apply_only_after_host_approval(tmp_path):
    memory, lifecycle_store, _lifecycle, proposals, _skills = services(tmp_path)
    created = proposals.create(
        LearningProposalDraft(
            kind=LearningProposalKind.MEMORY_CREATE,
            rationale="Durable explicit preference.",
            content="User prefers compact output.",
            confidence=0.9,
            metadata={
                "memory": {
                    "tags": ["preference"],
                    "importance": 5,
                    "source": "learning_review",
                }
            },
        )
    )

    assert memory.list_memories() == []
    accepted = proposals.approve(created.id, approved_by="operator")
    memory_id = accepted.accepted_memory_id
    assert memory_id is not None
    assert memory.get_memory(memory_id).content == "User prefers compact output."

    changed = lifecycle_store.create_proposal(
        kind="memory_update",
        target_name=memory_id,
        rationale="The user corrected the preference.",
        content="User prefers compact technical output.",
        confidence=1.0,
        metadata={"memory": {"tags": ["preference", "style"]}},
    )
    proposals.approve(changed.id, approved_by="operator")

    updated = memory.get_memory(memory_id)
    assert updated is not None
    assert updated.content == "User prefers compact technical output."
    assert updated.tags == ["preference", "style"]


def test_memory_approval_rolls_back_when_the_transaction_fails(
    tmp_path,
    monkeypatch,
):
    memory, _store, _lifecycle, proposals, _skills = services(tmp_path)
    proposal = proposals.create(
        LearningProposalDraft(
            kind=LearningProposalKind.MEMORY_CREATE,
            rationale="Durable preference.",
            content="User prefers compact output.",
        )
    )
    original = memory.save_memory_in_connection

    def fail_after_insert(conn, content, **kwargs):
        original(conn, content, **kwargs)
        raise RuntimeError("approval interrupted")

    monkeypatch.setattr(memory, "save_memory_in_connection", fail_after_insert)

    with pytest.raises(RuntimeError, match="interrupted"):
        proposals.approve(proposal.id, approved_by="operator")

    assert memory.list_memories() == []
    assert proposals.get(proposal.id).status is LearningProposalStatus.PENDING


def test_automatic_approval_is_opt_in_and_never_expands_authority(tmp_path):
    _memory, lifecycle_store, _lifecycle, proposals, _skills = services(
        tmp_path
    )
    proposal = lifecycle_store.create_proposal(
        kind="skill_create",
        target_name="review",
        rationale="Reusable workflow.",
        content=skill_content(),
        required_capabilities=("shell:execute",),
        metadata={"scope": "project"},
    )

    with pytest.raises(AutomaticLearningBlocked, match="disabled"):
        proposals.approve(
            proposal.id,
            approved_by="automatic-reviewer",
            automatic=True,
            granted_capabilities=("shell:execute",),
        )

    _memory, lifecycle_store, _lifecycle, enabled, _skills = services(
        tmp_path / "enabled",
        automatic_approval_enabled=True,
    )
    external = lifecycle_store.create_proposal(
        kind="memory_create",
        rationale="External suggestion.",
        content="Untrusted preference.",
        metadata={"external_source": True},
    )
    with pytest.raises(AutomaticLearningBlocked, match="external-source"):
        enabled.approve(
            external.id,
            approved_by="automatic-reviewer",
            automatic=True,
        )

    capability = lifecycle_store.create_proposal(
        kind="skill_create",
        target_name="review",
        rationale="Needs more authority.",
        content=skill_content(),
        required_capabilities=("shell:execute",),
        metadata={"scope": "project"},
    )
    with pytest.raises(AutomaticLearningBlocked, match="capability-increasing"):
        enabled.approve(
            capability.id,
            approved_by="automatic-reviewer",
            automatic=True,
        )


def test_reviewer_no_action_has_no_mutation_authority(tmp_path):
    _memory, store, _lifecycle, proposals, _skills = services(tmp_path)
    client = ScriptedLLMClient(
        [{"decision": "no_action", "rationale": "No durable learning."}]
    )
    coordinator = LearningReviewCoordinator(
        reviewer=RestrictedLearningReviewer(client),
        proposal_service=proposals,
        lifecycle_store=store,
    )

    outcome = coordinator.review(context())

    assert outcome.skipped is False
    assert outcome.proposal_ids == ()
    assert proposals.list() == ()
    assert len(client.call_log) == 1
    call = client.call_log[0]
    assert call["max_output_tokens"] == 2_000
    assert len(call["messages"]) == 2
    assert "no authority" in call["messages"][0]["content"]


def test_success_review_requires_explicit_opt_in_and_host_confirmation(tmp_path):
    _memory, store, _lifecycle, proposals, _skills = services(tmp_path)
    client = ScriptedLLMClient(
        [{"decision": "no_action", "rationale": "Nothing to retain."}]
    )
    coordinator = LearningReviewCoordinator(
        reviewer=RestrictedLearningReviewer(client),
        proposal_service=proposals,
        lifecycle_store=store,
        policy=LearningReviewPolicy(successful_turn_review_enabled=True),
    )

    skipped = coordinator.review(
        context(
            LearningReviewTrigger.SUCCESSFUL_TURN,
            successful=False,
            tool_calls=4,
        )
    )
    reviewed = coordinator.review(
        context(
            LearningReviewTrigger.SUCCESSFUL_TURN,
            successful=True,
            tool_calls=4,
        )
    )

    assert skipped.skipped is True
    assert reviewed.skipped is False
    assert len(client.call_log) == 1


def test_reviewer_proposes_but_does_not_install_a_skill(tmp_path):
    _memory, store, lifecycle, proposals, project_skills = services(tmp_path)
    response = {
        "decision": "propose",
        "rationale": "The workflow repeated.",
        "proposals": [
            {
                "kind": "skill_create",
                "target_name": "review",
                "rationale": "Capture the reusable workflow.",
                "content": skill_content(),
                "required_capabilities": [],
                "confidence": 0.9,
                "verification_steps": ["validate package"],
                "metadata": {"scope": "project"},
            }
        ],
    }
    client = ScriptedLLMClient([response])
    coordinator = LearningReviewCoordinator(
        reviewer=RestrictedLearningReviewer(client),
        proposal_service=proposals,
        lifecycle_store=store,
    )

    outcome = coordinator.review(context())

    assert len(outcome.proposal_ids) == 1
    assert not (project_skills / "review").exists()
    proposal = proposals.get(outcome.proposal_ids[0])
    assert proposal.status is LearningProposalStatus.PENDING

    lifecycle.approve(proposal.id, approved_by="operator")
    assert (project_skills / "review" / "SKILL.md").exists()


def test_daily_proposal_quota_is_reserved_before_another_model_call(tmp_path):
    _memory, store, _lifecycle, proposals, _skills = services(tmp_path)
    response = {
        "decision": "propose",
        "rationale": "Remember the preference.",
        "proposals": [
            {
                "kind": "memory_create",
                "rationale": "Explicit durable preference.",
                "content": "User prefers concise answers.",
                "confidence": 1.0,
                "verification_steps": [],
            }
        ],
    }
    client = ScriptedLLMClient([response, response])
    coordinator = LearningReviewCoordinator(
        reviewer=RestrictedLearningReviewer(client),
        proposal_service=proposals,
        lifecycle_store=store,
        quota=LearningReviewQuota(
            max_proposals_per_day=1,
            max_proposals_per_review=1,
            max_tokens_per_day=10_000,
            max_output_tokens=500,
        ),
    )

    coordinator.review(context())
    with pytest.raises(LearningReviewQuotaExceeded, match="proposal"):
        coordinator.review(context())

    assert len(client.call_log) == 1


def test_invalid_reviewer_output_is_failed_without_a_proposal(tmp_path):
    _memory, store, _lifecycle, proposals, _skills = services(tmp_path)
    client = ScriptedLLMClient(["not-json"])
    coordinator = LearningReviewCoordinator(
        reviewer=RestrictedLearningReviewer(client),
        proposal_service=proposals,
        lifecycle_store=store,
    )

    with pytest.raises(LearningReviewError, match="invalid JSON"):
        coordinator.review(context())

    assert proposals.list() == ()
    usage = store.review_usage_since("1970-01-01T00:00:00+00:00")
    assert usage.token_count > 0
    assert usage.proposal_count == 0


def test_reviewer_batch_is_all_or_nothing(tmp_path):
    _memory, store, _lifecycle, proposals, _skills = services(tmp_path)
    response = {
        "decision": "propose",
        "rationale": "Two possible memories.",
        "proposals": [
            {
                "kind": "memory_create",
                "rationale": "Valid preference.",
                "content": "User prefers concise answers.",
                "confidence": 1.0,
                "verification_steps": [],
            },
            {
                "kind": "memory_update",
                "target_name": "missing-memory",
                "rationale": "Invalid proposal without content.",
                "confidence": 1.0,
                "verification_steps": [],
            },
        ],
    }
    coordinator = LearningReviewCoordinator(
        reviewer=RestrictedLearningReviewer(ScriptedLLMClient([response])),
        proposal_service=proposals,
        lifecycle_store=store,
    )

    with pytest.raises(ValueError, match="require content"):
        coordinator.review(context())

    assert proposals.list() == ()


def test_reviewer_secret_is_rejected_before_proposal_persistence(tmp_path):
    _memory, store, _lifecycle, proposals, _skills = services(tmp_path)
    response = {
        "decision": "propose",
        "rationale": "Remember a credential.",
        "proposals": [
            {
                "kind": "memory_create",
                "rationale": "Unsafe value.",
                "content": "OPENAI_API_KEY=sk-abcdefghijklmnop",
                "confidence": 1.0,
                "verification_steps": [],
            }
        ],
    }
    coordinator = LearningReviewCoordinator(
        reviewer=RestrictedLearningReviewer(ScriptedLLMClient([response])),
        proposal_service=proposals,
        lifecycle_store=store,
    )

    with pytest.raises(ValueError, match="Credential-like"):
        coordinator.review(context())

    assert proposals.list() == ()


def test_cost_quota_requires_a_per_review_reservation():
    with pytest.raises(ValueError, match="max_cost_per_review"):
        LearningReviewQuota(max_cost_per_day=Decimal("1.00"))
