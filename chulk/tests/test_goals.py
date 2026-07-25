"""Outside-in tests for durable goal state, persistence, and recovery."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chulk.core.state import Plan, PlanStep
from chulk.goals import (
    GoalActionConflictError,
    GoalLeaseConflictError,
    GoalRetentionPolicy,
    GoalRevisionConflictError,
    GoalRisk,
    GoalService,
    GoalStatus,
    GoalStep,
    GoalStepStatus,
    GoalStore,
    InvalidGoalTransitionError,
)
from chulk.usage import BudgetScope, RunBudget


NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)


def _service(tmp_path, *, profile_id: str = "default", propagate=None) -> GoalService:
    store = GoalStore(
        tmp_path / "control.sqlite",
        profile_id=profile_id,
        clock=lambda: NOW,
    )
    return GoalService(
        store,
        clock=lambda: NOW,
        cancellation_propagator=propagate,
    )


def _create_goal(service: GoalService, *, risk: GoalRisk = GoalRisk.LOW):
    return service.create(
        title="Ship durable goals",
        acceptance_criteria=("Tests prove restart recovery",),
        steps=(
            GoalStep(
                id="implement",
                title="Implement",
                description="Build the goal store",
                acceptance_criterion_ids=("criterion-1",),
                risk=risk,
                max_attempts=2,
            ),
        ),
        budget=RunBudget(
            max_model_calls=3,
            max_tool_calls=4,
            max_tokens=10_000,
        ),
    )


def test_goal_lifecycle_requires_approval_evidence_and_valid_transitions(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)

    with pytest.raises(InvalidGoalTransitionError):
        service.run(goal.id, expected_revision=goal.revision, actor="owner")

    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    assert goal.status is GoalStatus.RUNNING
    assert goal.steps[0].status is GoalStepStatus.READY

    goal = service.start_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    with pytest.raises(InvalidGoalTransitionError, match="lacks evidence"):
        service.complete_step(
            goal.id,
            "implement",
            expected_revision=goal.revision,
            actor="runner",
        )

    goal = service.add_evidence(
        goal.id,
        expected_revision=goal.revision,
        summary="The restart test passed.",
        criterion_ids=("criterion-1",),
        step_id="implement",
        reference="trace:abc",
    )
    goal = service.complete_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    goal = service.complete(
        goal.id,
        expected_revision=goal.revision,
        actor="runner",
    )

    assert goal.status is GoalStatus.COMPLETED
    assert goal.missing_criterion_ids == ()
    assert [event.revision for event in service.store.events(goal.id)] == list(
        range(goal.revision + 1)
    )


def test_goal_service_emits_committed_typed_events(tmp_path) -> None:
    observed: list[tuple[str, int, str]] = []
    store = GoalStore(tmp_path / "control.sqlite", clock=lambda: NOW)
    service = GoalService(
        store,
        clock=lambda: NOW,
        event_callback=lambda event, goal: observed.append(
            (event.kind, event.revision, goal.status.value)
        ),
    )

    goal = _create_goal(service)
    service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )

    assert observed == [
        ("goal.created", 0, "draft"),
        ("goal.approved", 1, "approved"),
    ]


def test_goal_store_rejects_stale_operators_and_is_profile_scoped(tmp_path) -> None:
    service = _service(tmp_path, profile_id="work")
    goal = _create_goal(service)
    service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner-a",
    )

    with pytest.raises(GoalRevisionConflictError) as exc_info:
        service.approve(
            goal.id,
            expected_revision=goal.revision,
            approved_by="owner-b",
        )
    assert exc_info.value.actual == 1

    other = _service(tmp_path, profile_id="personal")
    with pytest.raises(LookupError):
        other.store.get(goal.id)


def test_plan_promotion_copies_source_and_does_not_follow_later_mutation(tmp_path) -> None:
    service = _service(tmp_path)
    plan = Plan(
        summary="Review and ship",
        steps=[
            PlanStep(
                id="review",
                title="Review",
                description="Review the change",
                acceptance_criteria=["Review passes"],
                retry_limit=2,
            ),
            PlanStep(
                id="ship",
                title="Ship",
                description="Ship the change",
                depends_on=["review"],
                acceptance_criteria=["Change is merged"],
            ),
        ],
    )

    goal = service.promote_plan(
        plan,
        profile_id="default",
        conversation_id="conversation-1",
        turn_id="turn-1",
        budget=RunBudget(max_model_calls=5),
    )
    plan.steps[0].title = "Mutated later"

    restored = GoalStore(tmp_path / "control.sqlite").get(goal.id)
    assert restored.steps[0].title == "Review"
    assert restored.source_conversation_id == "conversation-1"
    assert restored.source_turn_id == "turn-1"
    assert restored.budget.scope is BudgetScope.GOAL
    assert restored.source_plan is not None
    assert restored.source_plan["steps"][0]["title"] == "Review"
    with pytest.raises(TypeError):
        restored.source_plan["steps"][0]["title"] = "Cannot mutate"  # type: ignore[index]


def test_high_risk_step_requires_selected_step_approval(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service, risk=GoalRisk.HIGH)
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")

    with pytest.raises(InvalidGoalTransitionError, match="selected-step approval"):
        service.start_step(
            goal.id,
            "implement",
            expected_revision=goal.revision,
            actor="runner",
        )

    goal = service.approve_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.start_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    assert goal.steps[0].status is GoalStepStatus.RUNNING


def test_skip_is_audited_but_cannot_waive_goal_evidence(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    goal = service.skip_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        reason="Replaced by external verification",
        approved_by="owner",
    )

    assert goal.steps[0].status is GoalStepStatus.SKIPPED
    with pytest.raises(InvalidGoalTransitionError, match="lacks evidence"):
        service.complete(
            goal.id,
            expected_revision=goal.revision,
            actor="owner",
        )

    goal = service.add_evidence(
        goal.id,
        expected_revision=goal.revision,
        summary="External verification completed.",
        criterion_ids=("criterion-1",),
        recorded_by="owner",
    )
    assert service.complete(
        goal.id,
        expected_revision=goal.revision,
        actor="owner",
    ).status is GoalStatus.COMPLETED


def test_single_runner_lease_and_boundary_cancellation(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    goal = service.start_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    claim = service.store.claim(
        goal.id,
        runner_id="runner-a",
        expected_revision=goal.revision,
        now=NOW,
    )

    with pytest.raises(GoalLeaseConflictError):
        service.store.claim(
            goal.id,
            runner_id="runner-b",
            expected_revision=goal.revision,
            now=NOW,
        )
    assert service.store.assert_action_boundary(
        claim,
        step_id="implement",
        now=NOW,
    ).id == goal.id

    goal = service.request_cancel(
        goal.id,
        expected_revision=goal.revision,
        actor="owner",
    )
    with pytest.raises(GoalLeaseConflictError, match="cancellation"):
        service.store.assert_action_boundary(
            claim,
            step_id="implement",
            now=NOW,
        )


def test_pause_and_steering_are_observed_between_active_step_actions(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    goal = service.start_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    claim = service.store.claim(
        goal.id,
        runner_id="runner",
        expected_revision=goal.revision,
        now=NOW,
    )
    goal = service.steer(
        goal.id,
        expected_revision=goal.revision,
        instruction="Use the reversible path.",
        created_by="owner",
    )
    assert service.store.assert_action_boundary(
        claim,
        step_id="implement",
        now=NOW,
    ).steering[-1].instruction == "Use the reversible path."

    goal = service.pause(
        goal.id,
        expected_revision=goal.revision,
        actor="owner",
    )
    assert goal.status is GoalStatus.PAUSED
    with pytest.raises(GoalLeaseConflictError, match="not running"):
        service.store.assert_action_boundary(
            claim,
            step_id="implement",
            now=NOW,
        )
    goal = service.resume(
        goal.id,
        expected_revision=goal.revision,
        actor="owner",
    )
    assert goal.status is GoalStatus.RUNNING
    assert service.store.assert_action_boundary(
        claim,
        step_id="implement",
        now=NOW,
    ).status is GoalStatus.RUNNING


def test_expired_action_is_uncertain_after_restart_and_not_replayed(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    goal = service.start_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    claim = service.store.claim(
        goal.id,
        runner_id="runner-a",
        expected_revision=goal.revision,
        lease_seconds=5,
        now=NOW,
    )
    checkpoint = service.store.begin_action(
        claim,
        step_id="implement",
        idempotency_key="write-1",
        action_kind="tool",
        action_ref="write_file",
        now=NOW,
    )
    with pytest.raises(GoalLeaseConflictError, match="expired"):
        service.store.finish_action(
            claim,
            checkpoint.id,
            result={"ok": True},
            now=NOW + timedelta(seconds=6),
        )

    restarted = GoalStore(
        tmp_path / "control.sqlite",
        clock=lambda: NOW + timedelta(seconds=6),
    )
    recovered = restarted.recover_expired()

    assert len(recovered) == 1
    assert recovered[0].status is GoalStatus.BLOCKED
    assert recovered[0].steps[0].status is GoalStepStatus.UNCERTAIN
    assert restarted.action_checkpoints(goal.id)[0].state.value == "uncertain"
    assert checkpoint.id == restarted.action_checkpoints(goal.id)[0].id
    with pytest.raises(GoalLeaseConflictError):
        restarted.assert_action_boundary(
            claim,
            step_id="implement",
        )


def test_expired_action_is_recovered_after_pause_during_execution(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    goal = service.start_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    claim = service.store.claim(
        goal.id,
        runner_id="runner",
        expected_revision=goal.revision,
        lease_seconds=5,
        now=NOW,
    )
    checkpoint = service.store.begin_action(
        claim,
        step_id="implement",
        idempotency_key="paused-write",
        action_kind="tool",
        action_ref="write_file",
        now=NOW,
    )
    paused = service.pause(
        goal.id,
        expected_revision=goal.revision,
        actor="owner",
    )

    restarted = GoalStore(
        tmp_path / "control.sqlite",
        clock=lambda: NOW + timedelta(seconds=6),
    )
    recovered = restarted.recover_expired()

    assert paused.status is GoalStatus.PAUSED
    assert recovered[0].status is GoalStatus.BLOCKED
    assert recovered[0].steps[0].status is GoalStepStatus.UNCERTAIN
    assert restarted.action_checkpoints(goal.id)[0].id == checkpoint.id
    assert restarted.action_checkpoints(goal.id)[0].state.value == "uncertain"


def test_action_idempotency_key_prevents_duplicate_or_different_work(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    goal = service.start_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    claim = service.store.claim(
        goal.id,
        runner_id="runner",
        expected_revision=goal.revision,
        now=NOW,
    )
    checkpoint = service.store.begin_action(
        claim,
        step_id="implement",
        idempotency_key="action-1",
        action_kind="tool",
        action_ref="write_file",
        now=NOW,
    )
    service.store.finish_action(
        claim,
        checkpoint.id,
        result={"ok": True},
        now=NOW,
    )

    with pytest.raises(GoalActionConflictError, match="will not be replayed"):
        service.store.begin_action(
            claim,
            step_id="implement",
            idempotency_key="action-1",
            action_kind="tool",
            action_ref="write_file",
            now=NOW,
        )

    with pytest.raises(GoalActionConflictError, match="different work"):
        service.store.begin_action(
            claim,
            step_id="implement",
            idempotency_key="action-1",
            action_kind="tool",
            action_ref="run_cmd",
            now=NOW,
        )


def test_steering_is_append_only_and_cancellation_propagates_owned_resources(
    tmp_path,
) -> None:
    propagated: list[dict] = []
    service = _service(tmp_path, propagate=lambda **values: propagated.append(values))
    goal = _create_goal(service)
    goal = service.steer(
        goal.id,
        expected_revision=goal.revision,
        instruction="Prefer the smallest reversible change.",
        created_by="owner",
    )
    goal = service.link_resource(
        goal.id,
        expected_revision=goal.revision,
        kind="child_task",
        resource_id="child-1",
        actor="owner",
    )
    goal = service.link_resource(
        goal.id,
        expected_revision=goal.revision,
        kind="schedule",
        resource_id="job-1",
        actor="owner",
    )
    goal = service.link_resource(
        goal.id,
        expected_revision=goal.revision,
        kind="process",
        resource_id="process-1",
        actor="owner",
    )
    goal = service.request_cancel(
        goal.id,
        expected_revision=goal.revision,
        actor="owner",
    )

    assert goal.steering[0].instruction.startswith("Prefer")
    assert propagated == [
        {
            "profile_id": "default",
            "goal_id": goal.id,
            "child_task_ids": ("child-1",),
            "schedule_ids": ("job-1",),
            "process_ids": ("process-1",),
        }
    ]


def test_terminal_goal_export_retention_and_revisioned_purge(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    goal = service.start_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    goal = service.add_evidence(
        goal.id,
        expected_revision=goal.revision,
        summary="Verified without exposing sk-secret-value.",
        criterion_ids=("criterion-1",),
        step_id="implement",
        metadata={"api_key": "sk-secret-value"},
    )
    goal = service.complete_step(
        goal.id,
        "implement",
        expected_revision=goal.revision,
        actor="runner",
    )
    goal = service.complete(
        goal.id,
        expected_revision=goal.revision,
        actor="runner",
    )

    destination = tmp_path / "goal-export.json"
    service.store.export(destination, goal_id=goal.id)
    exported = destination.read_text(encoding="utf-8")
    assert goal.id in exported
    assert "goal.completed" in exported
    assert "sk-secret-value" not in exported

    policy = GoalRetentionPolicy()
    assert service.store.retention_candidates(
        policy,
        now=NOW + timedelta(days=89),
    ) == ()
    assert service.store.retention_candidates(
        policy,
        now=NOW + timedelta(days=90),
    ) == (goal,)
    assert service.store.purge_terminal({goal.id: goal.revision}) == (goal.id,)
    with pytest.raises(LookupError):
        service.store.get(goal.id)


def test_cancelling_without_an_active_runner_is_immediately_terminal(tmp_path) -> None:
    service = _service(tmp_path)
    goal = _create_goal(service)

    cancelled = service.request_cancel(
        goal.id,
        expected_revision=goal.revision,
        actor="owner",
    )

    assert cancelled.status is GoalStatus.CANCELLED
    assert cancelled.revision == 2
    assert [item.kind for item in service.store.events(goal.id)][-2:] == [
        "goal.cancellation_requested",
        "goal.cancelled",
    ]
