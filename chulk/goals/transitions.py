"""Pure state transitions for durable goals and goal steps."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Callable

from chulk.goals.models import (
    Goal,
    GoalApproval,
    GoalEvidence,
    GoalStatus,
    GoalSteering,
    GoalStep,
    GoalStepStatus,
)


class InvalidGoalTransitionError(ValueError):
    """Raised when an operation is not valid for the current goal snapshot."""


def approve_goal(goal: Goal, approval: GoalApproval, *, now: datetime) -> Goal:
    _require_status(goal, GoalStatus.DRAFT)
    if approval.scope != "goal":
        raise InvalidGoalTransitionError("goal approval requires goal scope")
    return replace(
        goal,
        status=GoalStatus.APPROVED,
        approvals=(*goal.approvals, approval),
        approved_at=now,
        cancellation_requested=False,
        last_error=None,
    )


def start_goal(goal: Goal, *, now: datetime) -> Goal:
    _require_status(goal, GoalStatus.APPROVED, GoalStatus.PAUSED, GoalStatus.BLOCKED)
    if goal.cancellation_requested:
        raise InvalidGoalTransitionError("cancelled goal cannot start")
    steps = _refresh_ready_steps(goal.steps)
    if not any(
        item.status in {GoalStepStatus.READY, GoalStepStatus.RUNNING}
        for item in steps
    ):
        raise InvalidGoalTransitionError("goal has no ready step")
    return replace(
        goal,
        status=GoalStatus.RUNNING,
        steps=steps,
        started_at=goal.started_at or now,
        last_error=None,
    )


def pause_goal(goal: Goal, *, now: datetime) -> Goal:
    del now
    _require_status(goal, GoalStatus.RUNNING, GoalStatus.BLOCKED)
    return replace(goal, status=GoalStatus.PAUSED)


def resume_goal(goal: Goal, *, now: datetime) -> Goal:
    _require_status(goal, GoalStatus.PAUSED, GoalStatus.BLOCKED)
    return start_goal(goal, now=now)


def steer_goal(goal: Goal, steering: GoalSteering) -> Goal:
    if goal.terminal:
        raise InvalidGoalTransitionError("terminal goal cannot be steered")
    return replace(goal, steering=(*goal.steering, steering))


def request_cancellation(goal: Goal) -> Goal:
    if goal.terminal:
        if goal.status is GoalStatus.CANCELLED:
            return goal
        raise InvalidGoalTransitionError("terminal goal cannot be cancelled")
    return replace(goal, cancellation_requested=True)


def cancel_goal(goal: Goal, *, now: datetime) -> Goal:
    if goal.status is GoalStatus.COMPLETED:
        raise InvalidGoalTransitionError("completed goal cannot be cancelled")
    if goal.status is GoalStatus.CANCELLED:
        return goal
    steps = tuple(
        replace(
            item,
            status=(
                GoalStepStatus.UNCERTAIN
                if item.status is GoalStepStatus.RUNNING
                else item.status
            ),
            blocked_reason=(
                "Cancellation interrupted an active action; outcome is uncertain."
                if item.status is GoalStepStatus.RUNNING
                else item.blocked_reason
            ),
        )
        for item in goal.steps
    )
    return replace(
        goal,
        status=GoalStatus.CANCELLED,
        steps=steps,
        cancellation_requested=True,
        completed_at=now,
    )


def fail_goal(goal: Goal, reason: str, *, now: datetime) -> Goal:
    if goal.terminal:
        raise InvalidGoalTransitionError("terminal goal cannot fail")
    clean_reason = reason.strip()
    if not clean_reason:
        raise ValueError("failure reason cannot be empty")
    return replace(
        goal,
        status=GoalStatus.FAILED,
        last_error=clean_reason,
        completed_at=now,
    )


def start_step(goal: Goal, step_id: str, *, now: datetime) -> Goal:
    _require_status(goal, GoalStatus.RUNNING)
    if goal.cancellation_requested:
        raise InvalidGoalTransitionError("cancellation requested before next action")
    if any(item.status is GoalStepStatus.RUNNING for item in goal.steps):
        raise InvalidGoalTransitionError("another goal step is already running")
    steps = _refresh_ready_steps(goal.steps)
    step = _step(steps, step_id)
    if step.status is not GoalStepStatus.READY:
        raise InvalidGoalTransitionError(
            f"goal step {step_id!r} is not ready"
        )
    if step.risk.value == "high" and not _step_is_approved(goal, step_id):
        raise InvalidGoalTransitionError(
            f"high-risk goal step {step_id!r} requires selected-step approval"
        )
    updated = replace(
        step,
        status=GoalStepStatus.RUNNING,
        attempt=step.attempt + 1,
        started_at=now,
        completed_at=None,
        blocked_reason=None,
    )
    return replace(goal, steps=_replace_step(steps, updated))


def record_evidence(goal: Goal, evidence: GoalEvidence) -> Goal:
    if goal.terminal:
        raise InvalidGoalTransitionError("terminal goal cannot accept evidence")
    if any(item.id == evidence.id for item in goal.evidence):
        return goal
    if evidence.step_id is None:
        return replace(goal, evidence=(*goal.evidence, evidence))
    step = _step(goal.steps, evidence.step_id)
    updated = replace(step, evidence_ids=(*step.evidence_ids, evidence.id))
    return replace(
        goal,
        evidence=(*goal.evidence, evidence),
        steps=_replace_step(goal.steps, updated),
    )


def complete_step(goal: Goal, step_id: str, *, now: datetime) -> Goal:
    _require_status(goal, GoalStatus.RUNNING)
    step = _step(goal.steps, step_id)
    if step.status is not GoalStepStatus.RUNNING:
        raise InvalidGoalTransitionError(
            f"goal step {step_id!r} is not running"
        )
    missing = set(step.acceptance_criterion_ids) - goal.evidenced_criterion_ids
    if missing:
        raise InvalidGoalTransitionError(
            f"goal step {step_id!r} lacks evidence for: {', '.join(sorted(missing))}"
        )
    updated = replace(
        step,
        status=GoalStepStatus.COMPLETED,
        completed_at=now,
        blocked_reason=None,
    )
    steps = _refresh_ready_steps(_replace_step(goal.steps, updated))
    return replace(goal, steps=steps)


def block_step(goal: Goal, step_id: str, reason: str) -> Goal:
    _require_status(goal, GoalStatus.RUNNING)
    clean_reason = reason.strip()
    if not clean_reason:
        raise ValueError("blocked reason cannot be empty")
    step = _step(goal.steps, step_id)
    if step.status not in {GoalStepStatus.RUNNING, GoalStepStatus.READY}:
        raise InvalidGoalTransitionError(
            f"goal step {step_id!r} cannot be blocked from {step.status.value}"
        )
    updated = replace(
        step,
        status=GoalStepStatus.BLOCKED,
        blocked_reason=clean_reason,
    )
    return replace(
        goal,
        status=GoalStatus.BLOCKED,
        steps=_replace_step(goal.steps, updated),
        last_error=clean_reason,
    )


def retry_step(goal: Goal, step_id: str) -> Goal:
    _require_status(goal, GoalStatus.BLOCKED, GoalStatus.RUNNING)
    step = _step(goal.steps, step_id)
    if step.status not in {
        GoalStepStatus.BLOCKED,
        GoalStepStatus.FAILED,
        GoalStepStatus.UNCERTAIN,
    }:
        raise InvalidGoalTransitionError(
            f"goal step {step_id!r} is not retryable"
        )
    if step.attempt >= step.max_attempts:
        raise InvalidGoalTransitionError(
            f"goal step {step_id!r} exhausted its attempt limit"
        )
    updated = replace(
        step,
        status=GoalStepStatus.PENDING,
        blocked_reason=None,
        completed_at=None,
    )
    steps = _refresh_ready_steps(_replace_step(goal.steps, updated))
    return replace(
        goal,
        status=GoalStatus.RUNNING,
        steps=steps,
        last_error=None,
    )


def skip_step(
    goal: Goal,
    step_id: str,
    *,
    reason: str,
    approval: GoalApproval,
    now: datetime,
) -> Goal:
    if goal.terminal:
        raise InvalidGoalTransitionError("terminal goal cannot skip a step")
    if approval.scope != "skip" or approval.step_id != step_id:
        raise InvalidGoalTransitionError("skip requires matching operator approval")
    clean_reason = reason.strip()
    if not clean_reason:
        raise ValueError("skip reason cannot be empty")
    step = _step(goal.steps, step_id)
    if step.status in {
        GoalStepStatus.COMPLETED,
        GoalStepStatus.SKIPPED,
        GoalStepStatus.RUNNING,
    }:
        raise InvalidGoalTransitionError(
            f"goal step {step_id!r} cannot be skipped from {step.status.value}"
        )
    updated = replace(
        step,
        status=GoalStepStatus.SKIPPED,
        skip_reason=clean_reason,
        completed_at=now,
    )
    steps = _refresh_ready_steps(_replace_step(goal.steps, updated))
    return replace(
        goal,
        steps=steps,
        approvals=(*goal.approvals, approval),
        status=(
            GoalStatus.RUNNING
            if goal.status in {GoalStatus.RUNNING, GoalStatus.BLOCKED}
            else goal.status
        ),
        last_error=None,
    )


def approve_step(goal: Goal, approval: GoalApproval) -> Goal:
    if approval.scope != "step" or approval.step_id is None:
        raise InvalidGoalTransitionError("step approval requires a step id")
    if goal.terminal:
        raise InvalidGoalTransitionError("terminal goal cannot approve a step")
    _step(goal.steps, approval.step_id)
    return replace(goal, approvals=(*goal.approvals, approval))


def mark_step_uncertain(goal: Goal, step_id: str, reason: str) -> Goal:
    step = _step(goal.steps, step_id)
    if step.status is not GoalStepStatus.RUNNING:
        return goal
    updated = replace(
        step,
        status=GoalStepStatus.UNCERTAIN,
        blocked_reason=reason.strip() or "Action outcome is uncertain after restart.",
    )
    return replace(
        goal,
        status=GoalStatus.BLOCKED,
        steps=_replace_step(goal.steps, updated),
        last_error=updated.blocked_reason,
    )


def complete_goal(goal: Goal, *, now: datetime) -> Goal:
    _require_status(goal, GoalStatus.RUNNING, GoalStatus.BLOCKED)
    unfinished = [
        item.id
        for item in goal.steps
        if item.status not in {
            GoalStepStatus.COMPLETED,
            GoalStepStatus.SKIPPED,
        }
    ]
    if unfinished:
        raise InvalidGoalTransitionError(
            f"goal has unfinished steps: {', '.join(unfinished)}"
        )
    if goal.missing_criterion_ids:
        raise InvalidGoalTransitionError(
            "goal lacks evidence for acceptance criteria: "
            f"{', '.join(goal.missing_criterion_ids)}"
        )
    return replace(
        goal,
        status=GoalStatus.COMPLETED,
        completed_at=now,
        cancellation_requested=False,
        last_error=None,
    )


def _refresh_ready_steps(steps: tuple[GoalStep, ...]) -> tuple[GoalStep, ...]:
    satisfied = {
        item.id
        for item in steps
        if item.status in {GoalStepStatus.COMPLETED, GoalStepStatus.SKIPPED}
    }
    return tuple(
        replace(
            item,
            status=(
                GoalStepStatus.READY
                if all(dependency in satisfied for dependency in item.depends_on)
                else GoalStepStatus.PENDING
            ),
        )
        if item.status in {GoalStepStatus.PENDING, GoalStepStatus.READY}
        else item
        for item in steps
    )


def _step_is_approved(goal: Goal, step_id: str) -> bool:
    return any(
        item.scope == "step" and item.step_id == step_id
        for item in goal.approvals
    )


def _step(steps: tuple[GoalStep, ...], step_id: str) -> GoalStep:
    for item in steps:
        if item.id == step_id:
            return item
    raise KeyError(step_id)


def _replace_step(
    steps: tuple[GoalStep, ...],
    updated: GoalStep,
) -> tuple[GoalStep, ...]:
    return tuple(updated if item.id == updated.id else item for item in steps)


def _require_status(goal: Goal, *statuses: GoalStatus) -> None:
    if goal.status not in statuses:
        allowed = ", ".join(item.value for item in statuses)
        raise InvalidGoalTransitionError(
            f"goal {goal.id!r} is {goal.status.value}; expected {allowed}"
        )


GoalMutation = Callable[[Goal], Goal]


__all__ = [
    "GoalMutation",
    "InvalidGoalTransitionError",
    "approve_goal",
    "approve_step",
    "block_step",
    "cancel_goal",
    "complete_goal",
    "complete_step",
    "fail_goal",
    "mark_step_uncertain",
    "pause_goal",
    "record_evidence",
    "request_cancellation",
    "resume_goal",
    "retry_step",
    "skip_step",
    "start_goal",
    "start_step",
    "steer_goal",
]
