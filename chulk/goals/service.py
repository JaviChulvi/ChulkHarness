"""High-level durable-goal operations and plan promotion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol
from uuid import uuid4

from chulk.goals.models import (
    Goal,
    GoalApproval,
    GoalCriterion,
    GoalEvidence,
    GoalEvent,
    GoalRisk,
    GoalSteering,
    GoalStep,
)
from chulk.goals.runtime import GoalExecutionContext
from chulk.goals.store import GoalStore
from chulk.goals.transitions import (
    approve_goal,
    approve_step,
    block_step,
    cancel_goal,
    complete_goal,
    complete_step,
    fail_goal,
    pause_goal,
    record_evidence,
    request_cancellation,
    resume_goal,
    retry_step,
    skip_step,
    start_goal,
    start_step,
    steer_goal,
)
from chulk.usage import BudgetScope, RunBudget, UsageDimensions


class PlanLike(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


class GoalCancellationPropagator(Protocol):
    """Host callback used to cancel resources owned by a goal."""

    def __call__(
        self,
        *,
        profile_id: str,
        goal_id: str,
        child_task_ids: tuple[str, ...],
        schedule_ids: tuple[str, ...],
        process_ids: tuple[str, ...],
    ) -> None: ...


class GoalEventCallback(Protocol):
    """Receives the committed typed event and matching goal snapshot."""

    def __call__(self, event: GoalEvent, goal: Goal) -> None: ...


class GoalService:
    """Operate a profile-scoped GoalStore through explicit audited transitions."""

    def __init__(
        self,
        store: GoalStore,
        *,
        clock: Callable[[], datetime] | None = None,
        cancellation_propagator: GoalCancellationPropagator | None = None,
        event_callback: GoalEventCallback | None = None,
    ) -> None:
        self.store = store
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.cancellation_propagator = cancellation_propagator
        self.event_callback = event_callback

    def create(
        self,
        *,
        title: str,
        acceptance_criteria: tuple[str | GoalCriterion, ...],
        steps: tuple[GoalStep, ...],
        budget: RunBudget,
        actor: str = "operator",
        goal_id: str | None = None,
        source_conversation_id: str | None = None,
        source_turn_id: str | None = None,
        source_plan: Mapping[str, Any] | None = None,
    ) -> Goal:
        now = self._now()
        criteria = tuple(
            item
            if isinstance(item, GoalCriterion)
            else GoalCriterion(id=f"criterion-{index}", description=item)
            for index, item in enumerate(acceptance_criteria, start=1)
        )
        goal = Goal(
            id=goal_id or uuid4().hex,
            profile_id=self.store.profile_id,
            title=title,
            acceptance_criteria=criteria,
            steps=steps,
            budget=_goal_budget(budget),
            source_conversation_id=source_conversation_id,
            source_turn_id=source_turn_id,
            source_plan=source_plan,
            created_at=now,
            updated_at=now,
        )
        created = self.store.create(goal, actor=actor)
        self._emit(created)
        return created

    def promote_plan(
        self,
        plan: PlanLike | Mapping[str, Any],
        *,
        profile_id: str,
        conversation_id: str,
        turn_id: str,
        budget: RunBudget,
        actor: str = "operator",
        goal_id: str | None = None,
    ) -> Goal:
        """Copy a one-turn plan into an independent durable goal snapshot."""
        if profile_id != self.store.profile_id:
            raise ValueError("plan profile does not match goal store profile")
        raw = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            raise ValueError("plan summary cannot be empty")
        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, (list, tuple)) or not raw_steps:
            raise ValueError("plan requires at least one step")
        criteria: list[GoalCriterion] = []
        steps: list[GoalStep] = []
        seen_criteria: dict[str, str] = {}
        for index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, Mapping):
                raise ValueError("plan steps must be objects")
            step_id = str(raw_step.get("id") or f"step-{index}").strip()
            description = str(
                raw_step.get("description") or raw_step.get("title") or ""
            ).strip()
            raw_criteria = raw_step.get("acceptance_criteria")
            criterion_descriptions = (
                tuple(str(item).strip() for item in raw_criteria)
                if isinstance(raw_criteria, (list, tuple))
                else (description,)
            )
            criterion_ids: list[str] = []
            for criterion_index, criterion_description in enumerate(
                criterion_descriptions,
                start=1,
            ):
                clean = criterion_description or description
                criterion_id = f"{step_id}-criterion-{criterion_index}"
                if criterion_id not in seen_criteria:
                    seen_criteria[criterion_id] = clean
                    criteria.append(
                        GoalCriterion(id=criterion_id, description=clean)
                    )
                criterion_ids.append(criterion_id)
            steps.append(
                GoalStep(
                    id=step_id,
                    title=str(raw_step.get("title") or description),
                    description=description,
                    acceptance_criterion_ids=tuple(criterion_ids),
                    depends_on=tuple(
                        str(item)
                        for item in raw_step.get("depends_on", ())
                    ),
                    risk=GoalRisk(
                        str(raw_step.get("risk", GoalRisk.LOW.value))
                    ),
                    expected_tools=tuple(
                        str(item)
                        for item in raw_step.get("expected_tools", ())
                    ),
                    max_attempts=max(
                        1,
                        int(raw_step.get("retry_limit", 0)) + 1,
                    ),
                )
            )
        return self.create(
            title=summary,
            acceptance_criteria=tuple(criteria),
            steps=tuple(steps),
            budget=budget,
            actor=actor,
            goal_id=goal_id,
            source_conversation_id=conversation_id,
            source_turn_id=turn_id,
            source_plan=raw,
        )

    def approve(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        approved_by: str,
        reason: str | None = None,
    ) -> Goal:
        approval = GoalApproval(
            id=uuid4().hex,
            approved_by=approved_by,
            reason=reason,
            created_at=self._now(),
        )
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.approved",
            approved_by,
            lambda goal: approve_goal(goal, approval, now=self._now()),
            {"approval_id": approval.id},
        )

    def run(self, goal_id: str, *, expected_revision: int, actor: str) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.running",
            actor,
            lambda goal: start_goal(goal, now=self._now()),
        )

    def pause(self, goal_id: str, *, expected_revision: int, actor: str) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.paused",
            actor,
            lambda goal: pause_goal(goal, now=self._now()),
        )

    def resume(self, goal_id: str, *, expected_revision: int, actor: str) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.resumed",
            actor,
            lambda goal: resume_goal(goal, now=self._now()),
        )

    def steer(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        instruction: str,
        created_by: str,
    ) -> Goal:
        steering = GoalSteering(
            id=uuid4().hex,
            instruction=instruction,
            created_by=created_by,
            created_at=self._now(),
        )
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.steered",
            created_by,
            lambda goal: steer_goal(goal, steering),
            {"steering_id": steering.id, "instruction": steering.instruction},
        )

    def approve_step(
        self,
        goal_id: str,
        step_id: str,
        *,
        expected_revision: int,
        approved_by: str,
        reason: str | None = None,
    ) -> Goal:
        approval = GoalApproval(
            id=uuid4().hex,
            approved_by=approved_by,
            scope="step",
            step_id=step_id,
            reason=reason,
            created_at=self._now(),
        )
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.step_approved",
            approved_by,
            lambda goal: approve_step(goal, approval),
            {"step_id": step_id, "approval_id": approval.id},
        )

    def start_step(
        self,
        goal_id: str,
        step_id: str,
        *,
        expected_revision: int,
        actor: str,
    ) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.step_started",
            actor,
            lambda goal: start_step(goal, step_id, now=self._now()),
            {"step_id": step_id},
        )

    def add_evidence(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        summary: str,
        criterion_ids: tuple[str, ...],
        step_id: str | None = None,
        kind: str = "observation",
        reference: str | None = None,
        recorded_by: str = "runner",
        metadata: Mapping[str, Any] | None = None,
    ) -> Goal:
        evidence = GoalEvidence(
            id=uuid4().hex,
            summary=summary,
            criterion_ids=criterion_ids,
            step_id=step_id,
            kind=kind,
            reference=reference,
            recorded_by=recorded_by,
            recorded_at=self._now(),
            metadata=metadata or {},
        )
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.evidence_recorded",
            recorded_by,
            lambda goal: record_evidence(goal, evidence),
            {
                "evidence_id": evidence.id,
                "step_id": step_id,
                "criterion_ids": list(criterion_ids),
            },
        )

    def complete_step(
        self,
        goal_id: str,
        step_id: str,
        *,
        expected_revision: int,
        actor: str,
    ) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.step_completed",
            actor,
            lambda goal: complete_step(goal, step_id, now=self._now()),
            {"step_id": step_id},
        )

    def block_step(
        self,
        goal_id: str,
        step_id: str,
        *,
        expected_revision: int,
        reason: str,
        actor: str,
    ) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.step_blocked",
            actor,
            lambda goal: block_step(goal, step_id, reason),
            {"step_id": step_id, "reason": reason},
        )

    def retry_step(
        self,
        goal_id: str,
        step_id: str,
        *,
        expected_revision: int,
        actor: str,
    ) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.step_retried",
            actor,
            lambda goal: retry_step(goal, step_id),
            {"step_id": step_id},
        )

    def skip_step(
        self,
        goal_id: str,
        step_id: str,
        *,
        expected_revision: int,
        reason: str,
        approved_by: str,
    ) -> Goal:
        approval = GoalApproval(
            id=uuid4().hex,
            approved_by=approved_by,
            scope="skip",
            step_id=step_id,
            reason=reason,
            created_at=self._now(),
        )
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.step_skipped",
            approved_by,
            lambda goal: skip_step(
                goal,
                step_id,
                reason=reason,
                approval=approval,
                now=self._now(),
            ),
            {"step_id": step_id, "reason": reason, "approval_id": approval.id},
        )

    def request_cancel(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        actor: str,
    ) -> Goal:
        goal = self._mutate(
            goal_id,
            expected_revision,
            "goal.cancellation_requested",
            actor,
            request_cancellation,
        )
        if self.cancellation_propagator is not None:
            self.cancellation_propagator(
                profile_id=goal.profile_id,
                goal_id=goal.id,
                child_task_ids=goal.child_task_ids,
                schedule_ids=goal.schedule_ids,
                process_ids=goal.process_ids,
            )
        if self.store.has_active_claim(goal.id):
            return goal
        return self.cancel(
            goal.id,
            expected_revision=goal.revision,
            actor=actor,
        )

    def cancel(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        actor: str,
    ) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.cancelled",
            actor,
            lambda goal: cancel_goal(goal, now=self._now()),
        )

    def complete(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        actor: str,
    ) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.completed",
            actor,
            lambda goal: complete_goal(goal, now=self._now()),
        )

    def fail(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        reason: str,
        actor: str,
    ) -> Goal:
        return self._mutate(
            goal_id,
            expected_revision,
            "goal.failed",
            actor,
            lambda goal: fail_goal(goal, reason, now=self._now()),
            {"reason": reason},
        )

    def link_resource(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        kind: str,
        resource_id: str,
        actor: str,
    ) -> Goal:
        """Link one trace, task, schedule, process, or artifact idempotently."""
        fields = {
            "child_task": "child_task_ids",
            "schedule": "schedule_ids",
            "process": "process_ids",
            "trace": "trace_ids",
            "artifact": "artifact_refs",
        }
        field_name = fields.get(kind)
        if field_name is None:
            raise ValueError(
                "goal resource kind must be child_task, schedule, process, "
                "trace, or artifact"
            )
        clean_id = resource_id.strip()
        if not clean_id:
            raise ValueError("goal resource id cannot be empty")

        def link(goal: Goal) -> Goal:
            values = getattr(goal, field_name)
            if clean_id in values:
                return goal
            if field_name == "child_task_ids":
                return replace(goal, child_task_ids=(*goal.child_task_ids, clean_id))
            if field_name == "schedule_ids":
                return replace(goal, schedule_ids=(*goal.schedule_ids, clean_id))
            if field_name == "process_ids":
                return replace(goal, process_ids=(*goal.process_ids, clean_id))
            if field_name == "trace_ids":
                return replace(goal, trace_ids=(*goal.trace_ids, clean_id))
            return replace(goal, artifact_refs=(*goal.artifact_refs, clean_id))

        return self._mutate(
            goal_id,
            expected_revision,
            "goal.resource_linked",
            actor,
            link,
            {"kind": kind, "resource_id": clean_id},
        )

    def runtime_options(
        self,
        goal_id: str,
        *,
        conversation_id: str | None = None,
        turn_id: str | None = None,
        channel: str | None = None,
    ) -> dict[str, object]:
        """Return create_agent kwargs that enforce this goal's shared budget."""
        goal = self.store.get(goal_id)
        return {
            "run_budget": goal.budget,
            "usage_dimensions": UsageDimensions(
                profile_id=goal.profile_id,
                channel=channel,
                conversation_id=conversation_id,
                turn_id=turn_id,
                goal_id=goal.id,
            ),
            "runtime_metadata": {
                "goal_id": goal.id,
                "goal_revision": goal.revision,
            },
        }

    def claim_execution(
        self,
        goal_id: str,
        step_id: str,
        *,
        expected_revision: int,
        runner_id: str,
        lease_seconds: int = 120,
    ) -> GoalExecutionContext:
        """Claim one running step for boundary-enforced agent execution."""
        goal = self.store.get(goal_id)
        if goal.step(step_id).status.value != "running":
            raise ValueError("goal execution requires a running step")
        claim = self.store.claim(
            goal_id,
            runner_id=runner_id,
            expected_revision=expected_revision,
            lease_seconds=lease_seconds,
        )
        return GoalExecutionContext(
            store=self.store,
            claim=claim,
            step_id=step_id,
        )

    def _mutate(
        self,
        goal_id: str,
        expected_revision: int,
        kind: str,
        actor: str,
        mutation,
        payload: Mapping[str, Any] | None = None,
    ) -> Goal:
        goal = self.store.mutate(
            goal_id,
            expected_revision=expected_revision,
            kind=kind,
            actor=actor,
            mutation=mutation,
            payload=payload,
        )
        if goal.revision != expected_revision:
            self._emit(goal)
        return goal

    def _emit(self, goal: Goal) -> None:
        if self.event_callback is None:
            return
        events = self.store.events(
            goal.id,
            after_revision=goal.revision - 1,
        )
        if events:
            self.event_callback(events[-1], goal)

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("goal clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


def _goal_budget(budget: RunBudget) -> RunBudget:
    return replace(budget, scope=BudgetScope.GOAL)


__all__ = [
    "GoalCancellationPropagator",
    "GoalEventCallback",
    "GoalService",
    "PlanLike",
]
