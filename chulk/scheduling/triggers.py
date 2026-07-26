"""Adapters from durable completion events into automation trigger envelopes."""

from __future__ import annotations

from chulk.children import ChildTask, ChildTaskStatus
from chulk.goals import Goal, GoalStatus
from chulk.scheduling.models import AutomationRun, TriggerEnvelope, TriggerKind
from chulk.scheduling.store import SQLiteScheduleStore


class AutomationCompletionBridge:
    """Translate completed owned resources without granting trigger authority."""

    def __init__(self, store: SQLiteScheduleStore) -> None:
        self.store = store

    def goal_completed(self, goal: Goal) -> tuple[TriggerEnvelope, ...]:
        if goal.profile_id != self.store.profile_id:
            raise ValueError("goal profile does not match automation profile")
        if goal.status is not GoalStatus.COMPLETED:
            return ()
        return self.store.emit_completion(
            kind=TriggerKind.GOAL_COMPLETION,
            source_resource_id=goal.id,
            source_event_id=f"goal:{goal.id}:revision:{goal.revision}:completed",
            payload={
                "goal_id": goal.id,
                "revision": goal.revision,
                "status": goal.status.value,
            },
            occurred_at=goal.updated_at,
        )

    def child_completed(self, task: ChildTask) -> tuple[TriggerEnvelope, ...]:
        if task.profile_id != self.store.profile_id:
            raise ValueError("child profile does not match automation profile")
        if task.status is not ChildTaskStatus.COMPLETED:
            return ()
        return self.store.emit_completion(
            kind=TriggerKind.CHILD_COMPLETION,
            source_resource_id=task.id,
            source_event_id=f"child:{task.id}:revision:{task.revision}:completed",
            payload={
                "child_task_id": task.id,
                "revision": task.revision,
                "status": task.status.value,
                "goal_id": task.goal_id,
            },
            occurred_at=task.completed_at or task.updated_at,
        )

    def job_completed(
        self,
        run: AutomationRun,
    ) -> tuple[TriggerEnvelope, ...]:
        if run.profile_id != self.store.profile_id:
            raise ValueError("run profile does not match automation profile")
        return self.store.emit_completion(
            kind=TriggerKind.JOB_COMPLETION,
            source_resource_id=run.job_id,
            source_event_id=f"automation-run:{run.id}:{run.status.value}",
            payload={
                "job_id": run.job_id,
                "run_id": run.id,
                "status": run.status.value,
                "reason": run.reason.value,
            },
            occurred_at=run.finished_at or run.updated_at,
        )


__all__ = ["AutomationCompletionBridge"]
