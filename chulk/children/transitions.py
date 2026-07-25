"""Pure state transitions for durable child tasks."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from chulk.children.models import ChildTask, ChildTaskResult, ChildTaskStatus


class InvalidChildTaskTransitionError(ValueError):
    """Raised when a child-task transition violates the state machine."""


def mark_ready(task: ChildTask) -> ChildTask:
    _require_status(task, ChildTaskStatus.PENDING, ChildTaskStatus.BLOCKED)
    if task.cancellation_requested:
        raise InvalidChildTaskTransitionError(
            "cancelled child task cannot become ready"
        )
    return replace(task, status=ChildTaskStatus.READY, terminal_reason=None)


def start_attempt(task: ChildTask) -> ChildTask:
    _require_status(task, ChildTaskStatus.READY)
    if task.cancellation_requested:
        raise InvalidChildTaskTransitionError(
            "child cancellation requested before execution"
        )
    return replace(
        task,
        status=ChildTaskStatus.RUNNING,
        attempt_count=task.attempt_count + 1,
        terminal_reason=None,
    )


def wait_for_children(task: ChildTask) -> ChildTask:
    _require_status(task, ChildTaskStatus.RUNNING)
    if task.spec.role.value != "orchestrator":
        raise InvalidChildTaskTransitionError("leaf child task cannot wait for children")
    return replace(task, status=ChildTaskStatus.WAITING)


def resume_orchestrator(task: ChildTask) -> ChildTask:
    _require_status(task, ChildTaskStatus.WAITING)
    if task.cancellation_requested:
        raise InvalidChildTaskTransitionError(
            "child cancellation requested before resume"
        )
    return replace(task, status=ChildTaskStatus.READY)


def complete_task(
    task: ChildTask,
    result: ChildTaskResult,
    *,
    now: datetime,
) -> ChildTask:
    _require_status(task, ChildTaskStatus.RUNNING)
    if task.cancellation_requested:
        raise InvalidChildTaskTransitionError(
            "cancelled child task cannot report completion"
        )
    return replace(
        task,
        status=ChildTaskStatus.COMPLETED,
        result=result,
        completed_at=now,
        terminal_reason=None,
    )


def fail_task(
    task: ChildTask,
    reason: str,
    *,
    now: datetime,
) -> ChildTask:
    _require_status(task, ChildTaskStatus.RUNNING)
    return _terminal(
        task,
        ChildTaskStatus.FAILED,
        reason,
        now=now,
    )


def exhaust_budget(
    task: ChildTask,
    reason: str,
    *,
    now: datetime,
) -> ChildTask:
    _require_status(task, ChildTaskStatus.RUNNING, ChildTaskStatus.READY)
    return _terminal(
        task,
        ChildTaskStatus.BUDGET_EXHAUSTED,
        reason,
        now=now,
    )


def block_task(task: ChildTask, reason: str) -> ChildTask:
    _require_status(
        task,
        ChildTaskStatus.PENDING,
        ChildTaskStatus.READY,
        ChildTaskStatus.WAITING,
    )
    return replace(
        task,
        status=ChildTaskStatus.BLOCKED,
        terminal_reason=_reason(reason),
    )


def request_cancellation(task: ChildTask) -> ChildTask:
    if task.terminal:
        return task
    return replace(task, cancellation_requested=True)


def cancel_task(
    task: ChildTask,
    reason: str,
    *,
    now: datetime,
) -> ChildTask:
    if task.terminal:
        if task.status is ChildTaskStatus.CANCELLED:
            return task
        raise InvalidChildTaskTransitionError(
            f"terminal child task cannot be cancelled from {task.status.value}"
        )
    return _terminal(
        replace(task, cancellation_requested=True),
        ChildTaskStatus.CANCELLED,
        reason,
        now=now,
    )


def mark_unknown(
    task: ChildTask,
    reason: str,
    *,
    now: datetime,
) -> ChildTask:
    _require_status(task, ChildTaskStatus.RUNNING)
    return _terminal(
        task,
        ChildTaskStatus.UNKNOWN,
        reason,
        now=now,
    )


def retry_task(task: ChildTask) -> ChildTask:
    _require_status(
        task,
        ChildTaskStatus.FAILED,
        ChildTaskStatus.BLOCKED,
        ChildTaskStatus.BUDGET_EXHAUSTED,
        ChildTaskStatus.UNKNOWN,
    )
    return replace(
        task,
        status=ChildTaskStatus.PENDING,
        result=None,
        terminal_reason=None,
        completed_at=None,
        cancellation_requested=False,
    )


def _terminal(
    task: ChildTask,
    status: ChildTaskStatus,
    reason: str,
    *,
    now: datetime,
) -> ChildTask:
    return replace(
        task,
        status=status,
        terminal_reason=_reason(reason),
        completed_at=now,
    )


def _reason(value: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError("child task reason cannot be empty")
    return clean


def _require_status(task: ChildTask, *statuses: ChildTaskStatus) -> None:
    if task.status not in statuses:
        expected = ", ".join(item.value for item in statuses)
        raise InvalidChildTaskTransitionError(
            f"child task {task.id!r} is {task.status.value}; expected {expected}"
        )


__all__ = [
    "InvalidChildTaskTransitionError",
    "block_task",
    "cancel_task",
    "complete_task",
    "exhaust_budget",
    "fail_task",
    "mark_ready",
    "mark_unknown",
    "request_cancellation",
    "resume_orchestrator",
    "retry_task",
    "start_attempt",
    "wait_for_children",
]
