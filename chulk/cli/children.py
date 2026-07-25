"""Revision-safe operator commands for durable child tasks."""

from __future__ import annotations

from collections.abc import Callable

from chulk.children import (
    ChildDeliveryStatus,
    ChildTaskStatus,
    DelegationService,
)
from chulk.cli.entrypoints import EXIT_OK, EXIT_RUNTIME_ERROR, json_text


def run_child_command(
    command: str,
    *,
    service: DelegationService,
    task_id: str | None = None,
    expected_revision: int | None = None,
    actor: str = "cli",
    reason: str | None = None,
    status: str | None = None,
    goal_id: str | None = None,
    parent_task_id: str | None = None,
    limit: int = 100,
    json_output: bool = False,
    output_func: Callable[[str], None] = print,
    error_func: Callable[[str], None] = print,
) -> int:
    """Run one profile-scoped child-task operator action."""
    try:
        if command == "list":
            tasks = service.store.list(
                status=ChildTaskStatus(status) if status is not None else None,
                goal_id=goal_id,
                parent_task_id=parent_task_id,
                limit=limit,
            )
            return _emit(
                {
                    "ok": True,
                    "profile_id": service.store.profile_id,
                    "tasks": [task.to_dict() for task in tasks],
                },
                _format_list(tasks),
                json_output,
                output_func,
            )
        if command == "inspect":
            task = service.store.get(_required(task_id, "child task id"))
            return _emit(
                {
                    "ok": True,
                    "task": task.to_dict(),
                    "events": [
                        {
                            "id": event.id,
                            "revision": event.revision,
                            "kind": event.kind,
                            "actor": event.actor,
                            "payload": dict(event.payload),
                            "created_at": event.created_at.isoformat(),
                        }
                        for event in service.store.events(task.id)
                    ],
                    "attempts": [
                        dict(attempt)
                        for attempt in service.store.attempts(task.id)
                    ],
                },
                _format_task(task),
                json_output,
                output_func,
            )
        if command == "deliveries":
            deliveries = service.store.list_deliveries(
                status=ChildDeliveryStatus(status)
                if status is not None
                else None,
                limit=limit,
            )
            return _emit(
                {
                    "ok": True,
                    "profile_id": service.store.profile_id,
                    "deliveries": [
                        {
                            "id": item.id,
                            "task_id": item.task_id,
                            "task_revision": item.task_revision,
                            "status": item.status.value,
                            "idempotency_key": item.idempotency_key,
                            "attempts": item.attempts,
                            "error": item.error,
                            "created_at": item.created_at.isoformat(),
                            "updated_at": item.updated_at.isoformat(),
                            "delivered_at": (
                                item.delivered_at.isoformat()
                                if item.delivered_at is not None
                                else None
                            ),
                        }
                        for item in deliveries
                    ],
                },
                _format_deliveries(deliveries),
                json_output,
                output_func,
            )
        if command == "recover":
            tasks = service.store.recover_expired(actor=actor)
            return _emit(
                {
                    "ok": True,
                    "action": "recover",
                    "tasks": [task.to_dict() for task in tasks],
                },
                f"Recovered {len(tasks)} expired child attempt(s).",
                json_output,
                output_func,
            )

        clean_task_id = _required(task_id, "child task id")
        revision = _revision(expected_revision)
        if command == "cancel":
            changed = service.cancel(
                clean_task_id,
                expected_revision=revision,
                actor=actor,
                reason=reason or "Cancellation requested by operator.",
            )
            task = next(
                item for item in changed if item.id == clean_task_id
            )
        elif command == "retry":
            task = service.store.retry(
                clean_task_id,
                expected_revision=revision,
                actor=actor,
            )
        else:
            raise ValueError(f"Unknown child command: {command}")
        return _emit(
            {"ok": True, "action": command, "task": task.to_dict()},
            _format_task(task),
            json_output,
            output_func,
        )
    except (LookupError, RuntimeError, ValueError) as exc:
        payload = {
            "ok": False,
            "status": "child_error",
            "error": str(exc),
        }
        if json_output:
            output_func(json_text(payload))
        else:
            error_func(f"child error: {exc}")
        return EXIT_RUNTIME_ERROR


def _format_list(tasks: tuple) -> str:
    if not tasks:
        return "Child tasks:\n  no tasks"
    lines = ["Child tasks:"]
    for task in tasks:
        lines.append(
            f"  {task.id[:12]}  {task.status.value:<16}  "
            f"r{task.revision}  depth={task.lineage.depth}  "
            f"{task.spec.instruction}"
        )
    return "\n".join(lines)


def _format_task(task) -> str:
    return "\n".join(
        (
            f"Child task {task.id}",
            f"  status       {task.status.value}",
            f"  revision     {task.revision}",
            f"  profile      {task.profile_id}",
            f"  role         {task.spec.role.value}",
            f"  depth        {task.lineage.depth}/{task.spec.max_depth}",
            f"  attempts     {task.attempt_count}",
            f"  parent       {task.lineage.parent_task_id or '-'}",
            f"  goal         {task.goal_id or '-'}",
            f"  cancel       "
            f"{'requested' if task.cancellation_requested else 'no'}",
            f"  instruction  {task.spec.instruction}",
            f"  reason       {task.terminal_reason or '-'}",
        )
    )


def _format_deliveries(deliveries: tuple) -> str:
    if not deliveries:
        return "Child completion deliveries:\n  no deliveries"
    lines = ["Child completion deliveries:"]
    for item in deliveries:
        lines.append(
            f"  {item.id[:12]}  {item.status.value:<10}  "
            f"task={item.task_id[:12]}  attempts={item.attempts}"
        )
    return "\n".join(lines)


def _required(value: str | None, label: str) -> str:
    clean = value.strip() if value is not None else ""
    if not clean:
        raise ValueError(f"{label} is required")
    return clean


def _revision(value: int | None) -> int:
    if value is None:
        raise ValueError("mutating child commands require --revision")
    if value < 0:
        raise ValueError("child task revision cannot be negative")
    return value


def _emit(
    payload: dict,
    text: str,
    json_output: bool,
    output_func: Callable[[str], None],
) -> int:
    output_func(json_text(payload) if json_output else text)
    return EXIT_OK


__all__ = ["run_child_command"]
