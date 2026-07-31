"""Owner-scoped CLI for durable automation definitions and run history."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from chulk.cli.entrypoints import EXIT_OK, EXIT_RUNTIME_ERROR, json_text
from chulk.scheduling import (
    AutomationJobStatus,
    RecurrenceKind,
    RecurrenceSpec,
    SQLiteScheduleStore,
    TriggerKind,
)


def run_automation_command(
    command: str,
    *,
    store: SQLiteScheduleStore,
    job_id: str | None = None,
    expected_revision: int | None = None,
    idempotency_key: str | None = None,
    status: str | None = None,
    limit: int = 100,
    actor: str = "cli",
    prompt: str | None = None,
    run_at: str | None = None,
    timezone_name: str = "UTC",
    interval_seconds: int | None = None,
    cron: str | None = None,
    rrule: str | None = None,
    trigger_kind: str | None = None,
    source_resource_id: str | None = None,
    json_output: bool = False,
    output_func: Callable[[str], None] = print,
    error_func: Callable[[str], None] = print,
) -> int:
    """Execute one profile-owned automation operation."""
    try:
        if command == "list":
            jobs = store.list(
                status=AutomationJobStatus(status) if status else None,
                limit=limit,
                include_terminal=status is None,
            )
            return _emit(
                {
                    "ok": True,
                    "profile_id": store.profile_id,
                    "jobs": [item.to_dict() for item in jobs],
                },
                _format_list(jobs),
                json_output,
                output_func,
            )
        if command == "recover":
            runs = store.recover_expired(actor=actor)
            return _emit(
                {"ok": True, "recovered": [item.to_dict() for item in runs]},
                f"Recovered {len(runs)} expired automation run(s).",
                json_output,
                output_func,
            )
        clean_id = _required(job_id, "job id")
        if command == "inspect":
            job = store.get(clean_id)
            payload = {
                "ok": True,
                "job": job.to_dict(),
                "triggers": [item.to_dict() for item in store.triggers(clean_id)],
            }
            return _emit(payload, _format_job(job), json_output, output_func)
        if command == "history":
            runs = store.runs(clean_id, limit=limit)
            return _emit(
                {"ok": True, "runs": [item.to_dict() for item in runs]},
                _format_history(runs),
                json_output,
                output_func,
            )
        if command == "triggers":
            triggers = store.triggers(clean_id)
            return _emit(
                {"ok": True, "triggers": [item.to_dict() for item in triggers]},
                "\n".join(
                    f"{item.id} {item.kind.value} enabled={item.enabled}"
                    for item in triggers
                )
                or "No automation triggers.",
                json_output,
                output_func,
            )
        if command == "webhook":
            trigger, token = store.create_webhook_trigger(clean_id)
            return _emit(
                {
                    "ok": True,
                    "trigger": trigger.to_dict(),
                    "credential": token,
                },
                (
                    f"Created webhook trigger {trigger.id}.\n"
                    f"Credential (shown once): {token}"
                ),
                json_output,
                output_func,
            )
        if command == "completion-trigger":
            trigger = store.create_completion_trigger(
                clean_id,
                kind=TriggerKind(_required(trigger_kind, "completion trigger kind")),
                source_resource_id=_required(
                    source_resource_id,
                    "source resource id",
                ),
            )
            return _emit(
                {"ok": True, "trigger": trigger.to_dict()},
                f"Created {trigger.kind.value} trigger {trigger.id}.",
                json_output,
                output_func,
            )
        revision = _revision(expected_revision)
        key = idempotency_key or f"cli:{command}:{uuid4().hex}"
        if command == "pause":
            job = store.pause(
                clean_id,
                expected_revision=revision,
                idempotency_key=key,
                actor=actor,
            )
        elif command == "resume":
            job = store.resume(
                clean_id,
                expected_revision=revision,
                idempotency_key=key,
                actor=actor,
            )
        elif command == "approve":
            job = store.approve(
                clean_id,
                expected_revision=revision,
                idempotency_key=key,
                actor=actor,
            )
        elif command == "run-now":
            job = store.run_now(
                clean_id,
                expected_revision=revision,
                idempotency_key=key,
                actor=actor,
            )
        elif command == "cancel":
            if not store.cancel(
                clean_id,
                expected_revision=revision,
                idempotency_key=key,
                actor=actor,
            ):
                raise ValueError("automation is already terminal")
            job = store.get(clean_id)
        elif command == "update":
            current = store.get(clean_id)
            selected_run_at = (
                _timestamp(run_at, "run_at")
                if run_at is not None
                else current.next_run_at
            )
            recurrence = (
                _recurrence(
                    timezone_name=timezone_name,
                    interval_seconds=interval_seconds,
                    cron=cron,
                    rrule=rrule,
                )
                if any(value is not None for value in (interval_seconds, cron, rrule))
                else None
            )
            job = store.update(
                clean_id,
                expected_revision=revision,
                idempotency_key=key,
                prompt=prompt,
                recurrence=recurrence,
                next_run_at=selected_run_at,
                actor=actor,
            )
        else:
            raise ValueError(f"Unknown automation command: {command}")
        return _emit(
            {
                "ok": True,
                "action": command,
                "idempotency_key": key,
                "job": job.to_dict(),
            },
            _format_job(job),
            json_output,
            output_func,
        )
    except (LookupError, PermissionError, RuntimeError, ValueError) as exc:
        payload = {"ok": False, "status": "automation_error", "error": str(exc)}
        if json_output:
            output_func(json_text(payload))
        else:
            error_func(f"automation error: {exc}")
        return EXIT_RUNTIME_ERROR


def _recurrence(
    *,
    timezone_name: str,
    interval_seconds: int | None,
    cron: str | None,
    rrule: str | None,
) -> RecurrenceSpec:
    values = sum(value is not None for value in (interval_seconds, cron, rrule))
    if values != 1:
        raise ValueError("update recurrence requires exactly one recurrence option")
    return RecurrenceSpec(
        kind=(
            RecurrenceKind.INTERVAL
            if interval_seconds is not None
            else RecurrenceKind.CRON
            if cron is not None
            else RecurrenceKind.RRULE
        ),
        timezone_name=timezone_name,
        interval_seconds=interval_seconds,
        cron=cron,
        rrule=rrule,
    )


def _format_list(jobs: tuple) -> str:
    if not jobs:
        return "No automation jobs."
    return "\n".join(
        f"{item.id} {item.status.value} {item.recurrence.kind.value} "
        f"next={item.next_run_at.isoformat()} runs={item.run_count}"
        for item in jobs
    )


def _format_job(job) -> str:
    return (
        f"Automation {job.id}\n"
        f"  status: {job.status.value}\n"
        f"  revision: {job.revision}\n"
        f"  profile: {job.profile_id}\n"
        f"  destination: {job.adapter}/{job.target.account_id}/{job.destination_id}\n"
        f"  recurrence: {job.recurrence.kind.value}\n"
        f"  next run: {job.next_run_at.isoformat()}\n"
        f"  run count: {job.run_count}"
    )


def _format_history(runs: tuple) -> str:
    if not runs:
        return "No automation runs."
    return "\n".join(
        f"{item.id} {item.status.value} {item.reason.value} "
        f"attempt={item.attempt} occurrence={item.occurrence_at.isoformat()}"
        for item in runs
    )


def _timestamp(value: str | None, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_required(value, label))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _revision(value: int | None) -> int:
    if value is None or value < 0:
        raise ValueError(
            "mutating automation commands require a non-negative --revision"
        )
    return value


def _required(value: str | None, label: str) -> str:
    clean = value.strip() if value is not None else ""
    if not clean:
        raise ValueError(f"{label} is required")
    return clean


def _emit(
    payload: dict,
    text: str,
    json_output: bool,
    output_func: Callable[[str], None],
) -> int:
    output_func(json_text(payload) if json_output else text)
    return EXIT_OK


__all__ = ["run_automation_command"]
