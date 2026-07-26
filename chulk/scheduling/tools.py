"""Destination-scoped tools for creating and managing scheduled jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from chulk.scheduling.models import (
    MisfirePolicy,
    RecurrenceKind,
    RecurrenceSpec,
    ScheduledJob,
)
from chulk.scheduling.store import SQLiteScheduleStore
from chulk.tools import Tool, ToolPermissionLevel, tool


def scheduled_job_tools(
    store: SQLiteScheduleStore,
    *,
    adapter: str,
    destination_id: str,
    timezone_name: str,
) -> list[Tool]:
    """Bind scheduling operations to one authenticated destination."""
    local_tz = ZoneInfo(timezone_name)

    def schedule_task(
        prompt: str,
        run_at: str,
        interval_seconds: int | None = None,
        cron: str | None = None,
        rrule: str | None = None,
        misfire_policy: str = "run_once",
        max_catch_up: int = 1,
        jitter_seconds: int = 0,
        max_runs: int | None = None,
        requires_approval: bool = False,
    ) -> str:
        when = _parse_run_at(run_at, local_tz)
        if when <= datetime.now(timezone.utc):
            raise ValueError("run_at must be in the future")
        if interval_seconds is not None and interval_seconds < 60:
            raise ValueError("interval_seconds must be at least 60")
        recurrence_fields = sum(
            value is not None for value in (interval_seconds, cron, rrule)
        )
        if recurrence_fields > 1:
            raise ValueError("choose only one of interval_seconds, cron, or rrule")
        kind = (
            RecurrenceKind.INTERVAL
            if interval_seconds is not None
            else RecurrenceKind.CRON
            if cron is not None
            else RecurrenceKind.RRULE
            if rrule is not None
            else RecurrenceKind.ONCE
        )
        recurrence = RecurrenceSpec(
            kind=kind,
            timezone_name=timezone_name,
            interval_seconds=interval_seconds,
            cron=cron,
            rrule=rrule,
            misfire_policy=MisfirePolicy(misfire_policy),
            max_catch_up=max_catch_up,
            jitter_seconds=jitter_seconds,
        )
        job = store.create(
            adapter=adapter,
            destination_id=destination_id,
            prompt=prompt,
            next_run_at=when,
            recurrence=recurrence,
            max_runs=max_runs,
            requires_approval=requires_approval,
        )
        recurrence_text = (
            f", {job.recurrence.kind.value} recurrence"
            if job.recurrence.kind is not RecurrenceKind.ONCE
            else ""
        )
        approval = ", pending approval" if requires_approval else ""
        return (
            f"Scheduled {job.id[:8]} for {job.next_run_at.isoformat()}"
            f"{recurrence_text}{approval}."
        )

    def list_scheduled_tasks() -> str:
        return format_jobs(
            store.list(adapter=adapter, destination_id=destination_id),
            timezone_name=timezone_name,
        )

    def current_time() -> str:
        return (
            f"Current local time: {datetime.now(local_tz).isoformat()}\n"
            f"Timezone: {timezone_name}"
        )

    def cancel_scheduled_task(job_id: str) -> str:
        clean_id = _resolve_job_id(
            store,
            adapter=adapter,
            destination_id=destination_id,
            value=job_id,
        )
        if not store.cancel(clean_id, adapter=adapter, destination_id=destination_id):
            raise ValueError("Scheduled task is no longer active")
        return f"Cancelled scheduled task {clean_id[:8]}."

    return [
        tool(
            schedule_task,
            name="schedule_task",
            description=(
                "Schedule a prompt for later delivery. run_at must be an ISO-8601 "
                f"date/time in {timezone_name} unless it includes an offset; "
                "choose at most one recurrence: interval_seconds (minimum 60), "
                "a five-field cron expression, or an RFC 5545 RRULE."
            ),
            permission_level=ToolPermissionLevel.WRITE,
            idempotent=False,
        ),
        tool(
            current_time,
            name="current_time",
            description=(
                "Return the current date, time, and configured timezone before resolving "
                "relative schedule requests such as tomorrow or next Monday."
            ),
        ),
        tool(
            list_scheduled_tasks,
            name="list_scheduled_tasks",
            description="List active scheduled tasks for this chat.",
        ),
        tool(
            cancel_scheduled_task,
            name="cancel_scheduled_task",
            description=(
                "Cancel a scheduled task by its full id or displayed 8-character prefix."
            ),
            permission_level=ToolPermissionLevel.WRITE,
            idempotent=True,
        ),
    ]


def format_jobs(
    jobs: tuple[ScheduledJob, ...],
    *,
    timezone_name: str = "UTC",
) -> str:
    if not jobs:
        return "No active scheduled tasks."
    lines = ["Active scheduled tasks:"]
    display_tz = ZoneInfo(timezone_name)
    for job in jobs:
        recurrence = ""
        if job.recurrence.kind is RecurrenceKind.INTERVAL:
            recurrence = f" every {job.interval_seconds}s"
        elif job.recurrence.kind is RecurrenceKind.CRON:
            recurrence = f" cron {job.recurrence.cron}"
        elif job.recurrence.kind is RecurrenceKind.RRULE:
            recurrence = f" RRULE {job.recurrence.rrule}"
        lines.append(
            f"- {job.id[:8]} at {job.next_run_at.astimezone(display_tz).isoformat()}"
            f"{recurrence} [{job.status.value}]: "
            f"{job.prompt[:120]}"
        )
    return "\n".join(lines)


def _parse_run_at(value: str, local_tz: ZoneInfo) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("run_at must be an ISO-8601 date/time") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(timezone.utc)


def _resolve_job_id(
    store: SQLiteScheduleStore,
    *,
    adapter: str,
    destination_id: str,
    value: str,
) -> str:
    clean = value.strip()
    matches = [
        job.id
        for job in store.list(adapter=adapter, destination_id=destination_id)
        if job.id.startswith(clean)
    ]
    if len(matches) != 1:
        raise ValueError("Scheduled task id is missing or ambiguous")
    return matches[0]


__all__ = ["format_jobs", "scheduled_job_tools"]
