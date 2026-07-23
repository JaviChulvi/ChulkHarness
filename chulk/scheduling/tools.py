"""Destination-scoped tools for creating and managing scheduled jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from zoneinfo import ZoneInfo

from chulk.scheduling.models import ScheduledJob
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

    def schedule_task(prompt: str, run_at: str, interval_seconds: int | None = None) -> str:
        when = _parse_run_at(run_at, local_tz)
        if when <= datetime.now(timezone.utc):
            raise ValueError("run_at must be in the future")
        if interval_seconds is not None and interval_seconds < 60:
            raise ValueError("interval_seconds must be at least 60")
        job = store.create(
            adapter=adapter,
            destination_id=destination_id,
            prompt=prompt,
            next_run_at=when,
            interval_seconds=interval_seconds,
        )
        recurrence = f", every {interval_seconds}s" if interval_seconds else ""
        return f"Scheduled {job.id[:8]} for {job.next_run_at.isoformat()}{recurrence}."

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
        cast(
            Tool,
            tool(
                schedule_task,
                name="schedule_task",
                description=(
                    "Schedule a prompt for later delivery. run_at must be an ISO-8601 "
                    f"date/time in {timezone_name} unless it includes an offset; "
                    "interval_seconds makes it recurring and must be at least 60."
                ),
                permission_level=ToolPermissionLevel.WRITE,
                idempotent=False,
            ),
        ),
        cast(
            Tool,
            tool(
                current_time,
                name="current_time",
                description=(
                    "Return the current date, time, and configured timezone before resolving "
                    "relative schedule requests such as tomorrow or next Monday."
                ),
            ),
        ),
        cast(
            Tool,
            tool(
                list_scheduled_tasks,
                name="list_scheduled_tasks",
                description="List active scheduled tasks for this chat.",
            ),
        ),
        cast(
            Tool,
            tool(
                cancel_scheduled_task,
                name="cancel_scheduled_task",
                description=(
                    "Cancel a scheduled task by its full id or displayed 8-character prefix."
                ),
                permission_level=ToolPermissionLevel.WRITE,
                idempotent=True,
            ),
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
        interval = job.interval_seconds
        recurrence = f" every {interval}s" if interval else ""
        lines.append(
            f"- {job.id[:8]} at {job.next_run_at.astimezone(display_tz).isoformat()}"
            f"{recurrence}: "
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
