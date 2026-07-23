from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chulk.scheduling import SQLiteScheduleStore
from chulk.scheduling.tools import scheduled_job_tools


def test_schedule_store_claims_completes_and_repeats_jobs(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    now = datetime(2026, 7, 23, 8, tzinfo=timezone.utc)
    once = store.create(
        adapter="telegram",
        destination_id="9",
        prompt="once",
        next_run_at=now,
    )
    recurring = store.create(
        adapter="telegram",
        destination_id="9",
        prompt="repeat",
        next_run_at=now,
        interval_seconds=3600,
    )

    claimed = store.claim_due(adapter="telegram", now=now)
    assert {job.id for job in claimed} == {once.id, recurring.id}

    store.complete(once.id, finished_at=now)
    store.complete(recurring.id, finished_at=now)
    assert store.get(once.id).status == "completed"
    updated = store.get(recurring.id)
    assert updated.status == "active"
    assert updated.next_run_at == now + timedelta(hours=1)


def test_schedule_store_scopes_list_and_cancel_to_destination(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    job = store.create(
        adapter="telegram",
        destination_id="9",
        prompt="private",
        next_run_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    assert store.list(adapter="telegram", destination_id="10") == ()
    assert not store.cancel(job.id, adapter="telegram", destination_id="10")
    assert store.cancel(job.id, adapter="telegram", destination_id="9")


def test_scheduling_tools_parse_local_time_and_manage_jobs(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    tools = {
        value.name: value
        for value in scheduled_job_tools(
            store,
            adapter="telegram",
            destination_id="9",
            timezone_name="Europe/Madrid",
        )
    }
    future = datetime.now(timezone.utc) + timedelta(days=2)
    local = future.astimezone().replace(tzinfo=None).isoformat(timespec="minutes")

    result = tools["schedule_task"].callable({"prompt": "hello", "run_at": local})
    assert result.success
    jobs = store.list(adapter="telegram", destination_id="9")
    assert len(jobs) == 1

    listed = tools["list_scheduled_tasks"].callable({})
    assert jobs[0].id[:8] in listed.observation
    cancelled = tools["cancel_scheduled_task"].callable({"job_id": jobs[0].id[:8]})
    assert cancelled.success


def test_schedule_tool_rejects_past_times(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    tool = scheduled_job_tools(
        store,
        adapter="telegram",
        destination_id="9",
        timezone_name="UTC",
    )[0]
    with pytest.raises(ValueError, match="future"):
        tool.callable({"prompt": "late", "run_at": "2020-01-01T00:00:00Z"})
