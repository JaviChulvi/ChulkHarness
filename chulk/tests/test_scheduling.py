from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3
from threading import Barrier

import pytest

from chulk.scheduling import SQLiteScheduleStore
from chulk.scheduling.tools import scheduled_job_tools
from chulk.storage import SQLITE_MIGRATIONS, initialize_sqlite_database


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

    claims = {job.id: job.claim_token for job in claimed}
    assert claims[once.id] is not None
    assert claims[recurring.id] is not None
    assert store.complete(once.id, claims[once.id], finished_at=now)
    assert store.complete(recurring.id, claims[recurring.id], finished_at=now)
    assert store.get(once.id).status == "completed"
    updated = store.get(recurring.id)
    assert updated.status == "active"
    assert updated.next_run_at == now + timedelta(hours=1)
    assert updated.scheduled_for == updated.next_run_at


def test_running_cancellation_cannot_be_revived_by_stale_completion(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    now = datetime(2026, 7, 23, 8, tzinfo=timezone.utc)
    job = store.create(
        adapter="telegram",
        destination_id="9",
        prompt="cancel me",
        next_run_at=now,
    )
    claimed = store.claim_due(adapter="telegram", now=now)[0]
    assert claimed.claim_token is not None

    assert store.cancel(job.id, adapter="telegram", destination_id="9")
    assert not store.complete(job.id, claimed.claim_token, finished_at=now)
    assert not store.fail(job.id, claimed.claim_token, "late", failed_at=now)
    assert store.get(job.id).status == "cancelled"


def test_two_workers_claim_a_due_job_only_once(tmp_path) -> None:
    path = tmp_path / "store.sqlite"
    first_store = SQLiteScheduleStore(path)
    second_store = SQLiteScheduleStore(path)
    now = datetime(2026, 7, 23, 8, tzinfo=timezone.utc)
    job = first_store.create(
        adapter="telegram",
        destination_id="9",
        prompt="one worker",
        next_run_at=now,
    )
    barrier = Barrier(2)

    def claim(store: SQLiteScheduleStore):
        barrier.wait()
        return store.claim_due(adapter="telegram", now=now)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(claim, (first_store, second_store)))

    claimed = [value for batch in results for value in batch]
    assert [value.id for value in claimed] == [job.id]
    assert claimed[0].claim_token is not None


def test_expired_claim_is_replaced_and_stale_worker_cannot_transition(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    now = datetime(2026, 7, 23, 8, tzinfo=timezone.utc)
    job = store.create(
        adapter="telegram",
        destination_id="9",
        prompt="lease",
        next_run_at=now,
    )
    stale = store.claim_due(
        adapter="telegram",
        now=now,
        lease_seconds=10,
    )[0]
    current = store.claim_due(
        adapter="telegram",
        now=now + timedelta(seconds=11),
        lease_seconds=10,
    )[0]
    assert stale.claim_token is not None
    assert current.claim_token is not None
    assert current.claim_token != stale.claim_token

    assert not store.complete(
        job.id,
        stale.claim_token,
        finished_at=now + timedelta(seconds=12),
    )
    assert not store.fail(
        job.id,
        stale.claim_token,
        "stale",
        failed_at=now + timedelta(seconds=12),
    )
    assert store.complete(
        job.id,
        current.claim_token,
        finished_at=now + timedelta(seconds=12),
    )


def test_active_worker_can_renew_lease_but_stale_worker_cannot(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    now = datetime(2026, 7, 23, 8, tzinfo=timezone.utc)
    job = store.create(
        adapter="telegram",
        destination_id="9",
        prompt="renew",
        next_run_at=now,
    )
    claimed = store.claim_due(adapter="telegram", now=now, lease_seconds=10)[0]
    assert claimed.claim_token is not None

    assert store.renew_lease(
        job.id,
        claimed.claim_token,
        now=now + timedelta(seconds=5),
        lease_seconds=10,
    )
    assert not store.renew_lease(
        job.id,
        "not-the-owner",
        now=now + timedelta(seconds=6),
        lease_seconds=10,
    )
    assert store.claim_due(
        adapter="telegram",
        now=now + timedelta(seconds=11),
        lease_seconds=10,
    ) == ()
    assert not store.renew_lease(
        job.id,
        claimed.claim_token,
        now=now + timedelta(seconds=16),
        lease_seconds=10,
    )


def test_recurring_retry_preserves_original_cadence(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    scheduled = datetime(2026, 7, 23, 8, tzinfo=timezone.utc)
    job = store.create(
        adapter="telegram",
        destination_id="9",
        prompt="repeat",
        next_run_at=scheduled,
        interval_seconds=3600,
    )
    first = store.claim_due(adapter="telegram", now=scheduled)[0]
    assert first.claim_token is not None
    assert store.fail(
        job.id,
        first.claim_token,
        "temporary",
        retry_seconds=600,
        failed_at=scheduled + timedelta(minutes=1),
    )
    retry_at = scheduled + timedelta(minutes=11)
    retry = store.claim_due(adapter="telegram", now=retry_at)[0]
    assert retry.scheduled_for == scheduled
    assert retry.claim_token is not None

    assert store.complete(
        job.id,
        retry.claim_token,
        finished_at=retry_at + timedelta(minutes=1),
    )
    updated = store.get(job.id)
    assert updated.next_run_at == scheduled + timedelta(hours=1)
    assert updated.scheduled_for == scheduled + timedelta(hours=1)


def test_legacy_scheduled_job_migration_backfills_claim_anchor(tmp_path) -> None:
    path = tmp_path / "legacy-schedule.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:4])
    scheduled = "2026-07-23T08:00:00+00:00"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO scheduled_jobs (
                id, adapter, destination_id, prompt, next_run_at,
                interval_seconds, status, created_at, updated_at
            ) VALUES ('legacy', 'telegram', '9', 'old', ?, NULL, 'active', ?, ?)
            """,
            (scheduled, scheduled, scheduled),
        )

    store = SQLiteScheduleStore(path)
    migrated = store.get("legacy")

    assert migrated.scheduled_for == datetime.fromisoformat(scheduled)
    assert migrated.claim_token is None
    assert list(tmp_path.glob("legacy-schedule.sqlite.backup-v4-*.sqlite"))


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
