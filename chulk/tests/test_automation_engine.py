from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from chulk.scheduling import (
    AutomationConflictError,
    AmbiguousTimePolicy,
    AutomationDeliveryState,
    AutomationExecutionResult,
    AutomationJobStatus,
    AutomationRetryPolicy,
    AutomationRunReason,
    AutomationRunStatus,
    AutomationSupervisor,
    MisfirePolicy,
    NonexistentTimePolicy,
    RecurrenceCalculator,
    RecurrenceKind,
    RecurrenceSpec,
    SQLiteScheduleStore,
    TriggerKind,
)
from chulk.storage import SQLITE_MIGRATIONS, initialize_sqlite_database


def test_interval_cron_and_rrule_recurrence_are_timezone_aware() -> None:
    calculator = RecurrenceCalculator()
    anchor = datetime(2026, 3, 27, 8, tzinfo=timezone.utc)
    interval = RecurrenceSpec(
        kind=RecurrenceKind.INTERVAL,
        interval_seconds=3_600,
    )
    assert calculator.next_after(
        interval,
        anchor + timedelta(hours=3, minutes=25),
        anchor=anchor,
    ) == anchor + timedelta(hours=4)

    cron = RecurrenceSpec(
        kind=RecurrenceKind.CRON,
        timezone_name="Europe/Madrid",
        cron="30 2 * * *",
    )
    # 02:30 does not exist on this DST transition; croniter advances safely.
    assert calculator.next_after(
        cron,
        datetime(2026, 3, 28, 22, tzinfo=timezone.utc),
        anchor=anchor,
    ) == datetime(2026, 3, 29, 1, tzinfo=timezone.utc)
    skipped_gap = RecurrenceSpec(
        kind=RecurrenceKind.CRON,
        timezone_name="Europe/Madrid",
        cron="30 2 * * *",
        nonexistent_time_policy=NonexistentTimePolicy.SKIP,
    )
    assert calculator.next_after(
        skipped_gap,
        datetime(2026, 3, 28, 22, tzinfo=timezone.utc),
        anchor=anchor,
    ) == datetime(2026, 3, 30, 0, 30, tzinfo=timezone.utc)
    latest_fold = RecurrenceSpec(
        kind=RecurrenceKind.CRON,
        timezone_name="Europe/Madrid",
        cron="30 2 * * *",
        ambiguous_time_policy=AmbiguousTimePolicy.LATEST,
    )
    assert calculator.next_after(
        latest_fold,
        datetime(2026, 10, 24, 22, tzinfo=timezone.utc),
        anchor=anchor,
    ) == datetime(2026, 10, 25, 1, 30, tzinfo=timezone.utc)

    rule = RecurrenceSpec(
        kind=RecurrenceKind.RRULE,
        timezone_name="Europe/Madrid",
        rrule="FREQ=DAILY;COUNT=3",
    )
    assert calculator.next_after(
        rule,
        anchor,
        anchor=anchor,
    ) == datetime(2026, 3, 28, 8, tzinfo=timezone.utc)
    assert (
        calculator.next_after(
            rule,
            datetime(2026, 3, 29, 8, tzinfo=timezone.utc),
            anchor=anchor,
        )
        is None
    )


def test_misfire_policies_skip_run_once_and_bound_catch_up() -> None:
    calculator = RecurrenceCalculator()
    scheduled = datetime(2026, 7, 1, 8, tzinfo=timezone.utc)
    observed = scheduled + timedelta(hours=5, minutes=5)

    skipped = calculator.due(
        RecurrenceSpec(
            kind=RecurrenceKind.INTERVAL,
            interval_seconds=3_600,
            misfire_policy=MisfirePolicy.SKIP,
            misfire_grace_seconds=0,
        ),
        scheduled,
        observed,
        anchor=scheduled,
    )
    assert skipped.occurrences == ()
    assert skipped.next_at == scheduled + timedelta(hours=6)

    run_once = calculator.due(
        RecurrenceSpec(
            kind=RecurrenceKind.INTERVAL,
            interval_seconds=3_600,
            misfire_policy=MisfirePolicy.RUN_ONCE,
            misfire_grace_seconds=0,
        ),
        scheduled,
        observed,
        anchor=scheduled,
    )
    assert run_once.occurrences == (scheduled + timedelta(hours=2),)
    assert run_once.next_at == scheduled + timedelta(hours=6)

    catch_up = calculator.due(
        RecurrenceSpec(
            kind=RecurrenceKind.INTERVAL,
            interval_seconds=3_600,
            misfire_policy=MisfirePolicy.CATCH_UP,
            misfire_grace_seconds=0,
            max_catch_up=3,
        ),
        scheduled,
        observed,
        anchor=scheduled,
    )
    assert catch_up.occurrences == (
        scheduled,
        scheduled + timedelta(hours=1),
        scheduled + timedelta(hours=2),
    )


def test_controls_are_revision_safe_and_idempotent(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite", profile_id="ops")
    future = datetime.now(timezone.utc) + timedelta(days=1)
    job = store.create(
        adapter="websocket",
        destination_id="owner",
        prompt="review queue",
        next_run_at=future,
        requires_approval=True,
    )
    assert job.status is AutomationJobStatus.PENDING_APPROVAL

    approved = store.approve(
        job.id,
        expected_revision=0,
        idempotency_key="approve-1",
    )
    replay = store.approve(
        job.id,
        expected_revision=0,
        idempotency_key="approve-1",
    )
    assert replay == approved
    with pytest.raises(AutomationConflictError):
        store.pause(
            job.id,
            expected_revision=0,
            idempotency_key="pause-stale",
        )

    paused = store.pause(
        job.id,
        expected_revision=approved.revision,
        idempotency_key="pause-1",
    )
    resumed = store.resume(
        job.id,
        expected_revision=paused.revision,
        idempotency_key="resume-1",
    )
    requested = store.run_now(
        job.id,
        expected_revision=resumed.revision,
        idempotency_key="manual-1",
        now=future - timedelta(hours=2),
    )
    assert (
        store.run_now(
            job.id,
            expected_revision=resumed.revision,
            idempotency_key="manual-1",
            now=future - timedelta(hours=2),
        )
        == requested
    )
    assert requested.revision == resumed.revision + 1
    claimed = store.claim_due(now=future - timedelta(hours=2))[0]
    run = store.get_run(claimed.active_run_id or "")
    assert run.reason is AutomationRunReason.MANUAL

    editable = store.create(
        adapter="websocket",
        destination_id="owner",
        prompt="old",
        next_run_at=future,
    )
    updated = store.update(
        editable.id,
        expected_revision=editable.revision,
        idempotency_key="update-1",
        prompt="new",
    )
    assert (
        store.update(
            editable.id,
            expected_revision=editable.revision,
            idempotency_key="update-1",
            prompt="new",
        )
        == updated
    )
    assert store.cancel(
        editable.id,
        expected_revision=updated.revision,
        idempotency_key="cancel-1",
    )
    assert store.cancel(
        editable.id,
        expected_revision=updated.revision,
        idempotency_key="cancel-1",
    )


def test_webhook_authentication_deduplication_and_profile_isolation(tmp_path) -> None:
    path = tmp_path / "store.sqlite"
    store = SQLiteScheduleStore(path, profile_id="alpha")
    other = SQLiteScheduleStore(path, profile_id="beta")
    job = store.create(
        adapter="webhook",
        destination_id="sink",
        prompt="process payload",
        next_run_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    trigger, credential = store.create_webhook_trigger(job.id)
    with pytest.raises(PermissionError):
        store.ingest_webhook(
            trigger.id,
            token="wrong",
            event_id="evt-1",
            payload={"secret": "sk-test-value"},
        )
    first = store.ingest_webhook(
        trigger.id,
        token=credential,
        event_id="evt-1",
        payload={"authorization": "Bearer private", "value": 7},
    )
    replay = store.ingest_webhook(
        trigger.id,
        token=credential,
        event_id="evt-1",
        payload={"value": 999},
    )
    assert replay.id == first.id
    assert "private" not in repr(first.payload)
    claimed = store.claim_due()[0]
    run = store.get_run(claimed.active_run_id or "")
    assert run.reason is AutomationRunReason.TRIGGER
    assert run.trigger_event_id == first.id
    with pytest.raises(LookupError):
        other.get(job.id)


def test_supervisor_records_result_delivery_and_completion_trigger(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    now = datetime.now(timezone.utc)
    source = store.create(
        adapter="test",
        destination_id="one",
        prompt="source",
        next_run_at=now,
    )
    downstream = store.create(
        adapter="test",
        destination_id="two",
        prompt="downstream",
        next_run_at=now + timedelta(days=30),
    )
    store.create_completion_trigger(
        downstream.id,
        kind=TriggerKind.JOB_COMPLETION,
        source_resource_id=source.id,
    )
    deliveries: list[str] = []

    class Runner:
        def run(self, job, run):
            return AutomationExecutionResult(
                summary=f"done {job.id}",
                trace_id="trace-1",
                usage={"total_tokens": 12},
                artifact_refs=("artifact-1",),
            )

        def cancel(self):
            pass

        def close(self):
            pass

    supervisor = AutomationSupervisor(
        store,
        lambda _job, _run: Runner(),
        worker_id="worker",
        delivery=lambda target, result, _job, _run: deliveries.append(
            f"{target.destination_id}:{result.summary}"
        ),
    )
    completed = supervisor.run_once()
    assert completed is not None
    assert completed.status is AutomationRunStatus.COMPLETED
    assert completed.delivery_state is AutomationDeliveryState.DELIVERED
    assert completed.trace_id == "trace-1"
    assert completed.duration_ms is not None
    assert [item.state for item in store.delivery_history(completed.id)] == [
        AutomationDeliveryState.PENDING,
        AutomationDeliveryState.DELIVERED,
    ]
    assert deliveries == [f"one:done {source.id}"]

    triggered = store.claim_due()[0]
    assert triggered.id == downstream.id
    assert (
        store.get_run(triggered.active_run_id or "").reason
        is AutomationRunReason.TRIGGER
    )


def test_expired_run_is_unknown_and_retried_with_a_new_fence(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    now = datetime(2026, 7, 1, 8, tzinfo=timezone.utc)
    job = store.create(
        adapter="test",
        destination_id="one",
        prompt="recover",
        next_run_at=now,
        retry_policy=AutomationRetryPolicy(max_attempts=2),
    )
    stale = store.claim_due(now=now, lease_seconds=10)[0]
    stale_run = store.get_run(stale.active_run_id or "")
    current = store.claim_due(now=now + timedelta(seconds=11), lease_seconds=10)[0]
    current_run = store.get_run(current.active_run_id or "")

    assert store.get_run(stale_run.id).status is AutomationRunStatus.UNKNOWN
    assert current_run.attempt == 2
    assert current.claim_token != stale.claim_token
    assert not store.complete(
        job.id,
        stale.claim_token or "",
        finished_at=now + timedelta(seconds=12),
    )


def test_legacy_running_schedule_migrates_to_unknown_history(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:5])
    scheduled = datetime.now(timezone.utc) - timedelta(minutes=5)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO scheduled_jobs (
                id, adapter, destination_id, prompt, next_run_at,
                interval_seconds, status, lease_until, last_run_at,
                last_error, created_at, updated_at, claim_token, scheduled_for
            ) VALUES ('legacy-running', 'telegram', '9', 'old', ?, NULL,
                      'running', ?, NULL, NULL, ?, ?, 'old-claim', ?)
            """,
            (
                scheduled.isoformat(),
                (scheduled + timedelta(minutes=1)).isoformat(),
                scheduled.isoformat(),
                scheduled.isoformat(),
                scheduled.isoformat(),
            ),
        )
    store = SQLiteScheduleStore(path)
    job = store.get("legacy-running")
    run = store.runs(job.id)[0]
    assert job.status is AutomationJobStatus.COMPLETED
    assert run.status is AutomationRunStatus.UNKNOWN
    assert store.claim_due(now=datetime.now(timezone.utc)) == ()
