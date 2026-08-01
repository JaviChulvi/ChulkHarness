"""Occurrence claiming and run lifecycle operations for scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.scheduling._store_support import (
    AutomationNotFoundError,
    DEFAULT_AUTOMATION_LEASE_SECONDS,
    _ScheduleStoreMixin,
    _decode,
    _encode,
    _json,
    _row_to_job,
    _row_to_run,
    _utc,
    _utc_now,
)
from chulk.scheduling.models import (
    AutomationDeliveryState,
    AutomationJobStatus,
    AutomationRun,
    AutomationRunReason,
    AutomationRunStatus,
    RecurrenceKind,
    ScheduledJob,
)


class _ScheduleRunsMixin(_ScheduleStoreMixin):
    def runs(self, job_id: str, *, limit: int = 100) -> tuple[AutomationRun, ...]:
        self.get(job_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM automation_runs
                WHERE profile_id = ? AND job_id = ?
                ORDER BY created_at DESC, id DESC LIMIT ?
                """,
                (self.profile_id, job_id, limit),
            ).fetchall()
        return tuple(_row_to_run(row) for row in rows)

    def get_run(self, run_id: str) -> AutomationRun:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM automation_runs WHERE profile_id = ? AND id = ?",
                (self.profile_id, run_id),
            ).fetchone()
        if row is None:
            raise AutomationNotFoundError(f"Automation run not found: {run_id}")
        return _row_to_run(row)

    def claim_due(
        self,
        *,
        adapter: str | None = None,
        now: datetime | None = None,
        limit: int = 10,
        lease_seconds: int = DEFAULT_AUTOMATION_LEASE_SECONDS,
        worker_id: str = "scheduler",
    ) -> tuple[ScheduledJob, ...]:
        observed = _utc(now or _utc_now(), "now")
        if limit < 1 or lease_seconds < 1:
            raise ValueError("limit and lease_seconds must be positive")
        self.recover_expired(now=observed, actor="lease-recovery")
        lease_until = observed + timedelta(seconds=lease_seconds)
        claims: list[ScheduledJob] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            clauses = [
                "j.profile_id = ?",
                "j.status = 'active'",
                "((j.next_run_at <= ? AND NOT ("
                "json_extract(j.recurrence_json, '$.kind') = 'once' AND EXISTS ("
                "SELECT 1 FROM automation_runs completed_once "
                "WHERE completed_once.profile_id = j.profile_id "
                "AND completed_once.job_id = j.id "
                "AND completed_once.reason = 'scheduled'"
                "))) OR EXISTS ("
                "SELECT 1 FROM automation_run_requests r "
                "WHERE r.profile_id = j.profile_id AND r.job_id = j.id "
                "AND r.status = 'pending' AND r.available_at <= ?))",
            ]
            params: list[object] = [
                self.profile_id,
                _encode(observed),
                _encode(observed),
            ]
            if adapter is not None:
                clauses.append("j.adapter = ?")
                params.append(adapter)
            params.append(self._claim_candidate_limit(limit))
            rows = conn.execute(
                f"""
                SELECT j.* FROM automation_jobs j
                WHERE {" AND ".join(clauses)}
                ORDER BY CASE WHEN j.next_run_at <= ? THEN j.next_run_at ELSE NULL END,
                         j.created_at, j.id
                LIMIT ?
                {self._claim_lock_clause()}
                """,
                (*params[:-1], _encode(observed), params[-1]),
            ).fetchall()
            for row in rows:
                if len(claims) >= limit:
                    break
                job_id = str(row["id"])
                job = self._get_in(conn, job_id)
                if (
                    job.status is not AutomationJobStatus.ACTIVE
                    or job.revision != int(row["revision"])
                ):
                    continue
                if job.max_runs is not None and job.run_count >= job.max_runs:
                    self._terminalize(
                        conn, job, AutomationJobStatus.COMPLETED, observed
                    )
                    continue
                request = conn.execute(
                    """
                    SELECT * FROM automation_run_requests
                    WHERE profile_id = ? AND job_id = ? AND status = 'pending'
                      AND available_at <= ?
                    ORDER BY available_at, created_at, id LIMIT 1
                    """,
                    (self.profile_id, job.id, _encode(observed)),
                ).fetchone()
                reason = AutomationRunReason.SCHEDULED
                occurrence = job.scheduled_for
                trigger_event_id = None
                request_id = None
                next_at: datetime | None = job.next_run_at
                if request is not None:
                    request_id = str(request["id"])
                    reason = AutomationRunReason(str(request["reason"]))
                    occurrence = _decode(str(request["occurrence_at"]))
                    trigger_event_id = (
                        str(request["trigger_event_id"])
                        if request["trigger_event_id"]
                        else None
                    )
                else:
                    decision = self.recurrence.due(
                        job.recurrence,
                        job.scheduled_for,
                        observed,
                        anchor=job.created_at
                        if job.recurrence.kind is not RecurrenceKind.INTERVAL
                        else job.scheduled_for,
                        identity=job.id,
                    )
                    if not decision.occurrences:
                        if decision.next_at is None:
                            self._terminalize(
                                conn, job, AutomationJobStatus.EXPIRED, observed
                            )
                        else:
                            conn.execute(
                                """
                                UPDATE automation_jobs
                                SET next_run_at = ?, scheduled_for = ?, updated_at = ?
                                WHERE profile_id = ? AND id = ?
                                """,
                                (
                                    _encode(decision.next_at),
                                    _encode(decision.next_at),
                                    _encode(observed),
                                    self.profile_id,
                                    job.id,
                                ),
                            )
                        continue
                    occurrence = decision.occurrences[0]
                    next_at = decision.next_at
                    for catch_up in decision.occurrences[1:]:
                        self._enqueue_request(
                            conn,
                            job_id=job.id,
                            reason=AutomationRunReason.SCHEDULED,
                            occurrence_at=catch_up,
                            available_at=observed,
                            idempotency_key=f"catch-up:{job.id}:{catch_up.isoformat()}",
                            now=observed,
                        )
                claim_token = uuid4().hex
                run_id = uuid4().hex
                attempt = self._next_attempt(
                    conn,
                    job_id=job.id,
                    occurrence_at=occurrence,
                    trigger_event_id=trigger_event_id,
                    reason=reason,
                )
                persisted_next = next_at or occurrence
                cursor = conn.execute(
                    """
                    UPDATE automation_jobs
                    SET status = 'running', claim_token = ?, lease_until = ?,
                        active_run_id = ?,
                        run_count = run_count + CASE WHEN ? = 'retry' THEN 0 ELSE 1 END,
                        next_run_at = ?, scheduled_for = ?, updated_at = ?
                    WHERE profile_id = ? AND id = ? AND status = 'active'
                    """,
                    (
                        claim_token,
                        _encode(lease_until),
                        run_id,
                        reason.value,
                        _encode(persisted_next),
                        _encode(occurrence),
                        _encode(observed),
                        self.profile_id,
                        job.id,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                conn.execute(
                    """
                    INSERT INTO automation_runs (
                        id, job_id, profile_id, occurrence_at, reason, status,
                        attempt, claim_token, worker_id, lease_until, started_at,
                        result_json, usage_json, cost_json, artifact_refs_json,
                        delivery_state, trigger_event_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, '{}',
                              '{}', '{}', '[]', 'none', ?, ?, ?)
                    """,
                    (
                        run_id,
                        job.id,
                        self.profile_id,
                        _encode(occurrence),
                        reason.value,
                        attempt,
                        claim_token,
                        worker_id,
                        _encode(lease_until),
                        _encode(observed),
                        trigger_event_id,
                        _encode(observed),
                        _encode(observed),
                    ),
                )
                if request_id is not None:
                    conn.execute(
                        """
                        UPDATE automation_run_requests
                        SET status = 'claimed', updated_at = ?
                        WHERE profile_id = ? AND id = ? AND status = 'pending'
                        """,
                        (_encode(observed), self.profile_id, request_id),
                    )
                self._event(
                    conn,
                    job.id,
                    "run_claimed",
                    job.revision,
                    worker_id,
                    run_id=run_id,
                    metadata={
                        "reason": reason.value,
                        "occurrence_at": occurrence.isoformat(),
                    },
                    now=observed,
                )
                claims.append(self._get_in(conn, job.id))
        return tuple(claims)

    def renew_lease(
        self,
        job_id: str,
        claim_token: str,
        *,
        now: datetime | None = None,
        lease_seconds: int = DEFAULT_AUTOMATION_LEASE_SECONDS,
    ) -> bool:
        observed = _utc(now or _utc_now(), "now")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        lease_until = observed + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_job_mutation(conn, job_id)
            row = conn.execute(
                """
                SELECT active_run_id FROM automation_jobs
                WHERE profile_id = ? AND id = ? AND status = 'running'
                  AND claim_token = ? AND lease_until >= ?
                """,
                (self.profile_id, job_id, claim_token, _encode(observed)),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                """
                UPDATE automation_jobs SET lease_until = ?, updated_at = ?
                WHERE profile_id = ? AND id = ? AND claim_token = ?
                """,
                (
                    _encode(lease_until),
                    _encode(observed),
                    self.profile_id,
                    job_id,
                    claim_token,
                ),
            )
            conn.execute(
                """
                UPDATE automation_runs SET lease_until = ?, updated_at = ?
                WHERE profile_id = ? AND id = ? AND claim_token = ?
                """,
                (
                    _encode(lease_until),
                    _encode(observed),
                    self.profile_id,
                    str(row["active_run_id"]),
                    claim_token,
                ),
            )
        return True

    def complete(
        self,
        job_id: str,
        claim_token: str,
        *,
        finished_at: datetime | None = None,
        result: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        usage: Mapping[str, Any] | None = None,
        cost: Mapping[str, Any] | None = None,
        artifact_refs: tuple[str, ...] = (),
        delivery_state: AutomationDeliveryState = AutomationDeliveryState.NONE,
    ) -> bool:
        finished = _utc(finished_at or _utc_now(), "finished_at")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._claimed_job(conn, job_id, claim_token, finished)
            if job is None:
                return False
            assert job.active_run_id is not None
            terminal = (
                job.recurrence.kind is RecurrenceKind.ONCE
                and not self._has_pending_requests(conn, job.id)
                and not self._has_enabled_triggers(conn, job.id)
            ) or (job.max_runs is not None and job.run_count >= job.max_runs)
            next_status = (
                AutomationJobStatus.COMPLETED
                if terminal
                else AutomationJobStatus.ACTIVE
            )
            conn.execute(
                """
                UPDATE automation_runs
                SET status = 'completed', claim_token = NULL, worker_id = NULL,
                    lease_until = NULL, finished_at = ?, result_json = ?,
                    trace_id = ?, usage_json = ?, cost_json = ?,
                    artifact_refs_json = ?, delivery_state = ?, updated_at = ?
                WHERE profile_id = ? AND id = ? AND claim_token = ?
                """,
                (
                    _encode(finished),
                    _json(dict(result or {})),
                    trace_id,
                    _json(dict(usage or {})),
                    _json(dict(cost or {})),
                    _json(list(dict.fromkeys(artifact_refs))),
                    AutomationDeliveryState(delivery_state).value,
                    _encode(finished),
                    self.profile_id,
                    job.active_run_id,
                    claim_token,
                ),
            )
            conn.execute(
                """
                UPDATE automation_jobs
                SET status = ?, claim_token = NULL, lease_until = NULL,
                    active_run_id = NULL, last_run_at = ?, last_error = NULL,
                    scheduled_for = next_run_at,
                    updated_at = ?
                WHERE profile_id = ? AND id = ? AND claim_token = ?
                """,
                (
                    next_status.value,
                    _encode(finished),
                    _encode(finished),
                    self.profile_id,
                    job_id,
                    claim_token,
                ),
            )
            if (
                AutomationDeliveryState(delivery_state)
                is not AutomationDeliveryState.NONE
            ):
                self._delivery_attempt(
                    conn,
                    run_id=job.active_run_id,
                    state=AutomationDeliveryState(delivery_state),
                    now=finished,
                )
            self._finish_claimed_request(conn, job.active_run_id, finished)
            self._event(
                conn,
                job.id,
                "run_completed",
                job.revision,
                "runner",
                run_id=job.active_run_id,
                now=finished,
            )
        return True

    def fail(
        self,
        job_id: str,
        claim_token: str,
        error: str,
        *,
        retry_seconds: int | None = None,
        failed_at: datetime | None = None,
        budget_exhausted: bool = False,
    ) -> bool:
        observed = _utc(failed_at or _utc_now(), "failed_at")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._claimed_job(conn, job_id, claim_token, observed)
            if job is None:
                return False
            assert job.active_run_id is not None
            run = self._run_in(conn, job.active_run_id)
            retry = run.attempt < job.retry_policy.max_attempts and not budget_exhausted
            delay = (
                retry_seconds
                if retry_seconds is not None
                else job.retry_policy.delay_for(run.attempt)
            )
            if retry and delay < 1:
                raise ValueError("retry_seconds must be positive")
            run_status = (
                AutomationRunStatus.BUDGET_EXHAUSTED
                if budget_exhausted
                else AutomationRunStatus.FAILED
            )
            conn.execute(
                """
                UPDATE automation_runs
                SET status = ?, claim_token = NULL, worker_id = NULL,
                    lease_until = NULL, finished_at = ?, error = ?, updated_at = ?
                WHERE profile_id = ? AND id = ? AND claim_token = ?
                """,
                (
                    run_status.value,
                    _encode(observed),
                    error[:2_000],
                    _encode(observed),
                    self.profile_id,
                    run.id,
                    claim_token,
                ),
            )
            if retry:
                self._enqueue_request(
                    conn,
                    job_id=job.id,
                    reason=AutomationRunReason.RETRY,
                    occurrence_at=run.occurrence_at,
                    available_at=observed + timedelta(seconds=delay),
                    trigger_event_id=run.trigger_event_id,
                    idempotency_key=f"retry:{run.id}:{run.attempt + 1}",
                    now=observed,
                )
            terminal = (
                not retry
                and job.recurrence.kind is RecurrenceKind.ONCE
                and not self._has_pending_requests(conn, job.id)
                and not self._has_enabled_triggers(conn, job.id)
            )
            conn.execute(
                """
                UPDATE automation_jobs
                SET status = ?, claim_token = NULL, lease_until = NULL,
                    active_run_id = NULL, last_run_at = ?, last_error = ?,
                    scheduled_for = CASE
                        WHEN ? THEN scheduled_for ELSE next_run_at
                    END,
                    updated_at = ?
                WHERE profile_id = ? AND id = ? AND claim_token = ?
                """,
                (
                    (
                        AutomationJobStatus.COMPLETED.value
                        if terminal
                        else AutomationJobStatus.ACTIVE.value
                    ),
                    _encode(observed),
                    error[:500],
                    int(retry),
                    _encode(observed),
                    self.profile_id,
                    job.id,
                    claim_token,
                ),
            )
            self._finish_claimed_request(conn, run.id, observed)
            self._event(
                conn,
                job.id,
                ("run_budget_exhausted" if budget_exhausted else "run_failed"),
                job.revision,
                "runner",
                run_id=run.id,
                metadata={"retry_scheduled": retry},
                now=observed,
            )
        return True

    def recover_expired(
        self,
        *,
        now: datetime | None = None,
        actor: str = "recovery",
    ) -> tuple[AutomationRun, ...]:
        observed = _utc(now or _utc_now(), "now")
        recovered: list[AutomationRun] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""
                SELECT * FROM automation_jobs
                WHERE profile_id = ? AND status = 'running' AND lease_until < ?
                ORDER BY lease_until, id
                {self._recovery_lock_clause()}
                """,
                (self.profile_id, _encode(observed)),
            ).fetchall()
            for row in rows:
                job = self._get_in(conn, str(row["id"]))
                if (
                    job.status is not AutomationJobStatus.RUNNING
                    or job.lease_until is None
                    or job.lease_until >= observed
                ):
                    continue
                assert job.active_run_id is not None
                run = self._run_in(conn, job.active_run_id)
                conn.execute(
                    """
                    UPDATE automation_runs
                    SET status = 'unknown', claim_token = NULL, worker_id = NULL,
                        lease_until = NULL, finished_at = ?, error = ?,
                        updated_at = ?
                    WHERE profile_id = ? AND id = ?
                    """,
                    (
                        _encode(observed),
                        "execution lease expired; outcome is unknown",
                        _encode(observed),
                        self.profile_id,
                        job.active_run_id,
                    ),
                )
                retry = run.attempt < job.retry_policy.max_attempts
                if retry:
                    self._enqueue_request(
                        conn,
                        job_id=job.id,
                        reason=AutomationRunReason.RETRY,
                        occurrence_at=run.occurrence_at,
                        available_at=observed,
                        trigger_event_id=run.trigger_event_id,
                        idempotency_key=f"lease-retry:{run.id}:{run.attempt + 1}",
                        now=observed,
                    )
                terminal = (
                    not retry
                    and job.recurrence.kind is RecurrenceKind.ONCE
                    and not self._has_pending_requests(conn, job.id)
                    and not self._has_enabled_triggers(conn, job.id)
                )
                conn.execute(
                    """
                    UPDATE automation_jobs
                    SET status = ?, claim_token = NULL, lease_until = NULL,
                        active_run_id = NULL, last_error = ?,
                        scheduled_for = CASE
                            WHEN ? THEN scheduled_for ELSE next_run_at
                        END,
                        updated_at = ?
                    WHERE profile_id = ? AND id = ?
                    """,
                    (
                        (
                            AutomationJobStatus.COMPLETED.value
                            if terminal
                            else AutomationJobStatus.ACTIVE.value
                        ),
                        "execution lease expired; outcome is unknown",
                        int(retry),
                        _encode(observed),
                        self.profile_id,
                        job.id,
                    ),
                )
                self._finish_claimed_request(conn, job.active_run_id, observed)
                self._event(
                    conn,
                    job.id,
                    "run_unknown",
                    job.revision,
                    actor,
                    run_id=job.active_run_id,
                    now=observed,
                )
                recovered.append(self._run_in(conn, job.active_run_id))
        return tuple(recovered)

    def _claimed_job(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        claim_token: str,
        observed: datetime,
    ) -> ScheduledJob | None:
        self._serialize_job_mutation(conn, job_id)
        row = conn.execute(
            """
            SELECT * FROM automation_jobs
            WHERE profile_id = ? AND id = ? AND status = 'running'
              AND claim_token = ? AND lease_until >= ?
            """,
            (self.profile_id, job_id, claim_token, _encode(observed)),
        ).fetchone()
        return _row_to_job(row) if row is not None else None

    def _enqueue_request(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        reason: AutomationRunReason,
        occurrence_at: datetime,
        idempotency_key: str,
        now: datetime,
        available_at: datetime | None = None,
        trigger_event_id: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO automation_run_requests (
                id, job_id, profile_id, reason, occurrence_at,
                available_at, trigger_event_id, idempotency_key, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            ON CONFLICT DO NOTHING
            """,
            (
                uuid4().hex,
                job_id,
                self.profile_id,
                reason.value,
                _encode(occurrence_at),
                _encode(available_at or occurrence_at),
                trigger_event_id,
                idempotency_key,
                _encode(now),
                _encode(now),
            ),
        )

    def _next_attempt(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        occurrence_at: datetime,
        trigger_event_id: str | None,
        reason: AutomationRunReason,
    ) -> int:
        if reason is AutomationRunReason.RETRY:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(attempt), 0) AS value FROM automation_runs
                WHERE profile_id = ? AND job_id = ?
                """,
                (self.profile_id, job_id),
            ).fetchone()
            return int(row["value"]) + 1
        row = conn.execute(
            """
            SELECT COALESCE(MAX(attempt), 0) AS value FROM automation_runs
            WHERE profile_id = ? AND job_id = ? AND occurrence_at = ?
              AND COALESCE(trigger_event_id, '') = COALESCE(?, '')
            """,
            (
                self.profile_id,
                job_id,
                _encode(occurrence_at),
                trigger_event_id,
            ),
        ).fetchone()
        return int(row["value"]) + 1

    def _has_pending_requests(self, conn: sqlite3.Connection, job_id: str) -> bool:
        return (
            conn.execute(
                """
                SELECT 1 FROM automation_run_requests
                WHERE profile_id = ? AND job_id = ? AND status = 'pending' LIMIT 1
                """,
                (self.profile_id, job_id),
            ).fetchone()
            is not None
        )

    def _finish_claimed_request(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        now: datetime,
    ) -> None:
        row = conn.execute(
            "SELECT trigger_event_id, reason, job_id, occurrence_at FROM automation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return
        conn.execute(
            """
            UPDATE automation_run_requests SET status = 'done', updated_at = ?
            WHERE profile_id = ? AND job_id = ? AND status = 'claimed'
              AND occurrence_at = ?
              AND COALESCE(trigger_event_id, '') = COALESCE(?, '')
            """,
            (
                _encode(now),
                self.profile_id,
                str(row["job_id"]),
                str(row["occurrence_at"]),
                row["trigger_event_id"],
            ),
        )

    def _terminalize(
        self,
        conn: sqlite3.Connection,
        job: ScheduledJob,
        status: AutomationJobStatus,
        now: datetime,
    ) -> None:
        conn.execute(
            """
            UPDATE automation_jobs SET status = ?, claim_token = NULL,
                lease_until = NULL, active_run_id = NULL, updated_at = ?
            WHERE profile_id = ? AND id = ?
            """,
            (status.value, _encode(now), self.profile_id, job.id),
        )
        self._event(
            conn,
            job.id,
            status.value,
            job.revision,
            "scheduler",
            now=now,
        )
