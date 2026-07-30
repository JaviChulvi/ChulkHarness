"""Transactional SQLite store for profile-owned automation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.gateway import DeliveryTarget
from chulk.scheduling.models import (
    AutomationDeliveryAttempt,
    AutomationDeliveryState,
    AutomationJobEvent,
    AutomationJobStatus,
    AutomationRetryPolicy,
    AutomationRun,
    AutomationRunReason,
    AutomationRunStatus,
    AutomationTrigger,
    AmbiguousTimePolicy,
    MisfirePolicy,
    NonexistentTimePolicy,
    RecurrenceKind,
    RecurrenceSpec,
    ScheduledJob,
    TriggerEnvelope,
    TriggerKind,
    TriggerTrust,
)
from chulk.scheduling.recurrence import RecurrenceCalculator
from chulk.storage import initialize_sqlite_database, sqlite_connection
from chulk.usage import BudgetScope, ExactCost, RunBudget, UnknownCostPolicy


DEFAULT_AUTOMATION_LEASE_SECONDS = 120


class AutomationConflictError(RuntimeError):
    """Raised when revision or idempotency expectations conflict."""


class AutomationNotFoundError(LookupError):
    """Raised when an automation resource is outside the owner profile."""


class SQLiteScheduleStore:
    """Persist jobs, occurrences, attempts, controls, and trigger deduplication."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        profile_id: str = "default",
        recurrence_calculator: RecurrenceCalculator | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.profile_id = _required(profile_id, "profile_id")
        self.recurrence = recurrence_calculator or RecurrenceCalculator()
        initialize_sqlite_database(self.db_path)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_connection(self.db_path)

    def _serialize_job_mutation(
        self,
        conn: sqlite3.Connection,
        job_id: str,
    ) -> None:
        """Let backends serialize state transitions for one job."""

    def _serialize_control_action(
        self,
        conn: sqlite3.Connection,
        idempotency_key: str,
    ) -> None:
        """Let backends serialize profile-wide control idempotency keys."""

    def _serialize_trigger_ingest(
        self,
        conn: sqlite3.Connection,
        trigger_id: str,
    ) -> None:
        """Let backends serialize source-event deduplication per trigger."""

    def _claim_lock_clause(self) -> str:
        """Return backend-specific locking for due-job candidates."""
        return ""

    def _claim_candidate_limit(self, limit: int) -> int:
        """Overfetch SQLite candidates that may terminalize before claiming."""
        return limit * 4

    def _recovery_lock_clause(self) -> str:
        """Return backend-specific locking for schedule recovery workers."""
        return ""

    def create(
        self,
        *,
        adapter: str,
        destination_id: str,
        prompt: str,
        next_run_at: datetime,
        interval_seconds: int | None = None,
        account_id: str = "primary",
        thread_id: str | None = None,
        recurrence: RecurrenceSpec | None = None,
        budget: RunBudget | None = None,
        retry_policy: AutomationRetryPolicy | None = None,
        max_runs: int | None = None,
        requires_approval: bool = False,
        idempotency_key: str | None = None,
        actor: str = "scheduler",
    ) -> ScheduledJob:
        next_run_at = _utc(next_run_at, "next_run_at")
        if recurrence is not None and interval_seconds is not None:
            raise ValueError("pass recurrence or interval_seconds, not both")
        selected_recurrence = recurrence or RecurrenceSpec(
            kind=(
                RecurrenceKind.INTERVAL
                if interval_seconds is not None
                else RecurrenceKind.ONCE
            ),
            interval_seconds=interval_seconds,
        )
        self.recurrence.validate(selected_recurrence, anchor=next_run_at)
        selected_budget = budget or RunBudget(scope=BudgetScope.JOB)
        if selected_budget.scope is not BudgetScope.JOB:
            raise ValueError("automation budget scope must be job")
        selected_retry = retry_policy or AutomationRetryPolicy()
        target = DeliveryTarget(
            adapter=adapter,
            account_id=account_id,
            destination_id=destination_id,
            thread_id=thread_id,
        )
        clean_prompt = _required(prompt, "prompt")
        if max_runs is not None and max_runs < 1:
            raise ValueError("max_runs must be positive")
        now = _utc_now()
        job_id = uuid4().hex
        status = (
            AutomationJobStatus.PENDING_APPROVAL
            if requires_approval
            else AutomationJobStatus.ACTIVE
        )
        key = idempotency_key.strip() if idempotency_key else None
        fingerprint = _fingerprint(
            {
                "adapter": target.adapter,
                "account_id": target.account_id,
                "destination_id": target.destination_id,
                "thread_id": target.thread_id,
                "prompt": clean_prompt,
                "recurrence": selected_recurrence.to_dict(),
            }
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if key:
                self._serialize_control_action(conn, key)
                existing = conn.execute(
                    """
                    SELECT job_id, fingerprint FROM automation_control_actions
                    WHERE profile_id = ? AND idempotency_key = ?
                    """,
                    (self.profile_id, key),
                ).fetchone()
                if existing is not None:
                    if str(existing["fingerprint"]) != fingerprint:
                        raise AutomationConflictError(
                            "idempotency key was already used with different input"
                        )
                    return self._get_in(conn, str(existing["job_id"]))
            conn.execute(
                """
                INSERT INTO automation_jobs (
                    id, profile_id, adapter, account_id, destination_id, thread_id,
                    prompt, recurrence_json, next_run_at, scheduled_for, status,
                    revision, run_count, max_runs, budget_json, retry_json,
                    requires_approval, approved_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    job_id,
                    self.profile_id,
                    target.adapter,
                    target.account_id,
                    target.destination_id,
                    target.thread_id,
                    clean_prompt,
                    _json(selected_recurrence.to_dict()),
                    _encode(next_run_at),
                    _encode(next_run_at),
                    status.value,
                    max_runs,
                    _json(selected_budget.to_dict()),
                    _json(selected_retry.to_dict()),
                    int(requires_approval),
                    _encode(now),
                    _encode(now),
                ),
            )
            self._event(
                conn,
                job_id,
                action="created",
                revision=0,
                actor=actor,
                metadata={"requires_approval": requires_approval},
                now=now,
            )
            if key:
                self._record_action(
                    conn,
                    key=key,
                    job_id=job_id,
                    action="create",
                    fingerprint=fingerprint,
                    revision=0,
                    now=now,
                )
            return self._get_in(conn, job_id)

    def get(self, job_id: str) -> ScheduledJob:
        with self._connect() as conn:
            return self._get_in(conn, job_id)

    def list(
        self,
        *,
        adapter: str | None = None,
        destination_id: str | None = None,
        status: AutomationJobStatus | str | None = None,
        limit: int = 100,
        include_terminal: bool = False,
    ) -> tuple[ScheduledJob, ...]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        clauses = ["profile_id = ?"]
        params: list[object] = [self.profile_id]
        if adapter is not None:
            clauses.append("adapter = ?")
            params.append(adapter)
        if destination_id is not None:
            clauses.append("destination_id = ?")
            params.append(destination_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(AutomationJobStatus(status).value)
        elif not include_terminal:
            clauses.append(
                "status IN ('pending_approval', 'active', 'running', 'paused')"
            )
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM automation_jobs
                WHERE {" AND ".join(clauses)}
                ORDER BY next_run_at, created_at, id
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return tuple(_row_to_job(row) for row in rows)

    def events(
        self, job_id: str, *, limit: int = 500
    ) -> tuple[AutomationJobEvent, ...]:
        self.get(job_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM automation_job_events
                WHERE profile_id = ? AND job_id = ?
                ORDER BY created_at, id LIMIT ?
                """,
                (self.profile_id, job_id, limit),
            ).fetchall()
        return tuple(_row_to_event(row) for row in rows)

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

    def delivery_history(
        self,
        run_id: str,
    ) -> tuple[AutomationDeliveryAttempt, ...]:
        self.get_run(run_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM automation_delivery_attempts
                WHERE profile_id = ? AND run_id = ?
                ORDER BY created_at, id
                """,
                (self.profile_id, run_id),
            ).fetchall()
        return tuple(_row_to_delivery_attempt(row) for row in rows)

    def pause(
        self,
        job_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        actor: str = "operator",
    ) -> ScheduledJob:
        return self._control(
            job_id,
            action="pause",
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            actor=actor,
            allowed={AutomationJobStatus.ACTIVE},
            status=AutomationJobStatus.PAUSED,
        )

    def resume(
        self,
        job_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        actor: str = "operator",
    ) -> ScheduledJob:
        return self._control(
            job_id,
            action="resume",
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            actor=actor,
            allowed={AutomationJobStatus.PAUSED},
            status=AutomationJobStatus.ACTIVE,
        )

    def approve(
        self,
        job_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        actor: str = "operator",
    ) -> ScheduledJob:
        return self._control(
            job_id,
            action="approve",
            expected_revision=expected_revision,
            idempotency_key=idempotency_key,
            actor=actor,
            allowed={AutomationJobStatus.PENDING_APPROVAL},
            status=AutomationJobStatus.ACTIVE,
            approved=True,
        )

    def cancel(
        self,
        job_id: str,
        *,
        adapter: str | None = None,
        destination_id: str | None = None,
        expected_revision: int | None = None,
        idempotency_key: str | None = None,
        actor: str = "operator",
    ) -> bool:
        """Cancel by owner profile, preserving the legacy destination-scoped API."""
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                self._serialize_control_action(conn, idempotency_key)
            self._serialize_job_mutation(conn, job_id)
            job = self._get_in(conn, job_id)
            if adapter is not None and job.adapter != adapter:
                return False
            if destination_id is not None and job.destination_id != destination_id:
                return False
            key = idempotency_key or f"legacy-cancel:{job_id}:{job.revision}"
            fingerprint = _fingerprint({"job_id": job_id})
            replay = self._action_replay(
                conn,
                key=key,
                job_id=job_id,
                action="cancel",
                fingerprint=fingerprint,
            )
            if replay is not None:
                return True
            if job.status in {
                AutomationJobStatus.CANCELLED,
                AutomationJobStatus.COMPLETED,
                AutomationJobStatus.EXPIRED,
            }:
                return False
            if expected_revision is not None and job.revision != expected_revision:
                raise AutomationConflictError(
                    f"expected revision {expected_revision}, found {job.revision}"
                )
            revision = job.revision + 1
            conn.execute(
                """
                UPDATE automation_jobs
                SET status = 'cancelled', revision = ?, claim_token = NULL,
                    lease_until = NULL, active_run_id = NULL, updated_at = ?
                WHERE profile_id = ? AND id = ?
                """,
                (revision, _encode(now), self.profile_id, job_id),
            )
            if job.active_run_id is not None:
                conn.execute(
                    """
                    UPDATE automation_runs
                    SET status = 'cancelled', claim_token = NULL, worker_id = NULL,
                        lease_until = NULL, finished_at = ?, updated_at = ?
                    WHERE profile_id = ? AND id = ?
                    """,
                    (_encode(now), _encode(now), self.profile_id, job.active_run_id),
                )
            conn.execute(
                """
                UPDATE automation_run_requests SET status = 'cancelled', updated_at = ?
                WHERE profile_id = ? AND job_id = ? AND status = 'pending'
                """,
                (_encode(now), self.profile_id, job_id),
            )
            self._event(conn, job_id, "cancelled", revision, actor, now=now)
            self._record_action(
                conn,
                key=key,
                job_id=job_id,
                action="cancel",
                fingerprint=fingerprint,
                revision=revision,
                now=now,
            )
        return True

    def update(
        self,
        job_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        prompt: str | None = None,
        recurrence: RecurrenceSpec | None = None,
        next_run_at: datetime | None = None,
        budget: RunBudget | None = None,
        retry_policy: AutomationRetryPolicy | None = None,
        max_runs: int | None = None,
        actor: str = "operator",
    ) -> ScheduledJob:
        now = _utc_now()
        fingerprint = _fingerprint(
            {
                "job_id": job_id,
                "prompt": prompt,
                "recurrence": recurrence.to_dict() if recurrence else None,
                "next_run_at": next_run_at.isoformat() if next_run_at else None,
                "budget": budget.to_dict() if budget else None,
                "retry": retry_policy.to_dict() if retry_policy else None,
                "max_runs": max_runs,
            }
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_control_action(conn, idempotency_key)
            self._serialize_job_mutation(conn, job_id)
            current = self._get_in(conn, job_id)
            replay = self._action_replay(
                conn,
                key=idempotency_key,
                job_id=job_id,
                action="update",
                fingerprint=fingerprint,
            )
            if replay is not None:
                return current
            if current.revision != expected_revision:
                raise AutomationConflictError(
                    f"expected revision {expected_revision}, found {current.revision}"
                )
            if current.status is AutomationJobStatus.RUNNING:
                raise AutomationConflictError("cannot update a running automation")
            selected_prompt = (
                _required(prompt, "prompt") if prompt is not None else current.prompt
            )
            selected_recurrence = recurrence or current.recurrence
            selected_next = (
                _utc(next_run_at, "next_run_at")
                if next_run_at is not None
                else current.next_run_at
            )
            selected_budget = budget or current.budget
            selected_retry = retry_policy or current.retry_policy
            selected_max_runs = current.max_runs if max_runs is None else max_runs
            if selected_budget.scope is not BudgetScope.JOB:
                raise ValueError("automation budget scope must be job")
            self.recurrence.validate(selected_recurrence, anchor=selected_next)
            revision = current.revision + 1
            conn.execute(
                """
                UPDATE automation_jobs
                SET prompt = ?, recurrence_json = ?, next_run_at = ?,
                    scheduled_for = ?, budget_json = ?, retry_json = ?,
                    max_runs = ?, revision = ?, updated_at = ?
                WHERE profile_id = ? AND id = ?
                """,
                (
                    selected_prompt,
                    _json(selected_recurrence.to_dict()),
                    _encode(selected_next),
                    _encode(selected_next),
                    _json(selected_budget.to_dict()),
                    _json(selected_retry.to_dict()),
                    selected_max_runs,
                    revision,
                    _encode(now),
                    self.profile_id,
                    job_id,
                ),
            )
            self._event(conn, job_id, "updated", revision, actor, now=now)
            self._record_action(
                conn,
                key=idempotency_key,
                job_id=job_id,
                action="update",
                fingerprint=fingerprint,
                revision=revision,
                now=now,
            )
            return self._get_in(conn, job_id)

    def run_now(
        self,
        job_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        actor: str = "operator",
        now: datetime | None = None,
    ) -> ScheduledJob:
        observed = _utc(now or _utc_now(), "now")
        fingerprint = _fingerprint({"job_id": job_id, "action": "run_now"})
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_control_action(conn, idempotency_key)
            self._serialize_job_mutation(conn, job_id)
            job = self._get_in(conn, job_id)
            replay = self._action_replay(
                conn,
                key=idempotency_key,
                job_id=job_id,
                action="run_now",
                fingerprint=fingerprint,
            )
            if replay is not None:
                return job
            if job.revision != expected_revision:
                raise AutomationConflictError(
                    f"expected revision {expected_revision}, found {job.revision}"
                )
            if job.status not in {
                AutomationJobStatus.ACTIVE,
            }:
                raise AutomationConflictError(
                    f"cannot run automation in {job.status.value} state"
                )
            self._enqueue_request(
                conn,
                job_id=job_id,
                reason=AutomationRunReason.MANUAL,
                occurrence_at=observed,
                idempotency_key=f"manual:{idempotency_key}",
                now=observed,
            )
            revision = job.revision + 1
            conn.execute(
                """
                UPDATE automation_jobs SET revision = ?, updated_at = ?
                WHERE profile_id = ? AND id = ?
                """,
                (revision, _encode(observed), self.profile_id, job_id),
            )
            self._event(
                conn, job_id, "run_now_requested", revision, actor, now=observed
            )
            self._record_action(
                conn,
                key=idempotency_key,
                job_id=job_id,
                action="run_now",
                fingerprint=fingerprint,
                revision=revision,
                now=observed,
            )
            return self._get_in(conn, job_id)

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

    def mark_delivery(
        self,
        run_id: str,
        state: AutomationDeliveryState,
        *,
        error: str | None = None,
    ) -> AutomationRun:
        selected = AutomationDeliveryState(state)
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE automation_runs SET delivery_state = ?, delivery_error = ?,
                    updated_at = ? WHERE profile_id = ? AND id = ?
                """,
                (
                    selected.value,
                    error[:500] if error else None,
                    _encode(now),
                    self.profile_id,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise AutomationNotFoundError(f"Automation run not found: {run_id}")
            self._delivery_attempt(
                conn,
                run_id=run_id,
                state=selected,
                error=error,
                now=now,
            )
            return self._run_in(conn, run_id)

    def create_webhook_trigger(
        self,
        job_id: str,
    ) -> tuple[AutomationTrigger, str]:
        token = secrets.token_urlsafe(32)
        return self._create_trigger(
            job_id,
            kind=TriggerKind.WEBHOOK,
            secret_digest=sha256(token.encode()).hexdigest(),
        ), token

    def create_completion_trigger(
        self,
        job_id: str,
        *,
        kind: TriggerKind,
        source_resource_id: str,
    ) -> AutomationTrigger:
        selected = TriggerKind(kind)
        if selected is TriggerKind.WEBHOOK:
            raise ValueError("use create_webhook_trigger for webhook triggers")
        return self._create_trigger(
            job_id,
            kind=selected,
            source_resource_id=_required(source_resource_id, "source_resource_id"),
        )

    def triggers(self, job_id: str | None = None) -> tuple[AutomationTrigger, ...]:
        clauses = ["profile_id = ?"]
        params: list[object] = [self.profile_id]
        if job_id is not None:
            self.get(job_id)
            clauses.append("job_id = ?")
            params.append(job_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM automation_triggers
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at, id
                """,
                tuple(params),
            ).fetchall()
        return tuple(_row_to_trigger(row) for row in rows)

    def ingest_webhook(
        self,
        trigger_id: str,
        *,
        token: str,
        event_id: str,
        payload: Mapping[str, Any],
        occurred_at: datetime | None = None,
    ) -> TriggerEnvelope:
        observed = _utc(occurred_at or _utc_now(), "occurred_at")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM automation_triggers
                WHERE profile_id = ? AND id = ? AND kind = 'webhook' AND enabled = 1
                """,
                (self.profile_id, trigger_id),
            ).fetchone()
            if row is None:
                raise AutomationNotFoundError(
                    f"Automation webhook trigger not found: {trigger_id}"
                )
            digest = sha256(token.encode()).hexdigest()
            if not hmac.compare_digest(digest, str(row["secret_digest"])):
                raise PermissionError("invalid webhook trigger credential")
        return self._ingest_trigger(
            _row_to_trigger(row),
            event_id=event_id,
            payload=payload,
            trust=TriggerTrust.TRUSTED,
            occurred_at=observed,
        )

    def emit_completion(
        self,
        *,
        kind: TriggerKind,
        source_resource_id: str,
        source_event_id: str,
        payload: Mapping[str, Any],
        occurred_at: datetime | None = None,
    ) -> tuple[TriggerEnvelope, ...]:
        selected = TriggerKind(kind)
        if selected is TriggerKind.WEBHOOK:
            raise ValueError("completion kind cannot be webhook")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM automation_triggers
                WHERE profile_id = ? AND kind = ? AND source_resource_id = ?
                  AND enabled = 1
                ORDER BY created_at, id
                """,
                (self.profile_id, selected.value, source_resource_id),
            ).fetchall()
        return tuple(
            self._ingest_trigger(
                _row_to_trigger(row),
                event_id=source_event_id,
                payload=payload,
                trust=TriggerTrust.OWNER,
                occurred_at=_utc(occurred_at or _utc_now(), "occurred_at"),
            )
            for row in rows
        )

    def _ingest_trigger(
        self,
        trigger: AutomationTrigger,
        *,
        event_id: str,
        payload: Mapping[str, Any],
        trust: TriggerTrust,
        occurred_at: datetime,
    ) -> TriggerEnvelope:
        envelope = TriggerEnvelope(
            id=uuid4().hex,
            profile_id=self.profile_id,
            trigger_id=trigger.id,
            trust=trust,
            payload=payload,
            occurred_at=occurred_at,
            source_event_id=_required(event_id, "event_id"),
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_trigger_ingest(conn, trigger.id)
            existing = conn.execute(
                """
                SELECT * FROM automation_trigger_events
                WHERE profile_id = ? AND trigger_id = ? AND source_event_id = ?
                """,
                (self.profile_id, trigger.id, envelope.source_event_id),
            ).fetchone()
            if existing is not None:
                return _row_to_envelope(existing)
            conn.execute(
                """
                INSERT INTO automation_trigger_events (
                    id, profile_id, trigger_id, source_event_id, trust,
                    payload_json, occurred_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.id,
                    self.profile_id,
                    trigger.id,
                    envelope.source_event_id,
                    envelope.trust.value,
                    _json(dict(envelope.payload)),
                    _encode(envelope.occurred_at),
                    _encode(_utc_now()),
                ),
            )
            self._enqueue_request(
                conn,
                job_id=trigger.job_id,
                reason=AutomationRunReason.TRIGGER,
                occurrence_at=envelope.occurred_at,
                trigger_event_id=envelope.id,
                idempotency_key=f"trigger:{trigger.id}:{envelope.source_event_id}",
                now=_utc_now(),
            )
        return envelope

    def _create_trigger(
        self,
        job_id: str,
        *,
        kind: TriggerKind,
        source_resource_id: str | None = None,
        secret_digest: str | None = None,
    ) -> AutomationTrigger:
        self.get(job_id)
        trigger_id = uuid4().hex
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_job_mutation(conn, job_id)
            job = self._get_in(conn, job_id)
            if job.status is AutomationJobStatus.CANCELLED:
                raise AutomationConflictError(
                    "cannot attach a trigger to a cancelled automation"
                )
            conn.execute(
                """
                INSERT INTO automation_triggers (
                    id, profile_id, job_id, kind, source_resource_id,
                    secret_digest, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    trigger_id,
                    self.profile_id,
                    job_id,
                    TriggerKind(kind).value,
                    source_resource_id,
                    secret_digest,
                    _encode(now),
                    _encode(now),
                ),
            )
            revision = job.revision + 1
            conn.execute(
                """
                UPDATE automation_jobs
                SET status = CASE
                        WHEN status IN ('completed', 'expired') THEN 'active'
                        ELSE status
                    END,
                    revision = ?, updated_at = ?
                WHERE profile_id = ? AND id = ?
                """,
                (revision, _encode(now), self.profile_id, job_id),
            )
            self._event(
                conn,
                job_id,
                "trigger_created",
                revision,
                "operator",
                metadata={"trigger_id": trigger_id, "kind": TriggerKind(kind).value},
                now=now,
            )
            row = conn.execute(
                "SELECT * FROM automation_triggers WHERE id = ?",
                (trigger_id,),
            ).fetchone()
        assert row is not None
        return _row_to_trigger(row)

    def _control(
        self,
        job_id: str,
        *,
        action: str,
        expected_revision: int,
        idempotency_key: str,
        actor: str,
        allowed: set[AutomationJobStatus],
        status: AutomationJobStatus,
        approved: bool = False,
    ) -> ScheduledJob:
        now = _utc_now()
        fingerprint = _fingerprint({"job_id": job_id, "action": action})
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_control_action(conn, idempotency_key)
            self._serialize_job_mutation(conn, job_id)
            current = self._get_in(conn, job_id)
            replay = self._action_replay(
                conn,
                key=idempotency_key,
                job_id=job_id,
                action=action,
                fingerprint=fingerprint,
            )
            if replay is not None:
                return current
            if current.revision != expected_revision:
                raise AutomationConflictError(
                    f"expected revision {expected_revision}, found {current.revision}"
                )
            if current.status not in allowed:
                raise AutomationConflictError(
                    f"cannot {action} automation in {current.status.value} state"
                )
            revision = current.revision + 1
            conn.execute(
                """
                UPDATE automation_jobs SET status = ?, revision = ?,
                    approved_at = CASE WHEN ? THEN ? ELSE approved_at END,
                    updated_at = ? WHERE profile_id = ? AND id = ?
                """,
                (
                    status.value,
                    revision,
                    int(approved),
                    _encode(now),
                    _encode(now),
                    self.profile_id,
                    job_id,
                ),
            )
            self._event(conn, job_id, action, revision, actor, now=now)
            self._record_action(
                conn,
                key=idempotency_key,
                job_id=job_id,
                action=action,
                fingerprint=fingerprint,
                revision=revision,
                now=now,
            )
            return self._get_in(conn, job_id)

    def _get_in(self, conn: sqlite3.Connection, job_id: str) -> ScheduledJob:
        row = conn.execute(
            "SELECT * FROM automation_jobs WHERE profile_id = ? AND id = ?",
            (self.profile_id, job_id),
        ).fetchone()
        if row is None:
            raise AutomationNotFoundError(f"Scheduled job not found: {job_id}")
        return _row_to_job(row)

    def _run_in(self, conn: sqlite3.Connection, run_id: str) -> AutomationRun:
        row = conn.execute(
            "SELECT * FROM automation_runs WHERE profile_id = ? AND id = ?",
            (self.profile_id, run_id),
        ).fetchone()
        if row is None:
            raise AutomationNotFoundError(f"Automation run not found: {run_id}")
        return _row_to_run(row)

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

    def _event(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        action: str,
        revision: int,
        actor: str,
        *,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO automation_job_events (
                id, job_id, profile_id, action, revision, actor, run_id,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                job_id,
                self.profile_id,
                action,
                revision,
                actor,
                run_id,
                _json(dict(metadata or {})),
                _encode(now),
            ),
        )

    def _delivery_attempt(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        state: AutomationDeliveryState,
        now: datetime,
        error: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO automation_delivery_attempts (
                id, run_id, profile_id, state, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                run_id,
                self.profile_id,
                state.value,
                error[:500] if error else None,
                _encode(now),
            ),
        )

    def _record_action(
        self,
        conn: sqlite3.Connection,
        *,
        key: str,
        job_id: str,
        action: str,
        fingerprint: str,
        revision: int,
        now: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO automation_control_actions (
                profile_id, idempotency_key, job_id, action, fingerprint,
                result_revision, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.profile_id,
                _required(key, "idempotency_key"),
                job_id,
                action,
                fingerprint,
                revision,
                _encode(now),
            ),
        )

    def _action_replay(
        self,
        conn: sqlite3.Connection,
        *,
        key: str,
        job_id: str,
        action: str,
        fingerprint: str,
    ) -> sqlite3.Row | None:
        row = conn.execute(
            """
            SELECT * FROM automation_control_actions
            WHERE profile_id = ? AND idempotency_key = ?
            """,
            (self.profile_id, _required(key, "idempotency_key")),
        ).fetchone()
        if row is None:
            return None
        if (
            str(row["job_id"]) != job_id
            or str(row["action"]) != action
            or str(row["fingerprint"]) != fingerprint
        ):
            raise AutomationConflictError(
                "idempotency key was already used for a different action"
            )
        return row

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

    def _has_enabled_triggers(self, conn: sqlite3.Connection, job_id: str) -> bool:
        return (
            conn.execute(
                """
                SELECT 1 FROM automation_triggers
                WHERE profile_id = ? AND job_id = ? AND enabled = 1 LIMIT 1
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


def _row_to_job(row: sqlite3.Row) -> ScheduledJob:
    recurrence_data = json.loads(str(row["recurrence_json"]))
    return ScheduledJob(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        target=DeliveryTarget(
            adapter=str(row["adapter"]),
            account_id=str(row["account_id"]),
            destination_id=str(row["destination_id"]),
            thread_id=str(row["thread_id"]) if row["thread_id"] else None,
        ),
        prompt=str(row["prompt"]),
        recurrence=_recurrence_from_dict(recurrence_data),
        next_run_at=_decode(str(row["next_run_at"])),
        scheduled_for=_decode(str(row["scheduled_for"])),
        status=AutomationJobStatus(str(row["status"])),
        budget=_budget_from_dict(json.loads(str(row["budget_json"]))),
        retry_policy=_retry_from_dict(json.loads(str(row["retry_json"]))),
        revision=int(row["revision"]),
        run_count=int(row["run_count"]),
        max_runs=int(row["max_runs"]) if row["max_runs"] is not None else None,
        requires_approval=bool(row["requires_approval"]),
        approved_at=_optional_datetime(row["approved_at"]),
        claim_token=str(row["claim_token"]) if row["claim_token"] else None,
        lease_until=_optional_datetime(row["lease_until"]),
        active_run_id=str(row["active_run_id"]) if row["active_run_id"] else None,
        last_run_at=_optional_datetime(row["last_run_at"]),
        last_error=str(row["last_error"]) if row["last_error"] else None,
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
    )


def _row_to_run(row: sqlite3.Row) -> AutomationRun:
    return AutomationRun(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        profile_id=str(row["profile_id"]),
        occurrence_at=_decode(str(row["occurrence_at"])),
        reason=AutomationRunReason(str(row["reason"])),
        status=AutomationRunStatus(str(row["status"])),
        attempt=int(row["attempt"]),
        claim_token=str(row["claim_token"]) if row["claim_token"] else None,
        worker_id=str(row["worker_id"]) if row["worker_id"] else None,
        lease_until=_optional_datetime(row["lease_until"]),
        started_at=_optional_datetime(row["started_at"]),
        finished_at=_optional_datetime(row["finished_at"]),
        result=json.loads(str(row["result_json"])),
        trace_id=str(row["trace_id"]) if row["trace_id"] else None,
        usage=json.loads(str(row["usage_json"])),
        cost=json.loads(str(row["cost_json"])),
        error=str(row["error"]) if row["error"] else None,
        artifact_refs=tuple(json.loads(str(row["artifact_refs_json"]))),
        delivery_state=AutomationDeliveryState(str(row["delivery_state"])),
        delivery_error=str(row["delivery_error"]) if row["delivery_error"] else None,
        trigger_event_id=(
            str(row["trigger_event_id"]) if row["trigger_event_id"] else None
        ),
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
    )


def _row_to_delivery_attempt(row: sqlite3.Row) -> AutomationDeliveryAttempt:
    return AutomationDeliveryAttempt(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        profile_id=str(row["profile_id"]),
        state=AutomationDeliveryState(str(row["state"])),
        error=str(row["error"]) if row["error"] else None,
        created_at=_decode(str(row["created_at"])),
    )


def _row_to_event(row: sqlite3.Row) -> AutomationJobEvent:
    return AutomationJobEvent(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        profile_id=str(row["profile_id"]),
        action=str(row["action"]),
        revision=int(row["revision"]),
        actor=str(row["actor"]),
        run_id=str(row["run_id"]) if row["run_id"] else None,
        metadata=json.loads(str(row["metadata_json"])),
        created_at=_decode(str(row["created_at"])),
    )


def _row_to_trigger(row: sqlite3.Row) -> AutomationTrigger:
    return AutomationTrigger(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        job_id=str(row["job_id"]),
        kind=TriggerKind(str(row["kind"])),
        source_resource_id=(
            str(row["source_resource_id"]) if row["source_resource_id"] else None
        ),
        secret_digest=str(row["secret_digest"]) if row["secret_digest"] else None,
        enabled=bool(row["enabled"]),
        created_at=_decode(str(row["created_at"])),
    )


def _row_to_envelope(row: sqlite3.Row) -> TriggerEnvelope:
    return TriggerEnvelope(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        trigger_id=str(row["trigger_id"]),
        trust=TriggerTrust(str(row["trust"])),
        payload=json.loads(str(row["payload_json"])),
        occurred_at=_decode(str(row["occurred_at"])),
        source_event_id=(
            str(row["source_event_id"]) if row["source_event_id"] else None
        ),
    )


def _recurrence_from_dict(value: Mapping[str, Any]) -> RecurrenceSpec:
    return RecurrenceSpec(
        kind=RecurrenceKind(str(value.get("kind", "once"))),
        timezone_name=str(value.get("timezone", "UTC")),
        interval_seconds=_optional_int(value.get("interval_seconds")),
        cron=_optional_string(value.get("cron")),
        rrule=_optional_string(value.get("rrule")),
        starts_at=_optional_timestamp(value.get("starts_at")),
        ends_at=_optional_timestamp(value.get("ends_at")),
        misfire_policy=MisfirePolicy(str(value.get("misfire_policy", "run_once"))),
        nonexistent_time_policy=NonexistentTimePolicy(
            str(value.get("nonexistent_time_policy", "shift_forward"))
        ),
        ambiguous_time_policy=AmbiguousTimePolicy(
            str(value.get("ambiguous_time_policy", "earliest"))
        ),
        misfire_grace_seconds=int(value.get("misfire_grace_seconds", 60)),
        max_catch_up=int(value.get("max_catch_up", 1)),
        jitter_seconds=int(value.get("jitter_seconds", 0)),
    )


def _retry_from_dict(value: Mapping[str, Any]) -> AutomationRetryPolicy:
    return AutomationRetryPolicy(
        max_attempts=int(value.get("max_attempts", 3)),
        initial_backoff_seconds=int(value.get("initial_backoff_seconds", 60)),
        max_backoff_seconds=int(value.get("max_backoff_seconds", 3_600)),
        multiplier=float(value.get("multiplier", 2.0)),
    )


def _budget_from_dict(value: Mapping[str, Any]) -> RunBudget:
    raw_cost = value.get("max_cost")
    cost = None
    if isinstance(raw_cost, Mapping) and raw_cost.get("amount") is not None:
        cost = ExactCost(
            Decimal(str(raw_cost["amount"])),
            currency=str(raw_cost.get("currency", "USD")),
            pricing_known=bool(raw_cost.get("pricing_known", True)),
            estimated=bool(raw_cost.get("estimated", False)),
            reported=bool(raw_cost.get("reported", False)),
        )
    return RunBudget(
        scope=BudgetScope(str(value.get("scope", "job"))),
        max_model_calls=_optional_int(value.get("max_model_calls")),
        max_tool_calls=_optional_int(value.get("max_tool_calls")),
        max_tokens=_optional_int(value.get("max_tokens")),
        max_cost=cost,
        deadline=_optional_timestamp(value.get("deadline")),
        unknown_cost_policy=UnknownCostPolicy(
            str(value.get("unknown_cost_policy", "fail_closed"))
        ),
    )


def _optional_timestamp(value: object) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value else None


def _optional_datetime(value: object) -> datetime | None:
    return _decode(str(value)) if value else None


def _optional_string(value: object) -> str | None:
    clean = str(value).strip() if value is not None else ""
    return clean or None


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None


def _required(value: str | None, label: str) -> str:
    clean = value.strip() if value is not None else ""
    if not clean:
        raise ValueError(f"{label} is required")
    return clean


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(_json(dict(value)).encode()).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _encode(value: datetime | None) -> str:
    return value.astimezone(timezone.utc).isoformat() if value is not None else ""


def _decode(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


__all__ = [
    "AutomationConflictError",
    "AutomationNotFoundError",
    "DEFAULT_AUTOMATION_LEASE_SECONDS",
    "SQLiteScheduleStore",
]
