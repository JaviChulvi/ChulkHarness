"""Job lifecycle operations for the scheduling store."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from chulk.gateway import DeliveryTarget
from chulk.scheduling._store_support import (
    AutomationConflictError,
    _ScheduleStoreMixin,
    _encode,
    _fingerprint,
    _json,
    _required,
    _row_to_event,
    _row_to_job,
    _utc,
    _utc_now,
)
from chulk.scheduling.models import (
    AutomationJobEvent,
    AutomationJobStatus,
    AutomationRetryPolicy,
    AutomationRunReason,
    RecurrenceKind,
    RecurrenceSpec,
    ScheduledJob,
)
from chulk.usage import BudgetScope, RunBudget


class _ScheduleJobsMixin(_ScheduleStoreMixin):
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
