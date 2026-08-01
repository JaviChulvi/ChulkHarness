"""Transactional persistence for profile-owned child-task graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.children.models import (
    ChildCompletionDelivery,
    ChildDeliveryStatus,
    ChildTask,
    ChildTaskClaim,
    ChildTaskEvent,
    ChildTaskResult,
    ChildTaskRole,
    ChildTaskStatus,
    child_result_from_dict,
    child_task_from_dict,
)
from chulk.children.transitions import (
    block_task,
    cancel_task,
    complete_task,
    exhaust_budget,
    fail_task,
    mark_ready,
    mark_unknown,
    request_cancellation,
    retry_task,
    start_attempt,
)
from chulk.redaction import redact_data
from chulk.storage import initialize_sqlite_database, sqlite_connection


DEFAULT_CHILD_LEASE_SECONDS = 120


class ChildTaskNotFoundError(LookupError):
    """Raised when a task is absent from the selected profile."""


class ChildTaskRevisionConflictError(RuntimeError):
    """Raised when a task mutation uses a stale revision."""

    def __init__(self, task_id: str, expected: int, actual: int) -> None:
        self.task_id = task_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"child task {task_id!r} revision conflict: "
            f"expected {expected}, actual {actual}"
        )


class ChildTaskLeaseConflictError(RuntimeError):
    """Raised when an attempt lease is absent, expired, or owned elsewhere."""


class ChildTaskConflictError(RuntimeError):
    """Raised when an idempotency key is reused for different child work."""


class ChildDeliveryConflictError(RuntimeError):
    """Raised when a completion delivery claim cannot be changed safely."""


class ChildTaskStore:
    """Durable task graph with revision CAS, leases, and completion delivery."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        profile_id: str = "default",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.profile_id = profile_id.strip()
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        initialize_sqlite_database(self.db_path)

    def create(
        self,
        task: ChildTask,
        *,
        actor: str = "parent",
        idempotency_key: str | None = None,
    ) -> ChildTask:
        """Insert a child after validating lineage and dependency ownership."""
        if task.profile_id != self.profile_id:
            raise ValueError("child task profile does not match store profile")
        if task.revision != 0:
            raise ValueError("new child task revision must be zero")
        if task.status is not ChildTaskStatus.PENDING:
            raise ValueError("new child task must start pending")
        clean_key = _optional(idempotency_key)
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if clean_key is not None:
                row = conn.execute(
                    """
                    SELECT task_id FROM child_task_creation_keys
                    WHERE profile_id = ? AND idempotency_key = ?
                    """,
                    (self.profile_id, clean_key),
                ).fetchone()
                if row is not None:
                    existing = _task_from_row(
                        _task_row(conn, str(row["task_id"]), self.profile_id)
                    )
                    if _creation_payload(existing) != _creation_payload(
                        _redacted_task(task)
                    ):
                        raise ChildTaskConflictError(
                            "child task idempotency key was reused with "
                            "different work"
                        )
                    return existing
            _validate_lineage(conn, task)
            _validate_goal_ownership(conn, task)
            dependency_statuses = _dependency_statuses(
                conn,
                task.dependency_ids,
                self.profile_id,
            )
            initial = _initial_state(task, dependency_statuses)
            stored = _redacted_task(initial)
            try:
                _insert_task(conn, stored)
                conn.executemany(
                    """
                    INSERT INTO child_task_dependencies (
                        task_id, dependency_id, profile_id
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        (stored.id, dependency_id, self.profile_id)
                        for dependency_id in stored.dependency_ids
                    ),
                )
                if clean_key is not None:
                    conn.execute(
                        """
                        INSERT INTO child_task_creation_keys (
                            profile_id, idempotency_key, task_id
                        ) VALUES (?, ?, ?)
                        """,
                        (self.profile_id, clean_key, stored.id),
                    )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"child task {stored.id!r} already exists") from exc
            _insert_event(
                conn,
                stored,
                kind="child.created",
                actor=actor,
                payload={
                    "parent_task_id": stored.lineage.parent_task_id,
                    "dependency_ids": list(stored.dependency_ids),
                    "role": stored.spec.role.value,
                },
                now=stored.created_at,
            )
        return stored

    def get(self, task_id: str) -> ChildTask:
        with sqlite_connection(self.db_path) as conn:
            return _task_from_row(_task_row(conn, task_id, self.profile_id))

    def list(
        self,
        *,
        status: ChildTaskStatus | str | None = None,
        goal_id: str | None = None,
        parent_task_id: str | None = None,
        limit: int = 100,
    ) -> tuple[ChildTask, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("child task list limit must be between 1 and 1000")
        clauses = ["profile_id = ?"]
        values: list[Any] = [self.profile_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(ChildTaskStatus(status).value)
        if goal_id is not None:
            clauses.append("goal_id = ?")
            values.append(goal_id)
        if parent_task_id is not None:
            clauses.append("parent_task_id = ?")
            values.append(parent_task_id)
        values.append(limit)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM child_tasks
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at, id
                LIMIT ?
                """,
                tuple(values),
            ).fetchall()
        return tuple(_task_from_row(row) for row in rows)

    def active_count(self) -> int:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count FROM child_tasks
                WHERE profile_id = ?
                  AND status IN (
                      'pending', 'ready', 'running', 'waiting', 'blocked'
                  )
                """,
                (self.profile_id,),
            ).fetchone()
        return int(row["count"])

    def events(
        self,
        task_id: str,
        *,
        after_revision: int = -1,
    ) -> tuple[ChildTaskEvent, ...]:
        self.get(task_id)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM child_task_events
                WHERE profile_id = ? AND task_id = ? AND revision > ?
                ORDER BY revision
                """,
                (self.profile_id, task_id, after_revision),
            ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def attempts(self, task_id: str) -> tuple[Mapping[str, Any], ...]:
        self.get(task_id)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM child_task_attempts
                WHERE profile_id = ? AND task_id = ?
                ORDER BY attempt_number
                """,
                (self.profile_id, task_id),
            ).fetchall()
        return tuple(_attempt_dict(row) for row in rows)

    def claim(
        self,
        task_id: str,
        *,
        expected_revision: int,
        worker_id: str,
        lease_seconds: int = DEFAULT_CHILD_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> ChildTaskClaim:
        """Claim a ready task and persist attempt intent before worker execution."""
        clean_worker = _required(worker_id, "worker id")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        observed = self._now(now)
        if self._expire_deadline_if_due(
            task_id,
            expected_revision=expected_revision,
            actor=clean_worker,
            now=observed,
        ):
            raise ChildTaskLeaseConflictError(
                "child task deadline expired before execution"
            )
        lease_until = observed + timedelta(seconds=lease_seconds)
        attempt_id = uuid4().hex
        claim_token = uuid4().hex
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = _task_from_row(_task_row(conn, task_id, self.profile_id))
            _check_revision(current, expected_revision)
            dependency_statuses = _dependency_statuses(
                conn,
                current.dependency_ids,
                self.profile_id,
            )
            if any(
                status is not ChildTaskStatus.COMPLETED
                for status in dependency_statuses.values()
            ):
                raise ChildTaskLeaseConflictError(
                    "child task dependencies are not completed"
                )
            _assert_parent_parallelism(conn, current)
            changed = start_attempt(current).with_revision(
                current.revision + 1,
                now=observed,
            )
            changed = _redacted_task(changed)
            cursor = conn.execute(
                """
                UPDATE child_tasks
                SET status = ?, revision = ?, snapshot_json = ?,
                    claim_token = ?, worker_id = ?, attempt_id = ?,
                    lease_until = ?, updated_at = ?
                WHERE id = ? AND profile_id = ? AND revision = ?
                  AND status = 'ready' AND cancellation_requested = 0
                  AND claim_token IS NULL
                """,
                (
                    changed.status.value,
                    changed.revision,
                    _json(changed.to_dict()),
                    claim_token,
                    clean_worker,
                    attempt_id,
                    lease_until.isoformat(),
                    observed.isoformat(),
                    task_id,
                    self.profile_id,
                    current.revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ChildTaskLeaseConflictError(
                    "child task claim changed concurrently"
                )
            conn.execute(
                """
                INSERT INTO child_task_attempts (
                    id, task_id, profile_id, attempt_number, worker_id,
                    claim_token, status, lease_until, result_json, error,
                    started_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, NULL, NULL, ?, ?, NULL)
                """,
                (
                    attempt_id,
                    task_id,
                    self.profile_id,
                    changed.attempt_count,
                    clean_worker,
                    claim_token,
                    lease_until.isoformat(),
                    observed.isoformat(),
                    observed.isoformat(),
                ),
            )
            _insert_event(
                conn,
                changed,
                kind="child.attempt_started",
                actor=clean_worker,
                payload={
                    "attempt_id": attempt_id,
                    "attempt_number": changed.attempt_count,
                },
                now=observed,
            )
        return ChildTaskClaim(
            task_id=task_id,
            profile_id=self.profile_id,
            attempt_id=attempt_id,
            attempt_number=changed.attempt_count,
            worker_id=clean_worker,
            claim_token=claim_token,
            lease_until=lease_until,
        )

    def claim_next(
        self,
        *,
        worker_id: str,
        role: ChildTaskRole | str | None = None,
        lease_seconds: int = DEFAULT_CHILD_LEASE_SECONDS,
    ) -> ChildTaskClaim | None:
        """Claim the oldest ready task, retrying only a concurrent claim race."""
        candidates = self.list(status=ChildTaskStatus.READY, limit=1000)
        if role is not None:
            selected_role = ChildTaskRole(role)
            candidates = tuple(
                task for task in candidates if task.spec.role is selected_role
            )
        for task in candidates:
            try:
                return self.claim(
                    task.id,
                    expected_revision=task.revision,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                )
            except (ChildTaskRevisionConflictError, ChildTaskLeaseConflictError):
                continue
        return None

    def heartbeat(
        self,
        claim: ChildTaskClaim,
        *,
        lease_seconds: int = DEFAULT_CHILD_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> ChildTaskClaim:
        if claim.profile_id != self.profile_id:
            raise ChildTaskLeaseConflictError("claim belongs to another profile")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        observed = self._now(now)
        lease_until = observed + timedelta(seconds=lease_seconds)
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE child_tasks
                SET lease_until = ?, updated_at = ?
                WHERE id = ? AND profile_id = ? AND status = 'running'
                  AND cancellation_requested = 0
                  AND attempt_id = ? AND worker_id = ? AND claim_token = ?
                  AND lease_until >= ?
                """,
                (
                    lease_until.isoformat(),
                    observed.isoformat(),
                    claim.task_id,
                    self.profile_id,
                    claim.attempt_id,
                    claim.worker_id,
                    claim.claim_token,
                    observed.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                raise ChildTaskLeaseConflictError(
                    "child lease is absent, expired, terminal, or cancelled"
                )
            conn.execute(
                """
                UPDATE child_task_attempts
                SET lease_until = ?, updated_at = ?
                WHERE id = ? AND task_id = ? AND profile_id = ?
                  AND status = 'running' AND claim_token = ?
                """,
                (
                    lease_until.isoformat(),
                    observed.isoformat(),
                    claim.attempt_id,
                    claim.task_id,
                    self.profile_id,
                    claim.claim_token,
                ),
            )
        return replace(claim, lease_until=lease_until)

    def assert_attempt_boundary(
        self,
        claim: ChildTaskClaim,
        *,
        now: datetime | None = None,
    ) -> ChildTask:
        observed = self._now(now)
        with sqlite_connection(self.db_path) as conn:
            return _assert_claim(conn, claim, self.profile_id, observed)

    def complete(
        self,
        claim: ChildTaskClaim,
        result: ChildTaskResult,
        *,
        actor: str | None = None,
        now: datetime | None = None,
    ) -> ChildTask:
        return self._finish_attempt(
            claim,
            ChildTaskStatus.COMPLETED,
            result=result,
            reason=None,
            actor=actor or claim.worker_id,
            now=now,
        )

    def fail(
        self,
        claim: ChildTaskClaim,
        reason: str,
        *,
        actor: str | None = None,
        now: datetime | None = None,
    ) -> ChildTask:
        return self._finish_attempt(
            claim,
            ChildTaskStatus.FAILED,
            result=None,
            reason=reason,
            actor=actor or claim.worker_id,
            now=now,
        )

    def exhaust_budget(
        self,
        claim: ChildTaskClaim,
        reason: str,
        *,
        actor: str | None = None,
        now: datetime | None = None,
    ) -> ChildTask:
        return self._finish_attempt(
            claim,
            ChildTaskStatus.BUDGET_EXHAUSTED,
            result=None,
            reason=reason,
            actor=actor or claim.worker_id,
            now=now,
        )

    def request_cancel(
        self,
        task_id: str,
        *,
        expected_revision: int,
        actor: str,
        reason: str = "Cancellation requested by parent.",
    ) -> tuple[ChildTask, ...]:
        """Persist cancellation intent and propagate it to every descendant."""
        observed = self._now()
        changed: list[ChildTask] = []
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            root = _task_from_row(_task_row(conn, task_id, self.profile_id))
            _check_revision(root, expected_revision)
            task_ids = _descendant_ids(conn, task_id, self.profile_id)
            for current_id in task_ids:
                current = _task_from_row(
                    _task_row(conn, current_id, self.profile_id)
                )
                if current.terminal:
                    continue
                updated = request_cancellation(current)
                if current.status is not ChildTaskStatus.RUNNING:
                    updated = cancel_task(updated, reason, now=observed)
                updated = _redacted_task(
                    updated.with_revision(current.revision + 1, now=observed)
                )
                _update_snapshot(
                    conn,
                    current,
                    updated,
                    clear_claim=updated.terminal,
                )
                if updated.terminal:
                    _terminalize_attempt(
                        conn,
                        current,
                        attempt_id=None,
                        status=updated.status,
                        result=None,
                        error=updated.terminal_reason,
                        now=observed,
                    )
                    _insert_delivery(conn, updated, now=observed)
                _insert_event(
                    conn,
                    updated,
                    kind=(
                        "child.cancelled"
                        if updated.terminal
                        else "child.cancellation_requested"
                    ),
                    actor=actor,
                    payload={"root_task_id": task_id, "reason": reason},
                    now=observed,
                )
                changed.append(updated)
        return tuple(changed)

    def retry(
        self,
        task_id: str,
        *,
        expected_revision: int,
        actor: str,
    ) -> ChildTask:
        observed = self._now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = _task_from_row(_task_row(conn, task_id, self.profile_id))
            _check_revision(current, expected_revision)
            updated = retry_task(current)
            dependency_statuses = _dependency_statuses(
                conn,
                updated.dependency_ids,
                self.profile_id,
            )
            updated = _initial_state(updated, dependency_statuses)
            updated = _redacted_task(
                updated.with_revision(current.revision + 1, now=observed)
            )
            _update_snapshot(conn, current, updated, clear_claim=True)
            _insert_event(
                conn,
                updated,
                kind="child.retried",
                actor=actor,
                payload={},
                now=observed,
            )
        return updated

    def recover_expired(
        self,
        *,
        now: datetime | None = None,
        actor: str = "recovery",
    ) -> tuple[ChildTask, ...]:
        """Mark in-flight expired attempts unknown without replaying them."""
        observed = self._now(now)
        recovered: list[ChildTask] = []
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM child_tasks
                WHERE profile_id = ? AND status = 'running'
                  AND lease_until IS NOT NULL AND lease_until < ?
                ORDER BY updated_at, id
                """,
                (self.profile_id, observed.isoformat()),
            ).fetchall()
            for row in rows:
                current = _task_from_row(row)
                updated = _redacted_task(
                    mark_unknown(
                        current,
                        "Worker lease expired after execution started; "
                        "the outcome is unknown and will not be replayed.",
                        now=observed,
                    ).with_revision(current.revision + 1, now=observed)
                )
                _update_snapshot(conn, current, updated, clear_claim=True)
                _terminalize_attempt(
                    conn,
                    current,
                    attempt_id=_optional(row["attempt_id"]),
                    status=ChildTaskStatus.UNKNOWN,
                    result=None,
                    error=updated.terminal_reason,
                    now=observed,
                )
                _insert_delivery(conn, updated, now=observed)
                _insert_event(
                    conn,
                    updated,
                    kind="child.outcome_unknown",
                    actor=actor,
                    payload={"attempt_id": row["attempt_id"]},
                    now=observed,
                )
                _block_dependents(conn, updated, now=observed, actor=actor)
                recovered.append(updated)
        return tuple(recovered)

    def claim_delivery(
        self,
        *,
        worker_id: str,
        lease_seconds: int = DEFAULT_CHILD_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> ChildCompletionDelivery | None:
        clean_worker = _required(worker_id, "delivery worker id")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        observed = self._now(now)
        lease_until = observed + timedelta(seconds=lease_seconds)
        token = uuid4().hex
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE child_completion_outbox
                SET status = 'unknown', claim_token = NULL, worker_id = NULL,
                    lease_until = NULL,
                    error = 'delivery lease expired', updated_at = ?
                WHERE profile_id = ? AND status = 'claimed' AND lease_until < ?
                """,
                (
                    observed.isoformat(),
                    self.profile_id,
                    observed.isoformat(),
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM child_completion_outbox
                WHERE profile_id = ? AND status IN ('pending', 'failed', 'unknown')
                ORDER BY created_at, id
                LIMIT 1
                """,
                (self.profile_id,),
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """
                UPDATE child_completion_outbox
                SET status = 'claimed', claim_token = ?, worker_id = ?,
                    lease_until = ?,
                    attempts = attempts + 1, updated_at = ?, error = NULL
                WHERE id = ? AND profile_id = ?
                  AND status IN ('pending', 'failed', 'unknown')
                """,
                (
                    token,
                    clean_worker,
                    lease_until.isoformat(),
                    observed.isoformat(),
                    row["id"],
                    self.profile_id,
                ),
            )
            if cursor.rowcount != 1:
                return None
            claimed = conn.execute(
                "SELECT * FROM child_completion_outbox WHERE id = ?",
                (row["id"],),
            ).fetchone()
        assert claimed is not None
        return _delivery_from_row(claimed)

    def finish_delivery(
        self,
        delivery: ChildCompletionDelivery,
        *,
        delivered: bool,
        error: str | None = None,
        now: datetime | None = None,
    ) -> ChildCompletionDelivery:
        """Ack or fail an exact claimed delivery token idempotently."""
        if delivery.profile_id != self.profile_id:
            raise ChildDeliveryConflictError("delivery belongs to another profile")
        if delivery.claim_token is None:
            raise ChildDeliveryConflictError("delivery is not claimed")
        observed = self._now(now)
        status = (
            ChildDeliveryStatus.DELIVERED
            if delivered
            else ChildDeliveryStatus.FAILED
        )
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE child_completion_outbox
                SET status = ?, claim_token = NULL, worker_id = NULL,
                    lease_until = NULL,
                    error = ?, delivered_at = ?, updated_at = ?
                WHERE id = ? AND profile_id = ? AND status = 'claimed'
                  AND claim_token = ? AND lease_until >= ?
                """,
                (
                    status.value,
                    _optional(error),
                    observed.isoformat() if delivered else None,
                    observed.isoformat(),
                    delivery.id,
                    self.profile_id,
                    delivery.claim_token,
                    observed.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                raise ChildDeliveryConflictError(
                    "delivery claim is absent, expired, or owned elsewhere"
                )
            row = conn.execute(
                "SELECT * FROM child_completion_outbox WHERE id = ?",
                (delivery.id,),
            ).fetchone()
        assert row is not None
        return _delivery_from_row(row)

    def list_deliveries(
        self,
        *,
        status: ChildDeliveryStatus | str | None = None,
        task_id: str | None = None,
        limit: int = 100,
    ) -> tuple[ChildCompletionDelivery, ...]:
        if not 1 <= limit <= 1000:
            raise ValueError("delivery list limit must be between 1 and 1000")
        clauses = ["profile_id = ?"]
        values: list[Any] = [self.profile_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(ChildDeliveryStatus(status).value)
        if task_id is not None:
            clauses.append("task_id = ?")
            values.append(task_id)
        values.append(limit)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM child_completion_outbox
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at, id
                LIMIT ?
                """,
                tuple(values),
            ).fetchall()
        return tuple(_delivery_from_row(row) for row in rows)

    def _finish_attempt(
        self,
        claim: ChildTaskClaim,
        status: ChildTaskStatus,
        *,
        result: ChildTaskResult | None,
        reason: str | None,
        actor: str,
        now: datetime | None,
    ) -> ChildTask:
        observed = self._now(now)
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = _assert_claim(
                conn,
                claim,
                self.profile_id,
                observed,
                allow_cancellation=True,
            )
            effective_status = status
            if current.cancellation_requested:
                safe_result = _redacted_result(result) if result is not None else None
                updated = replace(
                    cancel_task(
                        current,
                        "Cancellation was observed after the active child attempt.",
                        now=observed,
                    ),
                    result=safe_result,
                )
                effective_status = ChildTaskStatus.CANCELLED
            elif status is ChildTaskStatus.COMPLETED:
                if result is None:
                    raise ValueError("completed child task requires result")
                updated = complete_task(current, _redacted_result(result), now=observed)
            elif status is ChildTaskStatus.BUDGET_EXHAUSTED:
                updated = exhaust_budget(current, reason or "Budget exhausted.", now=observed)
            else:
                updated = fail_task(current, reason or "Child execution failed.", now=observed)
            updated = _redacted_task(
                updated.with_revision(current.revision + 1, now=observed)
            )
            _update_snapshot(conn, current, updated, clear_claim=True)
            _terminalize_attempt(
                conn,
                current,
                attempt_id=claim.attempt_id,
                status=effective_status,
                result=updated.result,
                error=updated.terminal_reason,
                now=observed,
            )
            _insert_delivery(conn, updated, now=observed)
            _insert_event(
                conn,
                updated,
                kind=f"child.{updated.status.value}",
                actor=actor,
                payload={"attempt_id": claim.attempt_id},
                now=observed,
            )
            if updated.status is ChildTaskStatus.COMPLETED:
                _ready_dependents(conn, updated, now=observed, actor=actor)
            else:
                _block_dependents(conn, updated, now=observed, actor=actor)
        return updated

    def _expire_deadline_if_due(
        self,
        task_id: str,
        *,
        expected_revision: int,
        actor: str,
        now: datetime,
    ) -> bool:
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = _task_from_row(_task_row(conn, task_id, self.profile_id))
            _check_revision(current, expected_revision)
            deadline = current.spec.budget.deadline
            if deadline is None or now < deadline:
                return False
            if current.status is not ChildTaskStatus.READY:
                return False
            updated = _redacted_task(
                exhaust_budget(
                    current,
                    "Child task deadline expired before execution.",
                    now=now,
                ).with_revision(current.revision + 1, now=now)
            )
            _update_snapshot(conn, current, updated, clear_claim=True)
            _insert_delivery(conn, updated, now=now)
            _insert_event(
                conn,
                updated,
                kind="child.budget_exhausted",
                actor=actor,
                payload={"dimension": "deadline"},
                now=now,
            )
            _block_dependents(conn, updated, now=now, actor=actor)
        return True

    def _now(self, value: datetime | None = None) -> datetime:
        observed = value or self.clock()
        if observed.tzinfo is None:
            raise ValueError("child task clock must return a timezone-aware datetime")
        return observed.astimezone(timezone.utc)


def _validate_lineage(conn: sqlite3.Connection, task: ChildTask) -> None:
    lineage = task.lineage
    if lineage.parent_task_id is None:
        if lineage.root_task_id not in {None, task.id}:
            raise ValueError("root child task root_task_id must be itself or empty")
        return
    parent = _task_from_row(
        _task_row(conn, lineage.parent_task_id, task.profile_id)
    )
    if parent.terminal or parent.cancellation_requested:
        raise ValueError("terminal or cancelling child task cannot create descendants")
    expected_root = parent.lineage.root_task_id or parent.id
    if lineage.root_task_id != expected_root:
        raise ValueError("child task root lineage does not match its parent")
    if lineage.depth != parent.lineage.depth + 1:
        raise ValueError("child task depth does not follow its parent")
    if parent.spec.role is not ChildTaskRole.ORCHESTRATOR:
        raise ValueError("leaf child task cannot create descendants")
    if lineage.depth > parent.spec.max_depth or lineage.depth > task.spec.max_depth:
        raise ValueError("child task exceeds the allowed delegation depth")


def _validate_goal_ownership(conn: sqlite3.Connection, task: ChildTask) -> None:
    if task.goal_id is None:
        if task.goal_step_id is not None:
            raise ValueError("goal_step_id requires goal_id")
        return
    row = conn.execute(
        "SELECT profile_id, snapshot_json FROM goals WHERE id = ?",
        (task.goal_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"goal {task.goal_id!r} does not exist")
    if str(row["profile_id"]) != task.profile_id:
        raise ValueError("child task goal belongs to another profile")
    if task.goal_step_id is None:
        return
    snapshot = json.loads(str(row["snapshot_json"]))
    steps = snapshot.get("steps", []) if isinstance(snapshot, dict) else []
    if not any(
        isinstance(step, dict) and step.get("id") == task.goal_step_id
        for step in steps
    ):
        raise ValueError(
            f"goal step {task.goal_step_id!r} does not exist in goal "
            f"{task.goal_id!r}"
        )


def _initial_state(
    task: ChildTask,
    dependency_statuses: Mapping[str, ChildTaskStatus],
) -> ChildTask:
    if not dependency_statuses or all(
        status is ChildTaskStatus.COMPLETED
        for status in dependency_statuses.values()
    ):
        return mark_ready(task)
    blocking = sorted(
        task_id
        for task_id, status in dependency_statuses.items()
        if status
        in {
            ChildTaskStatus.FAILED,
            ChildTaskStatus.CANCELLED,
            ChildTaskStatus.BUDGET_EXHAUSTED,
            ChildTaskStatus.UNKNOWN,
        }
    )
    if blocking:
        return block_task(
            task,
            f"Dependencies require operator action: {', '.join(blocking)}",
        )
    return task


def _assert_parent_parallelism(
    conn: sqlite3.Connection,
    task: ChildTask,
) -> None:
    parent_id = task.lineage.parent_task_id
    if parent_id is None:
        return
    parent = _task_from_row(_task_row(conn, parent_id, task.profile_id))
    active = int(
        conn.execute(
            """
            SELECT COUNT(*) AS count FROM child_tasks
            WHERE profile_id = ? AND parent_task_id = ?
              AND status IN ('running', 'waiting')
            """,
            (task.profile_id, parent_id),
        ).fetchone()["count"]
    )
    if active >= parent.spec.max_parallelism:
        raise ChildTaskLeaseConflictError(
            f"parent child-task parallelism limit "
            f"{parent.spec.max_parallelism} is already active"
        )


def _dependency_statuses(
    conn: sqlite3.Connection,
    dependency_ids: tuple[str, ...],
    profile_id: str,
) -> dict[str, ChildTaskStatus]:
    statuses: dict[str, ChildTaskStatus] = {}
    for dependency_id in dependency_ids:
        dependency = _task_from_row(
            _task_row(conn, dependency_id, profile_id)
        )
        statuses[dependency_id] = dependency.status
    return statuses


def _ready_dependents(
    conn: sqlite3.Connection,
    completed: ChildTask,
    *,
    now: datetime,
    actor: str,
) -> None:
    rows = conn.execute(
        """
        SELECT tasks.* FROM child_tasks tasks
        JOIN child_task_dependencies deps ON deps.task_id = tasks.id
        WHERE deps.profile_id = ? AND deps.dependency_id = ?
          AND tasks.status = 'pending'
        ORDER BY tasks.created_at, tasks.id
        """,
        (completed.profile_id, completed.id),
    ).fetchall()
    for row in rows:
        current = _task_from_row(row)
        statuses = _dependency_statuses(
            conn,
            current.dependency_ids,
            current.profile_id,
        )
        if not all(
            status is ChildTaskStatus.COMPLETED
            for status in statuses.values()
        ):
            continue
        updated = mark_ready(current).with_revision(
            current.revision + 1,
            now=now,
        )
        _update_snapshot(conn, current, updated, clear_claim=False)
        _insert_event(
            conn,
            updated,
            kind="child.ready",
            actor=actor,
            payload={"completed_dependency_id": completed.id},
            now=now,
        )


def _block_dependents(
    conn: sqlite3.Connection,
    terminal: ChildTask,
    *,
    now: datetime,
    actor: str,
) -> None:
    rows = conn.execute(
        """
        SELECT tasks.* FROM child_tasks tasks
        JOIN child_task_dependencies deps ON deps.task_id = tasks.id
        WHERE deps.profile_id = ? AND deps.dependency_id = ?
          AND tasks.status IN ('pending', 'ready', 'waiting')
        ORDER BY tasks.created_at, tasks.id
        """,
        (terminal.profile_id, terminal.id),
    ).fetchall()
    for row in rows:
        current = _task_from_row(row)
        updated = block_task(
            current,
            f"Dependency {terminal.id} ended as {terminal.status.value}.",
        ).with_revision(current.revision + 1, now=now)
        _update_snapshot(conn, current, updated, clear_claim=False)
        _insert_event(
            conn,
            updated,
            kind="child.blocked",
            actor=actor,
            payload={
                "dependency_id": terminal.id,
                "dependency_status": terminal.status.value,
            },
            now=now,
        )
        _block_dependents(conn, updated, now=now, actor=actor)


def _descendant_ids(
    conn: sqlite3.Connection,
    task_id: str,
    profile_id: str,
) -> tuple[str, ...]:
    rows = conn.execute(
        """
        WITH RECURSIVE descendants(id, depth) AS (
            SELECT id, 0 FROM child_tasks
            WHERE id = ? AND profile_id = ?
            UNION ALL
            SELECT child.id, descendants.depth + 1
            FROM child_tasks child
            JOIN descendants ON child.parent_task_id = descendants.id
            WHERE child.profile_id = ?
        )
        SELECT id FROM descendants ORDER BY depth DESC, id
        """,
        (task_id, profile_id, profile_id),
    ).fetchall()
    return tuple(str(row["id"]) for row in rows)


def _assert_claim(
    conn: sqlite3.Connection,
    claim: ChildTaskClaim,
    profile_id: str,
    now: datetime,
    *,
    allow_cancellation: bool = False,
) -> ChildTask:
    if claim.profile_id != profile_id:
        raise ChildTaskLeaseConflictError("claim belongs to another profile")
    row = _task_row(conn, claim.task_id, profile_id)
    task = _task_from_row(row)
    if (
        row["claim_token"] != claim.claim_token
        or row["worker_id"] != claim.worker_id
        or row["attempt_id"] != claim.attempt_id
    ):
        raise ChildTaskLeaseConflictError("child attempt belongs to another worker")
    lease_until = _optional_datetime(row["lease_until"])
    if lease_until is None or lease_until < now:
        raise ChildTaskLeaseConflictError("child attempt lease expired")
    if task.status is not ChildTaskStatus.RUNNING:
        raise ChildTaskLeaseConflictError("child task is not running")
    if task.cancellation_requested and not allow_cancellation:
        raise ChildTaskLeaseConflictError(
            "child cancellation requested before next action"
        )
    return task


def _insert_task(conn: sqlite3.Connection, task: ChildTask) -> None:
    conn.execute(
        """
        INSERT INTO child_tasks (
            id, profile_id, goal_id, goal_step_id, parent_task_id, root_task_id,
            depth, status, revision, snapshot_json, cancellation_requested,
            claim_token, worker_id, attempt_id, lease_until,
            created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?)
        """,
        (
            task.id,
            task.profile_id,
            task.goal_id,
            task.goal_step_id,
            task.lineage.parent_task_id,
            task.lineage.root_task_id,
            task.lineage.depth,
            task.status.value,
            task.revision,
            _json(task.to_dict()),
            int(task.cancellation_requested),
            task.created_at.isoformat(),
            task.updated_at.isoformat(),
            _iso(task.completed_at),
        ),
    )


def _update_snapshot(
    conn: sqlite3.Connection,
    current: ChildTask,
    updated: ChildTask,
    *,
    clear_claim: bool,
) -> None:
    cursor = conn.execute(
        """
        UPDATE child_tasks
        SET status = ?, revision = ?, snapshot_json = ?,
            cancellation_requested = ?, updated_at = ?, completed_at = ?,
            claim_token = CASE WHEN ? THEN NULL ELSE claim_token END,
            worker_id = CASE WHEN ? THEN NULL ELSE worker_id END,
            attempt_id = CASE WHEN ? THEN NULL ELSE attempt_id END,
            lease_until = CASE WHEN ? THEN NULL ELSE lease_until END
        WHERE id = ? AND profile_id = ? AND revision = ?
        """,
        (
            updated.status.value,
            updated.revision,
            _json(updated.to_dict()),
            int(updated.cancellation_requested),
            updated.updated_at.isoformat(),
            _iso(updated.completed_at),
            int(clear_claim),
            int(clear_claim),
            int(clear_claim),
            int(clear_claim),
            current.id,
            current.profile_id,
            current.revision,
        ),
    )
    if cursor.rowcount != 1:
        fresh = _task_from_row(
            _task_row(conn, current.id, current.profile_id)
        )
        raise ChildTaskRevisionConflictError(
            current.id,
            current.revision,
            fresh.revision,
        )


def _terminalize_attempt(
    conn: sqlite3.Connection,
    task: ChildTask,
    *,
    attempt_id: str | None,
    status: ChildTaskStatus,
    result: ChildTaskResult | None,
    error: str | None,
    now: datetime,
) -> None:
    if attempt_id is None:
        attempt = conn.execute(
            """
            SELECT * FROM child_task_attempts
            WHERE task_id = ? AND profile_id = ? AND status = 'running'
            ORDER BY attempt_number DESC LIMIT 1
            """,
            (task.id, task.profile_id),
        ).fetchone()
    else:
        attempt = conn.execute(
            """
            SELECT * FROM child_task_attempts
            WHERE id = ? AND task_id = ? AND profile_id = ?
              AND status = 'running'
            """,
            (attempt_id, task.id, task.profile_id),
        ).fetchone()
    if attempt is None:
        return
    conn.execute(
        """
        UPDATE child_task_attempts
        SET status = ?, result_json = ?, error = ?,
            updated_at = ?, completed_at = ?
        WHERE id = ? AND status = 'running'
        """,
        (
            status.value,
            _json(result.to_dict()) if result is not None else None,
            _optional(error),
            now.isoformat(),
            now.isoformat(),
            attempt["id"],
        ),
    )


def _insert_delivery(
    conn: sqlite3.Connection,
    task: ChildTask,
    *,
    now: datetime,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO child_completion_outbox (
            id, task_id, profile_id, task_revision, status, idempotency_key,
            claim_token, worker_id, lease_until, attempts, error,
            created_at, updated_at, delivered_at
        ) VALUES (?, ?, ?, ?, 'pending', ?, NULL, NULL, NULL, 0, NULL, ?, ?, NULL)
        """,
        (
            uuid4().hex,
            task.id,
            task.profile_id,
            task.revision,
            f"child:{task.id}:revision:{task.revision}",
            now.isoformat(),
            now.isoformat(),
        ),
    )


def _insert_event(
    conn: sqlite3.Connection,
    task: ChildTask,
    *,
    kind: str,
    actor: str,
    payload: Mapping[str, Any],
    now: datetime,
) -> None:
    conn.execute(
        """
        INSERT INTO child_task_events (
            id, task_id, profile_id, revision, kind, actor,
            payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid4().hex,
            task.id,
            task.profile_id,
            task.revision,
            _required(kind, "child event kind"),
            _required(actor, "child event actor"),
            _json(redact_data(dict(payload))),
            now.isoformat(),
        ),
    )


def _task_row(
    conn: sqlite3.Connection,
    task_id: str,
    profile_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM child_tasks WHERE id = ? AND profile_id = ?",
        (task_id, profile_id),
    ).fetchone()
    if row is None:
        raise ChildTaskNotFoundError(
            f"child task {task_id!r} does not exist"
        )
    return row


def _task_from_row(row: sqlite3.Row) -> ChildTask:
    value = json.loads(str(row["snapshot_json"]))
    if not isinstance(value, dict):
        raise ValueError("stored child task snapshot must be an object")
    value["status"] = str(row["status"])
    value["revision"] = int(row["revision"])
    value["cancellation_requested"] = bool(row["cancellation_requested"])
    value["updated_at"] = str(row["updated_at"])
    value["completed_at"] = row["completed_at"]
    return child_task_from_dict(value)


def _event_from_row(row: sqlite3.Row) -> ChildTaskEvent:
    payload = json.loads(str(row["payload_json"]))
    return ChildTaskEvent(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        profile_id=str(row["profile_id"]),
        revision=int(row["revision"]),
        kind=str(row["kind"]),
        actor=str(row["actor"]),
        payload=payload if isinstance(payload, dict) else {},
        created_at=_datetime(str(row["created_at"])),
    )


def _delivery_from_row(row: sqlite3.Row) -> ChildCompletionDelivery:
    return ChildCompletionDelivery(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        profile_id=str(row["profile_id"]),
        task_revision=int(row["task_revision"]),
        status=ChildDeliveryStatus(str(row["status"])),
        idempotency_key=str(row["idempotency_key"]),
        claim_token=_optional(row["claim_token"]),
        worker_id=_optional(row["worker_id"]),
        lease_until=_optional_datetime(row["lease_until"]),
        attempts=int(row["attempts"]),
        error=_optional(row["error"]),
        created_at=_datetime(str(row["created_at"])),
        updated_at=_datetime(str(row["updated_at"])),
        delivered_at=_optional_datetime(row["delivered_at"]),
    )


def _attempt_dict(row: sqlite3.Row) -> Mapping[str, Any]:
    return {
        key: row[key]
        for key in row.keys()
        if key not in {"claim_token", "result_json"}
    } | {
        "result": (
            json.loads(str(row["result_json"]))
            if row["result_json"] is not None
            else None
        )
    }


def _redacted_task(task: ChildTask) -> ChildTask:
    value = redact_data(task.to_dict())
    if not isinstance(value, dict):
        raise ValueError("redacted child task must remain an object")
    return child_task_from_dict(value)


def _redacted_result(result: ChildTaskResult) -> ChildTaskResult:
    value = redact_data(result.to_dict())
    if not isinstance(value, dict):
        raise ValueError("redacted child result must remain an object")
    return child_result_from_dict(value)


def _creation_payload(task: ChildTask) -> Mapping[str, Any]:
    return {
        "id": task.id,
        "profile_id": task.profile_id,
        "spec": task.spec.to_dict(),
        "lineage": task.lineage.to_dict(),
        "dependency_ids": list(task.dependency_ids),
        "goal_id": task.goal_id,
        "goal_step_id": task.goal_step_id,
        "parent_conversation_id": task.parent_conversation_id,
        "parent_turn_id": task.parent_turn_id,
        "parent_trace_id": task.parent_trace_id,
    }


def _check_revision(task: ChildTask, expected_revision: int) -> None:
    if task.revision != expected_revision:
        raise ChildTaskRevisionConflictError(
            task.id,
            expected_revision,
            task.revision,
        )


def _required(value: Any, label: str) -> str:
    clean = str(value).strip()
    if not clean:
        raise ValueError(f"{label} cannot be empty")
    return clean


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("stored child timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _optional_datetime(value: Any) -> datetime | None:
    return _datetime(str(value)) if value is not None else None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = [
    "DEFAULT_CHILD_LEASE_SECONDS",
    "ChildDeliveryConflictError",
    "ChildTaskLeaseConflictError",
    "ChildTaskConflictError",
    "ChildTaskNotFoundError",
    "ChildTaskRevisionConflictError",
    "ChildTaskStore",
]
