"""Run leasing, step transitions, cancellation, and recovery."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from chulk.hosting.scope import ExecutionScope
from chulk.runs.models import (
    AttemptRecord,
    AttemptStatus,
    Checkpoint,
    EffectStatus,
    RetryPolicy,
    RunClaim,
    RunRecord,
    RunStatus,
)

import chulk.runs._store_clock as _clock
from chulk.runs.errors import (
    InvalidRunTransitionError,
    RunConflictError,
    RunLeaseError,
    RunNotFoundError,
)
from chulk.runs._store_support import (
    _RunStoreBackend,
    _active_attempt,
    _assert_scope,
    _decode,
    _effect_from_row,
    _expire_linked_child_run_if_due,
    _insert_event,
    _iso,
    _json,
    _mark_active_attempt,
    _next_checkpoint_sequence,
    _object,
    _observed,
    _owned_run,
    _positive_seconds,
    _reject_direct_parent_transition,
    _required,
    _run_from_conn,
    _run_row,
    _safe_payload,
    _set_run_state,
    _settle_run_cancellation,
    _step_from_row,
    _step_row,
    _touch_run,
)


class _RunStoreTransitionMixin(_RunStoreBackend):
    def claim(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        run_id: str | None = None,
    ) -> RunClaim | None:
        worker_id = _required(worker_id, "worker id")
        _positive_seconds(lease_seconds)
        if scope.parent_run_id is not None:
            if run_id is not None and run_id != scope.run_id:
                raise RunNotFoundError("child scope cannot claim a sibling run")
            run_id = scope.run_id
        now = _clock.utc_now()
        lease_until = now + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            preflight_run_id = run_id
            if preflight_run_id is not None and _expire_linked_child_run_if_due(
                conn,
                scope,
                preflight_run_id,
                actor=worker_id,
                now=now,
            ):
                return None
            parameters: list[Any] = [
                scope.tenant_id,
                scope.workspace_id,
                scope.agent_id,
                scope.agent_version,
                _iso(now),
            ]
            run_clause = ""
            queue_scope_clause = ""
            if run_id is not None:
                run_clause = "AND id = ?"
                parameters.append(run_id)
            else:
                queue_scope_clause = """
                  AND NOT EXISTS (
                    SELECT 1 FROM durable_child_runs AS linked_child
                    WHERE linked_child.child_run_id = durable_runs.id
                  )
                """
            row = conn.execute(
                f"""
                SELECT * FROM durable_runs
                WHERE tenant_id = ? AND workspace_id = ?
                  AND agent_id = ? AND agent_version = ?
                  AND cancellation_requested = 0
                  AND (
                    status = 'queued'
                    OR (
                        status = 'waiting_for_retry'
                        AND next_retry_at IS NOT NULL
                        AND next_retry_at <= ?
                    )
                  )
                  {queue_scope_clause}
                  {run_clause}
                ORDER BY created_at, id
                LIMIT 1
                """,
                tuple(parameters),
            ).fetchone()
            if row is None:
                return None
            persisted_scope = ExecutionScope.from_dict(_object(row["scope_json"]))
            _assert_scope(scope, persisted_scope)
            if str(row["id"]) != preflight_run_id and _expire_linked_child_run_if_due(
                conn,
                scope,
                str(row["id"]),
                actor=worker_id,
                now=now,
            ):
                return None
            token = uuid4().hex
            revision = int(row["revision"]) + 1
            cursor = conn.execute(
                """
                UPDATE durable_runs
                SET status = 'running', revision = ?, claim_token = ?,
                    worker_id = ?, lease_until = ?, waiting_reason = NULL,
                    next_retry_at = NULL, updated_at = ?
                WHERE id = ? AND revision = ?
                  AND status IN ('queued', 'waiting_for_retry')
                """,
                (
                    revision,
                    token,
                    worker_id,
                    _iso(lease_until),
                    _iso(now),
                    str(row["id"]),
                    int(row["revision"]),
                ),
            )
            if cursor.rowcount != 1:
                raise RunConflictError("run changed while it was being claimed")
            _insert_event(
                conn,
                str(row["id"]),
                name="run.started",
                actor=worker_id,
                payload={"lease_until": _iso(lease_until)},
                now=now,
            )
        return RunClaim(
            run_id=str(row["id"]),
            scope_key=persisted_scope.key,
            worker_id=worker_id,
            lease_token=token,
            lease_until=lease_until,
            revision=revision,
        )

    def renew(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        lease_seconds: int = 120,
    ) -> RunClaim:
        _positive_seconds(lease_seconds)
        now = _clock.utc_now()
        lease_until = now + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = _owned_run(conn, scope, claim, now=now)
            cursor = conn.execute(
                """
                UPDATE durable_runs
                SET lease_until = ?, updated_at = ?
                WHERE id = ? AND claim_token = ? AND lease_until >= ?
                """,
                (
                    _iso(lease_until),
                    _iso(now),
                    claim.run_id,
                    claim.lease_token,
                    _iso(now),
                ),
            )
            if cursor.rowcount != 1:
                raise RunLeaseError("run lease could not be renewed")
        return RunClaim(
            run_id=claim.run_id,
            scope_key=claim.scope_key,
            worker_id=claim.worker_id,
            lease_token=claim.lease_token,
            lease_until=lease_until,
            revision=int(row["revision"]),
        )

    def start_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
    ) -> AttemptRecord:
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            if conn.execute(
                """
                SELECT 1 FROM durable_run_steps
                WHERE run_id = ? AND status = 'running'
                """,
                (claim.run_id,),
            ).fetchone():
                raise InvalidRunTransitionError(
                    "another durable run step is already active"
                )
            step = _step_row(conn, claim.run_id, step_id)
            if str(step["status"]) not in {"queued", "waiting_for_retry"}:
                raise InvalidRunTransitionError(
                    f"step cannot start from {step['status']}"
                )
            if (
                step["next_retry_at"] is not None
                and _decode(str(step["next_retry_at"])) > now
            ):
                raise InvalidRunTransitionError("step retry is not due")
            number = int(step["attempt_count"]) + 1
            retry_policy = RetryPolicy.from_dict(_object(step["retry_json"]))
            counted_attempts = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM durable_run_attempts
                    WHERE run_id = ? AND step_id = ? AND status != 'paused'
                    """,
                    (claim.run_id, step_id),
                ).fetchone()[0]
            )
            if counted_attempts >= retry_policy.max_attempts:
                raise InvalidRunTransitionError("step exhausted its attempt limit")
            attempt_id = uuid4().hex
            conn.execute(
                """
                INSERT INTO durable_run_attempts (
                    id, run_id, step_id, number, status, worker_id,
                    lease_token, started_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    attempt_id,
                    claim.run_id,
                    step_id,
                    number,
                    claim.worker_id,
                    claim.lease_token,
                    _iso(now),
                ),
            )
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = 'running', revision = revision + 1,
                    attempt_count = ?, next_retry_at = NULL, error = NULL,
                    started_at = COALESCE(started_at, ?), completed_at = NULL,
                    updated_at = ?
                WHERE run_id = ? AND id = ?
                """,
                (number, _iso(now), _iso(now), claim.run_id, step_id),
            )
            _touch_run(conn, claim.run_id, now)
            _insert_event(
                conn,
                claim.run_id,
                name="step.started",
                actor=claim.worker_id,
                step_id=step_id,
                payload={"attempt_id": attempt_id, "attempt": number},
                now=now,
            )
        return AttemptRecord(
            id=attempt_id,
            run_id=claim.run_id,
            step_id=step_id,
            number=number,
            status=AttemptStatus.RUNNING,
            worker_id=claim.worker_id,
            lease_token=claim.lease_token,
            started_at=now,
        )

    def checkpoint(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
    ) -> Checkpoint:
        kind = _required(kind, "checkpoint kind")
        safe_payload = _safe_payload(payload)
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            attempt = _active_attempt(conn, claim.run_id, step_id)
            sequence = _next_checkpoint_sequence(conn, claim.run_id)
            checkpoint = Checkpoint(
                id=uuid4().hex,
                run_id=claim.run_id,
                step_id=step_id,
                attempt_id=str(attempt["id"]),
                sequence=sequence,
                kind=kind,
                payload=safe_payload,
                created_at=now,
            )
            conn.execute(
                """
                INSERT INTO durable_run_checkpoints (
                    id, run_id, step_id, attempt_id, sequence, kind,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.run_id,
                    checkpoint.step_id,
                    checkpoint.attempt_id,
                    checkpoint.sequence,
                    checkpoint.kind,
                    _json(checkpoint.payload),
                    _iso(now),
                ),
            )
            conn.execute(
                """
                UPDATE durable_run_steps
                SET last_checkpoint_id = ?, revision = revision + 1,
                    updated_at = ?
                WHERE run_id = ? AND id = ? AND status = 'running'
                """,
                (checkpoint.id, _iso(now), claim.run_id, step_id),
            )
            _touch_run(conn, claim.run_id, now)
            _insert_event(
                conn,
                claim.run_id,
                name="step.checkpointed",
                actor=claim.worker_id,
                step_id=step_id,
                payload={
                    "checkpoint_id": checkpoint.id,
                    "kind": kind,
                    "sequence": sequence,
                },
                now=now,
            )
        return checkpoint

    def complete_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        now = _clock.utc_now()
        safe_result = _safe_payload(result or {})
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            _active_attempt(conn, claim.run_id, step_id)
            unfinished_effect = conn.execute(
                """
                SELECT 1 FROM durable_effects
                WHERE run_id = ? AND step_id = ?
                  AND status IN ('intended', 'executing', 'unknown')
                """,
                (claim.run_id, step_id),
            ).fetchone()
            if unfinished_effect is not None:
                raise InvalidRunTransitionError(
                    "step cannot complete with an intended, executing, "
                    "or unknown effect"
                )
            _mark_active_attempt(
                conn,
                claim.run_id,
                step_id,
                AttemptStatus.COMPLETED,
                now,
                None,
            )
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = 'completed', revision = revision + 1,
                    error = NULL, updated_at = ?, completed_at = ?
                WHERE run_id = ? AND id = ? AND status = 'running'
                """,
                (_iso(now), _iso(now), claim.run_id, step_id),
            )
            _touch_run(conn, claim.run_id, now)
            _insert_event(
                conn,
                claim.run_id,
                name="step.completed",
                actor=claim.worker_id,
                step_id=step_id,
                payload={"result": safe_result},
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, claim.run_id))

    def fail_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        reason: str,
        retryable: bool,
    ) -> RunRecord:
        reason = _required(reason, "step failure reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owned = _owned_run(conn, scope, claim, now=now)
            step = _step_from_row(_step_row(conn, claim.run_id, step_id))
            _active_attempt(conn, claim.run_id, step_id)
            if conn.execute(
                """
                SELECT 1 FROM durable_effects
                WHERE run_id = ? AND step_id = ?
                  AND status IN ('executing', 'unknown')
                """,
                (claim.run_id, step_id),
            ).fetchone():
                raise InvalidRunTransitionError(
                    "uncertain effects must be reconciled before retry or failure"
                )
            counted_attempts = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM durable_run_attempts
                    WHERE run_id = ? AND step_id = ? AND status != 'paused'
                    """,
                    (claim.run_id, step_id),
                ).fetchone()[0]
            )
            can_retry = retryable and counted_attempts < step.retry_policy.max_attempts
            if can_retry and bool(owned["cancellation_requested"]):
                _settle_run_cancellation(
                    conn,
                    claim.run_id,
                    actor=claim.worker_id,
                    reason=(
                        f"cancellation settled after retryable step failure: {reason}"
                    ),
                    now=now,
                )
                return _run_from_conn(conn, _run_row(conn, claim.run_id))
            _mark_active_attempt(
                conn,
                claim.run_id,
                step_id,
                AttemptStatus.FAILED,
                now,
                reason,
            )
            if can_retry:
                next_retry = now + step.retry_policy.delay_for_attempt(
                    step.attempt_count
                )
                conn.execute(
                    """
                    UPDATE durable_run_steps
                    SET status = 'waiting_for_retry',
                        revision = revision + 1, error = ?,
                        next_retry_at = ?, updated_at = ?
                    WHERE run_id = ? AND id = ?
                    """,
                    (
                        reason,
                        _iso(next_retry),
                        _iso(now),
                        claim.run_id,
                        step_id,
                    ),
                )
                _set_run_state(
                    conn,
                    claim.run_id,
                    RunStatus.WAITING_FOR_RETRY,
                    now=now,
                    error=reason,
                    waiting_reason=reason,
                    next_retry_at=next_retry,
                    clear_lease=True,
                )
                event_name = "run.retry_scheduled"
                payload = {
                    "reason": reason,
                    "next_retry_at": _iso(next_retry),
                }
            else:
                conn.execute(
                    """
                    UPDATE durable_run_steps
                    SET status = 'failed', revision = revision + 1,
                        error = ?, updated_at = ?, completed_at = ?
                    WHERE run_id = ? AND id = ?
                    """,
                    (
                        reason,
                        _iso(now),
                        _iso(now),
                        claim.run_id,
                        step_id,
                    ),
                )
                _set_run_state(
                    conn,
                    claim.run_id,
                    RunStatus.FAILED,
                    now=now,
                    error=reason,
                    clear_lease=True,
                    completed=True,
                )
                event_name = "run.failed"
                payload = {"reason": reason}
            _insert_event(
                conn,
                claim.run_id,
                name="step.failed",
                actor=claim.worker_id,
                step_id=step_id,
                payload={"reason": reason, "retryable": can_retry},
                now=now,
            )
            _insert_event(
                conn,
                claim.run_id,
                name=event_name,
                actor=claim.worker_id,
                step_id=step_id,
                payload=payload,
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, claim.run_id))

    def pause_for_approval(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        approval_id: str,
        payload: Mapping[str, Any],
    ) -> RunRecord:
        approval_id = _required(approval_id, "approval id")
        safe_payload = _safe_payload(payload)
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            attempt = _active_attempt(conn, claim.run_id, step_id)
            if conn.execute(
                """
                SELECT 1 FROM durable_effects
                WHERE run_id = ? AND step_id = ?
                  AND status IN ('executing', 'unknown')
                """,
                (claim.run_id, step_id),
            ).fetchone():
                raise InvalidRunTransitionError(
                    "uncertain effects must be reconciled before approval pause"
                )
            sequence = _next_checkpoint_sequence(conn, claim.run_id)
            checkpoint_id = uuid4().hex
            conn.execute(
                """
                INSERT INTO durable_run_checkpoints (
                    id, run_id, step_id, attempt_id, sequence, kind,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, 'approval', ?, ?)
                """,
                (
                    checkpoint_id,
                    claim.run_id,
                    step_id,
                    str(attempt["id"]),
                    sequence,
                    _json(
                        {
                            **safe_payload,
                            "approval_id": approval_id,
                        }
                    ),
                    _iso(now),
                ),
            )
            _mark_active_attempt(
                conn,
                claim.run_id,
                step_id,
                AttemptStatus.PAUSED,
                now,
                "waiting for approval",
            )
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = 'waiting_for_approval',
                    revision = revision + 1, last_checkpoint_id = ?,
                    error = NULL, updated_at = ?
                WHERE run_id = ? AND id = ?
                """,
                (checkpoint_id, _iso(now), claim.run_id, step_id),
            )
            _set_run_state(
                conn,
                claim.run_id,
                RunStatus.WAITING_FOR_APPROVAL,
                now=now,
                waiting_reason=f"approval:{approval_id}",
                clear_lease=True,
            )
            _insert_event(
                conn,
                claim.run_id,
                name="run.paused",
                actor=claim.worker_id,
                step_id=step_id,
                payload={
                    "reason": "waiting_for_approval",
                    "approval_id": approval_id,
                    "checkpoint_id": checkpoint_id,
                },
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, claim.run_id))

    def resume(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord:
        actor = _required(actor, "resume actor")
        reason = _required(reason, "resume reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = _run_from_conn(conn, _run_row(conn, run_id))
            _assert_scope(scope, run.scope)
            if run.status not in {
                RunStatus.WAITING_FOR_APPROVAL,
                RunStatus.WAITING_FOR_RETRY,
            }:
                raise InvalidRunTransitionError(
                    f"run cannot resume from {run.status.value}"
                )
            if run.cancellation_requested:
                raise InvalidRunTransitionError("cancelled run cannot resume")
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = 'queued', revision = revision + 1,
                    next_retry_at = NULL, error = NULL, updated_at = ?
                WHERE run_id = ?
                  AND status IN ('waiting_for_approval', 'waiting_for_retry')
                """,
                (_iso(now), run_id),
            )
            _set_run_state(
                conn,
                run_id,
                RunStatus.QUEUED,
                now=now,
                clear_lease=True,
            )
            _insert_event(
                conn,
                run_id,
                name="run.resumed",
                actor=actor,
                payload={"reason": reason},
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, run_id))

    def request_cancellation(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord:
        actor = _required(actor, "cancellation actor")
        reason = _required(reason, "cancellation reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = _run_from_conn(conn, _run_row(conn, run_id))
            _assert_scope(scope, run.scope)
            _reject_direct_parent_transition(
                conn,
                run_id,
                "request_parent_cancellation",
            )
            if run.status is RunStatus.COMPLETED:
                raise InvalidRunTransitionError("completed run cannot be cancelled")
            if run.status is RunStatus.CANCELLED:
                return run
            immediate = run.status in {
                RunStatus.QUEUED,
                RunStatus.WAITING_FOR_APPROVAL,
                RunStatus.WAITING_FOR_RETRY,
            }
            conn.execute(
                """
                UPDATE durable_runs
                SET cancellation_requested = 1, revision = revision + 1,
                    status = CASE WHEN ? THEN 'cancelled' ELSE status END,
                    error = CASE WHEN ? THEN ? ELSE error END,
                    claim_token = CASE WHEN ? THEN NULL ELSE claim_token END,
                    worker_id = CASE WHEN ? THEN NULL ELSE worker_id END,
                    lease_until = CASE WHEN ? THEN NULL ELSE lease_until END,
                    updated_at = ?,
                    completed_at = CASE WHEN ? THEN ? ELSE completed_at END
                WHERE id = ?
                """,
                (
                    int(immediate),
                    int(immediate),
                    reason,
                    int(immediate),
                    int(immediate),
                    int(immediate),
                    _iso(now),
                    int(immediate),
                    _iso(now),
                    run_id,
                ),
            )
            if immediate:
                conn.execute(
                    """
                    UPDATE durable_run_steps
                    SET status = CASE
                            WHEN status IN ('queued', 'waiting_for_approval',
                                            'waiting_for_retry')
                            THEN 'cancelled'
                            ELSE status
                        END,
                        revision = revision + 1, error = ?,
                        updated_at = ?, completed_at = ?
                    WHERE run_id = ? AND status NOT IN ('completed', 'failed')
                    """,
                    (reason, _iso(now), _iso(now), run_id),
                )
            _insert_event(
                conn,
                run_id,
                name="run.cancellation_requested",
                actor=actor,
                payload={"reason": reason, "immediate": immediate},
                now=now,
            )
            if immediate:
                _insert_event(
                    conn,
                    run_id,
                    name="run.cancelled",
                    actor=actor,
                    payload={"reason": reason},
                    now=now,
                )
            return _run_from_conn(conn, _run_row(conn, run_id))

    def cancel(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
        claim: RunClaim | None = None,
    ) -> RunRecord:
        actor = _required(actor, "cancellation actor")
        reason = _required(reason, "cancellation reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = _run_from_conn(conn, _run_row(conn, run_id))
            _assert_scope(scope, run.scope)
            _reject_direct_parent_transition(
                conn,
                run_id,
                "request_parent_cancellation",
            )
            if run.status is RunStatus.COMPLETED:
                raise InvalidRunTransitionError("completed run cannot be cancelled")
            if run.status is RunStatus.CANCELLED:
                return run
            if run.status is RunStatus.RUNNING:
                if claim is None:
                    raise RunLeaseError("active run cancellation requires its lease")
                _owned_run(conn, scope, claim, now=now)
            uncertain = conn.execute(
                """
                SELECT * FROM durable_effects
                WHERE run_id = ? AND status IN ('executing', 'unknown')
                ORDER BY created_at LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if uncertain is not None:
                effect = _effect_from_row(uncertain)
                if effect.status is EffectStatus.EXECUTING:
                    conn.execute(
                        """
                        UPDATE durable_effects
                        SET status = 'unknown', reconciliation_reason = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (reason, _iso(now), effect.id),
                    )
                conn.execute(
                    """
                    UPDATE durable_run_steps
                    SET status = 'unknown', revision = revision + 1,
                        error = ?, updated_at = ?
                    WHERE run_id = ? AND id = ?
                    """,
                    (reason, _iso(now), run_id, effect.step_id),
                )
                _set_run_state(
                    conn,
                    run_id,
                    RunStatus.UNKNOWN,
                    now=now,
                    error=reason,
                    clear_lease=True,
                )
                conn.execute(
                    """
                    UPDATE durable_runs
                    SET cancellation_requested = 1 WHERE id = ?
                    """,
                    (run_id,),
                )
                event_name = "run.unknown"
                payload = {
                    "reason": reason,
                    "effect_id": effect.id,
                    "cancellation_requested": True,
                }
            else:
                conn.execute(
                    """
                    UPDATE durable_run_attempts
                    SET status = 'cancelled', error = ?, completed_at = ?
                    WHERE run_id = ? AND status = 'running'
                    """,
                    (reason, _iso(now), run_id),
                )
                conn.execute(
                    """
                    UPDATE durable_run_steps
                    SET status = CASE
                            WHEN status != 'completed' THEN 'cancelled'
                            ELSE status
                        END,
                        revision = revision + 1, error = ?, updated_at = ?,
                        completed_at = CASE
                            WHEN status != 'completed' THEN ?
                            ELSE completed_at
                        END
                    WHERE run_id = ?
                    """,
                    (reason, _iso(now), _iso(now), run_id),
                )
                _set_run_state(
                    conn,
                    run_id,
                    RunStatus.CANCELLED,
                    now=now,
                    error=reason,
                    clear_lease=True,
                    completed=True,
                )
                conn.execute(
                    """
                    UPDATE durable_runs
                    SET cancellation_requested = 1 WHERE id = ?
                    """,
                    (run_id,),
                )
                event_name = "run.cancelled"
                payload = {"reason": reason}
            _insert_event(
                conn,
                run_id,
                name=event_name,
                actor=actor,
                payload=payload,
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, run_id))

    def complete(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        result: Mapping[str, Any],
    ) -> RunRecord:
        safe_result = _safe_payload(result)
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            incomplete = conn.execute(
                """
                SELECT 1 FROM durable_run_steps
                WHERE run_id = ? AND status != 'completed'
                """,
                (claim.run_id,),
            ).fetchone()
            if incomplete is not None:
                raise InvalidRunTransitionError(
                    "run cannot complete before every step completes"
                )
            row = _run_row(conn, claim.run_id)
            if bool(row["cancellation_requested"]):
                raise InvalidRunTransitionError(
                    "run cannot complete after cancellation was requested"
                )
            _set_run_state(
                conn,
                claim.run_id,
                RunStatus.COMPLETED,
                now=now,
                result=safe_result,
                clear_lease=True,
                completed=True,
            )
            _insert_event(
                conn,
                claim.run_id,
                name="run.completed",
                actor=claim.worker_id,
                payload={"result": safe_result},
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, claim.run_id))

    def fail(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        reason: str,
    ) -> RunRecord:
        return self._terminal_run(
            scope,
            claim,
            status=RunStatus.FAILED,
            event_name="run.failed",
            reason=reason,
        )

    def mark_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        reason: str,
    ) -> RunRecord:
        reason = _required(reason, "unknown run reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            active = conn.execute(
                """
                SELECT * FROM durable_run_steps
                WHERE run_id = ? AND status = 'running'
                ORDER BY id LIMIT 1
                """,
                (claim.run_id,),
            ).fetchone()
            if active is not None:
                _mark_active_attempt(
                    conn,
                    claim.run_id,
                    str(active["id"]),
                    AttemptStatus.UNKNOWN,
                    now,
                    reason,
                )
                conn.execute(
                    """
                    UPDATE durable_run_steps
                    SET status = 'unknown', revision = revision + 1,
                        error = ?, updated_at = ?
                    WHERE run_id = ? AND id = ?
                    """,
                    (
                        reason,
                        _iso(now),
                        claim.run_id,
                        str(active["id"]),
                    ),
                )
            _set_run_state(
                conn,
                claim.run_id,
                RunStatus.UNKNOWN,
                now=now,
                error=reason,
                clear_lease=True,
            )
            _insert_event(
                conn,
                claim.run_id,
                name="run.unknown",
                actor=claim.worker_id,
                step_id=str(active["id"]) if active is not None else None,
                payload={"reason": reason},
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, claim.run_id))

    def dead_letter(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord:
        actor = _required(actor, "dead-letter actor")
        reason = _required(reason, "dead-letter reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = _run_from_conn(conn, _run_row(conn, run_id))
            _assert_scope(scope, run.scope)
            _reject_direct_parent_transition(
                conn,
                run_id,
                "aggregate_children",
            )
            if run.status is RunStatus.COMPLETED:
                raise InvalidRunTransitionError("completed run cannot be dead-lettered")
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = CASE
                        WHEN status NOT IN ('completed', 'cancelled')
                        THEN 'dead_letter'
                        ELSE status
                    END,
                    revision = revision + 1, error = ?, updated_at = ?,
                    completed_at = CASE
                        WHEN status NOT IN ('completed', 'cancelled') THEN ?
                        ELSE completed_at
                    END
                WHERE run_id = ?
                """,
                (reason, _iso(now), _iso(now), run_id),
            )
            _set_run_state(
                conn,
                run_id,
                RunStatus.DEAD_LETTER,
                now=now,
                error=reason,
                clear_lease=True,
                completed=True,
            )
            _insert_event(
                conn,
                run_id,
                name="run.dead_lettered",
                actor=actor,
                payload={"reason": reason},
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, run_id))

    def steer(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        instruction: str,
        actor: str,
        idempotency_key: str,
    ) -> RunRecord:
        instruction = _required(instruction, "steering instruction")
        actor = _required(actor, "steering actor")
        idempotency_key = _required(idempotency_key, "steering idempotency key")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = _run_from_conn(conn, _run_row(conn, run_id))
            _assert_scope(scope, run.scope)
            if run.terminal:
                raise InvalidRunTransitionError("terminal run cannot be steered")
            existing = conn.execute(
                """
                SELECT * FROM durable_run_events
                WHERE run_id = ? AND idempotency_key = ?
                """,
                (run_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                payload = _object(existing["payload_json"])
                if payload.get("instruction") != instruction:
                    raise RunConflictError(
                        "steering idempotency key was reused for another instruction"
                    )
                return run
            metadata = dict(run.metadata)
            steering = list(metadata.get("steering", []))
            steering.append(
                {
                    "instruction": instruction,
                    "actor": actor,
                    "created_at": _iso(now),
                }
            )
            metadata["steering"] = steering
            conn.execute(
                """
                UPDATE durable_runs
                SET metadata_json = ?, revision = revision + 1, updated_at = ?
                WHERE id = ?
                """,
                (_json(metadata), _iso(now), run_id),
            )
            _insert_event(
                conn,
                run_id,
                name="run.steered",
                actor=actor,
                payload={
                    "instruction": instruction,
                    "status": run.status.value,
                },
                idempotency_key=idempotency_key,
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, run_id))

    def reconcile_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[RunRecord, ...]:
        observed = _observed(now)
        changed: list[RunRecord] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""
                SELECT * FROM durable_runs
                WHERE status = 'running' AND lease_until < ?
                ORDER BY lease_until, id
                {self._recovery_lock_clause()}
                """,
                (_iso(observed),),
            ).fetchall()
            for row in rows:
                run_id = str(row["id"])
                current = _run_row(conn, run_id)
                if (
                    str(current["status"]) != RunStatus.RUNNING.value
                    or current["claim_token"] != row["claim_token"]
                    or current["lease_until"] is None
                    or _decode(str(current["lease_until"])) >= observed
                ):
                    continue
                active_step = conn.execute(
                    """
                    SELECT * FROM durable_run_steps
                    WHERE run_id = ? AND status = 'running'
                    ORDER BY id LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                active_attempt = (
                    _active_attempt(conn, run_id, str(active_step["id"]))
                    if active_step is not None
                    else None
                )
                unsafe_effect = conn.execute(
                    """
                    SELECT * FROM durable_effects
                    WHERE run_id = ? AND status IN ('executing', 'unknown',
                                                    'completed')
                    ORDER BY created_at LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                non_intent_checkpoint = None
                if active_attempt is not None:
                    non_intent_checkpoint = conn.execute(
                        """
                        SELECT 1 FROM durable_run_checkpoints
                        WHERE attempt_id = ? AND kind != 'effect_intent'
                        LIMIT 1
                        """,
                        (str(active_attempt["id"]),),
                    ).fetchone()
                safe_to_requeue = (
                    unsafe_effect is None and non_intent_checkpoint is None
                )
                if safe_to_requeue:
                    if bool(current["cancellation_requested"]):
                        _settle_run_cancellation(
                            conn,
                            run_id,
                            actor="reconciler",
                            reason=(
                                "cancellation settled after the worker lease "
                                "expired before an unsafe effect"
                            ),
                            now=observed,
                        )
                    elif active_attempt is not None:
                        conn.execute(
                            """
                            UPDATE durable_run_attempts
                            SET status = 'failed',
                                error = 'lease expired before work started',
                                completed_at = ?
                            WHERE id = ? AND status = 'running'
                            """,
                            (_iso(observed), str(active_attempt["id"])),
                        )
                    if (
                        not bool(current["cancellation_requested"])
                        and active_step is not None
                    ):
                        conn.execute(
                            """
                            UPDATE durable_run_steps
                            SET status = 'queued', revision = revision + 1,
                                error = NULL, updated_at = ?
                            WHERE run_id = ? AND id = ?
                            """,
                            (
                                _iso(observed),
                                run_id,
                                str(active_step["id"]),
                            ),
                        )
                    if not bool(current["cancellation_requested"]):
                        _set_run_state(
                            conn,
                            run_id,
                            RunStatus.QUEUED,
                            now=observed,
                            clear_lease=True,
                        )
                        _insert_event(
                            conn,
                            run_id,
                            name="run.requeued",
                            actor="reconciler",
                            payload={"reason": "lease expired before work started"},
                            now=observed,
                        )
                else:
                    if (
                        unsafe_effect is not None
                        and str(unsafe_effect["status"]) == EffectStatus.EXECUTING.value
                    ):
                        conn.execute(
                            """
                            UPDATE durable_effects
                            SET status = 'unknown',
                                reconciliation_reason = ?,
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (
                                "worker lease expired after effect dispatch",
                                _iso(observed),
                                str(unsafe_effect["id"]),
                            ),
                        )
                    if active_attempt is not None:
                        conn.execute(
                            """
                            UPDATE durable_run_attempts
                            SET status = 'unknown', error = ?,
                                completed_at = ?
                            WHERE id = ? AND status = 'running'
                            """,
                            (
                                "worker lease expired at an uncertain checkpoint",
                                _iso(observed),
                                str(active_attempt["id"]),
                            ),
                        )
                    if active_step is not None:
                        conn.execute(
                            """
                            UPDATE durable_run_steps
                            SET status = 'unknown', revision = revision + 1,
                                error = ?, updated_at = ?
                            WHERE run_id = ? AND id = ?
                            """,
                            (
                                "worker lease expired at an uncertain checkpoint",
                                _iso(observed),
                                run_id,
                                str(active_step["id"]),
                            ),
                        )
                    _set_run_state(
                        conn,
                        run_id,
                        RunStatus.UNKNOWN,
                        now=observed,
                        error="worker lease expired at an uncertain checkpoint",
                        clear_lease=True,
                    )
                    _insert_event(
                        conn,
                        run_id,
                        name="run.unknown",
                        actor="reconciler",
                        step_id=(
                            str(active_step["id"]) if active_step is not None else None
                        ),
                        payload={
                            "reason": (
                                "worker lease expired at an uncertain checkpoint"
                            )
                        },
                        now=observed,
                    )
                changed.append(_run_from_conn(conn, _run_row(conn, run_id)))
        return tuple(changed)

    def _terminal_run(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        status: RunStatus,
        event_name: str,
        reason: str,
    ) -> RunRecord:
        reason = _required(reason, "terminal reason")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            uncertain = conn.execute(
                """
                SELECT 1 FROM durable_effects
                WHERE run_id = ? AND status IN ('executing', 'unknown')
                """,
                (claim.run_id,),
            ).fetchone()
            if uncertain is not None:
                raise InvalidRunTransitionError(
                    "run with an uncertain effect must be reconciled"
                )
            conn.execute(
                """
                UPDATE durable_run_attempts
                SET status = 'failed', error = ?, completed_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (reason, _iso(now), claim.run_id),
            )
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = CASE
                        WHEN status = 'running' THEN 'failed'
                        ELSE status
                    END,
                    revision = revision + 1,
                    error = CASE WHEN status = 'running' THEN ? ELSE error END,
                    updated_at = ?,
                    completed_at = CASE
                        WHEN status = 'running' THEN ?
                        ELSE completed_at
                    END
                WHERE run_id = ?
                """,
                (reason, _iso(now), _iso(now), claim.run_id),
            )
            _set_run_state(
                conn,
                claim.run_id,
                status,
                now=now,
                error=reason,
                clear_lease=True,
                completed=True,
            )
            _insert_event(
                conn,
                claim.run_id,
                name=event_name,
                actor=claim.worker_id,
                payload={"reason": reason},
                now=now,
            )
            return _run_from_conn(conn, _run_row(conn, claim.run_id))


__all__ = ["_RunStoreTransitionMixin"]
