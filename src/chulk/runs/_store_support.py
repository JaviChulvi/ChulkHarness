"""Shared queries and persistence helpers for durable run stores."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.hosting.scope import ExecutionScope, ExecutionScopeError
from chulk.redaction import redact_data
from chulk.runs.errors import (
    EffectConflictError,
    InvalidRunTransitionError,
    RunConflictError,
    RunLeaseError,
    RunNotFoundError,
)
from chulk.runs.models import (
    AttemptRecord,
    AttemptStatus,
    Checkpoint,
    EffectRecord,
    EffectStatus,
    ReconciliationDecision,
    RetryPolicy,
    RunClaim,
    RunEvent,
    RunRecord,
    RunStatus,
    RunSubmission,
    StepRecord,
    StepStatus,
)
from chulk.runs.parent_child import (
    ChildRunProgress,
    ChildRunRecord,
    ParentAggregationStatus,
    ParentCompletion,
    ParentCompletionStatus,
    ParentRunPolicy,
    ParentRunRecord,
    run_budget_from_dict,
)

import chulk.runs._store_clock as _clock


class _RunStoreBackend:
    """Connection and locking hooks supplied by the concrete store."""

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        raise NotImplementedError

    def _recovery_lock_clause(self) -> str:
        raise NotImplementedError

    def _parent_completion_claim_lock_clause(self) -> str:
        raise NotImplementedError


class _RunStoreSupportMixin(_RunStoreBackend):
    def submit(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        actor: str = "host",
    ) -> RunRecord:
        actor = _required(actor, "actor")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return _submit_run(
                conn,
                scope,
                submission,
                actor=actor,
                now=now,
            )

    def get(self, scope: ExecutionScope, run_id: str) -> RunRecord:
        with self._connect() as conn:
            record = _run_from_conn(conn, _run_row(conn, run_id))
        _assert_scope(scope, record.scope)
        return record

    def events(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]:
        self.get(scope, run_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM durable_run_events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence
                """,
                (run_id, after_sequence),
            ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def record_event(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        name: str,
        actor: str,
        payload: Mapping[str, Any],
        step_id: str | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunEvent:
        """Append a host-owned lifecycle event without changing run state."""
        name = _required(name, "event name")
        actor = _required(actor, "event actor")
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = _run_from_conn(conn, _run_row(conn, run_id))
            _assert_scope(scope, run.scope)
            return _insert_event(
                conn,
                run_id,
                name=name,
                actor=actor,
                payload=payload,
                step_id=step_id,
                causation_id=causation_id,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                now=now,
            )

    def attempts(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[AttemptRecord, ...]:
        self.get(scope, run_id)
        clause = " AND step_id = ?" if step_id is not None else ""
        parameters: tuple[Any, ...] = (
            (run_id, step_id) if step_id is not None else (run_id,)
        )
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM durable_run_attempts
                WHERE run_id = ?{clause}
                ORDER BY step_id, number
                """,
                parameters,
            ).fetchall()
        return tuple(_attempt_from_row(row) for row in rows)

    def checkpoints(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[Checkpoint, ...]:
        self.get(scope, run_id)
        clause = " AND step_id = ?" if step_id is not None else ""
        parameters: tuple[Any, ...] = (
            (run_id, step_id) if step_id is not None else (run_id,)
        )
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM durable_run_checkpoints
                WHERE run_id = ?{clause}
                ORDER BY sequence
                """,
                parameters,
            ).fetchall()
        return tuple(_checkpoint_from_row(row) for row in rows)

    def effects(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[EffectRecord, ...]:
        self.get(scope, run_id)
        clause = " AND step_id = ?" if step_id is not None else ""
        parameters: tuple[Any, ...] = (
            (run_id, step_id) if step_id is not None else (run_id,)
        )
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM durable_effects
                WHERE run_id = ?{clause}
                ORDER BY created_at, id
                """,
                parameters,
            ).fetchall()
        return tuple(_effect_from_row(row) for row in rows)


def _submit_run(
    conn: sqlite3.Connection,
    scope: ExecutionScope,
    submission: RunSubmission,
    *,
    actor: str,
    now: datetime,
    storage_idempotency_key: str | None = None,
) -> RunRecord:
    duplicate = _idempotent_run_row(
        conn,
        scope,
        submission,
        storage_idempotency_key=storage_idempotency_key,
    )
    if duplicate is not None:
        return _run_from_conn(
            conn,
            _run_row(conn, str(duplicate["id"])),
        )
    try:
        conn.execute(
            """
            INSERT INTO durable_runs (
                id, tenant_id, workspace_id, agent_id, agent_version,
                scope_key, scope_json, idempotency_key, input_digest,
                definition_digest, status, revision,
                cancellation_requested, budget_json, metadata_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, 0,
                      ?, ?, ?, ?)
            """,
            (
                scope.run_id,
                scope.tenant_id,
                scope.workspace_id,
                scope.agent_id,
                scope.agent_version,
                scope.key,
                _json(scope.to_dict()),
                storage_idempotency_key or submission.idempotency_key,
                submission.input_digest,
                submission.definition_digest,
                _json(submission.budget),
                _json(submission.metadata),
                _iso(now),
                _iso(now),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise RunConflictError(f"durable run {scope.run_id!r} already exists") from exc
    for step in submission.steps:
        conn.execute(
            """
            INSERT INTO durable_run_steps (
                run_id, id, name, status, revision, attempt_count,
                retry_json, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, 'queued', 0, 0, ?, ?, ?, ?)
            """,
            (
                scope.run_id,
                step.id,
                step.name,
                _json(step.retry_policy.to_dict()),
                _json(step.metadata),
                _iso(now),
                _iso(now),
            ),
        )
    _insert_event(
        conn,
        scope.run_id,
        name="run.queued",
        actor=actor,
        payload={
            "definition_digest": submission.definition_digest,
            "input_digest": submission.input_digest,
            "source_event_id": submission.source_event_id,
        },
        correlation_id=submission.correlation_id,
        idempotency_key=f"submit:{submission.idempotency_key}",
        now=now,
    )
    return _run_from_conn(conn, _run_row(conn, scope.run_id))


def _idempotent_run_row(
    conn: sqlite3.Connection,
    scope: ExecutionScope,
    submission: RunSubmission,
    *,
    storage_idempotency_key: str | None = None,
) -> sqlite3.Row | None:
    duplicate = conn.execute(
        """
        SELECT * FROM durable_runs
        WHERE tenant_id = ? AND workspace_id = ?
          AND idempotency_key = ?
        """,
        (
            scope.tenant_id,
            scope.workspace_id,
            storage_idempotency_key or submission.idempotency_key,
        ),
    ).fetchone()
    if duplicate is None:
        return None
    existing = _run_from_conn(
        conn,
        _run_row(conn, str(duplicate["id"])),
    )
    _assert_scope(scope, existing.scope)
    if (
        existing.input_digest != submission.input_digest
        or existing.definition_digest != submission.definition_digest
        or dict(existing.budget) != dict(submission.budget)
    ):
        raise RunConflictError(
            "run idempotency key was reused for different input, definition, or budget"
        )
    return duplicate


def _child_storage_idempotency_key(
    parent_run_id: str,
    idempotency_key: str,
) -> str:
    value = json.dumps(
        [parent_run_id, idempotency_key],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return f"child:{hashlib.sha256(value).hexdigest()}"


def _parent_row(conn: sqlite3.Connection, parent_run_id: str) -> sqlite3.Row:
    _run_row(conn, parent_run_id)
    row = conn.execute(
        """
        SELECT * FROM durable_run_parents WHERE parent_run_id = ?
        """,
        (parent_run_id,),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"durable parent run {parent_run_id!r} was not found")
    return row


def _child_row(conn: sqlite3.Connection, child_run_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM durable_child_runs WHERE child_run_id = ?
        """,
        (child_run_id,),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"durable child run {child_run_id!r} was not found")
    return row


def _child_rows(
    conn: sqlite3.Connection,
    parent_run_id: str,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM durable_child_runs
        WHERE parent_run_id = ? ORDER BY ordinal
        """,
        (parent_run_id,),
    ).fetchall()


def _parent_from_conn(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> ParentRunRecord:
    parent_run_id = str(row["parent_run_id"])
    aggregate_value = row["aggregate_result_json"]
    return ParentRunRecord(
        run=_run_from_conn(conn, _run_row(conn, parent_run_id)),
        policy=ParentRunPolicy.from_dict(_object(row["policy_json"])),
        children=tuple(
            _child_from_row(conn, child) for child in _child_rows(conn, parent_run_id)
        ),
        aggregation_status=ParentAggregationStatus(str(row["aggregation_status"])),
        aggregation_revision=int(row["aggregation_revision"]),
        aggregate_result=(
            _object(aggregate_value) if aggregate_value is not None else None
        ),
    )


def _child_from_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> ChildRunRecord:
    child_run_id = str(row["child_run_id"])
    progress_rows = conn.execute(
        """
        SELECT * FROM durable_child_progress
        WHERE child_run_id = ? ORDER BY sequence
        """,
        (child_run_id,),
    ).fetchall()
    evidence_value = row["terminal_evidence_json"]
    return ChildRunRecord(
        parent_run_id=str(row["parent_run_id"]),
        run=_run_from_conn(
            conn,
            _run_row(conn, child_run_id),
            idempotency_key=str(row["idempotency_key"]),
        ),
        ordinal=int(row["ordinal"]),
        definition_revision=str(row["definition_revision"]),
        progress=tuple(_progress_from_row(item) for item in progress_rows),
        terminal_evidence=(
            _object(evidence_value) if evidence_value is not None else None
        ),
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
    )


def _progress_from_row(row: sqlite3.Row) -> ChildRunProgress:
    return ChildRunProgress(
        child_run_id=str(row["child_run_id"]),
        sequence=int(row["sequence"]),
        payload=_object(row["payload_json"]),
        actor=str(row["actor"]),
        idempotency_key=str(row["idempotency_key"]),
        created_at=_decode(str(row["created_at"])),
    )


def _completion_row(
    conn: sqlite3.Connection,
    completion_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM durable_parent_completion_outbox WHERE id = ?
        """,
        (completion_id,),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"parent completion {completion_id!r} was not found")
    return row


def _completion_from_row(row: sqlite3.Row) -> ParentCompletion:
    return ParentCompletion(
        id=str(row["id"]),
        parent_run_id=str(row["parent_run_id"]),
        status=ParentCompletionStatus(str(row["status"])),
        payload=_object(row["payload_json"]),
        attempt_count=int(row["attempt_count"]),
        worker_id=str(row["worker_id"]) if row["worker_id"] is not None else None,
        lease_token=(
            str(row["lease_token"]) if row["lease_token"] is not None else None
        ),
        lease_until=_optional_datetime(row["lease_until"]),
        last_error=(str(row["last_error"]) if row["last_error"] is not None else None),
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
        delivered_at=_optional_datetime(row["delivered_at"]),
    )


def _expire_linked_child_run_if_due(
    conn: sqlite3.Connection,
    scope: ExecutionScope,
    run_id: str,
    *,
    actor: str,
    now: datetime,
) -> bool:
    row = conn.execute(
        """
        SELECT runs.scope_json, children.budget_json
        FROM durable_runs AS runs
        JOIN durable_child_runs AS children
          ON children.child_run_id = runs.id
        WHERE runs.id = ?
          AND runs.tenant_id = ? AND runs.workspace_id = ?
          AND runs.agent_id = ? AND runs.agent_version = ?
          AND runs.cancellation_requested = 0
          AND runs.status IN (
              'queued', 'waiting_for_approval', 'waiting_for_retry'
          )
          AND NOT EXISTS (
              SELECT 1 FROM durable_effects AS effects
              WHERE effects.run_id = runs.id
                AND effects.status IN ('executing', 'unknown')
          )
        """,
        (
            run_id,
            scope.tenant_id,
            scope.workspace_id,
            scope.agent_id,
            scope.agent_version,
        ),
    ).fetchone()
    if row is None:
        return False
    persisted_scope = ExecutionScope.from_dict(_object(row["scope_json"]))
    _assert_scope(scope, persisted_scope)
    budget = run_budget_from_dict(_object(row["budget_json"]))
    if budget.deadline is None or budget.deadline > now:
        return False
    current = _run_row(conn, run_id)
    if not bool(current["cancellation_requested"]) and str(current["status"]) in {
        RunStatus.QUEUED.value,
        RunStatus.WAITING_FOR_APPROVAL.value,
        RunStatus.WAITING_FOR_RETRY.value,
    }:
        current_budget = run_budget_from_dict(_object(current["budget_json"]))
        if current_budget.deadline is not None and current_budget.deadline <= now:
            _fail_expired_child_run(
                conn,
                run_id,
                actor=actor,
                deadline=current_budget.deadline,
                now=now,
            )
    return True


def _fail_expired_child_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    actor: str,
    deadline: datetime,
    now: datetime,
) -> None:
    reason = "child run budget deadline expired before claim"
    conn.execute(
        """
        UPDATE durable_run_steps
        SET status = 'failed', revision = revision + 1, error = ?,
            updated_at = ?, completed_at = ?
        WHERE run_id = ?
          AND status NOT IN ('completed', 'failed', 'cancelled', 'dead_letter')
        """,
        (reason, _iso(now), _iso(now), run_id),
    )
    _set_run_state(
        conn,
        run_id,
        RunStatus.FAILED,
        now=now,
        error=reason,
        clear_lease=True,
        completed=True,
    )
    _insert_event(
        conn,
        run_id,
        name="run.failed",
        actor=actor,
        payload={
            "reason": reason,
            "deadline": _iso(deadline),
        },
        now=now,
    )


def _settle_run_cancellation(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    actor: str,
    reason: str,
    now: datetime,
) -> None:
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
        SET status = 'cancelled', revision = revision + 1, error = ?,
            next_retry_at = NULL, updated_at = ?, completed_at = ?
        WHERE run_id = ?
          AND status NOT IN ('completed', 'failed', 'cancelled', 'dead_letter')
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
    _insert_event(
        conn,
        run_id,
        name="run.cancelled",
        actor=actor,
        payload={"reason": reason},
        now=now,
    )


def _request_child_cancellation(
    conn: sqlite3.Connection,
    parent_run_id: str,
    child_run_id: str,
    *,
    actor: str,
    reason: str,
    now: datetime,
) -> None:
    row = _run_row(conn, child_run_id)
    run = _run_from_conn(conn, row)
    if run.terminal:
        return
    if run.cancellation_requested and run.status in {
        RunStatus.RUNNING,
        RunStatus.UNKNOWN,
    }:
        return
    if run.status in {
        RunStatus.QUEUED,
        RunStatus.WAITING_FOR_APPROVAL,
        RunStatus.WAITING_FOR_RETRY,
    }:
        conn.execute(
            """
            UPDATE durable_run_attempts
            SET status = 'cancelled', error = ?, completed_at = ?
            WHERE run_id = ? AND status = 'running'
            """,
            (reason, _iso(now), child_run_id),
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
            (reason, _iso(now), _iso(now), child_run_id),
        )
        _set_run_state(
            conn,
            child_run_id,
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
            (child_run_id,),
        )
        child_event = "run.cancelled"
    else:
        conn.execute(
            """
            UPDATE durable_runs
            SET cancellation_requested = 1, revision = revision + 1,
                updated_at = ?
            WHERE id = ?
            """,
            (_iso(now), child_run_id),
        )
        child_event = "run.cancellation_requested"
    _insert_event(
        conn,
        child_run_id,
        name=child_event,
        actor=actor,
        payload={"reason": reason, "parent_run_id": parent_run_id},
        now=now,
    )
    _insert_event(
        conn,
        parent_run_id,
        name="child.cancellation_requested",
        actor=actor,
        payload={"child_run_id": child_run_id, "reason": reason},
        now=now,
    )


def _terminal_child_evidence(
    conn: sqlite3.Connection,
    run: RunRecord,
) -> dict[str, Any]:
    terminal = conn.execute(
        """
        SELECT * FROM durable_run_events
        WHERE run_id = ?
          AND name IN (
              'run.completed', 'run.failed', 'run.cancelled',
              'run.dead_lettered', 'effect.reconciled'
          )
        ORDER BY sequence DESC LIMIT 1
        """,
        (run.id,),
    ).fetchone()
    if terminal is None:
        raise InvalidRunTransitionError(
            f"child run {run.id!r} has no terminal evidence event"
        )
    effects = conn.execute(
        """
        SELECT id, logical_key, status, result_digest, reconciliation
        FROM durable_effects WHERE run_id = ? ORDER BY created_at, id
        """,
        (run.id,),
    ).fetchall()
    progress = conn.execute(
        """
        SELECT COALESCE(MAX(sequence), 0)
        FROM durable_child_progress WHERE child_run_id = ?
        """,
        (run.id,),
    ).fetchone()
    return {
        "terminal_event_id": str(terminal["id"]),
        "terminal_event_sequence": int(terminal["sequence"]),
        "status": run.status.value,
        "definition_digest": run.definition_digest,
        "input_digest": run.input_digest,
        "result": dict(run.result) if run.result is not None else None,
        "error": run.error,
        "last_progress_sequence": int(progress[0]),
        "effects": [
            {
                "id": str(effect["id"]),
                "logical_key": str(effect["logical_key"]),
                "status": str(effect["status"]),
                "result_digest": (
                    str(effect["result_digest"])
                    if effect["result_digest"] is not None
                    else None
                ),
                "reconciliation": (
                    str(effect["reconciliation"])
                    if effect["reconciliation"] is not None
                    else None
                ),
            }
            for effect in effects
        ],
    }


def _reject_direct_parent_transition(
    conn: sqlite3.Connection,
    run_id: str,
    owner: str,
) -> None:
    configured = conn.execute(
        """
        SELECT 1 FROM durable_run_parents WHERE parent_run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if configured is not None:
        raise InvalidRunTransitionError(
            f"configured parent transitions are owned by {owner}"
        )


def _assert_parent_scope(
    requested: ExecutionScope,
    persisted: ExecutionScope,
) -> None:
    try:
        requested.assert_resumable(persisted)
    except ExecutionScopeError as exc:
        raise RunNotFoundError(
            "durable parent run does not belong to this execution scope"
        ) from exc
    if requested.parent_run_id is not None or persisted.parent_run_id is not None:
        raise RunNotFoundError("parent run scope cannot be a child scope")


def _assert_child_scope(
    parent: ExecutionScope,
    child: ExecutionScope,
) -> None:
    if child.parent_run_id != parent.run_id:
        raise RunNotFoundError("child scope does not name the selected parent")
    if (
        child.tenant_id != parent.tenant_id
        or child.workspace_id != parent.workspace_id
        or child.actor_id != parent.actor_id
    ):
        raise RunNotFoundError("child scope crosses the parent authority boundary")
    if not child.grants.issubset(parent.grants):
        raise RunNotFoundError("child scope broadens parent grants")
    if child.run_id == parent.run_id:
        raise RunConflictError("child run id must differ from parent run id")


def _assert_parent_or_child_scope(
    requested: ExecutionScope,
    parent: ExecutionScope,
    child: ExecutionScope,
) -> None:
    if requested.run_id == parent.run_id:
        _assert_parent_scope(requested, parent)
        return
    if requested.key == child.key:
        return
    raise RunNotFoundError(
        "child run is visible only to its parent or its exact child scope"
    )


def _run_row(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    lock = getattr(conn, "_lock_run", None)
    if lock is not None:
        lock(run_id)
    row = conn.execute(
        "SELECT * FROM durable_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"durable run {run_id!r} was not found")
    return row


def _step_row(
    conn: sqlite3.Connection,
    run_id: str,
    step_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM durable_run_steps
        WHERE run_id = ? AND id = ?
        """,
        (run_id, step_id),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"durable run step {step_id!r} was not found")
    return row


def _effect_row(conn: sqlite3.Connection, effect_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM durable_effects WHERE id = ?",
        (effect_id,),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(f"durable effect {effect_id!r} was not found")
    return row


def _run_from_conn(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    idempotency_key: str | None = None,
) -> RunRecord:
    stored_idempotency_key = str(row["idempotency_key"])
    if idempotency_key is None and stored_idempotency_key.startswith("child:"):
        child = conn.execute(
            """
            SELECT idempotency_key FROM durable_child_runs
            WHERE child_run_id = ?
            """,
            (str(row["id"]),),
        ).fetchone()
        if child is not None:
            idempotency_key = str(child["idempotency_key"])
    step_rows = conn.execute(
        """
        SELECT * FROM durable_run_steps
        WHERE run_id = ? ORDER BY created_at, id
        """,
        (str(row["id"]),),
    ).fetchall()
    result = _object(row["result_json"]) if row["result_json"] is not None else None
    return RunRecord(
        id=str(row["id"]),
        scope=ExecutionScope.from_dict(_object(row["scope_json"])),
        idempotency_key=(
            idempotency_key if idempotency_key is not None else stored_idempotency_key
        ),
        input_digest=str(row["input_digest"]),
        definition_digest=str(row["definition_digest"]),
        status=RunStatus(str(row["status"])),
        revision=int(row["revision"]),
        steps=tuple(_step_from_row(step) for step in step_rows),
        cancellation_requested=bool(row["cancellation_requested"]),
        waiting_reason=(
            str(row["waiting_reason"]) if row["waiting_reason"] is not None else None
        ),
        next_retry_at=_optional_datetime(row["next_retry_at"]),
        budget=_object(row["budget_json"]),
        metadata=_object(row["metadata_json"]),
        result=result,
        error=str(row["error"]) if row["error"] is not None else None,
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
        completed_at=_optional_datetime(row["completed_at"]),
    )


def _step_from_row(row: sqlite3.Row) -> StepRecord:
    return StepRecord(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        name=str(row["name"]),
        status=StepStatus(str(row["status"])),
        revision=int(row["revision"]),
        attempt_count=int(row["attempt_count"]),
        retry_policy=RetryPolicy.from_dict(_object(row["retry_json"])),
        next_retry_at=_optional_datetime(row["next_retry_at"]),
        last_checkpoint_id=(
            str(row["last_checkpoint_id"])
            if row["last_checkpoint_id"] is not None
            else None
        ),
        error=str(row["error"]) if row["error"] is not None else None,
        metadata=_object(row["metadata_json"]),
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
        started_at=_optional_datetime(row["started_at"]),
        completed_at=_optional_datetime(row["completed_at"]),
    )


def _attempt_from_row(row: sqlite3.Row) -> AttemptRecord:
    return AttemptRecord(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        step_id=str(row["step_id"]),
        number=int(row["number"]),
        status=AttemptStatus(str(row["status"])),
        worker_id=str(row["worker_id"]),
        lease_token=str(row["lease_token"]),
        started_at=_decode(str(row["started_at"])),
        completed_at=_optional_datetime(row["completed_at"]),
        error=str(row["error"]) if row["error"] is not None else None,
    )


def _checkpoint_from_row(row: sqlite3.Row) -> Checkpoint:
    return Checkpoint(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        step_id=str(row["step_id"]),
        attempt_id=str(row["attempt_id"]),
        sequence=int(row["sequence"]),
        kind=str(row["kind"]),
        payload=_object(row["payload_json"]),
        created_at=_decode(str(row["created_at"])),
    )


def _effect_from_row(row: sqlite3.Row) -> EffectRecord:
    return EffectRecord(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        step_id=str(row["step_id"]),
        attempt_id=str(row["attempt_id"]),
        logical_key=str(row["logical_key"]),
        tool_name=str(row["tool_name"]),
        tool_version=str(row["tool_version"]),
        schema_version=str(row["schema_version"]),
        arguments_digest=str(row["arguments_digest"]),
        status=EffectStatus(str(row["status"])),
        result_digest=(
            str(row["result_digest"]) if row["result_digest"] is not None else None
        ),
        reconciliation=(
            ReconciliationDecision(str(row["reconciliation"]))
            if row["reconciliation"] is not None
            else None
        ),
        reconciled_by=(
            str(row["reconciled_by"]) if row["reconciled_by"] is not None else None
        ),
        reconciliation_reason=(
            str(row["reconciliation_reason"])
            if row["reconciliation_reason"] is not None
            else None
        ),
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
    )


def _event_from_row(row: sqlite3.Row) -> RunEvent:
    return RunEvent(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        sequence=int(row["sequence"]),
        name=str(row["name"]),
        actor=str(row["actor"]),
        step_id=str(row["step_id"]) if row["step_id"] is not None else None,
        payload=_object(row["payload_json"]),
        causation_id=(
            str(row["causation_id"]) if row["causation_id"] is not None else None
        ),
        correlation_id=(
            str(row["correlation_id"]) if row["correlation_id"] is not None else None
        ),
        idempotency_key=(
            str(row["idempotency_key"]) if row["idempotency_key"] is not None else None
        ),
        created_at=_decode(str(row["created_at"])),
    )


def _owned_run(
    conn: sqlite3.Connection,
    scope: ExecutionScope,
    claim: RunClaim,
    *,
    now: datetime,
) -> sqlite3.Row:
    row = _run_row(conn, claim.run_id)
    persisted = ExecutionScope.from_dict(_object(row["scope_json"]))
    _assert_scope(scope, persisted)
    if claim.scope_key != persisted.key:
        raise RunLeaseError("run claim scope does not match persisted authority")
    if (
        row["claim_token"] != claim.lease_token
        or row["worker_id"] != claim.worker_id
        or row["lease_until"] is None
        or _decode(str(row["lease_until"])) < now
        or str(row["status"]) != RunStatus.RUNNING.value
    ):
        raise RunLeaseError("run lease is stale, expired, or not owned")
    return row


def _active_attempt(
    conn: sqlite3.Connection,
    run_id: str,
    step_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM durable_run_attempts
        WHERE run_id = ? AND step_id = ? AND status = 'running'
        ORDER BY number DESC LIMIT 1
        """,
        (run_id, step_id),
    ).fetchone()
    if row is None:
        raise InvalidRunTransitionError("run step has no active attempt")
    return row


def _mark_active_attempt(
    conn: sqlite3.Connection,
    run_id: str,
    step_id: str,
    status: AttemptStatus,
    now: datetime,
    error: str | None,
) -> None:
    cursor = conn.execute(
        """
        UPDATE durable_run_attempts
        SET status = ?, error = ?, completed_at = ?
        WHERE run_id = ? AND step_id = ? AND status = 'running'
        """,
        (status.value, error, _iso(now), run_id, step_id),
    )
    if cursor.rowcount != 1:
        raise InvalidRunTransitionError("run step has no active attempt")


def _touch_run(
    conn: sqlite3.Connection,
    run_id: str,
    now: datetime,
) -> None:
    cursor = conn.execute(
        """
        UPDATE durable_runs
        SET revision = revision + 1, updated_at = ?
        WHERE id = ?
        """,
        (_iso(now), run_id),
    )
    if cursor.rowcount != 1:
        raise RunConflictError("durable run changed during transition")


def _set_run_state(
    conn: sqlite3.Connection,
    run_id: str,
    status: RunStatus,
    *,
    now: datetime,
    error: str | None = None,
    waiting_reason: str | None = None,
    next_retry_at: datetime | None = None,
    result: Mapping[str, Any] | None = None,
    clear_lease: bool = False,
    completed: bool = False,
) -> None:
    cursor = conn.execute(
        """
        UPDATE durable_runs
        SET status = ?, revision = revision + 1, error = ?,
            waiting_reason = ?, next_retry_at = ?, result_json = ?,
            claim_token = CASE WHEN ? THEN NULL ELSE claim_token END,
            worker_id = CASE WHEN ? THEN NULL ELSE worker_id END,
            lease_until = CASE WHEN ? THEN NULL ELSE lease_until END,
            updated_at = ?,
            completed_at = CASE WHEN ? THEN ? ELSE NULL END
        WHERE id = ?
        """,
        (
            status.value,
            error,
            waiting_reason,
            _iso(next_retry_at) if next_retry_at is not None else None,
            _json(result) if result is not None else None,
            int(clear_lease),
            int(clear_lease),
            int(clear_lease),
            _iso(now),
            int(completed),
            _iso(now),
            run_id,
        ),
    )
    if cursor.rowcount != 1:
        raise RunConflictError("durable run changed during transition")


def _insert_event(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    name: str,
    actor: str,
    payload: Mapping[str, Any],
    now: datetime,
    step_id: str | None = None,
    causation_id: str | None = None,
    correlation_id: str | None = None,
    idempotency_key: str | None = None,
) -> RunEvent:
    _lock_run_sequence_allocation(conn, run_id)
    if idempotency_key is not None:
        existing = conn.execute(
            """
            SELECT * FROM durable_run_events
            WHERE run_id = ? AND idempotency_key = ?
            """,
            (run_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            event = _event_from_row(existing)
            if event.name != name or dict(event.payload) != dict(
                _safe_payload(payload)
            ):
                raise RunConflictError(
                    "run event idempotency key was reused for another event"
                )
            return event
    sequence = int(
        conn.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1
            FROM durable_run_events WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    )
    event = RunEvent(
        id=uuid4().hex,
        run_id=run_id,
        sequence=sequence,
        name=name,
        actor=actor,
        step_id=step_id,
        payload=payload,
        causation_id=causation_id,
        correlation_id=correlation_id,
        idempotency_key=idempotency_key,
        created_at=now,
    )
    conn.execute(
        """
        INSERT INTO durable_run_events (
            id, run_id, sequence, name, actor, step_id, payload_json,
            causation_id, correlation_id, idempotency_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.id,
            event.run_id,
            event.sequence,
            event.name,
            event.actor,
            event.step_id,
            _json(event.payload),
            event.causation_id,
            event.correlation_id,
            event.idempotency_key,
            _iso(event.created_at),
        ),
    )
    return event


def _next_checkpoint_sequence(
    conn: sqlite3.Connection,
    run_id: str,
) -> int:
    _lock_run_sequence_allocation(conn, run_id)
    return int(
        conn.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1
            FROM durable_run_checkpoints WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
    )


def _lock_run_sequence_allocation(
    conn: sqlite3.Connection,
    run_id: str,
) -> None:
    lock = getattr(conn, "_lock_run_sequence_allocation", None)
    if lock is not None:
        lock(run_id)


def _assert_scope(requested: ExecutionScope, persisted: ExecutionScope) -> None:
    try:
        requested.assert_same_authority(persisted)
    except ExecutionScopeError as exc:
        raise RunNotFoundError(
            "durable run does not belong to this execution scope"
        ) from exc
    if (
        requested.parent_run_id is not None or persisted.parent_run_id is not None
    ) and requested.key != persisted.key:
        raise RunNotFoundError("linked child run requires its exact execution scope")


def _safe_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    redacted = redact_data(dict(value))
    if not isinstance(redacted, dict):
        raise ValueError("durable payload must remain an object after redaction")
    return redacted


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _safe_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _object(value: object) -> dict[str, Any]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("stored durable payload is not an object")
    return parsed


def _required(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


def _positive_seconds(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("lease_seconds must be a positive integer")


def _observed(value: datetime | None) -> datetime:
    observed = value or _clock.utc_now()
    if observed.tzinfo is None:
        raise ValueError("now must include a timezone")
    return observed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decode(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _optional_datetime(value: object) -> datetime | None:
    return _decode(str(value)) if value is not None else None


__all__ = [
    "EffectConflictError",
    "InvalidRunTransitionError",
    "RunConflictError",
    "RunLeaseError",
    "RunNotFoundError",
    "_RunStoreBackend",
    "_RunStoreSupportMixin",
]
