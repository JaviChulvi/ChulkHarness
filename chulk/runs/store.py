"""SQLite reference store for durable hosted runs."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.hosting.scope import ExecutionScope, ExecutionScopeError
from chulk.redaction import redact_data
from chulk.runs.models import (
    AttemptRecord,
    AttemptStatus,
    Checkpoint,
    EffectRecord,
    EffectStatus,
    ReconciliationDecision,
    ReconciliationRecord,
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
    ParentCompletionClaim,
    ParentCompletionStatus,
    ParentRunPolicy,
    ParentRunRecord,
    run_budget_from_dict,
    validate_child_budget_allocation,
)
from chulk.storage import initialize_sqlite_database, sqlite_connection


class RunNotFoundError(LookupError):
    """Raised when a run is absent or outside the caller's authority."""


class RunConflictError(RuntimeError):
    """Raised when an idempotency or optimistic transition conflicts."""


class RunLeaseError(RuntimeError):
    """Raised when a worker does not own a live run lease."""


class InvalidRunTransitionError(ValueError):
    """Raised when a requested run transition is not valid."""


class EffectConflictError(RuntimeError):
    """Raised when an external effect cannot be repeated safely."""


class SQLiteRunStore:
    """Transactional, scope-aware durable-run store with append-only events."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        initialize_sqlite_database(self.db_path)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_connection(self.db_path)

    def _recovery_lock_clause(self) -> str:
        """Return backend-specific locking for expired run workers."""
        return ""

    def _parent_completion_claim_lock_clause(self) -> str:
        """Return backend-specific locking for parent completion claims."""
        return ""

    def submit(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        actor: str = "host",
    ) -> RunRecord:
        actor = _required(actor, "actor")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return _submit_run(
                conn,
                scope,
                submission,
                actor=actor,
                now=now,
            )

    def submit_parent(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        policy: ParentRunPolicy,
        actor: str = "host",
    ) -> ParentRunRecord:
        """Atomically submit a run and freeze its child fan-out policy."""
        actor = _required(actor, "actor")
        if scope.parent_run_id is not None:
            raise InvalidRunTransitionError(
                "a child run cannot be configured as a parent"
            )
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            duplicate = _idempotent_run_row(conn, scope, submission)
            run = _submit_run(
                conn,
                scope,
                submission,
                actor=actor,
                now=now,
            )
            parent_row = conn.execute(
                """
                SELECT * FROM durable_run_parents WHERE parent_run_id = ?
                """,
                (run.id,),
            ).fetchone()
            if parent_row is not None:
                stored_policy = ParentRunPolicy.from_dict(
                    _object(parent_row["policy_json"])
                )
                if stored_policy != policy:
                    raise RunConflictError(
                        "parent run idempotency key was reused with another policy"
                    )
                return _parent_from_conn(conn, parent_row)
            if duplicate is not None:
                raise RunConflictError(
                    "an existing ordinary run cannot be changed into a parent"
                )
            conn.execute(
                """
                INSERT INTO durable_run_parents (
                    parent_run_id, policy_json, aggregation_status,
                    aggregation_revision, created_at, updated_at
                ) VALUES (?, ?, 'open', 0, ?, ?)
                """,
                (
                    run.id,
                    _json(policy.to_dict()),
                    _iso(now),
                    _iso(now),
                ),
            )
            _set_run_state(
                conn,
                run.id,
                RunStatus.WAITING_FOR_CHILDREN,
                now=now,
                waiting_reason="waiting for required child runs",
            )
            _insert_event(
                conn,
                run.id,
                name="run.waiting_for_children",
                actor=actor,
                payload={"policy": policy.to_dict()},
                idempotency_key=f"parent:{submission.idempotency_key}",
                now=now,
            )
            return _parent_from_conn(
                conn,
                _parent_row(conn, run.id),
            )

    def submit_child(
        self,
        parent_scope: ExecutionScope,
        child_scope: ExecutionScope,
        submission: RunSubmission,
        *,
        definition_revision: str,
        actor: str = "host",
    ) -> ChildRunRecord:
        """Atomically validate, submit, and link one bounded child run."""
        definition_revision = _required(
            definition_revision,
            "child definition revision",
        )
        if definition_revision != child_scope.agent_version:
            raise ValueError(
                "child definition revision must match its execution scope "
                "agent_version"
            )
        actor = _required(actor, "actor")
        child_budget = run_budget_from_dict(submission.budget)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = _parent_from_conn(
                conn,
                _parent_row(conn, parent_scope.run_id),
            )
            _assert_parent_scope(parent_scope, parent.run.scope)
            _assert_child_scope(parent.run.scope, child_scope)
            if (
                parent.aggregation_status is not ParentAggregationStatus.OPEN
                or parent.run.terminal
                or parent.run.cancellation_requested
            ):
                raise InvalidRunTransitionError(
                    "terminal or cancelling parent cannot accept child runs"
                )
            duplicate = conn.execute(
                """
                SELECT * FROM durable_child_runs
                WHERE parent_run_id = ? AND idempotency_key = ?
                """,
                (parent.run.id, submission.idempotency_key),
            ).fetchone()
            if duplicate is not None:
                existing = _child_from_row(conn, duplicate)
                if (
                    existing.run.scope.key != child_scope.key
                    or existing.run.input_digest != submission.input_digest
                    or existing.run.definition_digest
                    != submission.definition_digest
                    or existing.definition_revision != definition_revision
                    or dict(existing.run.budget) != dict(submission.budget)
                ):
                    raise RunConflictError(
                        "child idempotency key was reused for different work"
                    )
                return existing
            storage_idempotency_key = _child_storage_idempotency_key(
                parent.run.id,
                submission.idempotency_key,
            )
            if (
                _idempotent_run_row(
                    conn,
                    child_scope,
                    submission,
                    storage_idempotency_key=storage_idempotency_key,
                )
                is not None
            ):
                raise RunConflictError(
                    "child idempotency key already belongs to an unlinked run"
                )
            child_rows = _child_rows(conn, parent.run.id)
            if len(child_rows) >= parent.policy.max_children:
                raise InvalidRunTransitionError(
                    "parent child fan-out limit has been reached"
                )
            existing_budgets = tuple(
                run_budget_from_dict(_object(row["budget_json"]))
                for row in child_rows
            )
            validate_child_budget_allocation(
                parent.policy.budget,
                child_budget,
                existing_budgets,
                observed_at=now,
            )
            child_run = _submit_run(
                conn,
                child_scope,
                submission,
                actor=actor,
                now=now,
                storage_idempotency_key=storage_idempotency_key,
            )
            ordinal = len(child_rows) + 1
            try:
                conn.execute(
                    """
                    INSERT INTO durable_child_runs (
                        child_run_id, parent_run_id, ordinal, idempotency_key,
                        definition_revision, definition_digest, input_digest,
                        budget_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        child_run.id,
                        parent.run.id,
                        ordinal,
                        submission.idempotency_key,
                        definition_revision,
                        submission.definition_digest,
                        submission.input_digest,
                        _json(submission.budget),
                        _iso(now),
                        _iso(now),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RunConflictError(
                    "child run changed during concurrent submission"
                ) from exc
            _insert_event(
                conn,
                parent.run.id,
                name="child.submitted",
                actor=actor,
                payload={
                    "child_run_id": child_run.id,
                    "ordinal": ordinal,
                    "definition_revision": definition_revision,
                    "definition_digest": submission.definition_digest,
                    "input_digest": submission.input_digest,
                },
                idempotency_key=f"child:{submission.idempotency_key}",
                now=now,
            )
            return _child_from_row(
                conn,
                _child_row(conn, child_run.id),
            )

    def get_parent(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> ParentRunRecord:
        with self._connect() as conn:
            parent = _parent_from_conn(conn, _parent_row(conn, parent_run_id))
        _assert_parent_scope(scope, parent.run.scope)
        return parent

    def children(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> tuple[ChildRunRecord, ...]:
        return self.get_parent(scope, parent_run_id).children

    def get_child(
        self,
        scope: ExecutionScope,
        child_run_id: str,
    ) -> ChildRunRecord:
        with self._connect() as conn:
            child = _child_from_row(conn, _child_row(conn, child_run_id))
            parent = _run_from_conn(
                conn,
                _run_row(conn, child.parent_run_id),
            )
        _assert_parent_or_child_scope(scope, parent.scope, child.run.scope)
        return child

    def record_child_progress(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        sequence: int,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> ChildRunProgress:
        """Append progress only from the child's current live lease."""
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
        ):
            raise ValueError("child progress sequence must be positive")
        idempotency_key = _required(
            idempotency_key,
            "child progress idempotency key",
        )
        safe_payload = _safe_payload(payload)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            child_row = _child_row(conn, claim.run_id)
            parent_run_id = str(child_row["parent_run_id"])
            _run_row(conn, parent_run_id)
            _owned_run(conn, scope, claim, now=now)
            duplicate = conn.execute(
                """
                SELECT * FROM durable_child_progress
                WHERE child_run_id = ? AND idempotency_key = ?
                """,
                (claim.run_id, idempotency_key),
            ).fetchone()
            if duplicate is not None:
                existing = _progress_from_row(duplicate)
                if (
                    existing.sequence != sequence
                    or dict(existing.payload) != safe_payload
                    or existing.actor != claim.worker_id
                ):
                    raise RunConflictError(
                        "child progress idempotency key was reused"
                    )
                return existing
            expected = int(
                conn.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM durable_child_progress WHERE child_run_id = ?
                    """,
                    (claim.run_id,),
                ).fetchone()[0]
            )
            if sequence != expected:
                raise RunConflictError(
                    f"child progress sequence must be {expected}, got {sequence}"
                )
            conn.execute(
                """
                INSERT INTO durable_child_progress (
                    child_run_id, sequence, payload_json, actor,
                    idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.run_id,
                    sequence,
                    _json(safe_payload),
                    claim.worker_id,
                    idempotency_key,
                    _iso(now),
                ),
            )
            conn.execute(
                """
                UPDATE durable_child_runs
                SET updated_at = ? WHERE child_run_id = ?
                """,
                (_iso(now), claim.run_id),
            )
            _insert_event(
                conn,
                claim.run_id,
                name="child.progressed",
                actor=claim.worker_id,
                payload={"sequence": sequence, "progress": safe_payload},
                idempotency_key=f"progress:{idempotency_key}",
                now=now,
            )
            _insert_event(
                conn,
                parent_run_id,
                name="child.progressed",
                actor=claim.worker_id,
                payload={
                    "child_run_id": claim.run_id,
                    "sequence": sequence,
                    "progress": safe_payload,
                },
                idempotency_key=f"progress:{claim.run_id}:{idempotency_key}",
                now=now,
            )
            return _progress_from_row(
                conn.execute(
                    """
                    SELECT * FROM durable_child_progress
                    WHERE child_run_id = ? AND sequence = ?
                    """,
                    (claim.run_id, sequence),
                ).fetchone()
            )

    def request_child_cancellation(
        self,
        parent_scope: ExecutionScope,
        child_run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> ChildRunRecord:
        """Request one child cancellation through the owning parent."""
        actor = _required(actor, "cancellation actor")
        reason = _required(reason, "cancellation reason")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = _parent_from_conn(
                conn,
                _parent_row(conn, parent_scope.run_id),
            )
            _assert_parent_scope(parent_scope, parent.run.scope)
            row = _child_row(conn, child_run_id)
            if str(row["parent_run_id"]) != parent.run.id:
                raise RunNotFoundError(
                    "child run does not belong to this parent scope"
                )
            _request_child_cancellation(
                conn,
                parent.run.id,
                child_run_id,
                actor=actor,
                reason=reason,
                now=now,
            )
            return _child_from_row(conn, _child_row(conn, child_run_id))

    def request_parent_cancellation(
        self,
        parent_scope: ExecutionScope,
        *,
        actor: str,
        reason: str,
    ) -> ParentRunRecord:
        """Cancel the parent intent and propagate it to every live child."""
        actor = _required(actor, "cancellation actor")
        reason = _required(reason, "cancellation reason")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = _parent_from_conn(
                conn,
                _parent_row(conn, parent_scope.run_id),
            )
            _assert_parent_scope(parent_scope, parent.run.scope)
            if parent.aggregation_status is ParentAggregationStatus.COMPLETED:
                return parent
            changed = conn.execute(
                """
                UPDATE durable_runs
                SET cancellation_requested = 1, revision = revision + 1,
                    updated_at = ?
                WHERE id = ? AND cancellation_requested = 0
                """,
                (_iso(now), parent.run.id),
            )
            for child in parent.children:
                _request_child_cancellation(
                    conn,
                    parent.run.id,
                    child.run.id,
                    actor=actor,
                    reason=reason,
                    now=now,
                )
            if changed.rowcount == 1:
                _insert_event(
                    conn,
                    parent.run.id,
                    name="run.cancellation_requested",
                    actor=actor,
                    payload={
                        "reason": reason,
                        "status": parent.run.status.value,
                        "child_run_ids": [
                            child.run.id for child in parent.children
                        ],
                    },
                    idempotency_key="parent-cancellation",
                    now=now,
                )
            return _parent_from_conn(
                conn,
                _parent_row(conn, parent.run.id),
            )

    def aggregate_children(
        self,
        parent_scope: ExecutionScope,
        *,
        actor: str,
        idempotency_key: str,
    ) -> ParentRunRecord:
        """Terminalize a parent once all required child outcomes are durable."""
        actor = _required(actor, "aggregation actor")
        idempotency_key = _required(
            idempotency_key,
            "aggregation idempotency key",
        )
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = _parent_from_conn(
                conn,
                _parent_row(conn, parent_scope.run_id),
            )
            _assert_parent_scope(parent_scope, parent.run.scope)
            if parent.aggregation_status is ParentAggregationStatus.COMPLETED:
                return parent
            if (
                not parent.run.cancellation_requested
                and len(parent.children) < parent.policy.required_children
            ):
                raise InvalidRunTransitionError(
                    "parent does not yet have its required child set"
                )
            nonterminal = [
                child.run.id for child in parent.children if not child.run.terminal
            ]
            if nonterminal:
                raise InvalidRunTransitionError(
                    "parent cannot aggregate nonterminal children: "
                    + ", ".join(nonterminal)
                )
            if not parent.children and not parent.run.cancellation_requested:
                raise InvalidRunTransitionError(
                    "parent cannot aggregate an empty child set"
                )
            evidence: list[dict[str, Any]] = []
            for child in parent.children:
                terminal_evidence = _terminal_child_evidence(conn, child.run)
                conn.execute(
                    """
                    UPDATE durable_child_runs
                    SET terminal_evidence_json = ?, updated_at = ?
                    WHERE child_run_id = ?
                    """,
                    (
                        _json(terminal_evidence),
                        _iso(now),
                        child.run.id,
                    ),
                )
                evidence.append(
                    {
                        "child_run_id": child.run.id,
                        "ordinal": child.ordinal,
                        "status": child.run.status.value,
                        "result": (
                            dict(child.run.result)
                            if child.run.result is not None
                            else None
                        ),
                        "error": child.run.error,
                        "terminal_evidence": terminal_evidence,
                    }
                )
            if parent.run.cancellation_requested:
                status = RunStatus.CANCELLED
                reason = "parent cancellation completed after child settlement"
            elif all(
                child.run.status is RunStatus.COMPLETED
                for child in parent.children
            ):
                status = RunStatus.COMPLETED
                reason = None
            else:
                status = RunStatus.FAILED
                reason = "one or more child runs did not complete successfully"
            aggregate = {
                "parent_run_id": parent.run.id,
                "status": status.value,
                "children": evidence,
            }
            step_status = {
                RunStatus.COMPLETED: StepStatus.COMPLETED,
                RunStatus.FAILED: StepStatus.FAILED,
                RunStatus.CANCELLED: StepStatus.CANCELLED,
            }[status]
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = ?, revision = revision + 1, error = ?,
                    updated_at = ?, completed_at = ?
                WHERE run_id = ? AND status != 'completed'
                """,
                (
                    step_status.value,
                    reason,
                    _iso(now),
                    _iso(now),
                    parent.run.id,
                ),
            )
            _set_run_state(
                conn,
                parent.run.id,
                status,
                now=now,
                result=aggregate if status is RunStatus.COMPLETED else None,
                error=reason,
                clear_lease=True,
                completed=True,
            )
            conn.execute(
                """
                UPDATE durable_run_parents
                SET aggregation_status = 'completed',
                    aggregation_revision = aggregation_revision + 1,
                    aggregation_key = ?, aggregate_result_json = ?,
                    updated_at = ?
                WHERE parent_run_id = ? AND aggregation_status = 'open'
                """,
                (
                    idempotency_key,
                    _json(aggregate),
                    _iso(now),
                    parent.run.id,
                ),
            )
            _insert_event(
                conn,
                parent.run.id,
                name="run.children_aggregated",
                actor=actor,
                payload=aggregate,
                idempotency_key=f"aggregate:{idempotency_key}",
                now=now,
            )
            _insert_event(
                conn,
                parent.run.id,
                name={
                    RunStatus.COMPLETED: "run.completed",
                    RunStatus.FAILED: "run.failed",
                    RunStatus.CANCELLED: "run.cancelled",
                }[status],
                actor=actor,
                payload=(
                    {"result": aggregate}
                    if status is RunStatus.COMPLETED
                    else {"reason": reason, "children": evidence}
                ),
                idempotency_key="parent-terminal",
                now=now,
            )
            conn.execute(
                """
                INSERT INTO durable_parent_completion_outbox (
                    id, parent_run_id, status, payload_json, attempt_count,
                    created_at, updated_at
                ) VALUES (?, ?, 'pending', ?, 0, ?, ?)
                """,
                (
                    uuid4().hex,
                    parent.run.id,
                    _json(aggregate),
                    _iso(now),
                    _iso(now),
                ),
            )
            return _parent_from_conn(
                conn,
                _parent_row(conn, parent.run.id),
            )

    def parent_completion(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> ParentCompletion | None:
        """Inspect the terminal parent delivery without claiming it."""
        self.get_parent(scope, parent_run_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM durable_parent_completion_outbox
                WHERE parent_run_id = ?
                """,
                (parent_run_id,),
            ).fetchone()
        return _completion_from_row(row) if row is not None else None

    def claim_parent_completion(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        parent_run_id: str | None = None,
    ) -> ParentCompletionClaim | None:
        """Claim one pending completion; expired ambiguous claims fail closed."""
        worker_id = _required(worker_id, "completion worker id")
        _positive_seconds(lease_seconds)
        if scope.parent_run_id is not None:
            raise RunNotFoundError(
                "parent completion cannot be claimed from a child scope"
            )
        target_parent_run_id = parent_run_id or scope.run_id
        if target_parent_run_id != scope.run_id:
            raise RunNotFoundError(
                "parent completion does not belong to this execution scope"
            )
        now = _utc_now()
        lease_until = now + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            expired = conn.execute(
                f"""
                SELECT outbox.* FROM durable_parent_completion_outbox AS outbox
                JOIN durable_runs AS run ON run.id = outbox.parent_run_id
                WHERE run.tenant_id = ? AND run.workspace_id = ?
                  AND run.agent_id = ? AND run.agent_version = ?
                  AND outbox.parent_run_id = ?
                  AND outbox.status = 'claimed'
                  AND outbox.lease_until IS NOT NULL
                  AND outbox.lease_until <= ?
                ORDER BY outbox.created_at, outbox.id
                LIMIT 1 {self._parent_completion_claim_lock_clause()}
                """,
                (
                    scope.tenant_id,
                    scope.workspace_id,
                    scope.agent_id,
                    scope.agent_version,
                    target_parent_run_id,
                    _iso(now),
                ),
            ).fetchone()
            if expired is not None:
                conn.execute(
                    """
                    UPDATE durable_parent_completion_outbox
                    SET status = 'unknown', worker_id = NULL,
                        lease_token = NULL, lease_until = NULL,
                        last_error = ?, updated_at = ?
                    WHERE id = ? AND status = 'claimed'
                    """,
                    (
                        "delivery lease expired with unknown outcome",
                        _iso(now),
                        str(expired["id"]),
                    ),
                )
                _insert_event(
                    conn,
                    str(expired["parent_run_id"]),
                    name="parent.completion_unknown",
                    actor="recovery",
                    payload={
                        "completion_id": str(expired["id"]),
                        "reason": "delivery lease expired with unknown outcome",
                    },
                    idempotency_key=f"completion-expired:{expired['id']}",
                    now=now,
                )
            clauses = [
                "run.tenant_id = ?",
                "run.workspace_id = ?",
                "run.agent_id = ?",
                "run.agent_version = ?",
                "outbox.parent_run_id = ?",
                "outbox.status = 'pending'",
            ]
            values: list[Any] = [
                scope.tenant_id,
                scope.workspace_id,
                scope.agent_id,
                scope.agent_version,
                target_parent_run_id,
            ]
            row = conn.execute(
                f"""
                SELECT outbox.* FROM durable_parent_completion_outbox AS outbox
                JOIN durable_runs AS run ON run.id = outbox.parent_run_id
                WHERE {' AND '.join(clauses)}
                ORDER BY outbox.created_at, outbox.id
                LIMIT 1 {self._parent_completion_claim_lock_clause()}
                """,
                tuple(values),
            ).fetchone()
            if row is None:
                return None
            parent = _run_from_conn(
                conn,
                _run_row(conn, str(row["parent_run_id"])),
            )
            _assert_parent_scope(scope, parent.scope)
            token = uuid4().hex
            cursor = conn.execute(
                """
                UPDATE durable_parent_completion_outbox
                SET status = 'claimed', attempt_count = attempt_count + 1,
                    worker_id = ?, lease_token = ?, lease_until = ?,
                    last_error = NULL, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (
                    worker_id,
                    token,
                    _iso(lease_until),
                    _iso(now),
                    str(row["id"]),
                ),
            )
            if cursor.rowcount != 1:
                raise RunConflictError(
                    "parent completion changed during claim"
                )
            _insert_event(
                conn,
                parent.id,
                name="parent.completion_claimed",
                actor=worker_id,
                payload={"completion_id": str(row["id"])},
                now=now,
            )
            completion = _completion_from_row(
                _completion_row(conn, str(row["id"]))
            )
            return ParentCompletionClaim(
                completion=completion,
                worker_id=worker_id,
                lease_token=token,
                lease_until=lease_until,
            )

    def complete_parent_completion(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
    ) -> ParentCompletion:
        """Acknowledge a definitely delivered parent completion."""
        return self._transition_parent_completion_claim(
            scope,
            claim,
            status=ParentCompletionStatus.DELIVERED,
            event_name="parent.completion_delivered",
            reason=None,
        )

    def fail_parent_completion(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        reason: str,
    ) -> ParentCompletion:
        """Release a claim after a definite pre-delivery failure."""
        return self._transition_parent_completion_claim(
            scope,
            claim,
            status=ParentCompletionStatus.PENDING,
            event_name="parent.completion_failed",
            reason=_required(reason, "completion failure reason"),
        )

    def mark_parent_completion_unknown(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        reason: str,
    ) -> ParentCompletion:
        """Quarantine an ambiguous delivery until host reconciliation."""
        return self._transition_parent_completion_claim(
            scope,
            claim,
            status=ParentCompletionStatus.UNKNOWN,
            event_name="parent.completion_unknown",
            reason=_required(reason, "completion uncertainty reason"),
        )

    def reconcile_parent_completion(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
        *,
        delivered: bool,
        actor: str,
        reason: str,
    ) -> ParentCompletion:
        """Resolve an ambiguous delivery without allowing blind replay."""
        actor = _required(actor, "completion reconciliation actor")
        reason = _required(reason, "completion reconciliation reason")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = _parent_from_conn(
                conn,
                _parent_row(conn, parent_run_id),
            )
            _assert_parent_scope(scope, parent.run.scope)
            row = conn.execute(
                """
                SELECT * FROM durable_parent_completion_outbox
                WHERE parent_run_id = ?
                """,
                (parent_run_id,),
            ).fetchone()
            if row is None:
                raise RunNotFoundError(
                    "parent completion delivery was not found"
                )
            completion = _completion_from_row(row)
            if completion.status is not ParentCompletionStatus.UNKNOWN:
                raise InvalidRunTransitionError(
                    "only unknown parent completion can be reconciled"
                )
            status = (
                ParentCompletionStatus.DELIVERED
                if delivered
                else ParentCompletionStatus.PENDING
            )
            conn.execute(
                """
                UPDATE durable_parent_completion_outbox
                SET status = ?, worker_id = NULL, lease_token = NULL,
                    lease_until = NULL, last_error = ?, updated_at = ?,
                    delivered_at = CASE WHEN ? THEN ? ELSE NULL END
                WHERE id = ? AND status = 'unknown'
                """,
                (
                    status.value,
                    reason,
                    _iso(now),
                    int(delivered),
                    _iso(now),
                    completion.id,
                ),
            )
            _insert_event(
                conn,
                parent_run_id,
                name="parent.completion_reconciled",
                actor=actor,
                payload={
                    "completion_id": completion.id,
                    "delivered": delivered,
                    "reason": reason,
                },
                now=now,
            )
            return _completion_from_row(
                _completion_row(conn, completion.id)
            )

    def _transition_parent_completion_claim(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        status: ParentCompletionStatus,
        event_name: str,
        reason: str | None,
    ) -> ParentCompletion:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = _completion_row(conn, claim.completion.id)
            parent = _parent_from_conn(
                conn,
                _parent_row(conn, str(row["parent_run_id"])),
            )
            _assert_parent_scope(scope, parent.run.scope)
            if (
                str(row["status"]) != ParentCompletionStatus.CLAIMED.value
                or row["lease_token"] != claim.lease_token
                or row["worker_id"] != claim.worker_id
            ):
                raise RunLeaseError(
                    "parent completion claim is absent or stale"
                )
            if (
                status is not ParentCompletionStatus.UNKNOWN
                and _decode(str(row["lease_until"])) <= now
            ):
                raise RunLeaseError("parent completion claim has expired")
            delivered = status is ParentCompletionStatus.DELIVERED
            updated = conn.execute(
                """
                UPDATE durable_parent_completion_outbox
                SET status = ?, worker_id = NULL, lease_token = NULL,
                    lease_until = NULL, last_error = ?, updated_at = ?,
                    delivered_at = CASE WHEN ? THEN ? ELSE delivered_at END
                WHERE id = ? AND status = 'claimed' AND lease_token = ?
                """,
                (
                    status.value,
                    reason,
                    _iso(now),
                    int(delivered),
                    _iso(now),
                    claim.completion.id,
                    claim.lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise RunLeaseError(
                    "parent completion claim is absent or stale"
                )
            _insert_event(
                conn,
                parent.run.id,
                name=event_name,
                actor=claim.worker_id,
                payload={
                    "completion_id": claim.completion.id,
                    "reason": reason,
                },
                now=now,
            )
            return _completion_from_row(
                _completion_row(conn, claim.completion.id)
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
        now = _utc_now()
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
                raise RunNotFoundError(
                    "child scope cannot claim a sibling run"
                )
            run_id = scope.run_id
        now = _utc_now()
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
            if (
                str(row["id"]) != preflight_run_id
                and _expire_linked_child_run_if_due(
                    conn,
                    scope,
                    str(row["id"]),
                    actor=worker_id,
                    now=now,
                )
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
        now = _utc_now()
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
        now = _utc_now()
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
        now = _utc_now()
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

    def begin_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        logical_key: str,
        tool_name: str,
        tool_version: str,
        schema_version: str,
        arguments_digest: str,
    ) -> EffectRecord:
        logical_key = _required(logical_key, "logical effect key")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            attempt = _active_attempt(conn, claim.run_id, step_id)
            existing = conn.execute(
                """
                SELECT * FROM durable_effects
                WHERE run_id = ? AND logical_key = ?
                """,
                (claim.run_id, logical_key),
            ).fetchone()
            if existing is not None:
                effect = _effect_from_row(existing)
                expected = (
                    tool_name,
                    tool_version,
                    schema_version,
                    arguments_digest,
                )
                actual = (
                    effect.tool_name,
                    effect.tool_version,
                    effect.schema_version,
                    effect.arguments_digest,
                )
                if actual != expected:
                    raise EffectConflictError(
                        "logical effect key was reused for different tool input"
                    )
                if effect.status in {
                    EffectStatus.UNKNOWN,
                    EffectStatus.CANCELLED,
                }:
                    raise EffectConflictError(
                        f"effect is {effect.status.value}; reconciliation is required"
                    )
                if effect.status is EffectStatus.FAILED:
                    conn.execute(
                        """
                        UPDATE durable_effects
                        SET attempt_id = ?, status = 'intended',
                            reconciliation = NULL, reconciled_by = NULL,
                            reconciliation_reason = NULL, updated_at = ?
                        WHERE id = ? AND status = 'failed'
                        """,
                        (str(attempt["id"]), _iso(now), effect.id),
                    )
                    _insert_event(
                        conn,
                        claim.run_id,
                        name="effect.retried",
                        actor=claim.worker_id,
                        step_id=step_id,
                        payload={
                            "effect_id": effect.id,
                            "logical_key": effect.logical_key,
                        },
                        now=now,
                    )
                    return _effect_from_row(_effect_row(conn, effect.id))
                if (
                    effect.status is EffectStatus.INTENDED
                    and effect.attempt_id != str(attempt["id"])
                ):
                    conn.execute(
                        """
                        UPDATE durable_effects
                        SET attempt_id = ?, updated_at = ?
                        WHERE id = ? AND status = 'intended'
                        """,
                        (str(attempt["id"]), _iso(now), effect.id),
                    )
                    existing = _effect_row(conn, effect.id)
                    effect = _effect_from_row(existing)
                return effect
            effect_id = uuid4().hex
            conn.execute(
                """
                INSERT INTO durable_effects (
                    id, run_id, step_id, attempt_id, logical_key, tool_name,
                    tool_version, schema_version, arguments_digest, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'intended', ?, ?)
                """,
                (
                    effect_id,
                    claim.run_id,
                    step_id,
                    str(attempt["id"]),
                    logical_key,
                    _required(tool_name, "tool name"),
                    _required(tool_version, "tool version"),
                    _required(schema_version, "schema version"),
                    _required(arguments_digest, "arguments digest"),
                    _iso(now),
                    _iso(now),
                ),
            )
            _touch_run(conn, claim.run_id, now)
            _insert_event(
                conn,
                claim.run_id,
                name="effect.intended",
                actor=claim.worker_id,
                step_id=step_id,
                payload={
                    "effect_id": effect_id,
                    "logical_key": logical_key,
                    "tool_name": tool_name,
                    "tool_version": tool_version,
                    "schema_version": schema_version,
                    "arguments_digest": arguments_digest,
                },
                now=now,
            )
            return _effect_from_row(_effect_row(conn, effect_id))

    def mark_effect_started(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
    ) -> EffectRecord:
        return self._transition_effect(
            scope,
            claim,
            effect_id,
            expected={EffectStatus.INTENDED},
            status=EffectStatus.EXECUTING,
            event_name="effect.started",
        )

    def complete_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        result_digest: str,
    ) -> EffectRecord:
        return self._transition_effect(
            scope,
            claim,
            effect_id,
            expected={EffectStatus.EXECUTING},
            status=EffectStatus.COMPLETED,
            event_name="effect.completed",
            result_digest=_required(result_digest, "result digest"),
        )

    def fail_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord:
        return self._transition_effect(
            scope,
            claim,
            effect_id,
            expected={EffectStatus.INTENDED, EffectStatus.EXECUTING},
            status=EffectStatus.FAILED,
            event_name="effect.failed",
            reason=_required(reason, "effect failure reason"),
        )

    def mark_effect_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord:
        reason = _required(reason, "unknown effect reason")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            effect = _effect_from_row(_effect_row(conn, effect_id))
            if effect.run_id != claim.run_id:
                raise EffectConflictError("effect does not belong to the claimed run")
            if effect.status not in {
                EffectStatus.INTENDED,
                EffectStatus.EXECUTING,
            }:
                raise EffectConflictError(
                    f"effect cannot become unknown from {effect.status.value}"
                )
            conn.execute(
                """
                UPDATE durable_effects
                SET status = 'unknown', reconciliation_reason = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (reason, _iso(now), effect_id),
            )
            _mark_active_attempt(
                conn,
                claim.run_id,
                effect.step_id,
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
                (reason, _iso(now), claim.run_id, effect.step_id),
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
                name="effect.unknown",
                actor=claim.worker_id,
                step_id=effect.step_id,
                payload={"effect_id": effect_id, "reason": reason},
                now=now,
            )
            _insert_event(
                conn,
                claim.run_id,
                name="run.unknown",
                actor=claim.worker_id,
                step_id=effect.step_id,
                payload={"effect_id": effect_id, "reason": reason},
                now=now,
            )
            return _effect_from_row(_effect_row(conn, effect_id))

    def reconcile_effect(
        self,
        scope: ExecutionScope,
        effect_id: str,
        *,
        decision: ReconciliationDecision,
        actor: str,
        reason: str,
        result_digest: str | None = None,
    ) -> ReconciliationRecord:
        decision = ReconciliationDecision(decision)
        actor = _required(actor, "reconciliation actor")
        reason = _required(reason, "reconciliation reason")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute(
                "SELECT run_id FROM durable_effects WHERE id = ?",
                (effect_id,),
            ).fetchone()
            if owner is None:
                raise RunNotFoundError(f"durable effect {effect_id!r} was not found")
            run_id = str(owner["run_id"])
            run = _run_from_conn(conn, _run_row(conn, run_id))
            effect = _effect_from_row(_effect_row(conn, effect_id))
            _assert_scope(scope, run.scope)
            if effect.status is not EffectStatus.UNKNOWN:
                raise EffectConflictError("only unknown effects can be reconciled")
            if decision is ReconciliationDecision.CONFIRMED:
                new_effect_status = EffectStatus.COMPLETED
                new_step_status = (
                    StepStatus.CANCELLED
                    if run.cancellation_requested
                    else StepStatus.QUEUED
                )
                new_run_status = (
                    RunStatus.CANCELLED
                    if run.cancellation_requested
                    else RunStatus.QUEUED
                )
                if result_digest is None:
                    raise ValueError(
                        "confirmed reconciliation requires a result digest"
                    )
            elif decision is ReconciliationDecision.RETRY:
                if run.cancellation_requested:
                    new_effect_status = EffectStatus.CANCELLED
                    new_step_status = StepStatus.CANCELLED
                    new_run_status = RunStatus.CANCELLED
                else:
                    new_effect_status = EffectStatus.INTENDED
                    new_step_status = StepStatus.QUEUED
                    new_run_status = RunStatus.QUEUED
            elif decision is ReconciliationDecision.FAILED:
                new_effect_status = EffectStatus.FAILED
                new_step_status = StepStatus.FAILED
                new_run_status = RunStatus.FAILED
            else:
                new_effect_status = EffectStatus.CANCELLED
                new_step_status = StepStatus.CANCELLED
                new_run_status = RunStatus.CANCELLED
            updated = conn.execute(
                """
                UPDATE durable_effects
                SET status = ?, result_digest = ?, reconciliation = ?,
                    reconciled_by = ?, reconciliation_reason = ?, updated_at = ?
                WHERE id = ? AND status = 'unknown'
                """,
                (
                    new_effect_status.value,
                    result_digest,
                    decision.value,
                    actor,
                    reason,
                    _iso(now),
                    effect_id,
                ),
            )
            if updated.rowcount != 1:
                raise EffectConflictError("only unknown effects can be reconciled")
            terminal = new_step_status in {
                StepStatus.FAILED,
                StepStatus.CANCELLED,
            }
            conn.execute(
                """
                UPDATE durable_run_steps
                SET status = ?, revision = revision + 1, error = ?,
                    next_retry_at = NULL, updated_at = ?,
                    completed_at = CASE WHEN ? THEN ? ELSE NULL END
                WHERE run_id = ? AND id = ?
                """,
                (
                    new_step_status.value,
                    reason if terminal else None,
                    _iso(now),
                    int(terminal),
                    _iso(now),
                    effect.run_id,
                    effect.step_id,
                ),
            )
            _set_run_state(
                conn,
                effect.run_id,
                new_run_status,
                now=now,
                error=reason if terminal else None,
                clear_lease=True,
                completed=terminal,
            )
            _insert_event(
                conn,
                effect.run_id,
                name="effect.reconciled",
                actor=actor,
                step_id=effect.step_id,
                payload={
                    "effect_id": effect_id,
                    "decision": decision.value,
                    "reason": reason,
                    "result_digest": result_digest,
                },
                now=now,
            )
            updated_effect = _effect_from_row(_effect_row(conn, effect_id))
            updated_run = _run_from_conn(conn, _run_row(conn, effect.run_id))
        return ReconciliationRecord(
            effect=updated_effect,
            run=updated_run,
            decision=decision,
        )

    def complete_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        now = _utc_now()
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
        now = _utc_now()
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
            can_retry = (
                retryable
                and counted_attempts < step.retry_policy.max_attempts
            )
            if can_retry and bool(owned["cancellation_requested"]):
                _settle_run_cancellation(
                    conn,
                    claim.run_id,
                    actor=claim.worker_id,
                    reason=(
                        "cancellation settled after retryable step failure: "
                        f"{reason}"
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
        now = _utc_now()
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
        now = _utc_now()
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
        now = _utc_now()
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
        now = _utc_now()
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
        now = _utc_now()
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
        now = _utc_now()
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
        now = _utc_now()
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
                raise InvalidRunTransitionError(
                    "completed run cannot be dead-lettered"
                )
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
        now = _utc_now()
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
                            payload={
                                "reason": "lease expired before work started"
                            },
                            now=observed,
                        )
                else:
                    if unsafe_effect is not None and str(
                        unsafe_effect["status"]
                    ) == EffectStatus.EXECUTING.value:
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
                            str(active_step["id"])
                            if active_step is not None
                            else None
                        ),
                        payload={
                            "reason": (
                                "worker lease expired at an uncertain checkpoint"
                            )
                        },
                        now=observed,
                    )
                changed.append(
                    _run_from_conn(conn, _run_row(conn, run_id))
                )
        return tuple(changed)

    def _transition_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        expected: set[EffectStatus],
        status: EffectStatus,
        event_name: str,
        result_digest: str | None = None,
        reason: str | None = None,
    ) -> EffectRecord:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_run(conn, scope, claim, now=now)
            effect = _effect_from_row(_effect_row(conn, effect_id))
            if effect.run_id != claim.run_id:
                raise EffectConflictError("effect does not belong to the claimed run")
            if effect.status is status:
                return effect
            if effect.status not in expected:
                raise EffectConflictError(
                    f"effect cannot transition from {effect.status.value} "
                    f"to {status.value}"
                )
            conn.execute(
                """
                UPDATE durable_effects
                SET status = ?, result_digest = COALESCE(?, result_digest),
                    reconciliation_reason = COALESCE(?, reconciliation_reason),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    result_digest,
                    reason,
                    _iso(now),
                    effect_id,
                ),
            )
            _touch_run(conn, claim.run_id, now)
            _insert_event(
                conn,
                claim.run_id,
                name=event_name,
                actor=claim.worker_id,
                step_id=effect.step_id,
                payload={
                    "effect_id": effect_id,
                    "logical_key": effect.logical_key,
                    "result_digest": result_digest,
                    "reason": reason,
                },
                now=now,
            )
            return _effect_from_row(_effect_row(conn, effect_id))

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
        now = _utc_now()
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
        raise RunConflictError(
            f"durable run {scope.run_id!r} already exists"
        ) from exc
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
            "run idempotency key was reused for different input, definition, "
            "or budget"
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
        raise RunNotFoundError(
            f"durable parent run {parent_run_id!r} was not found"
        )
    return row


def _child_row(conn: sqlite3.Connection, child_run_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT * FROM durable_child_runs WHERE child_run_id = ?
        """,
        (child_run_id,),
    ).fetchone()
    if row is None:
        raise RunNotFoundError(
            f"durable child run {child_run_id!r} was not found"
        )
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
            _child_from_row(conn, child)
            for child in _child_rows(conn, parent_run_id)
        ),
        aggregation_status=ParentAggregationStatus(
            str(row["aggregation_status"])
        ),
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
        raise RunNotFoundError(
            f"parent completion {completion_id!r} was not found"
        )
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
        last_error=(
            str(row["last_error"]) if row["last_error"] is not None else None
        ),
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
    if (
        not bool(current["cancellation_requested"])
        and str(current["status"])
        in {
            RunStatus.QUEUED.value,
            RunStatus.WAITING_FOR_APPROVAL.value,
            RunStatus.WAITING_FOR_RETRY.value,
        }
    ):
        current_budget = run_budget_from_dict(_object(current["budget_json"]))
        if (
            current_budget.deadline is not None
            and current_budget.deadline <= now
        ):
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
        raise RunNotFoundError(
            "child scope crosses the parent authority boundary"
        )
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
        raise RunNotFoundError(
            f"durable run step {step_id!r} was not found"
        )
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
            idempotency_key
            if idempotency_key is not None
            else stored_idempotency_key
        ),
        input_digest=str(row["input_digest"]),
        definition_digest=str(row["definition_digest"]),
        status=RunStatus(str(row["status"])),
        revision=int(row["revision"]),
        steps=tuple(_step_from_row(step) for step in step_rows),
        cancellation_requested=bool(row["cancellation_requested"]),
        waiting_reason=(
            str(row["waiting_reason"])
            if row["waiting_reason"] is not None
            else None
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
            str(row["result_digest"])
            if row["result_digest"] is not None
            else None
        ),
        reconciliation=(
            ReconciliationDecision(str(row["reconciliation"]))
            if row["reconciliation"] is not None
            else None
        ),
        reconciled_by=(
            str(row["reconciled_by"])
            if row["reconciled_by"] is not None
            else None
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
            str(row["causation_id"])
            if row["causation_id"] is not None
            else None
        ),
        correlation_id=(
            str(row["correlation_id"])
            if row["correlation_id"] is not None
            else None
        ),
        idempotency_key=(
            str(row["idempotency_key"])
            if row["idempotency_key"] is not None
            else None
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
        requested.parent_run_id is not None
        or persisted.parent_run_id is not None
    ) and requested.key != persisted.key:
        raise RunNotFoundError(
            "linked child run requires its exact execution scope"
        )


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
    observed = value or _utc_now()
    if observed.tzinfo is None:
        raise ValueError("now must include a timezone")
    return observed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    "SQLiteRunStore",
]
