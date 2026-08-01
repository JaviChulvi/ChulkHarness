"""Parent-child durable run operations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.hosting.scope import ExecutionScope
from chulk.runs.models import (
    RunClaim,
    RunStatus,
    RunSubmission,
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

import chulk.runs._store_clock as _clock
from chulk.runs.errors import (
    InvalidRunTransitionError,
    RunConflictError,
    RunLeaseError,
    RunNotFoundError,
)
from chulk.runs._store_support import (
    _RunStoreBackend,
    _assert_child_scope,
    _assert_parent_or_child_scope,
    _assert_parent_scope,
    _child_from_row,
    _child_row,
    _child_rows,
    _child_storage_idempotency_key,
    _completion_from_row,
    _completion_row,
    _decode,
    _idempotent_run_row,
    _insert_event,
    _iso,
    _json,
    _object,
    _owned_run,
    _parent_from_conn,
    _parent_row,
    _positive_seconds,
    _progress_from_row,
    _request_child_cancellation,
    _required,
    _run_from_conn,
    _run_row,
    _safe_payload,
    _set_run_state,
    _submit_run,
    _terminal_child_evidence,
)


class _ParentChildRunStoreMixin(_RunStoreBackend):
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
        now = _clock.utc_now()
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
                "child definition revision must match its execution scope agent_version"
            )
        actor = _required(actor, "actor")
        child_budget = run_budget_from_dict(submission.budget)
        now = _clock.utc_now()
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
                    or existing.run.definition_digest != submission.definition_digest
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
                run_budget_from_dict(_object(row["budget_json"])) for row in child_rows
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
            child_row = _child_row(conn, child_run_id)
            parent = _run_from_conn(
                conn,
                _run_row(conn, str(child_row["parent_run_id"])),
            )
            child = _child_from_row(conn, child_row)
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
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("child progress sequence must be positive")
        idempotency_key = _required(
            idempotency_key,
            "child progress idempotency key",
        )
        safe_payload = _safe_payload(payload)
        now = _clock.utc_now()
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
                    raise RunConflictError("child progress idempotency key was reused")
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
        now = _clock.utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = _parent_from_conn(
                conn,
                _parent_row(conn, parent_scope.run_id),
            )
            _assert_parent_scope(parent_scope, parent.run.scope)
            row = _child_row(conn, child_run_id)
            if str(row["parent_run_id"]) != parent.run.id:
                raise RunNotFoundError("child run does not belong to this parent scope")
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
        now = _clock.utc_now()
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
                        "child_run_ids": [child.run.id for child in parent.children],
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
        now = _clock.utc_now()
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
                child.run.status is RunStatus.COMPLETED for child in parent.children
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
        now = _clock.utc_now()
        lease_until = now + timedelta(seconds=lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            parent = _run_from_conn(
                conn,
                _run_row(conn, target_parent_run_id),
            )
            _assert_parent_scope(scope, parent.scope)
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
                WHERE {" AND ".join(clauses)}
                ORDER BY outbox.created_at, outbox.id
                LIMIT 1 {self._parent_completion_claim_lock_clause()}
                """,
                tuple(values),
            ).fetchone()
            if row is None:
                return None
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
                raise RunConflictError("parent completion changed during claim")
            _insert_event(
                conn,
                parent.id,
                name="parent.completion_claimed",
                actor=worker_id,
                payload={"completion_id": str(row["id"])},
                now=now,
            )
            completion = _completion_from_row(_completion_row(conn, str(row["id"])))
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
        now = _clock.utc_now()
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
                raise RunNotFoundError("parent completion delivery was not found")
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
            return _completion_from_row(_completion_row(conn, completion.id))

    def _transition_parent_completion_claim(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        status: ParentCompletionStatus,
        event_name: str,
        reason: str | None,
    ) -> ParentCompletion:
        now = _clock.utc_now()
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
                raise RunLeaseError("parent completion claim is absent or stale")
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
                raise RunLeaseError("parent completion claim is absent or stale")
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
            return _completion_from_row(_completion_row(conn, claim.completion.id))


__all__ = ["_ParentChildRunStoreMixin"]
