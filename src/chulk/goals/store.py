"""Transactional SQLite persistence for revisioned durable goals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
from uuid import uuid4

from chulk.goals.models import (
    Goal,
    GoalActionCheckpoint,
    GoalActionState,
    GoalClaim,
    GoalModelRequest,
    GoalEvidence,
    GoalSliceAdmission,
    GoalStopReason,
    GoalEvent,
    GoalRetentionPolicy,
    GoalStatus,
    GoalStepStatus,
    goal_from_dict,
    verification_context_digest,
)
from chulk.goals.transitions import GoalMutation, mark_step_uncertain, block_step, start_step, complete_step, complete_goal, record_evidence
from chulk.redaction import redact_data
from chulk.storage import initialize_sqlite_database, sqlite_connection
from chulk.storage.private_files import write_private_text


DEFAULT_GOAL_LEASE_SECONDS = 120


class GoalNotFoundError(LookupError):
    """Raised when a goal is absent or belongs to another profile."""


class GoalRevisionConflictError(RuntimeError):
    """Raised when an operator mutates a stale goal revision."""

    def __init__(self, goal_id: str, expected: int, actual: int) -> None:
        self.goal_id = goal_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"goal {goal_id!r} revision conflict: expected {expected}, actual {actual}"
        )


class GoalLeaseConflictError(RuntimeError):
    """Raised when a different or expired runner owns the goal lease."""


class GoalActionConflictError(RuntimeError):
    """Raised when an action checkpoint cannot be safely changed."""


class GoalStore:
    """Profile-scoped goal repository with append-only transition events."""

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
        goal: Goal,
        *,
        actor: str = "operator",
        event_kind: str = "goal.created",
        event_payload: Mapping[str, Any] | None = None,
    ) -> Goal:
        """Persist a new immutable snapshot and its revision-zero event."""
        if goal.profile_id != self.profile_id:
            raise ValueError("goal profile does not match store profile")
        if goal.revision != 0:
            raise ValueError("new goal revision must be zero")
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _insert_goal(conn, goal)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"goal {goal.id!r} already exists") from exc
            _insert_event(
                conn,
                goal,
                kind=event_kind,
                actor=actor,
                payload=event_payload or {},
                now=goal.created_at,
            )
        return goal

    def get(self, goal_id: str) -> Goal:
        with sqlite_connection(self.db_path) as conn:
            row = _goal_row(conn, goal_id, self.profile_id)
        return _goal_from_row(row)

    def list(
        self,
        *,
        status: GoalStatus | str | None = None,
        limit: int = 100,
    ) -> tuple[Goal, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("goal list limit must be between 1 and 10000")
        clauses = ["profile_id = ?"]
        parameters: list[Any] = [self.profile_id]
        if status is not None:
            clauses.append("status = ?")
            parameters.append(GoalStatus(status).value)
        parameters.append(limit)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM goals
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, id
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        return tuple(_goal_from_row(row) for row in rows)

    def events(self, goal_id: str, *, after_revision: int = -1) -> tuple[GoalEvent, ...]:
        self.get(goal_id)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM goal_events
                WHERE profile_id = ? AND goal_id = ? AND revision > ?
                ORDER BY revision
                """,
                (self.profile_id, goal_id, after_revision),
            ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def mutate(
        self,
        goal_id: str,
        *,
        expected_revision: int,
        kind: str,
        actor: str,
        mutation: GoalMutation,
        payload: Mapping[str, Any] | None = None,
        claim: GoalClaim | None = None,
    ) -> Goal:
        """Apply one pure transition under a transactional revision CAS."""
        now = self._now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if claim is not None:
                if claim.goal_id != goal_id:
                    raise GoalLeaseConflictError("claim belongs to another goal")
                _assert_claim_owner(conn, claim, profile_id=self.profile_id, now=now)
            row = _goal_row(conn, goal_id, self.profile_id)
            current = _goal_from_row(row)
            if current.revision != expected_revision:
                raise GoalRevisionConflictError(
                    goal_id,
                    expected_revision,
                    current.revision,
                )
            changed = mutation(current)
            if changed.id != current.id or changed.profile_id != current.profile_id:
                raise ValueError("goal mutation cannot change goal ownership or identity")
            if changed.revision != current.revision:
                raise ValueError("goal mutation cannot assign its own revision")
            if changed == current:
                return current
            lease_until = _optional_datetime(row["lease_until"])
            if (
                current.status is not GoalStatus.RUNNING
                and changed.status is GoalStatus.RUNNING
                and lease_until is not None and lease_until >= now
            ):
                raise GoalLeaseConflictError("wait for the current goal execution to drain before resuming")
            updated = changed.with_revision(current.revision + 1, now=now)
            cursor = conn.execute(
                """
                UPDATE goals
                SET title = ?, status = ?, revision = ?, snapshot_json = ?,
                    cancellation_requested = ?, updated_at = ?, completed_at = ?,
                    claim_token = CASE
                        WHEN ? IN ('completed', 'cancelled', 'failed') THEN NULL
                        ELSE claim_token
                    END,
                    runner_id = CASE
                        WHEN ? IN ('completed', 'cancelled', 'failed') THEN NULL
                        ELSE runner_id
                    END,
                    lease_until = CASE
                        WHEN ? IN ('completed', 'cancelled', 'failed') THEN NULL
                        ELSE lease_until
                    END
                WHERE id = ? AND profile_id = ? AND revision = ?
                """,
                (
                    updated.title,
                    updated.status.value,
                    updated.revision,
                    _json(updated.to_dict()),
                    int(updated.cancellation_requested),
                    updated.updated_at.isoformat(),
                    _iso(updated.completed_at),
                    updated.status.value if claim is None else "owned",
                    updated.status.value if claim is None else "owned",
                    updated.status.value if claim is None else "owned",
                    updated.id,
                    updated.profile_id,
                    current.revision,
                ),
            )
            if cursor.rowcount != 1:
                fresh = _goal_from_row(_goal_row(conn, goal_id, self.profile_id))
                raise GoalRevisionConflictError(
                    goal_id,
                    expected_revision,
                    fresh.revision,
                )
            _insert_event(
                conn,
                updated,
                kind=kind,
                actor=actor,
                payload=payload or {},
                now=now,
            )
        return updated

    def claim(
        self,
        goal_id: str,
        *,
        runner_id: str,
        expected_revision: int,
        lease_seconds: int = DEFAULT_GOAL_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> GoalClaim:
        """Acquire the single-runner lease after validating a current revision."""
        clean_runner = runner_id.strip()
        if not clean_runner:
            raise ValueError("runner_id cannot be empty")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        observed = self._now(now)
        lease_until = observed + timedelta(seconds=lease_seconds)
        claim_token = uuid4().hex
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = _goal_row(conn, goal_id, self.profile_id)
            goal = _goal_from_row(row)
            if goal.revision != expected_revision:
                raise GoalRevisionConflictError(
                    goal_id,
                    expected_revision,
                    goal.revision,
                )
            if goal.status is not GoalStatus.RUNNING:
                raise GoalLeaseConflictError("only running goals can be claimed")
            if goal.cancellation_requested:
                raise GoalLeaseConflictError("goal cancellation is pending")
            existing_lease = _optional_datetime(row["lease_until"])
            if existing_lease is not None and existing_lease >= observed:
                raise GoalLeaseConflictError("goal already has an active runner lease")
            cursor = conn.execute(
                """
                UPDATE goals
                SET claim_token = ?, runner_id = ?, lease_until = ?, updated_at = ?
                WHERE id = ? AND profile_id = ? AND revision = ?
                  AND status = 'running' AND cancellation_requested = 0
                  AND (lease_until IS NULL OR lease_until < ?)
                """,
                (
                    claim_token,
                    clean_runner,
                    lease_until.isoformat(),
                    observed.isoformat(),
                    goal_id,
                    self.profile_id,
                    expected_revision,
                    observed.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                raise GoalLeaseConflictError("goal lease changed concurrently")
        return GoalClaim(
            goal_id=goal_id,
            profile_id=self.profile_id,
            runner_id=clean_runner,
            claim_token=claim_token,
            lease_until=lease_until,
        )

    def heartbeat(
        self,
        claim: GoalClaim,
        *,
        lease_seconds: int = DEFAULT_GOAL_LEASE_SECONDS,
        now: datetime | None = None,
    ) -> GoalClaim:
        """Renew an unexpired lease only for its exact owner token."""
        if claim.profile_id != self.profile_id:
            raise GoalLeaseConflictError("claim belongs to another profile")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        observed = self._now(now)
        lease_until = observed + timedelta(seconds=lease_seconds)
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE goals
                SET lease_until = ?, updated_at = ?
                WHERE id = ? AND profile_id = ? AND status IN ('running', 'paused', 'blocked')
                  AND claim_token = ? AND runner_id = ? AND lease_until >= ?
                """,
                (
                    lease_until.isoformat(),
                    observed.isoformat(),
                    claim.goal_id,
                    self.profile_id,
                    claim.claim_token,
                    claim.runner_id,
                    observed.isoformat(),
                ),
            )
        if cursor.rowcount != 1:
            raise GoalLeaseConflictError("goal lease is absent, expired, or cancelled")
        return replace(claim, lease_until=lease_until)

    def release_claim(self, claim: GoalClaim) -> bool:
        if claim.profile_id != self.profile_id:
            return False
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE goals
                SET claim_token = NULL, runner_id = NULL, lease_until = NULL,
                    updated_at = ?
                WHERE id = ? AND profile_id = ? AND claim_token = ? AND runner_id = ?
                """,
                (
                    self._now().isoformat(),
                    claim.goal_id,
                    self.profile_id,
                    claim.claim_token,
                    claim.runner_id,
                ),
            )
        return cursor.rowcount == 1

    def has_active_claim(
        self,
        goal_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        observed = self._now(now)
        with sqlite_connection(self.db_path) as conn:
            row = _goal_row(conn, goal_id, self.profile_id)
        lease_until = _optional_datetime(row["lease_until"])
        return (
            row["claim_token"] is not None
            and row["runner_id"] is not None
            and lease_until is not None
            and lease_until >= observed
        )

    def assert_action_boundary(
        self,
        claim: GoalClaim,
        *,
        step_id: str,
        now: datetime | None = None,
    ) -> Goal:
        """Return the latest snapshot only if work may safely start now."""
        observed = self._now(now)
        with sqlite_connection(self.db_path) as conn:
            return _assert_action_boundary(
                conn,
                claim,
                profile_id=self.profile_id,
                step_id=step_id,
                now=observed,
            )

    def begin_model_request(
        self, claim: GoalClaim, *, step_id: str, conversation_id: str,
        turn_id: str, request_index: int, purpose: str, goal_revision: int,
        steering_ids: tuple[str, ...],
    ) -> GoalModelRequest:
        """Persist the exact mandatory context at the request admission boundary."""
        now = self._now()
        receipt = GoalModelRequest(
            id=uuid4().hex, goal_id=claim.goal_id, profile_id=self.profile_id,
            step_id=step_id, conversation_id=conversation_id, turn_id=turn_id,
            request_index=request_index, purpose=purpose, goal_revision=goal_revision,
            steering_ids=steering_ids, created_at=now,
        )
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            goal = _assert_action_boundary(conn, claim, profile_id=self.profile_id,
                                          step_id=step_id, now=now)
            if goal.revision != goal_revision:
                raise GoalRevisionConflictError(goal.id, goal_revision, goal.revision)
            if steering_ids != tuple(item.id for item in goal.active_steering):
                raise GoalActionConflictError("model request omitted active steering")
            existing = conn.execute("""
                SELECT * FROM goal_model_requests
                WHERE conversation_id = ? AND turn_id = ? AND request_index = ?
            """, (conversation_id, turn_id, request_index)).fetchone()
            if existing is not None:
                raise GoalActionConflictError("model request already exists; reconcile before dispatch")
            conn.execute("""
                INSERT INTO goal_model_requests (
                    id, goal_id, profile_id, step_id, conversation_id, turn_id,
                    request_index, purpose, goal_revision, steering_ids_json,
                    claim_token, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (receipt.id, receipt.goal_id, receipt.profile_id, receipt.step_id,
                  receipt.conversation_id, receipt.turn_id, receipt.request_index,
                  receipt.purpose, receipt.goal_revision, _json(list(steering_ids)),
                  claim.claim_token, now.isoformat()))
        return receipt

    def acknowledge_model_response(
        self, claim: GoalClaim, request_id: str, *, response_ref: str,
    ) -> GoalModelRequest:
        """Acknowledge only after the owning session/journal persisted its response.

        A current runner can reconcile a prior receipt using a durable response
        reference. This updates incorporation, never evidence or completion.
        """
        clean_ref = response_ref.strip()
        if not clean_ref:
            raise ValueError("a durable response reference is required")
        now = self._now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _assert_claim_owner(conn, claim, profile_id=self.profile_id, now=now)
            row = conn.execute("""
                SELECT * FROM goal_model_requests
                WHERE id = ? AND goal_id = ? AND profile_id = ?
            """, (request_id, claim.goal_id, self.profile_id)).fetchone()
            if row is None:
                raise GoalActionConflictError("goal model request was not found")
            current = _model_request_from_row(row)
            if current.response_ref is not None:
                if current.response_ref != clean_ref:
                    raise GoalActionConflictError("model response reference changed")
                return current
            conn.execute("""
                UPDATE goal_model_requests SET response_ref = ?, incorporated_at = ?
                WHERE id = ? AND incorporated_at IS NULL
            """, (clean_ref, now.isoformat(), request_id))
            for steering_id in current.steering_ids:
                conn.execute("""
                    INSERT INTO goal_steering_incorporations
                        (goal_id, profile_id, steering_id, request_id)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(goal_id, steering_id) DO UPDATE SET request_id = excluded.request_id
                """, (claim.goal_id, self.profile_id, steering_id, request_id))
        return replace(current, response_ref=clean_ref, incorporated_at=now)

    def model_requests(
        self, goal_id: str, *, pending_only: bool = False, limit: int = 100,
    ) -> tuple[GoalModelRequest, ...]:
        self.get(goal_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("model request limit must be between 1 and 10000")
        pending = "AND incorporated_at IS NULL" if pending_only else ""
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(f"""
                SELECT * FROM goal_model_requests WHERE goal_id = ? AND profile_id = ?
                {pending} ORDER BY created_at, id LIMIT ?
            """, (goal_id, self.profile_id, limit)).fetchall()
        return tuple(_model_request_from_row(row) for row in rows)

    def incorporated_steering_ids(self, goal_id: str) -> frozenset[str]:
        self.get(goal_id)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute("""
                SELECT steering_id FROM goal_steering_incorporations
                WHERE goal_id = ? AND profile_id = ?
            """, (goal_id, self.profile_id)).fetchall()
        return frozenset(str(row["steering_id"]) for row in rows)

    def begin_action(
        self,
        claim: GoalClaim,
        *,
        step_id: str,
        idempotency_key: str,
        action_kind: str,
        action_ref: str | None = None,
        now: datetime | None = None,
    ) -> GoalActionCheckpoint:
        """Checkpoint intent before a potentially side-effecting action."""
        observed = self._now(now)
        clean_key = idempotency_key.strip()
        clean_kind = action_kind.strip()
        if not clean_key or not clean_kind:
            raise ValueError("idempotency_key and action_kind are required")
        checkpoint = GoalActionCheckpoint(
            id=uuid4().hex,
            goal_id=claim.goal_id,
            step_id=step_id,
            profile_id=self.profile_id,
            idempotency_key=clean_key,
            action_kind=clean_kind,
            action_ref=action_ref.strip() if action_ref and action_ref.strip() else None,
            state=GoalActionState.STARTED,
            claim_token=claim.claim_token,
            created_at=observed,
            updated_at=observed,
        )
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _assert_action_boundary(
                conn,
                claim,
                profile_id=self.profile_id,
                step_id=step_id,
                now=observed,
            )
            existing = conn.execute(
                """
                SELECT * FROM goal_action_checkpoints
                WHERE goal_id = ? AND idempotency_key = ?
                """,
                (claim.goal_id, clean_key),
            ).fetchone()
            if existing is not None:
                stored = _checkpoint_from_row(existing)
                if (
                    stored.step_id != step_id
                    or stored.action_kind != clean_kind
                    or stored.action_ref != checkpoint.action_ref
                ):
                    raise GoalActionConflictError(
                        "goal action idempotency key was reused for different work"
                    )
                if (
                    stored.state is GoalActionState.STARTED
                    and stored.claim_token != claim.claim_token
                ):
                    raise GoalActionConflictError(
                        "unfinished goal action belongs to an earlier runner claim"
                    )
                raise GoalActionConflictError(
                    "goal action was already checkpointed and will not be replayed"
                )
            conn.execute(
                """
                INSERT INTO goal_action_checkpoints (
                    id, goal_id, step_id, profile_id, idempotency_key,
                    action_kind, action_ref, state, claim_token,
                    result_json, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'started', ?, NULL, NULL, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.goal_id,
                    checkpoint.step_id,
                    checkpoint.profile_id,
                    checkpoint.idempotency_key,
                    checkpoint.action_kind,
                    checkpoint.action_ref,
                    checkpoint.claim_token,
                    checkpoint.created_at.isoformat(),
                    checkpoint.updated_at.isoformat(),
                ),
            )
        return checkpoint

    def finish_action(
        self,
        claim: GoalClaim,
        checkpoint_id: str,
        *,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
        now: datetime | None = None,
    ) -> GoalActionCheckpoint:
        """Terminalize the current runner's action checkpoint exactly once."""
        observed = self._now(now)
        state = GoalActionState.FAILED if error else GoalActionState.COMPLETED
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM goal_action_checkpoints
                WHERE id = ? AND goal_id = ? AND profile_id = ?
                """,
                (checkpoint_id, claim.goal_id, self.profile_id),
            ).fetchone()
            if row is None:
                raise GoalActionConflictError("goal action checkpoint was not found")
            current = _checkpoint_from_row(row)
            if current.state is not GoalActionState.STARTED:
                return current
            if current.claim_token != claim.claim_token:
                raise GoalActionConflictError("goal action belongs to another claim")
            _assert_claim_owner(
                conn,
                claim,
                profile_id=self.profile_id,
                now=observed,
            )
            conn.execute(
                """
                UPDATE goal_action_checkpoints
                SET state = ?, result_json = ?, error = ?, updated_at = ?
                WHERE id = ? AND state = 'started' AND claim_token = ?
                """,
                (
                    state.value,
                    _json(dict(result)) if result is not None else None,
                    error,
                    observed.isoformat(),
                    checkpoint_id,
                    claim.claim_token,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM goal_action_checkpoints WHERE id = ?",
                (checkpoint_id,),
            ).fetchone()
        assert updated is not None
        return _checkpoint_from_row(updated)

    def record_verification(
        self, claim: GoalClaim, *, step_id: str, operation_id: str,
        expected_revision: int, context_digest: str, evidence_digest: str,
        passed: bool, feedback: str,
    ) -> int:
        """Fence and persist a decision; the third unchanged rejection blocks atomically."""
        now = self._now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = _assert_claim_owner(conn, claim, profile_id=self.profile_id, now=now)
            existing = conn.execute(
                "SELECT * FROM goal_verifications WHERE goal_id = ? AND operation_id = ?",
                (claim.goal_id, operation_id),
            ).fetchone()
            if existing is not None:
                if (existing["context_digest"], existing["evidence_digest"], bool(existing["passed"]), existing["feedback"]) != (context_digest, evidence_digest, passed, feedback):
                    raise GoalActionConflictError("verification identity reused with a different decision")
                return int(existing["rejection_count"])
            _assert_action_boundary(conn, claim, profile_id=self.profile_id, step_id=step_id, now=now)
            if current.revision != expected_revision:
                raise GoalRevisionConflictError(current.id, expected_revision, current.revision)
            previous = conn.execute(
                "SELECT * FROM goal_verifications WHERE goal_id = ? AND step_id = ? ORDER BY sequence DESC LIMIT 1",
                (claim.goal_id, step_id),
            ).fetchone()
            count = 0 if passed else 1
            if not passed and previous is not None and previous["context_digest"] == context_digest and previous["evidence_digest"] == evidence_digest:
                count += int(previous["rejection_count"])
            conn.execute("""
                INSERT INTO goal_verifications
                    (goal_id, profile_id, step_id, operation_id, context_digest, evidence_digest,
                     passed, feedback, rejection_count, created_at, goal_revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (claim.goal_id, self.profile_id, step_id, operation_id, context_digest, evidence_digest,
                  int(passed), feedback, count, now.isoformat(), current.revision))
            if count >= 3:
                reason = f"verification_stagnation: three rejections without new evidence. {feedback}"
                changed = block_step(current, step_id, reason).with_revision(current.revision + 1, now=now)
                conn.execute("""UPDATE goals SET status = ?, revision = ?, snapshot_json = ?, updated_at = ?
                                WHERE id = ? AND profile_id = ? AND revision = ? AND claim_token = ?""",
                             (changed.status.value, changed.revision, _json(changed.to_dict()), now.isoformat(),
                              current.id, self.profile_id, current.revision, claim.claim_token))
                _insert_event(conn, changed, kind="goal.blocked", actor=claim.runner_id,
                              payload={"reason": reason, "operation_id": operation_id}, now=now)
        return count

    def execution_state(self, goal_id: str) -> dict[str, Any] | None:
        self.get(goal_id)
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM goal_executions WHERE goal_id = ? AND profile_id = ?",
                               (goal_id, self.profile_id)).fetchone()
        return dict(row) if row is not None else None

    def admit_slice(self, goal_id: str, *, runner_id: str, expected_revision: int,
                    turn_id: str, lease_seconds: int = 120) -> GoalSliceAdmission:
        """Select/start eligible work and claim it with preallocated durable identities."""
        if not runner_id.strip() or not turn_id.strip() or lease_seconds <= 0:
            raise ValueError("runner and turn identities and a positive lease are required")
        now = self._now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = _goal_row(conn, goal_id, self.profile_id)
            goal = _goal_from_row(row)
            if goal.revision != expected_revision:
                raise GoalRevisionConflictError(goal_id, expected_revision, goal.revision)
            if goal.status is not GoalStatus.RUNNING or goal.cancellation_requested:
                raise GoalLeaseConflictError("goal is not admitting work")
            if row["lease_until"] is not None and datetime.fromisoformat(row["lease_until"]) >= now:
                raise GoalLeaseConflictError("goal already has an active runner lease")
            conn.execute("""INSERT INTO goal_executions(goal_id, profile_id, conversation_id)
                            VALUES (?, ?, ?) ON CONFLICT(goal_id) DO NOTHING""",
                         (goal_id, self.profile_id, str(uuid4())))
            execution = conn.execute("SELECT * FROM goal_executions WHERE goal_id = ?", (goal_id,)).fetchone()
            assert execution is not None
            previous = execution["latest_turn_id"]
            active = next((item for item in goal.steps if item.status is GoalStepStatus.RUNNING), None)
            if active is not None and previous is not None:
                prior = conn.execute("SELECT state FROM goal_slices WHERE turn_id = ?", (previous,)).fetchone()
                if prior is not None and prior["state"] == "admitted":
                    return GoalSliceAdmission(goal, None, active.id, execution["conversation_id"], turn_id, previous, GoalStopReason.RECOVERY_REQUIRED)
            selected = active or next((item for item in goal.steps if item.status is GoalStepStatus.READY), None)
            if selected is None:
                reason = GoalStopReason.COMPLETED if not goal.missing_criterion_ids and all(item.status in {GoalStepStatus.COMPLETED, GoalStepStatus.SKIPPED} for item in goal.steps) else GoalStopReason.BLOCKED
                if reason is GoalStopReason.COMPLETED:
                    goal = _commit_execution_goal(conn, goal, complete_goal(goal, now=now), kind="goal.completed", actor=runner_id, now=now)
                return GoalSliceAdmission(goal, None, None, execution["conversation_id"], turn_id, previous, reason)
            if selected.risk.value == "high" and not any(item.scope == "step" and item.step_id == selected.id for item in goal.approvals):
                conn.execute("UPDATE goal_executions SET stop_reason = ? WHERE goal_id = ?", (GoalStopReason.APPROVAL_REQUIRED.value, goal_id))
                return GoalSliceAdmission(goal, None, selected.id, execution["conversation_id"], turn_id, previous, GoalStopReason.APPROVAL_REQUIRED)
            if active is None:
                goal = _commit_execution_goal(conn, goal, start_step(goal, selected.id, now=now), kind="goal.step_started", actor=runner_id, now=now)
                previous = None
            elif previous is None:
                # A pre-existing manually started step must have a known safe boundary.
                if conn.execute("SELECT 1 FROM goal_action_checkpoints WHERE goal_id = ? AND step_id = ? LIMIT 1", (goal_id, selected.id)).fetchone():
                    return GoalSliceAdmission(goal, None, selected.id, execution["conversation_id"], turn_id, None, GoalStopReason.RECOVERY_REQUIRED)
            claim = GoalClaim(goal_id=goal_id, profile_id=self.profile_id, runner_id=runner_id,
                              claim_token=uuid4().hex, lease_until=now + timedelta(seconds=lease_seconds))
            conn.execute("UPDATE goals SET claim_token = ?, runner_id = ?, lease_until = ? WHERE id = ? AND profile_id = ?",
                         (claim.claim_token, runner_id, claim.lease_until.isoformat(), goal_id, self.profile_id))
            conn.execute("""INSERT INTO goal_slices(turn_id, goal_id, profile_id, step_id, claim_token,
                            admitted_revision, state, admitted_at) VALUES (?, ?, ?, ?, ?, ?, 'admitted', ?)""",
                         (turn_id, goal_id, self.profile_id, selected.id, claim.claim_token, goal.revision, now.isoformat()))
            conn.execute("UPDATE goal_executions SET latest_turn_id = ?, latest_step_id = ?, stop_reason = NULL WHERE goal_id = ?",
                         (turn_id, selected.id, goal_id))
            return GoalSliceAdmission(goal, claim, selected.id, execution["conversation_id"], turn_id, previous, new_conversation=execution["latest_turn_id"] is None)

    def finish_slice(self, claim: GoalClaim, *, turn_id: str, reason: GoalStopReason,
                     usage: Mapping[str, Any], exhausted_budget: Mapping[str, Any] | None = None) -> None:
        now = self._now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _assert_claim_owner(conn, claim, profile_id=self.profile_id, now=now)
            conn.execute("UPDATE goal_slices SET state = ?, finished_at = ? WHERE turn_id = ? AND claim_token = ?",
                         (reason.value, now.isoformat(), turn_id, claim.claim_token))
            conn.execute("""UPDATE goal_executions SET stop_reason = ?, usage_json = ?, exhausted_budget_json = ?
                            WHERE goal_id = ? AND profile_id = ? AND latest_turn_id = ?""",
                         (reason.value, _json(dict(usage)), _json(dict(exhausted_budget)) if exhausted_budget else None,
                          claim.goal_id, self.profile_id, turn_id))

    def apply_verified_step(self, claim: GoalClaim, *, operation_id: str, expected_revision: int) -> Goal:
        now = self._now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = _assert_claim_owner(conn, claim, profile_id=self.profile_id, now=now)
            row = conn.execute("SELECT * FROM goal_verifications WHERE goal_id = ? AND operation_id = ? AND profile_id = ?",
                               (claim.goal_id, operation_id, self.profile_id)).fetchone()
            if row is None or not row["passed"]:
                raise GoalActionConflictError("completion requires a persisted passing host verification")
            if row["applied_revision"] is not None:
                return current
            _assert_action_boundary(conn, claim, profile_id=self.profile_id, step_id=row["step_id"], now=now)
            if current.revision != expected_revision or row["context_digest"] != verification_context_digest(current):
                raise GoalRevisionConflictError(current.id, int(row["goal_revision"]), current.revision)
            evidence = GoalEvidence(id=f"verification:{row['sequence']}", summary=row["feedback"],
                                    criterion_ids=current.step(row["step_id"]).acceptance_criterion_ids,
                                    step_id=row["step_id"], kind="verification", reference=f"goal-verification:{row['sequence']}",
                                    recorded_by=claim.runner_id, recorded_at=now)
            changed = complete_step(record_evidence(current, evidence), row["step_id"], now=now)
            if all(item.status in {GoalStepStatus.COMPLETED, GoalStepStatus.SKIPPED} for item in changed.steps) and not changed.missing_criterion_ids:
                changed = complete_goal(changed, now=now)
            updated = _commit_execution_goal(conn, current, changed, kind="goal.progress_verified", actor=claim.runner_id, now=now)
            conn.execute("UPDATE goal_verifications SET applied_revision = ? WHERE sequence = ?", (updated.revision, row["sequence"]))
        return updated

    def retention_candidates(
        self,
        policy: GoalRetentionPolicy,
        *,
        now: datetime | None = None,
    ) -> tuple[Goal, ...]:
        """Return terminal goals old enough for an explicit purge decision."""
        observed = self._now(now)
        candidates: list[Goal] = []
        for goal in self.list(limit=policy.max_export_goals):
            if goal.completed_at is None:
                continue
            retention = policy.retention_for(goal.status)
            if retention is not None and goal.completed_at + retention <= observed:
                candidates.append(goal)
        return tuple(candidates)

    def purge_terminal(self, expected_revisions: Mapping[str, int]) -> tuple[str, ...]:
        """Delete explicitly selected terminal goals using revision CAS checks."""
        if not expected_revisions:
            return ()
        purged: list[str] = []
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for goal_id, expected_revision in expected_revisions.items():
                row = _goal_row(conn, goal_id, self.profile_id)
                goal = _goal_from_row(row)
                if goal.revision != expected_revision:
                    raise GoalRevisionConflictError(
                        goal_id,
                        expected_revision,
                        goal.revision,
                    )
                if not goal.terminal:
                    raise ValueError(
                        f"goal {goal_id!r} is not terminal and cannot be purged"
                    )
                cursor = conn.execute(
                    """
                    DELETE FROM goals
                    WHERE id = ? AND profile_id = ? AND revision = ?
                    """,
                    (goal_id, self.profile_id, expected_revision),
                )
                if cursor.rowcount != 1:
                    raise GoalRevisionConflictError(goal_id, expected_revision, -1)
                purged.append(goal_id)
        return tuple(purged)

    def export(
        self,
        destination: Path | str,
        *,
        goal_id: str | None = None,
        include_events: bool = True,
        max_goals: int = 1_000,
        force: bool = False,
    ) -> Path:
        """Write a bounded, redacted profile-owned JSON export."""
        if not 1 <= max_goals <= 10_000:
            raise ValueError("max_goals must be between 1 and 10000")
        goals = (
            (self.get(goal_id),)
            if goal_id is not None
            else self.list(limit=max_goals)
        )
        payload = {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "goals": [
                {
                    "goal": goal.to_dict(),
                    "events": (
                        [item.to_dict() for item in self.events(goal.id)]
                        if include_events
                        else []
                    ),
                }
                for goal in goals
            ],
        }
        text = json.dumps(
            redact_data(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        return write_private_text(
            destination,
            text,
            overwrite=force,
            private_parent=False,
        )

    def action_checkpoints(
        self,
        goal_id: str,
        *,
        state: GoalActionState | str | None = None,
    ) -> tuple[GoalActionCheckpoint, ...]:
        self.get(goal_id)
        clauses = ["profile_id = ?", "goal_id = ?"]
        parameters: list[Any] = [self.profile_id, goal_id]
        if state is not None:
            clauses.append("state = ?")
            parameters.append(GoalActionState(state).value)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM goal_action_checkpoints
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at, id
                """,
                tuple(parameters),
            ).fetchall()
        return tuple(_checkpoint_from_row(row) for row in rows)

    def recover_expired(
        self,
        *,
        now: datetime | None = None,
        actor: str = "recovery",
    ) -> tuple[Goal, ...]:
        """Block expired running work and mark unfinished actions uncertain."""
        observed = self._now(now)
        recovered: list[Goal] = []
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM goals
                WHERE profile_id = ? AND status IN ('running', 'paused')
                  AND lease_until IS NOT NULL AND lease_until < ?
                ORDER BY updated_at, id
                """,
                (self.profile_id, observed.isoformat()),
            ).fetchall()
            for row in rows:
                goal = _goal_from_row(row)
                active_steps = [
                    item.id
                    for item in goal.steps
                    if item.status is GoalStepStatus.RUNNING
                ]
                changed = goal
                for step_id in active_steps:
                    changed = mark_step_uncertain(
                        changed,
                        step_id,
                        "Runner lease expired after action intent was persisted; "
                        "operator review is required before retry.",
                    )
                if changed is goal:
                    changed = replace(
                        goal,
                        status=GoalStatus.BLOCKED,
                        last_error="Runner lease expired before completion.",
                    )
                updated = changed.with_revision(goal.revision + 1, now=observed)
                conn.execute(
                    """
                    UPDATE goals
                    SET status = ?, revision = ?, snapshot_json = ?,
                        claim_token = NULL, runner_id = NULL, lease_until = NULL,
                        updated_at = ?
                    WHERE id = ? AND profile_id = ? AND revision = ?
                    """,
                    (
                        updated.status.value,
                        updated.revision,
                        _json(updated.to_dict()),
                        observed.isoformat(),
                        updated.id,
                        self.profile_id,
                        goal.revision,
                    ),
                )
                conn.execute(
                    """
                    UPDATE goal_action_checkpoints
                    SET state = 'uncertain', error = ?, updated_at = ?
                    WHERE goal_id = ? AND profile_id = ? AND state = 'started'
                    """,
                    (
                        "Runner lease expired before a durable action result was recorded.",
                        observed.isoformat(),
                        goal.id,
                        self.profile_id,
                    ),
                )
                _insert_event(
                    conn,
                    updated,
                    kind="goal.lease_expired",
                    actor=actor,
                    payload={"uncertain_step_ids": active_steps},
                    now=observed,
                )
                recovered.append(updated)
        return tuple(recovered)

    def _now(self, value: datetime | None = None) -> datetime:
        observed = value or self.clock()
        if observed.tzinfo is None:
            raise ValueError("goal clock must return a timezone-aware datetime")
        return observed.astimezone(timezone.utc)

def _commit_execution_goal(conn: sqlite3.Connection, current: Goal, changed: Goal, *,
                           kind: str, actor: str, now: datetime) -> Goal:
    updated = changed.with_revision(current.revision + 1, now=now)
    cursor = conn.execute("""UPDATE goals SET status = ?, revision = ?, snapshot_json = ?,
                             cancellation_requested = ?, updated_at = ?, completed_at = ?
                             WHERE id = ? AND profile_id = ? AND revision = ?""",
                          (updated.status.value, updated.revision, _json(updated.to_dict()),
                           int(updated.cancellation_requested), now.isoformat(), _iso(updated.completed_at),
                           current.id, current.profile_id, current.revision))
    if cursor.rowcount != 1:
        raise GoalRevisionConflictError(current.id, current.revision, updated.revision)
    _insert_event(conn, updated, kind=kind, actor=actor, payload={}, now=now)
    return updated


SQLiteGoalStore = GoalStore


def _insert_goal(conn: sqlite3.Connection, goal: Goal) -> None:
    conn.execute(
        """
        INSERT INTO goals (
            id, profile_id, title, status, revision, snapshot_json,
            cancellation_requested, created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            goal.id,
            goal.profile_id,
            goal.title,
            goal.status.value,
            goal.revision,
            _json(goal.to_dict()),
            int(goal.cancellation_requested),
            goal.created_at.isoformat(),
            goal.updated_at.isoformat(),
            _iso(goal.completed_at),
        ),
    )


def _insert_event(
    conn: sqlite3.Connection,
    goal: Goal,
    *,
    kind: str,
    actor: str,
    payload: Mapping[str, Any],
    now: datetime,
) -> None:
    clean_kind = kind.strip()
    clean_actor = actor.strip()
    if not clean_kind or not clean_actor:
        raise ValueError("goal event kind and actor are required")
    conn.execute(
        """
        INSERT INTO goal_events (
            id, goal_id, profile_id, revision, kind, actor,
            payload_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid4().hex,
            goal.id,
            goal.profile_id,
            goal.revision,
            clean_kind,
            clean_actor,
            _json(dict(payload)),
            now.isoformat(),
        ),
    )


def _goal_row(
    conn: sqlite3.Connection,
    goal_id: str,
    profile_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM goals WHERE id = ? AND profile_id = ?",
        (goal_id, profile_id),
    ).fetchone()
    if row is None:
        raise GoalNotFoundError(f"goal {goal_id!r} does not exist")
    return row


def _model_request_from_row(row: sqlite3.Row) -> GoalModelRequest:
    return GoalModelRequest(
        id=str(row["id"]), goal_id=str(row["goal_id"]), profile_id=str(row["profile_id"]),
        step_id=str(row["step_id"]), conversation_id=str(row["conversation_id"]),
        turn_id=str(row["turn_id"]), request_index=int(row["request_index"]),
        purpose=str(row["purpose"]), goal_revision=int(row["goal_revision"]),
        steering_ids=tuple(json.loads(row["steering_ids_json"])),
        created_at=_datetime(row["created_at"]), response_ref=row["response_ref"],
        incorporated_at=_optional_datetime(row["incorporated_at"]),
    )


def _goal_from_row(row: sqlite3.Row) -> Goal:
    value = json.loads(str(row["snapshot_json"]))
    if not isinstance(value, dict):
        raise ValueError("stored goal snapshot must be an object")
    value["revision"] = int(row["revision"])
    value["status"] = str(row["status"])
    value["cancellation_requested"] = bool(row["cancellation_requested"])
    value["updated_at"] = str(row["updated_at"])
    value["completed_at"] = row["completed_at"]
    return goal_from_dict(value)


def _assert_action_boundary(
    conn: sqlite3.Connection,
    claim: GoalClaim,
    *,
    profile_id: str,
    step_id: str,
    now: datetime,
) -> Goal:
    goal = _assert_claim_owner(
        conn,
        claim,
        profile_id=profile_id,
        now=now,
    )
    if goal.status is not GoalStatus.RUNNING:
        raise GoalLeaseConflictError("goal is not running")
    if goal.cancellation_requested:
        raise GoalLeaseConflictError("goal cancellation requested before next action")
    step = goal.step(step_id)
    if step.status is not GoalStepStatus.RUNNING:
        raise GoalLeaseConflictError("goal step is not running")
    return goal


def _assert_claim_owner(
    conn: sqlite3.Connection,
    claim: GoalClaim,
    *,
    profile_id: str,
    now: datetime,
) -> Goal:
    row = _goal_row(conn, claim.goal_id, profile_id)
    goal = _goal_from_row(row)
    if row["claim_token"] != claim.claim_token or row["runner_id"] != claim.runner_id:
        raise GoalLeaseConflictError("goal is claimed by another runner")
    lease_until = _optional_datetime(row["lease_until"])
    if lease_until is None or lease_until < now:
        raise GoalLeaseConflictError("goal runner lease expired")
    return goal


def _event_from_row(row: sqlite3.Row) -> GoalEvent:
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        payload = {}
    return GoalEvent(
        id=str(row["id"]),
        goal_id=str(row["goal_id"]),
        profile_id=str(row["profile_id"]),
        revision=int(row["revision"]),
        kind=str(row["kind"]),
        actor=str(row["actor"]),
        payload=payload,
        created_at=_datetime(str(row["created_at"])),
    )


def _checkpoint_from_row(row: sqlite3.Row) -> GoalActionCheckpoint:
    raw_result = row["result_json"]
    result = json.loads(str(raw_result)) if raw_result is not None else None
    if result is not None and not isinstance(result, dict):
        result = {"value": result}
    return GoalActionCheckpoint(
        id=str(row["id"]),
        goal_id=str(row["goal_id"]),
        step_id=str(row["step_id"]),
        profile_id=str(row["profile_id"]),
        idempotency_key=str(row["idempotency_key"]),
        action_kind=str(row["action_kind"]),
        action_ref=str(row["action_ref"]) if row["action_ref"] is not None else None,
        state=GoalActionState(str(row["state"])),
        claim_token=str(row["claim_token"]),
        result=result,
        error=str(row["error"]) if row["error"] is not None else None,
        created_at=_datetime(str(row["created_at"])),
        updated_at=_datetime(str(row["updated_at"])),
    )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("stored goal timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _optional_datetime(value: object) -> datetime | None:
    return _datetime(str(value)) if value is not None else None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = [
    "DEFAULT_GOAL_LEASE_SECONDS",
    "GoalActionConflictError",
    "GoalLeaseConflictError",
    "GoalNotFoundError",
    "GoalRevisionConflictError",
    "GoalStore",
    "SQLiteGoalStore",
]
