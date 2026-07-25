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
    GoalEvent,
    GoalStatus,
    GoalStepStatus,
    goal_from_dict,
)
from chulk.goals.transitions import GoalMutation, mark_step_uncertain
from chulk.storage import initialize_sqlite_database, sqlite_connection


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
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("goal list limit must be between 1 and 1000")
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
    ) -> Goal:
        """Apply one pure transition under a transactional revision CAS."""
        now = self._now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
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
            updated = changed.with_revision(current.revision + 1, now=now)
            cursor = conn.execute(
                """
                UPDATE goals
                SET title = ?, status = ?, revision = ?, snapshot_json = ?,
                    cancellation_requested = ?, updated_at = ?, completed_at = ?
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
        observed = self._now(now)
        lease_until = observed + timedelta(seconds=lease_seconds)
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE goals
                SET lease_until = ?, updated_at = ?
                WHERE id = ? AND profile_id = ? AND status = 'running'
                  AND cancellation_requested = 0
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
            row = _goal_row(conn, claim.goal_id, self.profile_id)
        goal = _goal_from_row(row)
        if (
            row["claim_token"] != claim.claim_token
            or row["runner_id"] != claim.runner_id
        ):
            raise GoalLeaseConflictError("goal is claimed by another runner")
        lease_until = _optional_datetime(row["lease_until"])
        if lease_until is None or lease_until < observed:
            raise GoalLeaseConflictError("goal runner lease expired")
        if goal.status is not GoalStatus.RUNNING:
            raise GoalLeaseConflictError("goal is not running")
        if goal.cancellation_requested:
            raise GoalLeaseConflictError("goal cancellation requested before next action")
        step = goal.step(step_id)
        if step.status is not GoalStepStatus.RUNNING:
            raise GoalLeaseConflictError("goal step is not running")
        return goal

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
        self.assert_action_boundary(claim, step_id=step_id, now=now)
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
            existing = conn.execute(
                """
                SELECT * FROM goal_action_checkpoints
                WHERE goal_id = ? AND idempotency_key = ?
                """,
                (claim.goal_id, clean_key),
            ).fetchone()
            if existing is not None:
                return _checkpoint_from_row(existing)
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
                WHERE profile_id = ? AND status = 'running'
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
