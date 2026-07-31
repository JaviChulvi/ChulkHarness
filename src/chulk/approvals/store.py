"""SQLite reference persistence for restart-safe approvals."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
from uuid import uuid4

from chulk.approvals.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalSubmission,
)
from chulk.hosting.scope import ExecutionScope, ExecutionScopeError
from chulk.redaction import redact_data
from chulk.storage import initialize_sqlite_database, sqlite_connection


class ApprovalNotFoundError(LookupError):
    """Raised when an approval is absent or outside the caller scope."""


class ApprovalConflictError(RuntimeError):
    """Raised when an approval decision or consumption conflicts."""


class SQLiteApprovalStore:
    """Transactional approval store sharing the durable run database."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        initialize_sqlite_database(self.db_path)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_connection(self.db_path)

    def _serialize_creation(
        self,
        conn: sqlite3.Connection,
        run_id: str,
    ) -> None:
        """Let backends serialize approval reuse for one owning run."""

    def _serialize_request_mutation(
        self,
        conn: sqlite3.Connection,
        approval_id: str,
    ) -> None:
        """Let backends serialize transitions for one approval request."""

    def create(
        self,
        scope: ExecutionScope,
        submission: ApprovalSubmission,
    ) -> ApprovalRequest:
        now = _utc_now()
        if submission.expires_at <= now:
            raise ValueError("approval expiry must be in the future")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_creation(conn, scope.run_id)
            _assert_run_scope(conn, scope)
            existing = conn.execute(
                """
                SELECT * FROM durable_approval_requests
                WHERE tenant_id = ? AND workspace_id = ? AND run_id = ?
                  AND step_id = ? AND tool_name = ? AND tool_version = ?
                  AND schema_version = ? AND arguments_digest = ?
                  AND policy_version = ? AND status IN ('pending', 'approved')
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (
                    scope.tenant_id,
                    scope.workspace_id,
                    scope.run_id,
                    submission.step_id,
                    submission.tool_name,
                    submission.tool_version,
                    submission.schema_version,
                    submission.arguments_digest,
                    submission.policy_version,
                ),
            ).fetchone()
            if existing is not None:
                return _from_row(existing)
            approval_id = uuid4().hex
            conn.execute(
                """
                INSERT INTO durable_approval_requests (
                    id, tenant_id, workspace_id, run_id, step_id, effect_id,
                    scope_json, tool_name, tool_version, schema_version,
                    arguments_digest, policy_version, preview_json, status,
                    revision, created_at, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending',
                          0, ?, ?, ?)
                """,
                (
                    approval_id,
                    scope.tenant_id,
                    scope.workspace_id,
                    scope.run_id,
                    submission.step_id,
                    submission.effect_id,
                    _json(scope.to_dict()),
                    submission.tool_name,
                    submission.tool_version,
                    submission.schema_version,
                    submission.arguments_digest,
                    submission.policy_version,
                    _json(submission.preview),
                    _iso(now),
                    _iso(submission.expires_at),
                    _iso(now),
                ),
            )
            _audit(
                conn,
                scope,
                "approval.requested",
                {
                    "approval_id": approval_id,
                    "step_id": submission.step_id,
                    "effect_id": submission.effect_id,
                    "tool_name": submission.tool_name,
                    "tool_version": submission.tool_version,
                    "schema_version": submission.schema_version,
                    "arguments_digest": submission.arguments_digest,
                    "policy_version": submission.policy_version,
                    "expires_at": _iso(submission.expires_at),
                },
                idempotency_key=f"approval:create:{approval_id}",
                now=now,
            )
            return _from_row(_row(conn, approval_id))

    def get(
        self,
        scope: ExecutionScope,
        approval_id: str,
    ) -> ApprovalRequest:
        with self._connect() as conn:
            request = _from_row(_row(conn, approval_id))
        _assert_scope(scope, request.scope)
        return request

    def list(
        self,
        scope: ExecutionScope,
        *,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[ApprovalRequest, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
            raise ValueError("approval list limit must be between 1 and 1000")
        clauses = ["tenant_id = ?", "workspace_id = ?"]
        parameters: list[Any] = [scope.tenant_id, scope.workspace_id]
        if run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(ApprovalStatus(status).value)
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM durable_approval_requests
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at, id LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()
        requests = tuple(_from_row(row) for row in rows)
        for request in requests:
            _assert_scope(scope, request.scope)
        return requests

    def decide(
        self,
        scope: ExecutionScope,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        reason: str,
        idempotency_key: str,
    ) -> ApprovalRequest:
        decision = ApprovalDecision(decision)
        decided_by = _required(decided_by, "approver identity")
        reason = _required(reason, "decision reason")
        idempotency_key = _required(idempotency_key, "decision idempotency key")
        now = _utc_now()
        status = (
            ApprovalStatus.APPROVED
            if decision is ApprovalDecision.APPROVE
            else ApprovalStatus.DENIED
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_request_mutation(conn, approval_id)
            request = _from_row(_row(conn, approval_id))
            _assert_scope(scope, request.scope)
            if request.status is not ApprovalStatus.PENDING:
                raw = _row(conn, approval_id)
                if (
                    raw["decision_key"] == idempotency_key
                    and raw["decision"] == decision.value
                ):
                    return request
                raise ApprovalConflictError(
                    f"approval request is already {request.status.value}"
                )
            if request.expires_at <= now:
                conn.execute(
                    """
                    UPDATE durable_approval_requests
                    SET status = 'expired', revision = revision + 1,
                        decision_reason = 'approval request expired',
                        updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (_iso(now), approval_id),
                )
                raise ApprovalConflictError("approval request has expired")
            try:
                cursor = conn.execute(
                    """
                    UPDATE durable_approval_requests
                    SET status = ?, revision = revision + 1, decision = ?,
                        decided_by = ?, decision_reason = ?, decision_key = ?,
                        decided_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'pending' AND revision = ?
                    """,
                    (
                        status.value,
                        decision.value,
                        decided_by,
                        reason,
                        idempotency_key,
                        _iso(now),
                        _iso(now),
                        approval_id,
                        request.revision,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ApprovalConflictError(
                    "decision idempotency key was already used"
                ) from exc
            if cursor.rowcount != 1:
                raise ApprovalConflictError(
                    "approval changed while it was being decided"
                )
            _audit(
                conn,
                request.scope,
                "approval.decided",
                {
                    "approval_id": approval_id,
                    "decision": decision.value,
                    "decided_by": decided_by,
                    "reason": reason,
                    "effect_id": request.effect_id,
                    "arguments_digest": request.arguments_digest,
                },
                idempotency_key=f"approval:decision:{idempotency_key}",
                now=now,
            )
            return _from_row(_row(conn, approval_id))

    def consume(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        expected_revision: int,
    ) -> ApprovalRequest:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_request_mutation(conn, approval_id)
            request = _from_row(_row(conn, approval_id))
            _assert_scope(scope, request.scope)
            if request.status is ApprovalStatus.CONSUMED:
                raise ApprovalConflictError("approval was already consumed")
            if request.status is not ApprovalStatus.APPROVED:
                raise ApprovalConflictError(
                    f"approval cannot be consumed from {request.status.value}"
                )
            if request.expires_at <= now:
                conn.execute(
                    """
                    UPDATE durable_approval_requests
                    SET status = 'expired', revision = revision + 1,
                        decision_reason = 'approval expired before consumption',
                        updated_at = ?
                    WHERE id = ? AND status = 'approved'
                    """,
                    (_iso(now), approval_id),
                )
                raise ApprovalConflictError(
                    "approval expired before consumption"
                )
            cursor = conn.execute(
                """
                UPDATE durable_approval_requests
                SET status = 'consumed', revision = revision + 1,
                    consumed_at = ?, updated_at = ?
                WHERE id = ? AND status = 'approved' AND revision = ?
                """,
                (
                    _iso(now),
                    _iso(now),
                    approval_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ApprovalConflictError(
                    "approval changed before it could be consumed"
                )
            _audit(
                conn,
                request.scope,
                "approval.consumed",
                {
                    "approval_id": approval_id,
                    "decided_by": request.decided_by,
                    "effect_id": request.effect_id,
                    "tool_name": request.tool_name,
                    "tool_version": request.tool_version,
                    "schema_version": request.schema_version,
                    "arguments_digest": request.arguments_digest,
                    "policy_version": request.policy_version,
                },
                idempotency_key=f"approval:consume:{approval_id}",
                now=now,
            )
            return _from_row(_row(conn, approval_id))

    def invalidate(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        reason: str,
    ) -> ApprovalRequest:
        reason = _required(reason, "invalidation reason")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_request_mutation(conn, approval_id)
            request = _from_row(_row(conn, approval_id))
            _assert_scope(scope, request.scope)
            if request.status is ApprovalStatus.INVALIDATED:
                return request
            if request.status not in {
                ApprovalStatus.PENDING,
                ApprovalStatus.APPROVED,
            }:
                raise ApprovalConflictError(
                    f"approval cannot be invalidated from {request.status.value}"
                )
            conn.execute(
                """
                UPDATE durable_approval_requests
                SET status = 'invalidated', revision = revision + 1,
                    decision_reason = ?, updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (reason, _iso(now), approval_id, request.revision),
            )
            _audit(
                conn,
                request.scope,
                "approval.invalidated",
                {
                    "approval_id": approval_id,
                    "reason": reason,
                    "arguments_digest": request.arguments_digest,
                },
                idempotency_key=(
                    f"approval:invalidate:{approval_id}:{request.revision}"
                ),
                now=now,
            )
            return _from_row(_row(conn, approval_id))

    def expire(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[ApprovalRequest, ...]:
        observed = _observed(now)
        expired: list[ApprovalRequest] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM durable_approval_requests
                WHERE status IN ('pending', 'approved') AND expires_at <= ?
                ORDER BY expires_at, id
                """,
                (_iso(observed),),
            ).fetchall()
            for row in rows:
                approval_id = str(row["id"])
                self._serialize_request_mutation(conn, approval_id)
                request = _from_row(_row(conn, approval_id))
                if (
                    request.status
                    not in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}
                    or request.expires_at > observed
                ):
                    continue
                cursor = conn.execute(
                    """
                    UPDATE durable_approval_requests
                    SET status = 'expired', revision = revision + 1,
                        decision_reason = 'approval request expired',
                        updated_at = ?
                    WHERE id = ? AND revision = ?
                    """,
                    (_iso(observed), request.id, request.revision),
                )
                if cursor.rowcount != 1:
                    continue
                _audit(
                    conn,
                    request.scope,
                    "approval.expired",
                    {"approval_id": request.id},
                    idempotency_key=(
                        f"approval:expire:{request.id}:{request.revision}"
                    ),
                    now=observed,
                )
                expired.append(_from_row(_row(conn, request.id)))
        return tuple(expired)

    def cancel(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        reason: str,
    ) -> ApprovalRequest:
        reason = _required(reason, "cancellation reason")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_request_mutation(conn, approval_id)
            request = _from_row(_row(conn, approval_id))
            _assert_scope(scope, request.scope)
            if request.status is ApprovalStatus.CANCELLED:
                return request
            if request.status not in {
                ApprovalStatus.PENDING,
                ApprovalStatus.APPROVED,
            }:
                raise ApprovalConflictError(
                    f"approval cannot be cancelled from {request.status.value}"
                )
            conn.execute(
                """
                UPDATE durable_approval_requests
                SET status = 'cancelled', revision = revision + 1,
                    decision_reason = ?, updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (reason, _iso(now), approval_id, request.revision),
            )
            _audit(
                conn,
                request.scope,
                "approval.cancelled",
                {"approval_id": approval_id, "reason": reason},
                idempotency_key=(
                    f"approval:cancel:{approval_id}:{request.revision}"
                ),
                now=now,
            )
            return _from_row(_row(conn, approval_id))


def _row(conn: sqlite3.Connection, approval_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM durable_approval_requests WHERE id = ?",
        (approval_id,),
    ).fetchone()
    if row is None:
        raise ApprovalNotFoundError(
            f"durable approval {approval_id!r} was not found"
        )
    return row


def _from_row(row: sqlite3.Row) -> ApprovalRequest:
    return ApprovalRequest(
        id=str(row["id"]),
        scope=ExecutionScope.from_dict(_object(row["scope_json"])),
        run_id=str(row["run_id"]),
        step_id=str(row["step_id"]),
        effect_id=(
            str(row["effect_id"]) if row["effect_id"] is not None else None
        ),
        tool_name=str(row["tool_name"]),
        tool_version=str(row["tool_version"]),
        schema_version=str(row["schema_version"]),
        arguments_digest=str(row["arguments_digest"]),
        policy_version=str(row["policy_version"]),
        preview=_object(row["preview_json"]),
        status=ApprovalStatus(str(row["status"])),
        revision=int(row["revision"]),
        decision=(
            ApprovalDecision(str(row["decision"]))
            if row["decision"] is not None
            else None
        ),
        decided_by=(
            str(row["decided_by"])
            if row["decided_by"] is not None
            else None
        ),
        decision_reason=(
            str(row["decision_reason"])
            if row["decision_reason"] is not None
            else None
        ),
        created_at=_decode(str(row["created_at"])),
        expires_at=_decode(str(row["expires_at"])),
        decided_at=_optional_datetime(row["decided_at"]),
        consumed_at=_optional_datetime(row["consumed_at"]),
        updated_at=_decode(str(row["updated_at"])),
    )


def _assert_run_scope(
    conn: sqlite3.Connection,
    scope: ExecutionScope,
) -> None:
    row = conn.execute(
        "SELECT scope_json FROM durable_runs WHERE id = ?",
        (scope.run_id,),
    ).fetchone()
    if row is None:
        raise ApprovalNotFoundError(
            "approval run does not exist in the durable run store"
        )
    persisted = ExecutionScope.from_dict(_object(row["scope_json"]))
    _assert_scope(scope, persisted)


def _assert_scope(
    requested: ExecutionScope,
    persisted: ExecutionScope,
) -> None:
    try:
        requested.assert_resumable(persisted)
    except ExecutionScopeError as exc:
        raise ApprovalNotFoundError(
            "durable approval does not belong to this execution scope"
        ) from exc


def _audit(
    conn: sqlite3.Connection,
    scope: ExecutionScope,
    event_type: str,
    payload: Mapping[str, Any],
    *,
    idempotency_key: str,
    now: datetime,
) -> None:
    safe = _safe_audit_payload(payload)
    conn.execute(
        """
        INSERT OR IGNORE INTO durable_audit_events (
            id, tenant_id, workspace_id, run_id, step_id, event_type,
            payload_json, idempotency_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid4().hex,
            scope.tenant_id,
            scope.workspace_id,
            scope.run_id,
            safe.get("step_id"),
            event_type,
            _json(safe),
            idempotency_key,
            _iso(now),
        ),
    )


def _safe_audit_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "prompt",
        "raw_arguments",
        "secret",
        "token",
    }

    def inspect(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).casefold() in forbidden:
                    raise ValueError(
                        f"durable audit payload cannot contain {key!r}"
                    )
                inspect(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                inspect(nested)

    inspect(value)
    redacted = redact_data(dict(value))
    if not isinstance(redacted, dict):
        raise ValueError("audit payload must remain an object after redaction")
    return redacted


def _json(value: Mapping[str, Any]) -> str:
    redacted = redact_data(dict(value))
    if not isinstance(redacted, dict):
        raise ValueError("approval payload must remain an object")
    return json.dumps(
        redacted,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _object(value: object) -> dict[str, Any]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("stored approval payload is not an object")
    return parsed


def _required(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


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
    "ApprovalConflictError",
    "ApprovalNotFoundError",
    "SQLiteApprovalStore",
]
