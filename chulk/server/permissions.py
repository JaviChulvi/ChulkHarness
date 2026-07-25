"""Durable permission rendezvous for local control clients."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from chulk.events import AgentEvent, EventName, PermissionPayload
from chulk.redaction import redact_data
from chulk.server.journal import PublicEventJournal
from chulk.storage import initialize_sqlite_database, sqlite_connection
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
)


DEFAULT_PERMISSION_TTL_SECONDS = 300
DEFAULT_ARGUMENT_PREVIEW_CHARS = 4_000
ImmediatePermissionCallback = Callable[
    [PermissionRequest, PermissionDecisionRecord],
    PermissionDecision | bool,
]


class PermissionRequestNotFoundError(LookupError):
    """Raised when a request is outside the owned conversation."""


class PermissionDecisionConflictError(RuntimeError):
    """Raised when an idempotency key is reused for a different decision."""


@dataclass(frozen=True, slots=True)
class PendingPermission:
    id: str
    profile_id: str
    conversation_id: str
    turn_id: str | None
    tool_name: str
    permission_level: str
    policy_name: str
    reason: str
    argument_preview: Mapping[str, Any]
    argument_sha256: str
    status: str
    decision: str | None
    decision_reason: str | None
    created_at: str
    expires_at: str
    decided_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "tool_name": self.tool_name,
            "permission_level": self.permission_level,
            "policy_name": self.policy_name,
            "reason": self.reason,
            "argument_preview": dict(self.argument_preview),
            "argument_sha256": self.argument_sha256,
            "status": self.status,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "decided_at": self.decided_at,
        }


class PermissionBroker:
    """Bridge synchronous tool approval callbacks to durable API decisions."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        profile_id: str,
        conversation_id: str,
        turn_id: Callable[[], str | None],
        journal: PublicEventJournal | None = None,
        ttl_seconds: int = DEFAULT_PERMISSION_TTL_SECONDS,
        argument_preview_chars: int = DEFAULT_ARGUMENT_PREVIEW_CHARS,
        immediate_callback: ImmediatePermissionCallback | None = None,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be greater than zero")
        if argument_preview_chars < 1:
            raise ValueError("argument_preview_chars must be greater than zero")
        self.db_path = Path(db_path).expanduser().resolve()
        self.profile_id = profile_id
        self.conversation_id = conversation_id
        self._turn_id = turn_id
        self.journal = journal
        self.ttl_seconds = ttl_seconds
        self.argument_preview_chars = argument_preview_chars
        self.immediate_callback = immediate_callback
        self._condition = threading.Condition()
        initialize_sqlite_database(self.db_path)
        self.mark_restart_uncertain()

    def callback(
        self,
        request: PermissionRequest,
        record: PermissionDecisionRecord,
    ) -> PermissionDecision:
        """Persist a request, then resolve immediately or await an API decision."""
        pending = self.create(request)
        if self.immediate_callback is not None:
            answer = self.immediate_callback(request, record)
            decision = _normalize_answer(answer)
            self.decide(
                pending.id,
                decision,
                idempotency_key=f"immediate:{pending.id}",
                reason="resolved by the configured immediate callback",
            )
        resolved = self.wait(pending.id)
        return (
            PermissionDecision.ALLOW
            if resolved.status == "allowed"
            else PermissionDecision.DENY
        )

    def create(self, request: PermissionRequest) -> PendingPermission:
        now = _utc_now()
        expires_at = now + timedelta(seconds=self.ttl_seconds)
        request_id = uuid4().hex
        preview = _bounded_preview(
            request.arguments,
            max_chars=self.argument_preview_chars,
        )
        digest = _argument_digest(request.arguments)
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO permission_requests (
                    id, profile_id, conversation_id, turn_id, tool_name,
                    permission_level, policy_name, reason,
                    argument_preview_json, argument_sha256, status,
                    created_at, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    request_id,
                    self.profile_id,
                    self.conversation_id,
                    self._turn_id(),
                    request.tool_name,
                    request.permission_level.value,
                    request.policy_name,
                    request.reason,
                    json.dumps(preview, separators=(",", ":"), sort_keys=True),
                    digest,
                    _encode(now),
                    _encode(expires_at),
                    _encode(now),
                ),
            )
        pending = self.get(request_id)
        self._publish(pending, EventName.PERMISSION_REQUESTED)
        return pending

    def wait(self, request_id: str) -> PendingPermission:
        while True:
            pending = self.get(request_id)
            if pending.status != "pending":
                return pending
            remaining = (_decode(pending.expires_at) - _utc_now()).total_seconds()
            if remaining <= 0:
                return self._expire(request_id)
            with self._condition:
                self._condition.wait(timeout=min(remaining, 0.25))

    def decide(
        self,
        request_id: str,
        decision: PermissionDecision | str,
        *,
        idempotency_key: str,
        reason: str | None = None,
    ) -> PendingPermission:
        normalized = _normalize_answer(decision)
        status = "allowed" if normalized is PermissionDecision.ALLOW else "denied"
        now = _utc_now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM permission_requests
                WHERE id = ? AND profile_id = ? AND conversation_id = ?
                """,
                (request_id, self.profile_id, self.conversation_id),
            ).fetchone()
            if row is None:
                raise PermissionRequestNotFoundError(
                    "permission request does not belong to this conversation"
                )
            existing_key = row["decision_key"]
            if existing_key is not None:
                if str(existing_key) != idempotency_key or row["decision"] != normalized.value:
                    raise PermissionDecisionConflictError(
                        "permission request already has a different decision"
                    )
                return _row(row)
            duplicate = conn.execute(
                """
                SELECT id FROM permission_requests
                WHERE profile_id = ? AND decision_key = ?
                """,
                (self.profile_id, idempotency_key),
            ).fetchone()
            if duplicate is not None:
                raise PermissionDecisionConflictError(
                    "idempotency key was already used for another permission request"
                )
            if str(row["status"]) != "pending":
                raise PermissionDecisionConflictError(
                    f"permission request is already {row['status']}"
                )
            if _decode(str(row["expires_at"])) <= now:
                conn.execute(
                    """
                    UPDATE permission_requests
                    SET status = 'expired', updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (_encode(now), request_id),
                )
                raise PermissionDecisionConflictError("permission request has expired")
            conn.execute(
                """
                UPDATE permission_requests
                SET status = ?, decision = ?, decision_reason = ?,
                    decision_key = ?, decided_at = ?, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (
                    status,
                    normalized.value,
                    reason,
                    idempotency_key,
                    _encode(now),
                    _encode(now),
                    request_id,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM permission_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
        assert updated is not None
        result = _row(updated)
        with self._condition:
            self._condition.notify_all()
        self._publish(result, EventName.PERMISSION_RESOLVED)
        return result

    def get(self, request_id: str) -> PendingPermission:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM permission_requests
                WHERE id = ? AND profile_id = ? AND conversation_id = ?
                """,
                (request_id, self.profile_id, self.conversation_id),
            ).fetchone()
        if row is None:
            raise PermissionRequestNotFoundError(
                "permission request does not belong to this conversation"
            )
        return _row(row)

    def list(self, *, status: str | None = None, limit: int = 100) -> tuple[PendingPermission, ...]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        parameters: list[Any] = [self.profile_id, self.conversation_id]
        clause = ""
        if status is not None:
            clause = " AND status = ?"
            parameters.append(status)
        parameters.append(limit)
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM permission_requests
                WHERE profile_id = ? AND conversation_id = ?{clause}
                ORDER BY created_at, id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return tuple(_row(row) for row in rows)

    def mark_restart_uncertain(self) -> int:
        now = _encode(_utc_now())
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE permission_requests
                SET status = 'uncertain',
                    decision_reason = 'server restarted before a decision',
                    updated_at = ?
                WHERE profile_id = ? AND conversation_id = ? AND status = 'pending'
                """,
                (now, self.profile_id, self.conversation_id),
            )
        return cursor.rowcount

    def cancel_pending(self, *, reason: str = "conversation cancelled") -> int:
        """Deny outstanding requests and wake blocked tool callbacks."""
        now = _encode(_utc_now())
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id FROM permission_requests
                WHERE profile_id = ? AND conversation_id = ? AND status = 'pending'
                """,
                (self.profile_id, self.conversation_id),
            ).fetchall()
            conn.execute(
                """
                UPDATE permission_requests
                SET status = 'denied', decision = 'deny', decision_reason = ?,
                    decided_at = ?, updated_at = ?
                WHERE profile_id = ? AND conversation_id = ? AND status = 'pending'
                """,
                (
                    reason,
                    now,
                    now,
                    self.profile_id,
                    self.conversation_id,
                ),
            )
        with self._condition:
            self._condition.notify_all()
        for row in rows:
            self._publish(self.get(str(row["id"])), EventName.PERMISSION_RESOLVED)
        return len(rows)

    def _expire(self, request_id: str) -> PendingPermission:
        now = _encode(_utc_now())
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE permission_requests
                SET status = 'expired', decision_reason = 'permission request expired',
                    updated_at = ?
                WHERE id = ? AND profile_id = ? AND conversation_id = ?
                  AND status = 'pending'
                """,
                (now, request_id, self.profile_id, self.conversation_id),
            )
        result = self.get(request_id)
        with self._condition:
            self._condition.notify_all()
        self._publish(result, EventName.PERMISSION_RESOLVED)
        return result

    def _publish(self, pending: PendingPermission, name: EventName) -> None:
        if self.journal is None:
            return
        self.journal.append(
            AgentEvent(
                name=name.value,
                profile_id=self.profile_id,
                conversation_id=self.conversation_id,
                turn_id=pending.turn_id,
                payload=PermissionPayload(
                    tool_name=pending.tool_name,
                    decision=pending.decision,
                    reason=pending.decision_reason or pending.reason,
                    policy_name=pending.policy_name,
                    extensions={
                        "permission_request_id": pending.id,
                        "permission_level": pending.permission_level,
                        "argument_sha256": pending.argument_sha256,
                        "status": pending.status,
                    },
                ),
                extensions={"source": "permission_broker"},
            )
        )


def _bounded_preview(arguments: Mapping[str, Any], *, max_chars: int) -> dict[str, Any]:
    redacted = redact_data(dict(arguments))
    if not isinstance(redacted, dict):
        return {}
    encoded = json.dumps(redacted, separators=(",", ":"), sort_keys=True)
    if len(encoded) <= max_chars:
        return redacted
    return {
        "truncated": True,
        "preview": encoded[:max_chars],
    }


def _argument_digest(arguments: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        arguments,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _normalize_answer(value: PermissionDecision | str | bool) -> PermissionDecision:
    if isinstance(value, bool):
        return PermissionDecision.ALLOW if value else PermissionDecision.DENY
    answer = value if isinstance(value, PermissionDecision) else PermissionDecision(value)
    if answer is PermissionDecision.ASK:
        raise ValueError("a pending permission must be resolved with allow or deny")
    return answer


def _row(row: Any) -> PendingPermission:
    preview = json.loads(str(row["argument_preview_json"]))
    if not isinstance(preview, dict):
        raise ValueError("stored permission argument preview is invalid")
    return PendingPermission(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        conversation_id=str(row["conversation_id"]),
        turn_id=str(row["turn_id"]) if row["turn_id"] is not None else None,
        tool_name=str(row["tool_name"]),
        permission_level=str(row["permission_level"]),
        policy_name=str(row["policy_name"]),
        reason=str(row["reason"]),
        argument_preview=preview,
        argument_sha256=str(row["argument_sha256"]),
        status=str(row["status"]),
        decision=str(row["decision"]) if row["decision"] is not None else None,
        decision_reason=(
            str(row["decision_reason"]) if row["decision_reason"] is not None else None
        ),
        created_at=str(row["created_at"]),
        expires_at=str(row["expires_at"]),
        decided_at=str(row["decided_at"]) if row["decided_at"] is not None else None,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _encode(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decode(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


__all__ = [
    "DEFAULT_ARGUMENT_PREVIEW_CHARS",
    "DEFAULT_PERMISSION_TTL_SECONDS",
    "PendingPermission",
    "PermissionBroker",
    "PermissionDecisionConflictError",
    "PermissionRequestNotFoundError",
]
