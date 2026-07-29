"""Owner-controlled durable inbox, execution, and delivery ledger."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.gateway.models import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    DeliveryTarget,
    InboundEnvelope,
    InboundPart,
    MediaPart,
    MediaReference,
    OutboundEnvelope,
    ReactionPart,
    ReplyPart,
    TextPart,
    TrustLevel,
)
from chulk.gateway.stores import GatewayRunTarget
from chulk.hosting.scope import ExecutionScope
from chulk.profiles.store import CONTROL_MIGRATIONS
from chulk.storage import initialize_sqlite_database, sqlite_connection


UNCERTAIN_EXECUTION_MESSAGE = (
    "The previous request stopped at an uncertain execution checkpoint. "
    "It will not be run again automatically because doing so could repeat side effects."
)
MAX_ENVELOPE_BYTES = 256_000


class GatewayBackpressureError(RuntimeError):
    """Raised when the bounded durable inbox cannot accept more work."""


@dataclass(frozen=True, slots=True)
class GatewayAdapterStatus:
    adapter: str
    account_id: str
    state: str
    cursor: str | None
    instance_token: str | None
    lease_until: datetime | None
    legacy_adopted_at: datetime | None
    stop_requested: bool


@dataclass(frozen=True, slots=True)
class InboxRecord:
    id: str
    profile_id: str
    envelope: InboundEnvelope
    conversation_key: str
    state: str
    execution_token: str | None
    execution_lease_until: datetime | None
    cancellation_requested: bool
    last_error: str | None
    created_at: datetime
    run_target: GatewayRunTarget | None = None
    dead_lettered_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class IngestResult:
    record: InboxRecord
    created: bool


@dataclass(frozen=True, slots=True)
class ExecutionClaim:
    record: InboxRecord
    execution_token: str


@dataclass(frozen=True, slots=True)
class OutboxRecord:
    id: str
    inbox_id: str
    profile_id: str
    envelope: OutboundEnvelope
    state: str
    attempt_count: int
    checkpoint: str | None
    delivery_token: str | None
    delivery_lease_until: datetime | None
    next_attempt_at: datetime | None
    last_error: str | None
    reconciliation_required: bool = False
    dead_lettered_at: datetime | None = None


class SQLiteGatewayLedger:
    """Coordinate channel work in the owner-only control database."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        initialize_sqlite_database(self.db_path, migrations=CONTROL_MIGRATIONS)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_connection(self.db_path)

    def _serialize_execution_claim(self, conn: sqlite3.Connection) -> None:
        """Let backends serialize concurrency admission inside the transaction."""

    def _serialize_pending_admission(self, conn: sqlite3.Connection) -> None:
        """Let backends serialize bounded inbox admission inside the transaction."""

    def _serialize_inbox_mutation(
        self,
        conn: sqlite3.Connection,
        inbox_id: str,
    ) -> None:
        """Let backends serialize dependent writes for one inbox record."""

    def _serialize_outbox_mutation(
        self,
        conn: sqlite3.Connection,
        outbox_id: str,
    ) -> None:
        """Let backends serialize delivery evidence for one outbox record."""

    def _execution_recovery_lock_clause(self) -> str:
        """Return backend-specific locking for expired execution workers."""
        return ""

    def start_adapter(
        self,
        adapter: str,
        account_id: str,
        *,
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> GatewayAdapterStatus:
        """Acquire one adapter/account lease, rejecting a live second owner."""
        adapter, account_id = _identity(adapter, account_id)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = _observed(now)
        token = uuid4().hex
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO gateway_adapters (
                    adapter, account_id, state, instance_token, lease_until,
                    started_at, stopped_at, updated_at
                ) VALUES (?, ?, 'running', ?, ?, ?, NULL, ?)
                ON CONFLICT(adapter, account_id) DO UPDATE SET
                    state = 'running',
                    instance_token = excluded.instance_token,
                    lease_until = excluded.lease_until,
                    stop_requested = 0,
                    started_at = excluded.started_at,
                    stopped_at = NULL,
                    updated_at = excluded.updated_at
                WHERE gateway_adapters.state != 'running'
                   OR gateway_adapters.lease_until IS NULL
                   OR gateway_adapters.lease_until <= excluded.started_at
                """,
                (
                    adapter,
                    account_id,
                    token,
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    _encode(observed),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"{adapter}/{account_id} is already running")
            row = _adapter_row(conn, adapter, account_id)
        assert row is not None
        return _row_to_adapter_status(row)

    def renew_adapter(
        self,
        adapter: str,
        account_id: str,
        instance_token: str,
        *,
        lease_seconds: int = 60,
        now: datetime | None = None,
    ) -> bool:
        if not instance_token:
            raise ValueError("instance_token is required")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = _observed(now)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE gateway_adapters
                SET lease_until = ?, updated_at = ?
                WHERE adapter = ? AND account_id = ? AND state = 'running'
                  AND instance_token = ? AND lease_until >= ?
                  AND stop_requested = 0
                """,
                (
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    adapter,
                    account_id,
                    instance_token,
                    _encode(observed),
                ),
            )
        return cursor.rowcount == 1

    def stop_adapter(
        self,
        adapter: str,
        account_id: str,
        instance_token: str | None = None,
    ) -> bool:
        """Stop an adapter, optionally requiring ownership of its active lease."""
        observed = _utc_now()
        where_token = " AND instance_token = ?" if instance_token is not None else ""
        arguments: tuple[object, ...] = (
            _encode(observed),
            _encode(observed),
            adapter,
            account_id,
        )
        if instance_token is not None:
            arguments += (instance_token,)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE gateway_adapters
                SET state = 'stopped', instance_token = NULL, lease_until = NULL,
                    stop_requested = 0, stopped_at = ?, updated_at = ?
                WHERE adapter = ? AND account_id = ?{where_token}
                """,
                arguments,
            )
        return cursor.rowcount == 1

    def adapter_status(
        self,
        adapter: str,
        account_id: str,
    ) -> GatewayAdapterStatus | None:
        with self._connect() as conn:
            row = _adapter_row(conn, adapter, account_id)
        return _row_to_adapter_status(row) if row is not None else None

    def list_adapter_statuses(self) -> tuple[GatewayAdapterStatus, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM gateway_adapters
                ORDER BY adapter, account_id
                """
            ).fetchall()
        return tuple(_row_to_adapter_status(row) for row in rows)

    def request_adapter_stop(self, adapter: str, account_id: str) -> bool:
        """Ask the active lease owner to stop without impersonating it."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE gateway_adapters
                SET stop_requested = 1, updated_at = ?
                WHERE adapter = ? AND account_id = ? AND state = 'running'
                """,
                (_encode(_utc_now()), adapter, account_id),
            )
        return cursor.rowcount == 1

    def save_cursor(
        self,
        adapter: str,
        account_id: str,
        cursor: str,
        *,
        instance_token: str,
    ) -> bool:
        if not cursor or not instance_token:
            raise ValueError("cursor and instance_token are required")
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE gateway_adapters
                SET cursor = ?, updated_at = ?
                WHERE adapter = ? AND account_id = ? AND state = 'running'
                  AND instance_token = ?
                """,
                (cursor, _encode(_utc_now()), adapter, account_id, instance_token),
            )
        return updated.rowcount == 1

    def ingest(
        self,
        envelope: InboundEnvelope,
        *,
        profile_id: str,
        conversation_key: str | None = None,
        max_pending: int | None = None,
        run_target: GatewayRunTarget | None = None,
    ) -> IngestResult:
        """Durably accept an envelope before its transport acknowledgement."""
        profile_id = _required(profile_id, "profile_id")
        if max_pending is not None and max_pending <= 0:
            raise ValueError("max_pending must be greater than zero")
        key = _required(
            conversation_key or conversation_key_for(envelope),
            "conversation_key",
        )
        encoded = _bounded_json(_inbound_to_dict(envelope))
        observed = _utc_now()
        record_id = uuid4().hex
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if max_pending is not None:
                self._serialize_pending_admission(conn)
            existing = conn.execute(
                """
                SELECT * FROM gateway_inbox
                WHERE adapter = ? AND account_id = ? AND idempotency_key = ?
                """,
                (
                    envelope.identity.adapter,
                    envelope.identity.account_id,
                    envelope.idempotency_key,
                ),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["event_id"]) != envelope.event_id
                    or str(existing["principal_id"])
                    != envelope.identity.principal_id
                    or str(existing["destination_id"]) != envelope.destination_id
                    or (
                        str(existing["thread_id"])
                        if existing["thread_id"] is not None
                        else None
                    )
                    != envelope.thread_id
                ):
                    raise ValueError(
                        "idempotency key collision does not match the stored event"
                    )
                _validate_stored_run_target(existing, run_target)
                return IngestResult(_row_to_inbox(existing), False)
            if max_pending is not None:
                pending = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM gateway_inbox AS inbox
                        WHERE inbox.state IN ('queued', 'processing')
                           OR EXISTS (
                               SELECT 1 FROM gateway_outbox AS outbox
                               WHERE outbox.inbox_id = inbox.id
                                 AND outbox.state IN ('pending', 'delivering')
                           )
                        """
                    ).fetchone()[0]
                )
                if pending >= max_pending:
                    raise GatewayBackpressureError(
                        "gateway inbox has reached its configured pending limit"
                    )
            conn.execute(
                """
                INSERT INTO gateway_inbox (
                    id, profile_id, adapter, account_id, event_id,
                    idempotency_key, conversation_key, principal_id,
                    destination_id, thread_id, envelope_json, state,
                    created_at, updated_at, execution_scope_json,
                    agent_definition_id, agent_definition_version,
                    agent_definition_digest, run_id
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    record_id,
                    profile_id,
                    envelope.identity.adapter,
                    envelope.identity.account_id,
                    envelope.event_id,
                    envelope.idempotency_key,
                    key,
                    envelope.identity.principal_id,
                    envelope.destination_id,
                    envelope.thread_id,
                    encoded,
                    _encode(observed),
                    _encode(observed),
                    (
                        json.dumps(
                            run_target.scope.to_dict(),
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        if run_target is not None
                        else None
                    ),
                    run_target.definition_id if run_target is not None else None,
                    (
                        run_target.definition_version
                        if run_target is not None
                        else None
                    ),
                    (
                        run_target.definition_digest
                        if run_target is not None
                        else None
                    ),
                    run_target.scope.run_id if run_target is not None else None,
                ),
            )
            row = _inbox_row(conn, record_id)
        assert row is not None
        return IngestResult(_row_to_inbox(row), True)

    def ignore(
        self,
        envelope: InboundEnvelope,
        *,
        profile_id: str,
        reason: str,
        conversation_key: str | None = None,
    ) -> InboxRecord:
        result = self.ingest(
            envelope,
            profile_id=profile_id,
            conversation_key=conversation_key,
        )
        if result.created:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE gateway_inbox
                    SET state = 'ignored', last_error = ?, updated_at = ?
                    WHERE id = ? AND state = 'queued'
                    """,
                    (reason[:500], _encode(_utc_now()), result.record.id),
                )
        record = self.get_inbox(result.record.id)
        assert record is not None
        return record

    def claim_execution(
        self,
        *,
        global_limit: int,
        profile_limit: int,
        profile_id: str | None = None,
        adapter_keys: tuple[tuple[str, str], ...] | None = None,
        queued_before: datetime | None = None,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> ExecutionClaim | None:
        """Claim the oldest eligible event while preserving conversation FIFO."""
        if global_limit <= 0 or profile_limit <= 0:
            raise ValueError("concurrency limits must be greater than zero")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = _observed(now)
        if queued_before is not None and queued_before.tzinfo is None:
            raise ValueError("queued_before must include a timezone")
        queue_cutoff = (
            queued_before.astimezone(timezone.utc)
            if queued_before is not None
            else None
        )
        token = uuid4().hex
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_execution_claim(conn)
            global_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM gateway_inbox WHERE state = 'processing'"
                ).fetchone()[0]
            )
            if global_count >= global_limit:
                return None
            selected_profile = profile_id
            if selected_profile is not None:
                profile_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM gateway_inbox
                        WHERE state = 'processing' AND profile_id = ?
                        """,
                        (selected_profile,),
                    ).fetchone()[0]
                )
                if profile_count >= profile_limit:
                    return None
            profile_clause = "AND candidate.profile_id = ?" if selected_profile else ""
            parameters: tuple[object, ...] = (
                (selected_profile,) if selected_profile is not None else ()
            )
            cutoff_clause = ""
            if queue_cutoff is not None:
                cutoff_clause = "AND candidate.created_at <= ?"
                parameters += (_encode(queue_cutoff),)
            adapter_clause = ""
            if adapter_keys is not None:
                if not adapter_keys:
                    return None
                adapter_clause = "AND (" + " OR ".join(
                    "(candidate.adapter = ? AND candidate.account_id = ?)"
                    for _item in adapter_keys
                ) + ")"
                parameters += tuple(
                    value for item in adapter_keys for value in item
                )
            row = conn.execute(
                f"""
                SELECT candidate.*
                FROM gateway_inbox AS candidate
                WHERE candidate.state = 'queued'
                  AND candidate.cancellation_requested = 0
                  {profile_clause}
                  {cutoff_clause}
                  {adapter_clause}
                  AND NOT EXISTS (
                    SELECT 1 FROM gateway_inbox AS active
                    WHERE active.conversation_key = candidate.conversation_key
                      AND active.state = 'processing'
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM gateway_inbox AS earlier
                    WHERE earlier.conversation_key = candidate.conversation_key
                      AND earlier.state = 'queued'
                      AND (
                        earlier.created_at < candidate.created_at
                        OR (
                            earlier.created_at = candidate.created_at
                            AND earlier.id < candidate.id
                        )
                      )
                  )
                  AND (
                    SELECT COUNT(*) FROM gateway_inbox AS profile_active
                    WHERE profile_active.profile_id = candidate.profile_id
                      AND profile_active.state = 'processing'
                  ) < ?
                ORDER BY candidate.created_at, candidate.id
                LIMIT 1
                """,
                (*parameters, profile_limit),
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """
                UPDATE gateway_inbox
                SET state = 'processing', execution_token = ?,
                    execution_lease_until = ?, updated_at = ?
                WHERE id = ? AND state = 'queued'
                """,
                (
                    token,
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    row["id"],
                ),
            )
            if cursor.rowcount != 1:
                return None
            claimed = _inbox_row(conn, str(row["id"]))
        assert claimed is not None
        return ExecutionClaim(_row_to_inbox(claimed), token)

    def renew_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> bool:
        if not execution_token:
            raise ValueError("execution_token is required")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = _observed(now)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE gateway_inbox
                SET execution_lease_until = ?, updated_at = ?
                WHERE id = ? AND state = 'processing'
                  AND execution_token = ? AND execution_lease_until >= ?
                """,
                (
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    inbox_id,
                    execution_token,
                    _encode(observed),
                ),
            )
        return cursor.rowcount == 1

    def complete_execution(
        self,
        inbox_id: str,
        execution_token: str,
        responses: tuple[OutboundEnvelope, ...],
    ) -> bool:
        """Commit execution and its complete outbox in one transaction."""
        if not execution_token:
            raise ValueError("execution_token is required")
        if not responses:
            raise ValueError("responses cannot be empty")
        observed = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_inbox_mutation(conn, inbox_id)
            row = _inbox_row(conn, inbox_id)
            if (
                row is None
                or row["state"] != "processing"
                or row["execution_token"] != execution_token
            ):
                return False
            profile_id = str(row["profile_id"])
            for sequence, response in enumerate(responses):
                if response.profile_id != profile_id:
                    raise ValueError("outbound profile_id must match the inbox owner")
                if response.sequence != sequence:
                    raise ValueError(
                        "outbound sequence values must be contiguous and start at zero"
                    )
                conn.execute(
                    """
                    INSERT INTO gateway_outbox (
                        id, inbox_id, profile_id, adapter, account_id,
                        sequence, envelope_json,
                        state, checkpoint, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        response.envelope_id,
                        inbox_id,
                        profile_id,
                        response.target.adapter,
                        response.target.account_id,
                        sequence,
                        _bounded_json(_outbound_to_dict(response)),
                        response.checkpoint,
                        _encode(observed),
                        _encode(observed),
                    ),
                )
            cursor = conn.execute(
                """
                UPDATE gateway_inbox
                SET state = 'executed', execution_token = NULL,
                    execution_lease_until = NULL, executed_at = ?,
                    updated_at = ?, last_error = NULL
                WHERE id = ? AND state = 'processing' AND execution_token = ?
                """,
                (_encode(observed), _encode(observed), inbox_id, execution_token),
            )
        return cursor.rowcount == 1

    def recover_expired_executions(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> tuple[InboxRecord, ...]:
        """Quarantine ambiguous work; execution is never replayed after lease loss."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        observed = _observed(now)
        recovered: list[InboxRecord] = []
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""
                SELECT * FROM gateway_inbox
                WHERE state = 'processing' AND execution_lease_until <= ?
                ORDER BY execution_lease_until, id
                LIMIT ?
                {self._execution_recovery_lock_clause()}
                """,
                (_encode(observed), limit),
            ).fetchall()
            for row in rows:
                current = _inbox_row(conn, str(row["id"]))
                if (
                    current is None
                    or current["state"] != "processing"
                    or current["execution_token"] is None
                    or current["execution_lease_until"] is None
                    or _decode(str(current["execution_lease_until"])) > observed
                ):
                    continue
                cursor = conn.execute(
                    """
                    UPDATE gateway_inbox
                    SET state = 'uncertain', execution_token = NULL,
                        execution_lease_until = NULL,
                        last_error = 'execution lease expired', updated_at = ?
                    WHERE id = ? AND state = 'processing'
                      AND execution_token = ? AND execution_lease_until <= ?
                    """,
                    (
                        _encode(observed),
                        current["id"],
                        current["execution_token"],
                        _encode(observed),
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                _insert_uncertain_outbox(conn, current, observed)
                persisted = _inbox_row(conn, str(current["id"]))
                if persisted is not None:
                    recovered.append(_row_to_inbox(persisted))
        return tuple(recovered)

    def quarantine_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        error: str,
    ) -> bool:
        """Terminally quarantine an execution whose side effects are unknown."""
        if not execution_token:
            raise ValueError("execution_token is required")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_inbox_mutation(conn, inbox_id)
            row = _inbox_row(conn, inbox_id)
            if (
                row is None
                or row["state"] != "processing"
                or row["execution_token"] != execution_token
            ):
                return False
            _insert_uncertain_outbox(conn, row, _utc_now())
            cursor = conn.execute(
                """
                UPDATE gateway_inbox
                SET state = 'uncertain', execution_token = NULL,
                    execution_lease_until = NULL, last_error = ?, updated_at = ?
                WHERE id = ? AND state = 'processing' AND execution_token = ?
                """,
                (
                    error[:500],
                    _encode(_utc_now()),
                    inbox_id,
                    execution_token,
                ),
            )
        return cursor.rowcount == 1

    def dead_letter_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        error: str,
    ) -> bool:
        """Stop a poison input before execution and expose a terminal outcome."""
        if not execution_token:
            raise ValueError("execution_token is required")
        observed = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE gateway_inbox
                SET state = 'uncertain', execution_token = NULL,
                    execution_lease_until = NULL, last_error = ?,
                    dead_lettered_at = ?, updated_at = ?
                WHERE id = ? AND state = 'processing'
                  AND execution_token = ?
                """,
                (
                    error[:500],
                    _encode(observed),
                    _encode(observed),
                    inbox_id,
                    execution_token,
                ),
            )
        return cursor.rowcount == 1

    def dead_letter_inbox(self, inbox_id: str, *, error: str) -> bool:
        """Persist a poison input as terminal before any execution claim."""
        observed = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE gateway_inbox
                SET state = 'uncertain', last_error = ?,
                    dead_lettered_at = ?, updated_at = ?
                WHERE id = ? AND state = 'queued'
                """,
                (
                    error[:500],
                    _encode(observed),
                    _encode(observed),
                    inbox_id,
                ),
            )
        return cursor.rowcount == 1

    def request_cancellation(self, inbox_id: str) -> bool:
        """Cancel queued work or signal the owner of an active execution."""
        observed = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE gateway_inbox
                SET state = CASE WHEN state = 'queued' THEN 'cancelled' ELSE state END,
                    cancellation_requested = 1, updated_at = ?
                WHERE id = ? AND state IN ('queued', 'processing')
                """,
                (_encode(observed), inbox_id),
            )
        return cursor.rowcount == 1

    def request_conversation_cancellation(
        self,
        *,
        profile_id: str,
        conversation_key: str,
        exclude_inbox_id: str,
    ) -> tuple[str, ...]:
        """Cancel earlier work while leaving the stop command itself queued."""
        observed = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT id FROM gateway_inbox
                WHERE profile_id = ? AND conversation_key = ? AND id != ?
                  AND state IN ('queued', 'processing')
                ORDER BY created_at, id
                """,
                (profile_id, conversation_key, exclude_inbox_id),
            ).fetchall()
            cancelled: list[str] = []
            for row in rows:
                inbox_id = str(row["id"])
                self._serialize_inbox_mutation(conn, inbox_id)
                cursor = conn.execute(
                    """
                    UPDATE gateway_inbox
                    SET state = CASE
                            WHEN state = 'queued' THEN 'cancelled'
                            ELSE state
                        END,
                        cancellation_requested = 1,
                        updated_at = ?
                    WHERE id = ? AND state IN ('queued', 'processing')
                    """,
                    (_encode(observed), inbox_id),
                )
                if cursor.rowcount == 1:
                    cancelled.append(inbox_id)
        return tuple(cancelled)

    def mark_execution_cancelled(self, inbox_id: str, execution_token: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE gateway_inbox
                SET state = 'cancelled', execution_token = NULL,
                    execution_lease_until = NULL, updated_at = ?
                WHERE id = ? AND state = 'processing'
                  AND execution_token = ? AND cancellation_requested = 1
                """,
                (_encode(_utc_now()), inbox_id, execution_token),
            )
        return cursor.rowcount == 1

    def get_inbox(self, inbox_id: str) -> InboxRecord | None:
        with self._connect() as conn:
            row = _inbox_row(conn, inbox_id)
        return _row_to_inbox(row) if row is not None else None

    def find_inbox(
        self,
        *,
        adapter: str,
        account_id: str,
        idempotency_key: str,
    ) -> InboxRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM gateway_inbox
                WHERE adapter = ? AND account_id = ? AND idempotency_key = ?
                """,
                (adapter, account_id, idempotency_key),
            ).fetchone()
        return _row_to_inbox(row) if row is not None else None

    def list_outbox(self, inbox_id: str) -> tuple[OutboxRecord, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM gateway_outbox
                WHERE inbox_id = ?
                ORDER BY sequence, id
                """,
                (inbox_id,),
            ).fetchall()
        return tuple(_row_to_outbox(row) for row in rows)

    def inbox_complete(self, inbox_id: str) -> bool:
        """Return whether an event is terminal and has no outstanding delivery."""
        with self._connect() as conn:
            inbox = _inbox_row(conn, inbox_id)
            if inbox is None:
                return False
            if inbox["state"] in {"ignored", "cancelled"}:
                return True
            if inbox["state"] not in {"executed", "uncertain"}:
                return False
            outstanding = conn.execute(
                """
                SELECT 1 FROM gateway_outbox
                WHERE inbox_id = ? AND state NOT IN ('delivered', 'failed')
                LIMIT 1
                """,
                (inbox_id,),
            ).fetchone()
        return outstanding is None

    def pending_count(self, *, profile_id: str | None = None) -> int:
        clause = " AND profile_id = ?" if profile_id is not None else ""
        parameters: tuple[object, ...] = (profile_id,) if profile_id is not None else ()
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) FROM gateway_inbox
                WHERE state IN ('queued', 'processing'){clause}
                """,
                parameters,
            ).fetchone()
        return int(row[0])

    def claim_delivery(
        self,
        *,
        adapter_keys: tuple[tuple[str, str], ...] | None = None,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> OutboxRecord | None:
        """Claim the oldest retry-ready outbound envelope."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = _observed(now)
        token = uuid4().hex
        adapter_clause = ""
        parameters: tuple[object, ...] = (
            _encode(observed),
            _encode(observed),
        )
        if adapter_keys is not None:
            if not adapter_keys:
                return None
            adapter_clause = "AND (" + " OR ".join(
                "(candidate.adapter = ? AND candidate.account_id = ?)"
                for _item in adapter_keys
            ) + ")"
            parameters += tuple(value for item in adapter_keys for value in item)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"""
                SELECT candidate.*
                FROM gateway_outbox AS candidate
                WHERE (
                    (
                        candidate.state = 'pending'
                        AND (
                            candidate.next_attempt_at IS NULL
                            OR candidate.next_attempt_at <= ?
                        )
                    )
                    OR (
                        candidate.state = 'delivering'
                        AND candidate.delivery_lease_until <= ?
                    )
                )
                {adapter_clause}
                AND candidate.reconciliation_required = 0
                AND NOT EXISTS (
                    SELECT 1 FROM gateway_outbox AS earlier
                    WHERE earlier.inbox_id = candidate.inbox_id
                      AND earlier.sequence < candidate.sequence
                      AND earlier.state NOT IN ('delivered', 'failed')
                )
                ORDER BY candidate.created_at, candidate.sequence, candidate.id
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """
                UPDATE gateway_outbox
                SET state = 'delivering', delivery_token = ?,
                    delivery_lease_until = ?, attempt_count = attempt_count + 1,
                    updated_at = ?
                WHERE id = ? AND (
                    state = 'pending'
                    OR (state = 'delivering' AND delivery_lease_until <= ?)
                )
                """,
                (
                    token,
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    row["id"],
                    _encode(observed),
                ),
            )
            if cursor.rowcount != 1:
                return None
            claimed = _outbox_row(conn, str(row["id"]))
        assert claimed is not None
        return _row_to_outbox(claimed)

    def record_delivery(
        self,
        outbox_id: str,
        delivery_token: str,
        receipt: DeliveryReceipt,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Record immutable delivery evidence and advance the durable checkpoint."""
        if receipt.envelope_id != outbox_id:
            raise ValueError("receipt envelope_id must match the outbox record")
        observed = _observed(now)
        if receipt.state is DeliveryState.DELIVERED:
            state = "delivered"
            next_attempt_at = None
            delivered_at = _encode(observed)
            reconciliation_required = 0
            dead_lettered_at = None
        elif receipt.state in {
            DeliveryState.ACCEPTED,
            DeliveryState.UNKNOWN,
        }:
            state = "pending"
            next_attempt_at = None
            delivered_at = None
            reconciliation_required = 1
            dead_lettered_at = None
        elif receipt.state is DeliveryState.RETRYABLE:
            state = "pending"
            delay = receipt.retry_after_seconds or 0
            next_attempt_at = _encode(observed + timedelta(seconds=delay))
            delivered_at = None
            reconciliation_required = 0
            dead_lettered_at = None
        elif receipt.state is DeliveryState.DEAD_LETTER:
            state = "failed"
            next_attempt_at = None
            delivered_at = None
            reconciliation_required = 0
            dead_lettered_at = _encode(observed)
        else:
            state = "failed"
            next_attempt_at = None
            delivered_at = None
            reconciliation_required = 0
            dead_lettered_at = None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_outbox_mutation(conn, outbox_id)
            row = _outbox_row(conn, outbox_id)
            if (
                row is None
                or row["state"] != "delivering"
                or row["delivery_token"] != delivery_token
            ):
                return False
            attempt = int(row["attempt_count"])
            if receipt.attempt != attempt:
                raise ValueError("receipt attempt must match the active delivery attempt")
            conn.execute(
                """
                INSERT INTO gateway_delivery_events (
                    id, outbox_id, attempt, state, receipt_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    outbox_id,
                    attempt,
                    receipt.state.value,
                    _bounded_json(_receipt_to_dict(receipt)),
                    receipt.recorded_at,
                ),
            )
            cursor = conn.execute(
                """
                UPDATE gateway_outbox
                SET state = ?, checkpoint = ?, delivery_token = NULL,
                    delivery_lease_until = NULL, next_attempt_at = ?,
                    last_error = ?, delivered_at = ?,
                    reconciliation_required = ?, dead_lettered_at = ?,
                    updated_at = ?
                WHERE id = ? AND state = 'delivering' AND delivery_token = ?
                """,
                (
                    state,
                    receipt.checkpoint,
                    next_attempt_at,
                    (
                        (receipt.error_message or receipt.error_code or "")[:500]
                        if state != "delivered"
                        else None
                    ),
                    delivered_at,
                    reconciliation_required,
                    dead_lettered_at,
                    _encode(observed),
                    outbox_id,
                    delivery_token,
                ),
            )
        return cursor.rowcount == 1

    def reconcile_delivery(
        self,
        outbox_id: str,
        receipt: DeliveryReceipt,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Resolve an ambiguous provider outcome without blindly resending."""
        if receipt.envelope_id != outbox_id:
            raise ValueError("receipt envelope_id must match the outbox record")
        if receipt.state in {
            DeliveryState.ACCEPTED,
            DeliveryState.UNKNOWN,
        }:
            raise ValueError("delivery reconciliation requires a terminal or retry outcome")
        observed = _observed(now)
        if receipt.state is DeliveryState.DELIVERED:
            state = "delivered"
            next_attempt_at = None
            delivered_at = _encode(observed)
            dead_lettered_at = None
        elif receipt.state is DeliveryState.RETRYABLE:
            state = "pending"
            delay = receipt.retry_after_seconds or 0
            next_attempt_at = _encode(observed + timedelta(seconds=delay))
            delivered_at = None
            dead_lettered_at = None
        elif receipt.state is DeliveryState.DEAD_LETTER:
            state = "failed"
            next_attempt_at = None
            delivered_at = None
            dead_lettered_at = _encode(observed)
        else:
            state = "failed"
            next_attempt_at = None
            delivered_at = None
            dead_lettered_at = None
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_outbox_mutation(conn, outbox_id)
            row = _outbox_row(conn, outbox_id)
            if (
                row is None
                or row["state"] != "pending"
                or not bool(row["reconciliation_required"])
            ):
                return False
            attempt = int(row["attempt_count"])
            if receipt.attempt != attempt:
                raise ValueError(
                    "receipt attempt must match the ambiguous delivery attempt"
                )
            conn.execute(
                """
                INSERT INTO gateway_delivery_events (
                    id, outbox_id, attempt, state, receipt_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    outbox_id,
                    attempt,
                    receipt.state.value,
                    _bounded_json(_receipt_to_dict(receipt)),
                    receipt.recorded_at,
                ),
            )
            cursor = conn.execute(
                """
                UPDATE gateway_outbox
                SET state = ?, checkpoint = ?, next_attempt_at = ?,
                    last_error = ?, delivered_at = ?,
                    reconciliation_required = 0, dead_lettered_at = ?,
                    updated_at = ?
                WHERE id = ? AND state = 'pending'
                  AND reconciliation_required = 1
                """,
                (
                    state,
                    receipt.checkpoint,
                    next_attempt_at,
                    (
                        (receipt.error_message or receipt.error_code or "")[:500]
                        if state != "delivered"
                        else None
                    ),
                    delivered_at,
                    dead_lettered_at,
                    _encode(observed),
                    outbox_id,
                ),
            )
        return cursor.rowcount == 1

    def list_reconciliation_required(
        self,
        *,
        limit: int = 100,
    ) -> tuple[OutboxRecord, ...]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM gateway_outbox
                WHERE state = 'pending' AND reconciliation_required = 1
                ORDER BY created_at, sequence, id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(_row_to_outbox(row) for row in rows)

    def get_outbox(self, outbox_id: str) -> OutboxRecord | None:
        with self._connect() as conn:
            row = _outbox_row(conn, outbox_id)
        return _row_to_outbox(row) if row is not None else None


def conversation_key_for(envelope: InboundEnvelope) -> str:
    """Return the stable FIFO key for one channel conversation."""
    return "\x1f".join(
        (
            envelope.identity.adapter,
            envelope.identity.account_id,
            envelope.destination_id,
            envelope.thread_id or "",
        )
    )


def _identity(adapter: str, account_id: str) -> tuple[str, str]:
    return _required(adapter, "adapter"), _required(account_id, "account_id")


def _required(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned or "\x00" in cleaned:
        raise ValueError(f"{name} is required")
    return cleaned


def _bounded_json(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_ENVELOPE_BYTES:
        raise ValueError("gateway envelope exceeds the durable payload limit")
    return encoded


def _media_to_dict(media: MediaReference) -> dict[str, Any]:
    return {
        "content_ref": media.content_ref,
        "content_type": media.content_type,
        "size_bytes": media.size_bytes,
        "file_name": media.file_name,
        "sha256": media.sha256,
        "external_content": media.external_content,
    }


def _inbound_to_dict(envelope: InboundEnvelope) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for part in envelope.parts:
        if isinstance(part, TextPart):
            parts.append(
                {
                    "kind": "text",
                    "text": part.text,
                    "external_content": part.external_content,
                }
            )
        elif isinstance(part, MediaPart):
            parts.append(
                {
                    "kind": "media",
                    "media": _media_to_dict(part.media),
                    "caption": part.caption,
                }
            )
        elif isinstance(part, ReplyPart):
            parts.append(
                {
                    "kind": "reply",
                    "event_id": part.event_id,
                    "excerpt": part.excerpt,
                    "external_content": part.external_content,
                }
            )
        else:
            assert isinstance(part, ReactionPart)
            parts.append(
                {
                    "kind": "reaction",
                    "reaction": part.reaction,
                    "target_event_id": part.target_event_id,
                    "external_content": part.external_content,
                }
            )
    return {
        "event_id": envelope.event_id,
        "idempotency_key": envelope.idempotency_key,
        "identity": {
            "adapter": envelope.identity.adapter,
            "account_id": envelope.identity.account_id,
            "principal_id": envelope.identity.principal_id,
        },
        "destination_id": envelope.destination_id,
        "parts": parts,
        "scope": envelope.scope.value,
        "authentication": envelope.authentication.value,
        "trust": envelope.trust.value,
        "thread_id": envelope.thread_id,
        "received_at": envelope.received_at,
        "external_content": envelope.external_content,
        "extensions": dict(envelope.extensions),
    }


def _inbound_from_dict(value: dict[str, Any]) -> InboundEnvelope:
    raw_parts = value["parts"]
    parts: list[InboundPart] = []
    for raw in raw_parts:
        kind = raw["kind"]
        if kind == "text":
            parts.append(
                TextPart(
                    str(raw["text"]),
                    external_content=bool(raw.get("external_content", True)),
                )
            )
        elif kind == "media":
            media = raw["media"]
            parts.append(
                MediaPart(
                    MediaReference(
                        content_ref=str(media["content_ref"]),
                        content_type=str(media["content_type"]),
                        size_bytes=int(media["size_bytes"]),
                        file_name=media.get("file_name"),
                        sha256=media.get("sha256"),
                        external_content=bool(media.get("external_content", True)),
                    ),
                    caption=raw.get("caption"),
                )
            )
        elif kind == "reply":
            parts.append(
                ReplyPart(
                    str(raw["event_id"]),
                    raw.get("excerpt"),
                    external_content=bool(raw.get("external_content", True)),
                )
            )
        elif kind == "reaction":
            parts.append(
                ReactionPart(
                    str(raw["reaction"]),
                    str(raw["target_event_id"]),
                    external_content=bool(raw.get("external_content", True)),
                )
            )
        else:
            raise ValueError(f"unknown inbound part kind: {kind}")
    identity = value["identity"]
    return InboundEnvelope(
        event_id=str(value["event_id"]),
        idempotency_key=str(value["idempotency_key"]),
        identity=ChannelIdentity(
            str(identity["adapter"]),
            str(identity["account_id"]),
            str(identity["principal_id"]),
        ),
        destination_id=str(value["destination_id"]),
        parts=tuple(parts),
        scope=ChannelScope(str(value["scope"])),
        authentication=AuthenticationState(str(value["authentication"])),
        trust=TrustLevel(str(value["trust"])),
        thread_id=value.get("thread_id"),
        received_at=str(value["received_at"]),
        external_content=bool(value.get("external_content", True)),
        extensions=value.get("extensions", {}),
    )


def _outbound_to_dict(envelope: OutboundEnvelope) -> dict[str, Any]:
    return {
        "profile_id": envelope.profile_id,
        "conversation_id": envelope.conversation_id,
        "target": {
            "adapter": envelope.target.adapter,
            "account_id": envelope.target.account_id,
            "destination_id": envelope.target.destination_id,
            "thread_id": envelope.target.thread_id,
        },
        "text": envelope.text,
        "attachments": [_media_to_dict(item) for item in envelope.attachments],
        "reply_to_event_id": envelope.reply_to_event_id,
        "envelope_id": envelope.envelope_id,
        "checkpoint": envelope.checkpoint,
        "sequence": envelope.sequence,
        "final": envelope.final,
        "extensions": dict(envelope.extensions),
    }


def _outbound_from_dict(value: dict[str, Any]) -> OutboundEnvelope:
    target = value["target"]
    return OutboundEnvelope(
        profile_id=str(value["profile_id"]),
        conversation_id=str(value["conversation_id"]),
        target=DeliveryTarget(
            str(target["adapter"]),
            str(target["account_id"]),
            str(target["destination_id"]),
            thread_id=target.get("thread_id"),
        ),
        text=value.get("text"),
        attachments=tuple(
            MediaReference(
                content_ref=str(item["content_ref"]),
                content_type=str(item["content_type"]),
                size_bytes=int(item["size_bytes"]),
                file_name=item.get("file_name"),
                sha256=item.get("sha256"),
                external_content=bool(item.get("external_content", True)),
            )
            for item in value.get("attachments", ())
        ),
        reply_to_event_id=value.get("reply_to_event_id"),
        envelope_id=str(value["envelope_id"]),
        checkpoint=value.get("checkpoint"),
        sequence=int(value.get("sequence", 0)),
        final=bool(value.get("final", True)),
        extensions=value.get("extensions", {}),
    )


def _receipt_to_dict(receipt: DeliveryReceipt) -> dict[str, Any]:
    return {
        "envelope_id": receipt.envelope_id,
        "state": receipt.state.value,
        "attempt": receipt.attempt,
        "adapter_message_id": receipt.adapter_message_id,
        "checkpoint": receipt.checkpoint,
        "retry_after_seconds": receipt.retry_after_seconds,
        "error_code": receipt.error_code,
        "error_message": receipt.error_message,
        "recorded_at": receipt.recorded_at,
        "extensions": dict(receipt.extensions),
    }


def _insert_uncertain_outbox(
    conn: sqlite3.Connection,
    inbox_row: sqlite3.Row,
    observed: datetime,
) -> None:
    envelope = _inbound_from_dict(json.loads(str(inbox_row["envelope_json"])))
    outbound = OutboundEnvelope(
        profile_id=str(inbox_row["profile_id"]),
        conversation_id=str(inbox_row["conversation_key"]),
        target=DeliveryTarget(
            envelope.identity.adapter,
            envelope.identity.account_id,
            envelope.destination_id,
            thread_id=envelope.thread_id,
        ),
        text=UNCERTAIN_EXECUTION_MESSAGE,
        reply_to_event_id=envelope.event_id,
    )
    conn.execute(
        """
        INSERT INTO gateway_outbox (
            id, inbox_id, profile_id, adapter, account_id,
            sequence, envelope_json,
            state, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 0, ?, 'pending', ?, ?)
        """,
        (
            outbound.envelope_id,
            inbox_row["id"],
            inbox_row["profile_id"],
            outbound.target.adapter,
            outbound.target.account_id,
            _bounded_json(_outbound_to_dict(outbound)),
            _encode(observed),
            _encode(observed),
        ),
    )


def _adapter_row(
    conn: sqlite3.Connection,
    adapter: str,
    account_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM gateway_adapters WHERE adapter = ? AND account_id = ?",
        (adapter, account_id),
    ).fetchone()


def _inbox_row(conn: sqlite3.Connection, inbox_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM gateway_inbox WHERE id = ?",
        (inbox_id,),
    ).fetchone()


def _outbox_row(conn: sqlite3.Connection, outbox_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM gateway_outbox WHERE id = ?",
        (outbox_id,),
    ).fetchone()


def _row_to_adapter_status(row: sqlite3.Row) -> GatewayAdapterStatus:
    return GatewayAdapterStatus(
        adapter=str(row["adapter"]),
        account_id=str(row["account_id"]),
        state=str(row["state"]),
        cursor=str(row["cursor"]) if row["cursor"] is not None else None,
        instance_token=(
            str(row["instance_token"]) if row["instance_token"] is not None else None
        ),
        lease_until=_optional_datetime(row["lease_until"]),
        legacy_adopted_at=_optional_datetime(row["legacy_adopted_at"]),
        stop_requested=bool(row["stop_requested"]),
    )


def _row_to_inbox(row: sqlite3.Row) -> InboxRecord:
    run_target = _stored_run_target(row)
    return InboxRecord(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        envelope=_inbound_from_dict(json.loads(str(row["envelope_json"]))),
        conversation_key=str(row["conversation_key"]),
        state=(
            "dead_letter"
            if row["dead_lettered_at"] is not None
            else str(row["state"])
        ),
        execution_token=(
            str(row["execution_token"]) if row["execution_token"] is not None else None
        ),
        execution_lease_until=_optional_datetime(row["execution_lease_until"]),
        cancellation_requested=bool(row["cancellation_requested"]),
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        created_at=_decode(str(row["created_at"])),
        run_target=run_target,
        dead_lettered_at=_optional_datetime(row["dead_lettered_at"]),
    )


def _row_to_outbox(row: sqlite3.Row) -> OutboxRecord:
    reconciliation_required = bool(row["reconciliation_required"])
    dead_lettered_at = _optional_datetime(row["dead_lettered_at"])
    state = str(row["state"])
    if dead_lettered_at is not None:
        state = DeliveryState.DEAD_LETTER.value
    elif reconciliation_required:
        state = DeliveryState.UNKNOWN.value
    return OutboxRecord(
        id=str(row["id"]),
        inbox_id=str(row["inbox_id"]),
        profile_id=str(row["profile_id"]),
        envelope=_outbound_from_dict(json.loads(str(row["envelope_json"]))),
        state=state,
        attempt_count=int(row["attempt_count"]),
        checkpoint=str(row["checkpoint"]) if row["checkpoint"] is not None else None,
        delivery_token=(
            str(row["delivery_token"]) if row["delivery_token"] is not None else None
        ),
        delivery_lease_until=_optional_datetime(row["delivery_lease_until"]),
        next_attempt_at=_optional_datetime(row["next_attempt_at"]),
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        reconciliation_required=reconciliation_required,
        dead_lettered_at=dead_lettered_at,
    )


def _stored_run_target(row: sqlite3.Row) -> GatewayRunTarget | None:
    encoded_scope = row["execution_scope_json"]
    if encoded_scope is None:
        return None
    decoded = json.loads(str(encoded_scope))
    if not isinstance(decoded, dict):
        raise ValueError("stored gateway execution scope is invalid")
    return GatewayRunTarget(
        scope=ExecutionScope.from_dict(decoded),
        definition_id=str(row["agent_definition_id"]),
        definition_version=str(row["agent_definition_version"]),
        definition_digest=str(row["agent_definition_digest"]),
    )


def _validate_stored_run_target(
    row: sqlite3.Row,
    requested: GatewayRunTarget | None,
) -> None:
    stored = _stored_run_target(row)
    if stored != requested:
        raise ValueError(
            "gateway idempotency key is already bound to a different hosted run"
        )


def _observed(value: datetime | None) -> datetime:
    observed = value or _utc_now()
    if observed.tzinfo is None:
        raise ValueError("now must include a timezone")
    return observed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _encode(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decode(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _optional_datetime(value: object) -> datetime | None:
    return _decode(str(value)) if value is not None else None


__all__ = [
    "ExecutionClaim",
    "GatewayBackpressureError",
    "GatewayAdapterStatus",
    "InboxRecord",
    "IngestResult",
    "OutboxRecord",
    "SQLiteGatewayLedger",
    "UNCERTAIN_EXECUTION_MESSAGE",
    "conversation_key_for",
]
