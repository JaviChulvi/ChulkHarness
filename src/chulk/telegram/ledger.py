"""Durable at-most-once execution and retryable delivery for adapter updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from chulk.storage import initialize_sqlite_database, sqlite_connection


UNCERTAIN_EXECUTION_RESPONSE = (
    "The previous request stopped at an uncertain execution checkpoint. "
    "It will not be run again automatically because doing so could repeat side effects."
)


@dataclass(frozen=True, slots=True)
class AdapterUpdateRecord:
    adapter: str
    update_id: int
    destination_id: str
    status: str
    response_parts: tuple[str, ...]
    next_response_part: int
    execution_token: str | None = None
    execution_lease_until: datetime | None = None
    delivery_token: str | None = None
    delivery_lease_until: datetime | None = None
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class AdapterUpdateClaim:
    record: AdapterUpdateRecord
    execution_token: str | None

    @property
    def should_execute(self) -> bool:
        return self.execution_token is not None


class SQLiteAdapterUpdateLedger:
    """Persist adapter execution before advancing polling cursors."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        initialize_sqlite_database(self.db_path)

    def ignore(self, *, adapter: str, update_id: int, destination_id: str) -> AdapterUpdateRecord:
        """Record a deliberately ignored update without creating an outbox entry."""
        clean_adapter, clean_destination = _validate_identity(adapter, update_id, destination_id)
        now = _utc_now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO adapter_updates (
                    adapter, update_id, destination_id, status, created_at,
                    updated_at, delivered_at
                ) VALUES (?, ?, ?, 'ignored', ?, ?, ?)
                ON CONFLICT(adapter, update_id) DO NOTHING
                """,
                (
                    clean_adapter,
                    update_id,
                    clean_destination,
                    _encode(now),
                    _encode(now),
                    _encode(now),
                ),
            )
            row = _select(conn, clean_adapter, update_id)
        assert row is not None
        return _row_to_record(row)

    def begin_execution(
        self,
        *,
        adapter: str,
        update_id: int,
        destination_id: str,
        now: datetime | None = None,
        lease_seconds: int = 300,
    ) -> AdapterUpdateClaim:
        """Claim a new update, quarantining expired in-flight execution."""
        clean_adapter, clean_destination = _validate_identity(adapter, update_id, destination_id)
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = (now or _utc_now()).astimezone(timezone.utc)
        execution_token = uuid4().hex
        lease_until = observed + timedelta(seconds=lease_seconds)
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = _select(conn, clean_adapter, update_id)
            if row is None:
                conn.execute(
                    """
                    INSERT INTO adapter_updates (
                        adapter, update_id, destination_id, status,
                        execution_token, execution_lease_until,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 'processing', ?, ?, ?, ?)
                    """,
                    (
                        clean_adapter,
                        update_id,
                        clean_destination,
                        execution_token,
                        _encode(lease_until),
                        _encode(observed),
                        _encode(observed),
                    ),
                )
                row = _select(conn, clean_adapter, update_id)
                assert row is not None
                return AdapterUpdateClaim(_row_to_record(row), execution_token)

            record = _row_to_record(row)
            if (
                record.status == "processing"
                and record.execution_lease_until is not None
                and record.execution_lease_until <= observed
            ):
                response_parts = json.dumps([UNCERTAIN_EXECUTION_RESPONSE])
                conn.execute(
                    """
                    UPDATE adapter_updates
                    SET status = 'executed', response_parts = ?,
                        next_response_part = 0, execution_token = NULL,
                        execution_lease_until = NULL,
                        last_error = 'execution lease expired',
                        executed_at = ?, updated_at = ?
                    WHERE adapter = ? AND update_id = ?
                      AND status = 'processing'
                    """,
                    (
                        response_parts,
                        _encode(observed),
                        _encode(observed),
                        clean_adapter,
                        update_id,
                    ),
                )
                row = _select(conn, clean_adapter, update_id)
                assert row is not None
                record = _row_to_record(row)
            return AdapterUpdateClaim(record, None)

    def renew_execution(
        self,
        *,
        adapter: str,
        update_id: int,
        execution_token: str,
        now: datetime | None = None,
        lease_seconds: int = 300,
    ) -> bool:
        """Renew an active execution claim without allowing resurrection."""
        if not execution_token:
            raise ValueError("execution_token is required")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = (now or _utc_now()).astimezone(timezone.utc)
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE adapter_updates
                SET execution_lease_until = ?, updated_at = ?
                WHERE adapter = ? AND update_id = ? AND status = 'processing'
                  AND execution_token = ? AND execution_lease_until >= ?
                """,
                (
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    adapter,
                    update_id,
                    execution_token,
                    _encode(observed),
                ),
            )
        return cursor.rowcount == 1

    def record_response(
        self,
        *,
        adapter: str,
        update_id: int,
        execution_token: str,
        response_parts: tuple[str, ...],
    ) -> bool:
        """Finish execution exactly once and create its durable response outbox."""
        if not execution_token:
            raise ValueError("execution_token is required")
        if not response_parts or any(not part for part in response_parts):
            raise ValueError("response_parts must contain non-empty messages")
        now = _utc_now()
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE adapter_updates
                SET status = 'executed', response_parts = ?,
                    next_response_part = 0, execution_token = NULL,
                    execution_lease_until = NULL, executed_at = ?,
                    updated_at = ?, last_error = NULL
                WHERE adapter = ? AND update_id = ? AND status = 'processing'
                  AND execution_token = ?
                """,
                (
                    json.dumps(response_parts),
                    _encode(now),
                    _encode(now),
                    adapter,
                    update_id,
                    execution_token,
                ),
            )
        return cursor.rowcount == 1

    def claim_delivery(
        self,
        *,
        adapter: str,
        update_id: int,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> AdapterUpdateRecord | None:
        """Claim pending response delivery, including an expired delivery lease."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        observed = (now or _utc_now()).astimezone(timezone.utc)
        token = uuid4().hex
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE adapter_updates
                SET status = 'delivering', delivery_token = ?,
                    delivery_lease_until = ?, updated_at = ?
                WHERE adapter = ? AND update_id = ?
                  AND (
                    status = 'executed'
                    OR (
                        status = 'delivering'
                        AND delivery_lease_until <= ?
                    )
                  )
                """,
                (
                    token,
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    adapter,
                    update_id,
                    _encode(observed),
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = _select(conn, adapter, update_id)
        assert row is not None
        return _row_to_record(row)

    def mark_response_part_delivered(
        self,
        *,
        adapter: str,
        update_id: int,
        delivery_token: str,
        expected_part: int,
    ) -> bool:
        """Checkpoint one delivered response part under the active claim."""
        now = _utc_now()
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT response_parts, next_response_part
                FROM adapter_updates
                WHERE adapter = ? AND update_id = ? AND status = 'delivering'
                  AND delivery_token = ?
                """,
                (adapter, update_id, delivery_token),
            ).fetchone()
            if row is None or int(row["next_response_part"]) != expected_part:
                return False
            parts = _decode_parts(row["response_parts"])
            next_part = expected_part + 1
            delivered = next_part >= len(parts)
            cursor = conn.execute(
                """
                UPDATE adapter_updates
                SET status = ?, next_response_part = ?,
                    delivery_token = ?, delivery_lease_until = ?,
                    delivered_at = ?, updated_at = ?, last_error = NULL
                WHERE adapter = ? AND update_id = ? AND status = 'delivering'
                  AND delivery_token = ? AND next_response_part = ?
                """,
                (
                    "delivered" if delivered else "delivering",
                    next_part,
                    None if delivered else delivery_token,
                    None if delivered else _encode(now + timedelta(seconds=120)),
                    _encode(now) if delivered else None,
                    _encode(now),
                    adapter,
                    update_id,
                    delivery_token,
                    expected_part,
                ),
            )
        return cursor.rowcount == 1

    def release_delivery(
        self,
        *,
        adapter: str,
        update_id: int,
        delivery_token: str,
        error: str,
    ) -> bool:
        """Return a failed delivery claim to the retryable outbox."""
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE adapter_updates
                SET status = 'executed', delivery_token = NULL,
                    delivery_lease_until = NULL, last_error = ?, updated_at = ?
                WHERE adapter = ? AND update_id = ? AND status = 'delivering'
                  AND delivery_token = ?
                """,
                (
                    error[:500],
                    _encode(_utc_now()),
                    adapter,
                    update_id,
                    delivery_token,
                ),
            )
        return cursor.rowcount == 1

    def get(self, *, adapter: str, update_id: int) -> AdapterUpdateRecord | None:
        with sqlite_connection(self.db_path) as conn:
            row = _select(conn, adapter, update_id)
        return _row_to_record(row) if row is not None else None

    def purge_terminal(self, *, before: datetime, limit: int = 1_000) -> int:
        """Delete a bounded batch of old delivered or ignored ledger entries."""
        if before.tzinfo is None:
            raise ValueError("before must include a timezone")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM adapter_updates
                WHERE rowid IN (
                    SELECT rowid FROM adapter_updates
                    WHERE status IN ('delivered', 'ignored') AND updated_at < ?
                    ORDER BY updated_at
                    LIMIT ?
                )
                """,
                (_encode(before), limit),
            )
        return cursor.rowcount


def _validate_identity(adapter: str, update_id: int, destination_id: str) -> tuple[str, str]:
    clean_adapter = adapter.strip()
    clean_destination = destination_id.strip()
    if not clean_adapter or not clean_destination:
        raise ValueError("adapter and destination_id are required")
    if update_id < 0:
        raise ValueError("update_id cannot be negative")
    return clean_adapter, clean_destination


def _select(
    conn: sqlite3.Connection,
    adapter: str,
    update_id: int,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM adapter_updates WHERE adapter = ? AND update_id = ?",
        (adapter, update_id),
    ).fetchone()


def _row_to_record(row: sqlite3.Row) -> AdapterUpdateRecord:
    return AdapterUpdateRecord(
        adapter=str(row["adapter"]),
        update_id=int(row["update_id"]),
        destination_id=str(row["destination_id"]),
        status=str(row["status"]),
        response_parts=_decode_parts(row["response_parts"]),
        next_response_part=int(row["next_response_part"]),
        execution_token=str(row["execution_token"]) if row["execution_token"] else None,
        execution_lease_until=(
            _decode(str(row["execution_lease_until"]))
            if row["execution_lease_until"]
            else None
        ),
        delivery_token=str(row["delivery_token"]) if row["delivery_token"] else None,
        delivery_lease_until=(
            _decode(str(row["delivery_lease_until"]))
            if row["delivery_lease_until"]
            else None
        ),
        last_error=str(row["last_error"]) if row["last_error"] else None,
    )


def _decode_parts(value: object) -> tuple[str, ...]:
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(str(part) for part in decoded if isinstance(part, str) and part)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _encode(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decode(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


__all__ = [
    "AdapterUpdateClaim",
    "AdapterUpdateRecord",
    "SQLiteAdapterUpdateLedger",
    "UNCERTAIN_EXECUTION_RESPONSE",
]
