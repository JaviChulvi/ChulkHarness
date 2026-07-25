"""Durable bounded delivery of stable public SDK events."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.events import AgentEvent
from chulk.storage import initialize_sqlite_database, sqlite_connection


DEFAULT_EVENT_RETENTION = 2_000
DEFAULT_EVENT_PAYLOAD_BYTES = 256_000


class PublicEventCursorExpiredError(LookupError):
    """Raised when a resume cursor is absent from the retained journal."""


@dataclass(frozen=True, slots=True)
class PublicEventRecord:
    event_id: str
    profile_id: str
    conversation_id: str
    sequence: int
    event: AgentEvent
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id,
            "sequence": self.sequence,
            "event": self.event.to_dict(),
            "created_at": self.created_at,
        }


class PublicEventJournal:
    """Append and resume profile-owned events without exposing raw trace rows."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        profile_id: str,
        retention: int = DEFAULT_EVENT_RETENTION,
        max_event_bytes: int = DEFAULT_EVENT_PAYLOAD_BYTES,
    ) -> None:
        if retention < 1:
            raise ValueError("retention must be greater than zero")
        if max_event_bytes < 1:
            raise ValueError("max_event_bytes must be greater than zero")
        self.db_path = Path(db_path).expanduser().resolve()
        self.profile_id = profile_id
        self.retention = retention
        self.max_event_bytes = max_event_bytes
        initialize_sqlite_database(self.db_path)

    def append(self, event: AgentEvent) -> PublicEventRecord:
        if event.profile_id not in {None, self.profile_id}:
            raise ValueError("event profile_id does not match journal ownership")
        payload = event.to_dict()
        payload["profile_id"] = self.profile_id
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        if len(encoded.encode("utf-8")) > self.max_event_bytes:
            raise ValueError("public event exceeds the configured payload limit")
        event_id = uuid4().hex
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                FROM public_events
                WHERE conversation_id = ?
                """,
                (event.conversation_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            conn.execute(
                """
                INSERT INTO public_events (
                    event_id, profile_id, conversation_id, turn_id, sequence,
                    event_name, schema_version, event_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    self.profile_id,
                    event.conversation_id,
                    event.turn_id,
                    sequence,
                    event.name,
                    event.schema_version,
                    encoded,
                    event.timestamp,
                ),
            )
            conn.execute(
                """
                DELETE FROM public_events
                WHERE conversation_id = ? AND sequence <= ?
                """,
                (event.conversation_id, sequence - self.retention),
            )
        return PublicEventRecord(
            event_id=event_id,
            profile_id=self.profile_id,
            conversation_id=event.conversation_id,
            sequence=sequence,
            event=AgentEvent.from_dict(payload),
            created_at=event.timestamp,
        )

    def list(
        self,
        conversation_id: str,
        *,
        after_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> tuple[PublicEventRecord, ...]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        if after_id is not None and after_sequence is not None:
            raise ValueError("pass after_id or after_sequence, not both")
        sequence = after_sequence or 0
        with sqlite_connection(self.db_path) as conn:
            if after_id is not None:
                row = conn.execute(
                    """
                    SELECT sequence FROM public_events
                    WHERE event_id = ? AND conversation_id = ? AND profile_id = ?
                    """,
                    (after_id, conversation_id, self.profile_id),
                ).fetchone()
                if row is None:
                    raise PublicEventCursorExpiredError(
                        "event cursor is not available in the retained journal"
                    )
                sequence = int(row["sequence"])
            rows = conn.execute(
                """
                SELECT * FROM public_events
                WHERE profile_id = ? AND conversation_id = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (self.profile_id, conversation_id, sequence, limit),
            ).fetchall()
        return tuple(_record(row) for row in rows)

    def latest_sequence(self, conversation_id: str) -> int:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS sequence
                FROM public_events
                WHERE profile_id = ? AND conversation_id = ?
                """,
                (self.profile_id, conversation_id),
            ).fetchone()
        return int(row["sequence"])


def _record(row: sqlite3.Row) -> PublicEventRecord:
    value = json.loads(str(row["event_json"]))
    if not isinstance(value, dict):
        raise ValueError("stored public event is invalid")
    return PublicEventRecord(
        event_id=str(row["event_id"]),
        profile_id=str(row["profile_id"]),
        conversation_id=str(row["conversation_id"]),
        sequence=int(row["sequence"]),
        event=AgentEvent.from_dict(value),
        created_at=str(row["created_at"]),
    )


__all__ = [
    "DEFAULT_EVENT_PAYLOAD_BYTES",
    "DEFAULT_EVENT_RETENTION",
    "PublicEventCursorExpiredError",
    "PublicEventJournal",
    "PublicEventRecord",
]
