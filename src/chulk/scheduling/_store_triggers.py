"""Trigger registration and ingestion operations for scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
import hmac
import secrets
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.scheduling._store_support import (
    AutomationConflictError,
    AutomationNotFoundError,
    _ScheduleStoreMixin,
    _encode,
    _json,
    _required,
    _row_to_envelope,
    _row_to_trigger,
    _utc,
    _utc_now,
)
from chulk.scheduling.models import (
    AutomationJobStatus,
    AutomationRunReason,
    AutomationTrigger,
    TriggerEnvelope,
    TriggerKind,
    TriggerTrust,
)


class _ScheduleTriggersMixin(_ScheduleStoreMixin):
    def create_webhook_trigger(
        self,
        job_id: str,
    ) -> tuple[AutomationTrigger, str]:
        token = secrets.token_urlsafe(32)
        return self._create_trigger(
            job_id,
            kind=TriggerKind.WEBHOOK,
            secret_digest=sha256(token.encode()).hexdigest(),
        ), token

    def create_completion_trigger(
        self,
        job_id: str,
        *,
        kind: TriggerKind,
        source_resource_id: str,
    ) -> AutomationTrigger:
        selected = TriggerKind(kind)
        if selected is TriggerKind.WEBHOOK:
            raise ValueError("use create_webhook_trigger for webhook triggers")
        return self._create_trigger(
            job_id,
            kind=selected,
            source_resource_id=_required(source_resource_id, "source_resource_id"),
        )

    def triggers(self, job_id: str | None = None) -> tuple[AutomationTrigger, ...]:
        clauses = ["profile_id = ?"]
        params: list[object] = [self.profile_id]
        if job_id is not None:
            self.get(job_id)
            clauses.append("job_id = ?")
            params.append(job_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM automation_triggers
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at, id
                """,
                tuple(params),
            ).fetchall()
        return tuple(_row_to_trigger(row) for row in rows)

    def ingest_webhook(
        self,
        trigger_id: str,
        *,
        token: str,
        event_id: str,
        payload: Mapping[str, Any],
        occurred_at: datetime | None = None,
    ) -> TriggerEnvelope:
        observed = _utc(occurred_at or _utc_now(), "occurred_at")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM automation_triggers
                WHERE profile_id = ? AND id = ? AND kind = 'webhook' AND enabled = 1
                """,
                (self.profile_id, trigger_id),
            ).fetchone()
            if row is None:
                raise AutomationNotFoundError(
                    f"Automation webhook trigger not found: {trigger_id}"
                )
            digest = sha256(token.encode()).hexdigest()
            if not hmac.compare_digest(digest, str(row["secret_digest"])):
                raise PermissionError("invalid webhook trigger credential")
        return self._ingest_trigger(
            _row_to_trigger(row),
            event_id=event_id,
            payload=payload,
            trust=TriggerTrust.TRUSTED,
            occurred_at=observed,
        )

    def emit_completion(
        self,
        *,
        kind: TriggerKind,
        source_resource_id: str,
        source_event_id: str,
        payload: Mapping[str, Any],
        occurred_at: datetime | None = None,
    ) -> tuple[TriggerEnvelope, ...]:
        selected = TriggerKind(kind)
        if selected is TriggerKind.WEBHOOK:
            raise ValueError("completion kind cannot be webhook")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM automation_triggers
                WHERE profile_id = ? AND kind = ? AND source_resource_id = ?
                  AND enabled = 1
                ORDER BY created_at, id
                """,
                (self.profile_id, selected.value, source_resource_id),
            ).fetchall()
        return tuple(
            self._ingest_trigger(
                _row_to_trigger(row),
                event_id=source_event_id,
                payload=payload,
                trust=TriggerTrust.OWNER,
                occurred_at=_utc(occurred_at or _utc_now(), "occurred_at"),
            )
            for row in rows
        )

    def _ingest_trigger(
        self,
        trigger: AutomationTrigger,
        *,
        event_id: str,
        payload: Mapping[str, Any],
        trust: TriggerTrust,
        occurred_at: datetime,
    ) -> TriggerEnvelope:
        envelope = TriggerEnvelope(
            id=uuid4().hex,
            profile_id=self.profile_id,
            trigger_id=trigger.id,
            trust=trust,
            payload=payload,
            occurred_at=occurred_at,
            source_event_id=_required(event_id, "event_id"),
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_trigger_ingest(conn, trigger.id)
            existing = conn.execute(
                """
                SELECT * FROM automation_trigger_events
                WHERE profile_id = ? AND trigger_id = ? AND source_event_id = ?
                """,
                (self.profile_id, trigger.id, envelope.source_event_id),
            ).fetchone()
            if existing is not None:
                return _row_to_envelope(existing)
            conn.execute(
                """
                INSERT INTO automation_trigger_events (
                    id, profile_id, trigger_id, source_event_id, trust,
                    payload_json, occurred_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.id,
                    self.profile_id,
                    trigger.id,
                    envelope.source_event_id,
                    envelope.trust.value,
                    _json(dict(envelope.payload)),
                    _encode(envelope.occurred_at),
                    _encode(_utc_now()),
                ),
            )
            self._enqueue_request(
                conn,
                job_id=trigger.job_id,
                reason=AutomationRunReason.TRIGGER,
                occurrence_at=envelope.occurred_at,
                trigger_event_id=envelope.id,
                idempotency_key=f"trigger:{trigger.id}:{envelope.source_event_id}",
                now=_utc_now(),
            )
        return envelope

    def _create_trigger(
        self,
        job_id: str,
        *,
        kind: TriggerKind,
        source_resource_id: str | None = None,
        secret_digest: str | None = None,
    ) -> AutomationTrigger:
        self.get(job_id)
        trigger_id = uuid4().hex
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._serialize_job_mutation(conn, job_id)
            job = self._get_in(conn, job_id)
            if job.status is AutomationJobStatus.CANCELLED:
                raise AutomationConflictError(
                    "cannot attach a trigger to a cancelled automation"
                )
            conn.execute(
                """
                INSERT INTO automation_triggers (
                    id, profile_id, job_id, kind, source_resource_id,
                    secret_digest, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    trigger_id,
                    self.profile_id,
                    job_id,
                    TriggerKind(kind).value,
                    source_resource_id,
                    secret_digest,
                    _encode(now),
                    _encode(now),
                ),
            )
            revision = job.revision + 1
            conn.execute(
                """
                UPDATE automation_jobs
                SET status = CASE
                        WHEN status IN ('completed', 'expired') THEN 'active'
                        ELSE status
                    END,
                    revision = ?, updated_at = ?
                WHERE profile_id = ? AND id = ?
                """,
                (revision, _encode(now), self.profile_id, job_id),
            )
            self._event(
                conn,
                job_id,
                "trigger_created",
                revision,
                "operator",
                metadata={"trigger_id": trigger_id, "kind": TriggerKind(kind).value},
                now=now,
            )
            row = conn.execute(
                "SELECT * FROM automation_triggers WHERE id = ?",
                (trigger_id,),
            ).fetchone()
        assert row is not None
        return _row_to_trigger(row)

    def _has_enabled_triggers(self, conn: sqlite3.Connection, job_id: str) -> bool:
        return (
            conn.execute(
                """
                SELECT 1 FROM automation_triggers
                WHERE profile_id = ? AND job_id = ? AND enabled = 1 LIMIT 1
                """,
                (self.profile_id, job_id),
            ).fetchone()
            is not None
        )
