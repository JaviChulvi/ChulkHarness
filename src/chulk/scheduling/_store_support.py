"""Shared persistence support and codecs for the scheduling store."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import sqlite3
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from chulk.gateway import DeliveryTarget
from chulk.scheduling.models import (
    AutomationDeliveryAttempt,
    AutomationDeliveryState,
    AutomationJobEvent,
    AutomationJobStatus,
    AutomationRetryPolicy,
    AutomationRun,
    AutomationRunReason,
    AutomationRunStatus,
    AutomationTrigger,
    AmbiguousTimePolicy,
    MisfirePolicy,
    NonexistentTimePolicy,
    RecurrenceKind,
    RecurrenceSpec,
    ScheduledJob,
    TriggerEnvelope,
    TriggerKind,
    TriggerTrust,
)
from chulk.usage import BudgetScope, ExactCost, RunBudget, UnknownCostPolicy

if TYPE_CHECKING:
    from chulk.scheduling.recurrence import RecurrenceCalculator


DEFAULT_AUTOMATION_LEASE_SECONDS = 120


class AutomationConflictError(RuntimeError):
    """Raised when revision or idempotency expectations conflict."""


class AutomationNotFoundError(LookupError):
    """Raised when an automation resource is outside the owner profile."""


class _ScheduleStoreMixin:
    """Declare the shared owner surface consumed across store concerns."""

    profile_id: str
    recurrence: RecurrenceCalculator
    _action_replay: Any
    _claim_candidate_limit: Any
    _claim_lock_clause: Any
    _claimed_job: Any
    _connect: Any
    _control: Any
    _delivery_attempt: Any
    _enqueue_request: Any
    _event: Any
    _finish_claimed_request: Any
    _get_in: Any
    _has_enabled_triggers: Any
    _has_pending_requests: Any
    _next_attempt: Any
    _record_action: Any
    _recovery_lock_clause: Any
    _run_in: Any
    _serialize_control_action: Any
    _serialize_job_mutation: Any
    _serialize_trigger_ingest: Any
    _terminalize: Any
    get: Any
    get_run: Any
    recover_expired: Any


class _ScheduleSupportMixin(_ScheduleStoreMixin):
    def _get_in(self, conn: sqlite3.Connection, job_id: str) -> ScheduledJob:
        row = conn.execute(
            "SELECT * FROM automation_jobs WHERE profile_id = ? AND id = ?",
            (self.profile_id, job_id),
        ).fetchone()
        if row is None:
            raise AutomationNotFoundError(f"Scheduled job not found: {job_id}")
        return _row_to_job(row)

    def _run_in(self, conn: sqlite3.Connection, run_id: str) -> AutomationRun:
        row = conn.execute(
            "SELECT * FROM automation_runs WHERE profile_id = ? AND id = ?",
            (self.profile_id, run_id),
        ).fetchone()
        if row is None:
            raise AutomationNotFoundError(f"Automation run not found: {run_id}")
        return _row_to_run(row)

    def _event(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        action: str,
        revision: int,
        actor: str,
        *,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO automation_job_events (
                id, job_id, profile_id, action, revision, actor, run_id,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                job_id,
                self.profile_id,
                action,
                revision,
                actor,
                run_id,
                _json(dict(metadata or {})),
                _encode(now),
            ),
        )

    def _record_action(
        self,
        conn: sqlite3.Connection,
        *,
        key: str,
        job_id: str,
        action: str,
        fingerprint: str,
        revision: int,
        now: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO automation_control_actions (
                profile_id, idempotency_key, job_id, action, fingerprint,
                result_revision, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.profile_id,
                _required(key, "idempotency_key"),
                job_id,
                action,
                fingerprint,
                revision,
                _encode(now),
            ),
        )

    def _action_replay(
        self,
        conn: sqlite3.Connection,
        *,
        key: str,
        job_id: str,
        action: str,
        fingerprint: str,
    ) -> sqlite3.Row | None:
        row = conn.execute(
            """
            SELECT * FROM automation_control_actions
            WHERE profile_id = ? AND idempotency_key = ?
            """,
            (self.profile_id, _required(key, "idempotency_key")),
        ).fetchone()
        if row is None:
            return None
        if (
            str(row["job_id"]) != job_id
            or str(row["action"]) != action
            or str(row["fingerprint"]) != fingerprint
        ):
            raise AutomationConflictError(
                "idempotency key was already used for a different action"
            )
        return row


def _row_to_job(row: sqlite3.Row) -> ScheduledJob:
    recurrence_data = json.loads(str(row["recurrence_json"]))
    return ScheduledJob(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        target=DeliveryTarget(
            adapter=str(row["adapter"]),
            account_id=str(row["account_id"]),
            destination_id=str(row["destination_id"]),
            thread_id=str(row["thread_id"]) if row["thread_id"] else None,
        ),
        prompt=str(row["prompt"]),
        recurrence=_recurrence_from_dict(recurrence_data),
        next_run_at=_decode(str(row["next_run_at"])),
        scheduled_for=_decode(str(row["scheduled_for"])),
        status=AutomationJobStatus(str(row["status"])),
        budget=_budget_from_dict(json.loads(str(row["budget_json"]))),
        retry_policy=_retry_from_dict(json.loads(str(row["retry_json"]))),
        revision=int(row["revision"]),
        run_count=int(row["run_count"]),
        max_runs=int(row["max_runs"]) if row["max_runs"] is not None else None,
        requires_approval=bool(row["requires_approval"]),
        approved_at=_optional_datetime(row["approved_at"]),
        claim_token=str(row["claim_token"]) if row["claim_token"] else None,
        lease_until=_optional_datetime(row["lease_until"]),
        active_run_id=str(row["active_run_id"]) if row["active_run_id"] else None,
        last_run_at=_optional_datetime(row["last_run_at"]),
        last_error=str(row["last_error"]) if row["last_error"] else None,
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
    )


def _row_to_run(row: sqlite3.Row) -> AutomationRun:
    return AutomationRun(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        profile_id=str(row["profile_id"]),
        occurrence_at=_decode(str(row["occurrence_at"])),
        reason=AutomationRunReason(str(row["reason"])),
        status=AutomationRunStatus(str(row["status"])),
        attempt=int(row["attempt"]),
        claim_token=str(row["claim_token"]) if row["claim_token"] else None,
        worker_id=str(row["worker_id"]) if row["worker_id"] else None,
        lease_until=_optional_datetime(row["lease_until"]),
        started_at=_optional_datetime(row["started_at"]),
        finished_at=_optional_datetime(row["finished_at"]),
        result=json.loads(str(row["result_json"])),
        trace_id=str(row["trace_id"]) if row["trace_id"] else None,
        usage=json.loads(str(row["usage_json"])),
        cost=json.loads(str(row["cost_json"])),
        error=str(row["error"]) if row["error"] else None,
        artifact_refs=tuple(json.loads(str(row["artifact_refs_json"]))),
        delivery_state=AutomationDeliveryState(str(row["delivery_state"])),
        delivery_error=str(row["delivery_error"]) if row["delivery_error"] else None,
        trigger_event_id=(
            str(row["trigger_event_id"]) if row["trigger_event_id"] else None
        ),
        created_at=_decode(str(row["created_at"])),
        updated_at=_decode(str(row["updated_at"])),
    )


def _row_to_delivery_attempt(row: sqlite3.Row) -> AutomationDeliveryAttempt:
    return AutomationDeliveryAttempt(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        profile_id=str(row["profile_id"]),
        state=AutomationDeliveryState(str(row["state"])),
        error=str(row["error"]) if row["error"] else None,
        created_at=_decode(str(row["created_at"])),
    )


def _row_to_event(row: sqlite3.Row) -> AutomationJobEvent:
    return AutomationJobEvent(
        id=str(row["id"]),
        job_id=str(row["job_id"]),
        profile_id=str(row["profile_id"]),
        action=str(row["action"]),
        revision=int(row["revision"]),
        actor=str(row["actor"]),
        run_id=str(row["run_id"]) if row["run_id"] else None,
        metadata=json.loads(str(row["metadata_json"])),
        created_at=_decode(str(row["created_at"])),
    )


def _row_to_trigger(row: sqlite3.Row) -> AutomationTrigger:
    return AutomationTrigger(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        job_id=str(row["job_id"]),
        kind=TriggerKind(str(row["kind"])),
        source_resource_id=(
            str(row["source_resource_id"]) if row["source_resource_id"] else None
        ),
        secret_digest=str(row["secret_digest"]) if row["secret_digest"] else None,
        enabled=bool(row["enabled"]),
        created_at=_decode(str(row["created_at"])),
    )


def _row_to_envelope(row: sqlite3.Row) -> TriggerEnvelope:
    return TriggerEnvelope(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        trigger_id=str(row["trigger_id"]),
        trust=TriggerTrust(str(row["trust"])),
        payload=json.loads(str(row["payload_json"])),
        occurred_at=_decode(str(row["occurred_at"])),
        source_event_id=(
            str(row["source_event_id"]) if row["source_event_id"] else None
        ),
    )


def _recurrence_from_dict(value: Mapping[str, Any]) -> RecurrenceSpec:
    return RecurrenceSpec(
        kind=RecurrenceKind(str(value.get("kind", "once"))),
        timezone_name=str(value.get("timezone", "UTC")),
        interval_seconds=_optional_int(value.get("interval_seconds")),
        cron=_optional_string(value.get("cron")),
        rrule=_optional_string(value.get("rrule")),
        starts_at=_optional_timestamp(value.get("starts_at")),
        ends_at=_optional_timestamp(value.get("ends_at")),
        misfire_policy=MisfirePolicy(str(value.get("misfire_policy", "run_once"))),
        nonexistent_time_policy=NonexistentTimePolicy(
            str(value.get("nonexistent_time_policy", "shift_forward"))
        ),
        ambiguous_time_policy=AmbiguousTimePolicy(
            str(value.get("ambiguous_time_policy", "earliest"))
        ),
        misfire_grace_seconds=int(value.get("misfire_grace_seconds", 60)),
        max_catch_up=int(value.get("max_catch_up", 1)),
        jitter_seconds=int(value.get("jitter_seconds", 0)),
    )


def _retry_from_dict(value: Mapping[str, Any]) -> AutomationRetryPolicy:
    return AutomationRetryPolicy(
        max_attempts=int(value.get("max_attempts", 3)),
        initial_backoff_seconds=int(value.get("initial_backoff_seconds", 60)),
        max_backoff_seconds=int(value.get("max_backoff_seconds", 3_600)),
        multiplier=float(value.get("multiplier", 2.0)),
    )


def _budget_from_dict(value: Mapping[str, Any]) -> RunBudget:
    raw_cost = value.get("max_cost")
    cost = None
    if isinstance(raw_cost, Mapping) and raw_cost.get("amount") is not None:
        cost = ExactCost(
            Decimal(str(raw_cost["amount"])),
            currency=str(raw_cost.get("currency", "USD")),
            pricing_known=bool(raw_cost.get("pricing_known", True)),
            estimated=bool(raw_cost.get("estimated", False)),
            reported=bool(raw_cost.get("reported", False)),
        )
    return RunBudget(
        scope=BudgetScope(str(value.get("scope", "job"))),
        max_model_calls=_optional_int(value.get("max_model_calls")),
        max_tool_calls=_optional_int(value.get("max_tool_calls")),
        max_tokens=_optional_int(value.get("max_tokens")),
        max_cost=cost,
        deadline=_optional_timestamp(value.get("deadline")),
        unknown_cost_policy=UnknownCostPolicy(
            str(value.get("unknown_cost_policy", "fail_closed"))
        ),
    )


def _optional_timestamp(value: object) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value else None


def _optional_datetime(value: object) -> datetime | None:
    return _decode(str(value)) if value else None


def _optional_string(value: object) -> str | None:
    clean = str(value).strip() if value is not None else ""
    return clean or None


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None


def _required(value: str | None, label: str) -> str:
    clean = value.strip() if value is not None else ""
    if not clean:
        raise ValueError(f"{label} is required")
    return clean


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(_json(dict(value)).encode()).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _encode(value: datetime | None) -> str:
    return value.astimezone(timezone.utc).isoformat() if value is not None else ""


def _decode(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)
