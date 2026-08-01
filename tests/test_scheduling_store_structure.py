"""Structural compatibility guards for the split scheduling store."""

from __future__ import annotations

from inspect import signature

import chulk.scheduling as scheduling
import pytest
from chulk.scheduling.store import (
    AutomationConflictError,
    AutomationNotFoundError,
    DEFAULT_AUTOMATION_LEASE_SECONDS,
    SQLiteScheduleStore,
)


PUBLIC_METHODS = {
    "approve",
    "cancel",
    "claim_due",
    "complete",
    "create",
    "create_completion_trigger",
    "create_webhook_trigger",
    "delivery_history",
    "emit_completion",
    "events",
    "fail",
    "get",
    "get_run",
    "ingest_webhook",
    "list",
    "mark_delivery",
    "pause",
    "recover_expired",
    "renew_lease",
    "resume",
    "run_now",
    "runs",
    "triggers",
    "update",
}

BACKEND_HOOKS = {
    "_claim_candidate_limit",
    "_claim_lock_clause",
    "_connect",
    "_recovery_lock_clause",
    "_serialize_control_action",
    "_serialize_job_mutation",
    "_serialize_trigger_ingest",
}


def test_schedule_store_keeps_its_public_identity_and_constructor() -> None:
    assert scheduling.SQLiteScheduleStore is SQLiteScheduleStore
    assert scheduling.AutomationConflictError is AutomationConflictError
    assert scheduling.AutomationNotFoundError is AutomationNotFoundError
    assert (
        scheduling.DEFAULT_AUTOMATION_LEASE_SECONDS
        == DEFAULT_AUTOMATION_LEASE_SECONDS
        == 120
    )
    assert SQLiteScheduleStore.__module__ == "chulk.scheduling.store"
    assert AutomationConflictError.__module__ == "chulk.scheduling.store"
    assert AutomationNotFoundError.__module__ == "chulk.scheduling.store"

    parameters = signature(SQLiteScheduleStore.__init__).parameters
    assert tuple(parameters) == (
        "self",
        "db_path",
        "profile_id",
        "recurrence_calculator",
    )
    assert parameters["profile_id"].default == "default"
    assert parameters["recurrence_calculator"].default is None


def test_schedule_store_keeps_public_methods_and_direct_backend_hooks() -> None:
    missing = {
        name
        for name in PUBLIC_METHODS
        if not callable(getattr(SQLiteScheduleStore, name, None))
    }
    assert not missing
    assert BACKEND_HOOKS <= SQLiteScheduleStore.__dict__.keys()


def test_postgres_schedule_store_still_subclasses_and_overrides_backend_hooks() -> None:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg")
    pytest.importorskip("alembic")

    from chulk.postgres import PostgreSQLScheduleStore

    assert issubclass(PostgreSQLScheduleStore, SQLiteScheduleStore)
    assert {
        "_claim_candidate_limit",
        "_claim_lock_clause",
        "_recovery_lock_clause",
        "_serialize_control_action",
        "_serialize_job_mutation",
        "_serialize_trigger_ingest",
    } <= PostgreSQLScheduleStore.__dict__.keys()
