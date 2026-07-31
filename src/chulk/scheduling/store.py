"""Transactional SQLite store for profile-owned automation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
import sqlite3

from chulk.scheduling._store_delivery import _ScheduleDeliveryMixin
from chulk.scheduling._store_jobs import _ScheduleJobsMixin
from chulk.scheduling._store_runs import _ScheduleRunsMixin
from chulk.scheduling._store_support import (
    AutomationConflictError,
    AutomationNotFoundError,
    DEFAULT_AUTOMATION_LEASE_SECONDS,
    _ScheduleSupportMixin,
    _required,
)
from chulk.scheduling._store_triggers import _ScheduleTriggersMixin
from chulk.scheduling.recurrence import RecurrenceCalculator
from chulk.storage import initialize_sqlite_database, sqlite_connection

AutomationConflictError.__module__ = __name__
AutomationNotFoundError.__module__ = __name__


class SQLiteScheduleStore(
    _ScheduleJobsMixin,
    _ScheduleRunsMixin,
    _ScheduleTriggersMixin,
    _ScheduleDeliveryMixin,
    _ScheduleSupportMixin,
):
    """Persist jobs, occurrences, attempts, controls, and trigger deduplication."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        profile_id: str = "default",
        recurrence_calculator: RecurrenceCalculator | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.profile_id = _required(profile_id, "profile_id")
        self.recurrence = recurrence_calculator or RecurrenceCalculator()
        initialize_sqlite_database(self.db_path)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_connection(self.db_path)

    def _serialize_job_mutation(
        self,
        conn: sqlite3.Connection,
        job_id: str,
    ) -> None:
        """Let backends serialize state transitions for one job."""

    def _serialize_control_action(
        self,
        conn: sqlite3.Connection,
        idempotency_key: str,
    ) -> None:
        """Let backends serialize profile-wide control idempotency keys."""

    def _serialize_trigger_ingest(
        self,
        conn: sqlite3.Connection,
        trigger_id: str,
    ) -> None:
        """Let backends serialize source-event deduplication per trigger."""

    def _claim_lock_clause(self) -> str:
        """Return backend-specific locking for due-job candidates."""
        return ""

    def _claim_candidate_limit(self, limit: int) -> int:
        """Overfetch SQLite candidates that may terminalize before claiming."""
        return limit * 4

    def _recovery_lock_clause(self) -> str:
        """Return backend-specific locking for schedule recovery workers."""
        return ""


__all__ = [
    "AutomationConflictError",
    "AutomationNotFoundError",
    "DEFAULT_AUTOMATION_LEASE_SECONDS",
    "SQLiteScheduleStore",
]
