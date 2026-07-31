"""SQLite reference store for durable hosted runs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
import sqlite3

from chulk.runs._store_effects import _RunStoreEffectMixin
from chulk.runs._store_parent_child import _ParentChildRunStoreMixin
from chulk.runs._store_support import _RunStoreSupportMixin
from chulk.runs._store_transitions import _RunStoreTransitionMixin
from chulk.runs.errors import (
    EffectConflictError,
    InvalidRunTransitionError,
    RunConflictError,
    RunLeaseError,
    RunNotFoundError,
)
from chulk.storage import initialize_sqlite_database, sqlite_connection


class SQLiteRunStore(
    _RunStoreSupportMixin,
    _ParentChildRunStoreMixin,
    _RunStoreEffectMixin,
    _RunStoreTransitionMixin,
):
    """Transactional, scope-aware durable-run store with append-only events."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        initialize_sqlite_database(self.db_path)

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return sqlite_connection(self.db_path)

    def _recovery_lock_clause(self) -> str:
        """Return backend-specific locking for expired run workers."""
        return ""

    def _parent_completion_claim_lock_clause(self) -> str:
        """Return backend-specific locking for parent completion claims."""
        return ""


__all__ = [
    "EffectConflictError",
    "InvalidRunTransitionError",
    "RunConflictError",
    "RunLeaseError",
    "RunNotFoundError",
    "SQLiteRunStore",
]
