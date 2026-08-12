"""Filesystem-free reference stores for hosted run contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import sqlite3
import threading

from chulk.runs.async_store import AsyncRunStoreAdapter
from chulk.runs.store import SQLiteRunStore
from chulk.storage.migrations import SQLITE_MIGRATIONS


class InMemoryRunStore(SQLiteRunStore):
    """SQLite semantics in a process-owned in-memory connection."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        for migration in SQLITE_MIGRATIONS:
            migration.apply(self._connection)
            self._connection.execute(
                f"PRAGMA user_version = {migration.version}"
            )
        self._connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._connection
                self._connection.commit()
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class AsyncInMemoryRunStore(AsyncRunStoreAdapter):
    """Async facade over one filesystem-free in-memory owner."""

    def __init__(self, store: InMemoryRunStore | None = None) -> None:
        selected = store or InMemoryRunStore()
        super().__init__(selected)
        self.store: InMemoryRunStore = selected


__all__ = ["AsyncInMemoryRunStore", "InMemoryRunStore"]
