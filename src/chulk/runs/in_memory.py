"""Filesystem-free reference stores for hosted run contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
import sqlite3
import threading
from typing import Any

from chulk.hosting.scope import ExecutionScope
from chulk.runs.async_store import AsyncRunStoreAdapter
from chulk.runs.models import RunClaim, RunEvent, RunRecord, RunSubmission
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
    """Native async facade over one filesystem-free in-memory owner."""

    def __init__(self, store: InMemoryRunStore | None = None) -> None:
        selected = store or InMemoryRunStore()
        super().__init__(selected)
        self.store: InMemoryRunStore = selected
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[InMemoryRunStore]:
        async with self._lock:
            yield self.store

    async def submit(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        actor: str = "host",
    ) -> RunRecord:
        async with self._operation() as store:
            return store.submit(scope, submission, actor=actor)

    async def get(
        self,
        scope: ExecutionScope,
        run_id: str,
    ) -> RunRecord:
        async with self._operation() as store:
            return store.get(scope, run_id)

    async def claim(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        run_id: str | None = None,
    ) -> RunClaim | None:
        async with self._operation() as store:
            return store.claim(
                scope,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                run_id=run_id,
            )

    async def renew(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        lease_seconds: int = 120,
    ) -> RunClaim:
        async with self._operation() as store:
            return store.renew(
                scope,
                claim,
                lease_seconds=lease_seconds,
            )

    async def events(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]:
        async with self._operation() as store:
            return store.events(
                scope,
                run_id,
                after_sequence=after_sequence,
            )

    async def call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        """Invoke an extended sync store operation without blocking on I/O."""
        async with self._operation() as store:
            target = getattr(store, method)
            return target(*args, **kwargs)

    async def close(self) -> None:
        async with self._operation() as store:
            store.close()


__all__ = ["AsyncInMemoryRunStore", "InMemoryRunStore"]
