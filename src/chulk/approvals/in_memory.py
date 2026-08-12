"""Filesystem-free approval stores for local mode and contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import sqlite3

from chulk.approvals.async_store import AsyncApprovalStoreAdapter
from chulk.approvals.store import SQLiteApprovalStore
from chulk.runs.in_memory import InMemoryRunStore


class InMemoryApprovalStore(SQLiteApprovalStore):
    """Approval persistence sharing a process-owned durable run database."""

    def __init__(self, runs: InMemoryRunStore) -> None:
        self.runs = runs

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self.runs._connect() as conn:
            yield conn


class AsyncInMemoryApprovalStore(AsyncApprovalStoreAdapter):
    """Async facade over the filesystem-free approval owner."""

    def __init__(self, store: InMemoryApprovalStore) -> None:
        super().__init__(store)
        self.store: InMemoryApprovalStore = store


__all__ = ["AsyncInMemoryApprovalStore", "InMemoryApprovalStore"]
