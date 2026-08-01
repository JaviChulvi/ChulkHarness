"""Filesystem-free approval stores for local mode and contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
import sqlite3
from typing import Any

from chulk.approvals.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalSubmission,
)
from chulk.approvals.async_store import AsyncApprovalStoreAdapter
from chulk.approvals.store import SQLiteApprovalStore
from chulk.hosting.scope import ExecutionScope
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
    """Native async facade over the filesystem-free approval owner."""

    def __init__(self, store: InMemoryApprovalStore) -> None:
        super().__init__(store)
        self.store: InMemoryApprovalStore = store
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[InMemoryApprovalStore]:
        async with self._lock:
            yield self.store

    async def create(
        self,
        scope: ExecutionScope,
        submission: ApprovalSubmission,
    ) -> ApprovalRequest:
        async with self._operation() as store:
            return store.create(scope, submission)

    async def get(
        self,
        scope: ExecutionScope,
        approval_id: str,
    ) -> ApprovalRequest:
        async with self._operation() as store:
            return store.get(scope, approval_id)

    async def decide(
        self,
        scope: ExecutionScope,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        reason: str,
        idempotency_key: str,
    ) -> ApprovalRequest:
        async with self._operation() as store:
            return store.decide(
                scope,
                approval_id,
                decision,
                decided_by=decided_by,
                reason=reason,
                idempotency_key=idempotency_key,
            )

    async def consume(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        expected_revision: int,
    ) -> ApprovalRequest:
        async with self._operation() as store:
            return store.consume(
                scope,
                approval_id,
                expected_revision=expected_revision,
            )

    async def call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        async with self._operation() as store:
            target = getattr(store, method)
            return target(*args, **kwargs)


__all__ = ["AsyncInMemoryApprovalStore", "InMemoryApprovalStore"]
