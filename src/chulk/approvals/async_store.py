"""Non-blocking async adapters for durable approval persistence."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from chulk.approvals.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalSubmission,
)
from chulk.approvals.protocols import ApprovalStore
from chulk.approvals.store import SQLiteApprovalStore
from chulk.hosting.scope import ExecutionScope


class AsyncApprovalStoreAdapter:
    def __init__(self, store: ApprovalStore) -> None:
        self.store = store

    async def call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.store, method)
        return await asyncio.to_thread(target, *args, **kwargs)

    async def create(
        self,
        scope: ExecutionScope,
        submission: ApprovalSubmission,
    ) -> ApprovalRequest:
        return await self.call("create", scope, submission)

    async def get(
        self,
        scope: ExecutionScope,
        approval_id: str,
    ) -> ApprovalRequest:
        return await self.call("get", scope, approval_id)

    async def list(
        self,
        scope: ExecutionScope,
        *,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[ApprovalRequest, ...]:
        return await self.call(
            "list",
            scope,
            run_id=run_id,
            status=status,
            limit=limit,
        )

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
        return await self.call(
            "decide",
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
        return await self.call(
            "consume",
            scope,
            approval_id,
            expected_revision=expected_revision,
        )

    async def invalidate(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        reason: str,
    ) -> ApprovalRequest:
        return await self.call(
            "invalidate",
            scope,
            approval_id,
            reason=reason,
        )

    async def cancel(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        reason: str,
    ) -> ApprovalRequest:
        return await self.call(
            "cancel",
            scope,
            approval_id,
            reason=reason,
        )

    async def expire(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[ApprovalRequest, ...]:
        return await self.call("expire", now=now)


class AsyncSQLiteApprovalStore(AsyncApprovalStoreAdapter):
    def __init__(self, db_path: Path | str) -> None:
        super().__init__(SQLiteApprovalStore(db_path))


__all__ = [
    "AsyncApprovalStoreAdapter",
    "AsyncSQLiteApprovalStore",
]
