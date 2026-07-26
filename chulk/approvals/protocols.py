"""Host persistence protocols for durable approvals."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from chulk.approvals.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalSubmission,
)
from chulk.hosting.scope import ExecutionScope


@runtime_checkable
class ApprovalStore(Protocol):
    def create(
        self,
        scope: ExecutionScope,
        submission: ApprovalSubmission,
    ) -> ApprovalRequest: ...

    def get(
        self,
        scope: ExecutionScope,
        approval_id: str,
    ) -> ApprovalRequest: ...

    def list(
        self,
        scope: ExecutionScope,
        *,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[ApprovalRequest, ...]: ...

    def decide(
        self,
        scope: ExecutionScope,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        reason: str,
        idempotency_key: str,
    ) -> ApprovalRequest: ...

    def consume(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        expected_revision: int,
    ) -> ApprovalRequest: ...

    def invalidate(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        reason: str,
    ) -> ApprovalRequest: ...

    def cancel(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        reason: str,
    ) -> ApprovalRequest: ...

    def expire(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[ApprovalRequest, ...]: ...


@runtime_checkable
class AsyncApprovalStore(Protocol):
    async def create(
        self,
        scope: ExecutionScope,
        submission: ApprovalSubmission,
    ) -> ApprovalRequest: ...

    async def get(
        self,
        scope: ExecutionScope,
        approval_id: str,
    ) -> ApprovalRequest: ...

    async def decide(
        self,
        scope: ExecutionScope,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        reason: str,
        idempotency_key: str,
    ) -> ApprovalRequest: ...

    async def consume(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        expected_revision: int,
    ) -> ApprovalRequest: ...

    async def list(
        self,
        scope: ExecutionScope,
        *,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[ApprovalRequest, ...]: ...

    async def invalidate(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        reason: str,
    ) -> ApprovalRequest: ...

    async def cancel(
        self,
        scope: ExecutionScope,
        approval_id: str,
        *,
        reason: str,
    ) -> ApprovalRequest: ...

    async def expire(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[ApprovalRequest, ...]: ...


__all__ = ["ApprovalStore", "AsyncApprovalStore"]
