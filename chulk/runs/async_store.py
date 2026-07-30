"""Non-blocking async adapters for the complete durable-run contract."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from chulk.hosting.scope import ExecutionScope
from chulk.runs.models import (
    AttemptRecord,
    Checkpoint,
    EffectRecord,
    ReconciliationDecision,
    ReconciliationRecord,
    RunClaim,
    RunEvent,
    RunRecord,
    RunSubmission,
)
from chulk.runs.parent_child import (
    ChildRunProgress,
    ChildRunRecord,
    ParentCompletion,
    ParentCompletionClaim,
    ParentRunPolicy,
    ParentRunRecord,
)
from chulk.runs.protocols import RunStore
from chulk.runs.store import SQLiteRunStore


class AsyncRunStoreAdapter:
    """Run blocking reference stores off-loop while preserving their CAS rules."""

    def __init__(self, store: RunStore) -> None:
        self.store = store

    async def call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.store, method)
        return await asyncio.to_thread(target, *args, **kwargs)

    async def submit(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        actor: str = "host",
    ) -> RunRecord:
        return await self.call("submit", scope, submission, actor=actor)

    async def submit_parent(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        policy: ParentRunPolicy,
        actor: str = "host",
    ) -> ParentRunRecord:
        return await self.call(
            "submit_parent",
            scope,
            submission,
            policy=policy,
            actor=actor,
        )

    async def submit_child(
        self,
        parent_scope: ExecutionScope,
        child_scope: ExecutionScope,
        submission: RunSubmission,
        *,
        definition_revision: str,
        actor: str = "host",
    ) -> ChildRunRecord:
        return await self.call(
            "submit_child",
            parent_scope,
            child_scope,
            submission,
            definition_revision=definition_revision,
            actor=actor,
        )

    async def get_parent(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> ParentRunRecord:
        return await self.call("get_parent", scope, parent_run_id)

    async def children(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> tuple[ChildRunRecord, ...]:
        return await self.call("children", scope, parent_run_id)

    async def get_child(
        self,
        scope: ExecutionScope,
        child_run_id: str,
    ) -> ChildRunRecord:
        return await self.call("get_child", scope, child_run_id)

    async def record_child_progress(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        sequence: int,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> ChildRunProgress:
        return await self.call(
            "record_child_progress",
            scope,
            claim,
            sequence=sequence,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    async def request_child_cancellation(
        self,
        parent_scope: ExecutionScope,
        child_run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> ChildRunRecord:
        return await self.call(
            "request_child_cancellation",
            parent_scope,
            child_run_id,
            actor=actor,
            reason=reason,
        )

    async def request_parent_cancellation(
        self,
        parent_scope: ExecutionScope,
        *,
        actor: str,
        reason: str,
    ) -> ParentRunRecord:
        return await self.call(
            "request_parent_cancellation",
            parent_scope,
            actor=actor,
            reason=reason,
        )

    async def aggregate_children(
        self,
        parent_scope: ExecutionScope,
        *,
        actor: str,
        idempotency_key: str,
    ) -> ParentRunRecord:
        return await self.call(
            "aggregate_children",
            parent_scope,
            actor=actor,
            idempotency_key=idempotency_key,
        )

    async def parent_completion(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> ParentCompletion | None:
        return await self.call("parent_completion", scope, parent_run_id)

    async def claim_parent_completion(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        parent_run_id: str | None = None,
    ) -> ParentCompletionClaim | None:
        return await self.call(
            "claim_parent_completion",
            scope,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            parent_run_id=parent_run_id,
        )

    async def complete_parent_completion(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
    ) -> ParentCompletion:
        return await self.call("complete_parent_completion", scope, claim)

    async def fail_parent_completion(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        reason: str,
    ) -> ParentCompletion:
        return await self.call(
            "fail_parent_completion",
            scope,
            claim,
            reason=reason,
        )

    async def mark_parent_completion_unknown(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        reason: str,
    ) -> ParentCompletion:
        return await self.call(
            "mark_parent_completion_unknown",
            scope,
            claim,
            reason=reason,
        )

    async def reconcile_parent_completion(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
        *,
        delivered: bool,
        actor: str,
        reason: str,
    ) -> ParentCompletion:
        return await self.call(
            "reconcile_parent_completion",
            scope,
            parent_run_id,
            delivered=delivered,
            actor=actor,
            reason=reason,
        )

    async def get(
        self,
        scope: ExecutionScope,
        run_id: str,
    ) -> RunRecord:
        return await self.call("get", scope, run_id)

    async def events(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]:
        return await self.call(
            "events",
            scope,
            run_id,
            after_sequence=after_sequence,
        )

    async def record_event(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        name: str,
        actor: str,
        payload: Mapping[str, Any],
        step_id: str | None = None,
        causation_id: str | None = None,
        correlation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunEvent:
        return await self.call(
            "record_event",
            scope,
            run_id,
            name=name,
            actor=actor,
            payload=payload,
            step_id=step_id,
            causation_id=causation_id,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

    async def attempts(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[AttemptRecord, ...]:
        return await self.call("attempts", scope, run_id, step_id=step_id)

    async def checkpoints(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[Checkpoint, ...]:
        return await self.call("checkpoints", scope, run_id, step_id=step_id)

    async def effects(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[EffectRecord, ...]:
        return await self.call("effects", scope, run_id, step_id=step_id)

    async def claim(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        run_id: str | None = None,
    ) -> RunClaim | None:
        return await self.call(
            "claim",
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
        return await self.call(
            "renew",
            scope,
            claim,
            lease_seconds=lease_seconds,
        )

    async def start_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
    ) -> AttemptRecord:
        return await self.call("start_step", scope, claim, step_id)

    async def checkpoint(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
    ) -> Checkpoint:
        return await self.call(
            "checkpoint",
            scope,
            claim,
            step_id,
            kind=kind,
            payload=payload,
        )

    async def begin_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        logical_key: str,
        tool_name: str,
        tool_version: str,
        schema_version: str,
        arguments_digest: str,
    ) -> EffectRecord:
        return await self.call(
            "begin_effect",
            scope,
            claim,
            step_id,
            logical_key=logical_key,
            tool_name=tool_name,
            tool_version=tool_version,
            schema_version=schema_version,
            arguments_digest=arguments_digest,
        )

    async def mark_effect_started(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
    ) -> EffectRecord:
        return await self.call(
            "mark_effect_started",
            scope,
            claim,
            effect_id,
        )

    async def complete_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        result_digest: str,
    ) -> EffectRecord:
        return await self.call(
            "complete_effect",
            scope,
            claim,
            effect_id,
            result_digest=result_digest,
        )

    async def fail_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord:
        return await self.call(
            "fail_effect",
            scope,
            claim,
            effect_id,
            reason=reason,
        )

    async def mark_effect_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord:
        return await self.call(
            "mark_effect_unknown",
            scope,
            claim,
            effect_id,
            reason=reason,
        )

    async def reconcile_effect(
        self,
        scope: ExecutionScope,
        effect_id: str,
        *,
        decision: ReconciliationDecision,
        actor: str,
        reason: str,
        result_digest: str | None = None,
    ) -> ReconciliationRecord:
        return await self.call(
            "reconcile_effect",
            scope,
            effect_id,
            decision=decision,
            actor=actor,
            reason=reason,
            result_digest=result_digest,
        )

    async def complete_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        return await self.call(
            "complete_step",
            scope,
            claim,
            step_id,
            result=result,
        )

    async def fail_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        reason: str,
        retryable: bool,
    ) -> RunRecord:
        return await self.call(
            "fail_step",
            scope,
            claim,
            step_id,
            reason=reason,
            retryable=retryable,
        )

    async def pause_for_approval(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        approval_id: str,
        payload: Mapping[str, Any],
    ) -> RunRecord:
        return await self.call(
            "pause_for_approval",
            scope,
            claim,
            step_id,
            approval_id=approval_id,
            payload=payload,
        )

    async def resume(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord:
        return await self.call(
            "resume",
            scope,
            run_id,
            actor=actor,
            reason=reason,
        )

    async def request_cancellation(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord:
        return await self.call(
            "request_cancellation",
            scope,
            run_id,
            actor=actor,
            reason=reason,
        )

    async def cancel(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
        claim: RunClaim | None = None,
    ) -> RunRecord:
        return await self.call(
            "cancel",
            scope,
            run_id,
            actor=actor,
            reason=reason,
            claim=claim,
        )

    async def complete(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        result: Mapping[str, Any],
    ) -> RunRecord:
        return await self.call("complete", scope, claim, result=result)

    async def fail(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        reason: str,
    ) -> RunRecord:
        return await self.call("fail", scope, claim, reason=reason)

    async def mark_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        reason: str,
    ) -> RunRecord:
        return await self.call("mark_unknown", scope, claim, reason=reason)

    async def dead_letter(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord:
        return await self.call(
            "dead_letter",
            scope,
            run_id,
            actor=actor,
            reason=reason,
        )

    async def steer(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        instruction: str,
        actor: str,
        idempotency_key: str,
    ) -> RunRecord:
        return await self.call(
            "steer",
            scope,
            run_id,
            instruction=instruction,
            actor=actor,
            idempotency_key=idempotency_key,
        )

    async def reconcile_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[RunRecord, ...]:
        return await self.call("reconcile_expired", now=now)

    async def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            await asyncio.to_thread(close)


class AsyncSQLiteRunStore(AsyncRunStoreAdapter):
    """Async facade over the SQLite reference adapter."""

    def __init__(self, db_path: Path | str) -> None:
        super().__init__(SQLiteRunStore(db_path))


__all__ = ["AsyncRunStoreAdapter", "AsyncSQLiteRunStore"]
