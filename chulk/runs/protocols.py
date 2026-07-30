"""Host-provided persistence boundaries for durable runs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class RunStore(Protocol):
    """Complete synchronous durable-run state boundary."""

    def submit(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        actor: str = "host",
    ) -> RunRecord: ...

    def submit_parent(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        policy: ParentRunPolicy,
        actor: str = "host",
    ) -> ParentRunRecord: ...

    def submit_child(
        self,
        parent_scope: ExecutionScope,
        child_scope: ExecutionScope,
        submission: RunSubmission,
        *,
        definition_revision: str,
        actor: str = "host",
    ) -> ChildRunRecord: ...

    def get_parent(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> ParentRunRecord: ...

    def children(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> tuple[ChildRunRecord, ...]: ...

    def get_child(
        self,
        scope: ExecutionScope,
        child_run_id: str,
    ) -> ChildRunRecord: ...

    def record_child_progress(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        sequence: int,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> ChildRunProgress: ...

    def request_child_cancellation(
        self,
        parent_scope: ExecutionScope,
        child_run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> ChildRunRecord: ...

    def request_parent_cancellation(
        self,
        parent_scope: ExecutionScope,
        *,
        actor: str,
        reason: str,
    ) -> ParentRunRecord: ...

    def aggregate_children(
        self,
        parent_scope: ExecutionScope,
        *,
        actor: str,
        idempotency_key: str,
    ) -> ParentRunRecord: ...

    def parent_completion(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> ParentCompletion | None: ...

    def claim_parent_completion(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        parent_run_id: str | None = None,
    ) -> ParentCompletionClaim | None: ...

    def complete_parent_completion(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
    ) -> ParentCompletion: ...

    def fail_parent_completion(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        reason: str,
    ) -> ParentCompletion: ...

    def mark_parent_completion_unknown(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        reason: str,
    ) -> ParentCompletion: ...

    def reconcile_parent_completion(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
        *,
        delivered: bool,
        actor: str,
        reason: str,
    ) -> ParentCompletion: ...

    def get(self, scope: ExecutionScope, run_id: str) -> RunRecord: ...

    def events(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]: ...

    def record_event(
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
    ) -> RunEvent: ...

    def attempts(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[AttemptRecord, ...]: ...

    def checkpoints(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[Checkpoint, ...]: ...

    def effects(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[EffectRecord, ...]: ...

    def claim(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        run_id: str | None = None,
    ) -> RunClaim | None: ...

    def renew(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        lease_seconds: int = 120,
    ) -> RunClaim: ...

    def start_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
    ) -> AttemptRecord: ...

    def checkpoint(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
    ) -> Checkpoint: ...

    def begin_effect(
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
    ) -> EffectRecord: ...

    def mark_effect_started(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
    ) -> EffectRecord: ...

    def complete_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        result_digest: str,
    ) -> EffectRecord: ...

    def fail_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord: ...

    def mark_effect_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord: ...

    def reconcile_effect(
        self,
        scope: ExecutionScope,
        effect_id: str,
        *,
        decision: ReconciliationDecision,
        actor: str,
        reason: str,
        result_digest: str | None = None,
    ) -> ReconciliationRecord: ...

    def complete_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> RunRecord: ...

    def fail_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        reason: str,
        retryable: bool,
    ) -> RunRecord: ...

    def pause_for_approval(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        approval_id: str,
        payload: Mapping[str, Any],
    ) -> RunRecord: ...

    def resume(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord: ...

    def request_cancellation(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord: ...

    def cancel(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
        claim: RunClaim | None = None,
    ) -> RunRecord: ...

    def complete(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        result: Mapping[str, Any],
    ) -> RunRecord: ...

    def fail(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        reason: str,
    ) -> RunRecord: ...

    def mark_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        reason: str,
    ) -> RunRecord: ...

    def dead_letter(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord: ...

    def steer(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        instruction: str,
        actor: str,
        idempotency_key: str,
    ) -> RunRecord: ...

    def reconcile_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[RunRecord, ...]: ...


@runtime_checkable
class AsyncRunStore(Protocol):
    """Native async equivalent of the durable-run state boundary."""

    async def submit(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        actor: str = "host",
    ) -> RunRecord: ...

    async def submit_parent(
        self,
        scope: ExecutionScope,
        submission: RunSubmission,
        *,
        policy: ParentRunPolicy,
        actor: str = "host",
    ) -> ParentRunRecord: ...

    async def submit_child(
        self,
        parent_scope: ExecutionScope,
        child_scope: ExecutionScope,
        submission: RunSubmission,
        *,
        definition_revision: str,
        actor: str = "host",
    ) -> ChildRunRecord: ...

    async def get_parent(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> ParentRunRecord: ...

    async def children(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> tuple[ChildRunRecord, ...]: ...

    async def get_child(
        self,
        scope: ExecutionScope,
        child_run_id: str,
    ) -> ChildRunRecord: ...

    async def record_child_progress(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        sequence: int,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> ChildRunProgress: ...

    async def request_child_cancellation(
        self,
        parent_scope: ExecutionScope,
        child_run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> ChildRunRecord: ...

    async def request_parent_cancellation(
        self,
        parent_scope: ExecutionScope,
        *,
        actor: str,
        reason: str,
    ) -> ParentRunRecord: ...

    async def aggregate_children(
        self,
        parent_scope: ExecutionScope,
        *,
        actor: str,
        idempotency_key: str,
    ) -> ParentRunRecord: ...

    async def parent_completion(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
    ) -> ParentCompletion | None: ...

    async def claim_parent_completion(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        parent_run_id: str | None = None,
    ) -> ParentCompletionClaim | None: ...

    async def complete_parent_completion(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
    ) -> ParentCompletion: ...

    async def fail_parent_completion(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        reason: str,
    ) -> ParentCompletion: ...

    async def mark_parent_completion_unknown(
        self,
        scope: ExecutionScope,
        claim: ParentCompletionClaim,
        *,
        reason: str,
    ) -> ParentCompletion: ...

    async def reconcile_parent_completion(
        self,
        scope: ExecutionScope,
        parent_run_id: str,
        *,
        delivered: bool,
        actor: str,
        reason: str,
    ) -> ParentCompletion: ...

    async def get(self, scope: ExecutionScope, run_id: str) -> RunRecord: ...

    async def claim(
        self,
        scope: ExecutionScope,
        *,
        worker_id: str,
        lease_seconds: int = 120,
        run_id: str | None = None,
    ) -> RunClaim | None: ...

    async def renew(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        lease_seconds: int = 120,
    ) -> RunClaim: ...

    async def events(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[RunEvent, ...]: ...

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
    ) -> RunEvent: ...

    async def attempts(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[AttemptRecord, ...]: ...

    async def checkpoints(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[Checkpoint, ...]: ...

    async def effects(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        step_id: str | None = None,
    ) -> tuple[EffectRecord, ...]: ...

    async def start_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
    ) -> AttemptRecord: ...

    async def checkpoint(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
    ) -> Checkpoint: ...

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
    ) -> EffectRecord: ...

    async def mark_effect_started(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
    ) -> EffectRecord: ...

    async def complete_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        result_digest: str,
    ) -> EffectRecord: ...

    async def fail_effect(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord: ...

    async def mark_effect_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        effect_id: str,
        *,
        reason: str,
    ) -> EffectRecord: ...

    async def reconcile_effect(
        self,
        scope: ExecutionScope,
        effect_id: str,
        *,
        decision: ReconciliationDecision,
        actor: str,
        reason: str,
        result_digest: str | None = None,
    ) -> ReconciliationRecord: ...

    async def complete_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> RunRecord: ...

    async def fail_step(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        reason: str,
        retryable: bool,
    ) -> RunRecord: ...

    async def pause_for_approval(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        step_id: str,
        *,
        approval_id: str,
        payload: Mapping[str, Any],
    ) -> RunRecord: ...

    async def resume(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord: ...

    async def request_cancellation(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord: ...

    async def cancel(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
        claim: RunClaim | None = None,
    ) -> RunRecord: ...

    async def complete(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        result: Mapping[str, Any],
    ) -> RunRecord: ...

    async def fail(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        reason: str,
    ) -> RunRecord: ...

    async def mark_unknown(
        self,
        scope: ExecutionScope,
        claim: RunClaim,
        *,
        reason: str,
    ) -> RunRecord: ...

    async def dead_letter(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        actor: str,
        reason: str,
    ) -> RunRecord: ...

    async def steer(
        self,
        scope: ExecutionScope,
        run_id: str,
        *,
        instruction: str,
        actor: str,
        idempotency_key: str,
    ) -> RunRecord: ...

    async def reconcile_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> tuple[RunRecord, ...]: ...


__all__ = ["AsyncRunStore", "RunStore"]
