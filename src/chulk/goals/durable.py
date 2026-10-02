"""Bind goal continuation to the existing durable run/effect owners."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

from chulk.goals.models import GoalStopReason
from chulk.hosting.scope import ExecutionScope
from chulk.runs.errors import RunLeaseError
from chulk.runs.execution import DurableEffectCoordinator, AsyncDurableEffectCoordinator
from chulk.runs.models import RunClaim, RunStatus, RunSubmission, StepDefinition, StepStatus
from chulk.runs.protocols import RunStore, AsyncRunStore


class GoalRecoveryRequired(RuntimeError):
    """A dispatched operation requires durable evidence or host reconciliation."""


def submission(goal_id: str) -> RunSubmission:
    # Goal steps remain authoritative; the durable run is one execution envelope.
    return RunSubmission(idempotency_key=f"goal:{goal_id}", input_digest=goal_id,
        definition_digest="goal-continuation-v1", steps=(StepDefinition("goal", "Goal execution"),),
        metadata={"goal_id": goal_id})


def require_runnable(status: RunStatus) -> None:
    if status in {RunStatus.UNKNOWN, RunStatus.FAILED, RunStatus.DEAD_LETTER}:
        raise GoalRecoveryRequired(f"Durable execution requires reconciliation: {status.value}")
    if status is RunStatus.WAITING_FOR_APPROVAL:
        raise GoalApprovalRequired("Durable approval is pending")


class GoalApprovalRequired(RuntimeError):
    """Release the goal worker while a durable approval is pending."""


@dataclass(slots=True)
class GoalRunBinding:
    runs: RunStore
    scope: ExecutionScope
    claim: RunClaim
    coordinator: DurableEffectCoordinator

    @classmethod
    def open(cls, runs: RunStore, scope: ExecutionScope, *, goal_id: str, worker_id: str,
             lease_seconds: int) -> GoalRunBinding:
        runs.submit(scope, submission(goal_id))
        runs.reconcile_expired(scope=scope)
        record = runs.get(scope, scope.run_id)
        require_runnable(record.status)
        claim = runs.claim(scope, worker_id=worker_id, lease_seconds=lease_seconds, run_id=scope.run_id)
        if claim is None:
            raise RunLeaseError("Durable goal execution is owned by another worker or is terminal")
        if record.steps[0].status is not StepStatus.COMPLETED:
            runs.start_step(scope, claim, "goal")
        return cls(runs, scope, claim, DurableEffectCoordinator(runs, scope=scope, claim=claim, step_id="goal"))

    def assert_boundary(self) -> None:
        self.runs.assert_claim(self.scope, self.claim)

    def heartbeat(self, *, lease_seconds: int) -> None:
        self.claim = self.runs.renew(self.scope, self.claim, lease_seconds=lease_seconds)
        self.coordinator.claim = self.claim

    def install(self, runtime: Any) -> None:
        from chulk.runs.events import RunEventPublisher
        self.scope.assert_resumable(runtime.execution_scope)
        if runtime.events.public_event_sink is not None:
            self.coordinator.publisher = RunEventPublisher(self.runs, runtime.events.public_event_sink, scope=self.scope)
            self.coordinator.publisher.publish()
        runtime._tool_executor.durable_effects = self.coordinator
        services = getattr(runtime, "resolved_services", None)
        if services is not None:
            from chulk.approvals.service import DurableApprovalCoordinator, DurableApprovalService
            runtime._tool_executor.durable_approvals = DurableApprovalCoordinator(
                DurableApprovalService(services.approvals, self.runs), self.coordinator,
                scope=self.scope, claim=self.claim, step_id="goal")

    def finish(self, reason: GoalStopReason, *, turn_id: str) -> None:
        current = self.runs.get(self.scope, self.scope.run_id)
        if current.status is not RunStatus.RUNNING:
            return
        if reason is GoalStopReason.COMPLETED:
            if current.steps[0].status is not StepStatus.COMPLETED:
                self.runs.complete_step(self.scope, self.claim, "goal", result={"turn_id": turn_id})
            self.runs.complete(self.scope, self.claim, result={"turn_id": turn_id})
        elif reason is GoalStopReason.CANCELLED:
            self.runs.cancel(self.scope, self.scope.run_id, actor=self.claim.worker_id,
                             reason="goal cancelled", claim=self.claim)
        elif reason is not GoalStopReason.LEASE_LOST:
            self.runs.yield_step(self.scope, self.claim, "goal", continuation={"turn_id": turn_id, "reason": reason.value})
        if self.coordinator.publisher is not None:
            self.coordinator.publisher.publish()


@dataclass(slots=True)
class AsyncGoalRunBinding:
    runs: AsyncRunStore
    scope: ExecutionScope
    claim: RunClaim
    coordinator: AsyncDurableEffectCoordinator

    @classmethod
    async def open(cls, runs: AsyncRunStore, scope: ExecutionScope, *, goal_id: str, worker_id: str,
                   lease_seconds: int) -> AsyncGoalRunBinding:
        await runs.submit(scope, submission(goal_id))
        await runs.reconcile_expired(scope=scope)
        record = await runs.get(scope, scope.run_id)
        require_runnable(record.status)
        claim = await runs.claim(scope, worker_id=worker_id, lease_seconds=lease_seconds, run_id=scope.run_id)
        if claim is None:
            raise RunLeaseError("Durable goal execution is owned by another worker or is terminal")
        if record.steps[0].status is not StepStatus.COMPLETED:
            await runs.start_step(scope, claim, "goal")
        return cls(runs, scope, claim, AsyncDurableEffectCoordinator(runs, scope=scope, claim=claim, step_id="goal"))

    async def assert_boundary(self) -> None:
        await self.runs.assert_claim(self.scope, self.claim)

    async def heartbeat(self, *, lease_seconds: int) -> None:
        self.claim = await self.runs.renew(self.scope, self.claim, lease_seconds=lease_seconds)
        self.coordinator.claim = self.claim

    async def install(self, runtime: Any) -> None:
        from chulk.runs.events import AsyncRunEventPublisher
        self.scope.assert_resumable(runtime.execution_scope)
        if runtime.events.public_event_sink is not None:
            self.coordinator.publisher = AsyncRunEventPublisher(self.runs, runtime.events.public_event_sink, scope=self.scope)
            await self.coordinator.publisher.publish()
        runtime._tool_executor.durable_effects = self.coordinator
        services = getattr(runtime, "resolved_services", None)
        if services is not None:
            from chulk.approvals.service import AsyncDurableApprovalCoordinator, AsyncDurableApprovalService
            from chulk.approvals.async_store import AsyncApprovalStoreAdapter
            approvals = services.approvals if inspect.iscoroutinefunction(services.approvals.create) else AsyncApprovalStoreAdapter(services.approvals)
            runtime._tool_executor.durable_approvals = AsyncDurableApprovalCoordinator(
                AsyncDurableApprovalService(approvals, self.runs), self.coordinator,
                scope=self.scope, claim=self.claim, step_id="goal")

    async def finish(self, reason: GoalStopReason, *, turn_id: str) -> None:
        current = await self.runs.get(self.scope, self.scope.run_id)
        if current.status is not RunStatus.RUNNING:
            return
        if reason is GoalStopReason.COMPLETED:
            if current.steps[0].status is not StepStatus.COMPLETED:
                await self.runs.complete_step(self.scope, self.claim, "goal", result={"turn_id": turn_id})
            await self.runs.complete(self.scope, self.claim, result={"turn_id": turn_id})
        elif reason is GoalStopReason.CANCELLED:
            await self.runs.cancel(self.scope, self.scope.run_id, actor=self.claim.worker_id,
                                   reason="goal cancelled", claim=self.claim)
        elif reason is not GoalStopReason.LEASE_LOST:
            await self.runs.yield_step(self.scope, self.claim, "goal", continuation={"turn_id": turn_id, "reason": reason.value})
        if self.coordinator.publisher is not None:
            await self.coordinator.publisher.publish()
