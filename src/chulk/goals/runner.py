"""Foreground coordination around the existing goal-bound action loop."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
import json
import inspect
import threading
from typing import Any, cast
from uuid import uuid4

from chulk.core.plan_execution import PlanStepVerifier, AsyncPlanStepVerifier
from chulk.errors import ConfigurationError
from chulk.goals.models import (
    Goal, GoalExecutionResult, GoalSliceAdmission, GoalSliceLimits, GoalStatus, GoalStopReason,
)
from chulk.goals.runtime import GoalExecutionContext
from chulk.goals.store import GoalLeaseConflictError, GoalRevisionConflictError
from chulk.goals.protocols import GoalExecutionStore, AsyncGoalExecutionStore
from chulk.goals.transitions import start_goal, block_step, cancel_goal
from chulk.hosting.async_utils import call_async_service
from chulk.usage import BudgetExceededError, UsageGroupBy, UsageLedger


AgentFactory = Callable[[GoalExecutionContext, str | None], Any]


class GoalRunner:
    """Run bounded portions in the foreground; the host owns worker scheduling.

    The factory builds an Agent with the supplied execution context and optional
    conversation to reopen. Its tools, credentials and authorizer remain host-owned.
    """

    def __init__(self, store: GoalExecutionStore, *, agent_factory: AgentFactory,
                 verifier: PlanStepVerifier | None = None,
                 slice_limits: GoalSliceLimits = GoalSliceLimits(),
                 runner_id: str | None = None, lease_seconds: int = 120,
                 renewal_seconds: float = 40,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        if not 0 < renewal_seconds < lease_seconds:
            raise ConfigurationError("Goal renewal interval must be positive and shorter than the lease")
        self.store = store
        self.clock = clock
        self.agent_factory = agent_factory
        self.verifier = verifier
        self.slice_limits = slice_limits
        self.runner_id = runner_id or f"goal-runner:{uuid4().hex}"
        self.lease_seconds = lease_seconds
        self.renewal_seconds = renewal_seconds

    def run(self, goal_id: str) -> GoalExecutionResult:
        while True:
            result = self.run_slice(goal_id)
            if result.stop_reason not in {GoalStopReason.YIELDED, GoalStopReason.STEP_COMPLETED}:
                return result

    def run_slice(self, goal_id: str) -> GoalExecutionResult:
        if inspect.iscoroutinefunction(self.store.get):
            raise ConfigurationError("Native async goal stores require AsyncGoalRunner")
        goal = self.store.get(goal_id)
        previous = self.store.execution_state(goal_id)
        stopped = _stopped(goal, previous)
        if stopped is not None:
            return stopped
        self._validate(goal)
        if goal.status is GoalStatus.APPROVED:
            goal = self.store.mutate(goal.id, expected_revision=goal.revision, kind="goal.running",
                                     actor=self.runner_id, mutation=lambda current: start_goal(current, now=self.clock()))
        admission = self.store.admit_slice(goal.id, runner_id=self.runner_id,
                                          expected_revision=goal.revision, turn_id=str(uuid4()), lease_seconds=self.lease_seconds)
        if admission.claim is None:
            return _admission_stop(admission, previous)
        context = self._context(admission)
        agent = None
        try:
            with self._renew(context):
                agent = self.agent_factory(context, None if admission.new_conversation else admission.conversation_id)
                self._validate_agent(agent, context)
                _runtime(agent)._plan_execution.verifier = self.verifier
                error: BaseException | None = None
                try:
                    if admission.previous_turn_id is not None:
                        _runtime(agent).continue_goal_slice(turn_id=admission.turn_id)
                    else:
                        _runtime(agent).start_goal_step(turn_id=admission.turn_id)
                except (BudgetExceededError, GoalLeaseConflictError, GoalRevisionConflictError, ConfigurationError) as exc:
                    error = exc
                usage = _usage(_group_usage(agent, goal_id))
                return self._finish(admission, context, agent, usage, error)
        finally:
            if agent is not None:
                agent.close()
            context.close()

    def _validate(self, goal: Goal) -> None:
        if self.verifier is None:
            raise ConfigurationError("Automatic goal execution requires a configured host verifier")
        if goal.budget.max_model_calls is None or goal.budget.max_model_calls < 1:
            raise ConfigurationError("Automatic goals require a finite positive global model-call budget")

    def _context(self, admission: GoalSliceAdmission) -> GoalExecutionContext:
        assert admission.claim is not None and admission.step_id is not None
        return GoalExecutionContext(self.store, admission.claim, admission.step_id,
                                    slice_limits=self.slice_limits, automatic=True,
                                    conversation_id=admission.conversation_id)

    @staticmethod
    def _validate_agent(agent: Any, context: GoalExecutionContext) -> None:
        runtime = _runtime(agent)
        if runtime.goal_execution is not context or runtime.state.conversation_id != context.conversation_id:
            raise ConfigurationError("Goal factory must bind the supplied context and execution conversation")
        if not runtime._model_transport.durable_responses:
            raise ConfigurationError("Automatic goals require durable execution recording")
        accounting = runtime._model_accounting.usage_accounting or runtime._model_accounting.async_usage_accounting
        if accounting is None or not getattr(accounting, "enforces_goal_budgets", False):
            raise ConfigurationError("Automatic goals require durable accounting that declares enforces_goal_budgets")

    def _finish(self, admission: GoalSliceAdmission, context: GoalExecutionContext,
                agent: Any, usage: dict[str, Any], error: BaseException | None) -> GoalExecutionResult:
        goal = self.store.get(admission.goal.id)
        if context.ownership_lost:
            error = GoalLeaseConflictError("execution claim renewal failed")
        reason, detail = _outcome(goal, _runtime(agent).state.turns[-1], error)
        if reason is GoalStopReason.LEASE_LOST:
            return _result(goal, admission, reason, usage, detail)
        if goal.cancellation_requested:
            goal = self.store.mutate(goal.id, expected_revision=goal.revision, kind="goal.cancelled", actor=self.runner_id,
                                     mutation=lambda current: cancel_goal(current, now=self.clock()), claim=context.claim)
            self.store.finish_slice(context.claim, turn_id=admission.turn_id, reason=GoalStopReason.CANCELLED, usage=usage)
            return _result(goal, admission, GoalStopReason.CANCELLED, usage, detail)
        if goal.status is GoalStatus.RUNNING:
            verification = _runtime(agent).state.turns[-1].extension_metadata.get("goal_verification")
            if verification and verification["passed"] and error is None and _runtime(agent).state.turns[-1].status == "completed":
                try:
                    goal = self.store.apply_verified_step(context.claim, operation_id=verification["operation_id"], expected_revision=goal.revision)
                    reason = GoalStopReason.COMPLETED if goal.status is GoalStatus.COMPLETED else GoalStopReason.STEP_COMPLETED
                except GoalRevisionConflictError:
                    reason, detail = GoalStopReason.BLOCKED, "Goal criteria changed during verification; verification must be repeated"
            if reason in {GoalStopReason.BUDGET_EXHAUSTED, GoalStopReason.FAILED, GoalStopReason.BLOCKED, GoalStopReason.REQUIRED_CONTEXT_OVERFLOW} and goal.status is GoalStatus.RUNNING:
                goal = self.store.mutate(goal.id, expected_revision=goal.revision, kind="goal.blocked", actor=self.runner_id,
                    mutation=lambda current: block_step(current, context.step_id, detail or reason.value), claim=context.claim)
        self.store.finish_slice(context.claim, turn_id=admission.turn_id, reason=reason, usage=usage,
                                exhausted_budget=goal.budget.to_dict() if reason is GoalStopReason.BUDGET_EXHAUSTED else None)
        return _result(goal, admission, reason, usage, detail, _runtime(agent).state.turns[-1].final_answer or "")

    @contextmanager
    def _renew(self, context: GoalExecutionContext):
        stop = threading.Event()
        def renew() -> None:
            while not stop.wait(self.renewal_seconds):
                try:
                    context.heartbeat(lease_seconds=self.lease_seconds)
                except Exception:
                    context.ownership_lost = True
                    return
        thread = threading.Thread(target=renew, name="goal-claim-renewal")
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()


class AsyncGoalRunner(GoalRunner):
    """Native asynchronous coordinator; sync bindings use the draining adapter."""

    def __init__(self, store: GoalExecutionStore | AsyncGoalExecutionStore, *, async_verifier: AsyncPlanStepVerifier | None = None, **kwargs: Any) -> None:
        super().__init__(cast(GoalExecutionStore, store), **kwargs)
        self.async_verifier = async_verifier

    def _validate(self, goal: Goal) -> None:
        if self.verifier is None and self.async_verifier is None:
            raise ConfigurationError("Automatic goal execution requires a configured host verifier")
        if goal.budget.max_model_calls is None or goal.budget.max_model_calls < 1:
            raise ConfigurationError("Automatic goals require a finite positive global model-call budget")

    async def run(self, goal_id: str) -> GoalExecutionResult:  # type: ignore[override]
        while True:
            result = await self.run_slice(goal_id)
            if result.stop_reason not in {GoalStopReason.YIELDED, GoalStopReason.STEP_COMPLETED}:
                return result

    async def run_slice(self, goal_id: str) -> GoalExecutionResult:  # type: ignore[override]
        goal = await call_async_service(self.store, "get", goal_id)
        previous = await call_async_service(self.store, "execution_state", goal_id)
        stopped = _stopped(goal, previous)
        if stopped is not None:
            return stopped
        self._validate(goal)
        if goal.status is GoalStatus.APPROVED:
            goal = await call_async_service(self.store, "mutate", goal.id, expected_revision=goal.revision,
                kind="goal.running", actor=self.runner_id, mutation=lambda current: start_goal(current, now=self.clock()))
        admission = await call_async_service(self.store, "admit_slice", goal.id, runner_id=self.runner_id,
            expected_revision=goal.revision, turn_id=str(uuid4()), lease_seconds=self.lease_seconds)
        if admission.claim is None:
            return _admission_stop(admission, previous)
        context = self._context(admission)
        agent = None
        try:
            async with self._renew_async(context):
                agent = await call_async_service(self.agent_factory, "__call__", context,
                                                 None if admission.new_conversation else admission.conversation_id)
                self._validate_agent(agent, context)
                _runtime(agent)._plan_execution.verifier = self.verifier
                _runtime(agent)._plan_execution.async_verifier = self.async_verifier
                error: BaseException | None = None
                try:
                    if admission.previous_turn_id is not None:
                        await _runtime(agent).continue_goal_slice_async(turn_id=admission.turn_id)
                    else:
                        await _runtime(agent).start_goal_step_async(turn_id=admission.turn_id)
                except (BudgetExceededError, GoalLeaseConflictError, GoalRevisionConflictError, ConfigurationError) as exc:
                    error = exc
                usage = _usage(await call_async_service(_group_usage, "__call__", agent, goal_id))
                return await self._finish_async(admission, context, agent, usage, error)
        finally:
            if agent is not None:
                await call_async_service(agent, "close")
            await context.close_async()

    async def _finish_async(self, admission: GoalSliceAdmission, context: GoalExecutionContext,
                agent: Any, usage: dict[str, Any], error: BaseException | None) -> GoalExecutionResult:
        goal = await call_async_service(self.store, "get", admission.goal.id)
        if context.ownership_lost:
            error = GoalLeaseConflictError("execution claim renewal failed")
        reason, detail = _outcome(goal, _runtime(agent).state.turns[-1], error)
        if reason is GoalStopReason.LEASE_LOST:
            return _result(goal, admission, reason, usage, detail)
        if goal.cancellation_requested:
            goal = await call_async_service(self.store, "mutate", goal.id, expected_revision=goal.revision, kind="goal.cancelled", actor=self.runner_id,
                                     mutation=lambda current: cancel_goal(current, now=self.clock()), claim=context.claim)
            await call_async_service(self.store, "finish_slice", context.claim, turn_id=admission.turn_id, reason=GoalStopReason.CANCELLED, usage=usage)
            return _result(goal, admission, GoalStopReason.CANCELLED, usage, detail)
        if goal.status is GoalStatus.RUNNING:
            verification = _runtime(agent).state.turns[-1].extension_metadata.get("goal_verification")
            if verification and verification["passed"] and error is None and _runtime(agent).state.turns[-1].status == "completed":
                try:
                    goal = await call_async_service(self.store, "apply_verified_step", context.claim, operation_id=verification["operation_id"], expected_revision=goal.revision)
                    reason = GoalStopReason.COMPLETED if goal.status is GoalStatus.COMPLETED else GoalStopReason.STEP_COMPLETED
                except GoalRevisionConflictError:
                    reason, detail = GoalStopReason.BLOCKED, "Goal criteria changed during verification; verification must be repeated"
            if reason in {GoalStopReason.BUDGET_EXHAUSTED, GoalStopReason.FAILED, GoalStopReason.BLOCKED, GoalStopReason.REQUIRED_CONTEXT_OVERFLOW} and goal.status is GoalStatus.RUNNING:
                goal = await call_async_service(self.store, "mutate", goal.id, expected_revision=goal.revision, kind="goal.blocked", actor=self.runner_id,
                    mutation=lambda current: block_step(current, context.step_id, detail or reason.value), claim=context.claim)
        await call_async_service(self.store, "finish_slice", context.claim, turn_id=admission.turn_id, reason=reason, usage=usage,
                                exhausted_budget=goal.budget.to_dict() if reason is GoalStopReason.BUDGET_EXHAUSTED else None)
        return _result(goal, admission, reason, usage, detail, _runtime(agent).state.turns[-1].final_answer or "")

    @asynccontextmanager
    async def _renew_async(self, context: GoalExecutionContext):
        stop = asyncio.Event()
        async def renew() -> None:
            while True:
                try:
                    await asyncio.wait_for(stop.wait(), self.renewal_seconds)
                    return
                except TimeoutError:
                    pass
                try:
                    await context.heartbeat_async(lease_seconds=self.lease_seconds)
                except Exception:
                    context.ownership_lost = True
                    return
        task = asyncio.create_task(renew())
        try:
            yield
        finally:
            stop.set()
            await task


def _usage(groups: Any) -> dict[str, Any]:
    return groups[0].to_dict() if groups else {"model_calls": 0, "tool_calls": 0, "total_tokens": 0}


def _stopped(goal: Goal, state: dict[str, Any] | None) -> GoalExecutionResult | None:
    reason = {GoalStatus.PAUSED: GoalStopReason.PAUSED, GoalStatus.CANCELLED: GoalStopReason.CANCELLED,
              GoalStatus.BLOCKED: GoalStopReason.BLOCKED, GoalStatus.FAILED: GoalStopReason.FAILED,
              GoalStatus.COMPLETED: GoalStopReason.COMPLETED, GoalStatus.DRAFT: GoalStopReason.APPROVAL_REQUIRED}.get(goal.status)
    if goal.cancellation_requested:
        reason = GoalStopReason.CANCELLED
    if reason not in {GoalStopReason.COMPLETED, GoalStopReason.CANCELLED} and state and state["exhausted_budget_json"] and json.loads(state["exhausted_budget_json"]) == goal.budget.to_dict():
        reason = GoalStopReason.BUDGET_EXHAUSTED
    if reason is None:
        return None
    return GoalExecutionResult(goal, reason, conversation_id=state["conversation_id"] if state else None,
        turn_id=state["latest_turn_id"] if state else None, usage=json.loads(state["usage_json"]) if state else {}, detail=goal.last_error)


def _admission_stop(admission: GoalSliceAdmission, state: dict[str, Any] | None) -> GoalExecutionResult:
    return GoalExecutionResult(admission.goal, admission.stop_reason or GoalStopReason.BLOCKED,
        conversation_id=admission.conversation_id, turn_id=admission.previous_turn_id,
        usage=json.loads(state["usage_json"]) if state else {})


def _outcome(goal: Goal, turn: Any, error: BaseException | None) -> tuple[GoalStopReason, str | None]:
    if isinstance(error, BudgetExceededError):
        return GoalStopReason.BUDGET_EXHAUSTED, str(error)
    if goal.cancellation_requested or goal.status is GoalStatus.CANCELLED:
        return GoalStopReason.CANCELLED, "Goal cancellation requested"
    if goal.status is GoalStatus.PAUSED:
        return GoalStopReason.PAUSED, None
    if goal.status is GoalStatus.BLOCKED:
        return GoalStopReason.BLOCKED, goal.last_error
    if isinstance(error, GoalLeaseConflictError):
        return GoalStopReason.LEASE_LOST, str(error)
    if isinstance(error, ConfigurationError) and error.details.failure_kind == "context_budget_exceeded":
        return GoalStopReason.REQUIRED_CONTEXT_OVERFLOW, str(error)
    if error is not None:
        return GoalStopReason.BLOCKED, str(error)
    if turn.status == "yielded":
        return GoalStopReason.YIELDED, None
    if turn.status == "completed":
        return GoalStopReason.BLOCKED, "Turn ended without a persisted passing goal verification"
    return GoalStopReason.BLOCKED, turn.final_answer or turn.status


def _result(goal: Goal, admission: GoalSliceAdmission, reason: GoalStopReason,
            usage: dict[str, Any], detail: str | None, content: str = "") -> GoalExecutionResult:
    return GoalExecutionResult(goal, reason, conversation_id=admission.conversation_id,
        turn_id=admission.turn_id, continuation_id=admission.turn_id if reason is GoalStopReason.YIELDED else None,
        usage=usage, content=content, detail=detail)


def _runtime(agent: Any) -> Any:
    return getattr(agent, "runtime", agent)


def _group_usage(agent: Any, goal_id: str) -> Any:
    if hasattr(agent, "group_usage"):
        return agent.group_usage(UsageGroupBy.GOAL, goal_id=goal_id)
    runtime = _runtime(agent)
    accounting = runtime._model_accounting.usage_accounting
    return UsageLedger(accounting.store.db_path, profile_id=runtime.profile_id).group(UsageGroupBy.GOAL, goal_id=goal_id)
