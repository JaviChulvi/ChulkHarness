"""Claim-bound execution context checked at every model and tool boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import inspect
import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import Any, TYPE_CHECKING
from chulk.hosting.scope import ExecutionScope

if TYPE_CHECKING:
    from chulk.goals.durable import GoalRunBinding, AsyncGoalRunBinding

from chulk.goals.models import Goal, GoalActionCheckpoint, GoalClaim, GoalModelRequest, GoalSliceLimits, verification_context_digest
from chulk.goals.protocols import GoalExecutionStore, AsyncGoalExecutionStore
from chulk.hosting.async_utils import call_async_service
from chulk.tools.registry import ToolResult
from chulk.usage import BudgetExceededError, BudgetScope, RunBudget
from chulk.errors import ConfigurationError
from chulk.core.state import TurnState, Plan, PlanStep


class GoalSliceExhausted(Exception):
    """Internal control signal; no operation was admitted in this phase."""

    def __init__(self, dimension: str) -> None:
        self.dimension = dimension
        super().__init__(f"Goal slice exhausted: {dimension}")


@dataclass(slots=True)
class GoalExecutionContext:
    """Bind one runner claim to one active goal step."""

    store: GoalExecutionStore | AsyncGoalExecutionStore
    claim: GoalClaim
    step_id: str
    slice_limits: GoalSliceLimits | None = None
    slice_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    automatic: bool = False
    conversation_id: str | None = None
    ownership_lost: bool = False
    recovering: bool = False
    new_conversation: bool = False
    execution_scope: ExecutionScope | None = None
    durable: GoalRunBinding | AsyncGoalRunBinding | None = None

    def _store_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        result = getattr(self.store, method)(*args, **kwargs)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise ConfigurationError("Native async goal stores require AsyncGoalRunner")
        return result

    def mark_started(self, turn_id: str) -> None:
        self._store_call("mark_slice_started", self.claim, turn_id)

    async def mark_started_async(self, turn_id: str) -> None:
        await call_async_service(self.store, "mark_slice_started", self.claim, turn_id)

    def slice_budget(self, turn: TurnState) -> RunBudget | None:
        if self.slice_limits is None:
            return None
        # Persist the deadline with the turn; reopening does not extend its slice.
        deadline = turn.extension_metadata.get("goal_slice_deadline")
        if deadline is None:
            deadline = (self.slice_clock().astimezone(timezone.utc) + timedelta(
                seconds=self.slice_limits.max_seconds
            )).isoformat()
            turn.extension_metadata["goal_slice_deadline"] = deadline
        return RunBudget(
            scope=BudgetScope.TURN,
            max_model_calls=self.slice_limits.max_model_calls,
            max_tool_calls=self.slice_limits.max_tool_calls,
            deadline=datetime.fromisoformat(deadline),
        )

    def on_budget_exhausted(self, error: BudgetExceededError) -> None:
        if self.slice_limits is None or error.scope is not BudgetScope.TURN:
            return
        if error.dimension != "deadline" and Decimal(error.requested) > Decimal(error.limit):
            raise ConfigurationError(
                f"Operation requires {error.requested} {error.dimension}; "
                f"a fresh goal slice allows {error.limit}"
            ) from error
        raise GoalSliceExhausted(error.dimension) from error

    @property
    def goal_id(self) -> str:
        return self.claim.goal_id

    @property
    def profile_id(self) -> str:
        return self.claim.profile_id

    def assert_boundary(self) -> Goal:
        """Observe pause, cancellation, steering, status, and lease changes."""
        if self.ownership_lost:
            from chulk.goals.store import GoalLeaseConflictError
            raise GoalLeaseConflictError("execution claim renewal failed")
        if self.durable is not None:
            result = self.durable.assert_boundary()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise ConfigurationError("Async durable binding requires async execution")
        return self._store_call("assert_action_boundary",
            self.claim,
            step_id=self.step_id,
        )

    async def assert_boundary_async(self) -> Goal:
        if self.ownership_lost:
            from chulk.goals.store import GoalLeaseConflictError
            raise GoalLeaseConflictError("execution claim renewal failed")
        if self.durable is not None:
            await call_async_service(self.durable, "assert_boundary")
        return await call_async_service(self.store, "assert_action_boundary", self.claim, step_id=self.step_id)

    def remaining_seconds(self, goal: Goal | None = None) -> float | None:
        goal = goal or self.assert_boundary()
        deadline = goal.budget.deadline
        if deadline is None:
            return None
        now = self.slice_clock().astimezone(timezone.utc)
        seconds = (deadline - now).total_seconds()
        if seconds <= 0:
            raise BudgetExceededError(scope=BudgetScope.GOAL, dimension="deadline", limit=deadline.isoformat(),
                                      committed=now.isoformat(), reserved="0", requested="0")
        return seconds

    async def remaining_seconds_async(self) -> float | None:
        return self.remaining_seconds(await self.assert_boundary_async())

    def call_with_deadline(self, callback: Callable, *args: Any, **kwargs: Any) -> Any:
        remaining = self.remaining_seconds() if self.automatic else None
        if remaining is None:
            return callback(*args, **kwargs)
        # A timed-out provider may still be billed: dispatch remains durably
        # uncertain and its reservation cannot be released or blindly replayed.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="goal-model-deadline")
        future = executor.submit(callback, *args, **kwargs)
        try:
            return future.result(timeout=remaining)
        except TimeoutError:
            if future.done():
                raise
            self.remaining_seconds()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    async def call_with_deadline_async(self, callback: Callable, *args: Any, **kwargs: Any) -> Any:
        remaining = await self.remaining_seconds_async() if self.automatic else None
        if remaining is None:
            return await callback(*args, **kwargs)
        try:
            return await asyncio.wait_for(callback(*args, **kwargs), timeout=remaining)
        except TimeoutError:
            await self.remaining_seconds_async()
            raise

    def context(self) -> dict[str, Any]:
        goal = self.assert_boundary()
        incorporated = self._store_call("incorporated_steering_ids", goal.id)
        return self._context_payload(goal, incorporated)

    def _context_payload(self, goal: Goal, incorporated: frozenset[str]) -> dict[str, Any]:
        step = goal.step(self.step_id)
        return {
            "goal_id": goal.id, "revision": goal.revision,
            **({"slice_limits": asdict(self.slice_limits)} if self.slice_limits is not None else {}),
            "description": goal.description, "constraints": list(goal.constraints),
            "acceptance_criteria": [item.to_dict() for item in goal.acceptance_criteria],
            "active_step": {
                "id": step.id, "description": step.description, "status": step.status.value,
                "criterion_ids": list(step.acceptance_criterion_ids), "attempt": step.attempt,
            },
            "progress": {
                "completed_step_count": sum(item.status.value in {"completed", "skipped"} for item in goal.steps),
                "step_count": len(goal.steps), "missing_criterion_ids": list(goal.missing_criterion_ids),
            },
            "instructions": [
                {"id": item.id, "instruction": item.instruction,
                 "incorporated": item.id in incorporated,
                 "fulfilled": item.id in goal.steering_fulfillments,
                 "fulfillment_evidence_ids": list(goal.steering_fulfillments.get(item.id, ()))}
                for item in goal.active_steering
            ],
            "evidence_refs": [
                {"id": item.id, "reference": item.reference,
                 "criterion_ids": list(item.criterion_ids)}
                for item in goal.evidence if item.step_id == step.id
            ][-6:],
            "evidence_count": len(goal.evidence),
        }

    async def context_async(self) -> dict[str, Any]:
        goal = await self.assert_boundary_async()
        incorporated = await call_async_service(self.store, "incorporated_steering_ids", goal.id)
        return self._context_payload(goal, incorporated)

    def project_plan(self, goal: Goal) -> Plan:
        """Project only the selected authoritative step into the existing turn loop."""
        step = goal.step(self.step_id)
        criteria = [item.description for item in goal.acceptance_criteria if item.id in step.acceptance_criterion_ids]
        plan = Plan(summary=goal.description or goal.title, steps=[PlanStep(id=step.id, title=step.title,
                    description=step.description, acceptance_criteria=criteria or [step.description], status="in_progress")])
        plan.approve()
        return plan

    def begin_model_request(
        self, *, context: dict[str, Any], conversation_id: str, turn_id: str,
        request_index: int, purpose: str,
    ) -> GoalModelRequest:
        return self._store_call("begin_model_request",
            self.claim, step_id=self.step_id, conversation_id=conversation_id,
            turn_id=turn_id, request_index=request_index, purpose=purpose,
            goal_revision=context["revision"],
            steering_ids=tuple(item["id"] for item in context["instructions"]),
        )

    async def begin_model_request_async(self, **kwargs: Any) -> GoalModelRequest:
        context = kwargs.pop("context")
        return await call_async_service(self.store, "begin_model_request", self.claim,
            step_id=self.step_id, goal_revision=context["revision"],
            steering_ids=tuple(item["id"] for item in context["instructions"]), **kwargs)

    def acknowledge_response(self, request_id: str, *, response_ref: str) -> GoalModelRequest:
        return self._store_call("acknowledge_model_response", self.claim, request_id, response_ref=response_ref)

    async def acknowledge_response_async(self, request_id: str, *, response_ref: str) -> GoalModelRequest:
        return await call_async_service(self.store, "acknowledge_model_response", self.claim, request_id, response_ref=response_ref)

    def begin_tool(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        attempt: int,
        tool_name: str,
        operation_id: str | None = None,
    ) -> GoalActionCheckpoint:
        return self._store_call("begin_action",
            self.claim,
            step_id=self.step_id,
            idempotency_key=(
                f"operation:{operation_id}:attempt:{attempt}" if operation_id else
                f"turn:{turn_id}:tool:{tool_call_index}:attempt:{attempt}"
            ),
            recover_existing=self.automatic and self.durable is not None,
            action_kind="tool",
            action_ref=tool_name,
        )

    async def begin_tool_async(self, *, turn_id: str, tool_call_index: int, attempt: int, tool_name: str, operation_id: str | None = None) -> GoalActionCheckpoint:
        return await call_async_service(self.store, "begin_action", self.claim, step_id=self.step_id,
            idempotency_key=f"operation:{operation_id}:attempt:{attempt}" if operation_id else f"turn:{turn_id}:tool:{tool_call_index}:attempt:{attempt}",
            action_kind="tool", action_ref=tool_name, recover_existing=self.automatic and self.durable is not None)

    def link_tool(self, checkpoint: GoalActionCheckpoint, effect_id: str) -> None:
        self._store_call("link_action_effect", self.claim, checkpoint.id, effect_id)

    async def link_tool_async(self, checkpoint: GoalActionCheckpoint, effect_id: str) -> None:
        await call_async_service(self.store, "link_action_effect", self.claim, checkpoint.id, effect_id)

    def finish_tool(
        self,
        checkpoint: GoalActionCheckpoint,
        result: ToolResult,
    ) -> GoalActionCheckpoint:
        """Persist the observed outcome without copying raw tool output."""
        return self._store_call("finish_action",
            self.claim,
            checkpoint.id,
            result=self._tool_result(result),
            error=result.error if not result.success else None,
        )

    @staticmethod
    def _tool_result(result: ToolResult) -> dict[str, Any]:
        return {"digest": sha256(result.to_observation().encode()).hexdigest(),
                "success": result.success, "tool_name": result.tool_name,
                "failure_kind": result.failure_kind, "exit_code": result.exit_code}

    async def finish_tool_async(self, checkpoint: GoalActionCheckpoint, result: ToolResult) -> GoalActionCheckpoint:
        return await call_async_service(self.store, "finish_action", self.claim, checkpoint.id,
            result=self._tool_result(result), error=result.error if not result.success else None)

    async def abort_tool_async(self, checkpoint: GoalActionCheckpoint, error: BaseException) -> GoalActionCheckpoint:
        return await call_async_service(self.store, "finish_action", self.claim, checkpoint.id,
            error=f"{type(error).__name__}: {error}")

    def abort_tool(
        self,
        checkpoint: GoalActionCheckpoint,
        error: BaseException,
    ) -> GoalActionCheckpoint:
        return self._store_call("finish_action",
            self.claim,
            checkpoint.id,
            error=f"{type(error).__name__}: {error}",
        )

    def verification_context(self) -> dict[str, Any]:
        goal = self.assert_boundary()
        checkpoints = self._store_call("action_checkpoints", goal.id)
        return self._verification_context(goal, checkpoints)

    async def verification_context_async(self) -> dict[str, Any]:
        goal = await self.assert_boundary_async()
        checkpoints = await call_async_service(self.store, "action_checkpoints", goal.id)
        return self._verification_context(goal, checkpoints)

    def _verification_context(self, goal: Goal, checkpoints: tuple[GoalActionCheckpoint, ...]) -> dict[str, Any]:
        evidence = sorted({
            str(item.result["digest"]) for item in checkpoints
            if item.step_id == self.step_id and item.result and "digest" in item.result
        } | {item.summary for item in goal.evidence if item.step_id == self.step_id})
        return {"expected_revision": goal.revision,
                "context_digest": verification_context_digest(goal),
                "evidence_digest": sha256(json.dumps(evidence).encode()).hexdigest()}

    @staticmethod
    def verification_operation_id(turn: TurnState) -> str:
        pending = turn.extension_metadata.get("goal_pending", {})
        return f"verification:{pending['operation_id']}" if pending.get("operation_id") else f"{turn.turn_id}:model:{turn.model_request_count}"

    def recorded_verification(self, turn: TurnState, context: dict[str, Any]) -> dict[str, Any] | None:
        recorded = self._store_call("verification", self.claim, self.verification_operation_id(turn))
        return self._current_verification(recorded, context)

    async def recorded_verification_async(self, turn: TurnState, context: dict[str, Any]) -> dict[str, Any] | None:
        recorded = await call_async_service(self.store, "verification", self.claim, self.verification_operation_id(turn))
        return self._current_verification(recorded, context)

    @staticmethod
    def _current_verification(recorded: dict[str, Any] | None, context: dict[str, Any]) -> dict[str, Any] | None:
        if recorded is not None and any(recorded[key] != context[key] for key in ("context_digest", "evidence_digest")):
            # The old operation cannot authorize changed work. The next request
            # obtains current criteria instead of silently reusing its decision.
            raise ConfigurationError("Verification context changed; request a fresh completion proposal")
        return recorded

    def record_verification(self, turn: TurnState, *, context: dict[str, Any], passed: bool, feedback: str) -> int:
        operation_id = self.verification_operation_id(turn)
        count = self._store_call("record_verification",
            self.claim, step_id=self.step_id, operation_id=operation_id,
            passed=passed, feedback=feedback, **context,
        )
        turn.extension_metadata["goal_verification"] = {"operation_id": operation_id,
            "passed": passed, "goal_revision": context["expected_revision"], "feedback": feedback}
        return count

    async def record_verification_async(self, turn: TurnState, *, context: dict[str, Any], passed: bool, feedback: str) -> int:
        operation_id = self.verification_operation_id(turn)
        count = await call_async_service(self.store, "record_verification", self.claim,
            step_id=self.step_id, operation_id=operation_id, passed=passed, feedback=feedback, **context)
        turn.extension_metadata["goal_verification"] = {"operation_id": operation_id,
            "passed": passed, "goal_revision": context["expected_revision"], "feedback": feedback}
        return count

    async def heartbeat_async(self, *, lease_seconds: int = 120) -> GoalClaim:
        self.claim = await call_async_service(self.store, "heartbeat", self.claim, lease_seconds=lease_seconds)
        if self.durable is not None:
            await call_async_service(self.durable, "heartbeat", lease_seconds=lease_seconds)
        return self.claim

    async def close_async(self) -> bool:
        return await call_async_service(self.store, "release_claim", self.claim)

    def heartbeat(self, *, lease_seconds: int = 120) -> GoalClaim:
        self.claim = self._store_call("heartbeat",
            self.claim,
            lease_seconds=lease_seconds,
        )
        if self.durable is not None:
            result = self.durable.heartbeat(lease_seconds=lease_seconds)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise ConfigurationError("Async durable binding requires async heartbeat")
        return self.claim

    def close(self) -> bool:
        return self._store_call("release_claim", self.claim)


__all__ = ["GoalExecutionContext"]
