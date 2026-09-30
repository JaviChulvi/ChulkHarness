"""Claim-bound execution context checked at every model and tool boundary."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from chulk.goals.models import Goal, GoalActionCheckpoint, GoalClaim, GoalModelRequest
from chulk.goals.store import GoalStore
from chulk.hosting.async_utils import call_async_service
from chulk.tools.registry import ToolResult


@dataclass(slots=True)
class GoalExecutionContext:
    """Bind one runner claim to one active goal step."""

    store: GoalStore
    claim: GoalClaim
    step_id: str

    @property
    def goal_id(self) -> str:
        return self.claim.goal_id

    @property
    def profile_id(self) -> str:
        return self.claim.profile_id

    def assert_boundary(self) -> Goal:
        """Observe pause, cancellation, steering, status, and lease changes."""
        return self.store.assert_action_boundary(
            self.claim,
            step_id=self.step_id,
        )

    async def assert_boundary_async(self) -> Goal:
        return await call_async_service(self, "assert_boundary")

    def context(self) -> dict[str, Any]:
        goal = self.assert_boundary()
        incorporated = self.store.incorporated_steering_ids(goal.id)
        step = goal.step(self.step_id)
        return {
            "goal_id": goal.id, "revision": goal.revision,
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
        return await call_async_service(self, "context")

    def begin_model_request(
        self, *, context: dict[str, Any], conversation_id: str, turn_id: str,
        request_index: int, purpose: str,
    ) -> GoalModelRequest:
        return self.store.begin_model_request(
            self.claim, step_id=self.step_id, conversation_id=conversation_id,
            turn_id=turn_id, request_index=request_index, purpose=purpose,
            goal_revision=context["revision"],
            steering_ids=tuple(item["id"] for item in context["instructions"]),
        )

    async def begin_model_request_async(self, **kwargs: Any) -> GoalModelRequest:
        return await call_async_service(self, "begin_model_request", **kwargs)

    def acknowledge_response(self, request_id: str, *, response_ref: str) -> GoalModelRequest:
        return self.store.acknowledge_model_response(self.claim, request_id, response_ref=response_ref)

    async def acknowledge_response_async(self, request_id: str, *, response_ref: str) -> GoalModelRequest:
        return await call_async_service(self, "acknowledge_response", request_id, response_ref=response_ref)

    def begin_tool(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        attempt: int,
        tool_name: str,
    ) -> GoalActionCheckpoint:
        return self.store.begin_action(
            self.claim,
            step_id=self.step_id,
            idempotency_key=(
                f"turn:{turn_id}:tool:{tool_call_index}:attempt:{attempt}"
            ),
            action_kind="tool",
            action_ref=tool_name,
        )

    def finish_tool(
        self,
        checkpoint: GoalActionCheckpoint,
        result: ToolResult,
    ) -> GoalActionCheckpoint:
        """Persist the observed outcome without copying raw tool output."""
        return self.store.finish_action(
            self.claim,
            checkpoint.id,
            result={
                "success": result.success,
                "tool_name": result.tool_name,
                "failure_kind": result.failure_kind,
                "exit_code": result.exit_code,
            },
            error=result.error if not result.success else None,
        )

    def abort_tool(
        self,
        checkpoint: GoalActionCheckpoint,
        error: BaseException,
    ) -> GoalActionCheckpoint:
        return self.store.finish_action(
            self.claim,
            checkpoint.id,
            error=f"{type(error).__name__}: {error}",
        )

    def heartbeat(self, *, lease_seconds: int = 120) -> GoalClaim:
        self.claim = self.store.heartbeat(
            self.claim,
            lease_seconds=lease_seconds,
        )
        return self.claim

    def close(self) -> bool:
        return self.store.release_claim(self.claim)


__all__ = ["GoalExecutionContext"]
