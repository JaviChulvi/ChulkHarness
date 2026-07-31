"""Claim-bound execution context checked at every model and tool boundary."""

from __future__ import annotations

from dataclasses import dataclass

from chulk.goals.models import Goal, GoalActionCheckpoint, GoalClaim
from chulk.goals.store import GoalStore
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
