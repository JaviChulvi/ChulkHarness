"""Plan-state mutations selected by the pure transition reducer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chulk.core.actions import PlanStepUpdateAction
from chulk.core.events import TraceEvent
from chulk.core.state import AgentState, ObservationRecord, Plan, PlanStep, ToolCallRecord, TurnState
from chulk.core.transitions import FinishToolEffect
from chulk.memory import ConversationMemory
from chulk.tools.registry import ToolResult


@dataclass
class PlanExecution:
    """Apply plan effects without deciding plan continuation policy."""

    state: AgentState
    memory: ConversationMemory
    trace: Callable[[str, dict | None], None]

    def present(self, turn: TurnState, plan: Plan) -> str:
        turn.wait_for_plan_approval(plan)
        self.state.active_plan = plan
        self.state.pending_plan_turn_id = turn.turn_id
        response = plan.to_user_text() + "\n\nUse /approve to execute this plan or /reject to cancel it."
        self.memory.add_assistant_message(response)
        self.state.messages = self.memory.recent()
        self.trace(
            TraceEvent.PLAN_CREATED,
            {"turn_id": turn.turn_id, "plan": plan.to_dict(), "turn": turn.to_dict()},
        )
        return response

    def request_revision(
        self,
        turn: TurnState,
        *,
        plan: Plan | None,
        feedback: str | None,
    ) -> None:
        turn.planning_feedback_count += 1
        if feedback is None:
            feedback = (
                "Planning feedback: the proposed plan is still mostly reconnaissance. "
                "Do not present read/list/search/explore/inspect steps as the approval plan. "
                "If more context is needed, call read_file or search_files now. "
                "Otherwise return a concrete implementation plan naming the modules/files to change, "
                "the behavior to add, and the tests to update."
            )
        metadata: dict[str, object] = {"revision_count": turn.planning_feedback_count}
        if plan is not None:
            metadata["rejected_plan"] = plan.to_dict()
        self.add_observation(
            turn,
            tool_name="planning_feedback",
            content=feedback,
            output_metadata=metadata,
        )
        self.trace(
            TraceEvent.PLAN_REVISION_REQUESTED,
            {
                "turn_id": turn.turn_id,
                "revision_count": turn.planning_feedback_count,
                "plan": plan.to_dict() if plan is not None else None,
                "feedback": feedback,
            },
        )

    def request_execution_feedback(self, turn: TurnState, *, feedback: str) -> None:
        turn.plan_execution_feedback_count += 1
        self.add_observation(
            turn,
            tool_name="plan_execution_feedback",
            content=feedback,
            output_metadata={"feedback_count": turn.plan_execution_feedback_count},
        )

    def start_step(self, turn: TurnState, step_id: str) -> None:
        plan = turn.active_plan
        if plan is None or not turn.plan_approved:
            raise RuntimeError("Reducer selected a plan step without an approved plan")
        step = plan.next_ready_step()
        if step is None or step.id != step_id:
            raise RuntimeError("Reducer-selected plan step is no longer ready")
        step.mark("in_progress")
        self.trace(
            TraceEvent.PLAN_STEP_STARTED,
            {"turn_id": turn.turn_id, "step": step.to_dict(), "plan": plan.to_dict()},
        )

    def apply_step_result(
        self,
        turn: TurnState,
        action: PlanStepUpdateAction,
    ) -> str | None:
        plan = turn.active_plan
        if plan is None or not turn.plan_approved:
            raise RuntimeError("Validated plan step update lost its approved plan")
        step = plan.active_step()
        if step is None or action.step_id != step.id:
            raise RuntimeError("Validated plan step update lost its active step")
        step.add_evidence(action.evidence, tool_name="plan_step_update")
        if action.status == "completed":
            step.mark("completed")
            self._trace_step(turn, step, TraceEvent.PLAN_STEP_COMPLETED)
            return None
        step.block(action.reason or action.evidence)
        self._trace_step(turn, step, TraceEvent.PLAN_STEP_BLOCKED)
        return _blocked_message(step)

    def apply_tool_result(
        self,
        turn: TurnState,
        effect: FinishToolEffect,
        *,
        step: PlanStep,
        record: ToolCallRecord,
        result: ToolResult,
        observation: str,
        output_metadata: dict,
    ) -> str | None:
        retry_metadata = effect.retry_metadata
        self._record_tool_evidence(
            step,
            record,
            observation,
            output_metadata,
            retry_metadata=retry_metadata,
        )
        if effect.disposition in {"none", "evidence"}:
            return None
        if effect.disposition == "retry_scheduled":
            if retry_metadata is None:
                raise RuntimeError("Retry disposition requires retry metadata")
            retries_used = retry_metadata["retries_used"]
            if not isinstance(retries_used, int):
                raise RuntimeError("Retry metadata has an invalid retries_used value")
            self.add_observation(
                turn,
                tool_name="plan_step_retry",
                content=(
                    f"Plan step {step.id} remains in_progress after {_tool_failure_reason(result)} "
                    f"Recovery attempt {retries_used} of {step.retry_limit} is available; "
                    f"{step.retries_remaining} retries remain. Correct the arguments or choose a safe "
                    "alternative that can satisfy the current step's acceptance criteria."
                ),
                output_metadata={
                    "step_id": step.id,
                    "step_status": step.status,
                    **retry_metadata,
                },
            )
            return None
        if effect.blocked_reason is None:
            raise RuntimeError("Blocked plan tool result requires a reason")
        step.block(effect.blocked_reason)
        self._trace_step(
            turn,
            step,
            TraceEvent.PLAN_STEP_BLOCKED,
            tool_name=result.tool_name,
            error=result.error,
        )
        return _blocked_message(step)

    def add_observation(
        self,
        turn: TurnState,
        *,
        tool_name: str,
        content: str,
        output_metadata: dict,
    ) -> None:
        metadata = {**output_metadata, "synthetic": True}
        turn.observations.append(
            ObservationRecord(
                tool_name=tool_name,
                content=content,
                output_metadata=metadata,
            )
        )
        self.state.observations.append(
            {
                "tool_name": tool_name,
                "observation": content,
                "output_metadata": metadata,
            }
        )
        self.memory.add_observation(content)
        self.trace(
            TraceEvent.TOOL_OBSERVATION,
            {
                "turn_id": turn.turn_id,
                "tool_name": tool_name,
                "observation": content,
                "output_metadata": metadata,
            },
        )

    def clear(self, turn: TurnState) -> None:
        if self.state.pending_plan_turn_id == turn.turn_id:
            self.state.pending_plan_turn_id = None
        if turn.active_plan is not None and self.state.active_plan is turn.active_plan:
            self.state.active_plan = None

    def _record_tool_evidence(
        self,
        step: PlanStep,
        record: ToolCallRecord,
        observation: str,
        output_metadata: dict,
        *,
        retry_metadata: dict[str, object] | None,
    ) -> None:
        metadata = {
            "phase": record.phase,
            "output_metadata": output_metadata,
            "tool_call": record.to_dict(),
        }
        if retry_metadata is not None:
            metadata["plan_step_retry"] = retry_metadata
        step.add_evidence(
            observation,
            tool_name=record.tool_name,
            tool_call_iteration=record.iteration,
            metadata=metadata,
        )

    def _trace_step(
        self,
        turn: TurnState,
        step: PlanStep,
        event_type: str,
        *,
        tool_name: str | None = None,
        error: str | None = None,
    ) -> None:
        self.trace(
            event_type,
            {
                "turn_id": turn.turn_id,
                "step": step.to_dict(),
                "plan": turn.active_plan.to_dict() if turn.active_plan else None,
                "tool_name": tool_name,
                "error": error,
            },
        )


def _tool_failure_reason(result: ToolResult) -> str:
    if result.error:
        return f"Tool {result.tool_name} failed with {result.error}."
    if result.exit_code is not None:
        return f"Tool {result.tool_name} failed with exit code {result.exit_code}."
    return f"Tool {result.tool_name} failed."


def _blocked_message(step: PlanStep) -> str:
    reason = step.blocked_reason or "Step blocked."
    return f"Plan step blocked: {step.title}. {reason}"
