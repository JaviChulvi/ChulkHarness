"""Single mutation boundary for reduced action-loop effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from chulk.core.events import TraceEvent
from chulk.core.observations import (
    MAX_TOOL_ACTION_CONTEXT_CHARS,
    format_tool_action_context,
    format_tool_observation,
)
from chulk.core.plan_execution import PlanExecution
from chulk.core.state import AgentState, ObservationRecord, PlanStep, ToolCallRecord, TurnState
from chulk.core.transitions import (
    ActionLoopSnapshot,
    ActionTransition,
    ApplyPlanStepUpdateEffect,
    BlockTurnEffect,
    CompleteAnswerEffect,
    ExecuteToolEffect,
    FailTurnEffect,
    FinishToolEffect,
    PresentPlanEffect,
    ProceedEffect,
    RequestPlanExecutionFeedbackEffect,
    RequestPlanRevisionEffect,
    RequestReflectionEffect,
    RequestReflectionRevisionEffect,
    StartPlanStepEffect,
    TransitionOutcome,
)
from chulk.llm import LLMClient
from chulk.memory import ConversationMemory
from chulk.tools.output import preview_text
from chulk.tools.registry import ToolResult


@dataclass(frozen=True)
class PendingToolExecution:
    effect: ExecuteToolEffect
    record: ToolCallRecord
    plan_step: PlanStep | None


@dataclass(frozen=True)
class PendingReflection:
    proposed_answer: str


PendingOperation: TypeAlias = PendingToolExecution | PendingReflection


@dataclass(frozen=True)
class TransitionApplication:
    """A validated application of the reducer-selected outcome."""

    outcome: TransitionOutcome
    response: str | None = None
    pending: PendingOperation | None = None


@dataclass
class TurnEffects:
    """Apply transition effects and preserve all observable turn mutations."""

    state: AgentState
    memory: ConversationMemory
    llm_client: LLMClient
    plan: PlanExecution
    trace: Callable[[str, dict | None], None]
    redact_text: Callable[[str, str, dict], tuple[str, dict]]
    artifact_writer: Callable[[str, str], dict | None]
    planning_tool_names: Callable[[], frozenset[str]]
    max_tool_calls_per_turn: int
    max_reflection_attempts: int
    max_observation_chars: int
    max_tool_stdout_chars: int
    max_tool_stderr_chars: int

    def snapshot(self, turn: TurnState, *, require_plan: bool) -> ActionLoopSnapshot:
        """Capture the immutable control state consumed by the reducer."""
        plan = turn.active_plan
        active_step = plan.active_step() if plan is not None and turn.plan_approved else None
        next_step = (
            plan.next_ready_step()
            if plan is not None and turn.plan_approved and active_step is None
            else None
        )
        return ActionLoopSnapshot(
            require_plan=require_plan,
            planning_feedback_count=turn.planning_feedback_count,
            planning_tool_limit_feedback_sent=turn.planning_tool_limit_feedback_sent,
            plan_execution_feedback_count=turn.plan_execution_feedback_count,
            active_plan_approved=plan is not None and turn.plan_approved,
            approved_plan_incomplete=bool(
                plan is not None and turn.plan_approved and plan.status() != "completed"
            ),
            active_plan_step_id=active_step.id if active_step else None,
            planning_tool_names=self.planning_tool_names(),
            tool_call_count=turn.tool_call_count,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            reflection_count=turn.reflection_count,
            max_reflection_attempts=self.max_reflection_attempts,
            active_plan_status=plan.status() if plan is not None and turn.plan_approved else None,
            active_plan_step_title=active_step.title if active_step else None,
            next_ready_plan_step_id=next_step.id if next_step else None,
            active_plan_step_retry_count=active_step.retry_count if active_step else 0,
            active_plan_step_retry_limit=active_step.retry_limit if active_step else 0,
            active_plan_step_tool_failure_count=(
                active_step.tool_failure_count if active_step else 0
            ),
        )

    def apply(
        self,
        turn: TurnState,
        transition: ActionTransition,
        *,
        pending: PendingOperation | None = None,
        tool_result: ToolResult | None = None,
    ) -> TransitionApplication:
        """Apply one effect and consume the reducer-provided outcome verbatim."""
        effect = transition.effect
        response: str | None = None
        next_pending: PendingOperation | None = None

        if isinstance(effect, ProceedEffect):
            pass
        elif isinstance(effect, StartPlanStepEffect):
            self.plan.start_step(turn, effect.step_id)
        elif isinstance(effect, BlockTurnEffect):
            response = self.block_turn(effect.message, turn)
        elif isinstance(effect, FailTurnEffect):
            response = self.fail_turn(effect.message, turn)
        elif isinstance(effect, RequestPlanRevisionEffect):
            if effect.mark_tool_limit_feedback_sent:
                turn.planning_tool_limit_feedback_sent = True
            self.plan.request_revision(
                turn,
                plan=effect.plan,
                feedback=effect.feedback,
            )
        elif isinstance(effect, PresentPlanEffect):
            response = self.plan.present(turn, effect.plan)
        elif isinstance(effect, RequestPlanExecutionFeedbackEffect):
            self.plan.request_execution_feedback(turn, feedback=effect.feedback)
        elif isinstance(effect, ApplyPlanStepUpdateEffect):
            blocked_message = self.plan.apply_step_result(turn, effect.action)
            if blocked_message is not None:
                response = self.block_turn(blocked_message, turn)
        elif isinstance(effect, CompleteAnswerEffect):
            response = self.complete_answer(effect.content, turn)
        elif isinstance(effect, RequestReflectionEffect):
            next_pending = PendingReflection(proposed_answer=effect.proposed_answer)
        elif isinstance(effect, RequestReflectionRevisionEffect):
            self._request_reflection_revision(turn, effect)
        elif isinstance(effect, ExecuteToolEffect):
            next_pending = self._start_tool(turn, effect)
        elif isinstance(effect, FinishToolEffect):
            if not isinstance(pending, PendingToolExecution) or tool_result is None:
                raise RuntimeError("FinishToolEffect requires its pending tool and result")
            blocked_message = self._finish_tool(
                turn,
                effect,
                pending=pending,
                result=tool_result,
            )
            if blocked_message is not None:
                response = self.block_turn(blocked_message, turn)
        else:
            raise TypeError(f"Unsupported transition effect: {type(effect).__name__}")

        return _validate_application(
            outcome=transition.outcome,
            response=response,
            pending=next_pending,
        )

    def complete_answer(self, content: str, turn: TurnState) -> str:
        content, redaction = self.redact_text(
            TraceEvent.FINAL_ANSWER,
            content,
            {"turn_id": turn.turn_id, "field": "content"},
        )
        if redaction.get("redacted") or redaction.get("redaction_error"):
            turn.extension_metadata["final_answer_redaction"] = redaction
        self._emit_final_stream(content, turn)
        self.memory.add_assistant_message(content)
        self.state.final_answer = content
        self.state.messages = self.memory.recent()
        turn.complete(content)
        self.plan.clear(turn)
        self.trace(TraceEvent.FINAL_ANSWER, {"turn_id": turn.turn_id, "content": content})
        self.trace(TraceEvent.TURN_FINISHED, self.state_snapshot(turn))
        return content

    def fail_turn(self, message: str, turn: TurnState | None = None) -> str:
        self.state.errors.append(message)
        self.state.final_answer = message
        self.memory.add_assistant_message(message)
        self.state.messages = self.memory.recent()
        if turn is not None:
            turn.fail(message)
            self.plan.clear(turn)
        self.trace(
            TraceEvent.TURN_FAILED,
            {"turn_id": turn.turn_id if turn else None, "message": message},
        )
        if turn is not None:
            self.trace(TraceEvent.TURN_FINISHED, self.state_snapshot(turn))
        return message

    def block_turn(self, message: str, turn: TurnState) -> str:
        self.state.errors.append(message)
        self.state.final_answer = message
        self.memory.add_assistant_message(message)
        self.state.messages = self.memory.recent()
        turn.block(message)
        self.plan.clear(turn)
        self.trace(
            TraceEvent.TURN_FAILED,
            {"turn_id": turn.turn_id, "message": message, "status": "blocked"},
        )
        self.trace(TraceEvent.TURN_FINISHED, self.state_snapshot(turn))
        return message

    def state_snapshot(self, turn: TurnState) -> dict:
        return {
            "turn": turn.to_dict(),
            "agent_state": {
                "conversation_id": self.state.conversation_id,
                "current_turn_id": self.state.current_turn_id,
                "message_count": len(self.memory.messages),
                "turn_count": len(self.state.turns),
                "loaded_memory_ids": self.state.loaded_memory_ids,
                "loaded_skill_names": self.state.loaded_skill_names,
                "available_tool_names": self.state.available_tool_names,
                "error_count": len(self.state.errors),
                "final_answer": self.state.final_answer,
                "active_plan": (
                    self.state.active_plan.to_dict() if self.state.active_plan else None
                ),
                "pending_plan_turn_id": self.state.pending_plan_turn_id,
                "last_context_report": self.state.last_context_report,
                "last_usage_report": self.state.last_usage_report,
                "conversation_summary": self.state.conversation_summary,
            },
        }

    def _request_reflection_revision(
        self,
        turn: TurnState,
        effect: RequestReflectionRevisionEffect,
    ) -> None:
        feedback = (
            "Reflection feedback: revise the proposed final answer before showing it to the user.\n"
            f"Reason: {effect.reason}\n"
            f"Instruction: {effect.feedback}"
        )
        metadata = {
            "reflection_count": turn.reflection_count,
            "approved": False,
            "reason": effect.reason,
            "feedback": effect.feedback,
            "proposed_answer": effect.proposed_answer,
        }
        self.plan.add_observation(
            turn,
            tool_name="reflection_feedback",
            content=feedback,
            output_metadata=metadata,
        )
        self.trace(
            TraceEvent.REFLECTION_REVISION_REQUESTED,
            {"turn_id": turn.turn_id, **metadata},
        )

    def _start_tool(
        self,
        turn: TurnState,
        effect: ExecuteToolEffect,
    ) -> PendingToolExecution:
        turn.tool_call_count += 1
        plan = turn.active_plan
        step = (
            plan.active_step()
            if effect.phase == "execution" and plan is not None and turn.plan_approved
            else None
        )
        record = ToolCallRecord(
            tool_name=effect.action.tool_name,
            arguments=effect.action.arguments,
            iteration=turn.tool_call_count,
            phase=effect.phase,
            plan_step_id=step.id if step else None,
        )
        turn.tool_calls.append(record)
        self.trace(
            TraceEvent.TOOL_CALL_STARTED,
            {
                **record.to_dict(),
                "turn_id": turn.turn_id,
                "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
            },
        )
        return PendingToolExecution(effect=effect, record=record, plan_step=step)

    def _finish_tool(
        self,
        turn: TurnState,
        effect: FinishToolEffect,
        *,
        pending: PendingToolExecution,
        result: ToolResult,
    ) -> str | None:
        action = pending.effect.action
        record = pending.record
        record.finish(result)
        state_record: dict[str, object] = {
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            "phase": pending.effect.phase,
            "success": result.success,
        }
        if record.plan_step_id is not None:
            state_record["plan_step_id"] = record.plan_step_id
        self.state.tool_calls.append(state_record)
        self.trace(
            TraceEvent.TOOL_CALL,
            {
                "turn_id": turn.turn_id,
                "tool_name": action.tool_name,
                "arguments": action.arguments,
                "phase": pending.effect.phase,
                "plan_step_id": record.plan_step_id,
                "success": result.success,
                "error": result.error,
            },
        )
        self.trace(
            TraceEvent.TOOL_CALL_COMPLETED if result.success else TraceEvent.TOOL_CALL_FAILED,
            {
                **record.to_dict(),
                "turn_id": turn.turn_id,
                "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
            },
        )
        observation, metadata = self._format_observation(action.tool_name, result)
        tool_action_context, action_context_metadata = self._format_tool_action_context(
            pending,
        )
        metadata["tool_action_context"] = action_context_metadata
        self.state.observations.append(
            {
                "tool_name": action.tool_name,
                "observation": observation,
                "output_metadata": metadata,
            }
        )
        turn.observations.append(
            ObservationRecord(
                tool_name=action.tool_name,
                content=observation,
                output_metadata=metadata,
            )
        )
        if pending.plan_step is not None:
            self.plan.record_tool_evidence(
                pending.plan_step,
                record,
                observation,
                metadata,
                retry_metadata=effect.retry_metadata,
            )
        self.memory.add_assistant_message(tool_action_context)
        self.memory.add_observation(observation)
        self.trace(
            TraceEvent.TOOL_OBSERVATION,
            {
                "turn_id": turn.turn_id,
                "observation_index": len(turn.observations),
                "tool_name": action.tool_name,
                "tool_action_context": tool_action_context,
                "observation": observation,
                "output_metadata": metadata,
                "turn": turn.to_dict(),
            },
        )
        if pending.plan_step is None:
            if effect.disposition != "none":
                raise RuntimeError("Reducer selected plan evidence without a plan step")
            return None
        return self.plan.apply_tool_result(
            turn,
            effect,
            step=pending.plan_step,
            record=record,
            result=result,
            observation=observation,
            output_metadata=metadata,
            evidence_recorded=True,
        )

    def _format_observation(
        self,
        requested_tool_name: str,
        result: ToolResult,
    ) -> tuple[str, dict]:
        observation, metadata = format_tool_observation(
            requested_tool_name=requested_tool_name,
            result=result,
            max_observation_chars=self.max_observation_chars,
            max_stdout_chars=self.max_tool_stdout_chars,
            max_stderr_chars=self.max_tool_stderr_chars,
            artifact_writer=self.artifact_writer,
        )
        observation, redaction = self.redact_text(
            TraceEvent.TOOL_OBSERVATION,
            observation,
            {
                "requested_tool_name": requested_tool_name,
                "resolved_tool_name": result.tool_name,
            },
        )
        if redaction.get("redacted") or redaction.get("redaction_error"):
            metadata["redaction"] = redaction
        return observation, metadata

    def _format_tool_action_context(
        self,
        pending: PendingToolExecution,
    ) -> tuple[str, dict]:
        record = pending.record
        max_chars = min(
            MAX_TOOL_ACTION_CONTEXT_CHARS,
            self.max_observation_chars,
        )
        context, metadata = format_tool_action_context(
            tool_name=record.tool_name,
            arguments=record.arguments,
            phase=record.phase,
            iteration=record.iteration,
            plan_step_id=record.plan_step_id,
            max_chars=max_chars,
        )
        context, redaction = self.redact_text(
            TraceEvent.TOOL_OBSERVATION,
            context,
            {
                "requested_tool_name": record.tool_name,
                "field": "tool_action_context",
                "iteration": record.iteration,
                "plan_step_id": record.plan_step_id,
            },
        )
        final_preview = preview_text(context, max_chars)
        metadata["final_context"] = final_preview.to_metadata()
        if redaction.get("redacted") or redaction.get("redaction_error"):
            metadata["redaction"] = redaction
        return final_preview.text, metadata

    def _emit_final_stream(self, content: str, turn: TurnState) -> None:
        if not content or not self._streaming_enabled():
            return
        payload = {
            "turn_id": turn.turn_id,
            "source": "validated_final_answer",
            "content_length": len(content),
        }
        self.trace(TraceEvent.MODEL_STREAM_STARTED, payload)
        streamed_length = 0
        try:
            for index, text in enumerate(_stream_chunks(content), start=1):
                streamed_length += len(text)
                self.trace(
                    TraceEvent.MODEL_STREAM_DELTA,
                    {
                        "turn_id": turn.turn_id,
                        "source": "validated_final_answer",
                        "chunk_index": index,
                        "text": text,
                    },
                )
        except Exception as exc:
            self.trace(
                TraceEvent.MODEL_STREAM_FAILED,
                {
                    "turn_id": turn.turn_id,
                    "source": "validated_final_answer",
                    "error": str(exc),
                    "streamed_length": streamed_length,
                },
            )
            raise
        self.trace(
            TraceEvent.MODEL_STREAM_COMPLETED,
            {
                "turn_id": turn.turn_id,
                "source": "validated_final_answer",
                "streamed_length": streamed_length,
            },
        )

    def _streaming_enabled(self) -> bool:
        provider = (
            getattr(self.llm_client, "last_success_provider", None) or self.llm_client
        )
        capabilities = getattr(provider, "capabilities", None)
        return bool(getattr(capabilities, "supports_streaming", False))


def _validate_application(
    *,
    outcome: TransitionOutcome,
    response: str | None,
    pending: PendingOperation | None,
) -> TransitionApplication:
    if outcome == TransitionOutcome.STOP:
        if response is None or pending is not None:
            raise RuntimeError("STOP transition requires one response and no pending work")
    elif outcome == TransitionOutcome.AWAIT_RESULT:
        if pending is None or response is not None:
            raise RuntimeError("AWAIT_RESULT transition requires pending transport work")
    elif outcome in {TransitionOutcome.CONTINUE, TransitionOutcome.PROCEED}:
        if response is not None or pending is not None:
            raise RuntimeError(f"{outcome.value} transition cannot return work or a response")
    else:  # pragma: no cover - enum exhaustiveness
        raise RuntimeError(f"Unsupported transition outcome: {outcome}")
    return TransitionApplication(outcome=outcome, response=response, pending=pending)


def _stream_chunks(text: str, *, max_chars: int = 80):
    for start in range(0, len(text), max_chars):
        yield text[start : start + max_chars]
