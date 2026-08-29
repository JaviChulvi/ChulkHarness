"""Plan-state mutations selected by the pure transition reducer."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from chulk.core.actions import PlanStepUpdateAction
from chulk.core.events import TraceEvent
from chulk.core.state import AgentState, ObservationRecord, Plan, PlanStep, ToolCallRecord, TurnState
from chulk.core.transitions import FinishToolEffect
from chulk.memory import ConversationMemory
from chulk.tools.registry import ToolResult


@dataclass(frozen=True)
class PlanStepVerificationRequest:
    """Detached plan-step facts supplied to a host verifier."""

    turn_id: str
    plan_summary: str
    step_id: str
    title: str
    description: str
    acceptance_criteria: tuple[str, ...]
    asserted_evidence: str
    recorded_evidence: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "plan_summary": self.plan_summary,
            "step_id": self.step_id,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": list(self.acceptance_criteria),
            "asserted_evidence": self.asserted_evidence,
            "recorded_evidence": list(self.recorded_evidence),
        }


@dataclass(frozen=True)
class PlanStepVerification:
    """Authoritative host decision for one model completion assertion."""

    passed: bool
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("plan step verification passed must be a boolean")
        clean_evidence = self.evidence.strip()
        if not clean_evidence:
            raise ValueError("plan step verification evidence cannot be empty")
        object.__setattr__(self, "evidence", clean_evidence)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "evidence": self.evidence}


PlanStepVerifier = Callable[[PlanStepVerificationRequest], PlanStepVerification]
AsyncPlanStepVerifier = Callable[
    [PlanStepVerificationRequest],
    Awaitable[PlanStepVerification],
]


@dataclass
class PlanExecution:
    """Apply plan effects without deciding plan continuation policy."""

    state: AgentState
    memory: ConversationMemory
    trace: Callable[[str, dict | None], None]
    verifier: PlanStepVerifier | None = None
    async_verifier: AsyncPlanStepVerifier | None = None

    def present(self, turn: TurnState, plan: Plan) -> str:
        turn.wait_for_plan_approval(plan)
        self.state.active_plan = plan
        self.state.pending_plan_turn_id = turn.turn_id
        self.state.messages = self.memory.recent()
        response = plan.to_user_text() + "\n\nUse /approve to execute this plan or /reject to cancel it."
        self.trace(
            TraceEvent.PLAN_CREATED,
            {
                "turn_id": turn.turn_id,
                "plan": plan.to_dict(),
                "display_message": response,
                "turn": turn.to_dict(),
            },
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
                "Do not present discovery or research as the approval plan. "
                "If more context is needed, use an available read-only reconnaissance action now. "
                "Otherwise return a concrete implementation plan naming the relevant components or "
                "resources, the behavior to change, and how the result will be verified."
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
            {
                "turn_id": turn.turn_id,
                "step": step.to_dict(),
                "plan": plan.to_dict(),
                "turn": turn.to_dict(),
            },
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
        verification: PlanStepVerification | None = None
        if action.status == "completed":
            if self.verifier is not None:
                verification = _require_verification(
                    self.verifier(_verification_request(turn, plan, step, action))
                )
            elif self.async_verifier is not None:
                raise RuntimeError(
                    "async plan step verifier requires asynchronous plan execution"
                )
        return self._apply_step_result(turn, step, action, verification)

    async def apply_step_result_async(
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
        verification: PlanStepVerification | None = None
        if action.status == "completed":
            request = _verification_request(turn, plan, step, action)
            if self.async_verifier is not None:
                verification = _require_verification(await self.async_verifier(request))
            elif self.verifier is not None:
                verification = _require_verification(
                    await asyncio.to_thread(self.verifier, request)
                )
        return self._apply_step_result(turn, step, action, verification)

    def _apply_step_result(
        self,
        turn: TurnState,
        step: PlanStep,
        action: PlanStepUpdateAction,
        verification: PlanStepVerification | None,
    ) -> str | None:
        if action.status == "completed" and verification is not None:
            if not verification.passed:
                self.add_observation(
                    turn,
                    tool_name="plan_step_verification",
                    content=(
                        f"Plan step verification rejected completion for {step.id}. "
                        f"{verification.evidence} Continue working against the acceptance "
                        "criteria before asserting completion again."
                    ),
                    output_metadata={
                        "step_id": step.id,
                        "asserted_evidence": action.evidence,
                        "verification": verification.to_dict(),
                    },
                )
                return None
        step.add_evidence(action.evidence, tool_name="plan_step_update")
        if action.status == "completed":
            if verification is not None:
                step.add_evidence(
                    verification.evidence,
                    tool_name="plan_step_verifier",
                    metadata={"external_verification": True},
                )
            step.mark("completed")
            self._trace_step(turn, step, TraceEvent.PLAN_STEP_COMPLETED)
            return None
        step.block(action.reason or action.evidence)
        message = _blocked_message(step)
        turn.block(message)
        self._trace_step(turn, step, TraceEvent.PLAN_STEP_BLOCKED)
        return message

    def prepare_tool_result_checkpoint(
        self,
        effect: FinishToolEffect,
        *,
        step: PlanStep,
    ) -> None:
        """Apply terminal tool disposition before the observation checkpoint."""
        if effect.disposition != "block":
            return
        if effect.blocked_reason is None:
            raise RuntimeError("Blocked plan tool result requires a reason")
        step.block(effect.blocked_reason)

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
        evidence_recorded: bool = False,
    ) -> str | None:
        retry_metadata = effect.retry_metadata
        if not evidence_recorded:
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
        if step.status != "blocked" or step.blocked_reason != effect.blocked_reason:
            step.block(effect.blocked_reason)
        message = _blocked_message(step)
        if turn.status != "blocked" or turn.final_answer != message:
            turn.block(message)
        self._trace_step(
            turn,
            step,
            TraceEvent.PLAN_STEP_BLOCKED,
            tool_name=result.tool_name,
            error=result.error,
        )
        return message

    def record_tool_evidence(
        self,
        step: PlanStep,
        record: ToolCallRecord,
        observation: str,
        output_metadata: dict,
        *,
        retry_metadata: dict[str, object] | None,
    ) -> None:
        """Attach tool evidence before the observation checkpoint is emitted."""
        self._record_tool_evidence(
            step,
            record,
            observation,
            output_metadata,
            retry_metadata=retry_metadata,
        )

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
                "observation_index": len(turn.observations),
                "tool_name": tool_name,
                "observation": content,
                "output_metadata": metadata,
                "turn": turn.to_dict(),
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
                "turn": turn.to_dict(),
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


def _verification_request(
    turn: TurnState,
    plan: Plan,
    step: PlanStep,
    action: PlanStepUpdateAction,
) -> PlanStepVerificationRequest:
    return PlanStepVerificationRequest(
        turn_id=turn.turn_id,
        plan_summary=plan.summary,
        step_id=step.id,
        title=step.title,
        description=step.description,
        acceptance_criteria=tuple(step.acceptance_criteria),
        asserted_evidence=action.evidence,
        recorded_evidence=tuple(record.content for record in step.evidence),
    )


def _require_verification(value: object) -> PlanStepVerification:
    if not isinstance(value, PlanStepVerification):
        raise TypeError("plan step verifier must return PlanStepVerification")
    return value
