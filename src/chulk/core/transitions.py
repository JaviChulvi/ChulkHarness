"""Pure action-loop transitions.

The reducer in this module decides what an agent action means for the current
turn.  It deliberately knows nothing about :class:`Agent`, providers, tools,
memory, or tracing.  Transport drivers apply the returned effect and feed the
next model action into the reducer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, TypeAlias

from chulk.core.actions import (
    AgentAction,
    FinalAnswerAction,
    PlanAction,
    PlanStepUpdateAction,
    ToolCallAction,
)
from chulk.core.planning import format_read_only_planning_tools, plan_looks_like_reconnaissance
from chulk.core.state import Plan


PLANNING_FINAL_ANSWER_FEEDBACK = (
    "Planning feedback: the user explicitly requested /plan, so do not answer directly. "
    "Use read-only reconnaissance tools if codebase context is needed, then return a plan action "
    "with concrete implementation steps that can be approved or rejected."
)
PLAN_EXECUTION_FINAL_ANSWER_FEEDBACK = (
    "Plan execution feedback: the approved plan is not complete. "
    "Continue the current executable step with a tool call, or return a plan_step_update "
    "if the step's acceptance criteria are already satisfied. Do not return final_answer yet."
)
PLANNING_TOOL_LIMIT_FEEDBACK = (
    "Planning feedback: the read-only reconnaissance tool budget is exhausted. "
    "Do not call more tools. Return a plan action now using the context already gathered. "
    "The plan must name concrete files/modules to change, behaviors to add, and tests to update."
)


class TransitionOutcome(str, Enum):
    """Expected control flow after a transition effect is applied."""

    PROCEED = "proceed"
    CONTINUE = "continue"
    STOP = "stop"
    AWAIT_RESULT = "await_result"


@dataclass(frozen=True)
class ActionLoopSnapshot:
    """The minimal immutable turn state needed to reduce one model action."""

    require_plan: bool
    planning_feedback_count: int = 0
    planning_tool_limit_feedback_sent: bool = False
    plan_execution_feedback_count: int = 0
    active_plan_approved: bool = False
    approved_plan_incomplete: bool = False
    active_plan_step_id: str | None = None
    planning_tool_names: frozenset[str] = field(default_factory=frozenset)
    tool_call_count: int = 0
    max_tool_calls_per_turn: int = 5
    reflection_count: int = 0
    max_reflection_attempts: int = 0
    active_plan_status: str | None = None
    active_plan_step_title: str | None = None
    next_ready_plan_step_id: str | None = None
    active_plan_step_retry_count: int = 0
    active_plan_step_retry_limit: int = 0
    active_plan_step_tool_failure_count: int = 0


@dataclass(frozen=True)
class ModelActionSignal:
    """A validated model action ready for policy reduction."""

    action: AgentAction


@dataclass(frozen=True)
class PrepareIterationSignal:
    """The loop is ready to prepare the next executable plan step."""


@dataclass(frozen=True)
class ProtocolFailureSignal:
    """The provider exhausted structured-action repair attempts."""

    message: str


@dataclass(frozen=True)
class ToolResultSignal:
    """Primitive tool-result facts used for continuation policy."""

    tool_name: str
    phase: Literal["planning", "execution"]
    success: bool
    has_plan_step: bool
    error: str | None = None
    failure_kind: str | None = None
    exit_code: int | None = None


@dataclass(frozen=True)
class ReflectionResultSignal:
    """One completed reflection review."""

    proposed_answer: str
    approved: bool
    reason: str
    feedback: str | None = None


@dataclass(frozen=True)
class PlanStepResultSignal:
    """A validated model assertion about the active plan step."""

    action: PlanStepUpdateAction


TransitionSignal: TypeAlias = (
    ModelActionSignal
    | PrepareIterationSignal
    | ProtocolFailureSignal
    | ToolResultSignal
    | ReflectionResultSignal
    | PlanStepResultSignal
)


@dataclass(frozen=True)
class ProceedEffect:
    """Proceed to the model request without another loop iteration."""


@dataclass(frozen=True)
class StartPlanStepEffect:
    step_id: str


@dataclass(frozen=True)
class BlockTurnEffect:
    message: str


@dataclass(frozen=True)
class FailTurnEffect:
    message: str


@dataclass(frozen=True)
class RequestPlanRevisionEffect:
    feedback: str | None = None
    plan: Plan | None = None
    mark_tool_limit_feedback_sent: bool = False


@dataclass(frozen=True)
class PresentPlanEffect:
    plan: Plan


@dataclass(frozen=True)
class RequestPlanExecutionFeedbackEffect:
    feedback: str


@dataclass(frozen=True)
class ApplyPlanStepUpdateEffect:
    action: PlanStepUpdateAction


@dataclass(frozen=True)
class CompleteAnswerEffect:
    content: str


@dataclass(frozen=True)
class RequestReflectionEffect:
    proposed_answer: str


@dataclass(frozen=True)
class RequestReflectionRevisionEffect:
    proposed_answer: str
    reason: str
    feedback: str | None = None


@dataclass(frozen=True)
class ExecuteToolEffect:
    action: ToolCallAction
    phase: Literal["planning", "execution"]


@dataclass(frozen=True)
class FinishToolEffect:
    """Record a tool result using the reducer-selected plan disposition."""

    disposition: Literal["none", "evidence", "retry_scheduled", "block", "fatal_safety"]
    retry_metadata: dict[str, object] | None = None
    blocked_reason: str | None = None


TransitionEffect: TypeAlias = (
    ProceedEffect
    | StartPlanStepEffect
    | BlockTurnEffect
    | FailTurnEffect
    | RequestPlanRevisionEffect
    | PresentPlanEffect
    | RequestPlanExecutionFeedbackEffect
    | ApplyPlanStepUpdateEffect
    | CompleteAnswerEffect
    | RequestReflectionEffect
    | RequestReflectionRevisionEffect
    | ExecuteToolEffect
    | FinishToolEffect
)


@dataclass(frozen=True)
class ActionTransition:
    """One reducer decision and its expected loop outcome."""

    effect: TransitionEffect
    outcome: TransitionOutcome


def reduce_transition(
    snapshot: ActionLoopSnapshot,
    signal: TransitionSignal,
) -> ActionTransition:
    """Reduce one loop signal without mutating state or performing I/O."""
    if isinstance(signal, PrepareIterationSignal):
        return _reduce_iteration_preparation(snapshot)
    if isinstance(signal, ProtocolFailureSignal):
        return _stop_with_failure(signal.message)
    if isinstance(signal, ToolResultSignal):
        return _reduce_tool_result(snapshot, signal)
    if isinstance(signal, ReflectionResultSignal):
        return _reduce_reflection_result(signal)
    if isinstance(signal, PlanStepResultSignal):
        return _reduce_plan_step_update(snapshot, signal.action)

    action = signal.action
    if isinstance(action, PlanAction):
        return _reduce_plan_action(snapshot, action)
    if isinstance(action, PlanStepUpdateAction):
        return _reduce_plan_step_update(snapshot, action)
    if isinstance(action, FinalAnswerAction):
        return _reduce_final_answer(snapshot, action)
    if isinstance(action, ToolCallAction):
        return _reduce_tool_call(snapshot, action)
    raise TypeError(f"Unsupported agent action: {type(action).__name__}")


def _reduce_plan_action(snapshot: ActionLoopSnapshot, action: PlanAction) -> ActionTransition:
    if not snapshot.require_plan:
        return _stop_with_failure("Model proposed a new plan after execution had already been approved.")

    if plan_looks_like_reconnaissance(
        [(step.title, step.description) for step in action.plan.steps]
    ):
        if snapshot.planning_feedback_count >= 2:
            return _stop_with_failure(
                "Planning failed because the model kept proposing reconnaissance as the plan."
            )
        return ActionTransition(
            effect=RequestPlanRevisionEffect(plan=action.plan),
            outcome=TransitionOutcome.CONTINUE,
        )

    return ActionTransition(
        effect=PresentPlanEffect(plan=action.plan),
        outcome=TransitionOutcome.STOP,
    )


def _reduce_plan_step_update(
    snapshot: ActionLoopSnapshot,
    action: PlanStepUpdateAction,
) -> ActionTransition:
    if snapshot.require_plan:
        return _stop_with_failure("Planning cannot update plan step status before user approval.")
    if not snapshot.active_plan_approved:
        return _stop_with_failure("Model returned plan_step_update without an approved active plan.")
    if snapshot.active_plan_step_id is None:
        return _stop_with_failure(
            "Model returned plan_step_update but no plan step is currently active."
        )
    if action.step_id != snapshot.active_plan_step_id:
        if snapshot.plan_execution_feedback_count >= 1:
            return _stop_with_failure(
                "Plan execution failed because the model updated the wrong plan step."
            )
        return ActionTransition(
            effect=RequestPlanExecutionFeedbackEffect(
                feedback=(
                    "Plan execution feedback: update only the current executable step. "
                    f"Current step id is {snapshot.active_plan_step_id}; "
                    f"the model tried to update {action.step_id}."
                )
            ),
            outcome=TransitionOutcome.CONTINUE,
        )

    return ActionTransition(
        effect=ApplyPlanStepUpdateEffect(action=action),
        outcome=(
            TransitionOutcome.CONTINUE
            if action.status == "completed"
            else TransitionOutcome.STOP
        ),
    )


def _reduce_final_answer(
    snapshot: ActionLoopSnapshot,
    action: FinalAnswerAction,
) -> ActionTransition:
    if snapshot.require_plan:
        if snapshot.planning_feedback_count >= 2:
            return _stop_with_failure(
                "Planning failed because the model answered directly instead of returning a plan."
            )
        return ActionTransition(
            effect=RequestPlanRevisionEffect(feedback=PLANNING_FINAL_ANSWER_FEEDBACK),
            outcome=TransitionOutcome.CONTINUE,
        )

    if snapshot.approved_plan_incomplete:
        if snapshot.plan_execution_feedback_count >= 1:
            return _stop_with_failure(
                "Plan execution failed because the model returned a final answer before completing "
                "the approved plan."
            )
        return ActionTransition(
            effect=RequestPlanExecutionFeedbackEffect(
                feedback=PLAN_EXECUTION_FINAL_ANSWER_FEEDBACK
            ),
            outcome=TransitionOutcome.CONTINUE,
        )

    reflect = (
        snapshot.max_reflection_attempts > 0
        and snapshot.reflection_count < snapshot.max_reflection_attempts
    )
    if reflect:
        return ActionTransition(
            effect=RequestReflectionEffect(proposed_answer=action.content),
            outcome=TransitionOutcome.AWAIT_RESULT,
        )
    return ActionTransition(
        effect=CompleteAnswerEffect(content=action.content),
        outcome=TransitionOutcome.STOP,
    )


def _reduce_tool_call(
    snapshot: ActionLoopSnapshot,
    action: ToolCallAction,
) -> ActionTransition:
    phase: Literal["planning", "execution"] = (
        "planning" if snapshot.require_plan else "execution"
    )
    if phase == "planning" and action.tool_name not in snapshot.planning_tool_names:
        allowed_tools = format_read_only_planning_tools(snapshot.planning_tool_names)
        return _stop_with_failure(
            "Planning can only use read-only reconnaissance tools before approval. "
            f"Allowed planning tools: {allowed_tools}. "
            "Return a plan action or retry with one of the allowed tools."
        )

    if snapshot.tool_call_count >= snapshot.max_tool_calls_per_turn:
        if phase == "planning" and not snapshot.planning_tool_limit_feedback_sent:
            return ActionTransition(
                effect=RequestPlanRevisionEffect(
                    feedback=PLANNING_TOOL_LIMIT_FEEDBACK,
                    mark_tool_limit_feedback_sent=True,
                ),
                outcome=TransitionOutcome.CONTINUE,
            )
        return _stop_with_failure(
            f"Tool call limit reached ({snapshot.max_tool_calls_per_turn}) "
            f"during {phase} before a final answer."
        )

    return ActionTransition(
        effect=ExecuteToolEffect(action=action, phase=phase),
        outcome=TransitionOutcome.AWAIT_RESULT,
    )


def _reduce_iteration_preparation(snapshot: ActionLoopSnapshot) -> ActionTransition:
    if snapshot.require_plan or not snapshot.active_plan_approved:
        return _proceed()
    if snapshot.active_plan_status == "completed":
        return _proceed()
    if snapshot.active_plan_status == "blocked":
        return ActionTransition(
            effect=BlockTurnEffect(message="Plan execution is blocked."),
            outcome=TransitionOutcome.STOP,
        )
    if snapshot.active_plan_step_id is not None:
        return _proceed()
    if snapshot.next_ready_plan_step_id is not None:
        return ActionTransition(
            effect=StartPlanStepEffect(step_id=snapshot.next_ready_plan_step_id),
            outcome=TransitionOutcome.PROCEED,
        )
    return ActionTransition(
        effect=BlockTurnEffect(
            message="Plan execution blocked because no pending step is ready."
        ),
        outcome=TransitionOutcome.STOP,
    )


def _reduce_reflection_result(signal: ReflectionResultSignal) -> ActionTransition:
    if signal.approved:
        return ActionTransition(
            effect=CompleteAnswerEffect(content=signal.proposed_answer),
            outcome=TransitionOutcome.STOP,
        )
    return ActionTransition(
        effect=RequestReflectionRevisionEffect(
            proposed_answer=signal.proposed_answer,
            reason=signal.reason,
            feedback=signal.feedback,
        ),
        outcome=TransitionOutcome.CONTINUE,
    )


def _reduce_tool_result(
    snapshot: ActionLoopSnapshot,
    signal: ToolResultSignal,
) -> ActionTransition:
    if not signal.success and signal.failure_kind == "fatal_safety":
        return ActionTransition(
            effect=FinishToolEffect(
                disposition="fatal_safety",
                blocked_reason=(
                    f"Fatal safety policy stopped the turn. "
                    f"{_format_tool_failure_reason(signal)}"
                ),
            ),
            outcome=TransitionOutcome.STOP,
        )
    if not signal.has_plan_step:
        return ActionTransition(
            effect=FinishToolEffect(disposition="none"),
            outcome=TransitionOutcome.CONTINUE,
        )
    if signal.success:
        return ActionTransition(
            effect=FinishToolEffect(disposition="evidence"),
            outcome=TransitionOutcome.CONTINUE,
        )

    retry_number = snapshot.active_plan_step_retry_count + 1
    tool_calls_remaining = max(
        0,
        snapshot.max_tool_calls_per_turn - snapshot.tool_call_count,
    )
    retry_scheduled = (
        retry_number <= snapshot.active_plan_step_retry_limit
        and tool_calls_remaining > 0
    )
    if retry_scheduled:
        disposition = "retry_scheduled"
        retries_used = retry_number
    elif retry_number > snapshot.active_plan_step_retry_limit:
        disposition = "retry_limit_exhausted"
        retries_used = snapshot.active_plan_step_retry_count
    else:
        disposition = "tool_call_limit_exhausted"
        retries_used = snapshot.active_plan_step_retry_count
    retry_metadata: dict[str, object] = {
        "disposition": disposition,
        "failure_number": snapshot.active_plan_step_tool_failure_count + 1,
        "retry_limit": snapshot.active_plan_step_retry_limit,
        "retries_used": retries_used,
        "retries_remaining": max(
            0,
            snapshot.active_plan_step_retry_limit - retries_used,
        ),
        "tool_calls_remaining": tool_calls_remaining,
        "failure_kind": signal.failure_kind,
        "error": signal.error,
    }
    if retry_scheduled:
        return ActionTransition(
            effect=FinishToolEffect(
                disposition="retry_scheduled",
                retry_metadata=retry_metadata,
            ),
            outcome=TransitionOutcome.CONTINUE,
        )

    reason = _format_tool_failure_reason(signal)
    blocked_reason = reason
    if snapshot.active_plan_step_retry_limit:
        if disposition == "retry_limit_exhausted":
            retry_label = (
                "retry" if snapshot.active_plan_step_retry_count == 1 else "retries"
            )
            blocked_reason = (
                f"{reason} Step retry limit exhausted after "
                f"{snapshot.active_plan_step_retry_count} {retry_label}."
            )
        else:
            blocked_reason = (
                f"{reason} No step retry could run because the turn tool-call limit "
                f"({snapshot.max_tool_calls_per_turn}) was reached."
            )
    return ActionTransition(
        effect=FinishToolEffect(
            disposition="block",
            retry_metadata=retry_metadata,
            blocked_reason=blocked_reason,
        ),
        outcome=TransitionOutcome.STOP,
    )


def _format_tool_failure_reason(signal: ToolResultSignal) -> str:
    if signal.error:
        return f"Tool {signal.tool_name} failed with {signal.error}."
    if signal.exit_code is not None:
        return f"Tool {signal.tool_name} failed with exit code {signal.exit_code}."
    return f"Tool {signal.tool_name} failed."


def _proceed() -> ActionTransition:
    return ActionTransition(
        effect=ProceedEffect(),
        outcome=TransitionOutcome.PROCEED,
    )


def _stop_with_failure(message: str) -> ActionTransition:
    return ActionTransition(
        effect=FailTurnEffect(message=message),
        outcome=TransitionOutcome.STOP,
    )
