"""Table tests for the pure action-loop transition reducer."""

from __future__ import annotations

from dataclasses import replace

import pytest

from chulk.core.actions import (
    FinalAnswerAction,
    PlanAction,
    PlanStepUpdateAction,
    ToolCallAction,
)
from chulk.core.state import Plan, PlanStep
from chulk.core.transitions import (
    ActionLoopSnapshot,
    ApplyPlanStepUpdateEffect,
    BlockTurnEffect,
    CompleteAnswerEffect,
    ExecuteToolEffect,
    FailTurnEffect,
    FinishToolEffect,
    ModelActionSignal,
    PlanStepResultSignal,
    PresentPlanEffect,
    PrepareIterationSignal,
    ProceedEffect,
    ProtocolFailureSignal,
    ReflectionResultSignal,
    RequestPlanExecutionFeedbackEffect,
    RequestPlanRevisionEffect,
    RequestReflectionEffect,
    RequestReflectionRevisionEffect,
    StartPlanStepEffect,
    ToolResultSignal,
    TransitionOutcome,
    reduce_transition,
)


def _snapshot(**changes: object) -> ActionLoopSnapshot:
    base = ActionLoopSnapshot(
        require_plan=False,
        planning_tool_names=frozenset({"read_file"}),
    )
    return replace(base, **changes)


def _plan(*, reconnaissance: bool = False) -> Plan:
    return Plan(
        summary="Implement the change.",
        steps=[
            PlanStep(
                id="implementation",
                title="Inspect the repository" if reconnaissance else "Implement the reducer",
                description=(
                    "Read files and explore the codebase"
                    if reconnaissance
                    else "Update the action loop and verify its behavior"
                ),
            )
        ],
    )


def _final_answer() -> FinalAnswerAction:
    return FinalAnswerAction(type="final_answer", content="Done.")


def _tool_call(name: str = "read_file") -> ToolCallAction:
    return ToolCallAction(type="tool_call", tool_name=name, arguments={"path": "README.md"})


def _step_update(
    *,
    step_id: str = "implementation",
    status: str = "completed",
) -> PlanStepUpdateAction:
    return PlanStepUpdateAction(
        type="plan_step_update",
        step_id=step_id,
        status=status,  # type: ignore[arg-type]
        evidence="Verified by tests.",
        reason="The dependency is unavailable." if status == "blocked" else None,
    )


@pytest.mark.parametrize(
    ("snapshot", "action", "effect_type", "outcome"),
    [
        (
            _snapshot(),
            PlanAction(type="plan", plan=_plan()),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(require_plan=True),
            PlanAction(type="plan", plan=_plan(reconnaissance=True)),
            RequestPlanRevisionEffect,
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(require_plan=True, planning_feedback_count=2),
            PlanAction(type="plan", plan=_plan(reconnaissance=True)),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(require_plan=True),
            PlanAction(type="plan", plan=_plan()),
            PresentPlanEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(require_plan=True),
            _final_answer(),
            RequestPlanRevisionEffect,
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(require_plan=True, planning_feedback_count=2),
            _final_answer(),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(active_plan_approved=True, approved_plan_incomplete=True),
            _final_answer(),
            RequestPlanExecutionFeedbackEffect,
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(
                active_plan_approved=True,
                approved_plan_incomplete=True,
                plan_execution_feedback_count=1,
            ),
            _final_answer(),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(),
            _final_answer(),
            CompleteAnswerEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(max_reflection_attempts=1),
            _final_answer(),
            RequestReflectionEffect,
            TransitionOutcome.AWAIT_RESULT,
        ),
        (
            _snapshot(max_reflection_attempts=1, reflection_count=1),
            _final_answer(),
            CompleteAnswerEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(require_plan=True),
            _tool_call("shell"),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(require_plan=True),
            _tool_call(),
            ExecuteToolEffect,
            TransitionOutcome.AWAIT_RESULT,
        ),
        (
            _snapshot(),
            _tool_call("shell"),
            ExecuteToolEffect,
            TransitionOutcome.AWAIT_RESULT,
        ),
        (
            _snapshot(
                require_plan=True,
                tool_call_count=5,
                max_tool_calls_per_turn=5,
            ),
            _tool_call(),
            RequestPlanRevisionEffect,
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(
                require_plan=True,
                tool_call_count=5,
                max_tool_calls_per_turn=5,
                planning_tool_limit_feedback_sent=True,
            ),
            _tool_call(),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(tool_call_count=5, max_tool_calls_per_turn=5),
            _tool_call("shell"),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(require_plan=True),
            _step_update(),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(),
            _step_update(),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(active_plan_approved=True),
            _step_update(),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(active_plan_approved=True, active_plan_step_id="implementation"),
            _step_update(step_id="other"),
            RequestPlanExecutionFeedbackEffect,
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(
                active_plan_approved=True,
                active_plan_step_id="implementation",
                plan_execution_feedback_count=1,
            ),
            _step_update(step_id="other"),
            FailTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(active_plan_approved=True, active_plan_step_id="implementation"),
            _step_update(),
            ApplyPlanStepUpdateEffect,
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(active_plan_approved=True, active_plan_step_id="implementation"),
            _step_update(status="blocked"),
            ApplyPlanStepUpdateEffect,
            TransitionOutcome.STOP,
        ),
    ],
    ids=[
        "plan-after-approval",
        "reconnaissance-plan-feedback",
        "reconnaissance-plan-exhausted",
        "present-plan",
        "direct-answer-during-planning",
        "direct-answer-planning-exhausted",
        "early-plan-answer-feedback",
        "early-plan-answer-exhausted",
        "final-answer",
        "final-answer-reflection",
        "final-answer-reflection-exhausted",
        "unsafe-planning-tool",
        "read-only-planning-tool",
        "execution-tool",
        "planning-tool-limit-feedback",
        "planning-tool-limit-exhausted",
        "execution-tool-limit",
        "step-update-during-planning",
        "step-update-without-plan",
        "step-update-without-active-step",
        "wrong-step-feedback",
        "wrong-step-exhausted",
        "complete-step",
        "block-step",
    ],
)
def test_reduce_transition_table(
    snapshot: ActionLoopSnapshot,
    action: object,
    effect_type: type[object],
    outcome: TransitionOutcome,
) -> None:
    transition = reduce_transition(snapshot, ModelActionSignal(action=action))  # type: ignore[arg-type]

    assert isinstance(transition.effect, effect_type)
    assert transition.outcome is outcome


def test_reduce_transition_returns_transport_details_without_mutating_inputs() -> None:
    snapshot = _snapshot(require_plan=True)
    action = _tool_call()

    transition = reduce_transition(snapshot, ModelActionSignal(action=action))

    assert transition == transition.__class__(
        effect=ExecuteToolEffect(action=action, phase="planning"),
        outcome=TransitionOutcome.AWAIT_RESULT,
    )
    assert snapshot.tool_call_count == 0
    assert action.arguments == {"path": "README.md"}


def test_planning_tool_limit_effect_explicitly_requests_one_way_flag_update() -> None:
    snapshot = _snapshot(
        require_plan=True,
        tool_call_count=5,
        max_tool_calls_per_turn=5,
    )

    transition = reduce_transition(snapshot, ModelActionSignal(action=_tool_call()))

    assert isinstance(transition.effect, RequestPlanRevisionEffect)
    assert transition.effect.mark_tool_limit_feedback_sent is True
    assert "reconnaissance tool budget is exhausted" in (transition.effect.feedback or "")


def test_tool_call_limit_is_shared_by_planning_and_execution() -> None:
    planning_transition = reduce_transition(
        _snapshot(require_plan=True, tool_call_count=4, max_tool_calls_per_turn=5),
        ModelActionSignal(action=_tool_call()),
    )
    execution_transition = reduce_transition(
        _snapshot(tool_call_count=5, max_tool_calls_per_turn=5),
        ModelActionSignal(action=_tool_call("shell")),
    )

    assert isinstance(planning_transition.effect, ExecuteToolEffect)
    assert isinstance(execution_transition.effect, FailTurnEffect)
    assert "tool call limit" in execution_transition.effect.message.lower()


@pytest.mark.parametrize(
    ("snapshot", "effect_type", "outcome"),
    [
        (_snapshot(), ProceedEffect, TransitionOutcome.PROCEED),
        (
            _snapshot(
                active_plan_approved=True,
                active_plan_status="approved",
                next_ready_plan_step_id="implementation",
            ),
            StartPlanStepEffect,
            TransitionOutcome.PROCEED,
        ),
        (
            _snapshot(active_plan_approved=True, active_plan_status="blocked"),
            BlockTurnEffect,
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(active_plan_approved=True, active_plan_status="approved"),
            BlockTurnEffect,
            TransitionOutcome.STOP,
        ),
    ],
    ids=["unplanned", "start-ready-step", "blocked-plan", "no-ready-step"],
)
def test_reduce_plan_step_preparation(
    snapshot: ActionLoopSnapshot,
    effect_type: type[object],
    outcome: TransitionOutcome,
) -> None:
    transition = reduce_transition(snapshot, PrepareIterationSignal())

    assert isinstance(transition.effect, effect_type)
    assert transition.outcome is outcome


def test_reduce_protocol_failure_to_terminal_effect() -> None:
    transition = reduce_transition(
        _snapshot(),
        ProtocolFailureSignal(message="Invalid structured action."),
    )

    assert transition == transition.__class__(
        effect=FailTurnEffect(message="Invalid structured action."),
        outcome=TransitionOutcome.STOP,
    )


@pytest.mark.parametrize(
    ("approved", "effect_type", "outcome"),
    [
        (True, CompleteAnswerEffect, TransitionOutcome.STOP),
        (False, RequestReflectionRevisionEffect, TransitionOutcome.CONTINUE),
    ],
)
def test_reduce_reflection_result(
    approved: bool,
    effect_type: type[object],
    outcome: TransitionOutcome,
) -> None:
    transition = reduce_transition(
        _snapshot(),
        ReflectionResultSignal(
            proposed_answer="Done.",
            approved=approved,
            reason="The answer is sufficiently grounded." if approved else "Missing evidence.",
            feedback=None if approved else "Name the evidence.",
        ),
    )

    assert isinstance(transition.effect, effect_type)
    assert transition.outcome is outcome


@pytest.mark.parametrize(
    ("snapshot", "signal", "disposition", "outcome"),
    [
        (
            _snapshot(),
            ToolResultSignal(
                tool_name="lookup",
                phase="execution",
                success=False,
                has_plan_step=False,
                error="unavailable",
            ),
            "none",
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(active_plan_step_id="implementation"),
            ToolResultSignal(
                tool_name="lookup",
                phase="execution",
                success=True,
                has_plan_step=True,
            ),
            "evidence",
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(
                active_plan_step_id="implementation",
                active_plan_step_retry_limit=1,
            ),
            ToolResultSignal(
                tool_name="lookup",
                phase="execution",
                success=False,
                has_plan_step=True,
                error="unavailable",
                failure_kind="environment",
            ),
            "retry_scheduled",
            TransitionOutcome.CONTINUE,
        ),
        (
            _snapshot(
                active_plan_step_id="implementation",
                active_plan_step_retry_count=1,
                active_plan_step_retry_limit=1,
                active_plan_step_tool_failure_count=1,
            ),
            ToolResultSignal(
                tool_name="lookup",
                phase="execution",
                success=False,
                has_plan_step=True,
                error="unavailable",
                failure_kind="environment",
            ),
            "block",
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(
                active_plan_step_id="implementation",
                active_plan_step_retry_limit=1,
                tool_call_count=5,
                max_tool_calls_per_turn=5,
            ),
            ToolResultSignal(
                tool_name="lookup",
                phase="execution",
                success=False,
                has_plan_step=True,
                error="unavailable",
                failure_kind="environment",
            ),
            "block",
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(),
            ToolResultSignal(
                tool_name="run_cmd",
                phase="execution",
                success=False,
                has_plan_step=False,
                error="blocked_command",
                failure_kind="fatal_safety",
            ),
            "fatal_safety",
            TransitionOutcome.STOP,
        ),
        (
            _snapshot(
                active_plan_step_id="implementation",
                active_plan_step_retry_limit=3,
            ),
            ToolResultSignal(
                tool_name="run_cmd",
                phase="execution",
                success=False,
                has_plan_step=True,
                error="containment_required",
                failure_kind="fatal_safety",
            ),
            "fatal_safety",
            TransitionOutcome.STOP,
        ),
    ],
    ids=[
        "unplanned-failure",
        "plan-success-evidence",
        "plan-retry",
        "plan-retry-exhausted",
        "plan-tool-limit-exhausted",
        "unplanned-fatal-safety",
        "planned-fatal-safety",
    ],
)
def test_reduce_tool_result(
    snapshot: ActionLoopSnapshot,
    signal: ToolResultSignal,
    disposition: str,
    outcome: TransitionOutcome,
) -> None:
    transition = reduce_transition(snapshot, signal)

    assert isinstance(transition.effect, FinishToolEffect)
    assert transition.effect.disposition == disposition
    assert transition.outcome is outcome


def test_reduce_plan_step_result_signal_explicitly() -> None:
    action = _step_update()
    transition = reduce_transition(
        _snapshot(active_plan_approved=True, active_plan_step_id="implementation"),
        PlanStepResultSignal(action=action),
    )

    assert transition == transition.__class__(
        effect=ApplyPlanStepUpdateEffect(action=action),
        outcome=TransitionOutcome.CONTINUE,
    )
