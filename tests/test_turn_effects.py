"""Contracts for consuming reducer outcomes at the mutation boundary."""

from __future__ import annotations

import json

import pytest

from chulk.core.actions import PlanStepUpdateAction, ToolCallAction
from chulk.core.events import TraceEvent
from chulk.core.plan_execution import (
    PlanExecution,
    PlanStepVerification,
    PlanStepVerificationRequest,
)
from chulk.core.state import AgentState, Plan, PlanStep, TurnState
from chulk.core.transitions import ExecuteToolEffect, FinishToolEffect, TransitionOutcome
from chulk.core.turn_effects import PendingReflection, TurnEffects, _validate_application
from chulk.memory import ConversationMemory
from chulk.testing import ScriptedLLMClient
from chulk.tools import ToolResult


@pytest.mark.parametrize(
    ("outcome", "response", "pending"),
    [
        (TransitionOutcome.STOP, None, None),
        (
            TransitionOutcome.AWAIT_RESULT,
            None,
            None,
        ),
        (TransitionOutcome.CONTINUE, "unexpected", None),
        (
            TransitionOutcome.PROCEED,
            None,
            PendingReflection(proposed_answer="unexpected"),
        ),
    ],
)
def test_transition_application_rejects_mismatched_reducer_outcome(
    outcome: TransitionOutcome,
    response: str | None,
    pending: PendingReflection | None,
) -> None:
    with pytest.raises(RuntimeError):
        _validate_application(outcome=outcome, response=response, pending=pending)


def test_transition_application_preserves_reducer_outcome() -> None:
    application = _validate_application(
        outcome=TransitionOutcome.STOP,
        response="done",
        pending=None,
    )

    assert application.outcome is TransitionOutcome.STOP
    assert application.response == "done"


def test_finished_tool_adds_bounded_redacted_action_before_observation() -> None:
    state = AgentState()
    memory = ConversationMemory()
    events: list[tuple[str, dict | None]] = []

    def trace(event_type: str, payload: dict | None) -> None:
        events.append((event_type, payload))

    def redact(_event_type: str, text: str, metadata: dict) -> tuple[str, dict]:
        redacted = text.replace("SECRET", "MASKED")
        return redacted, {"redacted": redacted != text, "field": metadata.get("field")}

    effects = TurnEffects(
        state=state,
        memory=memory,
        get_llm_client=lambda: ScriptedLLMClient([]),
        plan=PlanExecution(state=state, memory=memory, trace=trace),
        trace=trace,
        redact_text=redact,
        artifact_writer=lambda _name, _content: None,
        planning_tool_names=lambda: frozenset(),
        max_tool_calls_per_turn=5,
        max_reflection_attempts=0,
        max_observation_chars=1000,
        max_tool_stdout_chars=1000,
        max_tool_stderr_chars=1000,
    )
    turn = TurnState(user_message="Use the lookup tool.")
    action = ToolCallAction(
        type="tool_call",
        tool_name="lookup",
        arguments={"payload": "UNIQUE_ARGUMENT_" + ("x" * 5000), "token": "SECRET"},
    )
    pending = effects._start_tool(turn, ExecuteToolEffect(action=action, phase="execution"))

    blocked_message = effects._finish_tool(
        turn,
        FinishToolEffect(disposition="none"),
        pending=pending,
        result=ToolResult(
            tool_name="lookup",
            success=True,
            observation="lookup complete",
            exit_code=0,
        ),
    )

    assert blocked_message is None
    assert [message["role"] for message in memory.messages] == ["assistant", "observation"]
    action_context = memory.messages[0]["content"]
    assert len(action_context) <= 1000
    assert "UNIQUE_ARGUMENT_" in action_context
    assert "SECRET" not in action_context
    assert "MASKED" in action_context
    assert action_context.startswith("<executed_tool_action>\n")
    action_payload = json.loads(
        action_context.removeprefix("<executed_tool_action>\n").removesuffix(
            "\n</executed_tool_action>"
        )
    )
    assert action_payload["tool_name"] == "lookup"
    assert action_payload["arguments_truncated"] is True
    assert "lookup complete" in memory.messages[1]["content"]
    assert memory.messages[1]["content"].endswith("exit_code: 0")
    action_metadata = turn.observations[0].output_metadata["tool_action_context"]
    assert action_metadata["arguments"]["truncated"] is True
    assert action_metadata["redaction"]["redacted"] is True
    observation_events = [
        payload
        for event, payload in events
        if event == TraceEvent.TOOL_OBSERVATION and payload is not None
    ]
    assert observation_events[-1]["tool_action_context"] == action_context
    assert observation_events[-1]["observation_index"] == 1
    assert "lookup complete" in observation_events[-1]["turn"]["observations"][0][
        "content"
    ]
    completed_payload = next(
        payload
        for event, payload in events
        if event == TraceEvent.TOOL_CALL_COMPLETED and payload is not None
    )
    assert completed_payload["exit_code"] == 0


def test_planned_tool_observation_checkpoint_includes_step_evidence() -> None:
    state = AgentState()
    memory = ConversationMemory()
    events: list[tuple[str, dict | None]] = []

    def trace(event_type: str, payload: dict | None) -> None:
        events.append((event_type, payload))

    effects = TurnEffects(
        state=state,
        memory=memory,
        get_llm_client=lambda: ScriptedLLMClient([]),
        plan=PlanExecution(state=state, memory=memory, trace=trace),
        trace=trace,
        redact_text=lambda _event, text, _metadata: (text, {"redacted": False}),
        artifact_writer=lambda _name, _content: None,
        planning_tool_names=lambda: frozenset(),
        max_tool_calls_per_turn=5,
        max_reflection_attempts=0,
        max_observation_chars=1000,
        max_tool_stdout_chars=1000,
        max_tool_stderr_chars=1000,
    )
    plan = Plan(
        summary="Use one tool.",
        steps=[PlanStep(id="1", title="Lookup", description="Run lookup.")],
    )
    plan.approve()
    plan.steps[0].mark("in_progress")
    turn = TurnState(
        user_message="Use lookup.",
        active_plan=plan,
        plan_approved=True,
    )
    pending = effects._start_tool(
        turn,
        ExecuteToolEffect(
            action=ToolCallAction(type="tool_call", tool_name="lookup", arguments={}),
            phase="execution",
        ),
    )

    effects._finish_tool(
        turn,
        FinishToolEffect(disposition="evidence"),
        pending=pending,
        result=ToolResult(tool_name="lookup", success=True, observation="lookup complete"),
    )

    observation_payload = next(
        payload
        for event, payload in events
        if event == TraceEvent.TOOL_OBSERVATION and payload is not None
    )
    saved_step = observation_payload["turn"]["active_plan"]["steps"][0]
    assert saved_step["status"] == "in_progress"
    assert saved_step["evidence"][0]["content"].endswith("lookup complete")
    assert len(plan.steps[0].evidence) == 1


def test_terminal_tool_outcome_blocks_step_before_observation_checkpoint() -> None:
    state = AgentState()
    memory = ConversationMemory()
    events: list[tuple[str, dict | None]] = []

    def trace(event_type: str, payload: dict | None) -> None:
        events.append((event_type, payload))

    effects = TurnEffects(
        state=state,
        memory=memory,
        get_llm_client=lambda: ScriptedLLMClient([]),
        plan=PlanExecution(state=state, memory=memory, trace=trace),
        trace=trace,
        redact_text=lambda _event, text, _metadata: (text, {"redacted": False}),
        artifact_writer=lambda _name, _content: None,
        planning_tool_names=lambda: frozenset(),
        max_tool_calls_per_turn=5,
        max_reflection_attempts=0,
        max_observation_chars=1000,
        max_tool_stdout_chars=1000,
        max_tool_stderr_chars=1000,
    )
    plan = Plan(
        summary="Use one failing tool.",
        steps=[PlanStep(id="1", title="Lookup", description="Run lookup.")],
    )
    plan.approve()
    plan.steps[0].mark("in_progress")
    turn = TurnState(
        user_message="Use lookup.",
        active_plan=plan,
        plan_approved=True,
    )
    pending = effects._start_tool(
        turn,
        ExecuteToolEffect(
            action=ToolCallAction(type="tool_call", tool_name="lookup", arguments={}),
            phase="execution",
        ),
    )

    blocked_message = effects._finish_tool(
        turn,
        FinishToolEffect(
            disposition="block",
            blocked_reason="Tool retry limit exhausted.",
        ),
        pending=pending,
        result=ToolResult(
            tool_name="lookup",
            success=False,
            observation="lookup failed",
            error="timeout",
        ),
    )

    observation_payload = next(
        payload
        for event, payload in events
        if event == TraceEvent.TOOL_OBSERVATION and payload is not None
    )
    blocked_payload = next(
        payload
        for event, payload in events
        if event == TraceEvent.PLAN_STEP_BLOCKED and payload is not None
    )
    assert observation_payload["turn"]["active_plan"]["steps"][0]["status"] == "blocked"
    assert blocked_payload["turn"]["status"] == "blocked"
    assert blocked_payload["turn"]["final_answer"] == blocked_message
    assert turn.errors == [blocked_message]


def test_presented_plan_stays_in_canonical_state_not_model_history() -> None:
    state = AgentState()
    memory = ConversationMemory()
    memory.add_user_message("Plan this change.")
    memory.add_assistant_message("Read-only reconnaissance action.")
    memory.add_observation("Reconnaissance result.")
    execution = PlanExecution(state=state, memory=memory, trace=lambda _event, _payload: None)
    turn = TurnState(user_message="Plan this change.")
    plan = Plan(
        summary="Change the behavior.",
        steps=[PlanStep(id="1", title="Implement", description="Update the component.")],
    )

    response = execution.present(turn, plan)

    assert "Use /approve" in response
    assert state.messages == memory.messages
    assert response not in [message["content"] for message in state.messages]
    assert state.active_plan is plan
    assert turn.active_plan is plan


def test_default_plan_revision_feedback_is_domain_neutral() -> None:
    state = AgentState()
    memory = ConversationMemory()
    execution = PlanExecution(state=state, memory=memory, trace=lambda _event, _payload: None)
    turn = TurnState(user_message="Plan this change.")

    execution.request_revision(turn, plan=None, feedback=None)

    feedback = memory.messages[-1]["content"]
    assert "read_file" not in feedback
    assert "search_files" not in feedback
    assert "modules/files" not in feedback
    assert "available read-only reconnaissance action" in feedback


def test_plan_completion_verifier_rejection_keeps_step_in_progress() -> None:
    state = AgentState()
    memory = ConversationMemory()
    requests: list[PlanStepVerificationRequest] = []

    def verifier(request: PlanStepVerificationRequest) -> PlanStepVerification:
        requests.append(request)
        return PlanStepVerification(
            passed=False,
            evidence="The required test has not passed.",
        )

    execution = PlanExecution(
        state=state,
        memory=memory,
        trace=lambda _event, _payload: None,
        verifier=verifier,
    )
    turn, step = _active_plan_step(execution)

    result = execution.apply_step_result(
        turn,
        PlanStepUpdateAction(
            type="plan_step_update",
            step_id=step.id,
            status="completed",
            evidence="The model says the work is done.",
        ),
    )

    assert result is None
    assert step.status == "in_progress"
    assert step.evidence == []
    assert requests[0].acceptance_criteria == ("The focused test passes.",)
    assert requests[0].recorded_evidence == ()
    assert turn.observations[-1].tool_name == "plan_step_verification"
    assert "required test has not passed" in turn.observations[-1].content


@pytest.mark.asyncio
async def test_async_plan_completion_verifier_records_authoritative_evidence() -> None:
    state = AgentState()
    memory = ConversationMemory()

    async def verifier(
        request: PlanStepVerificationRequest,
    ) -> PlanStepVerification:
        assert request.asserted_evidence == "Focused test passed."
        return PlanStepVerification(
            passed=True,
            evidence="pytest tests/test_feature.py passed.",
        )

    execution = PlanExecution(
        state=state,
        memory=memory,
        trace=lambda _event, _payload: None,
        async_verifier=verifier,
    )
    turn, step = _active_plan_step(execution)

    result = await execution.apply_step_result_async(
        turn,
        PlanStepUpdateAction(
            type="plan_step_update",
            step_id=step.id,
            status="completed",
            evidence="Focused test passed.",
        ),
    )

    assert result is None
    assert step.status == "completed"
    assert [record.tool_name for record in step.evidence] == [
        "plan_step_update",
        "plan_step_verifier",
    ]
    assert step.evidence[-1].metadata == {"external_verification": True}


def _active_plan_step(execution: PlanExecution) -> tuple[TurnState, PlanStep]:
    step = PlanStep(
        id="implementation",
        title="Implement",
        description="Implement and verify the behavior.",
        acceptance_criteria=["The focused test passes."],
    )
    plan = Plan(summary="Implement the behavior.", steps=[step])
    turn = TurnState(user_message="Do the work.", active_plan=plan)
    turn.approve_plan()
    execution.start_step(turn, step.id)
    return turn, step
