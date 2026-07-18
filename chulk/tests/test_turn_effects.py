"""Contracts for consuming reducer outcomes at the mutation boundary."""

from __future__ import annotations

import json

import pytest

from chulk.core.actions import ToolCallAction
from chulk.core.events import TraceEvent
from chulk.core.plan_execution import PlanExecution
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
        llm_client=ScriptedLLMClient([]),
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
        result=ToolResult(tool_name="lookup", success=True, observation="lookup complete"),
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
    assert memory.messages[1]["content"].endswith("lookup complete")
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
    assert observation_events[-1]["turn"]["observations"][0]["content"].endswith(
        "lookup complete"
    )


def test_planned_tool_observation_checkpoint_includes_step_evidence() -> None:
    state = AgentState()
    memory = ConversationMemory()
    events: list[tuple[str, dict | None]] = []

    def trace(event_type: str, payload: dict | None) -> None:
        events.append((event_type, payload))

    effects = TurnEffects(
        state=state,
        memory=memory,
        llm_client=ScriptedLLMClient([]),
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


def test_presented_plan_stays_in_canonical_state_not_model_history() -> None:
    state = AgentState()
    memory = ConversationMemory()
    memory.add_user_message("Plan this change.")
    execution = PlanExecution(state=state, memory=memory, trace=lambda _event, _payload: None)
    turn = TurnState(user_message="Plan this change.")
    plan = Plan(
        summary="Change the behavior.",
        steps=[PlanStep(id="1", title="Implement", description="Update the component.")],
    )

    response = execution.present(turn, plan)

    assert "Use /approve" in response
    assert memory.messages == [{"role": "user", "content": "Plan this change."}]
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
