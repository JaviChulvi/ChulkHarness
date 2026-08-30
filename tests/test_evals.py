"""Tests for deterministic agent evaluations."""

from __future__ import annotations

import pytest

from chulk import Agent as PublicAgent
from chulk import AgentConfig
from chulk.core import TraceEvent
from tests.core_agent import build_core_agent as Agent
from chulk.core.actions import FinalAnswerAction, ToolCallAction
from chulk.evals import EvalExpectations, EvalRunner, EvalScenario, run_eval
from chulk.testing import ScriptedLLMClient
from chulk.tools import Tool, ToolRegistry


def test_run_eval_asserts_normalized_turn_outcomes() -> None:
    scenario = EvalScenario(
        name="direct answer",
        user_message="Are you ready?",
        scripted_responses=(FinalAnswerAction(type="final_answer", content="Ready."),),
        expectations=EvalExpectations(
            answer="Ready.",
            status="completed",
            tool_sequence=(),
            trace_event_sequence=(
                TraceEvent.TURN_STARTED,
                TraceEvent.MODEL_REQUEST_STARTED,
                TraceEvent.FINAL_ANSWER,
                TraceEvent.TURN_FINISHED,
            ),
        ),
    )

    result = run_eval(scenario)

    result.assert_passed()
    assert result.passed is True
    assert result.responses_remaining == 0
    assert result.exception is None


def test_eval_factory_exercises_tools_and_preserves_existing_callback() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="Look up a deterministic value.",
            args_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
            callable=lambda arguments: f"value:{arguments['key']}",
        )
    )
    callback_events: list[str] = []

    def factory(client: ScriptedLLMClient) -> Agent:
        return Agent(
            client,
            tool_registry=registry,
            event_callback=lambda event_type, payload: callback_events.append(event_type),
        )

    scenario = EvalScenario(
        name="tool path",
        user_message="Look up alpha",
        scripted_responses=(
            ToolCallAction(type="tool_call", tool_name="lookup", arguments={"key": "alpha"}),
            FinalAnswerAction(type="final_answer", content="Found it."),
        ),
        expectations=EvalExpectations(
            answer="Found it.",
            tool_sequence=("lookup",),
            trace_event_sequence=(
                TraceEvent.TOOL_CALL_STARTED,
                TraceEvent.TOOL_CALL_COMPLETED,
                TraceEvent.FINAL_ANSWER,
            ),
        ),
    )

    result = EvalRunner(factory).run(scenario)

    result.assert_passed()
    assert TraceEvent.TOOL_CALL_COMPLETED in callback_events
    assert registry.call_log[0]["tool_name"] == "lookup"


def test_eval_existing_agent_restores_injected_client_and_callback() -> None:
    original_client = ScriptedLLMClient(
        [FinalAnswerAction(type="final_answer", content="Original response.")]
    )
    callback_events: list[str] = []

    def callback(event_type: str, payload: dict) -> None:
        callback_events.append(event_type)

    agent = Agent(original_client, event_callback=callback)
    scenario = EvalScenario(
        name="injected agent",
        user_message="Use the scenario response",
        scripted_responses=(FinalAnswerAction(type="final_answer", content="Scenario response."),),
        expectations=EvalExpectations(answer="Scenario response.", tool_sequence=()),
    )

    result = run_eval(scenario, agent=agent)

    result.assert_passed()
    assert agent.llm_client is original_client
    assert agent.events.event_callback is callback
    assert agent.closed is False
    assert original_client.remaining == 1
    assert TraceEvent.FINAL_ANSWER in callback_events


def test_eval_accepts_the_public_agent_facade(tmp_path) -> None:
    original_client = ScriptedLLMClient(
        [FinalAnswerAction(type="final_answer", content="Original response.")]
    )
    scenario = EvalScenario(
        name="public facade",
        user_message="Use the eval script",
        scripted_responses=(FinalAnswerAction(type="final_answer", content="Evaluated."),),
        expectations=EvalExpectations(answer="Evaluated.", tool_sequence=()),
    )

    with PublicAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=original_client,
        tools=[],
        skills=[],
    ) as agent:
        result = run_eval(scenario, agent=agent)

        result.assert_passed()
        assert agent.runtime.llm_client is original_client
        assert original_client.remaining == 1


def test_eval_result_reports_all_mismatches() -> None:
    scenario = EvalScenario(
        name="mismatch",
        user_message="Run",
        scripted_responses=(FinalAnswerAction(type="final_answer", content="actual"),),
        expectations=EvalExpectations(
            answer="expected",
            status="blocked",
            tool_sequence=("missing_tool",),
            trace_event_sequence=(TraceEvent.TURN_FINISHED, TraceEvent.TURN_STARTED),
        ),
    )

    result = run_eval(scenario)

    assert result.passed is False
    assert len(result.failures) == 4
    with pytest.raises(AssertionError, match="expected answer") as error:
        result.assert_passed()
    assert "expected status" in str(error.value)
    assert "expected tool sequence" in str(error.value)
    assert "expected trace-event subsequence" in str(error.value)


def test_eval_records_execution_exceptions_instead_of_aborting_batch() -> None:
    scenario = EvalScenario(
        name="exhausted script",
        user_message="Run",
        scripted_responses=(),
        expectations=EvalExpectations(),
    )

    result = run_eval(scenario)

    assert result.passed is False
    assert result.exception is not None
    assert "LLMError" in result.exception
    assert result.failures[0].startswith("execution raised LLMError")


def test_eval_rejects_agent_and_factory_together() -> None:
    client = ScriptedLLMClient([])
    agent = Agent(client)
    scenario = EvalScenario(
        name="invalid setup",
        user_message="Run",
        scripted_responses=(),
        expectations=EvalExpectations(),
    )

    with pytest.raises(ValueError, match="not both"):
        EvalRunner(lambda scripted: Agent(scripted)).run(scenario, agent=agent)


def test_eval_runner_uses_a_fresh_agent_for_each_scenario() -> None:
    scenarios = tuple(
        EvalScenario(
            name=f"scenario {index}",
            user_message="Run",
            scripted_responses=(FinalAnswerAction(type="final_answer", content=str(index)),),
            expectations=EvalExpectations(answer=str(index), tool_sequence=()),
        )
        for index in range(2)
    )

    results = EvalRunner().run_many(scenarios)

    assert [result.passed for result in results] == [True, True]
    assert [result.answer for result in results] == ["0", "1"]
