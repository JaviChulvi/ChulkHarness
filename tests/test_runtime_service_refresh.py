"""Compatibility tests for mutable Agent runtime configuration."""

from __future__ import annotations

import pytest

from chulk.core import Agent
from chulk.core.context import ContextBudget
from chulk.core.state import AgentState
from chulk.memory import ConversationMemory
from chulk.testing import ScriptedLLMClient
from chulk.tools import ToolRegistry
from chulk.tools.permissions import ToolPermissionPolicy


def test_agent_refreshes_service_captured_runtime_configuration() -> None:
    agent = Agent(ScriptedLLMClient([]))
    llm = ScriptedLLMClient([])
    state = AgentState()
    memory = ConversationMemory()
    registry = ToolRegistry()
    permission_policy = ToolPermissionPolicy()
    context_budget = ContextBudget(max_prompt_tokens=1234, response_reserve_tokens=12)

    agent.llm_client = llm
    agent.state = state
    agent.memory = memory
    agent.tool_registry = registry
    agent.permission_policy = permission_policy
    agent.system_prompt = "replacement system prompt"
    agent.context_budget = context_budget
    agent.max_skill_content_chars = 111
    agent.max_tool_calls_per_turn = 7
    agent.max_json_repair_attempts = 4
    agent.max_reflection_attempts = 2
    agent.trace_max_prompt_chars = 2222
    agent.max_observation_chars = 333
    agent.max_tool_stdout_chars = 444
    agent.max_tool_stderr_chars = 555

    agent._refresh_action_runtime()

    assert agent._model_transport.llm_client is llm
    assert agent._model_transport.state is state
    assert agent._model_transport.memory is memory
    assert agent._model_transport.tool_registry is registry
    assert agent._model_transport.system_prompt == "replacement system prompt"
    assert agent._model_transport.context_budget is context_budget
    assert agent._model_transport.max_json_repair_attempts == 4
    assert agent._tool_executor.registry is registry
    assert agent._tool_executor.permission_policy is permission_policy
    assert agent._plan_execution.state is state
    assert agent._turn_effects.state is state
    assert agent._turn_effects.max_tool_calls_per_turn == 7
    assert agent._turn_effects.max_reflection_attempts == 2
    assert agent._turn_effects.max_observation_chars == 333
    assert agent._turn_effects.max_tool_stdout_chars == 444
    assert agent._turn_effects.max_tool_stderr_chars == 555


def test_replaced_state_and_memory_are_used_when_turn_setup_fails() -> None:
    events: list[tuple[str, dict]] = []
    fail_turn_start = True

    def capture(event_type: str, payload: dict) -> None:
        nonlocal fail_turn_start
        if event_type == "turn_started" and fail_turn_start:
            fail_turn_start = False
            raise RuntimeError("turn-start failure")
        events.append((event_type, payload))

    agent = Agent(ScriptedLLMClient([]), event_callback=capture)
    state = AgentState()
    memory = ConversationMemory()
    agent.state = state
    agent.memory = memory

    with pytest.raises(RuntimeError, match="turn-start failure"):
        agent.run_turn("Trigger setup failure")

    finished = next(payload for event, payload in events if event == "turn_finished")
    assert state.turns[-1].status == "failed"
    assert len(memory.messages) == 1
    assert finished["agent_state"]["turn_count"] == 1
    assert finished["agent_state"]["message_count"] == 1
    assert finished["agent_state"]["error_count"] == 1
    assert finished["agent_state"]["final_answer"].endswith("turn-start failure")


def test_reject_plan_snapshots_replaced_memory() -> None:
    events: list[tuple[str, dict]] = []
    agent = Agent(
        ScriptedLLMClient(
            [
                {
                    "type": "plan",
                    "plan": {
                        "summary": "Implement the change.",
                        "steps": [
                            {
                                "id": "implementation",
                                "title": "Implement",
                                "description": "Implement and verify the change.",
                            }
                        ],
                    },
                }
            ]
        ),
        event_callback=lambda event, payload: events.append((event, payload)),
    )
    agent.run_planned_turn("Create a plan")
    replacement_memory = ConversationMemory()
    agent.memory = replacement_memory
    events.clear()

    assert agent.reject_plan() == "Plan rejected. No tools were run."

    finished = next(payload for event, payload in events if event == "turn_finished")
    assert len(replacement_memory.messages) == 1
    assert finished["agent_state"]["message_count"] == 1
