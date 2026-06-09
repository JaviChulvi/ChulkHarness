"""Tests for the Phase 1 agent loop."""

import json

from src.core import Agent
from src.memory import ConversationMemory
from src.tools import ToolRegistry, calculator_tool, shell_tool


class RecordingLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.responses.pop(0)


def test_agent_sends_user_message_and_stores_response():
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "Hi Javier"})])
    agent = Agent(llm)

    response = agent.run_turn("hello")

    assert response == "Hi Javier"
    assert agent.memory.messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi Javier"},
    ]
    assert llm.requests[0][0]["role"] == "system"
    assert llm.requests[0][-1] == {"role": "user", "content": "hello"}


def test_agent_includes_recent_conversation_history():
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "first answer"}),
            json.dumps({"type": "final_answer", "content": "second answer"}),
        ]
    )
    agent = Agent(llm)

    agent.run_turn("first")
    agent.run_turn("second")

    second_request = llm.requests[1]

    assert {"role": "user", "content": "first"} in second_request
    assert {"role": "assistant", "content": "first answer"} in second_request
    assert second_request[-1] == {"role": "user", "content": "second"}


def test_agent_rejects_empty_user_message():
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "unused"})])
    agent = Agent(llm)

    try:
        agent.run_turn("   ")
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected empty messages to fail")


def test_conversation_memory_trims_to_limit():
    memory = ConversationMemory(max_messages=2)

    memory.add_user_message("one")
    memory.add_assistant_message("two")
    memory.add_user_message("three")

    assert memory.messages == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


def test_agent_calls_calculator_tool_then_returns_final_answer():
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool_name": "calculator",
                    "arguments": {"expression": "(2 + 3) * 4"},
                }
            ),
            json.dumps({"type": "final_answer", "content": "The result is 20."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry)

    response = agent.run_turn("what is (2 + 3) * 4?")

    assert response == "The result is 20."
    assert agent.state.tool_calls == [{"tool_name": "calculator", "arguments": {"expression": "(2 + 3) * 4"}, "success": True}]
    assert "calculator" in agent.state.observations[0]["observation"]
    assert len(llm.requests) == 2
    assert any(message["role"] == "observation" for message in llm.requests[1])


def test_agent_prompt_shows_available_tools():
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "ok"})])
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry)

    agent.run_turn("hello")

    system_prompt = llm.requests[0][0]["content"]
    assert "Available tools" in system_prompt
    assert "calculator" in system_prompt


def test_agent_can_run_safe_shell_tool(tmp_path):
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "run_cmd", "arguments": {"command": "printf hello"}}),
            json.dumps({"type": "final_answer", "content": "The command printed hello."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(shell_tool(tmp_path))
    agent = Agent(llm, tool_registry=registry)

    response = agent.run_turn("run printf hello")

    assert response == "The command printed hello."
    assert "stdout:\nhello" in agent.state.observations[0]["observation"]


def test_agent_handles_invalid_model_json():
    llm = RecordingLLMClient(["not json"])
    agent = Agent(llm)

    response = agent.run_turn("hello")

    assert "not valid action JSON" in response
    assert agent.state.errors


def test_agent_feeds_unknown_tool_observation_back_to_model():
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "missing_tool", "arguments": {}}),
            json.dumps({"type": "final_answer", "content": "I could not use that tool."}),
        ]
    )
    agent = Agent(llm)

    response = agent.run_turn("call missing tool")

    assert response == "I could not use that tool."
    assert "Unknown tool" in agent.state.observations[0]["observation"]


def test_agent_enforces_tool_call_limit():
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1 + 1"}}),
            json.dumps({"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "2 + 2"}}),
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry, max_tool_calls_per_turn=1)

    response = agent.run_turn("keep calculating")

    assert "Tool call limit reached" in response
