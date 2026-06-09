"""Tests for the Phase 1 agent loop."""

from src.core import Agent
from src.memory import ConversationMemory


class RecordingLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.responses.pop(0)


def test_agent_sends_user_message_and_stores_response():
    llm = RecordingLLMClient(["Hi Javier"])
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
    llm = RecordingLLMClient(["first answer", "second answer"])
    agent = Agent(llm)

    agent.run_turn("first")
    agent.run_turn("second")

    second_request = llm.requests[1]

    assert {"role": "user", "content": "first"} in second_request
    assert {"role": "assistant", "content": "first answer"} in second_request
    assert second_request[-1] == {"role": "user", "content": "second"}


def test_agent_rejects_empty_user_message():
    llm = RecordingLLMClient(["unused"])
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
