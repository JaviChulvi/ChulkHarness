"""Tests for the Phase 1 agent loop."""

import json

from src.core import Agent
from src.llm import LLMClient
from src.memory import ConversationMemory, SQLiteMemoryStore
from src.skills import SkillRegistry
from src.tools import ToolRegistry, calculator_tool, shell_tool
from src.tracing import JSONLTraceLogger


class RecordingLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.responses.pop(0)


def create_test_skill_registry(tmp_path):
    skills_dir = tmp_path / "skills"
    for name, description in {
        "shell": "Use this skill when the user request requires terminal inspection or command execution.",
        "memory": "Use this skill when the user request involves saving or retrieving durable information.",
        "files": "Use this skill when the user request requires reading, editing, or creating files.",
    }.items():
        skill_dir = skills_dir / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"# {name.title()} Skill\n\n{description}\n\nGuidelines:\n- Keep the work inspectable.\n",
            encoding="utf-8",
        )
    registry = SkillRegistry(skills_dir)
    registry.load_metadata()
    return registry


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
                    "content": None,
                    "tool_name": "calculator",
                    "arguments_json": json.dumps({"expression": "(2 + 3) * 4"}),
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


def test_agent_injects_profile_and_relevant_long_term_memories(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    profile_id = memory_store.save_memory(
        "Javier prefers exact file paths and direct implementation steps.",
        tags=["persona", "preference"],
        importance=9,
    )
    relevant_id = memory_store.save_memory(
        "ChulkHarness long-term memory is backed by SQLite.",
        tags=["project", "memory"],
        importance=5,
    )
    unrelated_id = memory_store.save_memory("The shell skill explains safe command usage.", tags=["skill"], importance=10)
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "SQLite memory is configured."})])
    agent = Agent(llm, memory_store=memory_store)

    response = agent.run_turn("How does SQLite memory work in ChulkHarness?")

    system_prompt = llm.requests[0][0]["content"]
    assert response == "SQLite memory is configured."
    assert profile_id in agent.state.loaded_memory_ids
    assert relevant_id in agent.state.loaded_memory_ids
    assert unrelated_id not in agent.state.loaded_memory_ids
    assert "Persona and workflow preferences" in system_prompt
    assert "Javier prefers exact file paths" in system_prompt
    assert "Relevant contextual memories" in system_prompt
    assert "SQLite" in system_prompt
    assert "It is not a skill, a tool, or an instruction playbook" in system_prompt


def test_agent_extracts_explicit_memories_and_writes_memory_trace_events(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "I will remember that."})])
    agent = Agent(llm, memory_store=memory_store, trace_logger=trace_logger)

    response = agent.run_turn("Please remember that ChulkHarness e2e memory uses SQLite.")

    trace_text = trace_logger.path.read_text(encoding="utf-8")
    saved_memory = memory_store.search_memory("e2e SQLite")[0]

    assert response == "I will remember that."
    assert agent.state.extracted_memory_ids == [saved_memory.id]
    assert saved_memory.id in agent.state.loaded_memory_ids
    assert "memory_extraction_completed" in trace_text
    assert "memory_search_started" in trace_text
    assert "memory_search_completed" in trace_text
    assert saved_memory.id in trace_text


def test_agent_injects_relevant_skill_without_loading_unrelated_skills(tmp_path):
    skill_registry = create_test_skill_registry(tmp_path)
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "I can run that command."})])
    agent = Agent(llm, skill_registry=skill_registry, trace_logger=trace_logger)

    response = agent.run_turn("run a shell command to print hello")

    system_prompt = llm.requests[0][0]["content"]
    trace_text = trace_logger.path.read_text(encoding="utf-8")

    assert response == "I can run that command."
    assert agent.state.loaded_skill_names == ["shell"]
    assert "Loaded skills are procedural instructions" in system_prompt
    assert "Skill: shell" in system_prompt
    assert "# Shell Skill" in system_prompt
    assert "Skill: memory" not in system_prompt
    assert skill_registry.get_skill("shell").loaded_content is not None
    assert skill_registry.get_skill("memory").loaded_content is None
    assert "skill_selection_completed" in trace_text
    assert "shell" in trace_text


def test_agent_selects_memory_and_file_skills_for_matching_requests(tmp_path):
    skill_registry = create_test_skill_registry(tmp_path)
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "Memory skill selected."}),
            json.dumps({"type": "final_answer", "content": "Files skill selected."}),
        ]
    )
    agent = Agent(llm, skill_registry=skill_registry)

    memory_response = agent.run_turn("please remember this durable project fact")
    memory_prompt = llm.requests[0][0]["content"]
    file_response = agent.run_turn("edit the README file")
    file_prompt = llm.requests[1][0]["content"]

    assert memory_response == "Memory skill selected."
    assert "Skill: memory" in memory_prompt
    assert agent.state.loaded_skill_names == ["files"]
    assert file_response == "Files skill selected."
    assert "Skill: files" in file_prompt
    assert "Skill: shell" not in file_prompt


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


def test_agent_repairs_invalid_model_json():
    llm = RecordingLLMClient(
        [
            "Claro, puedo ayudarte.",
            json.dumps({"type": "final_answer", "content": "Claro, puedo ayudarte."}),
        ]
    )
    agent = Agent(llm)

    response = agent.run_turn("hello")

    assert response == "Claro, puedo ayudarte."
    assert agent.state.json_repair_attempts == 1
    assert "JSON repair attempt" in agent.state.errors[0]
    assert llm.requests[1][-1]["role"] == "user"
    assert "could not be parsed" in llm.requests[1][-1]["content"]


def test_agent_fails_after_json_repair_limit():
    llm = RecordingLLMClient(["not json", "still not json"])
    agent = Agent(llm, max_json_repair_attempts=1)

    response = agent.run_turn("hello")

    assert "not valid action JSON" in response
    assert agent.state.json_repair_attempts == 1
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
