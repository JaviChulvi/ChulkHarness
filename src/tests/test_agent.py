"""Tests for the Phase 1 agent loop."""

import json
from pathlib import Path

from src.core import Agent, ObservationRecord, ToolCallRecord, TurnState
from src.core.context import ContextBudget
from src.llm import LLMClient
from src.memory import ConversationMemory, SQLiteMemoryStore
from src.skills import SkillRegistry
from src.tools import Tool, ToolRegistry, apply_patch_tool, calculator_tool, list_files_tool, read_file_tool, shell_tool, write_file_tool
from src.tools.registry import ToolResult
from src.tracing import JSONLTraceLogger


class RecordingLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.responses.pop(0)


class OutputLimitRecordingLLMClient(LLMClient):
    def __init__(self) -> None:
        self.max_output_tokens: list[int | None] = []

    def _complete_action_once(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        self.max_output_tokens.append(max_output_tokens)
        return json.dumps({"type": "final_answer", "content": "ok"})


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


def test_turn_state_records_serialize():
    tool_call = ToolCallRecord(
        tool_name="calculator",
        arguments={"expression": "1 + 1"},
        iteration=1,
    )
    tool_call.success = True
    observation = ObservationRecord(
        tool_name="calculator",
        content="Tool calculator finished with success.",
        output_metadata={"success": True},
    )
    turn = TurnState(
        user_message="what is 1 + 1?",
        available_tool_names=["calculator"],
        tool_calls=[tool_call],
        observations=[observation],
    )
    turn.model_request_count = 2
    turn.tool_call_count = 1
    turn.complete("2")

    payload = turn.to_dict()

    assert payload["turn_id"]
    assert payload["status"] == "completed"
    assert payload["started_at"]
    assert payload["ended_at"]
    assert payload["model_request_count"] == 2
    assert payload["tool_call_count"] == 1
    assert payload["available_tool_names"] == ["calculator"]
    assert payload["tool_calls"][0]["tool_name"] == "calculator"
    assert payload["observations"][0]["content"] == "Tool calculator finished with success."


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
    assert agent.state.tool_calls == [
        {"tool_name": "calculator", "arguments": {"expression": "(2 + 3) * 4"}, "phase": "execution", "success": True}
    ]
    assert "calculator" in agent.state.observations[0]["observation"]
    assert len(agent.state.turns) == 1
    turn = agent.state.turns[0]
    assert turn.status == "completed"
    assert turn.final_answer == "The result is 20."
    assert turn.model_request_count == 2
    assert turn.tool_call_count == 1
    assert turn.tool_calls[0].tool_name == "calculator"
    assert turn.tool_calls[0].phase == "execution"
    assert turn.tool_calls[0].success is True
    assert turn.observations[0].tool_name == "calculator"
    assert len(llm.requests) == 2
    assert any(message["role"] == "observation" for message in llm.requests[1])


def test_agent_calls_apply_patch_tool_then_returns_final_answer(tmp_path):
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "apply_patch",
                    "arguments_json": json.dumps(
                        {
                            "patch": "\n".join(
                                [
                                    "--- a/notes.txt",
                                    "+++ b/notes.txt",
                                    "@@ -1 +1 @@",
                                    "-hello",
                                    "+hello chulk",
                                ]
                            )
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "Updated notes.txt."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(apply_patch_tool(tmp_path))
    agent = Agent(llm, tool_registry=registry)

    response = agent.run_turn("update notes")

    assert response == "Updated notes.txt."
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello chulk\n"
    assert agent.state.tool_calls == [
        {"tool_name": "apply_patch", "arguments": {"patch": "--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n-hello\n+hello chulk"}, "phase": "execution", "success": True}
    ]
    assert "Applied patch" in agent.state.observations[0]["observation"]
    assert any(message["role"] == "observation" for message in llm.requests[1])


def test_agent_prompt_shows_available_tools():
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "ok"})])
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry, max_tool_calls_per_turn=4)

    agent.run_turn("hello")

    system_prompt = llm.requests[0][0]["content"]
    assert "Available tools" in system_prompt
    assert "calculator" in system_prompt
    assert "Tool-call limit" in system_prompt
    assert "at most 4 tool calls" in system_prompt


def test_agent_records_context_report_in_state_and_trace(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "context-session")
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "ok"})])
    agent = Agent(llm, trace_logger=trace_logger)

    agent.run_turn("hello")

    turn = agent.state.turns[0]
    report = agent.state.last_context_report
    trace_text = trace_logger.path.read_text(encoding="utf-8")

    assert isinstance(report, dict)
    assert turn.context_reports == [report]
    assert report["estimated_tokens"] > 0
    assert any(section["name"] == "history" for section in report["sections"])
    assert "context_report" in trace_text
    assert "estimated_tokens" in trace_text


def test_agent_passes_remaining_context_as_output_limit():
    llm = OutputLimitRecordingLLMClient()
    agent = Agent(llm, context_budget=ContextBudget(max_prompt_tokens=3000, response_reserve_tokens=200))

    response = agent.run_turn("hello")

    report = agent.state.last_context_report
    assert response == "ok"
    assert isinstance(report, dict)
    assert llm.max_output_tokens == [3000 - report["estimated_tokens"]]


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


def test_agent_traces_full_model_request_with_redaction(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "ok"})])
    agent = Agent(
        llm,
        trace_logger=trace_logger,
        system_prompt="System prompt includes OPENAI_API_KEY=sk-testsecret123456 and normal instructions.",
        trace_max_prompt_chars=10000,
    )

    response = agent.run_turn("hello")

    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    request_event = next(event for event in events if event["type"] == "model_request_started")
    payload = request_event["payload"]
    system_content = payload["messages"][0]["content"]

    assert response == "ok"
    assert payload["message_count"] == 2
    assert payload["truncated"] is False
    assert payload["prompt_char_count"] == payload["returned_prompt_char_count"]
    assert "normal instructions" in system_content
    assert "sk-testsecret123456" not in system_content
    assert "OPENAI_API_KEY= [redacted]" in system_content
    assert payload["messages"][-1] == {
        "role": "user",
        "content": "hello",
        "content_char_count": 5,
        "returned_content_char_count": 5,
        "truncated": False,
    }


def test_agent_truncates_large_model_request_trace(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "ok"})])
    agent = Agent(
        llm,
        trace_logger=trace_logger,
        system_prompt="x" * 200,
        trace_max_prompt_chars=40,
    )

    agent.run_turn("hello")

    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    payload = next(event["payload"] for event in events if event["type"] == "model_request_started")

    assert payload["truncated"] is True
    assert payload["returned_prompt_char_count"] == 40
    assert payload["prompt_char_count"] > 40
    assert payload["messages"][0]["truncated"] is True
    assert len(payload["messages"][0]["content"]) == 40


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


def test_agent_traces_tool_call_lifecycle(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1 + 1"}}),
            json.dumps({"type": "final_answer", "content": "The result is 2."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    response = agent.run_turn("what is 1 + 1?")

    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]
    completed_payload = next(event["payload"] for event in events if event["type"] == "tool_call_completed")
    turn_finished_payload = next(event["payload"] for event in events if event["type"] == "turn_finished")

    assert response == "The result is 2."
    assert "turn_started" in event_types
    assert "tool_call_started" in event_types
    assert "tool_call_completed" in event_types
    assert "model_response_parsed" in event_types
    assert completed_payload["tool_name"] == "calculator"
    assert completed_payload["iteration"] == 1
    assert completed_payload["success"] is True
    assert turn_finished_payload["turn"]["status"] == "completed"
    assert turn_finished_payload["turn"]["model_request_count"] == 2
    assert turn_finished_payload["turn"]["tool_call_count"] == 1
    assert turn_finished_payload["turn"]["tool_calls"][0]["success"] is True


def test_agent_planned_turn_creates_pending_plan_without_running_tools(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    plan_payload = {
        "summary": "Calculate the answer safely.",
        "steps": [
            {
                "id": "1",
                "title": "Run calculator",
                "description": "Use the calculator tool for the arithmetic.",
                "status": "pending",
            }
        ],
    }
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                }
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    response = agent.run_planned_turn("what is 2 + 2?")

    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]

    assert "Plan" in response
    assert "Use /approve" in response
    assert agent.has_pending_plan() is True
    assert agent.state.tool_calls == []
    assert agent.state.turns[0].status == "waiting_for_approval"
    assert agent.state.turns[0].active_plan is not None
    assert agent.state.turns[0].active_plan.summary == "Calculate the answer safely."
    assert "Planning: requested for this turn." in llm.requests[0][0]["content"]
    assert "return a plan action" in llm.requests[0][0]["content"]
    assert "plan_created" in event_types
    assert "tool_call_started" not in event_types


def test_agent_planned_turn_allows_read_only_reconnaissance_before_plan(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "core.py").write_text("class Agent:\n    pass\n", encoding="utf-8")
    plan_payload = {
        "summary": "Add subagent support based on the inspected runtime.",
        "steps": [
            {
                "id": "1",
                "title": "Extend agent runtime",
                "description": "Use the inspected src/core.py shape to add subagent orchestration.",
                "status": "pending",
            }
        ],
    }
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "read_file",
                    "arguments_json": json.dumps({"path": "src/core.py"}),
                }
            ),
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                }
            ),
        ]
    )
    registry = ToolRegistry()
    registry.register(read_file_tool(tmp_path))
    agent = Agent(llm, tool_registry=registry)

    response = agent.run_planned_turn("How would you add subagents?")
    turn = agent.state.turns[0]

    assert "Add subagent support" in response
    assert turn.status == "waiting_for_approval"
    assert turn.tool_call_count == 1
    assert turn.tool_calls[0].tool_name == "read_file"
    assert turn.tool_calls[0].phase == "planning"
    assert "class Agent" in agent.state.observations[0]["observation"]
    assert "read-only reconnaissance tools" in llm.requests[0][0]["content"]
    assert any(message["role"] == "observation" for message in llm.requests[1])


def test_agent_planned_turn_revises_reconnaissance_only_plan(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "config.py").write_text("class Config:\n    pass\n", encoding="utf-8")
    weak_plan_payload = {
        "summary": "Explore the current codebase before designing subagents.",
        "steps": [
            {
                "id": "1",
                "title": "Read main.py",
                "description": "Understand the agent loop.",
                "status": "pending",
            },
            {
                "id": "2",
                "title": "Read config.py",
                "description": "Understand configuration patterns.",
                "status": "pending",
            },
        ],
    }
    strong_plan_payload = {
        "summary": "Implement subagent support in the inspected runtime.",
        "steps": [
            {
                "id": "1",
                "title": "Add subagent state models",
                "description": "Extend src/core/state.py with records for child task requests and results.",
                "status": "pending",
            },
            {
                "id": "2",
                "title": "Implement subagent orchestration",
                "description": "Update src/core/agent.py to spawn isolated child agents and collect results.",
                "status": "pending",
            },
        ],
    }
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "list_files",
                    "arguments_json": json.dumps({"path": "src", "pattern": "*.py"}),
                }
            ),
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(weak_plan_payload),
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "read_file",
                    "arguments_json": json.dumps({"path": "src/main.py"}),
                }
            ),
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(strong_plan_payload),
                }
            ),
        ]
    )
    registry = ToolRegistry()
    registry.register(list_files_tool(tmp_path))
    registry.register(read_file_tool(tmp_path))
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    response = agent.run_planned_turn("How would you add subagent functionality?")
    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]
    turn = agent.state.turns[0]

    assert "Add subagent state models" in response
    assert "Read main.py" not in response
    assert turn.status == "waiting_for_approval"
    assert turn.tool_call_count == 2
    assert [tool_call.phase for tool_call in turn.tool_calls] == ["planning", "planning"]
    assert turn.planning_feedback_count == 1
    assert "plan_revision_requested" in event_types
    assert any(observation.tool_name == "planning_feedback" for observation in turn.observations)


def test_agent_planned_turn_revises_direct_answer_into_plan():
    plan_payload = {
        "summary": "Add subagent delegation support.",
        "steps": [
            {
                "id": "1",
                "title": "Add subagent action type",
                "description": "Extend src/core/actions.py with a delegation action for child-agent work.",
                "status": "pending",
            },
            {
                "id": "2",
                "title": "Implement delegation runtime",
                "description": "Update src/core/agent.py to create child agents and return their observations.",
                "status": "pending",
            },
        ],
    }
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "Here is how I would add subagents conceptually."}),
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                }
            ),
        ]
    )
    agent = Agent(llm)

    response = agent.run_planned_turn("How would you add subagents?")
    turn = agent.state.turns[0]

    assert "Add subagent action type" in response
    assert "conceptually" not in response
    assert turn.status == "waiting_for_approval"
    assert turn.planning_feedback_count == 1
    assert turn.observations[0].tool_name == "planning_feedback"
    assert "do not answer directly" in turn.observations[0].content


def test_agent_planned_turn_requests_plan_when_reconnaissance_budget_is_exhausted(tmp_path):
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 2\n", encoding="utf-8")
    plan_payload = {
        "summary": "Implement the feature with the gathered context.",
        "steps": [
            {
                "id": "1",
                "title": "Update agent runtime",
                "description": "Modify src/core/agent.py using the files already inspected.",
                "status": "pending",
            }
        ],
    }
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "read_file",
                    "arguments_json": json.dumps({"path": "a.py"}),
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "read_file",
                    "arguments_json": json.dumps({"path": "b.py"}),
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "read_file",
                    "arguments_json": json.dumps({"path": "c.py"}),
                }
            ),
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                }
            ),
        ]
    )
    registry = ToolRegistry()
    registry.register(read_file_tool(tmp_path))
    agent = Agent(llm, tool_registry=registry, max_tool_calls_per_turn=2)

    response = agent.run_planned_turn("Plan the feature")
    turn = agent.state.turns[0]

    assert "Update agent runtime" in response
    assert turn.status == "waiting_for_approval"
    assert turn.tool_call_count == 2
    assert turn.planning_tool_limit_feedback_sent is True
    assert turn.planning_feedback_count == 1
    assert "reconnaissance tool budget is exhausted" in turn.observations[-1].content


def test_agent_planned_turn_blocks_mutating_tools_before_plan(tmp_path):
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "write_file",
                    "arguments_json": json.dumps({"path": "created.txt", "content": "nope"}),
                }
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(write_file_tool(tmp_path))
    agent = Agent(llm, tool_registry=registry)

    response = agent.run_planned_turn("Create a file")

    assert "Planning can only use read-only reconnaissance tools before approval" in response
    assert not (tmp_path / "created.txt").exists()
    assert agent.state.turns[0].status == "failed"
    assert agent.state.turns[0].tool_call_count == 0


def test_agent_run_planned_turn_forces_plan_for_one_turn():
    plan_payload = {
        "summary": "Plan a design change.",
        "steps": [
            {
                "id": "1",
                "title": "Add subagent task model",
                "description": "Create a state record for delegated child-agent work.",
                "status": "pending",
            }
        ],
    }
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                }
            ),
            json.dumps({"type": "final_answer", "content": "Plan approved and completed."}),
        ]
    )
    agent = Agent(llm)

    response = agent.run_planned_turn("How would you add subagents?")
    approved_response = agent.approve_plan()

    assert "Use /approve" in response
    assert approved_response == "Plan approved and completed."
    assert "Planning: requested for this turn." in llm.requests[0][0]["content"]
    assert "Planning: approved for this turn." in llm.requests[1][0]["content"]


def test_agent_approve_plan_resumes_turn_and_tracks_plan_steps(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    plan_payload = {
        "summary": "Calculate the answer safely.",
        "steps": [
            {
                "id": "1",
                "title": "Run calculator",
                "description": "Use the calculator tool for the arithmetic.",
                "status": "pending",
            }
        ],
    }
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "calculator",
                    "arguments_json": json.dumps({"expression": "2 + 2"}),
                }
            ),
            json.dumps({"type": "final_answer", "content": "The result is 4."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    agent.run_planned_turn("what is 2 + 2?")
    response = agent.approve_plan()

    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]
    turn = agent.state.turns[0]

    assert response == "The result is 4."
    assert agent.has_pending_plan() is False
    assert agent.state.active_plan is None
    assert turn.status == "completed"
    assert turn.plan_approved is True
    assert turn.active_plan is not None
    assert turn.active_plan.steps[0].status == "completed"
    assert turn.model_request_count == 3
    assert turn.tool_call_count == 1
    assert "Planning: approved for this turn." in llm.requests[1][0]["content"]
    assert "plan_approved" in event_types
    assert "plan_step_started" in event_types
    assert "plan_step_completed" in event_types
    assert "turn_finished" in event_types


def test_agent_reject_plan_finishes_without_tools(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    plan_payload = {
        "summary": "Add project inspection support.",
        "steps": [
            {
                "id": "1",
                "title": "Add file inspection workflow",
                "description": "Implement a project inspection path using the existing file tools.",
                "status": "pending",
            }
        ],
    }
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                }
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    agent.run_planned_turn("inspect the project")
    response = agent.reject_plan()

    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]
    turn = agent.state.turns[0]

    assert response == "Plan rejected. No tools were run."
    assert agent.has_pending_plan() is False
    assert agent.state.tool_calls == []
    assert turn.status == "plan_rejected"
    assert turn.active_plan is not None
    assert turn.active_plan.status() == "rejected"
    assert "plan_rejected" in event_types
    assert "tool_call_started" not in event_types


def test_agent_truncates_tool_output_but_preserves_full_artifact(tmp_path):
    full_stdout = "HEAD-" + ("middle-" * 200) + "IMPORTANT_TAIL"

    def big_output_tool(_arguments):
        return ToolResult(
            tool_name="big_output",
            success=True,
            observation="Produced a long output.",
            stdout=full_stdout,
        )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="big_output",
            description="Return long output for testing.",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=big_output_tool,
        )
    )
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "big_output", "arguments": {}}),
            json.dumps({"type": "final_answer", "content": "Reviewed."}),
        ]
    )
    agent = Agent(
        llm,
        tool_registry=registry,
        trace_logger=trace_logger,
        max_tool_stdout_chars=120,
        max_observation_chars=1000,
    )

    response = agent.run_turn("produce long output")

    observation = agent.state.observations[0]["observation"]
    output_metadata = agent.state.observations[0]["output_metadata"]
    stdout_artifact = next(artifact for artifact in output_metadata["artifacts"] if artifact["field"] == "stdout")
    artifact_text = Path(stdout_artifact["path"]).read_text(encoding="utf-8")
    trace_text = trace_logger.path.read_text(encoding="utf-8")

    assert response == "Reviewed."
    assert "HEAD-" in observation
    assert "IMPORTANT_TAIL" in observation
    assert "full stdout saved" in observation
    assert output_metadata["stdout"]["truncated"] is True
    assert artifact_text == full_stdout
    assert "IMPORTANT_TAIL" in artifact_text
    assert "tool_observation" in trace_text
    assert stdout_artifact["sha256"] == output_metadata["stdout"]["sha256"]


def test_agent_preserves_artifact_when_full_observation_is_truncated(tmp_path):
    long_observation = "OBS_HEAD-" + ("obs-middle-" * 300) + "OBS_TAIL"
    full_stdout = "STDOUT_HEAD-" + ("stdout-middle-" * 200) + "STDOUT_TAIL"

    def verbose_tool(_arguments):
        return ToolResult(
            tool_name="verbose",
            success=True,
            observation=long_observation,
            stdout=full_stdout,
        )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="verbose",
            description="Return verbose output for testing.",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=verbose_tool,
        )
    )
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "verbose", "arguments": {}}),
            json.dumps({"type": "final_answer", "content": "Reviewed."}),
        ]
    )
    agent = Agent(
        llm,
        tool_registry=registry,
        trace_logger=trace_logger,
        max_tool_stdout_chars=120,
        max_observation_chars=500,
    )

    agent.run_turn("produce verbose output")

    observation = agent.state.observations[0]["observation"]
    output_metadata = agent.state.observations[0]["output_metadata"]
    observation_artifact = next(
        artifact for artifact in output_metadata["artifacts"] if artifact["field"] == "observation"
    )
    artifact_text = Path(observation_artifact["path"]).read_text(encoding="utf-8")

    assert len(observation) <= 500
    assert "full observation saved" in observation
    assert output_metadata["observation"]["truncated"] is True
    assert "OBS_TAIL" in artifact_text
    assert "full stdout saved" in artifact_text


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


def test_agent_feeds_invalid_tool_arguments_back_to_model(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": 123}}),
            json.dumps({"type": "final_answer", "content": "I corrected the tool arguments issue."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    response = agent.run_turn("calculate this")

    observation = agent.state.observations[0]["observation"]
    output_metadata = agent.state.observations[0]["output_metadata"]
    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    failed_payload = next(event["payload"] for event in events if event["type"] == "tool_call_failed")

    assert response == "I corrected the tool arguments issue."
    assert "failed before execution" in observation
    assert "expression: value has the wrong type" in observation
    assert "Expected: string" in observation
    assert output_metadata["success"] is False
    assert failed_payload["error"] == "invalid_arguments"
    assert failed_payload["metadata"]["validation_errors"][0]["path"] == "expression"
    assert any(message["role"] == "observation" and "value has the wrong type" in message["content"] for message in llm.requests[1])


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
    assert agent.state.turns[0].status == "failed"
    assert agent.state.turns[0].tool_call_count == 1
    assert "Tool call limit reached" in agent.state.turns[0].errors[0]
