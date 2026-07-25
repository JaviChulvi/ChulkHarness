"""Tests for the Phase 1 agent loop."""

import asyncio
from decimal import Decimal
import json
from chulk.core import Agent, ObservationRecord, Plan, PlanStep, ToolCallRecord, TraceEvent, TurnContextSection, TurnState
from chulk.core.actions import FinalAnswerAction, PlanAction, PlanStepUpdateAction
from chulk.core.context import ContextBudget
from chulk.llm import (
    FallbackChain,
    LLMActionError,
    LLMActionResult,
    LLMCapabilities,
    LLMClient,
    LLMCost,
    LLMError,
    LLMResponse,
    LLMUsage,
)
from chulk.mcp import MCPServerConfig, create_mcp_bridge_tools
from chulk.memory import ConversationMemory, SQLiteMemoryStore
from chulk.skills import SkillRegistry
from chulk.tools import (
    Tool,
    ToolExecutionContext,
    ToolFailureKind,
    ToolRegistry,
    apply_patch_tool,
    calculator_tool,
    list_files_tool,
    read_file_tool,
    shell_tool,
    write_file_tool,
)
from chulk.tools.permissions import PermissionDecision, ToolPermissionLevel, ToolPermissionPolicy
from chulk.tools.registry import ToolResult
from chulk.tracing import JSONLTraceLogger


class RecordingLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.responses.pop(0)


class StreamingRecordingLLMClient(RecordingLLMClient):
    capabilities = LLMCapabilities(supports_streaming=True)


class NativeActionRecordingLLMClient(LLMClient):
    capabilities = LLMCapabilities(supports_native_tool_calling=True)

    def __init__(self) -> None:
        self.requests: list[list[dict[str, str]]] = []
        self.tool_batches: list[list[object] | None] = []

    def complete_action(
        self,
        messages: list[dict[str, str]],
        *,
        max_repair_attempts: int = 2,
        max_output_tokens: int | None = None,
        tools: list[object] | None = None,
        **kwargs,
    ) -> LLMActionResult:
        self.requests.append(messages)
        self.tool_batches.append(tools)
        return LLMActionResult(
            action=FinalAnswerAction(type="final_answer", content="native ok"),
            raw_response='{"type":"final_answer","content":"native ok"}',
            metadata={"action_transport": "provider_native"},
        )


class PricedRecordingLLMClient(RecordingLLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"


class OutputLimitRecordingLLMClient(LLMClient):
    def __init__(self) -> None:
        self.action_kwargs: list[dict] = []

    def _complete_action_once(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.action_kwargs.append(kwargs)
        return json.dumps({"type": "final_answer", "content": "ok"})


class ChargedFailureActionLLMClient(LLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"

    def _complete_action_response_once(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        cost = LLMCost(amount=Decimal("0.000012"), pricing_known=True, provider=self.provider, model=self.model)
        raise LLMActionError(
            "provider charged then failed",
            usage=usage,
            cost=cost,
            code="server_error",
            retryable=True,
            fallback_eligible=True,
        )


class ChargedSuccessActionLLMClient(LLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"

    def _complete_action_response_once(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        usage = LLMUsage(input_tokens=20, output_tokens=10, total_tokens=30)
        cost = LLMCost(amount=Decimal("0.000024"), pricing_known=True, provider=self.provider, model=self.model)
        return LLMResponse(
            content=json.dumps({"type": "final_answer", "content": "fallback ok"}),
            usage=usage,
            cost=cost,
            provider=self.provider,
            model=self.model,
        )


class RecordingMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self):
        return [
            {
                "name": "search_docs",
                "description": "Search docs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

    def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return ToolResult("mcp_docs_search_docs", True, "called")


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
    turn.reflection_count = 1
    turn.reflections.append({"attempt": 1, "approved": True, "reason": "ok", "feedback": None})
    turn.complete("2")

    payload = turn.to_dict()

    assert payload["turn_id"]
    assert payload["status"] == "completed"
    assert payload["started_at"]
    assert payload["ended_at"]
    assert payload["model_request_count"] == 2
    assert payload["tool_call_count"] == 1
    assert payload["reflection_count"] == 1
    assert payload["reflections"][0]["approved"] is True
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


def test_agent_streams_validated_final_answer_when_provider_supports_streaming():
    llm = StreamingRecordingLLMClient([json.dumps({"type": "final_answer", "content": "Hello streamed world"})])
    events: list[tuple[str, dict]] = []
    agent = Agent(llm, event_callback=lambda event_type, payload: events.append((event_type, payload)))

    response = agent.run_turn("hello")

    assert response == "Hello streamed world"
    assert agent.state.final_answer == "Hello streamed world"
    stream_events = [event for event in events if event[0].startswith("model_stream_")]
    assert [event[0] for event in stream_events] == [
        TraceEvent.MODEL_STREAM_STARTED,
        TraceEvent.MODEL_STREAM_DELTA,
        TraceEvent.MODEL_STREAM_COMPLETED,
    ]
    assert "".join(event[1].get("text", "") for event in stream_events) == "Hello streamed world"


def test_agent_does_not_stream_final_answer_when_provider_lacks_streaming_capability():
    llm = RecordingLLMClient([json.dumps({"type": "final_answer", "content": "Hi Javier"})])
    events: list[tuple[str, dict]] = []
    agent = Agent(llm, event_callback=lambda event_type, payload: events.append((event_type, payload)))

    response = agent.run_turn("hello")

    assert response == "Hi Javier"
    assert not [event for event in events if event[0].startswith("model_stream_")]


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


def test_low_history_limit_keeps_current_user_and_complete_tool_result():
    llm = RecordingLLMClient(
        [
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
    agent = Agent(
        llm,
        memory=ConversationMemory(max_messages=1),
        tool_registry=registry,
    )

    assert agent.run_turn("Calculate 2 + 2.") == "The result is 4."

    follow_up_history = llm.requests[1][1:]
    assert [message["role"] for message in follow_up_history] == [
        "user",
        "assistant",
        "observation",
    ]
    assert follow_up_history[0]["content"] == "Calculate 2 + 2."
    assert follow_up_history[1]["content"].startswith("<executed_tool_action>")
    assert "Tool calculator finished with success." in follow_up_history[2]["content"]


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


def test_agent_reflection_approves_final_answer():
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "Hi Javier"}),
            json.dumps({"approved": True, "reason": "The answer addresses the greeting.", "feedback": None}),
        ]
    )
    agent = Agent(llm, max_reflection_attempts=1)

    response = agent.run_turn("hello")

    turn = agent.state.turns[0]
    assert response == "Hi Javier"
    assert turn.reflection_count == 1
    assert turn.reflections[0]["approved"] is True
    assert turn.model_request_count == 2
    assert "final-answer reviewer" in llm.requests[1][0]["content"]


def test_agent_reflection_feedback_triggers_one_more_action_loop(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "Done."}),
            json.dumps(
                {
                    "approved": False,
                    "reason": "The answer claims completion without mentioning the missing tool evidence.",
                    "feedback": "Explain that no tool evidence was gathered before answering.",
                }
            ),
            json.dumps({"type": "final_answer", "content": "I do not have tool evidence for completion."}),
        ]
    )
    agent = Agent(llm, trace_logger=trace_logger, max_reflection_attempts=1)

    response = agent.run_turn("did you inspect the file?")

    turn = agent.state.turns[0]
    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]

    assert response == "I do not have tool evidence for completion."
    assert turn.reflection_count == 1
    assert turn.model_request_count == 3
    assert turn.observations[-1].tool_name == "reflection_feedback"
    assert "no tool evidence" in turn.observations[-1].content
    assert any(message["role"] == "observation" and "Reflection feedback" in message["content"] for message in llm.requests[2])
    assert any(
        event["type"] == "tool_observation" and event["payload"]["tool_name"] == "reflection_feedback"
        for event in events
    )
    assert "reflection_started" in event_types
    assert "reflection_completed" in event_types
    assert "reflection_revision_requested" in event_types


def test_agent_reflection_attempt_limit_prevents_revision_loop():
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "First answer."}),
            json.dumps(
                {
                    "approved": False,
                    "reason": "Needs one more pass.",
                    "feedback": "Return a revised final answer.",
                }
            ),
            json.dumps({"type": "final_answer", "content": "Second answer."}),
        ]
    )
    agent = Agent(llm, max_reflection_attempts=1)

    response = agent.run_turn("answer carefully")

    turn = agent.state.turns[0]
    assert response == "Second answer."
    assert turn.reflection_count == 1
    assert len(turn.reflections) == 1
    assert len(llm.requests) == 3


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
    assert "<tools>" in system_prompt
    assert "<available_tools>" in system_prompt
    assert "<tool>" in system_prompt
    assert "<name>calculator</name>" in system_prompt
    assert "<arguments_schema_json>" in system_prompt
    assert "Available tools" in system_prompt
    assert "calculator" in system_prompt
    assert "<tool_call_rules>" in system_prompt
    assert "<max_tool_calls_per_turn>4</max_tool_calls_per_turn>" in system_prompt
    assert "Tool-call limit" in system_prompt
    assert "at most 4 tool calls" in system_prompt


def test_agent_uses_native_action_prompt_and_passes_tool_specs_when_supported(tmp_path):
    llm = NativeActionRecordingLLMClient()
    registry = ToolRegistry()
    registry.register(calculator_tool())
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "native-context")
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    response = agent.run_turn("what is 1 + 1?")

    system_prompt = llm.requests[0][0]["content"]
    assert response == "native ok"
    assert "provider-native tool interface" in system_prompt
    assert "You must respond with exactly one JSON object" not in system_prompt
    assert "<name>calculator</name>" not in system_prompt
    assert "<arguments_schema_json>" not in system_prompt
    assert llm.tool_batches[0] is not None
    assert [tool.name for tool in llm.tool_batches[0]] == ["calculator"]
    events = [
        json.loads(line)
        for line in trace_logger.path.read_text(encoding="utf-8").splitlines()
    ]
    request_payload = next(
        event["payload"]
        for event in events
        if event["type"] == TraceEvent.MODEL_REQUEST_STARTED
    )
    assert request_payload["action_transport"] == "provider_native"
    assert request_payload["native_tool_names"] == ["calculator"]
    declarations = request_payload["native_tool_declarations"]
    assert declarations["truncated"] is False
    assert declarations["items"][0]["parameters"]["type"] == "object"


def test_hosted_mcp_is_visible_in_native_context_without_tracing_authorization(tmp_path):
    class HostedNativeClient(NativeActionRecordingLLMClient):
        capabilities = LLMCapabilities(
            supports_native_tool_calling=True,
            supports_hosted_mcp_tools=True,
        )

    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
        authorization="secret-token",
    )
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "hosted-mcp-context")
    agent = Agent(
        HostedNativeClient(),
        mcp_servers=(server,),
        trace_logger=trace_logger,
    )

    assert agent.run_turn("search docs") == "native ok"

    report = agent.state.last_context_report
    native_section = next(
        section for section in report["sections"] if section["name"] == "native_tools"
    )
    assert native_section["metadata"]["tool_names"] == ["mcp:docs"]
    assert native_section["item_count"] == 1
    assert report["request_overhead_estimated_tokens"] > 0
    trace_text = trace_logger.path.read_text(encoding="utf-8")
    assert '"name": "mcp:docs"' in trace_text
    assert "secret-token" not in trace_text
    events = [json.loads(line) for line in trace_text.splitlines()]
    request_payload = next(
        event["payload"]
        for event in events
        if event["type"] == TraceEvent.MODEL_REQUEST_STARTED
    )
    assert request_payload["hosted_mcp_enabled"] is True
    assert request_payload["hosted_mcp_server_labels"] == ["docs"]


def test_agent_uses_one_json_contract_for_a_mixed_fallback_chain():
    class NativeFailingClient(LLMClient):
        capabilities = LLMCapabilities(
            supports_native_tool_calling=True,
            supports_hosted_mcp_tools=True,
        )
        provider = "native-primary"
        model = "native-model"

        def __init__(self) -> None:
            self.requests: list[list[dict[str, str]]] = []
            self.native_options: list[dict] = []

        def _complete_action_response_once(
            self,
            messages,
            *,
            tools=None,
            planning_tools=None,
            hosted_mcp_servers=None,
            mcp_approval_callback=None,
            **kwargs,
        ):
            self.requests.append(messages)
            self.native_options.append(
                {
                    "tools": tools,
                    "planning_tools": planning_tools,
                    "hosted_mcp_servers": hosted_mcp_servers,
                    "mcp_approval_callback": mcp_approval_callback,
                }
            )
            raise LLMError(
                "primary unavailable",
                code="server_error",
                retryable=True,
                fallback_eligible=True,
            )

    primary = NativeFailingClient()
    plan_payload = {
        "summary": "Use the JSON transport.",
        "steps": [
            {
                "id": "1",
                "title": "Complete the change",
                "description": "Complete the requested change.",
            }
        ],
    }
    secondary = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                    "step_update_json": "{}",
                }
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
    )
    bridge_tool = create_mcp_bridge_tools(
        [server],
        client_factory=lambda _server: RecordingMCPClient(),
    )[0]
    registry.register(bridge_tool)
    agent = Agent(
        FallbackChain([primary, secondary]),
        tool_registry=registry,
        mcp_servers=(server,),
        mcp_bridge_tool_names=[bridge_tool.name],
    )

    response = agent.run_planned_turn("hello")

    assert "Use /approve" in response
    assert primary.native_options == [
        {
            "tools": None,
            "planning_tools": None,
            "hosted_mcp_servers": None,
            "mcp_approval_callback": None,
        }
    ]
    for request in [primary.requests[0], secondary.requests[0]]:
        system_prompt = request[0]["content"]
        assert "<transport>json_object</transport>" in system_prompt
        assert "provider_native_tool_calling" not in system_prompt
        assert system_prompt.count("<name>calculator</name>") == 1
        assert system_prompt.count("<arguments_schema_json>") == 1
    assert agent.state.last_context_report["request_overhead_estimated_tokens"] == 0


def test_agent_async_keeps_native_options_off_for_a_mixed_fallback_chain():
    class AsyncNativeFailingClient(LLMClient):
        capabilities = LLMCapabilities(
            supports_native_tool_calling=True,
            supports_hosted_mcp_tools=True,
        )
        provider = "async-native-primary"
        model = "async-native-model"

        def __init__(self) -> None:
            self.requests: list[list[dict[str, str]]] = []
            self.native_options: list[dict] = []

        async def _acomplete_action_response_once(
            self,
            messages,
            *,
            tools=None,
            planning_tools=None,
            hosted_mcp_servers=None,
            mcp_approval_callback=None,
            **kwargs,
        ):
            self.requests.append(messages)
            self.native_options.append(
                {
                    "tools": tools,
                    "planning_tools": planning_tools,
                    "hosted_mcp_servers": hosted_mcp_servers,
                    "mcp_approval_callback": mcp_approval_callback,
                }
            )
            raise LLMError(
                "primary unavailable",
                code="server_error",
                retryable=True,
                fallback_eligible=True,
            )

    primary = AsyncNativeFailingClient()
    plan_payload = {
        "summary": "Use the async JSON transport.",
        "steps": [
            {
                "id": "1",
                "title": "Complete the change",
                "description": "Complete the requested change.",
            }
        ],
    }
    secondary = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan_payload),
                    "step_update_json": "{}",
                }
            )
        ]
    )
    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
    )
    registry = ToolRegistry()
    bridge_tool = create_mcp_bridge_tools(
        [server],
        client_factory=lambda _server: RecordingMCPClient(),
    )[0]
    registry.register(bridge_tool)
    agent = Agent(
        FallbackChain([primary, secondary]),
        tool_registry=registry,
        mcp_servers=(server,),
        mcp_bridge_tool_names=[bridge_tool.name],
    )

    response = asyncio.run(agent.run_planned_turn_async("hello"))

    assert "Use /approve" in response
    assert primary.native_options == [
        {
            "tools": None,
            "planning_tools": None,
            "hosted_mcp_servers": None,
            "mcp_approval_callback": None,
        }
    ]
    for request in [primary.requests[0], secondary.requests[0]]:
        assert "<transport>json_object</transport>" in request[0]["content"]


def test_native_planning_tools_follow_the_plan_lifecycle():
    class PlanningPolicyRecordingClient(LLMClient):
        capabilities = LLMCapabilities(supports_native_tool_calling=True)

        def __init__(self) -> None:
            self.actions = [
                PlanAction(
                    type="plan",
                    plan=Plan(
                        summary="Make one change.",
                        steps=[
                            PlanStep(
                                id="1",
                                title="Make the change",
                                description="Complete the requested change.",
                            )
                        ],
                    ),
                ),
                PlanStepUpdateAction(
                    type="plan_step_update",
                    step_id="1",
                    status="completed",
                    evidence="The change is complete.",
                ),
                FinalAnswerAction(type="final_answer", content="done"),
            ]
            self.planning_policies = []
            self.action_schemas = []

        def complete_action(
            self,
            messages,
            *,
            planning_tools=None,
            action_schema=None,
            **kwargs,
        ):
            self.planning_policies.append(planning_tools)
            self.action_schemas.append(action_schema)
            return LLMActionResult(
                action=self.actions.pop(0),
                raw_response='{"type":"recorded"}',
                metadata={"action_transport": "provider_native"},
            )

    llm = PlanningPolicyRecordingClient()
    agent = Agent(llm)

    plan_text = agent.run_planned_turn("make a change")
    response = agent.approve_plan()

    assert "Use /approve" in plan_text
    assert response == "done"
    assert [
        (policy.propose_plan, policy.update_plan_step)
        for policy in llm.planning_policies
    ] == [(True, False), (False, True), (False, False)]
    assert [
        schema["properties"]["type"]["enum"]
        for schema in llm.action_schemas
    ] == [["plan"], ["plan_step_update"], ["final_answer"]]


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


def test_agent_accepts_external_context_and_prompt_metadata(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "context-session")

    class ContextAwareLLM(RecordingLLMClient):
        def complete(self, messages):
            system_prompt = messages[0]["content"]
            assert "Quarterly planning source" in system_prompt
            assert "<profile>polp-search</profile>" in system_prompt
            assert "<locale>es-ES</locale>" in system_prompt
            return json.dumps({"type": "final_answer", "content": "used source"})

    agent = Agent(ContextAwareLLM([]), trace_logger=trace_logger)

    response = agent.run_turn(
        "answer from source",
        context_sections=[
            TurnContextSection(
                id="src-42",
                title="Quarterly planning source",
                content="Q3 launch is blocked by data migration.",
            )
        ],
        prompt_profile="polp-search",
        locale="es-ES",
        extension_metadata={"confidence": 0.81},
    )

    turn = agent.state.turns[0]
    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]

    assert response == "used source"
    assert turn.context_sections[0].id == "src-42"
    assert turn.prompt_profile == "polp-search"
    assert turn.locale == "es-ES"
    assert turn.extension_metadata == {"confidence": 0.81}
    assert any(event["type"] == TraceEvent.TURN_CONTEXT_SELECTED for event in events)
    assert turn.context_reports[0]["sections"][1]["name"] == "prompt_metadata"
    assert any(section["name"] == "external_context" for section in turn.context_reports[0]["sections"])


def test_agent_records_model_usage_and_cost_in_state_and_trace(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "usage-session")
    llm = PricedRecordingLLMClient([json.dumps({"type": "final_answer", "content": "ok"})])
    agent = Agent(llm, trace_logger=trace_logger)

    response = agent.run_turn("hello")

    turn = agent.state.turns[0]
    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    response_payload = next(event["payload"] for event in events if event["type"] == TraceEvent.MODEL_RESPONSE)
    finished_payload = next(event["payload"] for event in events if event["type"] == TraceEvent.TURN_FINISHED)

    assert response == "ok"
    assert len(turn.model_usage_reports) == 1
    assert turn.model_usage_totals["request_count"] == 1
    assert turn.model_usage_totals["usage"]["estimated"] is True
    assert turn.model_usage_totals["usage"]["total_tokens"] > 0
    assert turn.model_usage_totals["cost"]["pricing_known"] is True
    assert turn.model_usage_totals["cost"]["amount"] is not None
    assert agent.state.last_usage_report == turn.model_usage_totals
    assert response_payload["usage"]["estimated"] is True
    assert "total_tokens" in response_payload["usage"]
    assert response_payload["cost"]["pricing_known"] is True
    assert finished_payload["turn"]["model_usage_totals"]["request_count"] == 1
    assert finished_payload["agent_state"]["last_usage_report"]["request_count"] == 1
    assert finished_payload["agent_state"]["last_usage_report"]["cost"]["pricing_known"] is True


def test_agent_usage_totals_include_charged_failed_fallback_attempts():
    agent = Agent(FallbackChain([ChargedFailureActionLLMClient(), ChargedSuccessActionLLMClient()]))

    response = agent.run_turn("hello")

    turn = agent.state.turns[0]
    assert response == "fallback ok"
    assert turn.model_usage_totals["usage"]["input_tokens"] == 30
    assert turn.model_usage_totals["usage"]["output_tokens"] == 15
    assert turn.model_usage_totals["usage"]["total_tokens"] == 45
    assert turn.model_usage_totals["cost"]["amount"] == "0.000036"
    assert len(turn.model_usage_reports[0]["fallback_attempts"]) == 2
    assert turn.model_usage_reports[0]["fallback_attempts"][0]["success"] is False
    assert turn.model_usage_reports[0]["fallback_attempts"][0]["usage"]["total_tokens"] == 15


def test_agent_summarizes_omitted_history_before_action_request():
    llm = RecordingLLMClient(
        [
            "Earlier summary: keep the context compaction decision and update context tests.",
            json.dumps({"type": "final_answer", "content": "continued with summary"}),
        ]
    )
    memory = ConversationMemory(max_messages=10)
    memory.add_user_message("old decision " + ("a" * 2500))
    memory.add_assistant_message("old answer " + ("b" * 2500))
    agent = Agent(
        llm,
        memory=memory,
        context_budget=ContextBudget(max_prompt_tokens=1200, response_reserve_tokens=0),
    )

    response = agent.run_turn("latest question")

    action_request = llm.requests[-1]
    action_system_prompt = action_request[0]["content"]
    report = agent.state.last_context_report

    assert response == "continued with summary"
    assert len(llm.requests) == 2
    assert "You update a compact" in llm.requests[0][0]["content"]
    assert "Earlier summary: keep the context compaction decision" in action_system_prompt
    assert "old decision" not in json.dumps(action_request)
    assert agent.memory.summary_message_count == 2
    assert isinstance(report, dict)
    assert report["omitted_message_count"] == 0
    assert agent.state.turns[0].model_usage_totals["request_count"] == 2


def test_agent_does_not_pass_remaining_context_as_output_limit():
    llm = OutputLimitRecordingLLMClient()
    agent = Agent(llm, context_budget=ContextBudget(max_prompt_tokens=3000, response_reserve_tokens=200))

    response = agent.run_turn("hello")

    report = agent.state.last_context_report
    assert response == "ok"
    assert isinstance(report, dict)
    assert report["budget"]["input_token_budget"] == 2800
    assert llm.action_kwargs == [{}]


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
    assert "Unloaded skill metadata for this agent" not in system_prompt
    assert system_prompt.count(
        "Use this skill when the user request requires terminal inspection"
    ) == 1
    assert "Use this skill when the user request involves saving or retrieving" not in system_prompt
    assert "Procedural instructions selected for this turn" in system_prompt
    assert "<name>shell</name>" in system_prompt
    assert "# Shell Skill" in system_prompt
    assert "# Memory Skill" not in system_prompt
    assert skill_registry.get_skill("shell").loaded_content is not None
    assert skill_registry.get_skill("memory").loaded_content is None
    assert "skill_selection_completed" in trace_text
    assert "shell" in trace_text


def test_agent_traces_explainable_skill_version_digest_and_omissions(tmp_path):
    skills_dir = tmp_path / "skills"
    for name, description in (
        ("review", "Review Python code."),
        ("other", "Unrelated workflow."),
    ):
        skill_dir = skills_dir / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"""\
---
schema_version: 1
name: {name}
version: 2.1.0
description: {description}
---
# {name.title()}
""",
            encoding="utf-8",
        )
    skill_registry = SkillRegistry(skills_dir)
    skill_registry.load_metadata()
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient(
        [json.dumps({"type": "final_answer", "content": "reviewed"})]
    )
    agent = Agent(llm, skill_registry=skill_registry, trace_logger=trace_logger)

    agent.run_turn("/review inspect this")

    events = [
        json.loads(line)
        for line in trace_logger.path.read_text(encoding="utf-8").splitlines()
    ]
    payload = next(
        event["payload"]
        for event in events
        if event["type"] == "skill_selection_completed"
    )
    selected = payload["skills"][0]
    decisions = {
        decision["skill_name"]: decision for decision in payload["decisions"]
    }

    assert payload["explicit_skill_names"] == ["review"]
    assert selected["name"] == "review"
    assert selected["stage"] == "explicit"
    assert selected["reason"] == "explicit"
    assert selected["version"] == "2.1.0"
    assert selected["digest"].startswith("sha256:")
    assert selected["loaded_resources"] == ["SKILL.md"]
    assert decisions["review"]["status"] == "selected"
    assert decisions["other"]["reason"] == "no_keyword_match"


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
    assert "<name>memory</name>" in memory_prompt
    assert agent.state.loaded_skill_names == ["files"]
    assert file_response == "Files skill selected."
    assert "<name>files</name>" in file_prompt
    assert "# Shell Skill" not in file_prompt


def test_agent_can_run_safe_shell_tool(tmp_path):
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "run_cmd", "arguments": {"command": "printf hello"}}),
            json.dumps({"type": "final_answer", "content": "The command printed hello."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(shell_tool(tmp_path))
    policy = ToolPermissionPolicy(confirmation_decision=PermissionDecision.ALLOW)
    agent = Agent(llm, tool_registry=registry, permission_policy=policy)

    response = agent.run_turn("run printf hello")

    assert response == "The command printed hello."
    assert "stdout:\nhello" in agent.state.observations[0]["observation"]


def test_agent_blocks_confirmation_tool_without_permission_callback(tmp_path):
    calls = []
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "dangerous", "arguments": {"value": "run"}}),
            json.dumps({"type": "final_answer", "content": "I could not run the tool without approval."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="dangerous",
            description="Dangerous test tool.",
            args_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            callable=lambda arguments: calls.append(arguments) or ToolResult("dangerous", True, "ran"),
            requires_confirmation=True,
        )
    )
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    response = agent.run_turn("run the dangerous tool")

    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]
    turn = agent.state.turns[0]

    assert response == "I could not run the tool without approval."
    assert calls == []
    assert turn.tool_calls[0].success is False
    assert turn.tool_calls[0].error == "permission_denied"
    assert turn.tool_calls[0].failure_kind == ToolFailureKind.USER_BLOCKED
    assert turn.tool_calls[0].metadata["permission_decision"]["decision"] == "deny"
    assert "permission policy" in turn.observations[0].content
    assert any(message["role"] == "observation" and "permission policy" in message["content"] for message in llm.requests[1])
    assert "tool_permission_requested" in event_types
    assert "tool_permission_decided" in event_types


def test_agent_denied_mcp_bridge_permission_does_not_call_server(tmp_path):
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "mcp_docs_search_docs", "arguments": {"query": "MCP"}}),
            json.dumps({"type": "final_answer", "content": "I did not call the MCP server."}),
        ]
    )
    mcp_client = RecordingMCPClient()
    registry = ToolRegistry()
    bridge_tool = create_mcp_bridge_tools(
        [MCPServerConfig(label="docs", transport="streamable_http", server_url="https://mcp.example.com")],
        client_factory=lambda _server: mcp_client,
    )[0]
    registry.register(bridge_tool)
    approvals = []

    def deny(request, record):
        approvals.append((request.tool_name, record.decision))
        return PermissionDecision.DENY

    agent = Agent(llm, tool_registry=registry, permission_callback=deny, trace_logger=trace_logger)

    response = agent.run_turn("search docs")

    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    assert response == "I did not call the MCP server."
    assert mcp_client.calls == []
    assert approvals == [("mcp_docs_search_docs", PermissionDecision.ASK)]
    assert agent.state.turns[0].tool_calls[0].error == "permission_denied"
    assert any(event["type"] == TraceEvent.TOOL_PERMISSION_REQUESTED for event in events)
    assert any(event["type"] == TraceEvent.TOOL_PERMISSION_DECIDED for event in events)


def test_agent_runs_confirmation_tool_when_permission_callback_allows(tmp_path):
    calls = []
    approvals = []
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "dangerous", "arguments": {"value": "run"}}),
            json.dumps({"type": "final_answer", "content": "The approved tool ran."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="dangerous",
            description="Dangerous test tool.",
            args_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            callable=lambda arguments: calls.append(arguments) or ToolResult("dangerous", True, "ran"),
            requires_confirmation=True,
        )
    )

    def approve(request, record):
        approvals.append((request.tool_name, record.decision))
        return PermissionDecision.ALLOW

    agent = Agent(llm, tool_registry=registry, permission_callback=approve)

    response = agent.run_turn("run the dangerous tool")

    assert response == "The approved tool ran."
    assert calls == [{"value": "run"}]
    assert approvals == [("dangerous", PermissionDecision.ASK)]
    assert agent.state.turns[0].tool_calls[0].success is True


def test_agent_async_turn_awaits_tool_and_passes_context():
    calls = []

    async def lookup(arguments, context):
        calls.append((arguments, context.metadata["org_id"]))
        return ToolResult("lookup", True, f"found {arguments['query']}")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="Async lookup.",
            args_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            callable=lookup,
            accepts_context=True,
        )
    )
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "lookup", "arguments": {"query": "policy"}}),
            json.dumps({"type": "final_answer", "content": "policy found"}),
        ]
    )
    agent = Agent(llm, tool_registry=registry)

    response = asyncio.run(
        agent.run_turn_async(
            "lookup policy",
            tool_context=ToolExecutionContext(metadata={"org_id": "org-1"}),
        )
    )

    turn = agent.state.turns[0]
    assert response == "policy found"
    assert calls == [({"query": "policy"}, "org-1")]
    assert turn.tool_calls[0].success is True
    assert "found policy" in turn.observations[0].content


def test_agent_sync_turn_classifies_async_tool_misuse():
    async def lookup(_arguments):
        return ToolResult("lookup", True, "unused")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="lookup",
            description="Async lookup.",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=lookup,
        )
    )
    llm = RecordingLLMClient(
        [
            json.dumps({"type": "tool_call", "tool_name": "lookup", "arguments": {}}),
            json.dumps({"type": "final_answer", "content": "async not run"}),
        ]
    )
    agent = Agent(llm, tool_registry=registry)

    response = agent.run_turn("lookup")

    turn = agent.state.turns[0]
    assert response == "async not run"
    assert turn.tool_calls[0].success is False
    assert turn.tool_calls[0].failure_kind == ToolFailureKind.ASYNC_REQUIRED


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
    assert "available plan action" in llm.requests[0][0]["content"]
    assert "plan_created" in event_types
    assert "tool_call_started" not in event_types


def test_agent_planned_turn_allows_read_only_reconnaissance_before_plan(tmp_path):
    (tmp_path / "chulk").mkdir()
    (tmp_path / "chulk" / "core.py").write_text("class Agent:\n    pass\n", encoding="utf-8")
    plan_payload = {
        "summary": "Add subagent support based on the inspected runtime.",
        "steps": [
            {
                "id": "1",
                "title": "Extend agent runtime",
                "description": "Use the inspected chulk/core.py shape to add subagent orchestration.",
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
                    "arguments_json": json.dumps({"path": "chulk/core.py"}),
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
    assert "read-only tools" in llm.requests[0][0]["content"]
    assert any(message["role"] == "observation" for message in llm.requests[1])


def test_agent_planned_turn_uses_registered_permission_metadata_for_reconnaissance():
    calls = []
    plan_payload = {
        "summary": "Implement the feature using the inspected project metadata.",
        "steps": [
            {
                "id": "1",
                "title": "Implement project metadata support",
                "description": "Update the runtime to consume the inspected project metadata.",
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
                    "tool_name": "inspect_project_metadata",
                    "arguments_json": "{}",
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
    registry.register(
        Tool(
            name="inspect_project_metadata",
            description="Inspect project metadata without side effects.",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=lambda arguments: calls.append(arguments) or ToolResult(
                "inspect_project_metadata",
                True,
                "Project type is Python.",
            ),
            permission_level=ToolPermissionLevel.READ,
        )
    )
    agent = Agent(llm, tool_registry=registry)

    response = agent.run_planned_turn("Plan project metadata support")

    assert "Implement project metadata support" in response
    assert calls == [{}]
    assert agent.state.turns[0].tool_calls[0].phase == "planning"
    assert "<read_only_reconnaissance_tools>inspect_project_metadata</read_only_reconnaissance_tools>" in (
        llm.requests[0][0]["content"]
    )


def test_agent_planned_turn_blocks_read_named_tool_declared_as_write():
    calls = []
    llm = RecordingLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "read_file",
                    "arguments_json": "{}",
                }
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="A misleadingly named mutating tool.",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=lambda arguments: calls.append(arguments) or ToolResult("read_file", True, "mutated"),
            permission_level=ToolPermissionLevel.WRITE,
        )
    )
    agent = Agent(llm, tool_registry=registry)

    response = agent.run_planned_turn("Plan a change")

    assert "Planning can only use read-only reconnaissance tools before approval" in response
    assert "Allowed planning tools: none." in response
    assert calls == []
    assert agent.state.turns[0].tool_call_count == 0


def test_agent_planned_turn_revises_reconnaissance_only_plan(tmp_path):
    (tmp_path / "chulk").mkdir()
    (tmp_path / "chulk" / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_path / "chulk" / "config.py").write_text("class Config:\n    pass\n", encoding="utf-8")
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
                "description": "Extend chulk/core/state.py with records for child task requests and results.",
                "status": "pending",
            },
            {
                "id": "2",
                "title": "Implement subagent orchestration",
                "description": "Update chulk/core/agent.py to spawn isolated child agents and collect results.",
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
                    "arguments_json": json.dumps({"path": "chulk", "pattern": "*.py"}),
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
                    "arguments_json": json.dumps({"path": "chulk/main.py"}),
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
    assert any(
        event["type"] == "tool_observation" and event["payload"]["tool_name"] == "planning_feedback"
        for event in events
    )
    assert any(observation.tool_name == "planning_feedback" for observation in turn.observations)


def test_agent_planned_turn_revises_direct_answer_into_plan():
    plan_payload = {
        "summary": "Add subagent delegation support.",
        "steps": [
            {
                "id": "1",
                "title": "Add subagent action type",
                "description": "Extend chulk/core/actions.py with a delegation action for child-agent work.",
                "status": "pending",
            },
            {
                "id": "2",
                "title": "Implement delegation runtime",
                "description": "Update chulk/core/agent.py to create child agents and return their observations.",
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
                "description": "Modify chulk/core/agent.py using the files already inspected.",
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
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "The design step is complete.",
                            "reason": None,
                        }
                    ),
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
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "The calculator returned 4.",
                            "reason": None,
                        }
                    ),
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
    assert turn.model_request_count == 4
    assert turn.tool_call_count == 1
    assert "Planning: approved for this turn." in llm.requests[1][0]["content"]
    assert turn.tool_calls[0].plan_step_id == "1"
    assert turn.active_plan.steps[0].evidence
    assert "calculator returned 4" in turn.active_plan.steps[0].evidence[-1].content
    assert "plan_approved" in event_types
    assert "plan_step_started" in event_types
    assert "plan_step_completed" in event_types
    assert "turn_finished" in event_types


def test_agent_executes_multiple_plan_steps_with_dependencies():
    plan_payload = {
        "summary": "Calculate two values.",
        "steps": [
            {
                "id": "1",
                "title": "Calculate four",
                "description": "Use the calculator for 2 + 2.",
                "status": "pending",
                "acceptance_criteria": ["Calculator result for 2 + 2 is known."],
                "retry_limit": 0,
            },
            {
                "id": "2",
                "title": "Calculate six",
                "description": "Use the calculator for 3 + 3.",
                "status": "pending",
                "depends_on": ["1"],
                "acceptance_criteria": ["Calculator result for 3 + 3 is known."],
                "retry_limit": 0,
            },
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
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "2 + 2 returned 4.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "calculator",
                    "arguments_json": json.dumps({"expression": "3 + 3"}),
                }
            ),
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "2",
                            "status": "completed",
                            "evidence": "3 + 3 returned 6.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "The results are 4 and 6."}),
        ]
    )
    registry = ToolRegistry()
    registry.register(calculator_tool())
    agent = Agent(llm, tool_registry=registry)

    agent.run_planned_turn("calculate two values")
    response = agent.approve_plan()
    turn = agent.state.turns[0]

    assert response == "The results are 4 and 6."
    assert turn.active_plan is not None
    assert [step.status for step in turn.active_plan.steps] == ["completed", "completed"]
    assert [call.plan_step_id for call in turn.tool_calls] == ["1", "2"]
    assert turn.active_plan.steps[1].depends_on == ["1"]
    assert turn.active_plan.steps[0].evidence[-1].content == "2 + 2 returned 4."
    assert turn.active_plan.steps[1].evidence[-1].content == "3 + 3 returned 6."


def test_agent_rejects_premature_final_answer_until_plan_steps_complete():
    plan_payload = {
        "summary": "Complete one tracked step.",
        "steps": [
            {
                "id": "1",
                "title": "Confirm work",
                "description": "Record explicit step completion.",
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
            json.dumps({"type": "final_answer", "content": "Done too early."}),
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "The tracked step is complete.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "Done after completion."}),
        ]
    )
    agent = Agent(llm)

    agent.run_planned_turn("track one step")
    response = agent.approve_plan()
    turn = agent.state.turns[0]

    assert response == "Done after completion."
    assert turn.plan_execution_feedback_count == 1
    assert any(observation.tool_name == "plan_execution_feedback" for observation in turn.observations)
    assert turn.active_plan is not None
    assert turn.active_plan.status() == "completed"


def test_agent_blocks_plan_immediately_when_step_tool_fails(tmp_path):
    def failing_tool(_arguments):
        return ToolResult(
            tool_name="fail_tool",
            success=False,
            observation="The tool failed deliberately.",
            error="deliberate_failure",
        )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fail_tool",
            description="Always fail.",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=failing_tool,
        )
    )
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "test-session")
    plan_payload = {
        "summary": "Try a failing tool.",
        "steps": [
            {
                "id": "1",
                "title": "Run failing tool",
                "description": "Call a tool that fails.",
                "status": "pending",
            },
            {
                "id": "2",
                "title": "Never run",
                "description": "This step depends on the failed step.",
                "status": "pending",
                "depends_on": ["1"],
            },
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
                    "tool_name": "fail_tool",
                    "arguments_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "Should not be used."}),
        ]
    )
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    agent.run_planned_turn("run the failing plan")
    response = agent.approve_plan()
    turn = agent.state.turns[0]
    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]
    event_types = [event["type"] for event in events]

    assert response == "Plan step blocked: Run failing tool. Tool fail_tool failed with deliberate_failure."
    assert turn.status == "blocked"
    assert agent.state.active_plan is None
    assert turn.active_plan is not None
    assert [step.status for step in turn.active_plan.steps] == ["blocked", "pending"]
    assert turn.active_plan.steps[0].blocked_reason == "Tool fail_tool failed with deliberate_failure."
    assert turn.tool_call_count == 1
    assert turn.model_request_count == 2
    assert "plan_step_blocked" in event_types
    assert llm.responses == [json.dumps({"type": "final_answer", "content": "Should not be used."})]


def test_agent_retries_failed_plan_step_within_budget_then_completes():
    call_count = 0

    def flaky_tool(_arguments):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ToolResult(
                tool_name="flaky_tool",
                success=False,
                observation="The first attempt failed.",
                error="transient_failure",
                failure_kind=ToolFailureKind.ENVIRONMENT,
            )
        return ToolResult(tool_name="flaky_tool", success=True, observation="The recovery attempt succeeded.")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="flaky_tool",
            description="Fail once, then succeed.",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=flaky_tool,
        )
    )
    plan_payload = {
        "summary": "Recover one flaky operation.",
        "steps": [
            {
                "id": "1",
                "title": "Run flaky operation",
                "description": "Run the operation and recover from one transient failure.",
                "status": "pending",
                "acceptance_criteria": ["The operation succeeds."],
                "retry_limit": 1,
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
                    "tool_name": "flaky_tool",
                    "arguments_json": "{}",
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "flaky_tool",
                    "arguments_json": "{}",
                }
            ),
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "completed",
                            "evidence": "The recovery attempt succeeded.",
                            "reason": None,
                        }
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "The flaky operation recovered and completed."}),
        ]
    )
    agent = Agent(llm, tool_registry=registry)

    pending_response = agent.run_planned_turn("run the flaky operation")
    response = asyncio.run(agent.approve_plan_async())
    turn = agent.state.turns[0]
    assert turn.active_plan is not None
    step = turn.active_plan.steps[0]
    retry_evidence = step.evidence[0].metadata["plan_step_retry"]

    assert "retries   0/1 used; 1 remaining" in pending_response
    assert response == "The flaky operation recovered and completed."
    assert call_count == 2
    assert turn.status == "completed"
    assert step.status == "completed"
    assert step.retry_count == 1
    assert step.retries_remaining == 0
    assert step.tool_failure_count == 1
    assert retry_evidence["disposition"] == "retry_scheduled"
    assert retry_evidence["retries_used"] == 1
    assert any(observation.tool_name == "plan_step_retry" for observation in turn.observations)
    assert "Retry budget: 1/1 used; 0 remaining" in llm.requests[2][0]["content"]


def test_agent_blocks_failed_plan_step_only_after_retry_budget_is_exhausted(tmp_path):
    def failing_tool(_arguments):
        return ToolResult(
            tool_name="fail_tool",
            success=False,
            observation="The tool failed deliberately.",
            error="deliberate_failure",
            failure_kind=ToolFailureKind.ENVIRONMENT,
        )

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fail_tool",
            description="Always fail.",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=failing_tool,
        )
    )
    trace_logger = JSONLTraceLogger(tmp_path / "traces", "retry-session")
    plan_payload = {
        "summary": "Try a failing tool with one recovery attempt.",
        "steps": [
            {
                "id": "1",
                "title": "Run failing tool",
                "description": "Call a tool and use one recovery attempt if needed.",
                "status": "pending",
                "retry_limit": 1,
            }
        ],
    }
    tool_call = json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": "fail_tool",
            "arguments_json": "{}",
        }
    )
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
            tool_call,
            tool_call,
            json.dumps({"type": "final_answer", "content": "Should not be used."}),
        ]
    )
    agent = Agent(llm, tool_registry=registry, trace_logger=trace_logger)

    agent.run_planned_turn("run the failing plan")
    response = agent.approve_plan()
    turn = agent.state.turns[0]
    assert turn.active_plan is not None
    step = turn.active_plan.steps[0]
    retry_records = [record.metadata["plan_step_retry"] for record in step.evidence]
    events = [json.loads(line) for line in trace_logger.path.read_text(encoding="utf-8").splitlines()]

    assert response == (
        "Plan step blocked: Run failing tool. Tool fail_tool failed with deliberate_failure. "
        "Step retry limit exhausted after 1 retry."
    )
    assert turn.status == "blocked"
    assert step.status == "blocked"
    assert step.retry_count == 1
    assert step.retries_remaining == 0
    assert step.tool_failure_count == 2
    assert [record["disposition"] for record in retry_records] == ["retry_scheduled", "retry_limit_exhausted"]
    assert retry_records[-1]["failure_number"] == 2
    assert turn.tool_call_count == 2
    assert turn.model_request_count == 3
    assert sum(event["type"] == "plan_step_blocked" for event in events) == 1
    assert llm.responses == [json.dumps({"type": "final_answer", "content": "Should not be used."})]


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
    artifact_text = trace_logger.read_artifact(
        stdout_artifact["artifact_id"],
        mode="head",
        max_bytes=20_000,
    ).content
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
    artifact_text = trace_logger.read_artifact(
        observation_artifact["artifact_id"],
        mode="head",
        max_bytes=20_000,
    ).content

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
    llm = RecordingLLMClient(["not json", "## Markdown answer\n\nStill not action JSON."])
    agent = Agent(llm, max_json_repair_attempts=1)

    response = agent.run_turn("hello")

    assert "not valid action JSON" in response
    assert "I did not execute any tools from the invalid response." in response
    assert "Unparsed model output" in response
    assert "## Markdown answer" in response
    assert agent.state.json_repair_attempts == 1
    assert agent.state.errors
    assert agent.state.turns[0].status == "failed"
    assert agent.state.turns[0].tool_calls == []


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
