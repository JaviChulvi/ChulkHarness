"""Tests for LLM client behavior."""

from decimal import Decimal
import json
from types import SimpleNamespace

from chulk.core.actions import (
    FinalAnswerAction,
    PlanAction,
    ToolCallAction,
    action_json_schema_for,
)
from chulk.core.state import Plan, PlanStep
from chulk.llm import (
    DeepSeekChatCompletionsClient,
    FallbackChain,
    LLMActionError,
    LLMActionResult,
    LLM_PROVIDER_REGISTRY,
    LLMCapabilities,
    LLMClient,
    LLMConfigurationError,
    LLMCost,
    LLMError,
    LLMResponse,
    LLMStreamChunk,
    LLMUsage,
    LocalOpenAICompatibleClient,
    OpenAIResponsesClient,
    PlanningToolAvailability,
    create_llm_client,
    conservative_model_capabilities,
    resolve_model_capabilities,
)
from chulk.llm.capabilities import resolve_runtime_model_capabilities
from chulk.llm.tools import PLAN_TOOL_NAME
from chulk.mcp import MCPServerConfig


class FakeResponsesResource:
    def __init__(
        self,
        output_text: str = "model answer",
        stream_events: list | None = None,
        usage=None,
        output: list | None = None,
        responses: list | None = None,
        fail_with_tools: bool = False,
    ) -> None:
        self.output_text = output_text
        self.stream_events = stream_events or []
        self.usage = usage
        self.output = output
        self.responses = list(responses or [])
        self.fail_with_tools = fail_with_tools
        self.kwargs = None
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        if kwargs.get("tools") and self.fail_with_tools:
            raise RuntimeError("tools unsupported")
        if kwargs.get("stream"):
            return iter(self.stream_events)
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return SimpleNamespace(id="resp_static", output_text=self.output_text, usage=self.usage, output=self.output)


class FakeOpenAIClient:
    def __init__(
        self,
        output_text: str = "model answer",
        stream_events: list | None = None,
        usage=None,
        output: list | None = None,
        responses: list | None = None,
        fail_with_tools: bool = False,
    ) -> None:
        self.responses = FakeResponsesResource(
            output_text,
            stream_events=stream_events,
            usage=usage,
            output=output,
            responses=responses,
            fail_with_tools=fail_with_tools,
        )


class RetryOptionsRecordingOpenAIClient(FakeOpenAIClient):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.retry_options: list[dict] = []

    def with_options(self, **kwargs):
        self.retry_options.append(kwargs)
        return self


class FakeChatCompletionsResource:
    def __init__(
        self,
        content: str | None = "deepseek answer",
        usage=None,
        tool_calls: list | None = None,
        fail_with_tools: bool = False,
    ) -> None:
        self.content = content
        self.usage = usage
        self.tool_calls = tool_calls
        self.fail_with_tools = fail_with_tools
        self.kwargs = None
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        if kwargs.get("tools") and self.fail_with_tools:
            raise RuntimeError("tools unsupported")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content, tool_calls=self.tool_calls),
                )
            ],
            usage=self.usage,
        )


class FakeDeepSeekClient:
    def __init__(
        self,
        content: str | None = "deepseek answer",
        usage=None,
        tool_calls: list | None = None,
        fail_with_tools: bool = False,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=FakeChatCompletionsResource(
                content,
                usage=usage,
                tool_calls=tool_calls,
                fail_with_tools=fail_with_tools,
            )
        )


class FakeLocalClient:
    def __init__(
        self,
        content: str | None = "local answer",
        tool_calls: list | None = None,
        fail_with_tools: bool = False,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=FakeChatCompletionsResource(
                content,
                tool_calls=tool_calls,
                fail_with_tools=fail_with_tools,
            )
        )


class ScriptedLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.responses.pop(0)


class InternalTypeErrorLLMClient(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> str:
        self.calls += 1
        raise TypeError("provider implementation failed internally")


class ScriptedStreamingLLMClient(LLMClient):
    capabilities = LLMCapabilities(supports_streaming=True)

    def __init__(self, chunks: list[LLMStreamChunk]) -> None:
        self.chunks = chunks
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return "".join(chunk.text for chunk in self.chunks if chunk.type == "text_delta")

    def stream_complete(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None):
        self.requests.append(messages)
        yield from self.chunks


class FailingLLMClient(LLMClient):
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise LLMError(
            "provider unavailable",
            code="server_error",
            retryable=True,
            fallback_eligible=True,
        )


class UsageFailingLLMClient(LLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"

    def complete_response(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> LLMResponse:
        usage = LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15, source="provider")
        cost = LLMCost(
            amount=Decimal("0.000012"),
            pricing_known=True,
            estimated=False,
            provider=self.provider,
            model=self.model,
        )
        raise LLMActionError(
            "provider charged then failed",
            usage=usage,
            cost=cost,
            code="server_error",
            retryable=True,
            fallback_eligible=True,
        )


class UsageSuccessfulLLMClient(LLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"

    def complete_response(self, messages: list[dict[str, str]], *, max_output_tokens: int | None = None) -> LLMResponse:
        usage = LLMUsage(input_tokens=20, output_tokens=10, total_tokens=30, source="provider")
        cost = LLMCost(
            amount=Decimal("0.000024"),
            pricing_known=True,
            estimated=False,
            provider=self.provider,
            model=self.model,
        )
        return LLMResponse(content="fallback answer", usage=usage, cost=cost, provider=self.provider, model=self.model)


def fake_calculator_tool():
    return SimpleNamespace(
        name="calculator",
        description="Evaluate arithmetic.",
        args_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    )


def fake_mcp_bridge_tool():
    return SimpleNamespace(
        name="mcp_docs_search_docs",
        description="Bridge docs search.",
        args_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        metadata={"mcp_bridge": True},
    )


def test_openai_responses_client_sends_instructions_and_input():
    fake_client = FakeOpenAIClient()
    client = OpenAIResponsesClient(model="test-model", client=fake_client)

    response = client.complete(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ]
    )

    assert response == "model answer"
    assert fake_client.responses.kwargs == {
        "model": "test-model",
        "instructions": "Be concise.",
        "input": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ],
    }


def test_base_stream_complete_falls_back_to_one_shot_completion():
    client = ScriptedLLMClient(["plain answer"])

    chunks = list(client.stream_complete([{"role": "user", "content": "Hello"}]))

    assert chunks[0] == LLMStreamChunk(type="text_delta", text="plain answer")
    assert chunks[1].type == "completed"
    assert chunks[1].usage is not None
    assert chunks[1].usage.estimated is True
    assert chunks[1].cost is not None
    assert chunks[1].cost.pricing_known is False


def test_base_complete_response_estimates_usage_and_preserves_complete_compatibility():
    client = ScriptedLLMClient(["plain answer", "plain answer"])

    assert client.complete([{"role": "user", "content": "Hello"}]) == "plain answer"
    response = client.complete_response([{"role": "user", "content": "Hello"}])

    assert response.content == "plain answer"
    assert response.usage is not None
    assert response.usage.estimated is True
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.cost is not None
    assert response.cost.pricing_known is False


def test_base_complete_response_does_not_retry_internal_type_error():
    client = InternalTypeErrorLLMClient()

    try:
        client.complete_response([{"role": "user", "content": "Hello"}], max_output_tokens=50)
    except TypeError as exc:
        assert "internally" in str(exc)
    else:
        raise AssertionError("Expected the provider TypeError to escape")

    assert client.calls == 1


def test_openai_responses_client_streams_text_deltas():
    fake_client = FakeOpenAIClient(
        stream_events=[
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta="hello "),
            SimpleNamespace(type="response.output_text.delta", delta="world"),
            SimpleNamespace(type="response.completed"),
        ]
    )
    client = OpenAIResponsesClient(model="test-model", client=fake_client)

    chunks = list(client.stream_complete([{"role": "user", "content": "Hello"}]))

    assert [chunk.text for chunk in chunks if chunk.type == "text_delta"] == ["hello ", "world"]
    assert chunks[-1].type == "completed"
    assert fake_client.responses.kwargs == {
        "model": "test-model",
        "instructions": None,
        "input": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }


def test_openai_responses_client_extracts_usage_and_cost():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        total_tokens=1200,
        input_tokens_details=SimpleNamespace(cached_tokens=100),
    )
    fake_client = FakeOpenAIClient(usage=usage)
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)

    response = client.complete_response([{"role": "user", "content": "Hello"}])

    assert response.content == "model answer"
    assert response.usage is not None
    assert response.usage.input_tokens == 1000
    assert response.usage.cached_input_tokens == 100
    assert response.usage.cache_miss_input_tokens == 900
    assert response.cost is not None
    assert response.cost.to_dict()["amount"] == "0.00069"
    assert response.cost.pricing_known is True
    assert response.cost.estimated is False


def test_openai_responses_client_extracts_cache_write_usage_and_cost():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        total_tokens=1200,
        input_tokens_details=SimpleNamespace(
            cached_tokens=100,
            cache_write_tokens=200,
        ),
    )
    fake_client = FakeOpenAIClient(usage=usage)
    client = OpenAIResponsesClient(model="gpt-5.6-sol", client=fake_client)

    response = client.complete_response([{"role": "user", "content": "Hello"}])

    assert response.usage is not None
    assert response.usage.cache_hit_input_tokens == 100
    assert response.usage.cache_write_input_tokens == 200
    assert response.usage.cache_miss_input_tokens == 700
    assert response.cost is not None
    assert response.cost.input_cost == Decimal("0.0035")
    assert response.cost.cached_input_cost == Decimal("0.00005")
    assert response.cost.cache_write_input_cost == Decimal("0.00125")
    assert response.cost.output_cost == Decimal("0.006")
    assert response.cost.amount == Decimal("0.0108")
    assert response.cost.pricing_known is True
    assert response.cost.estimated is False


def test_openai_responses_client_uses_native_tool_calls_when_tools_are_provided():
    fake_client = FakeOpenAIClient(
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="calculator",
                arguments=json.dumps({"expression": "1 + 1"}),
                call_id="call_1",
            )
        ],
    )
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)

    result = client.complete_action([{"role": "user", "content": "what is 1+1?"}], tools=[fake_calculator_tool()])

    assert result.action == ToolCallAction(type="tool_call", tool_name="calculator", arguments={"expression": "1 + 1"})
    assert fake_client.responses.kwargs["tool_choice"] == "auto"
    assert fake_client.responses.kwargs["parallel_tool_calls"] is False
    assert "tools" in fake_client.responses.kwargs
    assert "text" not in fake_client.responses.kwargs
    tool_names = {tool["name"] for tool in fake_client.responses.kwargs["tools"]}
    assert tool_names == {"calculator"}
    assert result.metadata["action_transport"] == "provider_native"
    assert result.metadata["provider_tool_call"]["call_id"] == "call_1"


def test_openai_rejects_multiple_function_calls_and_uses_json_fallback():
    fallback_payload = json.dumps(
        {
            "type": "final_answer",
            "content": "safe fallback",
            "tool_name": None,
            "arguments_json": "{}",
            "plan_json": "{}",
            "step_update_json": "{}",
        }
    )
    fake_client = FakeOpenAIClient(
        responses=[
            SimpleNamespace(
                id="resp_parallel",
                output_text="",
                usage=None,
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="calculator",
                        arguments=json.dumps({"expression": "1 + 1"}),
                        call_id="call_1",
                    ),
                    SimpleNamespace(
                        type="function_call",
                        name="calculator",
                        arguments=json.dumps({"expression": "2 + 2"}),
                        call_id="call_2",
                    ),
                ],
            ),
            SimpleNamespace(
                id="resp_fallback",
                output_text=fallback_payload,
                usage=None,
                output=[],
            ),
        ]
    )
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)

    result = client.complete_action(
        [{"role": "user", "content": "calculate both"}],
        tools=[fake_calculator_tool()],
    )

    assert result.action == FinalAnswerAction(
        type="final_answer",
        content="safe fallback",
    )
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert "multiple function calls" in result.metadata["native_tool_call_error"]
    assert len(fake_client.responses.calls) == 2
    assert fake_client.responses.calls[0]["parallel_tool_calls"] is False
    assert "tools" not in fake_client.responses.calls[1]


def test_openai_native_text_with_no_effective_tools_omits_tool_fields():
    fake_client = FakeOpenAIClient(output_text="native final")
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)

    result = client.complete_action(
        [{"role": "user", "content": "hello"}],
        tools=[],
        planning_tools=PlanningToolAvailability(),
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="native final")
    assert "tools" not in fake_client.responses.kwargs
    assert "tool_choice" not in fake_client.responses.kwargs
    assert result.metadata["action_transport"] == "provider_native"


def test_openai_responses_client_sends_hosted_mcp_tools_and_filters_bridge_tools():
    fake_client = FakeOpenAIClient(output_text="native final")
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)
    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
        server_description="Docs MCP",
        allowed_tools=("search_docs",),
        authorization="secret-token",
    )

    result = client.complete_action(
        [{"role": "user", "content": "search docs"}],
        tools=[fake_calculator_tool(), fake_mcp_bridge_tool()],
        hosted_mcp_servers=[server],
    )

    assert result.action.content == "native final"
    request_tools = fake_client.responses.kwargs["tools"]
    function_tool_names = {tool["name"] for tool in request_tools if tool["type"] == "function"}
    assert "calculator" in function_tool_names
    assert "mcp_docs_search_docs" not in function_tool_names
    assert {
        "type": "mcp",
        "server_label": "docs",
        "server_url": "https://mcp.example.com",
        "require_approval": "always",
        "server_description": "Docs MCP",
        "allowed_tools": ["search_docs"],
        "authorization": "secret-token",
    } in request_tools


def test_openai_responses_client_continues_after_mcp_approval_request():
    fake_client = FakeOpenAIClient(
        responses=[
            SimpleNamespace(
                id="resp_1",
                output_text="",
                usage=None,
                output=[
                    SimpleNamespace(
                        type="mcp_approval_request",
                        id="approval_1",
                        server_label="docs",
                        name="search_docs",
                        arguments=json.dumps({"query": "MCP"}),
                    )
                ],
            ),
            SimpleNamespace(id="resp_2", output_text="approved answer", usage=None, output=[]),
        ]
    )
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)
    approvals = []

    result = client.complete_action(
        [{"role": "user", "content": "search docs"}],
        hosted_mcp_servers=[
            MCPServerConfig(label="docs", transport="streamable_http", server_url="https://mcp.example.com")
        ],
        mcp_approval_callback=lambda approval: approvals.append(approval) or True,
    )

    assert result.action.content == "approved answer"
    assert approvals[0]["server_label"] == "docs"
    assert approvals[0]["name"] == "search_docs"
    assert len(fake_client.responses.calls) == 2
    continuation = fake_client.responses.calls[1]
    assert continuation["previous_response_id"] == "resp_1"
    assert continuation["input"] == [
        {
            "type": "mcp_approval_response",
            "approval_request_id": "approval_1",
            "approve": True,
        }
    ]
    assert result.metadata["provider_mcp_approval"] == [
        {
            "approval_request_id": "approval_1",
            "server_label": "docs",
            "name": "search_docs",
            "approved": True,
        }
    ]


def test_openai_mcp_approval_continuation_keeps_planning_contract_and_fails_closed_on_text():
    fake_client = FakeOpenAIClient(
        responses=[
            SimpleNamespace(
                id="resp_1",
                output_text="",
                usage=None,
                output=[
                    SimpleNamespace(
                        type="mcp_approval_request",
                        id="approval_1",
                        server_label="docs",
                        name="search_docs",
                        arguments=json.dumps({"query": "MCP"}),
                    )
                ],
            ),
            SimpleNamespace(
                id="resp_2",
                output_text="phase-illegal final text",
                usage=None,
                output=[],
            ),
        ]
    )
    primary = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)
    secondary = ScriptedLLMClient(
        [json.dumps({"type": "final_answer", "content": "must not run"})]
    )
    fallback = FallbackChain([primary, secondary])
    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
    )

    try:
        fallback.complete_action(
            [
                {"role": "system", "content": "Finish with a plan step update."},
                {"role": "user", "content": "Search the docs and update the step."},
            ],
            planning_tools=PlanningToolAvailability(update_plan_step=True),
            hosted_mcp_servers=[server],
            mcp_approval_callback=lambda _approval: True,
        )
    except LLMError as exc:
        assert exc.code == "action_shape_error"
        assert exc.retryable is False
        assert exc.fallback_eligible is False
        assert "planning action was required" in str(exc)
    else:
        raise AssertionError("Expected post-MCP text to fail closed during plan execution")

    continuation = fake_client.responses.calls[1]
    assert continuation["tool_choice"] == "required"
    assert continuation["instructions"] == "Finish with a plan step update."
    assert len(fallback.last_attempts) == 1
    assert secondary.requests == []


def test_openai_hosted_mcp_approval_requires_callback():
    fake_client = FakeOpenAIClient(
        responses=[
            SimpleNamespace(
                id="resp_1",
                output_text="",
                usage=None,
                output=[
                    SimpleNamespace(
                        type="mcp_approval_request",
                        id="approval_1",
                        server_label="docs",
                        name="search_docs",
                        arguments=json.dumps({"query": "MCP"}),
                    )
                ],
            )
        ]
    )
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)

    try:
        client.complete_action(
            [{"role": "user", "content": "search docs"}],
            tools=[],
            hosted_mcp_servers=[
                MCPServerConfig(label="docs", transport="streamable_http", server_url="https://mcp.example.com")
            ],
        )
    except LLMError as exc:
        assert "permission callback" in str(exc)
    else:
        raise AssertionError("Expected hosted MCP approval without callback to fail")


def test_openai_hosted_mcp_shape_error_advances_fallback_chain():
    primary = OpenAIResponsesClient(
        model="gpt-4.1-mini",
        client=FakeOpenAIClient(output_text="", output=[]),
    )
    secondary = ScriptedLLMClient(
        [json.dumps({"type": "final_answer", "content": "fallback ok"})]
    )
    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
    )

    result = FallbackChain([primary, secondary]).complete_action(
        [{"role": "user", "content": "search docs"}],
        tools=[],
        planning_tools=PlanningToolAvailability(),
        hosted_mcp_servers=[server],
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="fallback ok")
    assert len(secondary.requests) == 1


def test_openai_hosted_mcp_call_without_final_action_does_not_replay_on_fallback():
    primary = OpenAIResponsesClient(
        model="gpt-4.1-mini",
        client=FakeOpenAIClient(
            output_text="",
            output=[
                SimpleNamespace(
                    type="mcp_call",
                    id="call_1",
                    server_label="write_api",
                    name="create_record",
                    arguments=json.dumps({"value": "created"}),
                    output="created",
                    status="completed",
                )
            ],
        ),
    )
    secondary = ScriptedLLMClient(
        [json.dumps({"type": "final_answer", "content": "must not run"})]
    )
    fallback = FallbackChain([primary, secondary])
    server = MCPServerConfig(
        label="write_api",
        transport="streamable_http",
        server_url="https://mcp.example.com",
    )

    try:
        fallback.complete_action(
            [{"role": "user", "content": "create a record"}],
            tools=[],
            hosted_mcp_servers=[server],
        )
    except LLMError as exc:
        assert exc.code == "action_shape_error"
        assert exc.retryable is False
        assert exc.fallback_eligible is False
    else:
        raise AssertionError("Expected a hosted MCP response without a final action to fail closed")

    assert len(fallback.last_attempts) == 1
    assert secondary.requests == []


def test_openai_timeout_after_approved_mcp_call_does_not_replay_on_fallback():
    fake_client = RetryOptionsRecordingOpenAIClient(
        responses=[
            SimpleNamespace(
                id="resp_1",
                output_text="",
                usage=None,
                output=[
                    SimpleNamespace(
                        type="mcp_approval_request",
                        id="approval_1",
                        server_label="write_api",
                        name="create_record",
                        arguments=json.dumps({"value": "created"}),
                    )
                ],
            ),
            TimeoutError("response timed out after approval"),
        ]
    )
    primary = OpenAIResponsesClient(
        model="gpt-4.1-mini",
        client=fake_client,
    )
    secondary = ScriptedLLMClient(
        [json.dumps({"type": "final_answer", "content": "must not run"})]
    )
    fallback = FallbackChain([primary, secondary])
    server = MCPServerConfig(
        label="write_api",
        transport="streamable_http",
        server_url="https://mcp.example.com",
    )

    try:
        fallback.complete_action(
            [{"role": "user", "content": "create a record"}],
            tools=[],
            hosted_mcp_servers=[server],
            mcp_approval_callback=lambda _approval: True,
        )
    except LLMError as exc:
        assert exc.code == "timeout"
        assert exc.retryable is False
        assert exc.fallback_eligible is False
        assert "may have executed" in str(exc)
    else:
        raise AssertionError("Expected an ambiguous post-approval failure to fail closed")

    assert len(primary._client.responses.calls) == 2
    assert fake_client.retry_options == [{"max_retries": 0}]
    assert len(fallback.last_attempts) == 1
    assert secondary.requests == []


def test_openai_pre_execution_mcp_rejection_can_advance_fallback():
    fake_client = RetryOptionsRecordingOpenAIClient(
        responses=[
            LLMError(
                "hosted MCP tools are unsupported",
                provider="openai",
                model="gpt-4.1-mini",
                code="unsupported_feature",
                fallback_eligible=True,
            )
        ]
    )
    primary = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)
    secondary = ScriptedLLMClient(
        [json.dumps({"type": "final_answer", "content": "fallback ok"})]
    )
    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
        approval="never",
    )

    result = FallbackChain([primary, secondary]).complete_action(
        [{"role": "user", "content": "search docs"}],
        tools=[],
        hosted_mcp_servers=[server],
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="fallback ok")
    assert fake_client.retry_options == [{"max_retries": 0}]
    assert len(secondary.requests) == 1


def test_openai_mcp_execution_blocks_action_repair_and_fallback():
    fake_client = RetryOptionsRecordingOpenAIClient(
        output_text="",
        output=[
            SimpleNamespace(
                type="mcp_call",
                id="call_1",
                server_label="write_api",
                name="create_record",
                arguments=json.dumps({"value": "created"}),
                output="created",
                status="completed",
            ),
            SimpleNamespace(
                type="function_call",
                name=PLAN_TOOL_NAME,
                arguments="{}",
            ),
        ],
    )
    primary = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)
    secondary = ScriptedLLMClient(
        [json.dumps({"type": "final_answer", "content": "must not run"})]
    )
    fallback = FallbackChain([primary, secondary])
    server = MCPServerConfig(
        label="write_api",
        transport="streamable_http",
        server_url="https://mcp.example.com",
        approval="never",
    )

    try:
        fallback.complete_action(
            [{"role": "user", "content": "create a record"}],
            tools=[],
            hosted_mcp_servers=[server],
        )
    except LLMActionError as exc:
        assert exc.repair_attempts == 0
        assert "repair disabled" in str(exc)
        assert exc.fallback_eligible is False
    else:
        raise AssertionError("Expected invalid post-MCP action to fail without repair")

    assert len(fake_client.responses.calls) == 1
    assert fake_client.retry_options == [{"max_retries": 0}]
    assert len(fallback.last_attempts) == 1
    assert secondary.requests == []


def test_openai_malformed_hosted_mcp_approval_advances_fallback_chain():
    primary = OpenAIResponsesClient(
        model="gpt-4.1-mini",
        client=FakeOpenAIClient(
            responses=[
                SimpleNamespace(
                    id="resp_1",
                    output_text="",
                    usage=None,
                    output=[
                        SimpleNamespace(
                            type="mcp_approval_request",
                            id=None,
                            server_label="docs",
                            name="search_docs",
                            arguments=json.dumps({"query": "MCP"}),
                        )
                    ],
                )
            ]
        ),
    )
    secondary = ScriptedLLMClient(
        [json.dumps({"type": "final_answer", "content": "fallback ok"})]
    )
    server = MCPServerConfig(
        label="docs",
        transport="streamable_http",
        server_url="https://mcp.example.com",
    )
    fallback = FallbackChain([primary, secondary])

    result = fallback.complete_action(
        [{"role": "user", "content": "search docs"}],
        tools=[],
        hosted_mcp_servers=[server],
        mcp_approval_callback=lambda _approval: True,
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="fallback ok")
    assert [attempt.success for attempt in fallback.last_attempts] == [False, True]
    assert fallback.last_attempts[0].error_code == "invalid_response"
    assert len(secondary.requests) == 1


def test_openai_responses_native_final_text_becomes_final_answer():
    fake_client = FakeOpenAIClient(output_text="native final")
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)

    result = client.complete_action([{"role": "user", "content": "hello"}], tools=[fake_calculator_tool()])

    assert result.action.content == "native final"
    assert result.metadata["action_transport"] == "provider_native"


def test_openai_responses_native_plan_tool_becomes_plan_action():
    fake_client = FakeOpenAIClient(
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name=PLAN_TOOL_NAME,
                arguments=json.dumps(
                    {
                        "summary": "Make a change.",
                        "steps": [
                            {
                                "id": "1",
                                "title": "Edit file",
                                "description": "Update the file.",
                                "status": "pending",
                                "depends_on": [],
                                "acceptance_criteria": ["File is updated."],
                                "retry_limit": 0,
                            }
                        ],
                    }
                ),
            )
        ],
    )
    client = OpenAIResponsesClient(model="gpt-4.1-mini", client=fake_client)

    result = client.complete_action(
        [{"role": "user", "content": "plan this"}],
        planning_tools=PlanningToolAvailability(propose_plan=True),
    )

    assert isinstance(result.action, PlanAction)
    assert result.action.plan.summary == "Make a change."
    assert result.action.plan.steps[0].title == "Edit file"
    assert [tool["name"] for tool in fake_client.responses.kwargs["tools"]] == [
        PLAN_TOOL_NAME
    ]
    assert fake_client.responses.kwargs["tool_choice"] == "required"


def test_unknown_model_reports_usage_without_cost():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        total_tokens=1200,
        input_tokens_details=SimpleNamespace(cached_tokens=100),
    )
    fake_client = FakeOpenAIClient(usage=usage)
    client = OpenAIResponsesClient(model="unknown-model", client=fake_client)

    response = client.complete_response([{"role": "user", "content": "Hello"}])

    assert response.usage is not None
    assert response.cost is not None
    assert response.cost.pricing_known is False
    assert response.cost.amount is None


def test_fallback_chain_streams_from_first_successful_provider():
    fallback = FallbackChain(
        [
            FailingLLMClient(),
            ScriptedStreamingLLMClient(
                [
                    LLMStreamChunk(type="text_delta", text="fallback "),
                    LLMStreamChunk(type="text_delta", text="worked"),
                    LLMStreamChunk(type="completed"),
                ]
            ),
        ]
    )

    chunks = list(fallback.stream_complete([{"role": "user", "content": "Hello"}]))

    assert "".join(chunk.text for chunk in chunks if chunk.type == "text_delta") == "fallback worked"
    assert [attempt.success for attempt in fallback.last_attempts] == [False, True]
    assert fallback.last_success_provider is fallback.providers[1]


def test_fallback_chain_records_attempt_usage_and_cost_separately():
    fallback = FallbackChain([UsageFailingLLMClient(), UsageSuccessfulLLMClient()])

    response = fallback.complete_response([{"role": "user", "content": "Hello"}])

    assert response.content == "fallback answer"
    assert [attempt.success for attempt in fallback.last_attempts] == [False, True]
    assert fallback.last_attempts[0].usage is not None
    assert fallback.last_attempts[0].usage.total_tokens == 15
    assert fallback.last_attempts[0].cost is not None
    assert fallback.last_attempts[0].cost.amount == Decimal("0.000012")
    assert fallback.last_attempts[1].usage is response.usage


def test_fallback_chain_keeps_action_attempts_across_json_repair():
    provider = ScriptedLLMClient(
        [
            "plain prose",
            json.dumps({"type": "final_answer", "content": "repaired"}),
        ]
    )
    fallback = FallbackChain([provider])

    result = fallback.complete_action([{"role": "user", "content": "Hello"}], max_repair_attempts=1)

    assert result.action.content == "repaired"
    assert result.repair_attempts == 1
    assert len(fallback.last_attempts) == 2
    assert [attempt.success for attempt in fallback.last_attempts] == [True, True]
    assert result.usage is not None
    assert result.usage.total_tokens > fallback.last_attempts[-1].usage.total_tokens


def test_fallback_chain_honors_a_custom_public_complete_action():
    class PublicActionClient(LLMClient):
        provider = "custom-action"

        def __init__(self) -> None:
            self.planning_tools = None

        def complete(self, messages, *, max_output_tokens=None):
            raise AssertionError("complete() must not replace complete_action()")

        def complete_action(self, messages, *, planning_tools=None, **kwargs):
            self.planning_tools = planning_tools
            return LLMActionResult(
                action=FinalAnswerAction(type="final_answer", content="custom action"),
                raw_response='{"type":"custom"}',
            )

    provider = PublicActionClient()
    policy = PlanningToolAvailability(propose_plan=True)

    result = FallbackChain([provider]).complete_action(
        [{"role": "user", "content": "hello"}],
        planning_tools=policy,
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="custom action")
    assert provider.planning_tools == policy
    assert result.raw_response == '{"type":"custom"}'


def test_fallback_chain_serializes_custom_plan_action_as_a_strict_proposal():
    plan = Plan(
        summary="Inspect and update the file.",
        steps=[
            PlanStep(
                id="inspect",
                title="Inspect",
                description="Read the current implementation.",
                acceptance_criteria=["The implementation is understood."],
            ),
            PlanStep(
                id="update",
                title="Update",
                description="Apply the requested change.",
                depends_on=["inspect"],
                acceptance_criteria=["The requested behavior is implemented."],
                retry_limit=1,
            ),
        ],
    )

    class PublicPlanActionClient(LLMClient):
        def complete_action(self, messages, **kwargs):
            return LLMActionResult(
                action=PlanAction(type="plan", plan=plan),
                raw_response='{"type":"custom-plan"}',
            )

    result = FallbackChain([PublicPlanActionClient()]).complete_action(
        [{"role": "user", "content": "Plan this"}],
        action_schema=action_json_schema_for(["plan"]),
    )

    assert result.action == PlanAction(type="plan", plan=plan)
    assert result.raw_response == '{"type":"custom-plan"}'


def test_custom_public_action_provider_does_not_collapse_other_provider_repairs():
    class PublicActionClient(LLMClient):
        def complete_action(self, messages, **kwargs):
            raise AssertionError("fallback provider should not be reached")

    scripted = ScriptedLLMClient(
        [
            "plain prose",
            json.dumps({"type": "final_answer", "content": "repaired"}),
        ]
    )
    fallback = FallbackChain([scripted, PublicActionClient()])

    result = fallback.complete_action(
        [{"role": "user", "content": "Hello"}],
        max_repair_attempts=1,
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="repaired")
    assert result.repair_attempts == 1
    assert len(fallback.last_attempts) == 2
    assert [attempt.success for attempt in fallback.last_attempts] == [True, True]
    assert result.usage is not None
    assert fallback.last_attempts[-1].usage is not None
    assert result.usage.total_tokens > fallback.last_attempts[-1].usage.total_tokens


def test_openai_responses_client_requires_api_key_without_injected_client():
    try:
        OpenAIResponsesClient(model="test-model", api_key=None)
    except LLMConfigurationError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing API key to fail")


def test_openai_responses_client_uses_strict_action_schema():
    fake_client = FakeOpenAIClient(
        json.dumps({"type": "final_answer", "content": "structured answer", "tool_name": None, "arguments_json": "{}"})
    )
    client = OpenAIResponsesClient(model="test-model", client=fake_client)

    result = client.complete_action(
        [
            {"role": "system", "content": "Return action JSON."},
            {"role": "user", "content": "Hello"},
        ]
    )

    response_format = fake_client.responses.kwargs["text"]["format"]
    schema = response_format["schema"]
    assert result.action.content == "structured answer"
    assert response_format["type"] == "json_schema"
    assert response_format["strict"] is True
    assert "plan" in schema["properties"]["type"]["enum"]
    assert "plan_step_update" in schema["properties"]["type"]["enum"]
    assert "arguments_json" in schema["properties"]
    assert "plan_json" in schema["properties"]
    assert "step_update_json" in schema["properties"]
    assert "arguments" not in schema["properties"]


def test_openai_responses_client_uses_phase_narrowed_action_schema():
    plan_payload = {
        "summary": "Make the change.",
        "steps": [
            {
                "id": "1",
                "title": "Make the change",
                "description": "Complete the requested work.",
                "status": "pending",
            }
        ],
    }
    fake_client = FakeOpenAIClient(
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
    )
    client = OpenAIResponsesClient(model="test-model", client=fake_client)

    result = client.complete_action(
        [{"role": "user", "content": "Plan this"}],
        action_schema=action_json_schema_for(["plan"]),
    )

    schema = fake_client.responses.kwargs["text"]["format"]["schema"]
    assert isinstance(result.action, PlanAction)
    assert schema["properties"]["type"]["enum"] == ["plan"]


def test_openai_responses_client_applies_output_limits():
    fake_client = FakeOpenAIClient(
        json.dumps({"type": "final_answer", "content": "structured answer", "tool_name": None, "arguments_json": "{}"})
    )
    client = OpenAIResponsesClient(model="test-model", client=fake_client, max_output_tokens=100)

    client.complete([{"role": "user", "content": "Hello"}], max_output_tokens=25)

    assert fake_client.responses.kwargs["max_output_tokens"] == 25

    client.complete_action(
        [
            {"role": "system", "content": "Return action JSON."},
            {"role": "user", "content": "Hello"},
        ],
        max_output_tokens=250,
    )

    assert fake_client.responses.kwargs["max_output_tokens"] == 100


def test_deepseek_client_sends_chat_completion_messages():
    fake_client = FakeDeepSeekClient()
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    response = client.complete(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ]
    )

    assert response == "deepseek answer"
    assert fake_client.chat.completions.kwargs == {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ],
        "stream": False,
    }


def test_deepseek_client_uses_native_tool_calls_when_tools_are_provided():
    fake_client = FakeDeepSeekClient(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="calculator", arguments=json.dumps({"expression": "2 + 2"})),
            )
        ],
    )
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    result = client.complete_action([{"role": "user", "content": "what is 2+2?"}], tools=[fake_calculator_tool()])

    assert result.action == ToolCallAction(type="tool_call", tool_name="calculator", arguments={"expression": "2 + 2"})
    assert fake_client.chat.completions.kwargs["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in fake_client.chat.completions.kwargs
    assert "tools" in fake_client.chat.completions.kwargs
    assert "response_format" not in fake_client.chat.completions.kwargs
    tool_names = {tool["function"]["name"] for tool in fake_client.chat.completions.kwargs["tools"]}
    assert tool_names == {"calculator"}
    assert result.metadata["action_transport"] == "provider_native"
    assert result.metadata["provider_tool_call"]["id"] == "call_1"


def test_deepseek_client_receives_mcp_bridge_tools_as_function_tools():
    fake_client = FakeDeepSeekClient(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_mcp",
                type="function",
                function=SimpleNamespace(
                    name="mcp_docs_search_docs",
                    arguments=json.dumps({"query": "MCP"}),
                ),
            )
        ],
    )
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    result = client.complete_action([{"role": "user", "content": "search docs"}], tools=[fake_mcp_bridge_tool()])

    assert result.action == ToolCallAction(
        type="tool_call",
        tool_name="mcp_docs_search_docs",
        arguments={"query": "MCP"},
    )
    tool_names = {tool["function"]["name"] for tool in fake_client.chat.completions.kwargs["tools"]}
    assert "mcp_docs_search_docs" in tool_names


def test_deepseek_native_final_text_becomes_final_answer():
    fake_client = FakeDeepSeekClient(content="native final")
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    result = client.complete_action([{"role": "user", "content": "hello"}], tools=[fake_calculator_tool()])

    assert result.action.content == "native final"
    assert result.metadata["action_transport"] == "provider_native"


def test_deepseek_client_extracts_cache_usage_and_cost():
    usage = SimpleNamespace(
        prompt_tokens=3000,
        completion_tokens=500,
        total_tokens=3500,
        prompt_cache_hit_tokens=1000,
        prompt_cache_miss_tokens=2000,
    )
    fake_client = FakeDeepSeekClient(usage=usage)
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    response = client.complete_response([{"role": "user", "content": "Hello"}])

    assert response.usage is not None
    assert response.usage.input_tokens == 3000
    assert response.usage.cache_hit_input_tokens == 1000
    assert response.usage.cache_miss_input_tokens == 2000
    assert response.usage.cache_split_estimated is False
    assert response.cost is not None
    assert response.cost.to_dict()["amount"] == "0.0004228"
    assert response.cost.pricing_known is True


def test_deepseek_missing_cache_split_treats_prompt_as_estimated_miss():
    usage = SimpleNamespace(prompt_tokens=3000, completion_tokens=500, total_tokens=3500)
    fake_client = FakeDeepSeekClient(usage=usage)
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    response = client.complete_response([{"role": "user", "content": "Hello"}])

    assert response.usage is not None
    assert response.usage.cache_hit_input_tokens == 0
    assert response.usage.cache_miss_input_tokens == 3000
    assert response.usage.cache_split_estimated is True
    assert response.cost is not None
    assert response.cost.estimated is True


def test_deepseek_client_requires_api_key_without_injected_client():
    try:
        DeepSeekChatCompletionsClient(model="deepseek-v4-flash", api_key=None)
    except LLMConfigurationError as exc:
        assert "DEEPSEEK_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected missing API key to fail")


def test_deepseek_client_uses_json_object_mode_for_actions():
    fake_client = FakeDeepSeekClient(
        json.dumps({"type": "final_answer", "content": "structured answer", "tool_name": None, "arguments_json": "{}"})
    )
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    result = client.complete_action(
        [
            {"role": "system", "content": "Return action JSON."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert result.action.content == "structured answer"
    assert fake_client.chat.completions.kwargs["response_format"] == {"type": "json_object"}


def test_deepseek_client_sends_only_explicit_max_tokens():
    fake_client = FakeDeepSeekClient(
        json.dumps({"type": "final_answer", "content": "structured answer", "tool_name": None, "arguments_json": "{}"})
    )
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    client.complete([{"role": "user", "content": "Hello"}], max_output_tokens=25)

    assert fake_client.chat.completions.kwargs["max_tokens"] == 25

    client.complete_action(
        [
            {"role": "system", "content": "Return action JSON."},
            {"role": "user", "content": "Hello"},
        ],
        max_output_tokens=250,
    )

    assert fake_client.chat.completions.kwargs["max_tokens"] == 250


def test_deepseek_client_can_parse_plan_action_from_json_mode():
    plan_json = json.dumps(
        {
            "summary": "Inspect before editing.",
            "steps": [
                {
                    "id": "1",
                    "title": "Inspect files",
                    "description": "List relevant files before editing.",
                    "status": "pending",
                }
            ],
        }
    )
    fake_client = FakeDeepSeekClient(
        json.dumps(
            {
                "type": "plan",
                "content": None,
                "tool_name": None,
                "arguments_json": "{}",
                "plan_json": plan_json,
            }
        )
    )
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    result = client.complete_action(
        [
            {"role": "system", "content": "Return action JSON."},
            {"role": "user", "content": "Inspect the files"},
        ]
    )

    assert result.action.type == "plan"
    assert result.action.plan.summary == "Inspect before editing."
    assert result.action.plan.steps[0].status == "pending"
    assert fake_client.chat.completions.kwargs["response_format"] == {"type": "json_object"}


def test_local_client_sends_local_template_friendly_chat_completion_messages():
    fake_client = FakeLocalClient()
    client = LocalOpenAICompatibleClient(model="google/gemma-4-12b-qat", client=fake_client)

    response = client.complete(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ]
    )

    assert response == "local answer"
    assert fake_client.chat.completions.kwargs == {
        "model": "google/gemma-4-12b-qat",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Instructions:\nBe concise.\n\nUser message:\n\nContinue"},
        ],
        "stream": False,
    }


def test_local_client_guarantees_user_message_for_system_only_requests():
    fake_client = FakeLocalClient()
    client = LocalOpenAICompatibleClient(model="google/gemma-4-12b-qat", client=fake_client)

    client.complete([{"role": "system", "content": "Return action JSON."}])

    assert fake_client.chat.completions.kwargs["messages"] == [
        {"role": "user", "content": "Instructions:\nReturn action JSON."}
    ]


def test_local_client_converts_observations_to_user_messages():
    fake_client = FakeLocalClient()
    client = LocalOpenAICompatibleClient(model="google/gemma-4-12b-qat", client=fake_client)

    client.complete(
        [
            {"role": "system", "content": "Return action JSON."},
            {"role": "user", "content": "Find the file."},
            {"role": "observation", "content": "Tool read_file finished with success."},
        ]
    )

    assert fake_client.chat.completions.kwargs["messages"] == [
        {
            "role": "user",
            "content": (
                "Instructions:\nReturn action JSON.\n\n"
                "User message:\n\n"
                "Find the file.\n\n"
                "observation: Tool read_file finished with success."
            ),
        }
    ]


def test_local_client_actions_use_plain_chat_completion_contract():
    fake_client = FakeLocalClient(
        json.dumps({"type": "final_answer", "content": "structured answer", "tool_name": None, "arguments_json": "{}"})
    )
    client = LocalOpenAICompatibleClient(model="google/gemma-4-12b-qat", client=fake_client)

    result = client.complete_action(
        [
            {"role": "system", "content": "Return action JSON."},
            {"role": "user", "content": "Hello"},
        ],
        max_output_tokens=250,
    )

    assert result.action.content == "structured answer"
    assert fake_client.chat.completions.kwargs["max_tokens"] == 250
    assert "response_format" not in fake_client.chat.completions.kwargs
    assert fake_client.chat.completions.kwargs["messages"] == [
        {"role": "user", "content": "Instructions:\nReturn action JSON.\n\nUser message:\n\nHello"}
    ]


def test_local_client_uses_native_tool_calls_when_tools_are_provided():
    fake_client = FakeLocalClient(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_local",
                type="function",
                function=SimpleNamespace(name="calculator", arguments=json.dumps({"expression": "3 + 4"})),
            )
        ],
    )
    client = LocalOpenAICompatibleClient(model="google/gemma-4-12b-qat", client=fake_client)

    result = client.complete_action([{"role": "user", "content": "what is 3+4?"}], tools=[fake_calculator_tool()])

    assert result.action == ToolCallAction(type="tool_call", tool_name="calculator", arguments={"expression": "3 + 4"})
    assert fake_client.chat.completions.kwargs["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in fake_client.chat.completions.kwargs
    assert "tools" in fake_client.chat.completions.kwargs
    tool_names = {tool["function"]["name"] for tool in fake_client.chat.completions.kwargs["tools"]}
    assert tool_names == {"calculator"}
    assert result.metadata["action_transport"] == "provider_native"
    assert result.metadata["provider_tool_call"]["id"] == "call_local"


def test_local_client_receives_mcp_bridge_tools_as_function_tools():
    fake_client = FakeLocalClient(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_local_mcp",
                type="function",
                function=SimpleNamespace(
                    name="mcp_docs_search_docs",
                    arguments=json.dumps({"query": "local MCP"}),
                ),
            )
        ],
    )
    client = LocalOpenAICompatibleClient(model="google/gemma-4-12b-qat", client=fake_client)

    result = client.complete_action([{"role": "user", "content": "search docs"}], tools=[fake_mcp_bridge_tool()])

    assert result.action == ToolCallAction(
        type="tool_call",
        tool_name="mcp_docs_search_docs",
        arguments={"query": "local MCP"},
    )
    tool_names = {tool["function"]["name"] for tool in fake_client.chat.completions.kwargs["tools"]}
    assert "mcp_docs_search_docs" in tool_names


def test_local_client_falls_back_to_json_when_native_tools_are_rejected():
    fake_client = FakeLocalClient(
        json.dumps({"type": "final_answer", "content": "fallback answer", "tool_name": None, "arguments_json": "{}"}),
        fail_with_tools=True,
    )
    client = LocalOpenAICompatibleClient(model="google/gemma-4-12b-qat", client=fake_client)

    result = client.complete_action([{"role": "user", "content": "hello"}], tools=[fake_calculator_tool()])

    assert result.action.content == "fallback answer"
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert "tools unsupported" in result.metadata["native_tool_call_error"]
    assert len(fake_client.chat.completions.calls) == 2
    assert "tools" in fake_client.chat.completions.calls[0]
    assert "tools" not in fake_client.chat.completions.calls[1]
    assert "Respond with exactly one JSON object" in fake_client.chat.completions.calls[1]["messages"][0]["content"]


def test_llm_client_repairs_invalid_action_json():
    client = ScriptedLLMClient(
        [
            "plain prose",
            json.dumps({"type": "final_answer", "content": "repaired"}),
        ]
    )

    result = client.complete_action([{"role": "user", "content": "Hello"}], max_repair_attempts=1)

    assert result.action.content == "repaired"
    assert result.repair_attempts == 1
    assert "not valid JSON" in result.errors[0]
    assert "could not be parsed" in client.requests[1][-1]["content"]
    assert "Return exactly one valid JSON object" in client.requests[1][-1]["content"]


def test_llm_client_repairs_pre_advanced_plan_steps():
    def plan_response(status: str) -> str:
        return json.dumps(
            {
                "type": "plan",
                "content": None,
                "tool_name": None,
                "arguments_json": "{}",
                "plan_json": json.dumps(
                    {
                        "summary": "Make the change.",
                        "steps": [
                            {
                                "id": "1",
                                "title": "Make the change",
                                "description": "Complete the requested work.",
                                "status": status,
                            }
                        ],
                    }
                ),
                "step_update_json": "{}",
            }
        )

    client = ScriptedLLMClient(
        [plan_response("completed"), plan_response("pending")]
    )

    result = client.complete_action(
        [{"role": "user", "content": "Plan this"}],
        max_repair_attempts=1,
    )

    assert isinstance(result.action, PlanAction)
    assert result.action.plan.steps[0].status == "pending"
    assert result.repair_attempts == 1
    assert "status must be pending" in result.errors[0]


def test_llm_client_replaces_prior_repair_instead_of_accumulating_it():
    client = ScriptedLLMClient(
        [
            "FIRST_INVALID_RESPONSE",
            "SECOND_INVALID_RESPONSE",
            json.dumps({"type": "final_answer", "content": "repaired"}),
        ]
    )
    messages = [
        {
            "role": "system",
            "content": "<response_protocol>Existing action contract.</response_protocol>",
        },
        {"role": "user", "content": "Hello"},
    ]

    result = client.complete_action(messages, max_repair_attempts=2)

    assert result.action.content == "repaired"
    assert [len(request) for request in client.requests] == [2, 3, 3]
    assert "FIRST_INVALID_RESPONSE" in client.requests[1][-1]["content"]
    assert "FIRST_INVALID_RESPONSE" not in client.requests[2][-1]["content"]
    assert "SECOND_INVALID_RESPONSE" in client.requests[2][-1]["content"]
    assert "Return exactly one valid JSON object" not in client.requests[2][-1]["content"]


def test_create_llm_client_selects_deepseek_provider():
    client = create_llm_client(
        provider="deepseek",
        model="deepseek-v4-flash",
        openai_api_key=None,
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://api.deepseek.com",
        timeout_seconds=60,
        max_retries=2,
    )

    assert isinstance(client, DeepSeekChatCompletionsClient)


def test_create_llm_client_selects_local_provider():
    client = create_llm_client(
        provider="local",
        model="google/gemma-4-12b-qat",
        openai_api_key=None,
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        local_api_key=None,
        local_base_url="http://localhost:1234/v1",
        timeout_seconds=60,
        max_retries=2,
    )

    assert isinstance(client, LocalOpenAICompatibleClient)
    assert client.base_url == "http://localhost:1234/v1"
    assert client.model_capabilities.context_window_tokens == 131_072


def test_llm_provider_registry_exposes_provider_capabilities():
    openai_provider = LLM_PROVIDER_REGISTRY["openai"]
    deepseek_provider = LLM_PROVIDER_REGISTRY["deepseek"]
    local_provider = LLM_PROVIDER_REGISTRY["local"]

    assert openai_provider.capabilities.supports_structured_output is True
    assert openai_provider.capabilities.api_style == "responses"
    assert openai_provider.capabilities.supports_native_tool_calling is True
    assert openai_provider.capabilities.supports_hosted_mcp_tools is True
    assert deepseek_provider.capabilities.supports_json_mode is True
    assert deepseek_provider.capabilities.api_style == "chat_completions"
    assert deepseek_provider.capabilities.supports_native_tool_calling is True
    assert deepseek_provider.capabilities.supports_hosted_mcp_tools is False
    assert local_provider.capabilities.api_style == "chat_completions"
    assert local_provider.capabilities.supports_structured_output is False
    assert local_provider.capabilities.supports_native_tool_calling is True
    assert local_provider.capabilities.supports_hosted_mcp_tools is False


def test_resolve_model_capabilities_returns_context_window_and_reserve():
    openai_caps = resolve_model_capabilities("openai", "gpt-4.1-mini")
    deepseek_caps = resolve_model_capabilities("deepseek", "deepseek-v4-flash")
    local_caps = resolve_model_capabilities("local", "google/gemma-4-12b-qat")
    local_qwen_caps = resolve_model_capabilities("local", "qwen/qwen3.5-35b-a3b")

    assert openai_caps.context_window_tokens == 1_047_576
    assert openai_caps.default_response_reserve_tokens == 8_192
    assert openai_caps.max_input_tokens is None
    assert openai_caps.max_output_tokens == 32_768
    assert openai_caps.input_budget_tokens == 1_039_384
    assert deepseek_caps.context_window_tokens == 1_048_576
    assert deepseek_caps.default_response_reserve_tokens == 16_384
    assert deepseek_caps.max_input_tokens is None
    assert deepseek_caps.max_output_tokens == 393_216
    assert deepseek_caps.input_budget_tokens == 1_032_192
    assert local_caps.context_window_tokens == 262_144
    assert local_caps.default_response_reserve_tokens == 4_096
    assert local_qwen_caps.context_window_tokens == 262_144
    assert local_qwen_caps.default_response_reserve_tokens == 4_096


def test_conservative_model_capabilities_use_smallest_context_and_largest_reserve():
    openai_caps = resolve_model_capabilities("openai", "gpt-4.1-mini")
    local_caps = resolve_model_capabilities("local", "google/gemma-4-12b-qat")

    combined = conservative_model_capabilities([openai_caps, local_caps])

    assert combined.context_window_tokens == 262_144
    assert combined.default_response_reserve_tokens == 8_192
    assert combined.input_budget_tokens == 253_952


def test_fallback_chain_exposes_conservative_bound_model_capabilities():
    primary = ScriptedLLMClient(["primary"])
    primary.model_capabilities = resolve_model_capabilities("openai", "gpt-4.1-mini")
    fallback = ScriptedLLMClient(["fallback"])
    fallback.model_capabilities = resolve_model_capabilities("local", "google/gemma-4-12b-qat")

    capabilities = FallbackChain([primary, fallback]).model_capabilities

    assert capabilities is not None
    assert capabilities.context_window_tokens == 262_144
    assert capabilities.default_response_reserve_tokens == 8_192


def test_factory_attaches_model_capabilities_to_bound_client():
    client = create_llm_client(
        provider="local",
        model="custom-model",
        openai_api_key=None,
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        local_api_key="local",
        local_base_url="http://localhost:1234/v1",
        timeout_seconds=1,
        max_retries=0,
    )

    assert client.model_capabilities.context_window_tokens == 131_072


def test_factory_applies_local_context_before_fallback_aggregation():
    local = create_llm_client(
        provider="local",
        model="google/gemma-4-12b-qat",
        local_api_key="local",
        local_base_url="http://localhost:1234/v1",
        local_context_window_tokens=65_536,
        timeout_seconds=1,
        max_retries=0,
    )
    hosted = ScriptedLLMClient(["hosted"])
    hosted.model_capabilities = resolve_model_capabilities("openai", "gpt-4.1-mini")

    capabilities = FallbackChain([hosted, local]).model_capabilities

    assert local.model_capabilities.context_window_tokens == 65_536
    assert capabilities is not None
    assert capabilities.context_window_tokens == 65_536


def test_runtime_local_context_uses_deployment_limit_for_unknown_models():
    capabilities = resolve_runtime_model_capabilities(
        "local",
        "custom-model",
        local_context_window_tokens=262_144,
    )

    assert capabilities.context_window_tokens == 262_144


def test_runtime_local_context_never_exceeds_known_architecture_limit():
    capabilities = resolve_runtime_model_capabilities(
        "local",
        "gemma3:12b",
        local_context_window_tokens=262_144,
    )

    assert capabilities.context_window_tokens == 131_072


def test_runtime_local_context_rejects_zero_input_budget():
    try:
        resolve_runtime_model_capabilities(
            "local",
            "custom-model",
            local_context_window_tokens=4_096,
        )
    except ValueError as exc:
        assert "must exceed" in str(exc)
        assert "4096" in str(exc)
    else:
        raise AssertionError("Expected local context equal to the reserve to fail")


def test_resolve_model_capabilities_supports_known_family_aliases():
    caps = resolve_model_capabilities("openai", "gpt-4.1-mini-2025-04-14")

    assert caps.model == "gpt-4.1-mini-2025-04-14"
    assert caps.context_window_tokens == 1_047_576


def test_resolve_model_capabilities_uses_conservative_defaults_for_local_models():
    caps = resolve_model_capabilities("local", "ollama/custom-model:latest")

    assert caps.model == "ollama/custom-model:latest"
    assert caps.context_window_tokens == 131_072


def test_resolve_model_capabilities_rejects_unknown_models():
    try:
        resolve_model_capabilities("openai", "unknown-model")
    except ValueError as exc:
        assert "No token capability metadata" in str(exc)
        assert "src/chulk/llm/model_catalog.py" in str(exc)
    else:
        raise AssertionError("Expected unknown model capability lookup to fail")
