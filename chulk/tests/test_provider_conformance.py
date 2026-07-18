"""Shared behavior contract for built-in LLM provider transports."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chulk import runtime as runtime_module
from chulk.cli import terminal as terminal_module
from chulk.core.actions import FinalAnswerAction, ToolCallAction
from chulk.llm import (
    FallbackChain,
    LLMClient,
    LLMClientSettings,
    LLMProviderConnection,
    LLMProviderProfile,
    LLMUsage,
    LLM_PROVIDER_REGISTRY,
    LLMCapabilities,
    DeepSeekChatCompletionsClient,
    LocalOpenAICompatibleClient,
    PlanningToolAvailability,
    create_llm_client,
    provider_connection_from_config,
)
from chulk.llm.capabilities import LLMModelCapabilities, MODEL_CAPABILITIES
from chulk.llm.tools import PLAN_TOOL_NAME
from chulk.testing import ScriptedLLMClient


PROVIDERS = [
    pytest.param(DeepSeekChatCompletionsClient, "deepseek-v4-flash", "deepseek", id="deepseek"),
    pytest.param(LocalOpenAICompatibleClient, "local/test-model", "local", id="local"),
]


class FakeChatCompletions:
    def __init__(
        self,
        *,
        responses: list[object] | None = None,
        stream_chunks: list[object] | None = None,
        fail_with_tools: bool = False,
    ) -> None:
        self.responses = list(responses or [_response(content="answer")])
        self.stream_chunks = list(stream_chunks or [])
        self.fail_with_tools = fail_with_tools
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools") and self.fail_with_tools:
            raise RuntimeError("native tools unsupported")
        if kwargs.get("stream"):
            return iter(self.stream_chunks)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class FakeClient:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class FakeAsyncChatCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _client(client_type, model: str, completions: FakeChatCompletions):
    return client_type(model=model, client=FakeClient(completions))


def _response(*, content: str | None, tool_calls: list[object] | None = None, usage: object = None) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=usage,
    )


def _tool_call(call_id: str, *, expression: str) -> object:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name="calculator",
            arguments=json.dumps({"expression": expression}),
        ),
    )


def _calculator_tool() -> object:
    return SimpleNamespace(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        args_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    )


@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
def test_provider_text_and_dynamic_output_limit_contract(client_type, model, provider):
    completions = FakeChatCompletions(responses=[_response(content="first"), _response(content="second")])
    client = _client(client_type, model, completions)

    assert client.complete([{"role": "user", "content": "hello"}]) == "first"
    assert "max_tokens" not in completions.calls[0]

    assert client.complete([{"role": "user", "content": "hello"}], max_output_tokens=987_654) == "second"
    assert completions.calls[1]["max_tokens"] == 987_654


@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
def test_provider_native_single_tool_contract_omits_optional_parallel_parameter(
    client_type, model, provider
):
    completions = FakeChatCompletions(
        responses=[_response(content=None, tool_calls=[_tool_call("call_1", expression="2 + 2")])]
    )
    client = _client(client_type, model, completions)

    result = client.complete_action(
        [{"role": "user", "content": "what is 2+2?"}],
        tools=[_calculator_tool()],
    )

    assert result.action == ToolCallAction(
        type="tool_call",
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )
    assert completions.calls[0]["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in completions.calls[0]
    assert result.metadata["provider_tool_call"]["id"] == "call_1"


@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
def test_provider_native_text_with_no_effective_tools_omits_tool_fields(
    client_type, model, provider
):
    completions = FakeChatCompletions(responses=[_response(content="native final")])
    client = _client(client_type, model, completions)

    result = client.complete_action(
        [{"role": "user", "content": "hello"}],
        tools=[],
        planning_tools=PlanningToolAvailability(),
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="native final")
    assert "tools" not in completions.calls[0]
    assert "tool_choice" not in completions.calls[0]
    assert result.metadata["action_transport"] == "provider_native"


@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
def test_provider_planning_policy_enables_native_transport_without_regular_tools(
    client_type, model, provider
):
    completions = FakeChatCompletions(responses=[_response(content="native final")])
    client = _client(client_type, model, completions)

    result = client.complete_action(
        [{"role": "user", "content": "plan this"}],
        planning_tools=PlanningToolAvailability(propose_plan=True),
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="native final")
    declarations = completions.calls[0]["tools"]
    assert [item["function"]["name"] for item in declarations] == [PLAN_TOOL_NAME]
    assert result.metadata["action_transport"] == "provider_native"


@pytest.mark.asyncio
@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
async def test_provider_async_native_tool_contract_omits_optional_parallel_parameter(
    client_type, model, provider
):
    async_completions = FakeAsyncChatCompletions(
        _response(content=None, tool_calls=[_tool_call("call_1", expression="2 + 2")])
    )
    client = client_type(
        model=model,
        client=FakeClient(FakeChatCompletions()),
        async_client=FakeClient(async_completions),
    )

    result = await client.acomplete_action(
        [{"role": "user", "content": "what is 2+2?"}],
        tools=[_calculator_tool()],
    )

    assert result.action == ToolCallAction(
        type="tool_call",
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )
    assert async_completions.calls[0]["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in async_completions.calls[0]


@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
def test_provider_json_fallback_contract(client_type, model, provider):
    fallback_json = json.dumps({"type": "final_answer", "content": "fallback"})
    completions = FakeChatCompletions(
        responses=[_response(content=fallback_json)],
        fail_with_tools=True,
    )
    client = _client(client_type, model, completions)

    result = client.complete_action(
        [{"role": "user", "content": "hello"}],
        tools=[_calculator_tool()],
        max_output_tokens=321,
    )

    assert result.action.content == "fallback"
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert "native tools unsupported" in result.metadata["native_tool_call_error"]
    assert "parallel_tool_calls" not in completions.calls[0]
    assert completions.calls[0]["max_tokens"] == 321
    assert "tools" not in completions.calls[1]
    assert completions.calls[1]["max_tokens"] == 321


@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
def test_provider_rejects_unexpected_parallel_response_and_falls_back(client_type, model, provider):
    fallback_json = json.dumps({"type": "final_answer", "content": "safe fallback"})
    completions = FakeChatCompletions(
        responses=[
            _response(
                content=None,
                tool_calls=[
                    _tool_call("call_1", expression="2 + 2"),
                    _tool_call("call_2", expression="3 + 3"),
                ],
            ),
            _response(content=fallback_json),
        ]
    )
    client = _client(client_type, model, completions)

    result = client.complete_action(
        [{"role": "user", "content": "calculate"}],
        tools=[_calculator_tool()],
    )

    assert result.action.content == "safe fallback"
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert "multiple tool calls" in result.metadata["native_tool_call_error"]
    assert len(completions.calls) == 2


@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
def test_provider_streaming_contract(client_type, model, provider):
    usage = SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6)
    completions = FakeChatCompletions(
        stream_chunks=[
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"), finish_reason=None)],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"), finish_reason="stop")],
                usage=usage,
            ),
        ]
    )
    client = _client(client_type, model, completions)

    chunks = list(client.stream_complete([{"role": "user", "content": "hello"}], max_output_tokens=42))

    assert [chunk.text for chunk in chunks if chunk.type == "text_delta"] == ["hel", "lo"]
    assert chunks[-1].type == "completed"
    assert chunks[-1].usage == LLMUsage(
        input_tokens=4,
        output_tokens=2,
        total_tokens=6,
        cache_miss_input_tokens=4,
        estimated=False,
        cache_split_estimated=True,
        source="provider",
        raw={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    )
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["max_tokens"] == 42


@pytest.mark.parametrize(("client_type", "model", "provider"), PROVIDERS)
def test_provider_usage_contract(client_type, model, provider):
    usage = SimpleNamespace(
        prompt_tokens=30,
        completion_tokens=5,
        total_tokens=35,
        prompt_cache_hit_tokens=10,
        prompt_cache_miss_tokens=20,
    )
    completions = FakeChatCompletions(responses=[_response(content="answer", usage=usage)])
    client = _client(client_type, model, completions)

    response = client.complete_response([{"role": "user", "content": "hello"}])

    assert response.provider == provider
    assert response.model == model
    assert response.usage is not None
    assert response.usage.input_tokens == 30
    assert response.usage.output_tokens == 5
    assert response.usage.cache_hit_input_tokens == 10
    assert response.usage.cache_miss_input_tokens == 20


def test_provider_profile_binds_typed_connection_from_existing_config_fields():
    config = SimpleNamespace(
        openai_api_key="openai-key",
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://deepseek.example/v1",
        local_api_key="local-key",
        local_base_url="http://localhost:11434/v1",
    )

    connection = provider_connection_from_config("deepseek", config)
    settings = LLMClientSettings(
        model="deepseek-v4-flash",
        connection=connection,
        timeout_seconds=5,
        max_retries=1,
    )

    assert settings.connection == LLMProviderConnection(
        api_key="deepseek-key",
        base_url="https://deepseek.example/v1",
    )
    assert settings.openai_api_key is None
    assert settings.deepseek_api_key is None
    assert settings.local_api_key is None


def test_llm_client_settings_accepts_original_positional_contract():
    settings = LLMClientSettings(
        "legacy-model",
        "openai-key",
        "deepseek-key",
        "https://deepseek.example/v1",
        "local-key",
        "http://localhost:11434/v1",
        12.5,
        4,
    )

    assert settings.model == "legacy-model"
    assert settings.openai_api_key == "openai-key"
    assert settings.deepseek_api_key == "deepseek-key"
    assert settings.deepseek_base_url == "https://deepseek.example/v1"
    assert settings.local_api_key == "local-key"
    assert settings.local_base_url == "http://localhost:11434/v1"
    assert settings.timeout_seconds == 12.5
    assert settings.max_retries == 4
    assert settings.connection == LLMProviderConnection()


def test_llm_client_settings_accepts_original_keyword_contract():
    settings = LLMClientSettings(
        model="legacy-model",
        openai_api_key="openai-key",
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://deepseek.example/v1",
        local_api_key="local-key",
        local_base_url="http://localhost:11434/v1",
        timeout_seconds=8,
        max_retries=1,
    )

    assert settings.openai_api_key == "openai-key"
    assert settings.deepseek_api_key == "deepseek-key"
    assert settings.local_api_key == "local-key"
    assert settings.timeout_seconds == 8
    assert settings.max_retries == 1


def test_custom_provider_callback_receives_original_settings_fields(monkeypatch):
    provider = "legacy-custom"
    model = "legacy-model"
    captured: dict[str, object] = {}
    client = ScriptedLLMClient(["answer"])

    def create_legacy_client(settings: LLMClientSettings):
        captured.update(
            {
                "model": settings.model,
                "openai_api_key": settings.openai_api_key,
                "deepseek_api_key": settings.deepseek_api_key,
                "deepseek_base_url": settings.deepseek_base_url,
                "local_api_key": settings.local_api_key,
                "local_base_url": settings.local_base_url,
                "timeout_seconds": settings.timeout_seconds,
                "max_retries": settings.max_retries,
            }
        )
        return client

    monkeypatch.setitem(
        LLM_PROVIDER_REGISTRY,
        provider,
        LLMProviderProfile(
            name=provider,
            capabilities=LLMCapabilities(),
            create_client=create_legacy_client,
        ),
    )
    monkeypatch.setitem(
        MODEL_CAPABILITIES,
        (provider, model),
        LLMModelCapabilities(
            provider=provider,
            model=model,
            context_window_tokens=8_192,
            default_response_reserve_tokens=2_048,
        ),
    )

    result = create_llm_client(
        provider=provider,
        model=model,
        openai_api_key="openai-key",
        deepseek_api_key="deepseek-key",
        deepseek_base_url="https://deepseek.example/v1",
        local_api_key="local-key",
        local_base_url="http://localhost:11434/v1",
        timeout_seconds=15,
        max_retries=3,
    )

    assert result is client
    assert captured == {
        "model": model,
        "openai_api_key": "openai-key",
        "deepseek_api_key": "deepseek-key",
        "deepseek_base_url": "https://deepseek.example/v1",
        "local_api_key": "local-key",
        "local_base_url": "http://localhost:11434/v1",
        "timeout_seconds": 15,
        "max_retries": 3,
    }


def test_mcp_routing_uses_provider_capability_instead_of_provider_name(monkeypatch):
    profile = LLMProviderProfile(
        name="hosted-compatible",
        capabilities=LLMCapabilities(supports_hosted_mcp_tools=True),
        create_client=lambda settings: None,  # type: ignore[arg-type, return-value]
    )
    monkeypatch.setitem(LLM_PROVIDER_REGISTRY, profile.name, profile)
    config = SimpleNamespace(
        llm_provider=profile.name,
        llm_fallback_providers=(),
        mcp_servers=(object(),),
    )

    assert runtime_module._mcp_bridge_required(config, config.mcp_servers) is False
    assert runtime_module._mcp_provider_path(config, config.mcp_servers) == "hosted"
    assert terminal_module._mcp_provider_path(config) == "hosted"


def test_mcp_routing_registers_a_bridge_for_an_injected_mixed_fallback_chain():
    class HostedClient(LLMClient):
        capabilities = LLMCapabilities(supports_hosted_mcp_tools=True)

    class BridgeClient(LLMClient):
        capabilities = LLMCapabilities(supports_hosted_mcp_tools=False)

    config = SimpleNamespace(
        llm_provider="openai",
        llm_fallback_providers=(),
    )
    chain = FallbackChain([HostedClient(), BridgeClient()])
    servers = (object(),)

    assert runtime_module._mcp_bridge_required(
        config,
        servers,
        llm_client=chain,
    ) is True
    assert runtime_module._mcp_provider_path(
        config,
        servers,
        llm_client=chain,
    ) == "hosted+bridge"
