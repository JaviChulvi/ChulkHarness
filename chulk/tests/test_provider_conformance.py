"""Shared behavior contract for built-in LLM provider transports."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chulk.cli import terminal as terminal_module
from chulk.core.actions import ToolCallAction
from chulk.llm import (
    LLMClientSettings,
    LLMProviderConnection,
    LLMProviderProfile,
    LLMUsage,
    LLM_PROVIDER_REGISTRY,
    LLMCapabilities,
    DeepSeekChatCompletionsClient,
    LocalOpenAICompatibleClient,
    provider_connection_from_config,
)
from chulk import runtime as runtime_module


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
def test_provider_native_single_tool_contract_disables_parallel_calls(client_type, model, provider):
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
    assert completions.calls[0]["parallel_tool_calls"] is False
    assert result.metadata["provider_tool_call"]["id"] == "call_1"


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
    assert completions.calls[0]["parallel_tool_calls"] is False
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
    assert not hasattr(settings, "openai_api_key")
    assert not hasattr(settings, "deepseek_api_key")
    assert not hasattr(settings, "local_api_key")


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
