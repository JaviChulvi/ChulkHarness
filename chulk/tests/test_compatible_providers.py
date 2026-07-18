"""Tests for hosted OpenAI-compatible provider adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chulk import AgentConfig
from chulk.config import ConfigValueError, load_config
from chulk.core.actions import ToolCallAction
from chulk.llm import OpenAICompatibleProvider, OpenRouterProvider, supported_llm_providers
from chulk.llm.base import LLMConfigurationError, LLMError
from chulk.llm.factory import LLMProviderConnection, provider_connection_from_config
from chulk.llm.providers import compatible as compatible_module
from chulk.llm.providers.compatible import (
    DEFAULT_OPENROUTER_BASE_URL,
    HOSTED_OPENAI_COMPATIBLE_CAPABILITIES,
    OPENROUTER_CAPABILITIES,
    HostedOpenAICompatibleClient,
    OpenRouterChatCompletionsClient,
    openrouter_default_headers,
)
from chulk.main import create_cli_llm


MESSAGES = [{"role": "user", "content": "hello"}]


class FakeChatCompletions:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = list(responses or [_response(content="answer")])
        self.calls: list[dict] = []
        self.error: Exception | None = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return iter(self.responses)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class FakeClient:
    def __init__(self, completions: FakeChatCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _response(
    *, content: str | None, tool_calls: list[object] | None = None, usage=None
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        usage=usage,
    )


def _tool_call() -> object:
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(
            name="calculator", arguments=json.dumps({"expression": "2 + 2"})
        ),
    )


def _calculator_tool() -> object:
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


def _hosted(completions: FakeChatCompletions) -> HostedOpenAICompatibleClient:
    return HostedOpenAICompatibleClient(
        model="vendor/model",
        api_key="test-key",
        base_url="https://models.example/v1",
        client=FakeClient(completions),
    )


def _openrouter(completions: FakeChatCompletions) -> OpenRouterChatCompletionsClient:
    return OpenRouterChatCompletionsClient(
        model="vendor/model",
        api_key="test-key",
        client=FakeClient(completions),
    )


@pytest.mark.parametrize("client_factory", [_hosted, _openrouter])
def test_compatible_providers_follow_text_and_output_limit_contract(
    client_factory,
) -> None:
    completions = FakeChatCompletions(
        [_response(content="first"), _response(content="second")]
    )
    client = client_factory(completions)

    assert client.complete(MESSAGES) == "first"
    assert "max_tokens" not in completions.calls[0]
    assert client.complete(MESSAGES, max_output_tokens=1234) == "second"
    assert completions.calls[1]["max_tokens"] == 1234


@pytest.mark.parametrize("client_factory", [_hosted, _openrouter])
def test_compatible_providers_follow_native_single_tool_contract(
    client_factory,
) -> None:
    completions = FakeChatCompletions(
        [_response(content=None, tool_calls=[_tool_call()])]
    )
    client = client_factory(completions)

    result = client.complete_action(MESSAGES, tools=[_calculator_tool()])

    assert result.action == ToolCallAction(
        type="tool_call",
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )
    assert "parallel_tool_calls" not in completions.calls[0]
    assert result.metadata["action_transport"] == "provider_native"


@pytest.mark.parametrize("client_factory", [_hosted, _openrouter])
@pytest.mark.parametrize(
    ("unsupported_kwargs", "field"),
    [
        ({"hosted_mcp_servers": [object()]}, "hosted_mcp_servers"),
        ({"mcp_approval_callback": lambda _request: True}, "mcp_approval_callback"),
    ],
)
def test_compatible_providers_reject_unsupported_hosted_mcp_arguments(
    client_factory,
    unsupported_kwargs: dict,
    field: str,
) -> None:
    completions = FakeChatCompletions()
    client = client_factory(completions)

    with pytest.raises(LLMError, match=field) as raised:
        client.complete_action(MESSAGES, **unsupported_kwargs)

    assert raised.value.code == "unsupported_feature"
    assert completions.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("client_factory", [_hosted, _openrouter])
async def test_compatible_providers_async_reject_unsupported_hosted_mcp_arguments(
    client_factory,
) -> None:
    completions = FakeChatCompletions()
    client = client_factory(completions)

    with pytest.raises(LLMError, match="hosted_mcp_servers") as raised:
        await client.acomplete_action(MESSAGES, hosted_mcp_servers=[object()])

    assert raised.value.code == "unsupported_feature"
    assert completions.calls == []


@pytest.mark.parametrize(
    ("factory", "provider"),
    [(_hosted, "openai-compatible"), (_openrouter, "openrouter")],
)
def test_compatible_providers_normalize_usage(factory, provider: str) -> None:
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    client = factory(FakeChatCompletions([_response(content="answer", usage=usage)]))

    response = client.complete_response(MESSAGES)

    assert response.provider == provider
    assert response.model == "vendor/model"
    assert response.usage is not None
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 8


@pytest.mark.parametrize("client_factory", [_hosted, _openrouter])
def test_compatible_providers_follow_streaming_contract(client_factory) -> None:
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    completions = FakeChatCompletions(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hel"), finish_reason=None
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="lo"), finish_reason="stop"
                    )
                ],
                usage=usage,
            ),
        ]
    )
    client = client_factory(completions)

    chunks = list(client.stream_complete(MESSAGES, max_output_tokens=64))

    assert [chunk.text for chunk in chunks if chunk.type == "text_delta"] == [
        "hel",
        "lo",
    ]
    assert chunks[-1].type == "completed"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 7
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["max_tokens"] == 64


@pytest.mark.parametrize(
    ("factory", "provider"),
    [(_hosted, "openai-compatible"), (_openrouter, "openrouter")],
)
def test_compatible_providers_preserve_typed_request_errors(
    factory, provider: str
) -> None:
    error_type = type("RateLimitError", (Exception,), {})
    error = error_type("slow down")
    error.status_code = 429
    completions = FakeChatCompletions()
    completions.error = error
    client = factory(completions)

    with pytest.raises(LLMError) as raised:
        client.complete(MESSAGES)

    assert raised.value.provider == provider
    assert raised.value.model == "vendor/model"
    assert raised.value.code == "rate_limit"
    assert raised.value.retryable is True
    assert raised.value.fallback_eligible is True


@pytest.mark.parametrize(
    ("kwargs", "provider", "field"),
    [
        (
            {"model": "", "api_key": "key", "base_url": "https://models.example/v1"},
            "openai-compatible",
            "model",
        ),
        (
            {
                "model": "vendor/model",
                "api_key": None,
                "base_url": "https://models.example/v1",
            },
            "openai-compatible",
            "api_key",
        ),
        (
            {"model": "vendor/model", "api_key": "key", "base_url": ""},
            "openai-compatible",
            "base_url",
        ),
    ],
)
def test_hosted_compatible_requires_explicit_connection_fields(
    kwargs, provider: str, field: str
) -> None:
    with pytest.raises(LLMConfigurationError, match=field) as raised:
        HostedOpenAICompatibleClient(**kwargs, client=FakeClient(FakeChatCompletions()))

    assert raised.value.provider == provider
    assert raised.value.code == "configuration_error"


def test_openrouter_requires_model_and_api_key_even_with_fake_client() -> None:
    fake = FakeClient(FakeChatCompletions())

    with pytest.raises(LLMConfigurationError, match="model"):
        OpenRouterChatCompletionsClient(model=" ", api_key="key", client=fake)
    with pytest.raises(LLMConfigurationError, match="api_key"):
        OpenRouterChatCompletionsClient(model="vendor/model", api_key=None, client=fake)


def test_missing_required_values_raise_typed_configuration_errors() -> None:
    fake = FakeClient(FakeChatCompletions())

    with pytest.raises(LLMConfigurationError, match="model"):
        HostedOpenAICompatibleClient(client=fake)
    with pytest.raises(LLMConfigurationError, match="model"):
        OpenRouterChatCompletionsClient(client=fake)


def test_openrouter_defaults_and_attribution_headers_are_explicit(monkeypatch) -> None:
    captured: dict = {}
    fake_client = FakeClient(FakeChatCompletions())

    def fake_sdk_client(**kwargs):
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(compatible_module, "_openai_sdk_client", fake_sdk_client)

    client = OpenRouterChatCompletionsClient(
        model="vendor/model",
        api_key="test-key",
        site_url=" https://app.example ",
        app_name=" Chulk Harness ",
        timeout_seconds=7,
        max_retries=1,
    )

    assert client.base_url == DEFAULT_OPENROUTER_BASE_URL
    assert client.default_headers == {
        "HTTP-Referer": "https://app.example",
        "X-OpenRouter-Title": "Chulk Harness",
    }
    assert captured == {
        "api_key": "test-key",
        "base_url": DEFAULT_OPENROUTER_BASE_URL,
        "timeout_seconds": 7,
        "max_retries": 1,
        "default_headers": client.default_headers,
        "model": "vendor/model",
    }


def test_openrouter_header_builder_omits_blank_optional_values() -> None:
    assert openrouter_default_headers(site_url=" ", app_name=None) == {}


def test_capabilities_are_explicit_and_conservative() -> None:
    for capabilities in (
        HOSTED_OPENAI_COMPATIBLE_CAPABILITIES,
        OPENROUTER_CAPABILITIES,
    ):
        assert capabilities.api_style == "chat_completions"
        assert capabilities.supports_streaming is True
        assert capabilities.supports_native_tool_calling is True
        assert capabilities.supports_json_mode is False
        assert capabilities.supports_hosted_mcp_tools is False


def test_compatible_provider_environment_uses_explicit_models_and_connections(tmp_path) -> None:
    generic = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "openai-compatible",
            "CHULK_MODEL": "vendor/generic-model",
            "CHULK_OPENAI_COMPATIBLE_API_KEY": "generic-key",
            "CHULK_OPENAI_COMPATIBLE_BASE_URL": "https://models.example/v1",
        }
    )
    router = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "openrouter",
            "CHULK_MODEL": "vendor/router-model",
            "OPENROUTER_API_KEY": "router-key",
        }
    )

    assert provider_connection_from_config("openai-compatible", generic) == LLMProviderConnection(
        api_key="generic-key",
        base_url="https://models.example/v1",
    )
    assert provider_connection_from_config("openrouter", router) == LLMProviderConnection(
        api_key="router-key",
        base_url=DEFAULT_OPENROUTER_BASE_URL,
    )
    assert {"openai-compatible", "openrouter"} <= supported_llm_providers()


@pytest.mark.parametrize("provider", ["openai-compatible", "openrouter"])
def test_hosted_compatible_environment_requires_an_explicit_model(tmp_path, provider: str) -> None:
    with pytest.raises(ConfigValueError, match="CHULK_MODEL"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_PROVIDER": provider,
            }
        )


def test_public_compatible_builders_and_cli_specs_use_the_same_config(tmp_path) -> None:
    generic = AgentConfig.openai_compatible(
        project_root=tmp_path,
        model="vendor/generic-model",
        api_key="generic-key",
        base_url="https://models.example/v1",
    ).to_config()
    router = AgentConfig.openrouter(
        project_root=tmp_path,
        model="vendor/router-model",
        api_key="router-key",
    ).to_config()

    generic_chain = create_cli_llm(generic)
    router_chain = create_cli_llm(router)
    assert isinstance(generic_chain.providers[0], OpenAICompatibleProvider)
    assert isinstance(router_chain.providers[0], OpenRouterProvider)
    assert generic.openai_compatible_api_key == "generic-key"
    assert generic.openai_compatible_base_url == "https://models.example/v1"
    assert router.openrouter_api_key == "router-key"
    assert router.openrouter_base_url == DEFAULT_OPENROUTER_BASE_URL


def test_compatible_fallback_entries_require_models(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit model"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_FALLBACK_PROVIDERS": "openrouter",
            }
        )
