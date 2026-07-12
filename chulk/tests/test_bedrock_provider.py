"""Tests for the AWS Bedrock OpenAI-compatible provider adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from chulk import AgentConfig
from chulk.core.actions import FinalAnswerAction, ToolCallAction
from chulk.config import ConfigValueError, load_config
from chulk.llm.base import LLMConfigurationError, LLMError
from chulk.llm.capabilities import (
    BEDROCK_DEFAULT_CONTEXT_WINDOW_TOKENS,
    BEDROCK_DEFAULT_RESPONSE_RESERVE_TOKENS,
    resolve_model_capabilities,
)
from chulk.llm.factory import (
    LLMProviderConnection,
    create_llm_client,
    provider_capabilities,
    provider_connection_from_config,
    supported_llm_providers,
)
from chulk.llm.public import BedrockProvider
from chulk.llm.providers.bedrock import (
    BEDROCK_CAPABILITIES,
    BedrockOpenAICompatibleClient,
)
from chulk.main import create_cli_llm, format_config


MESSAGES = [{"role": "user", "content": "hello"}]
MODEL = "openai.gpt-oss-120b-1:0"
BASE_URL = "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"


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
    *,
    content: str | None,
    tool_calls: list[object] | None = None,
    usage: object = None,
) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
        usage=usage,
    )


def _tool_call(call_id: str = "call_1") -> object:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name="calculator",
            arguments=json.dumps({"expression": "2 + 2"}),
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


def _client(completions: FakeChatCompletions) -> BedrockOpenAICompatibleClient:
    return BedrockOpenAICompatibleClient(
        model=MODEL,
        api_key="test-key",
        base_url=BASE_URL,
        client=FakeClient(completions),
    )


@pytest.mark.parametrize(
    ("model", "api_key", "base_url", "missing_field"),
    [
        (" ", "key", BASE_URL, "model"),
        (MODEL, None, BASE_URL, "api_key"),
        (MODEL, "key", "", "base_url"),
    ],
)
def test_bedrock_requires_explicit_connection_fields_even_with_fake_client(
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    missing_field: str,
) -> None:
    with pytest.raises(LLMConfigurationError, match=missing_field) as raised:
        BedrockOpenAICompatibleClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=FakeClient(FakeChatCompletions()),
        )

    assert raised.value.provider == "bedrock"
    assert raised.value.code == "configuration_error"
    assert raised.value.retryable is False


def test_bedrock_preserves_caller_selected_connection() -> None:
    client = BedrockOpenAICompatibleClient(
        model=f" {MODEL} ",
        api_key=" test-key ",
        base_url=f" {BASE_URL} ",
        client=FakeClient(FakeChatCompletions()),
    )

    assert client.model == MODEL
    assert client.base_url == BASE_URL


def test_bedrock_text_request_uses_only_an_explicit_output_limit() -> None:
    completions = FakeChatCompletions(
        [_response(content="first"), _response(content="second")]
    )
    client = _client(completions)

    assert client.complete(MESSAGES) == "first"
    assert "max_tokens" not in completions.calls[0]
    assert client.complete(MESSAGES, max_output_tokens=123) == "second"
    assert completions.calls[1]["max_tokens"] == 123
    assert completions.calls[1]["model"] == MODEL


def test_bedrock_normalizes_chat_completions_usage() -> None:
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=4,
        total_tokens=14,
        prompt_cache_hit_tokens=3,
        prompt_cache_miss_tokens=7,
    )
    client = _client(FakeChatCompletions([_response(content="answer", usage=usage)]))

    response = client.complete_response(MESSAGES)

    assert response.provider == "bedrock"
    assert response.model == MODEL
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
    assert response.usage.cache_hit_input_tokens == 3
    assert response.usage.cache_miss_input_tokens == 7


def test_bedrock_streams_text_and_final_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
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

    chunks = list(_client(completions).stream_complete(MESSAGES, max_output_tokens=42))

    assert [chunk.text for chunk in chunks if chunk.type == "text_delta"] == [
        "hel",
        "lo",
    ]
    assert chunks[-1].type == "completed"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 5
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["max_tokens"] == 42


def test_bedrock_normalizes_one_native_tool_call() -> None:
    completions = FakeChatCompletions(
        [_response(content=None, tool_calls=[_tool_call()])]
    )

    result = _client(completions).complete_action(
        MESSAGES,
        tools=[_calculator_tool()],
    )

    assert result.action == ToolCallAction(
        type="tool_call",
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )
    assert result.metadata["action_transport"] == "provider_native"
    assert result.metadata["provider_tool_call"]["id"] == "call_1"
    assert completions.calls[0]["tool_choice"] == "auto"
    assert completions.calls[0]["parallel_tool_calls"] is False


def test_bedrock_rejects_parallel_native_actions_and_uses_json_fallback() -> None:
    fallback = json.dumps({"type": "final_answer", "content": "safe fallback"})
    completions = FakeChatCompletions(
        [
            _response(
                content=None,
                tool_calls=[_tool_call("call_1"), _tool_call("call_2")],
            ),
            _response(content=fallback),
        ]
    )

    result = _client(completions).complete_action(
        MESSAGES,
        tools=[_calculator_tool()],
    )

    assert result.action == FinalAnswerAction(
        type="final_answer",
        content="safe fallback",
    )
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert "multiple tool calls" in result.metadata["native_tool_call_error"]
    assert len(completions.calls) == 2


def test_bedrock_preserves_typed_provider_errors() -> None:
    error_type = type("RateLimitError", (Exception,), {})
    error = error_type("slow down")
    error.status_code = 429
    completions = FakeChatCompletions()
    completions.error = error

    with pytest.raises(LLMError) as raised:
        _client(completions).complete(MESSAGES)

    assert raised.value.provider == "bedrock"
    assert raised.value.model == MODEL
    assert raised.value.code == "rate_limit"
    assert raised.value.retryable is True
    assert raised.value.fallback_eligible is True


def test_bedrock_capabilities_are_explicit_and_conservative() -> None:
    assert BEDROCK_CAPABILITIES.api_style == "chat_completions"
    assert BEDROCK_CAPABILITIES.supports_streaming is True
    assert BEDROCK_CAPABILITIES.supports_native_tool_calling is True
    assert BEDROCK_CAPABILITIES.supports_json_mode is False
    assert BEDROCK_CAPABILITIES.supports_hosted_mcp_tools is False


def test_bedrock_environment_prefers_canonical_connection_values(tmp_path: Path) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "bedrock",
            "CHULK_MODEL": MODEL,
            "CHULK_BEDROCK_API_KEY": "canonical-key",
            "BEDROCK_API_KEY": "bedrock-key",
            "AWS_BEARER_TOKEN_BEDROCK": "aws-key",
            "CHULK_BEDROCK_BASE_URL": BASE_URL,
            "CHULK_BASE_URL": "https://legacy.example/openai/v1",
        }
    )

    assert config.bedrock_api_key == "canonical-key"
    assert config.bedrock_base_url == BASE_URL
    assert provider_connection_from_config("bedrock", config) == LLMProviderConnection(
        api_key="canonical-key",
        base_url=BASE_URL,
    )


@pytest.mark.parametrize(
    ("environment", "expected_key"),
    [
        (
            {
                "BEDROCK_API_KEY": "bedrock-key",
                "AWS_BEARER_TOKEN_BEDROCK": "aws-key",
            },
            "bedrock-key",
        ),
        ({"AWS_BEARER_TOKEN_BEDROCK": "aws-key"}, "aws-key"),
    ],
)
def test_bedrock_environment_supports_api_key_aliases_in_order(
    tmp_path: Path,
    environment: dict[str, str],
    expected_key: str,
) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "bedrock",
            "CHULK_MODEL": MODEL,
            "CHULK_BEDROCK_BASE_URL": BASE_URL,
            **environment,
        }
    )

    assert config.bedrock_api_key == expected_key


def test_bedrock_environment_supports_legacy_base_url_alias(tmp_path: Path) -> None:
    legacy_base_url = "https://legacy.example/openai/v1"
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "bedrock",
            "CHULK_MODEL": MODEL,
            "BEDROCK_API_KEY": "test-key",
            "CHULK_BASE_URL": legacy_base_url,
        }
    )

    assert config.bedrock_base_url == legacy_base_url


def test_bedrock_environment_requires_explicit_model(tmp_path: Path) -> None:
    with pytest.raises(ConfigValueError, match="CHULK_MODEL"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_PROVIDER": "bedrock",
                "CHULK_BEDROCK_API_KEY": "test-key",
                "CHULK_BEDROCK_BASE_URL": BASE_URL,
            }
        )


def test_bedrock_environment_requires_explicit_base_url(tmp_path: Path) -> None:
    with pytest.raises(ConfigValueError, match="CHULK_BEDROCK_BASE_URL"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_PROVIDER": "bedrock",
                "CHULK_MODEL": MODEL,
                "CHULK_BEDROCK_API_KEY": "test-key",
            }
        )


def test_bedrock_fallback_requires_explicit_base_url(tmp_path: Path) -> None:
    with pytest.raises(ConfigValueError, match="CHULK_BEDROCK_BASE_URL"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_PROVIDER": "openai",
                "CHULK_MODEL": "gpt-4.1-mini",
                "CHULK_LLM_FALLBACK_PROVIDERS": f"bedrock:{MODEL}",
                "CHULK_BEDROCK_API_KEY": "test-key",
            }
        )


def test_bedrock_factory_builds_shared_transport_with_bound_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = FakeChatCompletions()
    sdk_client = FakeClient(completions)
    captured: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return sdk_client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fake_constructor))

    client = create_llm_client(
        provider="bedrock",
        model=MODEL,
        connection=LLMProviderConnection(api_key="test-key", base_url=BASE_URL),
        timeout_seconds=12.5,
        max_retries=3,
    )

    assert isinstance(client, BedrockOpenAICompatibleClient)
    assert client.complete(MESSAGES) == "answer"
    assert client.model_capabilities == resolve_model_capabilities("bedrock", MODEL)
    assert captured == {
        "api_key": "test-key",
        "base_url": BASE_URL,
        "timeout": 12.5,
        "max_retries": 3,
    }


def test_bedrock_profile_and_model_capabilities_are_registered() -> None:
    assert "bedrock" in supported_llm_providers()
    assert provider_capabilities("bedrock") is BEDROCK_CAPABILITIES

    capabilities = resolve_model_capabilities("bedrock", MODEL)
    assert capabilities.context_window_tokens == BEDROCK_DEFAULT_CONTEXT_WINDOW_TOKENS
    assert (
        capabilities.default_response_reserve_tokens
        == BEDROCK_DEFAULT_RESPONSE_RESERVE_TOKENS
    )


def test_bedrock_agent_config_and_cli_use_public_provider_spec(tmp_path: Path) -> None:
    config = AgentConfig.bedrock(
        project_root=tmp_path,
        model=MODEL,
        api_key="test-key",
        base_url=BASE_URL,
    ).to_config()

    chain = create_cli_llm(config)

    assert config.llm_provider == "bedrock"
    assert config.model == MODEL
    assert config.bedrock_api_key == "test-key"
    assert config.bedrock_base_url == BASE_URL
    assert isinstance(chain.providers[0], BedrockProvider)
    assert chain.providers[0].model == MODEL


def test_bedrock_show_config_redacts_credentials(tmp_path: Path) -> None:
    secret = "bedrock-secret-that-must-not-be-rendered"
    config = AgentConfig.bedrock(
        project_root=tmp_path,
        model=MODEL,
        api_key=secret,
        base_url=BASE_URL,
    ).to_config()

    output = format_config(config)

    assert "bedrock_api_key: set" in output
    assert f"bedrock_base_url: {BASE_URL}" in output
    assert secret not in output
