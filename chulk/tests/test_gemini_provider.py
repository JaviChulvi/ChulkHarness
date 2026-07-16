"""Tests for the native Gemini GenerateContent provider."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from chulk._sdk.config import AgentConfig
from chulk.config import ConfigValueError, load_config
from chulk.core.actions import FinalAnswerAction, ToolCallAction
from chulk.llm import GeminiGenerateContentClient as ExportedGeminiGenerateContentClient
from chulk.llm import GeminiProvider as ExportedGeminiProvider
from chulk.llm.base import LLMConfigurationError, LLMError
from chulk.llm.capabilities import (
    GEMINI_DEFAULT_CONTEXT_WINDOW_TOKENS,
    GEMINI_DEFAULT_RESPONSE_RESERVE_TOKENS,
    resolve_model_capabilities,
)
from chulk.llm.factory import (
    LLMProviderConnection,
    create_llm_client,
    provider_capabilities,
    provider_connection_from_config,
    supported_llm_providers,
)
from chulk.llm.providers.gemini import (
    GEMINI_CAPABILITIES,
    GeminiGenerateContentClient,
)
from chulk.llm.public import GeminiProvider
from chulk.llm.usage import LLMUsage
from chulk.main import create_cli_llm, format_config


MESSAGES = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Hello"},
]


class FakeModels:
    def __init__(
        self,
        *,
        responses: list[object] | None = None,
        stream_chunks: list[object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.stream_chunks = list(stream_chunks or [])
        self.error = error
        self.generate_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError("No fake Gemini response configured")
        return self.responses.pop(0)

    def generate_content_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return iter(self.stream_chunks)


def _client(
    models: FakeModels,
    *,
    model: str = "gemini-test",
) -> GeminiGenerateContentClient:
    return GeminiGenerateContentClient(
        model=model,
        client=SimpleNamespace(models=models),
    )


def _response(
    *,
    text: str | None = None,
    function_calls: list[object] | None = None,
    usage: object = None,
    finish_reason: str | None = None,
) -> object:
    candidates = []
    if finish_reason is not None:
        candidates = [SimpleNamespace(finish_reason=finish_reason)]
    return SimpleNamespace(
        text=text,
        function_calls=function_calls or [],
        usage_metadata=usage,
        candidates=candidates,
    )


def _function_call(name: str, arguments: dict) -> object:
    return SimpleNamespace(name=name, args=arguments)


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


def test_gemini_requires_an_api_key_without_an_injected_client() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        GeminiGenerateContentClient(model="gemini-test")


def test_gemini_rejects_blank_api_key_without_an_injected_client() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        GeminiGenerateContentClient(model="gemini-test", api_key="   ")


def test_gemini_strips_api_key_and_shapes_sdk_client_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    fake_genai = SimpleNamespace(Client=fake_constructor)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))

    GeminiGenerateContentClient(
        model=" gemini-test ",
        api_key="  test-key  ",
        base_url=" https://gemini.example ",
        timeout_seconds=12.5,
        max_retries=3,
    )

    assert captured == {
        "api_key": "test-key",
        "http_options": {
            "base_url": "https://gemini.example",
            "timeout": 12_500,
            "retry_options": {"attempts": 4},
        },
    }


def test_gemini_text_request_uses_native_message_shape_and_dynamic_output_limit() -> None:
    models = FakeModels(
        responses=[_response(text="first"), _response(text="second")],
    )
    client = _client(models)

    assert client.complete(MESSAGES) == "first"
    assert models.generate_calls[0] == {
        "model": "gemini-test",
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "config": {"system_instruction": "Be concise."},
    }

    assert client.complete(MESSAGES, max_output_tokens=987_654) == "second"
    assert models.generate_calls[1]["config"]["max_output_tokens"] == 987_654


def test_gemini_normalizes_provider_usage() -> None:
    usage = SimpleNamespace(
        prompt_token_count=30,
        candidates_token_count=9,
        total_token_count=41,
        cached_content_token_count=12,
        thoughts_token_count=2,
    )
    models = FakeModels(responses=[_response(text="answer", usage=usage)])

    response = _client(models).complete_response(MESSAGES)

    assert response.provider == "gemini"
    assert response.model == "gemini-test"
    assert response.usage == LLMUsage(
        input_tokens=30,
        output_tokens=11,
        total_tokens=41,
        cached_input_tokens=12,
        cache_hit_input_tokens=12,
        cache_miss_input_tokens=18,
        reasoning_tokens=2,
        estimated=False,
        source="provider",
        raw={
            "cached_content_token_count": 12,
            "candidates_token_count": 9,
            "prompt_token_count": 30,
            "thoughts_token_count": 2,
            "total_token_count": 41,
        },
    )


def test_gemini_bills_tool_use_and_thinking_tokens_in_the_right_buckets() -> None:
    usage = SimpleNamespace(
        prompt_token_count=199_999,
        tool_use_prompt_token_count=2,
        candidates_token_count=1,
        thoughts_token_count=1,
        total_token_count=200_003,
        cached_content_token_count=0,
    )
    models = FakeModels(responses=[_response(text="answer", usage=usage)])

    response = _client(
        models,
        model="gemini-3.1-pro-preview",
    ).complete_response(MESSAGES)

    assert response.usage is not None
    assert response.usage.input_tokens == 200_001
    assert response.usage.cache_miss_input_tokens == 200_001
    assert response.usage.output_tokens == 2
    assert response.usage.reasoning_tokens == 1
    assert response.usage.total_tokens == 200_003
    assert response.cost is not None
    assert response.cost.input_cost == Decimal("0.800004")
    assert response.cost.output_cost == Decimal("0.000036")
    assert response.cost.amount == Decimal("0.800040")


def test_gemini_native_streaming_yields_text_and_final_usage() -> None:
    usage = SimpleNamespace(
        prompt_token_count=4,
        candidates_token_count=2,
        total_token_count=6,
        cached_content_token_count=0,
        thoughts_token_count=0,
    )
    models = FakeModels(
        stream_chunks=[
            _response(text="hello "),
            _response(text="world", usage=usage, finish_reason="STOP"),
        ]
    )

    chunks = list(_client(models).stream_complete(MESSAGES, max_output_tokens=42))

    assert [chunk.text for chunk in chunks if chunk.type == "text_delta"] == ["hello ", "world"]
    assert chunks[-1].type == "completed"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 6
    assert chunks[-1].metadata["finish_reason"] == "STOP"
    assert models.stream_calls[0]["config"]["max_output_tokens"] == 42


def test_gemini_json_action_mode_uses_schema_without_native_tools() -> None:
    payload = json.dumps(
        {
            "type": "final_answer",
            "content": "done",
            "tool_name": None,
            "arguments_json": "{}",
            "plan_json": "{}",
            "step_update_json": "{}",
        }
    )
    models = FakeModels(responses=[_response(text=payload)])

    result = _client(models).complete_action(MESSAGES, max_output_tokens=512)

    assert result.action == FinalAnswerAction(type="final_answer", content="done")
    assert result.metadata["action_transport"] == "chulk_json"
    config = models.generate_calls[0]["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"]["additionalProperties"] is False
    assert config["max_output_tokens"] == 512
    assert "tools" not in config


def test_gemini_native_single_function_call_is_normalized_to_one_action() -> None:
    models = FakeModels(
        responses=[
            _response(
                function_calls=[
                    _function_call("calculator", {"expression": "2 + 2"}),
                ]
            )
        ]
    )

    result = _client(models).complete_action(
        MESSAGES,
        tools=[_calculator_tool()],
    )

    assert result.action == ToolCallAction(
        type="tool_call",
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )
    assert result.metadata["action_transport"] == "provider_native"
    assert result.metadata["provider_tool_call"]["name"] == "calculator"
    config = models.generate_calls[0]["config"]
    assert config["automatic_function_calling"] == {"disable": True}
    assert config["tool_config"] == {"function_calling_config": {"mode": "AUTO"}}
    declarations = config["tools"][0]["function_declarations"]
    assert any(item["name"] == "calculator" for item in declarations)
    assert any(item["name"] == "chulk_propose_plan" for item in declarations)


def test_gemini_native_text_is_a_direct_final_answer() -> None:
    models = FakeModels(responses=[_response(text="A direct answer")])

    result = _client(models).complete_action(MESSAGES, tools=[_calculator_tool()])

    assert result.action == FinalAnswerAction(type="final_answer", content="A direct answer")
    assert result.metadata["action_transport"] == "provider_native"
    assert result.metadata["provider_tool_call"] is None
    assert len(models.generate_calls) == 1


def test_gemini_rejects_multiple_function_calls_and_uses_json_fallback() -> None:
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
    models = FakeModels(
        responses=[
            _response(
                function_calls=[
                    _function_call("calculator", {"expression": "2 + 2"}),
                    _function_call("calculator", {"expression": "3 + 3"}),
                ]
            ),
            _response(text=fallback_payload),
        ]
    )

    result = _client(models).complete_action(MESSAGES, tools=[_calculator_tool()])

    assert result.action == FinalAnswerAction(type="final_answer", content="safe fallback")
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert "multiple function calls" in result.metadata["native_tool_call_error"]
    assert len(models.generate_calls) == 2
    assert "tools" in models.generate_calls[0]["config"]
    assert models.generate_calls[1]["config"]["response_mime_type"] == "application/json"


class GeminiSDKError(Exception):
    def __init__(self, message: str, *, code: int) -> None:
        super().__init__(message)
        self.code = code


class GeminiHTTPError(Exception):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = status_code
        self.response = SimpleNamespace(status_code=status_code)


class FailOnceModels(FakeModels):
    def __init__(self, *, error: Exception, responses: list[object]) -> None:
        super().__init__(responses=responses)
        self.one_time_error: Exception | None = error

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.one_time_error is not None:
            error = self.one_time_error
            self.one_time_error = None
            raise error
        if not self.responses:
            raise AssertionError("No fake Gemini response configured")
        return self.responses.pop(0)


@pytest.mark.parametrize(
    ("error", "code", "retryable", "fallback_eligible"),
    [
        (GeminiSDKError("API key not valid", code=400), "authentication_error", False, False),
        (GeminiSDKError("quota exceeded", code=429), "rate_limit", True, True),
        (GeminiSDKError("backend unavailable", code=503), "server_error", True, True),
    ],
)
def test_gemini_sdk_failures_have_typed_provider_metadata(
    error: Exception,
    code: str,
    retryable: bool,
    fallback_eligible: bool,
) -> None:
    models = FakeModels(error=error)

    with pytest.raises(LLMError) as raised:
        _client(models).complete(MESSAGES)

    assert raised.value.code == code
    assert raised.value.retryable is retryable
    assert raised.value.fallback_eligible is fallback_eligible
    assert raised.value.provider == "gemini"
    assert raised.value.model == "gemini-test"
    assert len(models.generate_calls) == 1


def test_gemini_http_400_invalid_key_overrides_generic_classification() -> None:
    models = FakeModels(
        error=GeminiHTTPError("API key not valid", status_code=400),
    )

    with pytest.raises(LLMError) as raised:
        _client(models).complete(MESSAGES)

    assert raised.value.code == "authentication_error"
    assert raised.value.retryable is False
    assert raised.value.fallback_eligible is False


def test_gemini_http_400_unsupported_function_calling_uses_json_fallback() -> None:
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
    models = FailOnceModels(
        error=GeminiHTTPError(
            "Function calling is not supported for this model",
            status_code=400,
        ),
        responses=[_response(text=fallback_payload)],
    )

    result = _client(models).complete_action(MESSAGES, tools=[_calculator_tool()])

    assert result.action == FinalAnswerAction(
        type="final_answer",
        content="safe fallback",
    )
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert "function calling is not supported" in result.metadata[
        "native_tool_call_error"
    ].lower()
    assert "tools" in models.generate_calls[0]["config"]
    assert models.generate_calls[1]["config"]["response_mime_type"] == "application/json"


@pytest.mark.parametrize("value", [False, True, 0, -1])
def test_gemini_rejects_invalid_output_limit_before_calling_provider(value: int) -> None:
    models = FakeModels(responses=[_response(text="unused")])

    with pytest.raises(ValueError, match="max_output_tokens"):
        _client(models).complete(MESSAGES, max_output_tokens=value)

    assert models.generate_calls == []


def test_gemini_environment_uses_documented_key_precedence_and_optional_base_url(
    tmp_path: Path,
) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "gemini",
            "CHULK_MODEL": "gemini-test",
            "CHULK_GEMINI_API_KEY": " canonical-key ",
            "GEMINI_API_KEY": "gemini-key",
            "GOOGLE_API_KEY": "google-key",
            "CHULK_GEMINI_BASE_URL": " https://gemini.example ",
        }
    )

    assert config.gemini_api_key == "canonical-key"
    assert config.gemini_base_url == "https://gemini.example"
    assert provider_connection_from_config("gemini", config) == LLMProviderConnection(
        api_key="canonical-key",
        base_url="https://gemini.example",
    )


@pytest.mark.parametrize(
    ("environment", "expected_key"),
    [
        (
            {
                "CHULK_GEMINI_API_KEY": "   ",
                "GEMINI_API_KEY": "gemini-key",
                "GOOGLE_API_KEY": "google-key",
            },
            "gemini-key",
        ),
        ({"GOOGLE_API_KEY": "google-key"}, "google-key"),
    ],
)
def test_gemini_environment_supports_standard_key_aliases_in_order(
    tmp_path: Path,
    environment: dict[str, str],
    expected_key: str,
) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "gemini",
            "CHULK_MODEL": "gemini-test",
            **environment,
        }
    )

    assert config.gemini_api_key == expected_key


def test_gemini_environment_requires_explicit_model(tmp_path: Path) -> None:
    with pytest.raises(ConfigValueError, match="CHULK_MODEL"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_PROVIDER": "gemini",
                "CHULK_GEMINI_API_KEY": "test-key",
            }
        )


def test_gemini_fallback_requires_explicit_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit model"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_FALLBACK_PROVIDERS": "gemini",
            }
        )


def test_gemini_factory_builds_native_client_with_bound_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_client = object()
    captured: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return sdk_client

    fake_genai = SimpleNamespace(Client=fake_constructor)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))

    client = create_llm_client(
        provider="gemini",
        model="gemini-test",
        connection=LLMProviderConnection(
            api_key="test-key",
            base_url="https://gemini.example",
        ),
        timeout_seconds=12.5,
        max_retries=3,
    )

    assert isinstance(client, GeminiGenerateContentClient)
    assert client.model_capabilities == resolve_model_capabilities("gemini", "gemini-test")
    assert client.model_capabilities.context_window_tokens == GEMINI_DEFAULT_CONTEXT_WINDOW_TOKENS
    assert (
        client.model_capabilities.default_response_reserve_tokens
        == GEMINI_DEFAULT_RESPONSE_RESERVE_TOKENS
    )
    assert captured["api_key"] == "test-key"
    assert captured["http_options"] == {
        "base_url": "https://gemini.example",
        "timeout": 12_500,
        "retry_options": {"attempts": 4},
    }


def test_gemini_agent_config_cli_and_safe_config_output(tmp_path: Path) -> None:
    secret = "gemini-secret-that-must-not-be-rendered"
    config = AgentConfig.gemini(
        project_root=tmp_path,
        model="gemini-test",
        api_key=secret,
        base_url="https://gemini.example",
    ).to_config()
    chain = create_cli_llm(config)
    output = format_config(config)

    assert config.llm_provider == "gemini"
    assert config.model == "gemini-test"
    assert config.gemini_api_key == secret
    assert config.gemini_base_url == "https://gemini.example"
    primary_provider = chain.providers[0]
    assert isinstance(primary_provider, GeminiProvider)
    assert primary_provider.model == "gemini-test"
    assert "gemini_api_key: set" in output
    assert "gemini_base_url: https://gemini.example" in output
    assert secret not in output


def test_gemini_agent_config_requires_a_nonblank_model() -> None:
    with pytest.raises(ValueError, match="non-empty model"):
        AgentConfig.gemini(model="   ")


def test_gemini_is_exported_and_registered_with_native_capabilities() -> None:
    assert ExportedGeminiGenerateContentClient is GeminiGenerateContentClient
    assert ExportedGeminiProvider is GeminiProvider
    assert "gemini" in supported_llm_providers()
    assert provider_capabilities("gemini") is GEMINI_CAPABILITIES
    assert GEMINI_CAPABILITIES.api_style == "generate_content"
    assert GEMINI_CAPABILITIES.supports_structured_output is True
    assert GEMINI_CAPABILITIES.supports_native_tool_calling is True
    assert GEMINI_CAPABILITIES.supports_hosted_mcp_tools is False
