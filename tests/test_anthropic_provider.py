"""Tests for the native Anthropic Messages provider."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from chulk._sdk.config import AgentConfig
from chulk.config import ConfigValueError, load_config
from chulk.core.actions import FinalAnswerAction, ToolCallAction
from chulk.llm import AnthropicMessagesClient as ExportedAnthropicMessagesClient
from chulk.llm import AnthropicProvider as ExportedAnthropicProvider
from chulk.llm import PlanningToolAvailability
from chulk.llm.base import LLMConfigurationError, LLMError
from chulk.llm.capabilities import resolve_model_capabilities
from chulk.llm.factory import (
    LLMProviderConnection,
    create_llm_client,
    provider_connection_from_config,
    supported_llm_providers,
)
from chulk.llm.providers.anthropic import (
    ANTHROPIC_CAPABILITIES,
    DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS,
    AnthropicMessagesClient,
    normalize_anthropic_usage,
)
from chulk.llm.public import AnthropicProvider
from chulk.llm.tools import PLAN_TOOL_NAME
from chulk.main import create_cli_llm, format_config


class FakeMessages:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeAnthropicClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.messages = FakeMessages(outcomes)


class AuthenticationError(RuntimeError):
    status_code = 401


def _response(*blocks: object, usage: object | None = None) -> object:
    return SimpleNamespace(
        content=list(blocks),
        usage=usage or SimpleNamespace(input_tokens=4, output_tokens=2),
        stop_reason="end_turn",
    )


def _text(value: str) -> object:
    return SimpleNamespace(type="text", text=value)


def _tool_use(call_id: str, *, expression: object = "2 + 2") -> object:
    return SimpleNamespace(
        type="tool_use",
        id=call_id,
        name="calculator",
        input={"expression": expression},
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


def _final_answer_json(content: str) -> str:
    return json.dumps({"type": "final_answer", "content": content})


def test_anthropic_text_request_splits_system_and_honors_output_limit() -> None:
    usage = SimpleNamespace(input_tokens=7, output_tokens=3)
    fake = FakeAnthropicClient([_response(_text("hello"), usage=usage), _response(_text("again"))])
    client = AnthropicMessagesClient(model="claude-test", client=fake)
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "developer", "content": "Use plain language."},
        {"role": "user", "content": "Hi"},
    ]

    response = client.complete_response(messages)
    second = client.complete(messages, max_output_tokens=123)

    assert response.content == "hello"
    assert response.provider == "anthropic"
    assert response.model == "claude-test"
    assert response.usage is not None
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3
    assert second == "again"
    assert fake.messages.calls[0] == {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS,
        "system": "Be concise.\n\nUse plain language.",
    }
    assert fake.messages.calls[1]["max_tokens"] == 123


def test_anthropic_usage_normalizes_cache_tokens() -> None:
    usage = normalize_anthropic_usage(
        {
            "input_tokens": 5,
            "output_tokens": 7,
            "cache_creation_input_tokens": 11,
            "cache_read_input_tokens": 13,
        }
    )

    assert usage is not None
    assert usage.input_tokens == 29
    assert usage.output_tokens == 7
    assert usage.total_tokens == 36
    assert usage.cached_input_tokens == 13
    assert usage.cache_hit_input_tokens == 13
    assert usage.cache_miss_input_tokens == 16
    assert usage.estimated is False


def test_anthropic_uses_one_native_tool_call_and_disables_parallel_use() -> None:
    fake = FakeAnthropicClient([_response(_tool_use("toolu_1"))])
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    result = client.complete_action(
        [{"role": "user", "content": "what is 2+2?"}],
        tools=[_calculator_tool()],
        max_output_tokens=321,
    )

    assert result.action == ToolCallAction(
        type="tool_call",
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )
    request = fake.messages.calls[0]
    assert request["max_tokens"] == 321
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    request_tools = request["tools"]
    assert isinstance(request_tools, list)
    tool_names = {str(tool["name"]) for tool in request_tools if isinstance(tool, dict)}
    assert tool_names == {"calculator"}
    assert result.metadata["action_transport"] == "provider_native"
    assert result.metadata["provider_tool_call"]["id"] == "toolu_1"


def test_anthropic_native_text_becomes_final_answer() -> None:
    fake = FakeAnthropicClient([_response(_text("native final"))])
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    result = client.complete_action(
        [{"role": "user", "content": "hello"}],
        tools=[_calculator_tool()],
    )

    assert isinstance(result.action, FinalAnswerAction)
    assert result.action.content == "native final"
    assert result.metadata["action_transport"] == "provider_native"
    assert result.metadata["provider_tool_call"] is None


def test_anthropic_native_text_with_no_effective_tools_omits_tool_fields() -> None:
    fake = FakeAnthropicClient([_response(_text("native final"))])
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    result = client.complete_action(
        [{"role": "user", "content": "hello"}],
        tools=[],
        planning_tools=PlanningToolAvailability(),
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="native final")
    assert "tools" not in fake.messages.calls[0]
    assert "tool_choice" not in fake.messages.calls[0]
    assert result.metadata["action_transport"] == "provider_native"


def test_anthropic_planning_policy_enables_native_transport_without_regular_tools() -> None:
    fake = FakeAnthropicClient([_response(_text("native final"))])
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    result = client.complete_action(
        [{"role": "user", "content": "plan this"}],
        planning_tools=PlanningToolAvailability(propose_plan=True),
    )

    assert result.action == FinalAnswerAction(type="final_answer", content="native final")
    assert [item["name"] for item in fake.messages.calls[0]["tools"]] == [
        PLAN_TOOL_NAME
    ]
    assert fake.messages.calls[0]["tool_choice"] == {
        "type": "any",
        "disable_parallel_tool_use": True,
    }
    assert result.metadata["action_transport"] == "provider_native"


def test_anthropic_rejects_multiple_tool_calls_and_uses_json_fallback() -> None:
    fake = FakeAnthropicClient(
        [
            _response(_tool_use("toolu_1"), _tool_use("toolu_2", expression="3 + 3")),
            _response(_text(_final_answer_json("safe fallback"))),
        ]
    )
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    result = client.complete_action(
        [{"role": "user", "content": "calculate"}],
        tools=[_calculator_tool()],
    )

    assert isinstance(result.action, FinalAnswerAction)
    assert result.action.content == "safe fallback"
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert "multiple tool calls" in result.metadata["native_tool_call_error"]
    assert len(fake.messages.calls) == 2
    assert "tools" not in fake.messages.calls[1]
    assert "exactly one JSON object" in str(fake.messages.calls[1]["system"])


def test_anthropic_falls_back_only_for_unsupported_native_transport() -> None:
    fake = FakeAnthropicClient(
        [
            RuntimeError("native tools unsupported"),
            _response(_text(_final_answer_json("fallback"))),
        ]
    )
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    result = client.complete_action(
        [{"role": "user", "content": "hello"}],
        tools=[_calculator_tool()],
    )

    assert isinstance(result.action, FinalAnswerAction)
    assert result.action.content == "fallback"
    assert result.metadata["action_transport"] == "chulk_json_fallback"
    assert len(fake.messages.calls) == 2


def test_anthropic_does_not_fallback_for_authentication_failures() -> None:
    fake = FakeAnthropicClient([AuthenticationError("bad key")])
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    with pytest.raises(LLMError) as error:
        client.complete_action(
            [{"role": "user", "content": "hello"}],
            tools=[_calculator_tool()],
        )

    assert error.value.code == "authentication_error"
    assert error.value.provider == "anthropic"
    assert error.value.model == "claude-test"
    assert len(fake.messages.calls) == 1


def test_anthropic_streams_text_and_normalizes_final_usage() -> None:
    events = iter(
        [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=4,
                        output_tokens=0,
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                    )
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="hel"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="lo"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=2),
            ),
            SimpleNamespace(type="message_stop"),
        ]
    )
    fake = FakeAnthropicClient([events])
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    chunks = list(
        client.stream_complete(
            [{"role": "user", "content": "hello"}],
            max_output_tokens=42,
        )
    )

    assert [chunk.text for chunk in chunks if chunk.type == "text_delta"] == ["hel", "lo"]
    assert chunks[-1].type == "completed"
    assert chunks[-1].metadata["stop_reason"] == "end_turn"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.input_tokens == 4
    assert chunks[-1].usage.output_tokens == 2
    assert fake.messages.calls[0]["stream"] is True
    assert fake.messages.calls[0]["max_tokens"] == 42


def test_anthropic_json_action_path_uses_default_max_tokens() -> None:
    fake = FakeAnthropicClient([_response(_text(_final_answer_json("structured")))])
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    result = client.complete_action([{"role": "user", "content": "hello"}])

    assert isinstance(result.action, FinalAnswerAction)
    assert result.action.content == "structured"
    assert result.metadata["action_transport"] == "chulk_json"
    assert fake.messages.calls[0]["max_tokens"] == DEFAULT_ANTHROPIC_MAX_OUTPUT_TOKENS


def test_anthropic_rejects_hosted_mcp_without_making_a_request() -> None:
    fake = FakeAnthropicClient([])
    client = AnthropicMessagesClient(model="claude-test", client=fake)

    with pytest.raises(LLMError) as error:
        client.complete_action(
            [{"role": "user", "content": "hello"}],
            tools=[_calculator_tool()],
            hosted_mcp_servers=[{"type": "mcp"}],
        )

    assert error.value.code == "unsupported_feature"
    assert fake.messages.calls == []


def test_anthropic_requires_key_without_injected_client() -> None:
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicMessagesClient(model="claude-test")


def test_anthropic_requires_nonempty_model() -> None:
    with pytest.raises(LLMConfigurationError, match="CHULK_MODEL"):
        AnthropicMessagesClient(model="   ", client=object())


def test_anthropic_strips_api_key_before_sdk_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=fake_constructor))

    AnthropicMessagesClient(model=" claude-test ", api_key="  test-key  ")

    assert captured["api_key"] == "test-key"


def test_anthropic_dependency_is_loaded_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fail_anthropic_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "anthropic":
            raise ImportError("not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_anthropic_import)

    with pytest.raises(LLMConfigurationError, match="anthropic package is required"):
        AnthropicMessagesClient(model="claude-test", api_key="test-key")


@pytest.mark.parametrize("value", [False, True, 0, -1])
def test_anthropic_rejects_invalid_output_limits(value: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        AnthropicMessagesClient(model="claude-test", client=object(), max_output_tokens=value)


def test_anthropic_environment_uses_canonical_key_alias_and_optional_base_url(tmp_path: Path) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "anthropic",
            "CHULK_MODEL": "claude-test",
            "CHULK_ANTHROPIC_API_KEY": "canonical-key",
            "ANTHROPIC_API_KEY": "fallback-key",
            "CHULK_ANTHROPIC_BASE_URL": "https://anthropic.example",
        }
    )

    assert config.anthropic_api_key == "canonical-key"
    assert config.anthropic_base_url == "https://anthropic.example"
    assert provider_connection_from_config("anthropic", config) == LLMProviderConnection(
        api_key="canonical-key",
        base_url="https://anthropic.example",
    )


def test_anthropic_environment_falls_back_to_standard_key_alias(tmp_path: Path) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "anthropic",
            "CHULK_MODEL": "claude-test",
            "ANTHROPIC_API_KEY": "standard-key",
        }
    )

    assert config.anthropic_api_key == "standard-key"
    assert config.anthropic_base_url is None


def test_anthropic_environment_requires_explicit_model(tmp_path: Path) -> None:
    with pytest.raises(ConfigValueError, match="CHULK_MODEL"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_PROVIDER": "anthropic",
                "CHULK_ANTHROPIC_API_KEY": "test-key",
            }
        )


def test_anthropic_factory_builds_native_client_with_bound_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_client = object()
    captured: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return sdk_client

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=fake_constructor))
    client = create_llm_client(
        provider="anthropic",
        model="claude-test",
        connection=LLMProviderConnection(
            api_key="test-key",
            base_url="https://anthropic.example",
        ),
        timeout_seconds=12.5,
        max_retries=3,
    )

    assert isinstance(client, AnthropicMessagesClient)
    assert client.model_capabilities is not None
    assert client.model_capabilities == resolve_model_capabilities("anthropic", "claude-test")
    assert client.model_capabilities.context_window_tokens == 200_000
    assert captured == {
        "api_key": "test-key",
        "base_url": "https://anthropic.example",
        "timeout": 12.5,
        "max_retries": 3,
    }


def test_anthropic_sdk_builder_cli_spec_and_safe_config_output(tmp_path: Path) -> None:
    config = AgentConfig.anthropic(
        project_root=tmp_path,
        model="claude-test",
        api_key="secret-anthropic-key",
        base_url="https://anthropic.example",
    ).to_config()
    chain = create_cli_llm(config)
    output = format_config(config)

    assert config.llm_provider == "anthropic"
    assert config.model == "claude-test"
    assert config.anthropic_api_key == "secret-anthropic-key"
    assert config.anthropic_base_url == "https://anthropic.example"
    primary_provider = chain.providers[0]
    assert isinstance(primary_provider, AnthropicProvider)
    assert primary_provider.model == "claude-test"
    assert "anthropic_api_key: set" in output
    assert "anthropic_base_url: https://anthropic.example" in output
    assert "secret-anthropic-key" not in output


def test_anthropic_cli_fallback_spec_requires_and_preserves_explicit_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit model"):
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_FALLBACK_PROVIDERS": "anthropic",
            }
        )

    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_FALLBACK_PROVIDERS": "anthropic:claude-fallback",
        }
    )
    chain = create_cli_llm(config)

    fallback_provider = chain.providers[1]
    assert isinstance(fallback_provider, AnthropicProvider)
    assert fallback_provider.model == "claude-fallback"


def test_anthropic_is_exported_and_registered_with_native_capabilities() -> None:
    assert ExportedAnthropicMessagesClient is AnthropicMessagesClient
    assert ExportedAnthropicProvider is AnthropicProvider
    assert "anthropic" in supported_llm_providers()
    assert ANTHROPIC_CAPABILITIES.api_style == "messages"
    assert ANTHROPIC_CAPABILITIES.supports_native_tool_calling is True
    assert ANTHROPIC_CAPABILITIES.supports_hosted_mcp_tools is False
