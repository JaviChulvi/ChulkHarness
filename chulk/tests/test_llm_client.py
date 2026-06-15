"""Tests for LLM client behavior."""

import json
from types import SimpleNamespace

from chulk.llm import (
    DeepSeekChatCompletionsClient,
    LLM_PROVIDER_REGISTRY,
    LLMClient,
    LLMConfigurationError,
    LocalOpenAICompatibleClient,
    OpenAIResponsesClient,
    create_llm_client,
    resolve_model_capabilities,
)


class FakeResponsesResource:
    def __init__(self, output_text: str = "model answer") -> None:
        self.output_text = output_text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeOpenAIClient:
    def __init__(self, output_text: str = "model answer") -> None:
        self.responses = FakeResponsesResource(output_text)


class FakeChatCompletionsResource:
    def __init__(self, content: str = "deepseek answer") -> None:
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class FakeDeepSeekClient:
    def __init__(self, content: str = "deepseek answer") -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletionsResource(content))


class FakeLocalClient:
    def __init__(self, content: str = "local answer") -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletionsResource(content))


class ScriptedLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        return self.responses.pop(0)


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
    assert "arguments_json" in schema["properties"]
    assert "plan_json" in schema["properties"]
    assert "arguments" not in schema["properties"]


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


def test_deepseek_client_does_not_send_max_tokens():
    fake_client = FakeDeepSeekClient(
        json.dumps({"type": "final_answer", "content": "structured answer", "tool_name": None, "arguments_json": "{}"})
    )
    client = DeepSeekChatCompletionsClient(model="deepseek-v4-flash", client=fake_client)

    client.complete([{"role": "user", "content": "Hello"}], max_output_tokens=25)

    assert "max_tokens" not in fake_client.chat.completions.kwargs

    client.complete_action(
        [
            {"role": "system", "content": "Return action JSON."},
            {"role": "user", "content": "Hello"},
        ],
        max_output_tokens=250,
    )

    assert "max_tokens" not in fake_client.chat.completions.kwargs


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
    assert "max_tokens" not in fake_client.chat.completions.kwargs
    assert "response_format" not in fake_client.chat.completions.kwargs
    assert fake_client.chat.completions.kwargs["messages"] == [
        {"role": "user", "content": "Instructions:\nReturn action JSON.\n\nUser message:\n\nHello"}
    ]


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


def test_llm_provider_registry_exposes_provider_capabilities():
    openai_provider = LLM_PROVIDER_REGISTRY["openai"]
    deepseek_provider = LLM_PROVIDER_REGISTRY["deepseek"]
    local_provider = LLM_PROVIDER_REGISTRY["local"]

    assert openai_provider.capabilities.supports_structured_output is True
    assert openai_provider.capabilities.api_style == "responses"
    assert deepseek_provider.capabilities.supports_json_mode is True
    assert deepseek_provider.capabilities.api_style == "chat_completions"
    assert local_provider.capabilities.api_style == "chat_completions"
    assert local_provider.capabilities.supports_structured_output is False


def test_resolve_model_capabilities_returns_context_window_and_reserve():
    openai_caps = resolve_model_capabilities("openai", "gpt-4.1-mini")
    deepseek_caps = resolve_model_capabilities("deepseek", "deepseek-v4-flash")
    local_caps = resolve_model_capabilities("local", "google/gemma-4-12b-qat")
    local_qwen_caps = resolve_model_capabilities("local", "qwen/qwen3.5-35b-a3b")

    assert openai_caps.context_window_tokens == 1_047_576
    assert openai_caps.default_response_reserve_tokens == 8_192
    assert openai_caps.input_budget_tokens == 1_039_384
    assert deepseek_caps.context_window_tokens == 1_000_000
    assert deepseek_caps.default_response_reserve_tokens == 16_384
    assert deepseek_caps.input_budget_tokens == 983_616
    assert local_caps.context_window_tokens == 131_072
    assert local_caps.default_response_reserve_tokens == 4_096
    assert local_qwen_caps.context_window_tokens == 262_144
    assert local_qwen_caps.default_response_reserve_tokens == 4_096


def test_resolve_model_capabilities_supports_known_family_aliases():
    caps = resolve_model_capabilities("openai", "gpt-4.1-mini-2099-01-01")

    assert caps.model == "gpt-4.1-mini-2099-01-01"
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
        assert "chulk/llm/capabilities.py" in str(exc)
    else:
        raise AssertionError("Expected unknown model capability lookup to fail")
