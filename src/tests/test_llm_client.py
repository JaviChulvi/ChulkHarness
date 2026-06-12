"""Tests for LLM client behavior."""

import json
from types import SimpleNamespace

from src.llm import (
    DeepSeekChatCompletionsClient,
    LLM_PROVIDER_REGISTRY,
    LLMClient,
    LLMConfigurationError,
    OpenAIResponsesClient,
    create_llm_client,
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
    assert "arguments_json" in schema["properties"]
    assert "arguments" not in schema["properties"]


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


def test_llm_provider_registry_exposes_provider_capabilities():
    openai_provider = LLM_PROVIDER_REGISTRY["openai"]
    deepseek_provider = LLM_PROVIDER_REGISTRY["deepseek"]

    assert openai_provider.capabilities.supports_structured_output is True
    assert openai_provider.capabilities.api_style == "responses"
    assert deepseek_provider.capabilities.supports_json_mode is True
    assert deepseek_provider.capabilities.api_style == "chat_completions"
