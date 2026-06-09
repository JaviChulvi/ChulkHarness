"""Tests for LLM client behavior."""

from types import SimpleNamespace

from src.llm import (
    DeepSeekChatCompletionsClient,
    LLMConfigurationError,
    OpenAIResponsesClient,
    create_llm_client,
)


class FakeResponsesResource:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text="model answer")


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponsesResource()


class FakeChatCompletionsResource:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="deepseek answer"),
                )
            ]
        )


class FakeDeepSeekClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletionsResource())


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
