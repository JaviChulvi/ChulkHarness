from __future__ import annotations

from collections.abc import Callable
import sys
from types import ModuleType

import pytest

from chulk.llm import FallbackChain, LLMClient
from chulk.llm.providers import (
    AnthropicMessagesClient,
    DeepSeekChatCompletionsClient,
    GeminiGenerateContentClient,
    OpenAIResponsesClient,
)


class SyncTransport:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class AsyncTransport:
    def __init__(self) -> None:
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


class FailingTransport:
    def close(self) -> None:
        raise RuntimeError("close failed")


ProviderFactory = Callable[..., LLMClient]


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kwargs: OpenAIResponsesClient(model="test", **kwargs),
        lambda **kwargs: AnthropicMessagesClient(model="test", **kwargs),
        lambda **kwargs: GeminiGenerateContentClient(model="test", **kwargs),
        lambda **kwargs: DeepSeekChatCompletionsClient(model="test", **kwargs),
    ],
    ids=["openai-responses", "anthropic", "gemini", "chat-completions"],
)
def test_injected_provider_transports_are_caller_owned_by_default(
    factory: ProviderFactory,
) -> None:
    sync_transport = SyncTransport()
    async_transport = AsyncTransport()
    client = factory(
        client=sync_transport,
        async_client=async_transport,
    )

    client.close()
    client.close()

    assert sync_transport.close_count == 0
    assert async_transport.close_count == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kwargs: OpenAIResponsesClient(model="test", **kwargs),
        lambda **kwargs: AnthropicMessagesClient(model="test", **kwargs),
        lambda **kwargs: GeminiGenerateContentClient(model="test", **kwargs),
        lambda **kwargs: DeepSeekChatCompletionsClient(model="test", **kwargs),
    ],
    ids=["openai-responses", "anthropic", "gemini", "chat-completions"],
)
def test_explicitly_owned_provider_transports_close_once_from_sync_host(
    factory: ProviderFactory,
) -> None:
    sync_transport = SyncTransport()
    async_transport = AsyncTransport()
    client = factory(
        client=sync_transport,
        async_client=async_transport,
        owns_client=True,
        owns_async_client=True,
    )

    client.close()
    client.close()

    assert sync_transport.close_count == 1
    assert async_transport.close_count == 1


@pytest.mark.asyncio
async def test_owned_provider_transports_close_once_from_async_host() -> None:
    sync_transport = SyncTransport()
    async_transport = AsyncTransport()
    client = OpenAIResponsesClient(
        model="test",
        client=sync_transport,
        async_client=async_transport,
        owns_client=True,
        owns_async_client=True,
    )

    await client.aclose()
    await client.aclose()

    assert sync_transport.close_count == 1
    assert async_transport.close_count == 1


def test_provider_cleanup_continues_after_one_transport_fails() -> None:
    remaining = SyncTransport()
    client = OpenAIResponsesClient(
        model="test",
        client=FailingTransport(),
        async_client=remaining,
        owns_client=True,
        owns_async_client=True,
    )

    with pytest.raises(ExceptionGroup, match="provider resources"):
        client.close()

    assert remaining.close_count == 1


def test_openai_constructor_failure_closes_created_sync_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_transport = SyncTransport()
    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = lambda **_kwargs: sync_transport  # type: ignore[attr-defined]

    class BrokenAsyncOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("async construction failed")

    fake_openai.AsyncOpenAI = BrokenAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    with pytest.raises(RuntimeError, match="async construction failed"):
        OpenAIResponsesClient(model="test", api_key="secret")

    assert sync_transport.close_count == 1


class CloseableClient(LLMClient):
    def __init__(self) -> None:
        self.close_count = 0
        self.aclose_count = 0

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return "done"

    def close(self) -> None:
        self.close_count += 1

    async def aclose(self) -> None:
        self.aclose_count += 1


def test_fallback_chain_closes_each_provider_once() -> None:
    first = CloseableClient()
    second = CloseableClient()
    chain = FallbackChain([first, second])

    chain.close()
    chain.close()

    assert first.close_count == 1
    assert second.close_count == 1


@pytest.mark.asyncio
async def test_fallback_chain_async_close_uses_provider_async_lifecycle() -> None:
    first = CloseableClient()
    second = CloseableClient()
    chain = FallbackChain([first, second])

    await chain.aclose()
    await chain.aclose()

    assert first.aclose_count == 1
    assert second.aclose_count == 1
