"""Local OpenAI-compatible provider client."""

from __future__ import annotations

from typing import Any

from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import local_chat_messages
from chulk.llm.providers.chat_completions import (
    ChatCompletionsTransportProfile,
    OpenAICompatibleChatCompletionsClient,
)
from chulk.llm.usage import normalize_chat_completions_usage


DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LOCAL_API_KEY = "local"

LOCAL_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=False,
    supports_streaming=True,
    supports_native_tool_calling=True,
    api_style="chat_completions",
)

LOCAL_TRANSPORT_PROFILE = ChatCompletionsTransportProfile(
    provider="local",
    display_name="Local LLM",
    capabilities=LOCAL_CAPABILITIES,
    normalize_messages=local_chat_messages,
    normalize_usage=normalize_chat_completions_usage,
    default_api_key=DEFAULT_LOCAL_API_KEY,
)


class LocalOpenAICompatibleClient(OpenAICompatibleChatCompletionsClient):
    """LLM client backed by a local OpenAI-compatible Chat Completions server."""

    capabilities = LOCAL_CAPABILITIES
    provider = "local"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_LOCAL_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        super().__init__(
            profile=LOCAL_TRANSPORT_PROFILE,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
            async_client=async_client,
        )
