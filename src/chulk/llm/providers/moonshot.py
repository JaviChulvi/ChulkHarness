"""Moonshot AI provider client."""

from __future__ import annotations

from typing import Any

from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import chat_messages
from chulk.llm.providers.chat_completions import (
    ChatCompletionsTransportProfile,
    OpenAICompatibleChatCompletionsClient,
)
from chulk.llm.usage import normalize_chat_completions_usage


DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"

MOONSHOT_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=True,
    supports_streaming=True,
    supports_native_tool_calling=True,
    api_style="chat_completions",
)

MOONSHOT_TRANSPORT_PROFILE = ChatCompletionsTransportProfile(
    provider="moonshot",
    display_name="Moonshot AI",
    capabilities=MOONSHOT_CAPABILITIES,
    normalize_messages=chat_messages,
    normalize_usage=normalize_chat_completions_usage,
    json_response_format={"type": "json_object"},
    missing_api_key_message=(
        "MOONSHOT_API_KEY or CHULK_MOONSHOT_API_KEY is required for Moonshot AI"
    ),
    max_output_tokens_field="max_completion_tokens",
    models_supporting_required_tool_choice=frozenset({"kimi-k3"}),
)


class MoonshotChatCompletionsClient(OpenAICompatibleChatCompletionsClient):
    """LLM client backed by Moonshot AI's Chat Completions API."""

    capabilities = MOONSHOT_CAPABILITIES
    provider = "moonshot"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_MOONSHOT_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
        async_client: Any | None = None,
        owns_client: bool | None = None,
        owns_async_client: bool | None = None,
    ) -> None:
        super().__init__(
            profile=MOONSHOT_TRANSPORT_PROFILE,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
            async_client=async_client,
            owns_client=owns_client,
            owns_async_client=owns_async_client,
        )
