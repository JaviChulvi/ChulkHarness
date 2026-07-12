"""DeepSeek provider client."""

from __future__ import annotations

from typing import Any

from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import chat_messages
from chulk.llm.providers.chat_completions import (
    ChatCompletionsTransportProfile,
    OpenAICompatibleChatCompletionsClient,
)
from chulk.llm.usage import normalize_deepseek_usage


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEEPSEEK_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=True,
    supports_streaming=True,
    supports_native_tool_calling=True,
    api_style="chat_completions",
)

DEEPSEEK_TRANSPORT_PROFILE = ChatCompletionsTransportProfile(
    provider="deepseek",
    display_name="DeepSeek",
    capabilities=DEEPSEEK_CAPABILITIES,
    normalize_messages=chat_messages,
    normalize_usage=normalize_deepseek_usage,
    json_response_format={"type": "json_object"},
    missing_api_key_message="DEEPSEEK_API_KEY or CHULK_DEEPSEEK_API_KEY is required for DeepSeek",
)


class DeepSeekChatCompletionsClient(OpenAICompatibleChatCompletionsClient):
    """LLM client backed by DeepSeek's OpenAI-compatible Chat Completions API."""

    capabilities = DEEPSEEK_CAPABILITIES
    provider = "deepseek"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            profile=DEEPSEEK_TRANSPORT_PROFILE,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
        )
