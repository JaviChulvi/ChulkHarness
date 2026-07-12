"""AWS Bedrock client for its OpenAI-compatible Chat Completions API."""

from __future__ import annotations

from typing import Any

from chulk.llm.base import LLMConfigurationError
from chulk.llm.capabilities import LLMCapabilities
from chulk.llm.messages import chat_messages
from chulk.llm.providers.chat_completions import (
    ChatCompletionsTransportProfile,
    OpenAICompatibleChatCompletionsClient,
)
from chulk.llm.usage import normalize_chat_completions_usage


BEDROCK_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=False,
    supports_streaming=True,
    supports_native_tool_calling=True,
    api_style="chat_completions",
)

BEDROCK_TRANSPORT_PROFILE = ChatCompletionsTransportProfile(
    provider="bedrock",
    display_name="AWS Bedrock",
    capabilities=BEDROCK_CAPABILITIES,
    normalize_messages=chat_messages,
    normalize_usage=normalize_chat_completions_usage,
    missing_api_key_message=(
        "BEDROCK_API_KEY or AWS_BEARER_TOKEN_BEDROCK is required for AWS Bedrock"
    ),
)


class BedrockOpenAICompatibleClient(OpenAICompatibleChatCompletionsClient):
    """Client for a caller-selected Bedrock OpenAI-compatible endpoint."""

    capabilities = BEDROCK_CAPABILITIES
    provider = "bedrock"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        normalized_model = _required_value(model, "model")
        normalized_api_key = _required_value(
            api_key,
            "api_key",
            model=normalized_model,
        )
        normalized_base_url = _required_value(
            base_url,
            "base_url",
            model=normalized_model,
        )
        super().__init__(
            profile=BEDROCK_TRANSPORT_PROFILE,
            model=normalized_model,
            api_key=normalized_api_key,
            base_url=normalized_base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
            async_client=async_client,
        )


def _required_value(
    value: str | None,
    field: str,
    *,
    model: str | None = None,
) -> str:
    if value is None or not value.strip():
        raise LLMConfigurationError(
            f"{field} is required for provider bedrock",
            provider="bedrock",
            model=model,
        )
    return value.strip()
