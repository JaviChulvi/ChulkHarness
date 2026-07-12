"""Hosted OpenAI-compatible and OpenRouter provider clients."""

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


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

HOSTED_OPENAI_COMPATIBLE_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=False,
    supports_streaming=True,
    supports_native_tool_calling=True,
    api_style="chat_completions",
)

OPENROUTER_CAPABILITIES = LLMCapabilities(
    supports_structured_output=False,
    supports_json_mode=False,
    supports_streaming=True,
    supports_native_tool_calling=True,
    api_style="chat_completions",
)

HOSTED_OPENAI_COMPATIBLE_TRANSPORT_PROFILE = ChatCompletionsTransportProfile(
    provider="openai-compatible",
    display_name="Hosted OpenAI-compatible provider",
    capabilities=HOSTED_OPENAI_COMPATIBLE_CAPABILITIES,
    normalize_messages=chat_messages,
    normalize_usage=normalize_chat_completions_usage,
    missing_api_key_message=(
        "CHULK_OPENAI_COMPATIBLE_API_KEY is required for the hosted OpenAI-compatible provider"
    ),
)

OPENROUTER_TRANSPORT_PROFILE = ChatCompletionsTransportProfile(
    provider="openrouter",
    display_name="OpenRouter",
    capabilities=OPENROUTER_CAPABILITIES,
    normalize_messages=chat_messages,
    normalize_usage=normalize_chat_completions_usage,
    missing_api_key_message="OPENROUTER_API_KEY or CHULK_OPENROUTER_API_KEY is required for OpenRouter",
)


class HostedOpenAICompatibleClient(OpenAICompatibleChatCompletionsClient):
    """Client for a user-selected hosted Chat Completions endpoint."""

    capabilities = HOSTED_OPENAI_COMPATIBLE_CAPABILITIES
    provider = "openai-compatible"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        normalized_model = _required_value(model, "model", provider=self.provider)
        normalized_api_key = _required_value(
            api_key,
            "api_key",
            provider=self.provider,
            model=normalized_model,
        )
        normalized_base_url = _required_value(
            base_url,
            "base_url",
            provider=self.provider,
            model=normalized_model,
        )
        super().__init__(
            profile=HOSTED_OPENAI_COMPATIBLE_TRANSPORT_PROFILE,
            model=normalized_model,
            api_key=normalized_api_key,
            base_url=normalized_base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
        )


class OpenRouterChatCompletionsClient(OpenAICompatibleChatCompletionsClient):
    """Client for OpenRouter's OpenAI-compatible Chat Completions API."""

    capabilities = OPENROUTER_CAPABILITIES
    provider = "openrouter"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        site_url: str | None = None,
        app_name: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        normalized_model = _required_value(model, "model", provider=self.provider)
        normalized_api_key = _required_value(
            api_key,
            "api_key",
            provider=self.provider,
            model=normalized_model,
        )
        normalized_base_url = _required_value(
            base_url,
            "base_url",
            provider=self.provider,
            model=normalized_model,
        )
        self.default_headers = openrouter_default_headers(
            site_url=site_url, app_name=app_name
        )
        if client is None and self.default_headers:
            client = _openai_sdk_client(
                api_key=normalized_api_key,
                base_url=normalized_base_url,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                default_headers=self.default_headers,
                model=normalized_model,
            )
        super().__init__(
            profile=OPENROUTER_TRANSPORT_PROFILE,
            model=normalized_model,
            api_key=normalized_api_key,
            base_url=normalized_base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=client,
        )


def openrouter_default_headers(
    *,
    site_url: str | None = None,
    app_name: str | None = None,
) -> dict[str, str]:
    """Return the optional OpenRouter attribution headers."""
    headers: dict[str, str] = {}
    if site_url is not None and site_url.strip():
        headers["HTTP-Referer"] = site_url.strip()
    if app_name is not None and app_name.strip():
        headers["X-OpenRouter-Title"] = app_name.strip()
    return headers


def _required_value(
    value: str | None,
    field: str,
    *,
    provider: str,
    model: str | None = None,
) -> str:
    if value is None or not value.strip():
        raise LLMConfigurationError(
            f"{field} is required for provider {provider}",
            provider=provider,
            model=model,
        )
    return value.strip()


def _openai_sdk_client(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    max_retries: int,
    default_headers: dict[str, str],
    model: str,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMConfigurationError(
            "The openai package is required. Install it with: pip install -e '.[openai]'",
            provider="openrouter",
            model=model,
        ) from exc
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
        default_headers=default_headers,
    )
