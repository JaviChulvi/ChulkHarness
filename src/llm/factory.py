"""LLM provider registry and client factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.llm.base import LLMClient, LLMConfigurationError
from src.llm.capabilities import LLMCapabilities
from src.llm.providers.deepseek import DEEPSEEK_CAPABILITIES, DeepSeekChatCompletionsClient
from src.llm.providers.openai import OPENAI_CAPABILITIES, OpenAIResponsesClient


@dataclass(frozen=True)
class LLMClientSettings:
    """Provider-neutral settings used to construct an LLM client."""

    model: str
    openai_api_key: str | None
    deepseek_api_key: str | None
    deepseek_base_url: str
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class LLMProvider:
    """Factory metadata for one configured LLM provider."""

    name: str
    capabilities: LLMCapabilities
    create_client: Callable[[LLMClientSettings], LLMClient]


LLM_PROVIDER_REGISTRY: dict[str, LLMProvider] = {
    "openai": LLMProvider(
        name="openai",
        capabilities=OPENAI_CAPABILITIES,
        create_client=lambda settings: _create_openai_client(settings),
    ),
    "deepseek": LLMProvider(
        name="deepseek",
        capabilities=DEEPSEEK_CAPABILITIES,
        create_client=lambda settings: _create_deepseek_client(settings),
    ),
}


def supported_llm_providers() -> set[str]:
    """Return configured provider names."""
    return set(LLM_PROVIDER_REGISTRY)


def create_llm_client(
    *,
    provider: str,
    model: str,
    openai_api_key: str | None,
    deepseek_api_key: str | None,
    deepseek_base_url: str,
    timeout_seconds: float,
    max_retries: int,
) -> LLMClient:
    """Create an LLM client for the selected provider."""
    normalized_provider = provider.lower()
    provider_spec = LLM_PROVIDER_REGISTRY.get(normalized_provider)
    if provider_spec is None:
        raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")

    return provider_spec.create_client(
        LLMClientSettings(
            model=model,
            openai_api_key=openai_api_key,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=deepseek_base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    )


def _create_openai_client(settings: LLMClientSettings) -> LLMClient:
    return OpenAIResponsesClient(
        model=settings.model,
        api_key=settings.openai_api_key,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def _create_deepseek_client(settings: LLMClientSettings) -> LLMClient:
    return DeepSeekChatCompletionsClient(
        model=settings.model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
