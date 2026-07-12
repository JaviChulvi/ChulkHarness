"""LLM provider profiles and client construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from chulk.llm.base import LLMClient, LLMConfigurationError
from chulk.llm.capabilities import LLMCapabilities, resolve_model_capabilities
from chulk.llm.providers.compatible import (
    DEFAULT_OPENROUTER_BASE_URL,
    HOSTED_OPENAI_COMPATIBLE_CAPABILITIES,
    OPENROUTER_CAPABILITIES,
    HostedOpenAICompatibleClient,
    OpenRouterChatCompletionsClient,
)
from chulk.llm.providers.deepseek import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEEPSEEK_CAPABILITIES,
    DeepSeekChatCompletionsClient,
)
from chulk.llm.providers.local import (
    DEFAULT_LOCAL_BASE_URL,
    LOCAL_CAPABILITIES,
    LocalOpenAICompatibleClient,
)
from chulk.llm.providers.openai import OPENAI_CAPABILITIES, OpenAIResponsesClient


class LLMConnectionConfig(Protocol):
    """Runtime configuration fields used to bind built-in providers."""

    @property
    def openai_api_key(self) -> str | None: ...

    @property
    def deepseek_api_key(self) -> str | None: ...

    @property
    def deepseek_base_url(self) -> str: ...

    @property
    def local_api_key(self) -> str | None: ...

    @property
    def local_base_url(self) -> str: ...

    @property
    def openai_compatible_api_key(self) -> str | None: ...

    @property
    def openai_compatible_base_url(self) -> str | None: ...

    @property
    def openrouter_api_key(self) -> str | None: ...

    @property
    def openrouter_base_url(self) -> str: ...


@dataclass(frozen=True)
class LLMProviderConnection:
    """Connection details for one selected provider."""

    api_key: str | None = None
    base_url: str | None = None

    def with_overrides(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "LLMProviderConnection":
        """Return a connection with explicit non-None overrides applied."""
        return LLMProviderConnection(
            api_key=self.api_key if api_key is None else api_key,
            base_url=self.base_url if base_url is None else base_url,
        )


@dataclass(frozen=True)
class LLMClientSettings:
    """Provider-neutral settings used to construct an LLM client."""

    model: str
    connection: LLMProviderConnection
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class LLMProviderProfile:
    """Factory, connection, and capability metadata for one provider."""

    name: str
    capabilities: LLMCapabilities
    create_client: Callable[[LLMClientSettings], LLMClient]
    default_connection: LLMProviderConnection = LLMProviderConnection()
    connection_from_config: Callable[[LLMConnectionConfig], LLMProviderConnection] | None = None

    def bind_connection(self, config: LLMConnectionConfig) -> LLMProviderConnection:
        """Resolve this provider's connection from the shared runtime config."""
        configured = (
            self.connection_from_config(config)
            if self.connection_from_config is not None
            else self.default_connection
        )
        return self.default_connection.with_overrides(
            api_key=configured.api_key,
            base_url=configured.base_url,
        )


# Compatibility alias retained for callers that imported the original registry type.
LLMProvider = LLMProviderProfile


LLM_PROVIDER_REGISTRY: dict[str, LLMProviderProfile] = {
    "openai": LLMProviderProfile(
        name="openai",
        capabilities=OPENAI_CAPABILITIES,
        default_connection=LLMProviderConnection(),
        connection_from_config=lambda config: LLMProviderConnection(api_key=config.openai_api_key),
        create_client=lambda settings: _create_openai_client(settings),
    ),
    "deepseek": LLMProviderProfile(
        name="deepseek",
        capabilities=DEEPSEEK_CAPABILITIES,
        default_connection=LLMProviderConnection(base_url=DEFAULT_DEEPSEEK_BASE_URL),
        connection_from_config=lambda config: LLMProviderConnection(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        ),
        create_client=lambda settings: _create_deepseek_client(settings),
    ),
    "local": LLMProviderProfile(
        name="local",
        capabilities=LOCAL_CAPABILITIES,
        default_connection=LLMProviderConnection(base_url=DEFAULT_LOCAL_BASE_URL),
        connection_from_config=lambda config: LLMProviderConnection(
            api_key=config.local_api_key,
            base_url=config.local_base_url,
        ),
        create_client=lambda settings: _create_local_client(settings),
    ),
    "openai-compatible": LLMProviderProfile(
        name="openai-compatible",
        capabilities=HOSTED_OPENAI_COMPATIBLE_CAPABILITIES,
        connection_from_config=lambda config: LLMProviderConnection(
            api_key=config.openai_compatible_api_key,
            base_url=config.openai_compatible_base_url,
        ),
        create_client=lambda settings: _create_hosted_compatible_client(settings),
    ),
    "openrouter": LLMProviderProfile(
        name="openrouter",
        capabilities=OPENROUTER_CAPABILITIES,
        default_connection=LLMProviderConnection(base_url=DEFAULT_OPENROUTER_BASE_URL),
        connection_from_config=lambda config: LLMProviderConnection(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
        ),
        create_client=lambda settings: _create_openrouter_client(settings),
    ),
}


def supported_llm_providers() -> set[str]:
    """Return configured provider names."""
    return set(LLM_PROVIDER_REGISTRY)


def provider_capabilities(provider: str) -> LLMCapabilities:
    """Return capability metadata for one registered provider."""
    return _provider_profile(provider).capabilities


def provider_connection_from_config(
    provider: str,
    config: LLMConnectionConfig,
) -> LLMProviderConnection:
    """Resolve one registered provider connection from runtime config."""
    return _provider_profile(provider).bind_connection(config)


def create_llm_client(
    *,
    provider: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
    connection: LLMProviderConnection | None = None,
    openai_api_key: str | None = None,
    deepseek_api_key: str | None = None,
    deepseek_base_url: str | None = None,
    local_api_key: str | None = None,
    local_base_url: str | None = None,
    openai_compatible_api_key: str | None = None,
    openai_compatible_base_url: str | None = None,
    openrouter_api_key: str | None = None,
    openrouter_base_url: str | None = None,
) -> LLMClient:
    """Create an LLM client for the selected provider.

    ``connection`` is the provider-neutral construction path. The named
    credential arguments remain accepted for compatibility with existing SDK
    callers and are normalized into the same connection object.
    """
    normalized_provider = provider.lower()
    provider_profile = _provider_profile(normalized_provider)
    try:
        model_capabilities = resolve_model_capabilities(normalized_provider, model)
    except ValueError as exc:
        raise LLMConfigurationError(str(exc)) from exc

    selected_connection = connection or _legacy_connection(
        provider_profile,
        openai_api_key=openai_api_key,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        local_api_key=local_api_key,
        local_base_url=local_base_url,
        openai_compatible_api_key=openai_compatible_api_key,
        openai_compatible_base_url=openai_compatible_base_url,
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url,
    )
    selected_connection = provider_profile.default_connection.with_overrides(
        api_key=selected_connection.api_key,
        base_url=selected_connection.base_url,
    )
    client = provider_profile.create_client(
        LLMClientSettings(
            model=model,
            connection=selected_connection,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    )
    client.model_capabilities = model_capabilities
    return client


def _provider_profile(provider: str) -> LLMProviderProfile:
    normalized_provider = provider.lower()
    provider_profile = LLM_PROVIDER_REGISTRY.get(normalized_provider)
    if provider_profile is None:
        raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")
    return provider_profile


def _legacy_connection(
    profile: LLMProviderProfile,
    *,
    openai_api_key: str | None,
    deepseek_api_key: str | None,
    deepseek_base_url: str | None,
    local_api_key: str | None,
    local_base_url: str | None,
    openai_compatible_api_key: str | None,
    openai_compatible_base_url: str | None,
    openrouter_api_key: str | None,
    openrouter_base_url: str | None,
) -> LLMProviderConnection:
    if profile.name == "openai":
        return LLMProviderConnection(api_key=openai_api_key)
    if profile.name == "deepseek":
        return LLMProviderConnection(api_key=deepseek_api_key, base_url=deepseek_base_url)
    if profile.name == "local":
        return LLMProviderConnection(api_key=local_api_key, base_url=local_base_url)
    if profile.name == "openai-compatible":
        return LLMProviderConnection(
            api_key=openai_compatible_api_key,
            base_url=openai_compatible_base_url,
        )
    if profile.name == "openrouter":
        return LLMProviderConnection(api_key=openrouter_api_key, base_url=openrouter_base_url)
    return profile.default_connection


def _create_openai_client(settings: LLMClientSettings) -> LLMClient:
    return OpenAIResponsesClient(
        model=settings.model,
        api_key=settings.connection.api_key,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def _create_deepseek_client(settings: LLMClientSettings) -> LLMClient:
    return DeepSeekChatCompletionsClient(
        model=settings.model,
        api_key=settings.connection.api_key,
        base_url=settings.connection.base_url or DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def _create_local_client(settings: LLMClientSettings) -> LLMClient:
    return LocalOpenAICompatibleClient(
        model=settings.model,
        api_key=settings.connection.api_key,
        base_url=settings.connection.base_url or DEFAULT_LOCAL_BASE_URL,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def _create_hosted_compatible_client(settings: LLMClientSettings) -> LLMClient:
    return HostedOpenAICompatibleClient(
        model=settings.model,
        api_key=settings.connection.api_key,
        base_url=settings.connection.base_url,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def _create_openrouter_client(settings: LLMClientSettings) -> LLMClient:
    return OpenRouterChatCompletionsClient(
        model=settings.model,
        api_key=settings.connection.api_key,
        base_url=settings.connection.base_url or DEFAULT_OPENROUTER_BASE_URL,
        timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
