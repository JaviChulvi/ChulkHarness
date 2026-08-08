"""LLM provider clients and shared interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chulk._lazy import public_dir, resolve_export

if TYPE_CHECKING:
    from chulk.llm.client import (
        AnthropicMessagesClient,
        BedrockOpenAICompatibleClient,
        DeepSeekChatCompletionsClient,
        GeminiGenerateContentClient,
        HostedOpenAICompatibleClient,
        LLMActionError,
        LLMActionResult,
        LLMClient,
        LLMCapabilities,
        LLMClientSettings,
        LLMConfigurationError,
        LLMCost,
        LLMError,
        LLMErrorClassification,
        LLMErrorCode,
        LLMModelCapabilities,
        LLMProvider,
        LLMProviderConnection,
        LLMProviderProfile,
        LLM_PROVIDER_REGISTRY,
        LLMResponse,
        LLMStreamChunk,
        LLMUsage,
        LocalOpenAICompatibleClient,
        MoonshotChatCompletionsClient,
        OpenAIResponsesClient,
        OpenRouterChatCompletionsClient,
        PlanningToolAvailability,
        create_llm_client,
        provider_capabilities,
        provider_connection_from_config,
        resolve_model_capabilities,
        supported_llm_providers,
    )
    from chulk.llm.capabilities import conservative_model_capabilities
    from chulk.llm.public import (
        AnthropicProvider,
        BedrockProvider,
        BindableLLM,
        DeepSeekProvider,
        FallbackChain,
        FallbackStrategy,
        GeminiProvider,
        LocalProvider,
        MoonshotProvider,
        OpenAICompatibleProvider,
        OpenAIProvider,
        OpenRouterProvider,
        ProviderAttempt,
    )

__all__ = [
    "AnthropicMessagesClient",
    "AnthropicProvider",
    "BedrockOpenAICompatibleClient",
    "BedrockProvider",
    "BindableLLM",
    "DeepSeekProvider",
    "DeepSeekChatCompletionsClient",
    "FallbackChain",
    "FallbackStrategy",
    "GeminiGenerateContentClient",
    "GeminiProvider",
    "HostedOpenAICompatibleClient",
    "LLMActionError",
    "LLMActionResult",
    "LLMClient",
    "LLMCapabilities",
    "LLMClientSettings",
    "LLMConfigurationError",
    "LLMCost",
    "LLMError",
    "LLMErrorClassification",
    "LLMErrorCode",
    "LLMModelCapabilities",
    "LLMProvider",
    "LLMProviderConnection",
    "LLMProviderProfile",
    "LLM_PROVIDER_REGISTRY",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMUsage",
    "LocalOpenAICompatibleClient",
    "LocalProvider",
    "MoonshotChatCompletionsClient",
    "MoonshotProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "OpenAIResponsesClient",
    "OpenRouterChatCompletionsClient",
    "OpenRouterProvider",
    "PlanningToolAvailability",
    "ProviderAttempt",
    "create_llm_client",
    "conservative_model_capabilities",
    "provider_capabilities",
    "provider_connection_from_config",
    "resolve_model_capabilities",
    "supported_llm_providers",
]


_EXPORT_MODULES = (
    "chulk.llm.client",
    "chulk.llm.capabilities",
    "chulk.llm.public",
)


if not TYPE_CHECKING:

    def __getattr__(name: str) -> object:
        return resolve_export(
            name,
            public_names=__all__,
            owner_modules=_EXPORT_MODULES,
            namespace=globals(),
        )

    def __dir__() -> list[str]:
        return public_dir(__all__, globals())
