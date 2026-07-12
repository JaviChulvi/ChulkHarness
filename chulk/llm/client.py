"""Compatibility imports for LLM provider clients and shared interfaces."""

from chulk.llm.base import (
    LLMActionError,
    LLMActionResult,
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMErrorClassification,
    LLMErrorCode,
    LLMStreamChunk,
)
from chulk.llm.capabilities import LLMCapabilities, LLMModelCapabilities, resolve_model_capabilities
from chulk.llm.factory import (
    LLM_PROVIDER_REGISTRY,
    LLMClientSettings,
    LLMProvider,
    LLMProviderConnection,
    LLMProviderProfile,
    create_llm_client,
    provider_capabilities,
    provider_connection_from_config,
    supported_llm_providers,
)
from chulk.llm.providers.anthropic import AnthropicMessagesClient
from chulk.llm.providers.bedrock import BedrockOpenAICompatibleClient
from chulk.llm.providers.deepseek import DeepSeekChatCompletionsClient
from chulk.llm.providers.gemini import GeminiGenerateContentClient
from chulk.llm.providers.compatible import (
    HostedOpenAICompatibleClient,
    OpenRouterChatCompletionsClient,
)
from chulk.llm.providers.local import LocalOpenAICompatibleClient
from chulk.llm.providers.openai import OpenAIResponsesClient
from chulk.llm.usage import LLMCost, LLMResponse, LLMUsage

__all__ = [
    "AnthropicMessagesClient",
    "BedrockOpenAICompatibleClient",
    "DeepSeekChatCompletionsClient",
    "HostedOpenAICompatibleClient",
    "GeminiGenerateContentClient",
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
    "OpenAIResponsesClient",
    "OpenRouterChatCompletionsClient",
    "create_llm_client",
    "provider_capabilities",
    "provider_connection_from_config",
    "resolve_model_capabilities",
    "supported_llm_providers",
]
