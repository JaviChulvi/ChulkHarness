"""Compatibility imports for LLM provider clients and shared interfaces."""

from chulk.llm.base import (
    LLMActionError,
    LLMActionResult,
    LLMClient,
    LLMConfigurationError,
    LLMError,
    LLMStreamChunk,
)
from chulk.llm.capabilities import LLMCapabilities, LLMModelCapabilities, resolve_model_capabilities
from chulk.llm.factory import (
    LLM_PROVIDER_REGISTRY,
    LLMClientSettings,
    LLMProvider,
    create_llm_client,
    supported_llm_providers,
)
from chulk.llm.providers.deepseek import DeepSeekChatCompletionsClient
from chulk.llm.providers.local import LocalOpenAICompatibleClient
from chulk.llm.providers.openai import OpenAIResponsesClient
from chulk.llm.usage import LLMCost, LLMResponse, LLMUsage

__all__ = [
    "DeepSeekChatCompletionsClient",
    "LLMActionError",
    "LLMActionResult",
    "LLMClient",
    "LLMCapabilities",
    "LLMClientSettings",
    "LLMConfigurationError",
    "LLMCost",
    "LLMError",
    "LLMModelCapabilities",
    "LLMProvider",
    "LLM_PROVIDER_REGISTRY",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMUsage",
    "LocalOpenAICompatibleClient",
    "OpenAIResponsesClient",
    "create_llm_client",
    "resolve_model_capabilities",
    "supported_llm_providers",
]
