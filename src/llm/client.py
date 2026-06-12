"""Compatibility imports for LLM provider clients and shared interfaces."""

from src.llm.base import LLMActionError, LLMActionResult, LLMClient, LLMConfigurationError, LLMError
from src.llm.capabilities import LLMCapabilities
from src.llm.factory import (
    LLM_PROVIDER_REGISTRY,
    LLMClientSettings,
    LLMProvider,
    create_llm_client,
    supported_llm_providers,
)
from src.llm.providers.deepseek import DeepSeekChatCompletionsClient
from src.llm.providers.openai import OpenAIResponsesClient

__all__ = [
    "DeepSeekChatCompletionsClient",
    "LLMActionError",
    "LLMActionResult",
    "LLMClient",
    "LLMCapabilities",
    "LLMClientSettings",
    "LLMConfigurationError",
    "LLMError",
    "LLMProvider",
    "LLM_PROVIDER_REGISTRY",
    "OpenAIResponsesClient",
    "create_llm_client",
    "supported_llm_providers",
]
