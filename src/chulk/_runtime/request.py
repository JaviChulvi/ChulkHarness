"""Model client construction for runtime assembly."""

from __future__ import annotations

from chulk.config import Config
from chulk.llm import (
    LLMClient,
    LLMModelCapabilities,
)
from chulk.llm.capabilities import resolve_runtime_model_capabilities
from chulk.llm.factory import _bind_llm_client


def default_llm_client_factory(config: Config) -> LLMClient:
    """Create the configured model client."""
    return _bind_llm_client(
        config,
        provider=config.llm_provider,
        model=config.model,
        local_context_window_tokens=config.local_context_window_tokens,
        timeout_seconds=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )


def client_model_capabilities(
    client: LLMClient,
    config: Config,
) -> LLMModelCapabilities:
    """Return bound-client capabilities or resolve the configured fallback."""
    capabilities = getattr(client, "model_capabilities", None)
    if isinstance(capabilities, LLMModelCapabilities):
        return capabilities
    return resolve_runtime_model_capabilities(
        config.llm_provider,
        config.model,
        local_context_window_tokens=config.local_context_window_tokens,
    )
