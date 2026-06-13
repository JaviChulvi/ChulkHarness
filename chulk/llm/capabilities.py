"""Provider capability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OPENAI_GPT_4_1_CONTEXT_WINDOW_TOKENS = 1_047_576
OPENAI_GPT_4_1_MAX_OUTPUT_TOKENS = 32_768
OPENAI_GPT_4_1_DEFAULT_RESPONSE_RESERVE_TOKENS = 8_192
DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS = 1_000_000
DEEPSEEK_V4_MAX_OUTPUT_TOKENS = 384_000
DEEPSEEK_V4_DEFAULT_RESPONSE_RESERVE_TOKENS = 16_384

OPENAI_GPT_4_1_LIMITS = (
    OPENAI_GPT_4_1_CONTEXT_WINDOW_TOKENS,
    OPENAI_GPT_4_1_MAX_OUTPUT_TOKENS,
    OPENAI_GPT_4_1_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
DEEPSEEK_V4_LIMITS = (
    DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS,
    DEEPSEEK_V4_MAX_OUTPUT_TOKENS,
    DEEPSEEK_V4_DEFAULT_RESPONSE_RESERVE_TOKENS,
)


@dataclass(frozen=True)
class LLMCapabilities:
    """Capabilities exposed by one provider implementation."""

    supports_structured_output: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = False
    api_style: Literal["responses", "chat_completions"] = "chat_completions"


@dataclass(frozen=True)
class LLMModelCapabilities:
    """Concrete token limits for one provider/model pair."""

    provider: str
    model: str
    context_window_tokens: int
    max_output_tokens: int
    default_response_reserve_tokens: int

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be greater than zero")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be greater than zero")
        if self.default_response_reserve_tokens < 1:
            raise ValueError("default_response_reserve_tokens must be greater than zero")
        if self.default_response_reserve_tokens > self.max_output_tokens:
            raise ValueError("default_response_reserve_tokens cannot exceed max_output_tokens")

    @property
    def input_budget_tokens(self) -> int:
        return max(0, self.context_window_tokens - self.default_response_reserve_tokens)

    def to_dict(self) -> dict[str, int | str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "default_response_reserve_tokens": self.default_response_reserve_tokens,
            "input_budget_tokens": self.input_budget_tokens,
        }


MODEL_CAPABILITIES: dict[tuple[str, str], LLMModelCapabilities] = {}


def register_model_capabilities(capabilities: LLMModelCapabilities) -> None:
    """Register token metadata for one exact provider/model pair."""
    key = _model_key(capabilities.provider, capabilities.model)
    MODEL_CAPABILITIES[key] = capabilities


def resolve_model_capabilities(provider: str, model: str) -> LLMModelCapabilities:
    """Return required token metadata for a configured model."""
    key = _model_key(provider, model)
    capabilities = MODEL_CAPABILITIES.get(key)
    if capabilities is not None:
        return capabilities

    capabilities = _resolve_model_family_capabilities(provider, model)
    if capabilities is not None:
        return capabilities

    supported = ", ".join(
        f"{item_provider}/{item_model}" for item_provider, item_model in sorted(MODEL_CAPABILITIES)
    )
    raise ValueError(
        f"No token capability metadata configured for {provider}/{model}. "
        f"Add this model to chulk/llm/capabilities.py. Supported models: {supported}"
    )


def _model_key(provider: str, model: str) -> tuple[str, str]:
    return provider.lower().strip(), model.lower().strip()


def _resolve_model_family_capabilities(provider: str, model: str) -> LLMModelCapabilities | None:
    normalized_provider, normalized_model = _model_key(provider, model)
    family_prefixes = [
        ("openai", "gpt-4.1-mini-", *OPENAI_GPT_4_1_LIMITS),
        ("openai", "gpt-4.1-nano-", *OPENAI_GPT_4_1_LIMITS),
        ("openai", "gpt-4.1-", *OPENAI_GPT_4_1_LIMITS),
        ("deepseek", "deepseek-v4-flash-", *DEEPSEEK_V4_LIMITS),
        ("deepseek", "deepseek-v4-pro-", *DEEPSEEK_V4_LIMITS),
    ]
    for family_provider, prefix, context_window, max_output, response_reserve in family_prefixes:
        if normalized_provider == family_provider and normalized_model.startswith(prefix):
            return LLMModelCapabilities(
                provider=normalized_provider,
                model=normalized_model,
                context_window_tokens=context_window,
                max_output_tokens=max_output,
                default_response_reserve_tokens=response_reserve,
            )
    return None


for _provider, _model, _context_window, _max_output, _response_reserve in [
    ("openai", "gpt-4.1", *OPENAI_GPT_4_1_LIMITS),
    ("openai", "gpt-4.1-mini", *OPENAI_GPT_4_1_LIMITS),
    ("openai", "gpt-4.1-mini-2025-04-14", *OPENAI_GPT_4_1_LIMITS),
    ("openai", "gpt-4.1-nano", *OPENAI_GPT_4_1_LIMITS),
    ("deepseek", "deepseek-v4-flash", *DEEPSEEK_V4_LIMITS),
    ("deepseek", "deepseek-v4-pro", *DEEPSEEK_V4_LIMITS),
    ("deepseek", "deepseek-chat", *DEEPSEEK_V4_LIMITS),
    ("deepseek", "deepseek-reasoner", *DEEPSEEK_V4_LIMITS),
]:
    register_model_capabilities(
        LLMModelCapabilities(
            provider=_provider,
            model=_model,
            context_window_tokens=_context_window,
            max_output_tokens=_max_output,
            default_response_reserve_tokens=_response_reserve,
        )
    )
