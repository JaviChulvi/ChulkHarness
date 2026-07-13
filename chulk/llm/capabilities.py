"""Provider capability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chulk.llm.model_catalog import MODEL_CATALOG, ModelSpec, resolve_model_spec

LOCAL_DEFAULT_CONTEXT_WINDOW_TOKENS = 131_072
LOCAL_DEFAULT_RESPONSE_RESERVE_TOKENS = 4_096
COMPATIBLE_DEFAULT_CONTEXT_WINDOW_TOKENS = 8_192
COMPATIBLE_DEFAULT_RESPONSE_RESERVE_TOKENS = 2_048
ANTHROPIC_DEFAULT_CONTEXT_WINDOW_TOKENS = 200_000
ANTHROPIC_DEFAULT_RESPONSE_RESERVE_TOKENS = 4_096
BEDROCK_DEFAULT_CONTEXT_WINDOW_TOKENS = 8_192
BEDROCK_DEFAULT_RESPONSE_RESERVE_TOKENS = 2_048
GEMINI_DEFAULT_CONTEXT_WINDOW_TOKENS = 1_048_576
GEMINI_DEFAULT_RESPONSE_RESERVE_TOKENS = 8_192

_OPENAI_GPT_4_1_SPEC = resolve_model_spec("openai", "gpt-4.1")
_DEEPSEEK_V4_SPEC = resolve_model_spec("deepseek", "deepseek-v4-flash")
_LOCAL_QWEN_SPEC = resolve_model_spec("local", "qwen/qwen3.5-35b-a3b")
assert _OPENAI_GPT_4_1_SPEC is not None
assert _DEEPSEEK_V4_SPEC is not None
assert _LOCAL_QWEN_SPEC is not None

OPENAI_GPT_4_1_CONTEXT_WINDOW_TOKENS = (
    _OPENAI_GPT_4_1_SPEC.limits.context_window_tokens
)
OPENAI_GPT_4_1_DEFAULT_RESPONSE_RESERVE_TOKENS = (
    _OPENAI_GPT_4_1_SPEC.limits.default_response_reserve_tokens
)
DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS = _DEEPSEEK_V4_SPEC.limits.context_window_tokens
DEEPSEEK_V4_DEFAULT_RESPONSE_RESERVE_TOKENS = (
    _DEEPSEEK_V4_SPEC.limits.default_response_reserve_tokens
)
LOCAL_QWEN_3_5_35B_CONTEXT_WINDOW_TOKENS = (
    _LOCAL_QWEN_SPEC.limits.context_window_tokens
)

OPENAI_GPT_4_1_LIMITS = (
    OPENAI_GPT_4_1_CONTEXT_WINDOW_TOKENS,
    OPENAI_GPT_4_1_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
DEEPSEEK_V4_LIMITS = (
    DEEPSEEK_V4_CONTEXT_WINDOW_TOKENS,
    DEEPSEEK_V4_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
LOCAL_DEFAULT_LIMITS = (
    LOCAL_DEFAULT_CONTEXT_WINDOW_TOKENS,
    LOCAL_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
LOCAL_QWEN_3_5_35B_LIMITS = (
    LOCAL_QWEN_3_5_35B_CONTEXT_WINDOW_TOKENS,
    LOCAL_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
COMPATIBLE_DEFAULT_LIMITS = (
    COMPATIBLE_DEFAULT_CONTEXT_WINDOW_TOKENS,
    COMPATIBLE_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
ANTHROPIC_DEFAULT_LIMITS = (
    ANTHROPIC_DEFAULT_CONTEXT_WINDOW_TOKENS,
    ANTHROPIC_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
BEDROCK_DEFAULT_LIMITS = (
    BEDROCK_DEFAULT_CONTEXT_WINDOW_TOKENS,
    BEDROCK_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
GEMINI_DEFAULT_LIMITS = (
    GEMINI_DEFAULT_CONTEXT_WINDOW_TOKENS,
    GEMINI_DEFAULT_RESPONSE_RESERVE_TOKENS,
)


@dataclass(frozen=True)
class LLMCapabilities:
    """Capabilities exposed by one provider implementation."""

    supports_structured_output: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = False
    supports_native_tool_calling: bool = False
    supports_hosted_mcp_tools: bool = False
    api_style: Literal[
        "responses",
        "chat_completions",
        "messages",
        "generate_content",
    ] = "chat_completions"


@dataclass(frozen=True)
class LLMModelCapabilities:
    """Concrete token limits for one provider/model pair."""

    provider: str
    model: str
    context_window_tokens: int
    default_response_reserve_tokens: int
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name, required_value in (
            ("context_window_tokens", self.context_window_tokens),
            (
                "default_response_reserve_tokens",
                self.default_response_reserve_tokens,
            ),
        ):
            if (
                isinstance(required_value, bool)
                or not isinstance(required_value, int)
                or required_value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.default_response_reserve_tokens > self.context_window_tokens:
            raise ValueError(
                "default_response_reserve_tokens cannot exceed context_window_tokens"
            )
        for name, optional_value in (
            ("max_input_tokens", self.max_input_tokens),
            ("max_output_tokens", self.max_output_tokens),
        ):
            if optional_value is None:
                continue
            if (
                isinstance(optional_value, bool)
                or not isinstance(optional_value, int)
                or optional_value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
            if optional_value > self.context_window_tokens:
                raise ValueError(f"{name} cannot exceed context_window_tokens")

    @property
    def input_budget_tokens(self) -> int:
        context_budget = max(
            0,
            self.context_window_tokens - self.default_response_reserve_tokens,
        )
        if self.max_input_tokens is None:
            return context_budget
        return min(context_budget, self.max_input_tokens)

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "provider": self.provider,
            "model": self.model,
            "context_window_tokens": self.context_window_tokens,
            "default_response_reserve_tokens": self.default_response_reserve_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "input_budget_tokens": self.input_budget_tokens,
        }


MODEL_CAPABILITIES: dict[tuple[str, str], LLMModelCapabilities] = {}


def register_model_capabilities(capabilities: LLMModelCapabilities) -> None:
    """Register token metadata for one exact provider/model pair."""
    key = _model_key(capabilities.provider, capabilities.model)
    MODEL_CAPABILITIES[key] = capabilities


def conservative_model_capabilities(
    capabilities: list[LLMModelCapabilities] | tuple[LLMModelCapabilities, ...],
) -> LLMModelCapabilities:
    """Return limits that are safe for every model in a fallback path."""
    if not capabilities:
        raise ValueError("At least one model capability record is required")
    context_window = min(item.context_window_tokens for item in capabilities)
    response_reserve = min(
        max(item.default_response_reserve_tokens for item in capabilities),
        context_window,
    )
    providers = ",".join(dict.fromkeys(item.provider for item in capabilities))
    models = ",".join(dict.fromkeys(item.model for item in capabilities))
    max_input_tokens = _conservative_optional_limit(
        capabilities,
        "max_input_tokens",
    )
    max_output_tokens = _conservative_optional_limit(
        capabilities,
        "max_output_tokens",
    )
    return LLMModelCapabilities(
        provider=providers,
        model=models,
        context_window_tokens=context_window,
        default_response_reserve_tokens=response_reserve,
        max_input_tokens=(
            min(max_input_tokens, context_window)
            if max_input_tokens is not None
            else None
        ),
        max_output_tokens=(
            min(max_output_tokens, context_window)
            if max_output_tokens is not None
            else None
        ),
    )


def resolve_model_capabilities(provider: str, model: str) -> LLMModelCapabilities:
    """Return required token metadata for a configured model."""
    key = _model_key(provider, model)
    capabilities = MODEL_CAPABILITIES.get(key)
    if capabilities is not None:
        return capabilities

    spec = resolve_model_spec(provider, model)
    if spec is not None:
        return _capabilities_from_spec(provider, model, spec)

    capabilities = _resolve_model_family_capabilities(provider, model)
    if capabilities is not None:
        return capabilities

    supported_keys = set(MODEL_CATALOG.canonical_keys) | set(MODEL_CAPABILITIES)
    supported = ", ".join(
        f"{item_provider}/{item_model}"
        for item_provider, item_model in sorted(supported_keys)
    )
    raise ValueError(
        f"No token capability metadata configured for {provider}/{model}. "
        f"Add this model to chulk/llm/model_catalog.py. Supported models: {supported}"
    )


def _model_key(provider: str, model: str) -> tuple[str, str]:
    return provider.lower().strip(), model.lower().strip()


def _resolve_model_family_capabilities(provider: str, model: str) -> LLMModelCapabilities | None:
    normalized_provider, normalized_model = _model_key(provider, model)
    if normalized_provider == "anthropic":
        return LLMModelCapabilities(
            provider=normalized_provider,
            model=normalized_model,
            context_window_tokens=ANTHROPIC_DEFAULT_CONTEXT_WINDOW_TOKENS,
            default_response_reserve_tokens=ANTHROPIC_DEFAULT_RESPONSE_RESERVE_TOKENS,
        )
    if normalized_provider == "bedrock":
        return LLMModelCapabilities(
            provider=normalized_provider,
            model=normalized_model,
            context_window_tokens=BEDROCK_DEFAULT_CONTEXT_WINDOW_TOKENS,
            default_response_reserve_tokens=BEDROCK_DEFAULT_RESPONSE_RESERVE_TOKENS,
        )
    if normalized_provider == "gemini":
        return LLMModelCapabilities(
            provider=normalized_provider,
            model=normalized_model,
            context_window_tokens=GEMINI_DEFAULT_CONTEXT_WINDOW_TOKENS,
            default_response_reserve_tokens=GEMINI_DEFAULT_RESPONSE_RESERVE_TOKENS,
        )
    if normalized_provider in {"openai-compatible", "openrouter"}:
        return LLMModelCapabilities(
            provider=normalized_provider,
            model=normalized_model,
            context_window_tokens=COMPATIBLE_DEFAULT_CONTEXT_WINDOW_TOKENS,
            default_response_reserve_tokens=COMPATIBLE_DEFAULT_RESPONSE_RESERVE_TOKENS,
        )
    if normalized_provider == "local":
        return LLMModelCapabilities(
            provider=normalized_provider,
            model=normalized_model,
            context_window_tokens=LOCAL_DEFAULT_CONTEXT_WINDOW_TOKENS,
            default_response_reserve_tokens=LOCAL_DEFAULT_RESPONSE_RESERVE_TOKENS,
        )
    return None


def _capabilities_from_spec(
    provider: str,
    model: str,
    spec: ModelSpec,
) -> LLMModelCapabilities:
    normalized_provider, normalized_model = _model_key(provider, model)
    return LLMModelCapabilities(
        provider=normalized_provider,
        model=normalized_model,
        context_window_tokens=spec.limits.context_window_tokens,
        default_response_reserve_tokens=(
            spec.limits.default_response_reserve_tokens
        ),
        max_input_tokens=spec.limits.max_input_tokens,
        max_output_tokens=spec.limits.max_output_tokens,
    )


def _conservative_optional_limit(
    capabilities: list[LLMModelCapabilities] | tuple[LLMModelCapabilities, ...],
    field_name: Literal["max_input_tokens", "max_output_tokens"],
) -> int | None:
    values = [getattr(item, field_name) for item in capabilities]
    if any(value is None for value in values):
        return None
    return min(value for value in values if value is not None)
