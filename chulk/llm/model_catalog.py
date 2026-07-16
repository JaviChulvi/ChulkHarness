"""Immutable model metadata used by capability and pricing lookups."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from typing import Literal


USD = "USD"
ModelStatus = Literal["stable", "preview", "deprecated"]


def _normalized_identifier(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validated_provenance(
    label: str,
    source_urls: tuple[str, ...],
    last_checked: date | None,
    *,
    required: bool,
) -> tuple[str, ...]:
    if not isinstance(source_urls, tuple):
        raise TypeError(f"{label} source_urls must be a tuple")
    for source_url in source_urls:
        if not isinstance(source_url, str):
            raise TypeError(f"{label} source URLs must be strings")
    urls = tuple(url.strip() for url in source_urls)
    if required and (not urls or last_checked is None):
        raise ValueError(f"{label} source_urls and last_checked are required")
    if bool(urls) != (last_checked is not None):
        raise ValueError(
            f"{label} source_urls and last_checked must be configured together"
        )
    if last_checked is not None and not isinstance(last_checked, date):
        raise TypeError(f"{label} last_checked must be a date")
    if len(set(urls)) != len(urls):
        raise ValueError(f"Duplicate {label} source URLs configured")
    for source_url in urls:
        if not source_url.startswith(("https://", "http://")):
            raise ValueError(f"{label} source must be an HTTP(S) URL: {source_url!r}")
    return urls


@dataclass(frozen=True, slots=True)
class ModelLimits:
    """Published token limits plus Chulk's internal prompt-budget policy."""

    context_window_tokens: int
    default_response_reserve_tokens: int
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    source_urls: tuple[str, ...] = ()
    last_checked: date | None = None

    def __post_init__(self) -> None:
        _require_positive_int("context_window_tokens", self.context_window_tokens)
        _require_positive_int(
            "default_response_reserve_tokens",
            self.default_response_reserve_tokens,
        )
        if self.default_response_reserve_tokens > self.context_window_tokens:
            raise ValueError(
                "default_response_reserve_tokens cannot exceed context_window_tokens"
            )
        for name, value in (
            ("max_input_tokens", self.max_input_tokens),
            ("max_output_tokens", self.max_output_tokens),
        ):
            if value is None:
                continue
            _require_positive_int(name, value)
            if value > self.context_window_tokens:
                raise ValueError(f"{name} cannot exceed context_window_tokens")
        source_urls = _validated_provenance(
            "limit",
            self.source_urls,
            self.last_checked,
            required=True,
        )
        object.__setattr__(self, "source_urls", source_urls)

    @property
    def primary_source_url(self) -> str:
        return self.source_urls[0]


@dataclass(frozen=True, slots=True)
class LongContextPricing:
    """Full-request rates used above one published input-token threshold."""

    applies_above_input_tokens: int
    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None
    cache_write_input_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        _require_positive_int(
            "applies_above_input_tokens",
            self.applies_above_input_tokens,
        )
        _validate_decimal_rates(
            (
                ("input_per_million", self.input_per_million),
                ("output_per_million", self.output_per_million),
                ("cached_input_per_million", self.cached_input_per_million),
                (
                    "cache_write_input_per_million",
                    self.cache_write_input_per_million,
                ),
            )
        )


def _validate_decimal_rates(
    rates: tuple[tuple[str, Decimal | None], ...],
) -> None:
    for name, value in rates:
        if value is None:
            continue
        if not isinstance(value, Decimal):
            raise TypeError(f"{name} must be a Decimal")
        if not value.is_finite() or value < 0:
            raise ValueError(f"{name} must be a finite non-negative Decimal")


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """Current standard per-million-token prices for one model."""

    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None
    currency: str = USD
    source_urls: tuple[str, ...] = ()
    last_checked: date | None = None
    long_context: LongContextPricing | None = None
    cache_write_input_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        _validate_decimal_rates(
            (
                ("input_per_million", self.input_per_million),
                ("output_per_million", self.output_per_million),
                ("cached_input_per_million", self.cached_input_per_million),
                (
                    "cache_write_input_per_million",
                    self.cache_write_input_per_million,
                ),
            )
        )
        if self.long_context is not None and not isinstance(
            self.long_context,
            LongContextPricing,
        ):
            raise TypeError("long_context must be a LongContextPricing record or None")
        if not isinstance(self.currency, str):
            raise TypeError("currency must be a string")
        currency = self.currency.strip().upper()
        if not currency:
            raise ValueError("currency cannot be empty")
        source_urls = _validated_provenance(
            "pricing",
            self.source_urls,
            self.last_checked,
            required=True,
        )
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "source_urls", source_urls)

    @property
    def primary_source_url(self) -> str:
        return self.source_urls[0]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One canonical model record shared by every model-metadata consumer."""

    provider: str
    model: str
    limits: ModelLimits
    pricing: TokenPricing | None = None
    aliases: tuple[str, ...] = ()
    status: ModelStatus = "stable"
    replacement_model: str | None = None
    retired_on: date | None = None
    lifecycle_source_urls: tuple[str, ...] = ()
    lifecycle_last_checked: date | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.limits, ModelLimits):
            raise TypeError("limits must be a ModelLimits record")
        if self.pricing is not None and not isinstance(self.pricing, TokenPricing):
            raise TypeError("pricing must be a TokenPricing record or None")
        if self.pricing is not None and self.pricing.long_context is not None:
            published_input_bound = (
                self.limits.max_input_tokens or self.limits.context_window_tokens
            )
            if (
                self.pricing.long_context.applies_above_input_tokens
                >= published_input_bound
            ):
                raise ValueError(
                    "long-context pricing threshold must be below the model's "
                    "published input bound"
                )
            if (
                self.pricing.cache_write_input_per_million is not None
                and self.pricing.long_context.cache_write_input_per_million is None
            ):
                raise ValueError(
                    "long-context cache-write pricing is required when the "
                    "standard tier has a cache-write rate"
                )
        if not isinstance(self.aliases, tuple):
            raise TypeError("aliases must be a tuple")
        provider = _normalized_identifier("provider", self.provider)
        model = _normalized_identifier("model", self.model)
        aliases = tuple(_normalized_identifier("alias", alias) for alias in self.aliases)
        if model in aliases:
            raise ValueError(f"Model alias duplicates canonical model: {provider}/{model}")
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"Duplicate aliases configured for {provider}/{model}")
        if self.status not in {"stable", "preview", "deprecated"}:
            raise ValueError(f"Unsupported model status: {self.status!r}")
        lifecycle_source_urls = _validated_provenance(
            "lifecycle",
            self.lifecycle_source_urls,
            self.lifecycle_last_checked,
            required=self.status != "stable",
        )
        if self.retired_on is not None and self.status != "deprecated":
            raise ValueError("retired_on is only valid for deprecated models")
        if self.retired_on is not None and not isinstance(self.retired_on, date):
            raise TypeError("retired_on must be a date")
        replacement = self.replacement_model
        if replacement is not None:
            if self.status != "deprecated":
                raise ValueError(
                    "replacement_model is only valid for deprecated models"
                )
            replacement = _normalized_identifier("replacement model", replacement)
            if replacement == model:
                raise ValueError("replacement_model cannot reference the same model")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "replacement_model", replacement)
        object.__setattr__(self, "lifecycle_source_urls", lifecycle_source_urls)


class ModelCatalog:
    """Validated, immutable indexes over model specifications."""

    __slots__ = ("_by_key", "_specs")

    def __init__(self, specs: Iterable[ModelSpec]) -> None:
        records = tuple(specs)
        by_key: dict[tuple[str, str], ModelSpec] = {}

        for spec in records:
            if not isinstance(spec, ModelSpec):
                raise TypeError("ModelCatalog entries must be ModelSpec records")
            for identity in (spec.model, *spec.aliases):
                key = (spec.provider, identity)
                existing = by_key.get(key)
                if existing is not None:
                    raise ValueError(
                        "Duplicate model identity "
                        f"{spec.provider}/{identity}: {existing.model} and {spec.model}"
                    )
                by_key[key] = spec

        for spec in records:
            if spec.replacement_model is None:
                continue
            replacement = by_key.get((spec.provider, spec.replacement_model))
            if replacement is None or replacement.model != spec.replacement_model:
                raise ValueError(
                    "replacement_model must name a canonical model in the same "
                    f"provider: {spec.provider}/{spec.replacement_model}"
                )

        self._specs = records
        self._by_key: Mapping[tuple[str, str], ModelSpec] = MappingProxyType(by_key)

    @property
    def specs(self) -> tuple[ModelSpec, ...]:
        return self._specs

    @property
    def canonical_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((spec.provider, spec.model) for spec in self._specs))

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def resolve(self, provider: str | None, model: str | None) -> ModelSpec | None:
        """Resolve an exact model or explicit alias to one shared record."""
        if not provider or not model:
            return None
        normalized_provider = provider.strip().lower()
        normalized_model = model.strip().lower()
        if not normalized_provider or not normalized_model:
            return None

        return self._by_key.get((normalized_provider, normalized_model))


_OPENAI_LIMITS_CHECKED_ON = date(2026, 7, 13)
_OPENAI_PRICING_CHECKED_ON = date(2026, 7, 13)
_OPENAI_LIFECYCLE_CHECKED_ON = date(2026, 7, 13)
_ANTHROPIC_LIMITS_CHECKED_ON = date(2026, 7, 13)
_ANTHROPIC_LIFECYCLE_CHECKED_ON = date(2026, 7, 13)
_GEMINI_LIMITS_CHECKED_ON = date(2026, 7, 13)
_GEMINI_PRICING_CHECKED_ON = date(2026, 7, 13)
_GEMINI_LIFECYCLE_CHECKED_ON = date(2026, 7, 13)
_DEEPSEEK_LIMITS_CHECKED_ON = date(2026, 7, 13)
_DEEPSEEK_PRICING_CHECKED_ON = date(2026, 7, 13)
_DEEPSEEK_LIFECYCLE_CHECKED_ON = date(2026, 7, 13)
_LOCAL_LIMITS_CHECKED_ON = date(2026, 7, 13)
_OPENAI_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
_OPENAI_DEPRECATIONS_URL = "https://developers.openai.com/api/docs/deprecations"
_ANTHROPIC_LIMIT_URLS = (
    "https://platform.claude.com/docs/en/about-claude/models/overview",
    "https://platform.claude.com/docs/en/api/models",
)
_ANTHROPIC_DEPRECATIONS_URL = (
    "https://platform.claude.com/docs/en/about-claude/model-deprecations"
)
_GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
_GEMINI_DEPRECATIONS_URL = "https://ai.google.dev/gemini-api/docs/deprecations"
_GEMINI_CHANGELOG_URL = "https://ai.google.dev/gemini-api/docs/changelog"
_GEMINI_MODELS_URL = "https://ai.google.dev/gemini-api/docs/models"
_GEMINI_FLASH_LITE_ALIAS_URL = (
    "https://developers.googleblog.com/en/continuing-to-bring-you-our-latest-"
    "models-with-an-improved-gemini-2-5-flash-and-flash-lite-release/"
)
_DEEPSEEK_V4_URL = "https://api-docs.deepseek.com/quick_start/pricing/"
_DEEPSEEK_V4_LIMIT_URLS = (
    _DEEPSEEK_V4_URL,
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json",
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json",
)
_DEEPSEEK_V4_ANNOUNCEMENT_URL = (
    "https://api-docs.deepseek.com/news/news260424/"
)
_GEMMA_4_API_LIMIT_URLS = (
    "https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api",
    "https://ai.google.dev/gemma/docs/core/model_card_4",
)
_LM_STUDIO_GEMMA_4_12B_URLS = (
    "https://lmstudio.ai/models/google/gemma-4-12b-qat",
    "https://lmstudio.ai/models/gemma-4",
)
_OLLAMA_GEMMA_4_12B_URL = "https://ollama.com/library/gemma4:12b"
_OLLAMA_GEMMA_4_12B_QAT_URL = (
    "https://ollama.com/library/gemma4:12b-it-qat"
)
_OLLAMA_GEMMA_3_12B_URL = "https://ollama.com/library/gemma3:12b"
_LM_STUDIO_QWEN_3_5_URL = (
    "https://lmstudio.ai/models/qwen/qwen3.5-35b-a3b"
)


def _pricing(
    *,
    input_rate: str,
    output_rate: str,
    cached_input_rate: str | None,
    source_urls: tuple[str, ...],
    last_checked: date,
    cache_write_input_rate: str | None = None,
    long_context: LongContextPricing | None = None,
) -> TokenPricing:
    return TokenPricing(
        input_per_million=Decimal(input_rate),
        cached_input_per_million=(
            Decimal(cached_input_rate) if cached_input_rate is not None else None
        ),
        cache_write_input_per_million=(
            Decimal(cache_write_input_rate)
            if cache_write_input_rate is not None
            else None
        ),
        output_per_million=Decimal(output_rate),
        source_urls=source_urls,
        last_checked=last_checked,
        long_context=long_context,
    )


def _long_context_pricing(
    *,
    above_input_tokens: int,
    input_rate: str,
    output_rate: str,
    cached_input_rate: str | None,
    cache_write_input_rate: str | None = None,
) -> LongContextPricing:
    return LongContextPricing(
        applies_above_input_tokens=above_input_tokens,
        input_per_million=Decimal(input_rate),
        cached_input_per_million=(
            Decimal(cached_input_rate) if cached_input_rate is not None else None
        ),
        cache_write_input_per_million=(
            Decimal(cache_write_input_rate)
            if cache_write_input_rate is not None
            else None
        ),
        output_per_million=Decimal(output_rate),
    )


def _openai_spec(
    model: str,
    *,
    context_window_tokens: int,
    max_output_tokens: int,
    default_response_reserve_tokens: int = 8_192,
    source_model: str | None = None,
    limits_source_urls: tuple[str, ...] | None = None,
    aliases: tuple[str, ...] = (),
    input_rate: str | None = None,
    cached_input_rate: str | None = None,
    cache_write_input_rate: str | None = None,
    output_rate: str | None = None,
    pricing_source_urls: tuple[str, ...] | None = None,
    long_context: LongContextPricing | None = None,
    status: ModelStatus = "stable",
    replacement_model: str | None = None,
    retired_on: date | None = None,
    lifecycle_source_urls: tuple[str, ...] = (),
    limits_last_checked: date = _OPENAI_LIMITS_CHECKED_ON,
    pricing_last_checked: date = _OPENAI_PRICING_CHECKED_ON,
    lifecycle_last_checked: date = _OPENAI_LIFECYCLE_CHECKED_ON,
) -> ModelSpec:
    source_url = (
        "https://developers.openai.com/api/docs/models/"
        f"{source_model or model}"
    )
    if (input_rate is None) != (output_rate is None):
        raise ValueError("OpenAI input and output pricing must be configured together")
    if cache_write_input_rate is not None and input_rate is None:
        raise ValueError("OpenAI cache-write pricing requires token pricing")
    pricing = (
        _pricing(
            input_rate=input_rate,
            cached_input_rate=cached_input_rate,
            cache_write_input_rate=cache_write_input_rate,
            output_rate=output_rate,
            source_urls=pricing_source_urls or (source_url,),
            last_checked=pricing_last_checked,
            long_context=long_context,
        )
        if input_rate is not None and output_rate is not None
        else None
    )
    if not lifecycle_source_urls and status == "deprecated":
        lifecycle_source_urls = (_OPENAI_DEPRECATIONS_URL,)
    return ModelSpec(
        provider="openai",
        model=model,
        aliases=aliases,
        limits=ModelLimits(
            context_window_tokens=context_window_tokens,
            default_response_reserve_tokens=default_response_reserve_tokens,
            max_output_tokens=max_output_tokens,
            source_urls=limits_source_urls or (source_url,),
            last_checked=limits_last_checked,
        ),
        pricing=pricing,
        status=status,
        replacement_model=replacement_model,
        retired_on=retired_on,
        lifecycle_source_urls=lifecycle_source_urls,
        lifecycle_last_checked=(
            lifecycle_last_checked if lifecycle_source_urls else None
        ),
    )


def _anthropic_spec(
    model: str,
    *,
    context_window_tokens: int,
    max_output_tokens: int,
    aliases: tuple[str, ...] = (),
    status: ModelStatus = "stable",
    replacement_model: str | None = None,
    retired_on: date | None = None,
    limits_last_checked: date = _ANTHROPIC_LIMITS_CHECKED_ON,
    lifecycle_last_checked: date = _ANTHROPIC_LIFECYCLE_CHECKED_ON,
) -> ModelSpec:
    lifecycle_sources: tuple[str, ...] = ()
    if status == "preview":
        lifecycle_sources = (_ANTHROPIC_LIMIT_URLS[0],)
    elif status == "deprecated":
        lifecycle_sources = (_ANTHROPIC_DEPRECATIONS_URL,)
    return ModelSpec(
        provider="anthropic",
        model=model,
        aliases=aliases,
        limits=ModelLimits(
            context_window_tokens=context_window_tokens,
            default_response_reserve_tokens=4_096,
            max_input_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
            source_urls=_ANTHROPIC_LIMIT_URLS,
            last_checked=limits_last_checked,
        ),
        status=status,
        replacement_model=replacement_model,
        retired_on=retired_on,
        lifecycle_source_urls=lifecycle_sources,
        lifecycle_last_checked=(lifecycle_last_checked if lifecycle_sources else None),
    )


def _gemini_spec(
    model: str,
    *,
    context_window_tokens: int,
    max_output_tokens: int | None,
    source_model: str | None = None,
    limits_source_urls: tuple[str, ...] | None = None,
    aliases: tuple[str, ...] = (),
    input_rate: str | None = None,
    cached_input_rate: str | None = None,
    output_rate: str | None = None,
    long_context: LongContextPricing | None = None,
    status: ModelStatus = "stable",
    replacement_model: str | None = None,
    lifecycle_source_urls: tuple[str, ...] = (),
    limits_last_checked: date = _GEMINI_LIMITS_CHECKED_ON,
    pricing_last_checked: date = _GEMINI_PRICING_CHECKED_ON,
    lifecycle_last_checked: date = _GEMINI_LIFECYCLE_CHECKED_ON,
) -> ModelSpec:
    source_url = (
        "https://ai.google.dev/gemini-api/docs/models/"
        f"{source_model or model}"
    )
    if (input_rate is None) != (output_rate is None):
        raise ValueError("Gemini input and output pricing must be configured together")
    pricing = (
        _pricing(
            input_rate=input_rate,
            cached_input_rate=cached_input_rate,
            output_rate=output_rate,
            source_urls=(_GEMINI_PRICING_URL,),
            last_checked=pricing_last_checked,
            long_context=long_context,
        )
        if input_rate is not None and output_rate is not None
        else None
    )
    if not lifecycle_source_urls:
        if status == "preview":
            lifecycle_source_urls = (source_url,)
        elif status == "deprecated":
            lifecycle_source_urls = (_GEMINI_DEPRECATIONS_URL,)
    return ModelSpec(
        provider="gemini",
        model=model,
        aliases=aliases,
        limits=ModelLimits(
            context_window_tokens=context_window_tokens,
            default_response_reserve_tokens=8_192,
            max_input_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
            source_urls=limits_source_urls or (source_url,),
            last_checked=limits_last_checked,
        ),
        pricing=pricing,
        status=status,
        replacement_model=replacement_model,
        lifecycle_source_urls=lifecycle_source_urls,
        lifecycle_last_checked=(
            lifecycle_last_checked if lifecycle_source_urls else None
        ),
    )


_OPENAI_MODEL_SPECS = (
    _openai_spec(
        "gpt-5.6-sol",
        aliases=("gpt-5.6",),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_rate="5.00",
        cached_input_rate="0.50",
        cache_write_input_rate="6.25",
        output_rate="30.00",
        pricing_source_urls=(_OPENAI_PRICING_URL,),
        long_context=_long_context_pricing(
            above_input_tokens=272_000,
            input_rate="10.00",
            cached_input_rate="1.00",
            cache_write_input_rate="12.50",
            output_rate="45.00",
        ),
    ),
    _openai_spec(
        "gpt-5.6-terra",
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_rate="2.50",
        cached_input_rate="0.25",
        cache_write_input_rate="3.125",
        output_rate="15.00",
        pricing_source_urls=(_OPENAI_PRICING_URL,),
        long_context=_long_context_pricing(
            above_input_tokens=272_000,
            input_rate="5.00",
            cached_input_rate="0.50",
            cache_write_input_rate="6.25",
            output_rate="22.50",
        ),
    ),
    _openai_spec(
        "gpt-5.6-luna",
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_rate="1.00",
        cached_input_rate="0.10",
        cache_write_input_rate="1.25",
        output_rate="6.00",
        pricing_source_urls=(_OPENAI_PRICING_URL,),
        long_context=_long_context_pricing(
            above_input_tokens=272_000,
            input_rate="2.00",
            cached_input_rate="0.20",
            cache_write_input_rate="2.50",
            output_rate="9.00",
        ),
    ),
    _openai_spec(
        "chat-latest",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="5.00",
        cached_input_rate="0.50",
        output_rate="30.00",
    ),
    _openai_spec(
        "gpt-5.5",
        aliases=("gpt-5.5-2026-04-23",),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_rate="5.00",
        cached_input_rate="0.50",
        output_rate="30.00",
        long_context=_long_context_pricing(
            above_input_tokens=272_000,
            input_rate="10.00",
            cached_input_rate="1.00",
            output_rate="45.00",
        ),
    ),
    _openai_spec(
        "gpt-5.5-pro",
        aliases=("gpt-5.5-pro-2026-04-23",),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_rate="30.00",
        output_rate="180.00",
        pricing_source_urls=(_OPENAI_PRICING_URL,),
        long_context=_long_context_pricing(
            above_input_tokens=272_000,
            input_rate="60.00",
            cached_input_rate=None,
            output_rate="270.00",
        ),
    ),
    _openai_spec(
        "gpt-5.4",
        aliases=("gpt-5.4-2026-03-05",),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_rate="2.50",
        cached_input_rate="0.25",
        output_rate="15.00",
        long_context=_long_context_pricing(
            above_input_tokens=272_000,
            input_rate="5.00",
            cached_input_rate="0.50",
            output_rate="22.50",
        ),
    ),
    _openai_spec(
        "gpt-5.4-pro",
        aliases=("gpt-5.4-pro-2026-03-05",),
        context_window_tokens=1_050_000,
        max_output_tokens=128_000,
        input_rate="30.00",
        output_rate="180.00",
        long_context=_long_context_pricing(
            above_input_tokens=272_000,
            input_rate="60.00",
            cached_input_rate=None,
            output_rate="270.00",
        ),
    ),
    _openai_spec(
        "gpt-5.4-mini",
        aliases=("gpt-5.4-mini-2026-03-17",),
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="0.75",
        cached_input_rate="0.075",
        output_rate="4.50",
    ),
    _openai_spec(
        "gpt-5.4-nano",
        aliases=("gpt-5.4-nano-2026-03-17",),
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="0.20",
        cached_input_rate="0.02",
        output_rate="1.25",
    ),
    _openai_spec(
        "gpt-5.3-codex",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.75",
        cached_input_rate="0.175",
        output_rate="14.00",
    ),
    _openai_spec(
        "gpt-5.2",
        aliases=("gpt-5.2-2025-12-11",),
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.75",
        cached_input_rate="0.175",
        output_rate="14.00",
    ),
    _openai_spec(
        "gpt-5.2-pro",
        aliases=("gpt-5.2-pro-2025-12-11",),
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="21.00",
        output_rate="168.00",
    ),
    _openai_spec(
        "gpt-5.1",
        aliases=("gpt-5.1-2025-11-13",),
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
    ),
    _openai_spec(
        "gpt-5",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
    ),
    _openai_spec(
        "gpt-5-mini",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="0.25",
        cached_input_rate="0.025",
        output_rate="2.00",
    ),
    _openai_spec(
        "gpt-5-nano",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="0.05",
        cached_input_rate="0.005",
        output_rate="0.40",
    ),
    _openai_spec(
        "gpt-5-pro",
        context_window_tokens=400_000,
        max_output_tokens=272_000,
        input_rate="15.00",
        output_rate="120.00",
    ),
    _openai_spec(
        "o3-pro",
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        input_rate="20.00",
        output_rate="80.00",
    ),
    _openai_spec(
        "o3",
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        input_rate="2.00",
        cached_input_rate="0.50",
        output_rate="8.00",
    ),
    _openai_spec(
        "gpt-4.1",
        aliases=("gpt-4.1-2025-04-14",),
        context_window_tokens=1_047_576,
        max_output_tokens=32_768,
        input_rate="2.00",
        cached_input_rate="0.50",
        output_rate="8.00",
    ),
    _openai_spec(
        "gpt-4.1-mini",
        aliases=("gpt-4.1-mini-2025-04-14",),
        context_window_tokens=1_047_576,
        max_output_tokens=32_768,
        input_rate="0.40",
        cached_input_rate="0.10",
        output_rate="1.60",
    ),
    _openai_spec(
        "gpt-4o",
        aliases=("gpt-4o-2024-08-06", "gpt-4o-2024-11-20"),
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        input_rate="2.50",
        cached_input_rate="1.25",
        output_rate="10.00",
        status="deprecated",
        lifecycle_source_urls=(
            "https://developers.openai.com/api/docs/models/gpt-4o",
        ),
    ),
    _openai_spec(
        "gpt-4o-mini",
        aliases=("gpt-4o-mini-2024-07-18",),
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        input_rate="0.15",
        cached_input_rate="0.075",
        output_rate="0.60",
    ),
    # These identities remain callable but retire before current recommended models.
    _openai_spec(
        "gpt-5.3-chat-latest",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        input_rate="1.75",
        cached_input_rate="0.175",
        output_rate="14.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 8, 10),
    ),
    _openai_spec(
        "gpt-5.2-chat-latest",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        input_rate="1.75",
        cached_input_rate="0.175",
        output_rate="14.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 8, 10),
    ),
    _openai_spec(
        "gpt-5.2-codex",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.75",
        cached_input_rate="0.175",
        output_rate="14.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 7, 23),
    ),
    _openai_spec(
        "gpt-5.1-chat-latest",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 7, 23),
    ),
    _openai_spec(
        "gpt-5.1-codex",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 7, 23),
    ),
    _openai_spec(
        "gpt-5.1-codex-max",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 7, 23),
    ),
    _openai_spec(
        "gpt-5.1-codex-mini",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="0.25",
        cached_input_rate="0.025",
        output_rate="2.00",
        status="deprecated",
        replacement_model="gpt-5.4-mini",
        retired_on=date(2026, 7, 23),
    ),
    _openai_spec(
        "gpt-5-chat-latest",
        context_window_tokens=128_000,
        max_output_tokens=16_384,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 7, 23),
    ),
    _openai_spec(
        "gpt-5-codex",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 7, 23),
    ),
    _openai_spec(
        "computer-use-preview",
        aliases=("computer-use-preview-2025-03-11",),
        context_window_tokens=8_192,
        max_output_tokens=1_024,
        default_response_reserve_tokens=1_024,
        input_rate="3.00",
        output_rate="12.00",
        status="deprecated",
        replacement_model="gpt-5.4-mini",
        retired_on=date(2026, 7, 23),
    ),
    _openai_spec(
        "gpt-4-turbo",
        aliases=("gpt-4-turbo-2024-04-09",),
        context_window_tokens=128_000,
        max_output_tokens=4_096,
        input_rate="10.00",
        output_rate="30.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 10, 23),
    ),
    _openai_spec(
        "gpt-4o-2024-05-13",
        source_model="gpt-4o",
        limits_source_urls=(
            "https://learn.microsoft.com/en-us/azure/ai-foundry/"
            "foundry-models/concepts/models"
            "?view=azure-node-latest",
            "https://developers.openai.com/api/docs/models/gpt-4o",
        ),
        context_window_tokens=128_000,
        max_output_tokens=4_096,
        input_rate="5.00",
        output_rate="15.00",
        pricing_source_urls=(
            "https://openai.com/index/introducing-structured-outputs-in-the-api/",
        ),
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 10, 23),
    ),
    _openai_spec(
        "gpt-5-2025-08-07",
        source_model="gpt-5",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 12, 11),
    ),
    _openai_spec(
        "gpt-5-mini-2025-08-07",
        source_model="gpt-5-mini",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="0.25",
        cached_input_rate="0.025",
        output_rate="2.00",
        status="deprecated",
        replacement_model="gpt-5.4-mini",
        retired_on=date(2026, 12, 11),
    ),
    _openai_spec(
        "gpt-5-nano-2025-08-07",
        source_model="gpt-5-nano",
        context_window_tokens=400_000,
        max_output_tokens=128_000,
        input_rate="0.05",
        cached_input_rate="0.005",
        output_rate="0.40",
        status="deprecated",
        replacement_model="gpt-5.4-nano",
        retired_on=date(2026, 12, 11),
    ),
    _openai_spec(
        "gpt-5-pro-2025-10-06",
        source_model="gpt-5-pro",
        context_window_tokens=400_000,
        max_output_tokens=272_000,
        input_rate="15.00",
        output_rate="120.00",
        status="deprecated",
        replacement_model="gpt-5.5-pro",
        retired_on=date(2026, 12, 11),
    ),
    _openai_spec(
        "o3-2025-04-16",
        source_model="o3",
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        input_rate="2.00",
        cached_input_rate="0.50",
        output_rate="8.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 12, 11),
    ),
    _openai_spec(
        "o3-pro-2025-06-10",
        source_model="o3-pro",
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        input_rate="20.00",
        output_rate="80.00",
        status="deprecated",
        replacement_model="gpt-5.5-pro",
        retired_on=date(2026, 12, 11),
    ),
    _openai_spec(
        "gpt-4.1-nano",
        aliases=("gpt-4.1-nano-2025-04-14",),
        context_window_tokens=1_047_576,
        max_output_tokens=32_768,
        input_rate="0.10",
        cached_input_rate="0.025",
        output_rate="0.40",
        status="deprecated",
        replacement_model="gpt-5.4-nano",
        retired_on=date(2026, 10, 23),
    ),
    _openai_spec(
        "o4-mini",
        aliases=("o4-mini-2025-04-16",),
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        input_rate="1.10",
        cached_input_rate="0.275",
        output_rate="4.40",
        status="deprecated",
        replacement_model="gpt-5.4-mini",
        retired_on=date(2026, 10, 23),
    ),
    _openai_spec(
        "o3-mini",
        aliases=("o3-mini-2025-01-31",),
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        input_rate="1.10",
        cached_input_rate="0.55",
        output_rate="4.40",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 10, 23),
    ),
    _openai_spec(
        "o1",
        aliases=("o1-2024-12-17",),
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        input_rate="15.00",
        cached_input_rate="7.50",
        output_rate="60.00",
        status="deprecated",
        replacement_model="gpt-5.5",
        retired_on=date(2026, 10, 23),
    ),
    _openai_spec(
        "o1-pro",
        aliases=("o1-pro-2025-03-19",),
        context_window_tokens=200_000,
        max_output_tokens=100_000,
        input_rate="150.00",
        output_rate="600.00",
        status="deprecated",
        replacement_model="gpt-5.5-pro",
        retired_on=date(2026, 10, 23),
    ),
)

_ANTHROPIC_MODEL_SPECS = (
    _anthropic_spec(
        "claude-fable-5",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    _anthropic_spec(
        "claude-mythos-5",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    _anthropic_spec(
        "claude-opus-4-8",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    _anthropic_spec(
        "claude-opus-4-7",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    _anthropic_spec(
        "claude-opus-4-6",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    _anthropic_spec(
        "claude-opus-4-5-20251101",
        aliases=("claude-opus-4-5",),
        context_window_tokens=200_000,
        max_output_tokens=64_000,
    ),
    _anthropic_spec(
        "claude-opus-4-1-20250805",
        aliases=("claude-opus-4-1",),
        context_window_tokens=200_000,
        max_output_tokens=32_000,
        status="deprecated",
        replacement_model="claude-opus-4-8",
        retired_on=date(2026, 8, 5),
    ),
    _anthropic_spec(
        "claude-sonnet-5",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    _anthropic_spec(
        "claude-sonnet-4-6",
        context_window_tokens=1_000_000,
        max_output_tokens=128_000,
    ),
    _anthropic_spec(
        "claude-sonnet-4-5-20250929",
        aliases=("claude-sonnet-4-5",),
        context_window_tokens=200_000,
        max_output_tokens=64_000,
    ),
    _anthropic_spec(
        "claude-haiku-4-5-20251001",
        aliases=("claude-haiku-4-5",),
        context_window_tokens=200_000,
        max_output_tokens=64_000,
    ),
)

_GEMINI_PRO_LONG_CONTEXT = _long_context_pricing(
    above_input_tokens=200_000,
    input_rate="4.00",
    cached_input_rate="0.40",
    output_rate="18.00",
)
_GEMINI_MODEL_SPECS = (
    _gemini_spec(
        "gemini-3.5-flash",
        aliases=("gemini-flash-latest",),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        input_rate="1.50",
        cached_input_rate="0.15",
        output_rate="9.00",
        lifecycle_source_urls=(_GEMINI_CHANGELOG_URL,),
    ),
    _gemini_spec(
        "gemini-3.1-flash-lite",
        aliases=("gemini-flash-lite-latest",),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        input_rate="0.25",
        cached_input_rate="0.025",
        output_rate="1.50",
        lifecycle_source_urls=(
            _GEMINI_FLASH_LITE_ALIAS_URL,
            _GEMINI_MODELS_URL,
        ),
    ),
    _gemini_spec(
        "gemini-3.1-pro-preview",
        aliases=("gemini-3-pro-preview", "gemini-pro-latest"),
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        input_rate="2.00",
        cached_input_rate="0.20",
        output_rate="12.00",
        long_context=_GEMINI_PRO_LONG_CONTEXT,
        status="preview",
        lifecycle_source_urls=(_GEMINI_CHANGELOG_URL,),
    ),
    _gemini_spec(
        "gemini-3.1-pro-preview-customtools",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        source_model="gemini-3.1-pro-preview",
        input_rate="2.00",
        cached_input_rate="0.20",
        output_rate="12.00",
        long_context=_GEMINI_PRO_LONG_CONTEXT,
        status="preview",
    ),
    _gemini_spec(
        "gemini-3-flash-preview",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        input_rate="0.50",
        cached_input_rate="0.05",
        output_rate="3.00",
        status="preview",
    ),
    _gemini_spec(
        "gemini-2.5-pro",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        input_rate="1.25",
        cached_input_rate="0.125",
        output_rate="10.00",
        long_context=_long_context_pricing(
            above_input_tokens=200_000,
            input_rate="2.50",
            cached_input_rate="0.25",
            output_rate="15.00",
        ),
        status="deprecated",
        replacement_model="gemini-3.1-pro-preview",
    ),
    _gemini_spec(
        "gemini-2.5-flash",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        input_rate="0.30",
        cached_input_rate="0.03",
        output_rate="2.50",
        status="deprecated",
        replacement_model="gemini-3.5-flash",
    ),
    _gemini_spec(
        "gemini-2.5-flash-lite",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        input_rate="0.10",
        cached_input_rate="0.01",
        output_rate="0.40",
        status="deprecated",
        replacement_model="gemini-3.1-flash-lite",
    ),
    _gemini_spec(
        "gemini-2.5-computer-use-preview-10-2025",
        context_window_tokens=128_000,
        max_output_tokens=64_000,
        input_rate="1.25",
        output_rate="10.00",
        status="preview",
    ),
    _gemini_spec(
        "gemini-robotics-er-1.6-preview",
        context_window_tokens=131_072,
        max_output_tokens=65_536,
        input_rate="1.00",
        output_rate="5.00",
        status="preview",
    ),
    _gemini_spec(
        "gemma-4-26b-a4b-it",
        context_window_tokens=262_144,
        max_output_tokens=None,
        limits_source_urls=_GEMMA_4_API_LIMIT_URLS,
        input_rate="0",
        cached_input_rate="0",
        output_rate="0",
    ),
    _gemini_spec(
        "gemma-4-31b-it",
        context_window_tokens=262_144,
        max_output_tokens=None,
        limits_source_urls=_GEMMA_4_API_LIMIT_URLS,
        input_rate="0",
        cached_input_rate="0",
        output_rate="0",
    ),
)

_DEEPSEEK_V4_LIMITS = ModelLimits(
    context_window_tokens=1_048_576,
    default_response_reserve_tokens=16_384,
    max_output_tokens=393_216,
    source_urls=_DEEPSEEK_V4_LIMIT_URLS,
    last_checked=_DEEPSEEK_LIMITS_CHECKED_ON,
)
_DEEPSEEK_FLASH_PRICING = _pricing(
    input_rate="0.14",
    cached_input_rate="0.0028",
    output_rate="0.28",
    source_urls=(_DEEPSEEK_V4_URL,),
    last_checked=_DEEPSEEK_PRICING_CHECKED_ON,
)
_DEEPSEEK_PRO_PRICING = _pricing(
    input_rate="0.435",
    cached_input_rate="0.003625",
    output_rate="0.87",
    source_urls=(_DEEPSEEK_V4_URL,),
    last_checked=_DEEPSEEK_PRICING_CHECKED_ON,
)
_DEEPSEEK_MODEL_SPECS = (
    ModelSpec(
        provider="deepseek",
        model="deepseek-v4-flash",
        limits=_DEEPSEEK_V4_LIMITS,
        pricing=_DEEPSEEK_FLASH_PRICING,
        status="preview",
        lifecycle_source_urls=(_DEEPSEEK_V4_ANNOUNCEMENT_URL,),
        lifecycle_last_checked=_DEEPSEEK_LIFECYCLE_CHECKED_ON,
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-v4-pro",
        limits=_DEEPSEEK_V4_LIMITS,
        pricing=_DEEPSEEK_PRO_PRICING,
        status="preview",
        lifecycle_source_urls=(_DEEPSEEK_V4_ANNOUNCEMENT_URL,),
        lifecycle_last_checked=_DEEPSEEK_LIFECYCLE_CHECKED_ON,
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-chat",
        limits=_DEEPSEEK_V4_LIMITS,
        pricing=_DEEPSEEK_FLASH_PRICING,
        status="deprecated",
        replacement_model="deepseek-v4-flash",
        retired_on=date(2026, 7, 24),
        lifecycle_source_urls=(_DEEPSEEK_V4_URL,),
        lifecycle_last_checked=_DEEPSEEK_LIFECYCLE_CHECKED_ON,
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-reasoner",
        limits=_DEEPSEEK_V4_LIMITS,
        pricing=_DEEPSEEK_FLASH_PRICING,
        status="deprecated",
        replacement_model="deepseek-v4-flash",
        retired_on=date(2026, 7, 24),
        lifecycle_source_urls=(_DEEPSEEK_V4_URL,),
        lifecycle_last_checked=_DEEPSEEK_LIFECYCLE_CHECKED_ON,
    ),
)

_LOCAL_MODEL_SPECS = (
    ModelSpec(
        provider="local",
        model="google/gemma-4-12b-qat",
        limits=ModelLimits(
            context_window_tokens=262_144,
            default_response_reserve_tokens=4_096,
            source_urls=_LM_STUDIO_GEMMA_4_12B_URLS,
            last_checked=_LOCAL_LIMITS_CHECKED_ON,
        ),
    ),
    ModelSpec(
        provider="local",
        model="gemma4:12b",
        limits=ModelLimits(
            context_window_tokens=262_144,
            default_response_reserve_tokens=4_096,
            source_urls=(_OLLAMA_GEMMA_4_12B_URL,),
            last_checked=_LOCAL_LIMITS_CHECKED_ON,
        ),
    ),
    ModelSpec(
        provider="local",
        model="gemma4:12b-it-qat",
        limits=ModelLimits(
            context_window_tokens=262_144,
            default_response_reserve_tokens=4_096,
            source_urls=(_OLLAMA_GEMMA_4_12B_QAT_URL,),
            last_checked=_LOCAL_LIMITS_CHECKED_ON,
        ),
    ),
    ModelSpec(
        provider="local",
        model="gemma3:12b",
        limits=ModelLimits(
            context_window_tokens=131_072,
            default_response_reserve_tokens=4_096,
            source_urls=(_OLLAMA_GEMMA_3_12B_URL,),
            last_checked=_LOCAL_LIMITS_CHECKED_ON,
        ),
    ),
    ModelSpec(
        provider="local",
        model="qwen/qwen3.5-35b-a3b",
        limits=ModelLimits(
            context_window_tokens=262_144,
            default_response_reserve_tokens=4_096,
            source_urls=(_LM_STUDIO_QWEN_3_5_URL,),
            last_checked=_LOCAL_LIMITS_CHECKED_ON,
        ),
    ),
)

MODEL_SPECS = (
    *_OPENAI_MODEL_SPECS,
    *_ANTHROPIC_MODEL_SPECS,
    *_GEMINI_MODEL_SPECS,
    *_DEEPSEEK_MODEL_SPECS,
    *_LOCAL_MODEL_SPECS,
)

MODEL_CATALOG = ModelCatalog(MODEL_SPECS)


def resolve_model_spec(provider: str | None, model: str | None) -> ModelSpec | None:
    """Resolve bundled metadata without allocating a new model record."""
    return MODEL_CATALOG.resolve(provider, model)


__all__ = [
    "MODEL_CATALOG",
    "MODEL_SPECS",
    "LongContextPricing",
    "ModelCatalog",
    "ModelLimits",
    "ModelSpec",
    "ModelStatus",
    "TokenPricing",
    "resolve_model_spec",
]
