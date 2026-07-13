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
class TokenPricing:
    """Current per-million-token prices for one model."""

    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None
    currency: str = USD
    source_urls: tuple[str, ...] = ()
    last_checked: date | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
            ("cached_input_per_million", self.cached_input_per_million),
        ):
            if value is None:
                continue
            if not isinstance(value, Decimal):
                raise TypeError(f"{name} must be a Decimal")
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite non-negative Decimal")
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


_OPENAI_GPT_4_1_URL = "https://developers.openai.com/api/docs/models/gpt-4.1"
_OPENAI_GPT_4_1_MINI_URL = (
    "https://developers.openai.com/api/docs/models/gpt-4.1-mini"
)
_OPENAI_GPT_4_1_NANO_URL = (
    "https://developers.openai.com/api/docs/models/gpt-4.1-nano"
)
_DEEPSEEK_V4_URL = "https://api-docs.deepseek.com/quick_start/pricing/"
_GEMMA_4_URLS = (
    "https://ai.google.dev/gemma/docs/core/model_card_4",
    "https://www.ollama.com/library/gemma4",
)
_GEMMA_3_URL = "https://ai.google.dev/gemma/docs/core/model_card_3"
_QWEN_3_5_URL = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B"

_OPENAI_LIMITS_CHECKED_ON = date(2026, 7, 13)
_OPENAI_PRICING_CHECKED_ON = date(2026, 7, 13)
_DEEPSEEK_LIMITS_CHECKED_ON = date(2026, 7, 13)
_DEEPSEEK_PRICING_CHECKED_ON = date(2026, 7, 13)
_LOCAL_LIMITS_CHECKED_ON = date(2026, 7, 13)
_LIFECYCLE_CHECKED_ON = date(2026, 7, 13)

_OPENAI_GPT_4_1_LIMITS = ModelLimits(
    context_window_tokens=1_047_576,
    default_response_reserve_tokens=8_192,
    max_output_tokens=32_768,
    source_urls=(_OPENAI_GPT_4_1_URL,),
    last_checked=_OPENAI_LIMITS_CHECKED_ON,
)
_OPENAI_GPT_4_1_MINI_LIMITS = ModelLimits(
    context_window_tokens=1_047_576,
    default_response_reserve_tokens=8_192,
    max_output_tokens=32_768,
    source_urls=(_OPENAI_GPT_4_1_MINI_URL,),
    last_checked=_OPENAI_LIMITS_CHECKED_ON,
)
_OPENAI_GPT_4_1_NANO_LIMITS = ModelLimits(
    context_window_tokens=1_047_576,
    default_response_reserve_tokens=8_192,
    max_output_tokens=32_768,
    source_urls=(_OPENAI_GPT_4_1_NANO_URL,),
    last_checked=_OPENAI_LIMITS_CHECKED_ON,
)
_DEEPSEEK_V4_LIMITS = ModelLimits(
    context_window_tokens=1_000_000,
    default_response_reserve_tokens=16_384,
    max_output_tokens=384_000,
    source_urls=(_DEEPSEEK_V4_URL,),
    last_checked=_DEEPSEEK_LIMITS_CHECKED_ON,
)
_GEMMA_4_LIMITS = ModelLimits(
    context_window_tokens=262_144,
    default_response_reserve_tokens=4_096,
    source_urls=_GEMMA_4_URLS,
    last_checked=_LOCAL_LIMITS_CHECKED_ON,
)
_GEMMA_3_LIMITS = ModelLimits(
    context_window_tokens=131_072,
    default_response_reserve_tokens=4_096,
    source_urls=(_GEMMA_3_URL,),
    last_checked=_LOCAL_LIMITS_CHECKED_ON,
)
_QWEN_3_5_LIMITS = ModelLimits(
    context_window_tokens=262_144,
    default_response_reserve_tokens=4_096,
    source_urls=(_QWEN_3_5_URL,),
    last_checked=_LOCAL_LIMITS_CHECKED_ON,
)

_DEEPSEEK_FLASH_PRICING = TokenPricing(
    input_per_million=Decimal("0.14"),
    cached_input_per_million=Decimal("0.0028"),
    output_per_million=Decimal("0.28"),
    source_urls=(_DEEPSEEK_V4_URL,),
    last_checked=_DEEPSEEK_PRICING_CHECKED_ON,
)
_DEEPSEEK_PRO_PRICING = TokenPricing(
    input_per_million=Decimal("0.435"),
    cached_input_per_million=Decimal("0.003625"),
    output_per_million=Decimal("0.87"),
    source_urls=(_DEEPSEEK_V4_URL,),
    last_checked=_DEEPSEEK_PRICING_CHECKED_ON,
)


MODEL_SPECS = (
    ModelSpec(
        provider="openai",
        model="gpt-4.1",
        aliases=("gpt-4.1-2025-04-14",),
        limits=_OPENAI_GPT_4_1_LIMITS,
        pricing=TokenPricing(
            input_per_million=Decimal("2.00"),
            cached_input_per_million=Decimal("0.50"),
            output_per_million=Decimal("8.00"),
            source_urls=(_OPENAI_GPT_4_1_URL,),
            last_checked=_OPENAI_PRICING_CHECKED_ON,
        ),
    ),
    ModelSpec(
        provider="openai",
        model="gpt-4.1-mini",
        aliases=("gpt-4.1-mini-2025-04-14",),
        limits=_OPENAI_GPT_4_1_MINI_LIMITS,
        pricing=TokenPricing(
            input_per_million=Decimal("0.40"),
            cached_input_per_million=Decimal("0.10"),
            output_per_million=Decimal("1.60"),
            source_urls=(_OPENAI_GPT_4_1_MINI_URL,),
            last_checked=_OPENAI_PRICING_CHECKED_ON,
        ),
    ),
    ModelSpec(
        provider="openai",
        model="gpt-4.1-nano",
        aliases=("gpt-4.1-nano-2025-04-14",),
        limits=_OPENAI_GPT_4_1_NANO_LIMITS,
        pricing=TokenPricing(
            input_per_million=Decimal("0.10"),
            cached_input_per_million=Decimal("0.025"),
            output_per_million=Decimal("0.40"),
            source_urls=(_OPENAI_GPT_4_1_NANO_URL,),
            last_checked=_OPENAI_PRICING_CHECKED_ON,
        ),
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-v4-flash",
        limits=_DEEPSEEK_V4_LIMITS,
        pricing=_DEEPSEEK_FLASH_PRICING,
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-v4-pro",
        limits=_DEEPSEEK_V4_LIMITS,
        pricing=_DEEPSEEK_PRO_PRICING,
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
        lifecycle_last_checked=_LIFECYCLE_CHECKED_ON,
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
        lifecycle_last_checked=_LIFECYCLE_CHECKED_ON,
    ),
    ModelSpec(
        provider="local",
        model="google/gemma-4-12b-qat",
        aliases=("gemma4:12b",),
        limits=_GEMMA_4_LIMITS,
    ),
    ModelSpec(
        provider="local",
        model="gemma3:12b",
        limits=_GEMMA_3_LIMITS,
    ),
    ModelSpec(
        provider="local",
        model="qwen/qwen3.5-35b-a3b",
        limits=_QWEN_3_5_LIMITS,
    ),
)

MODEL_CATALOG = ModelCatalog(MODEL_SPECS)


def resolve_model_spec(provider: str | None, model: str | None) -> ModelSpec | None:
    """Resolve bundled metadata without allocating a new model record."""
    return MODEL_CATALOG.resolve(provider, model)


__all__ = [
    "MODEL_CATALOG",
    "MODEL_SPECS",
    "ModelCatalog",
    "ModelLimits",
    "ModelSpec",
    "ModelStatus",
    "TokenPricing",
    "resolve_model_spec",
]
