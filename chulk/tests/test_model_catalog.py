from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal

import pytest

from chulk.llm.capabilities import (
    LLMModelCapabilities,
    conservative_model_capabilities,
    resolve_model_capabilities,
)
from chulk.llm.model_catalog import (
    MODEL_CATALOG,
    MODEL_SPECS,
    ModelCatalog,
    ModelLimits,
    ModelSpec,
    TokenPricing,
    resolve_model_spec,
)
from chulk.llm.pricing import estimate_cost, resolve_pricing
from chulk.llm.usage import LLMUsage


LIMIT_SOURCE_URL = "https://models.example/limits"
PRICING_SOURCE_URL = "https://models.example/pricing"
LIMITS_CHECKED_ON = date(2026, 7, 12)
PRICING_CHECKED_ON = date(2026, 7, 13)


def _spec(
    model: str,
    *,
    aliases: tuple[str, ...] = (),
) -> ModelSpec:
    return ModelSpec(
        provider="test",
        model=model,
        aliases=aliases,
        limits=ModelLimits(
            context_window_tokens=100_000,
            default_response_reserve_tokens=1_000,
            max_input_tokens=90_000,
            max_output_tokens=10_000,
            source_urls=(LIMIT_SOURCE_URL,),
            last_checked=LIMITS_CHECKED_ON,
        ),
        pricing=TokenPricing(
            input_per_million=Decimal("1.25"),
            cached_input_per_million=Decimal("0.25"),
            output_per_million=Decimal("5.00"),
            source_urls=(PRICING_SOURCE_URL,),
            last_checked=PRICING_CHECKED_ON,
        ),
    )


def test_catalog_returns_one_shared_record_for_exact_and_explicit_alias() -> None:
    canonical = resolve_model_spec("openai", "gpt-4.1-mini")
    alias = resolve_model_spec(" OPENAI ", " GPT-4.1-MINI-2025-04-14 ")

    assert canonical is not None
    assert alias is canonical
    assert resolve_pricing("openai", "gpt-4.1-mini") is canonical.pricing


def test_catalog_rejects_unlisted_snapshots_and_sibling_names() -> None:
    assert resolve_model_spec("openai", "gpt-4.1-mini-2099-01-01") is None
    assert resolve_model_spec("openai", "gpt-4.1-new-model") is None
    assert resolve_pricing("openai", "gpt-4.1-new-model") is None


def test_new_model_is_appended_as_one_declarative_record() -> None:
    new_spec = _spec("future-model", aliases=("future-model-latest",))
    catalog = ModelCatalog((*MODEL_SPECS, new_spec))

    assert catalog.resolve("test", "future-model") is new_spec
    assert catalog.resolve("test", "future-model-latest") is new_spec
    assert len(catalog) == len(MODEL_CATALOG) + 1


def test_catalog_rejects_duplicate_identities() -> None:
    with pytest.raises(ValueError, match="Duplicate model identity"):
        ModelCatalog((_spec("first", aliases=("shared",)), _spec("shared")))


def test_catalog_records_are_immutable_and_validate_values() -> None:
    spec = _spec("immutable")

    with pytest.raises(FrozenInstanceError):
        spec.model = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="max_output_tokens"):
        ModelLimits(
            context_window_tokens=100,
            default_response_reserve_tokens=10,
            max_output_tokens=101,
        )
    with pytest.raises(TypeError, match="Decimal"):
        TokenPricing(
            input_per_million=1.0,  # type: ignore[arg-type]
            output_per_million=Decimal("2.0"),
        )
    with pytest.raises(ValueError, match="source_urls and last_checked"):
        ModelLimits(
            context_window_tokens=100,
            default_response_reserve_tokens=10,
        )
    with pytest.raises(TypeError, match="limits must be"):
        ModelSpec(
            provider="test",
            model="malformed",
            limits={"context_window_tokens": 100},  # type: ignore[arg-type]
        )


def test_catalog_rejects_unknown_replacement_models() -> None:
    deprecated = ModelSpec(
        provider="test",
        model="old-model",
        limits=_spec("source").limits,
        status="deprecated",
        replacement_model="typo-model",
        lifecycle_source_urls=("https://models.example/lifecycle",),
        lifecycle_last_checked=PRICING_CHECKED_ON,
    )

    with pytest.raises(ValueError, match="replacement_model"):
        ModelCatalog((deprecated,))
    with pytest.raises(ValueError, match="same model"):
        ModelSpec(
            provider="test",
            model="self-replacement",
            limits=_spec("source").limits,
            status="deprecated",
            replacement_model="self-replacement",
            lifecycle_source_urls=("https://models.example/lifecycle",),
            lifecycle_last_checked=PRICING_CHECKED_ON,
        )


def test_published_limits_and_section_provenance_are_available() -> None:
    openai = resolve_model_spec("openai", "gpt-4.1-mini")
    deepseek = resolve_model_spec("deepseek", "deepseek-v4-pro")
    gemma = resolve_model_spec("local", "gemma4:12b")

    assert openai is not None
    assert openai.limits.context_window_tokens == 1_047_576
    assert openai.limits.max_input_tokens is None
    assert openai.limits.max_output_tokens == 32_768
    assert openai.limits.last_checked == date(2026, 7, 13)
    assert openai.limits.source_urls == (
        "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
    )
    assert openai.pricing is not None
    assert openai.pricing.last_checked == date(2026, 7, 13)
    assert openai.pricing.source_urls == openai.limits.source_urls

    assert deepseek is not None
    assert deepseek.limits.context_window_tokens == 1_000_000
    assert deepseek.limits.max_input_tokens is None
    assert deepseek.limits.max_output_tokens == 384_000

    assert gemma is not None
    assert gemma.model == "google/gemma-4-12b-qat"
    assert gemma.limits.context_window_tokens == 262_144
    assert gemma.limits.max_output_tokens is None
    assert len(gemma.limits.source_urls) == 2


def test_limit_and_pricing_provenance_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec("separate-provenance")
    monkeypatch.setattr(
        "chulk.llm.pricing.resolve_model_spec",
        lambda _provider, _model: spec,
    )

    cost = estimate_cost(
        "test",
        "separate-provenance",
        LLMUsage(input_tokens=1_000, output_tokens=100),
    )

    assert spec.limits.source_urls == (LIMIT_SOURCE_URL,)
    assert spec.limits.last_checked == LIMITS_CHECKED_ON
    assert spec.pricing is not None
    assert spec.pricing.source_urls == (PRICING_SOURCE_URL,)
    assert spec.pricing.last_checked == PRICING_CHECKED_ON
    assert cost is not None
    assert cost.pricing_source == PRICING_SOURCE_URL
    assert cost.pricing_last_checked == PRICING_CHECKED_ON.isoformat()


def test_deepseek_legacy_names_have_lifecycle_and_replacement_metadata() -> None:
    flash = resolve_model_spec("deepseek", "deepseek-v4-flash")
    chat = resolve_model_spec("deepseek", "deepseek-chat")
    reasoner = resolve_model_spec("deepseek", "deepseek-reasoner")

    assert flash is not None
    assert chat is not None
    assert reasoner is not None
    assert chat.status == reasoner.status == "deprecated"
    assert chat.replacement_model == reasoner.replacement_model == "deepseek-v4-flash"
    assert chat.retired_on == reasoner.retired_on == date(2026, 7, 24)
    assert chat.lifecycle_source_urls == (
        "https://api-docs.deepseek.com/quick_start/pricing/",
    )
    assert chat.lifecycle_last_checked == date(2026, 7, 13)
    assert chat.limits is flash.limits
    assert chat.pricing is flash.pricing
    assert reasoner.pricing is flash.pricing


def test_capability_adapter_exposes_hard_limits_without_breaking_old_constructor() -> None:
    capabilities = resolve_model_capabilities("openai", "gpt-4.1-mini")
    custom = LLMModelCapabilities(
        provider="custom",
        model="custom-model",
        context_window_tokens=8_192,
        default_response_reserve_tokens=1_024,
    )

    assert capabilities.max_input_tokens is None
    assert capabilities.max_output_tokens == 32_768
    assert capabilities.to_dict()["max_output_tokens"] == 32_768
    assert custom.max_input_tokens is None
    assert custom.max_output_tokens is None


def test_capability_input_limit_bounds_budget_and_rejects_invalid_limits() -> None:
    capabilities = LLMModelCapabilities(
        provider="custom",
        model="input-bound",
        context_window_tokens=100_000,
        default_response_reserve_tokens=8_000,
        max_input_tokens=64_000,
        max_output_tokens=16_000,
    )

    assert capabilities.input_budget_tokens == 64_000
    with pytest.raises(ValueError, match="positive integer"):
        LLMModelCapabilities(
            provider="custom",
            model="invalid",
            context_window_tokens=100_000,
            default_response_reserve_tokens=8_000,
            max_output_tokens=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        LLMModelCapabilities(
            provider="custom",
            model="invalid",
            context_window_tokens=100_000,
            default_response_reserve_tokens=8_000,
            max_output_tokens=100_001,
        )


def test_conservative_capabilities_only_claim_limits_known_by_every_model() -> None:
    openai = resolve_model_capabilities("openai", "gpt-4.1-mini")
    deepseek = resolve_model_capabilities("deepseek", "deepseek-v4-flash")
    unknown_output = LLMModelCapabilities(
        provider="custom",
        model="unknown-output",
        context_window_tokens=64_000,
        default_response_reserve_tokens=2_000,
    )

    known = conservative_model_capabilities((openai, deepseek))
    partially_unknown = conservative_model_capabilities((openai, unknown_output))

    assert known.max_output_tokens == 32_768
    assert partially_unknown.max_output_tokens is None


def test_cost_estimation_uses_normalized_cache_buckets_for_every_provider() -> None:
    usage = LLMUsage(
        input_tokens=1_000,
        output_tokens=100,
        cached_input_tokens=200,
    )

    cost = estimate_cost("openai", "gpt-4.1-mini", usage)

    assert cost is not None
    assert cost.amount == Decimal("0.000500")
    assert cost.input_cost == Decimal("0.00032")
    assert cost.cached_input_cost == Decimal("0.00002")
    assert cost.output_cost == Decimal("0.00016")
    assert cost.estimated is True
    assert cost.pricing_source == (
        "https://developers.openai.com/api/docs/models/gpt-4.1-mini"
    )
    assert cost.pricing_last_checked == "2026-07-13"


def test_cost_uses_cache_buckets_when_aggregate_input_is_missing() -> None:
    usage = LLMUsage(
        input_tokens=0,
        output_tokens=100,
        cache_hit_input_tokens=200,
        cache_miss_input_tokens=800,
    )

    cost = estimate_cost("openai", "gpt-4.1-mini", usage)

    assert cost is not None
    assert cost.input_cost == Decimal("0.00032")
    assert cost.cached_input_cost == Decimal("0.00002")
    assert cost.amount == Decimal("0.000500")
    assert cost.estimated is True


def test_cost_marks_conflicting_cached_token_fields_as_estimated() -> None:
    usage = LLMUsage(
        input_tokens=1_000,
        output_tokens=100,
        cached_input_tokens=300,
        cache_hit_input_tokens=200,
        cache_miss_input_tokens=800,
    )

    cost = estimate_cost("openai", "gpt-4.1-mini", usage)

    assert cost is not None
    assert cost.cached_input_cost == Decimal("0.00002")
    assert cost.input_cost == Decimal("0.00032")
    assert cost.estimated is True


def test_unknown_pricing_and_missing_usage_keep_existing_semantics() -> None:
    usage = LLMUsage(input_tokens=10, output_tokens=5)

    unknown = estimate_cost("local", "unpriced-model", usage)

    assert estimate_cost("local", "unpriced-model", None) is None
    assert unknown is not None
    assert unknown.amount is None
    assert unknown.pricing_known is False
