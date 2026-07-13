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
    LongContextPricing,
    ModelCatalog,
    ModelLimits,
    ModelSpec,
    TokenPricing,
    resolve_model_spec,
)
from chulk.llm.pricing import estimate_cost, resolve_pricing
from chulk.llm.usage import (
    LLMCost,
    LLMUsage,
    aggregate_cost,
    aggregate_usage,
    normalize_chat_completions_usage,
)


LIMIT_SOURCE_URL = "https://models.example/limits"
PRICING_SOURCE_URL = "https://models.example/pricing"
LIMITS_CHECKED_ON = date(2026, 7, 12)
PRICING_CHECKED_ON = date(2026, 7, 13)

EXPECTED_CANONICAL_MODELS = {
    "openai": {
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "chat-latest",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.3-codex",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "o3-pro",
        "o3",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5.3-chat-latest",
        "gpt-5.2-chat-latest",
        "gpt-5.2-codex",
        "gpt-5.1-chat-latest",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5-chat-latest",
        "gpt-5-codex",
        "computer-use-preview",
        "gpt-4-turbo",
        "gpt-4o-2024-05-13",
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-5-pro-2025-10-06",
        "o3-2025-04-16",
        "o3-pro-2025-06-10",
        "gpt-4.1-nano",
        "o4-mini",
        "o3-mini",
        "o1",
        "o1-pro",
    },
    "anthropic": {
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-opus-4-1-20250805",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    },
    "gemini": {
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-computer-use-preview-10-2025",
        "gemini-robotics-er-1.6-preview",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    },
    "deepseek": {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-chat",
        "deepseek-reasoner",
    },
    "local": {
        "google/gemma-4-12b-qat",
        "gemma3:12b",
        "qwen/qwen3.5-35b-a3b",
    },
}


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


def test_catalog_contains_every_verified_direct_provider_model() -> None:
    actual = {
        provider: {spec.model for spec in MODEL_CATALOG if spec.provider == provider}
        for provider in EXPECTED_CANONICAL_MODELS
    }

    assert actual == EXPECTED_CANONICAL_MODELS
    assert len(MODEL_CATALOG) == 77
    assert all(spec.limits.source_urls for spec in MODEL_CATALOG)
    assert all(spec.limits.last_checked == date(2026, 7, 13) for spec in MODEL_CATALOG)


@pytest.mark.parametrize(
    ("provider", "alias", "canonical"),
    (
        ("openai", "gpt-5.6", "gpt-5.6-sol"),
        ("openai", "gpt-5.4-mini-2026-03-17", "gpt-5.4-mini"),
        ("openai", "gpt-4o-2024-11-20", "gpt-4o"),
        ("openai", "gpt-4-turbo-2024-04-09", "gpt-4-turbo"),
        ("anthropic", "claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
        ("gemini", "gemini-flash-latest", "gemini-3.5-flash"),
        ("gemini", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"),
        ("gemini", "gemini-pro-latest", "gemini-3.1-pro-preview"),
        ("gemini", "gemini-3-pro-preview", "gemini-3.1-pro-preview"),
    ),
)
def test_published_aliases_resolve_to_their_canonical_records(
    provider: str,
    alias: str,
    canonical: str,
) -> None:
    resolved = resolve_model_spec(provider, alias)

    assert resolved is resolve_model_spec(provider, canonical)


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
    with pytest.raises(ValueError, match="positive integer"):
        LongContextPricing(
            applies_above_input_tokens=0,
            input_per_million=Decimal("2.0"),
            output_per_million=Decimal("4.0"),
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


def test_new_cache_fields_preserve_existing_positional_constructors() -> None:
    pricing = TokenPricing(
        Decimal("1"),
        Decimal("2"),
        Decimal("0.5"),
        "EUR",
        (PRICING_SOURCE_URL,),
        PRICING_CHECKED_ON,
        None,
    )
    usage = LLMUsage(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        True,
        True,
        "legacy",
        {"old": True},
    )
    cost = LLMCost(
        Decimal("1"),
        "EUR",
        True,
        True,
        Decimal("0.2"),
        Decimal("0.1"),
        Decimal("0.7"),
        "test",
        "legacy",
        PRICING_SOURCE_URL,
        "2026-07-13",
    )

    assert pricing.currency == "EUR"
    assert pricing.cache_write_input_per_million is None
    assert usage.cache_miss_input_tokens == 6
    assert usage.reasoning_tokens == 7
    assert usage.raw == {"old": True}
    assert usage.cache_write_input_tokens == 0
    assert cost.output_cost == Decimal("0.7")
    assert cost.pricing_last_checked == "2026-07-13"
    assert cost.cache_write_input_cost is None


def test_catalog_rejects_unreachable_long_context_price_tiers() -> None:
    limits = ModelLimits(
        context_window_tokens=100_000,
        default_response_reserve_tokens=1_000,
        max_input_tokens=100_000,
        source_urls=(LIMIT_SOURCE_URL,),
        last_checked=LIMITS_CHECKED_ON,
    )
    pricing = TokenPricing(
        input_per_million=Decimal("1.0"),
        output_per_million=Decimal("2.0"),
        source_urls=(PRICING_SOURCE_URL,),
        last_checked=PRICING_CHECKED_ON,
        long_context=LongContextPricing(
            applies_above_input_tokens=100_000,
            input_per_million=Decimal("2.0"),
            output_per_million=Decimal("3.0"),
        ),
    )

    with pytest.raises(ValueError, match="threshold"):
        ModelSpec(
            provider="test",
            model="unreachable-tier",
            limits=limits,
            pricing=pricing,
        )


def test_catalog_requires_explicit_long_context_cache_write_pricing() -> None:
    pricing = TokenPricing(
        input_per_million=Decimal("1.0"),
        cached_input_per_million=Decimal("0.1"),
        cache_write_input_per_million=Decimal("1.25"),
        output_per_million=Decimal("2.0"),
        source_urls=(PRICING_SOURCE_URL,),
        last_checked=PRICING_CHECKED_ON,
        long_context=LongContextPricing(
            applies_above_input_tokens=50_000,
            input_per_million=Decimal("2.0"),
            cached_input_per_million=Decimal("0.2"),
            output_per_million=Decimal("3.0"),
        ),
    )

    with pytest.raises(ValueError, match="cache-write pricing is required"):
        ModelSpec(
            provider="test",
            model="missing-cache-write-tier",
            limits=_spec("source").limits,
            pricing=pricing,
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
    anthropic = resolve_model_spec("anthropic", "claude-sonnet-5")
    gemini = resolve_model_spec("gemini", "gemini-3.5-flash")
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

    assert anthropic is not None
    assert anthropic.limits.context_window_tokens == 1_000_000
    assert anthropic.limits.max_input_tokens == 1_000_000
    assert anthropic.limits.max_output_tokens == 128_000
    assert anthropic.pricing is None
    assert anthropic.limits.source_urls == (
        "https://platform.claude.com/docs/en/about-claude/models/overview",
        "https://platform.claude.com/docs/en/api/models",
    )

    assert gemini is not None
    assert gemini.limits.context_window_tokens == 1_048_576
    assert gemini.limits.max_input_tokens == 1_048_576
    assert gemini.limits.max_output_tokens == 65_536
    assert gemini.pricing is not None
    assert gemini.pricing.source_urls == (
        "https://ai.google.dev/gemini-api/docs/pricing",
    )

    assert deepseek is not None
    assert deepseek.limits.context_window_tokens == 1_048_576
    assert deepseek.limits.max_input_tokens is None
    assert deepseek.limits.max_output_tokens == 393_216
    assert deepseek.status == "preview"

    assert gemma is not None
    assert gemma.model == "google/gemma-4-12b-qat"
    assert gemma.limits.context_window_tokens == 262_144
    assert gemma.limits.max_output_tokens is None
    assert len(gemma.limits.source_urls) == 2


def test_newest_provider_models_keep_exact_limits_and_pricing() -> None:
    chat_latest = resolve_model_spec("openai", "chat-latest")
    sol = resolve_model_spec("openai", "gpt-5.6-sol")
    gemma = resolve_model_spec("gemini", "gemma-4-26b-a4b-it")
    robotics = resolve_model_spec("gemini", "gemini-robotics-er-1.6-preview")
    computer_use = resolve_model_spec(
        "gemini",
        "gemini-2.5-computer-use-preview-10-2025",
    )
    openai_computer_use = resolve_model_spec("openai", "computer-use-preview")

    assert chat_latest is not None
    assert chat_latest.limits.context_window_tokens == 400_000
    assert chat_latest.limits.max_output_tokens == 128_000
    assert chat_latest.pricing is not None
    assert chat_latest.pricing.input_per_million == Decimal("5.00")

    assert sol is not None
    assert sol.pricing is not None
    assert sol.pricing.cache_write_input_per_million == Decimal("6.25")
    assert sol.pricing.long_context is not None
    assert sol.pricing.long_context.cache_write_input_per_million == Decimal(
        "12.50"
    )

    assert gemma is not None
    assert gemma.limits.context_window_tokens == 262_144
    assert gemma.limits.max_output_tokens == 32_768
    assert gemma.limits.source_urls == (
        "https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api",
        "https://ai.google.dev/api/models",
    )
    assert gemma.pricing is not None
    assert gemma.pricing.input_per_million == Decimal("0")
    assert gemma.pricing.output_per_million == Decimal("0")

    assert robotics is not None
    assert robotics.pricing is not None
    assert robotics.pricing.input_per_million == Decimal("1.00")
    assert robotics.pricing.output_per_million == Decimal("5.00")

    assert computer_use is not None
    assert computer_use.limits.context_window_tokens == 131_072
    assert computer_use.limits.max_output_tokens == 65_536

    assert openai_computer_use is not None
    assert openai_computer_use.limits.default_response_reserve_tokens == 1_024
    assert (
        resolve_model_capabilities("openai", "computer-use-preview").input_budget_tokens
        == 7_168
    )


def test_deprecated_models_record_replacements_without_guessing_alias_lifecycle() -> None:
    nano = resolve_model_spec("openai", "gpt-4.1-nano")
    gpt_5 = resolve_model_spec("openai", "gpt-5")
    gpt_5_snapshot = resolve_model_spec("openai", "gpt-5-2025-08-07")
    opus = resolve_model_spec("anthropic", "claude-opus-4-1")
    gemini = resolve_model_spec("gemini", "gemini-2.5-pro")
    codex = resolve_model_spec("openai", "gpt-5.2-codex")
    old_4o = resolve_model_spec("openai", "gpt-4o-2024-05-13")
    current_4o = resolve_model_spec("openai", "gpt-4o")

    assert nano is not None
    assert nano.status == "deprecated"
    assert nano.replacement_model == "gpt-5.4-nano"
    assert nano.retired_on == date(2026, 10, 23)
    assert gpt_5 is not None
    assert gpt_5.status == "stable"
    assert gpt_5_snapshot is not None
    assert gpt_5_snapshot is not gpt_5
    assert gpt_5_snapshot.status == "deprecated"
    assert gpt_5_snapshot.replacement_model == "gpt-5.5"
    assert gpt_5_snapshot.retired_on == date(2026, 12, 11)
    assert opus is not None
    assert opus.status == "deprecated"
    assert opus.replacement_model == "claude-opus-4-8"
    assert opus.retired_on == date(2026, 8, 5)
    assert gemini is not None
    assert gemini.status == "deprecated"
    assert gemini.replacement_model == "gemini-3.1-pro-preview"
    # Google publishes an earliest possible shutdown, not a guaranteed date.
    assert gemini.retired_on is None
    assert codex is not None
    assert codex.status == "deprecated"
    assert codex.replacement_model == "gpt-5.5"
    assert codex.retired_on == date(2026, 7, 23)
    assert old_4o is not None
    assert current_4o is not None
    assert old_4o is not current_4o
    assert old_4o.status == "deprecated"
    assert old_4o.replacement_model == "gpt-5.5"
    assert old_4o.retired_on == date(2026, 10, 23)
    assert old_4o.limits.max_output_tokens == 4_096
    assert old_4o.pricing is not None
    assert current_4o.pricing is not None
    assert old_4o.pricing.input_per_million == Decimal("5.00")
    assert current_4o.pricing.input_per_million == Decimal("2.50")


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


def test_conservative_capabilities_preserve_the_smallest_known_input_cap() -> None:
    capped = LLMModelCapabilities(
        provider="capped",
        model="known-input-cap",
        context_window_tokens=100_000,
        default_response_reserve_tokens=8_000,
        max_input_tokens=64_000,
    )
    uncapped = LLMModelCapabilities(
        provider="uncapped",
        model="unknown-input-cap",
        context_window_tokens=100_000,
        default_response_reserve_tokens=8_000,
    )

    combined = conservative_model_capabilities((capped, uncapped))

    assert combined.max_input_tokens == 64_000
    assert combined.input_budget_tokens == 64_000


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


def test_long_context_pricing_switches_the_full_request_above_the_threshold() -> None:
    at_threshold = estimate_cost(
        "openai",
        "gpt-5.4",
        LLMUsage(
            input_tokens=272_000,
            output_tokens=1_000,
            cached_input_tokens=72_000,
        ),
    )
    above_threshold = estimate_cost(
        "openai",
        "gpt-5.4",
        LLMUsage(
            input_tokens=272_001,
            output_tokens=1_000,
            cached_input_tokens=72_001,
        ),
    )

    assert at_threshold is not None
    assert at_threshold.input_cost == Decimal("0.500000")
    assert at_threshold.cached_input_cost == Decimal("0.018000")
    assert at_threshold.output_cost == Decimal("0.015000")
    assert at_threshold.amount == Decimal("0.533000")
    assert above_threshold is not None
    assert above_threshold.input_cost == Decimal("1.000000")
    assert above_threshold.cached_input_cost == Decimal("0.0360005")
    assert above_threshold.output_cost == Decimal("0.022500")
    assert above_threshold.amount == Decimal("1.0585005")


def test_long_context_tier_uses_cache_buckets_when_total_input_is_missing() -> None:
    cost = estimate_cost(
        "openai",
        "gpt-5.4",
        LLMUsage(
            input_tokens=0,
            output_tokens=1_000,
            cache_hit_input_tokens=72_001,
            cache_miss_input_tokens=200_000,
        ),
    )

    assert cost is not None
    assert cost.amount == Decimal("1.0585005")
    assert cost.estimated is True


def test_long_context_tier_prices_cache_writes_at_the_published_rate() -> None:
    cost = estimate_cost(
        "openai",
        "gpt-5.6-sol",
        LLMUsage(
            input_tokens=272_001,
            output_tokens=1_000,
            cache_hit_input_tokens=72_001,
            cache_write_input_tokens=50_000,
            cache_miss_input_tokens=150_000,
        ),
    )

    assert cost is not None
    assert cost.input_cost == Decimal("1.500000")
    assert cost.cached_input_cost == Decimal("0.072001")
    assert cost.cache_write_input_cost == Decimal("0.625000")
    assert cost.output_cost == Decimal("0.045000")
    assert cost.amount == Decimal("2.242001")


def test_openai_cache_write_pricing_uses_a_distinct_usage_bucket() -> None:
    openai = estimate_cost(
        "openai",
        "gpt-5.6-sol",
        LLMUsage(
            input_tokens=1_000,
            output_tokens=100,
            cache_hit_input_tokens=200,
            cache_write_input_tokens=300,
            cache_miss_input_tokens=500,
        ),
    )

    assert openai is not None
    assert openai.pricing_known is True
    assert openai.input_cost == Decimal("0.002500")
    assert openai.cached_input_cost == Decimal("0.000100")
    assert openai.cache_write_input_cost == Decimal("0.001875")
    assert openai.output_cost == Decimal("0.003000")
    assert openai.amount == Decimal("0.007475")
    assert openai.pricing_source == (
        "https://developers.openai.com/api/docs/pricing"
    )


def test_cache_write_usage_and_cost_aggregate_without_losing_the_bucket() -> None:
    usage = aggregate_usage(
        [
            LLMUsage(input_tokens=10, cache_write_input_tokens=2),
            LLMUsage(input_tokens=20, cache_write_input_tokens=3),
        ]
    )
    cost = aggregate_cost(
        [
            LLMCost(
                amount=Decimal("0.10"),
                pricing_known=True,
                cache_write_input_cost=Decimal("0.02"),
            ),
            LLMCost(
                amount=Decimal("0.20"),
                pricing_known=True,
                cache_write_input_cost=Decimal("0.03"),
            ),
        ]
    )

    assert usage is not None
    assert usage.cache_write_input_tokens == 5
    assert cost is not None
    assert cost.amount == Decimal("0.30")
    assert cost.cache_write_input_cost == Decimal("0.05")


def test_chat_completions_normalizes_nested_cache_write_details() -> None:
    usage = normalize_chat_completions_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 30,
            },
        }
    )

    assert usage is not None
    assert usage.cache_hit_input_tokens == 20
    assert usage.cache_write_input_tokens == 30
    assert usage.cache_miss_input_tokens == 50
    assert usage.cache_split_estimated is False


def test_invalid_cache_write_split_is_clamped_and_marked_estimated() -> None:
    cost = estimate_cost(
        "openai",
        "gpt-5.6-sol",
        LLMUsage(
            input_tokens=100,
            cache_hit_input_tokens=80,
            cache_write_input_tokens=50,
        ),
    )

    assert cost is not None
    assert cost.cached_input_cost == Decimal("0.000040")
    assert cost.cache_write_input_cost == Decimal("0.000125")
    assert cost.amount == Decimal("0.000165")
    assert cost.estimated is True


def test_cache_write_buckets_without_aggregate_input_are_still_priced() -> None:
    cost = estimate_cost(
        "openai",
        "gpt-5.6-sol",
        LLMUsage(
            output_tokens=10,
            cache_hit_input_tokens=20,
            cache_write_input_tokens=30,
            cache_miss_input_tokens=50,
        ),
    )

    assert cost is not None
    assert cost.amount == Decimal("0.0007475")
    assert cost.estimated is True


def test_anthropic_complex_cache_write_pricing_stays_explicitly_unknown() -> None:
    anthropic = estimate_cost(
        "anthropic",
        "claude-sonnet-5",
        LLMUsage(input_tokens=100, output_tokens=20),
    )

    assert anthropic is not None
    assert anthropic.pricing_known is False


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
