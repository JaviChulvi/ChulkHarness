"""Provider-neutral cost estimates backed by the immutable model catalog."""

from __future__ import annotations

from decimal import Decimal

from chulk.llm.model_catalog import TokenPricing, resolve_model_spec
from chulk.llm.usage import LLMCost, LLMUsage


TOKENS_PER_MILLION = Decimal("1000000")


def resolve_pricing(
    provider: str | None,
    model: str | None,
) -> TokenPricing | None:
    """Return the shared pricing record for a known model."""
    spec = resolve_model_spec(provider, model)
    return spec.pricing if spec is not None else None


def estimate_cost(
    provider: str | None,
    model: str | None,
    usage: LLMUsage | None,
) -> LLMCost | None:
    """Estimate cost from normalized usage and known model pricing."""
    if usage is None:
        return None
    spec = resolve_model_spec(provider, model)
    if spec is None or spec.pricing is None:
        return LLMCost(
            amount=None,
            pricing_known=False,
            estimated=usage.estimated,
            provider=provider,
            model=model,
        )
    pricing = spec.pricing

    cached_tokens, uncached_tokens, split_estimated = _input_token_split(usage)
    cached_rate = (
        pricing.cached_input_per_million
        if pricing.cached_input_per_million is not None
        else pricing.input_per_million
    )
    input_cost = _token_cost(uncached_tokens, pricing.input_per_million)
    cached_input_cost = _token_cost(cached_tokens, cached_rate)
    output_cost = _token_cost(usage.output_tokens, pricing.output_per_million)
    amount = input_cost + cached_input_cost + output_cost
    return LLMCost(
        amount=amount,
        currency=pricing.currency,
        pricing_known=True,
        estimated=(
            usage.estimated or usage.cache_split_estimated or split_estimated
        ),
        input_cost=input_cost,
        cached_input_cost=cached_input_cost,
        output_cost=output_cost,
        provider=provider,
        model=model,
        pricing_source=pricing.primary_source_url,
        pricing_last_checked=(
            pricing.last_checked.isoformat()
            if pricing.last_checked is not None
            else None
        ),
    )


def _input_token_split(usage: LLMUsage) -> tuple[int, int, bool]:
    input_tokens = max(usage.input_tokens, 0)
    reported_cached_tokens = (
        usage.cache_hit_input_tokens
        if usage.cache_hit_input_tokens > 0
        else usage.cached_input_tokens
    )
    cached_fields_conflict = (
        usage.cached_input_tokens > 0
        and usage.cache_hit_input_tokens > 0
        and usage.cached_input_tokens != usage.cache_hit_input_tokens
    )
    reported_uncached_tokens = max(usage.cache_miss_input_tokens, 0)

    if input_tokens == 0 and (reported_cached_tokens or reported_uncached_tokens):
        return reported_cached_tokens, reported_uncached_tokens, True

    cached_tokens = min(reported_cached_tokens, input_tokens)
    expected_uncached_tokens = input_tokens - cached_tokens
    if (
        reported_cached_tokens <= input_tokens
        and reported_uncached_tokens == expected_uncached_tokens
    ):
        return cached_tokens, reported_uncached_tokens, cached_fields_conflict
    return cached_tokens, expected_uncached_tokens, True


def _token_cost(tokens: int, rate_per_million: Decimal) -> Decimal:
    return Decimal(max(0, int(tokens))) * rate_per_million / TOKENS_PER_MILLION


__all__ = ["estimate_cost", "resolve_pricing"]
