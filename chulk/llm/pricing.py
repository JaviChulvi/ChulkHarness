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

    (
        cached_tokens,
        cache_write_tokens,
        uncached_tokens,
        split_estimated,
    ) = _input_token_split(usage)
    input_rate = pricing.input_per_million
    cached_rate = pricing.cached_input_per_million
    cache_write_rate = pricing.cache_write_input_per_million
    output_rate = pricing.output_per_million
    long_context = pricing.long_context
    if (
        long_context is not None
        and cached_tokens + cache_write_tokens + uncached_tokens
        > long_context.applies_above_input_tokens
    ):
        input_rate = long_context.input_per_million
        cached_rate = long_context.cached_input_per_million
        cache_write_rate = long_context.cache_write_input_per_million
        output_rate = long_context.output_per_million
    if cached_rate is None:
        cached_rate = input_rate
    if cache_write_rate is None:
        cache_write_rate = input_rate

    input_cost = _token_cost(uncached_tokens, input_rate)
    cached_input_cost = _token_cost(cached_tokens, cached_rate)
    cache_write_input_cost = _token_cost(cache_write_tokens, cache_write_rate)
    output_cost = _token_cost(usage.output_tokens, output_rate)
    amount = input_cost + cached_input_cost + cache_write_input_cost + output_cost
    return LLMCost(
        amount=amount,
        currency=pricing.currency,
        pricing_known=True,
        estimated=(
            usage.estimated or usage.cache_split_estimated or split_estimated
        ),
        input_cost=input_cost,
        cached_input_cost=cached_input_cost,
        cache_write_input_cost=cache_write_input_cost,
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


def _input_token_split(usage: LLMUsage) -> tuple[int, int, int, bool]:
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
    reported_cache_write_tokens = max(usage.cache_write_input_tokens, 0)

    if input_tokens == 0 and (
        reported_cached_tokens
        or reported_cache_write_tokens
        or reported_uncached_tokens
    ):
        return (
            reported_cached_tokens,
            reported_cache_write_tokens,
            reported_uncached_tokens,
            True,
        )

    cached_tokens = min(reported_cached_tokens, input_tokens)
    cache_write_tokens = min(
        reported_cache_write_tokens,
        input_tokens - cached_tokens,
    )
    expected_uncached_tokens = input_tokens - cached_tokens - cache_write_tokens
    if (
        reported_cached_tokens <= input_tokens
        and reported_cache_write_tokens <= input_tokens - reported_cached_tokens
        and reported_uncached_tokens == expected_uncached_tokens
    ):
        return (
            cached_tokens,
            cache_write_tokens,
            reported_uncached_tokens,
            cached_fields_conflict,
        )
    return cached_tokens, cache_write_tokens, expected_uncached_tokens, True


def _token_cost(tokens: int, rate_per_million: Decimal) -> Decimal:
    return Decimal(max(0, int(tokens))) * rate_per_million / TOKENS_PER_MILLION


__all__ = ["estimate_cost", "resolve_pricing"]
