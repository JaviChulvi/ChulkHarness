"""Static LLM pricing metadata and cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from chulk.llm.usage import LLMCost, LLMUsage


USD = "USD"
PRICING_LAST_CHECKED = "2026-06-17"
TOKENS_PER_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class ModelPricing:
    """Per-1M-token pricing metadata for one model."""

    provider: str
    model: str
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal
    source_url: str
    currency: str = USD
    last_checked: str = PRICING_LAST_CHECKED


MODEL_PRICING: dict[tuple[str, str], ModelPricing] = {}


def register_pricing(pricing: ModelPricing) -> None:
    MODEL_PRICING[_key(pricing.provider, pricing.model)] = pricing


def resolve_pricing(provider: str | None, model: str | None) -> ModelPricing | None:
    if not provider or not model:
        return None
    key = _key(provider, model)
    pricing = MODEL_PRICING.get(key)
    if pricing is not None:
        return pricing

    normalized_provider, normalized_model = key
    if normalized_provider == "openai":
        for prefix, base_model in [
            ("gpt-4.1-mini-", "gpt-4.1-mini"),
            ("gpt-4.1-nano-", "gpt-4.1-nano"),
            ("gpt-4.1-", "gpt-4.1"),
        ]:
            if normalized_model.startswith(prefix):
                return MODEL_PRICING.get((normalized_provider, base_model))
    return None


def estimate_cost(provider: str | None, model: str | None, usage: LLMUsage | None) -> LLMCost | None:
    """Estimate cost from normalized usage and known model pricing."""
    if usage is None:
        return None
    pricing = resolve_pricing(provider, model)
    if pricing is None:
        return LLMCost(
            amount=None,
            pricing_known=False,
            estimated=usage.estimated,
            provider=provider,
            model=model,
        )

    if pricing.provider == "deepseek":
        cached_tokens = usage.cache_hit_input_tokens or usage.cached_input_tokens
        uncached_tokens = usage.cache_miss_input_tokens
    else:
        cached_tokens = usage.cached_input_tokens or usage.cache_hit_input_tokens
        uncached_tokens = max(usage.input_tokens - cached_tokens, 0)

    input_cost = _token_cost(uncached_tokens, pricing.input_per_million)
    cached_input_cost = _token_cost(cached_tokens, pricing.cached_input_per_million)
    output_cost = _token_cost(usage.output_tokens, pricing.output_per_million)
    amount = input_cost + cached_input_cost + output_cost
    return LLMCost(
        amount=amount,
        currency=pricing.currency,
        pricing_known=True,
        estimated=usage.estimated or usage.cache_split_estimated,
        input_cost=input_cost,
        cached_input_cost=cached_input_cost,
        output_cost=output_cost,
        provider=provider,
        model=model,
        pricing_source=pricing.source_url,
        pricing_last_checked=pricing.last_checked,
    )


def _token_cost(tokens: int, rate_per_million: Decimal) -> Decimal:
    return Decimal(max(0, int(tokens))) * rate_per_million / TOKENS_PER_MILLION


def _key(provider: str, model: str) -> tuple[str, str]:
    return provider.lower().strip(), model.lower().strip()


for _provider, _model, _input, _cached, _output, _source in [
    (
        "openai",
        "gpt-4.1",
        "2.00",
        "0.50",
        "8.00",
        "https://developers.openai.com/api/docs/models/gpt-4.1",
    ),
    (
        "openai",
        "gpt-4.1-mini",
        "0.40",
        "0.10",
        "1.60",
        "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
    ),
    (
        "openai",
        "gpt-4.1-nano",
        "0.10",
        "0.025",
        "0.40",
        "https://developers.openai.com/api/docs/models/gpt-4.1-nano",
    ),
    (
        "deepseek",
        "deepseek-v4-flash",
        "0.14",
        "0.0028",
        "0.28",
        "https://api-docs.deepseek.com/quick_start/pricing",
    ),
    (
        "deepseek",
        "deepseek-v4-pro",
        "0.435",
        "0.003625",
        "0.87",
        "https://api-docs.deepseek.com/quick_start/pricing",
    ),
]:
    register_pricing(
        ModelPricing(
            provider=_provider,
            model=_model,
            input_per_million=Decimal(_input),
            cached_input_per_million=Decimal(_cached),
            output_per_million=Decimal(_output),
            source_url=_source,
        )
    )
