"""Normalized LLM usage and response accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class LLMUsage:
    """Provider-agnostic token usage for one model request."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_hit_input_tokens: int = 0
    cache_miss_input_tokens: int = 0
    reasoning_tokens: int = 0
    estimated: bool = False
    cache_split_estimated: bool = False
    source: str = "provider"
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = self.total_tokens or self.input_tokens + self.output_tokens
        object.__setattr__(self, "input_tokens", max(0, int(self.input_tokens)))
        object.__setattr__(self, "output_tokens", max(0, int(self.output_tokens)))
        object.__setattr__(self, "total_tokens", max(0, int(total)))
        object.__setattr__(self, "cached_input_tokens", max(0, int(self.cached_input_tokens)))
        object.__setattr__(self, "cache_hit_input_tokens", max(0, int(self.cache_hit_input_tokens)))
        object.__setattr__(self, "cache_miss_input_tokens", max(0, int(self.cache_miss_input_tokens)))
        object.__setattr__(self, "reasoning_tokens", max(0, int(self.reasoning_tokens)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_hit_input_tokens": self.cache_hit_input_tokens,
            "cache_miss_input_tokens": self.cache_miss_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "estimated": self.estimated,
            "cache_split_estimated": self.cache_split_estimated,
            "source": self.source,
            "raw": self.raw,
        }


@dataclass(frozen=True)
class LLMCost:
    """Estimated monetary cost for one model request."""

    amount: Decimal | None
    currency: str = "USD"
    pricing_known: bool = False
    estimated: bool = False
    input_cost: Decimal | None = None
    cached_input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    provider: str | None = None
    model: str | None = None
    pricing_source: str | None = None
    pricing_last_checked: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": _decimal_text(self.amount),
            "currency": self.currency,
            "pricing_known": self.pricing_known,
            "estimated": self.estimated,
            "input_cost": _decimal_text(self.input_cost),
            "cached_input_cost": _decimal_text(self.cached_input_cost),
            "output_cost": _decimal_text(self.output_cost),
            "provider": self.provider,
            "model": self.model,
            "pricing_source": self.pricing_source,
            "pricing_last_checked": self.pricing_last_checked,
        }


@dataclass(frozen=True)
class LLMResponse:
    """Text response plus optional provider accounting metadata."""

    content: str
    usage: LLMUsage | None = None
    cost: LLMCost | None = None
    provider: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_openai_usage(usage: object) -> LLMUsage | None:
    """Return normalized usage from an OpenAI Responses-style usage object."""
    if usage is None:
        return None
    input_tokens = _int_value(_value(usage, "input_tokens"))
    output_tokens = _int_value(_value(usage, "output_tokens"))
    total_tokens = _int_value(_value(usage, "total_tokens")) or input_tokens + output_tokens
    input_details = _value(usage, "input_tokens_details")
    output_details = _value(usage, "output_tokens_details")
    cached_tokens = _int_value(_value(input_details, "cached_tokens"))
    reasoning_tokens = _int_value(_value(output_details, "reasoning_tokens"))
    if not any([input_tokens, output_tokens, total_tokens, cached_tokens, reasoning_tokens]):
        return None
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_tokens,
        cache_hit_input_tokens=cached_tokens,
        cache_miss_input_tokens=max(input_tokens - cached_tokens, 0),
        reasoning_tokens=reasoning_tokens,
        estimated=False,
        source="provider",
        raw=_public_dict(usage),
    )


def normalize_deepseek_usage(usage: object) -> LLMUsage | None:
    """Return normalized usage from a DeepSeek chat-completions usage object."""
    if usage is None:
        return None
    prompt_tokens = _int_value(_value(usage, "prompt_tokens"))
    completion_tokens = _int_value(_value(usage, "completion_tokens"))
    total_tokens = _int_value(_value(usage, "total_tokens")) or prompt_tokens + completion_tokens
    cache_hit = _int_value(_value(usage, "prompt_cache_hit_tokens"))
    cache_miss = _int_value(_value(usage, "prompt_cache_miss_tokens"))
    cache_split_estimated = False
    if prompt_tokens and not cache_hit and not cache_miss:
        cache_miss = prompt_tokens
        cache_split_estimated = True
    completion_details = _value(usage, "completion_tokens_details")
    reasoning_tokens = _int_value(_value(completion_details, "reasoning_tokens"))
    if not any([prompt_tokens, completion_tokens, total_tokens, cache_hit, cache_miss, reasoning_tokens]):
        return None
    return LLMUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cache_hit,
        cache_hit_input_tokens=cache_hit,
        cache_miss_input_tokens=cache_miss,
        reasoning_tokens=reasoning_tokens,
        estimated=False,
        cache_split_estimated=cache_split_estimated,
        source="provider",
        raw=_public_dict(usage),
    )


def estimate_usage(messages: list[dict[str, str]], content: str) -> LLMUsage:
    """Estimate usage with Chulk's deterministic context estimator."""
    from chulk.core.context import estimate_message_tokens, estimate_tokens

    input_tokens = sum(estimate_message_tokens(message) for message in messages)
    output_tokens = estimate_tokens(content)
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_miss_input_tokens=input_tokens,
        estimated=True,
        cache_split_estimated=True,
        source="estimated",
    )


def aggregate_usage(usages: list[LLMUsage | None], *, source: str = "aggregate") -> LLMUsage | None:
    """Aggregate multiple usage records."""
    records = [usage for usage in usages if usage is not None]
    if not records:
        return None
    return LLMUsage(
        input_tokens=sum(item.input_tokens for item in records),
        output_tokens=sum(item.output_tokens for item in records),
        total_tokens=sum(item.total_tokens for item in records),
        cached_input_tokens=sum(item.cached_input_tokens for item in records),
        cache_hit_input_tokens=sum(item.cache_hit_input_tokens for item in records),
        cache_miss_input_tokens=sum(item.cache_miss_input_tokens for item in records),
        reasoning_tokens=sum(item.reasoning_tokens for item in records),
        estimated=any(item.estimated for item in records),
        cache_split_estimated=any(item.cache_split_estimated for item in records),
        source=source,
    )


def aggregate_cost(costs: list[LLMCost | None]) -> LLMCost | None:
    """Aggregate multiple cost records, preserving unknown-pricing state."""
    records = [cost for cost in costs if cost is not None]
    if not records:
        return None
    known_amounts = [cost.amount for cost in records if cost.amount is not None]
    amount = sum(known_amounts, Decimal("0")) if known_amounts else None
    return LLMCost(
        amount=amount,
        pricing_known=all(cost.pricing_known for cost in records),
        estimated=any(cost.estimated for cost in records),
        input_cost=_sum_optional(cost.input_cost for cost in records),
        cached_input_cost=_sum_optional(cost.cached_input_cost for cost in records),
        output_cost=_sum_optional(cost.output_cost for cost in records),
        provider="mixed" if len({cost.provider for cost in records}) > 1 else records[0].provider,
        model="mixed" if len({cost.model for cost in records}) > 1 else records[0].model,
    )


def usage_from_dict(payload: object) -> LLMUsage | None:
    if not isinstance(payload, dict):
        return None
    return LLMUsage(
        input_tokens=_int_value(payload.get("input_tokens")),
        output_tokens=_int_value(payload.get("output_tokens")),
        total_tokens=_int_value(payload.get("total_tokens")),
        cached_input_tokens=_int_value(payload.get("cached_input_tokens")),
        cache_hit_input_tokens=_int_value(payload.get("cache_hit_input_tokens")),
        cache_miss_input_tokens=_int_value(payload.get("cache_miss_input_tokens")),
        reasoning_tokens=_int_value(payload.get("reasoning_tokens")),
        estimated=bool(payload.get("estimated")),
        cache_split_estimated=bool(payload.get("cache_split_estimated")),
        source=str(payload.get("source") or "provider"),
        raw=payload.get("raw") if isinstance(payload.get("raw"), dict) else {},
    )


def cost_from_dict(payload: object) -> LLMCost | None:
    if not isinstance(payload, dict):
        return None
    amount = _decimal_value(payload.get("amount"))
    return LLMCost(
        amount=amount,
        currency=str(payload.get("currency") or "USD"),
        pricing_known=bool(payload.get("pricing_known")),
        estimated=bool(payload.get("estimated")),
        input_cost=_decimal_value(payload.get("input_cost")),
        cached_input_cost=_decimal_value(payload.get("cached_input_cost")),
        output_cost=_decimal_value(payload.get("output_cost")),
        provider=payload.get("provider"),
        model=payload.get("model"),
        pricing_source=payload.get("pricing_source"),
        pricing_last_checked=payload.get("pricing_last_checked"),
    )


def _value(source: object, key: str) -> object:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _int_value(value: object) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _decimal_value(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _public_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): _public_value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return _public_dict(dumped)
        except Exception:
            pass
    result: dict[str, Any] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if callable(item):
            continue
        if isinstance(item, (str, int, float, bool, type(None), dict)):
            result[key] = _public_value(item)
    return result


def _public_value(value: object) -> Any:
    if isinstance(value, dict):
        return {str(key): _public_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        try:
            return _public_value(value.model_dump())
        except Exception:
            return str(value)
    return str(value)


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value.normalize(), "f")


def _sum_optional(values) -> Decimal | None:
    items = [value for value in values if value is not None]
    if not items:
        return None
    return sum(items, Decimal("0"))
