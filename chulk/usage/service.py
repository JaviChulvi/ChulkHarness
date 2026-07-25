"""Model-transport integration for durable usage and budget reservations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from chulk.core.context import estimate_message_tokens
from chulk.llm import LLMClient
from chulk.llm.pricing import estimate_cost
from chulk.llm.usage import LLMCost, LLMUsage, cost_from_dict, usage_from_dict
from chulk.usage.models import (
    BudgetReservation,
    ExactCost,
    ResourceKind,
    RunBudget,
    UsageDimensions,
    UsageEntry,
)
from chulk.usage.store import SQLiteUsageStore


@dataclass(frozen=True, slots=True)
class ModelMeter:
    """Safe identity and pricing inputs for one possible provider attempt."""

    provider: str | None
    model: str | None
    model_profile_id: str | None = None
    credential_ref: str | None = None


class ModelUsageAccounting:
    """Reserve before model calls and commit normalized results afterwards."""

    def __init__(
        self,
        store: SQLiteUsageStore,
        *,
        client: LLMClient,
        dimensions: UsageDimensions,
        budget: RunBudget,
        max_output_tokens: int,
        trace_path: Path | str | None = None,
    ) -> None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        self.store = store
        self.client = client
        if dimensions.conversation_id is None:
            raise ValueError("model usage accounting requires a conversation_id")
        self.dimensions = dimensions
        self.conversation_id = dimensions.conversation_id
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self.trace_path = str(trace_path) if trace_path is not None else None
        self._reservations: dict[tuple[str, int], BudgetReservation] = {}
        self.store.release_expired()

    def reserve_model_request(
        self,
        *,
        turn_id: str,
        request_index: int,
        messages: list[dict[str, str]],
        purpose: str,
        repair_attempts: int = 0,
    ) -> BudgetReservation:
        """Hold a conservative allowance for every provider that may be tried."""
        key = (turn_id, request_index)
        existing = self._reservations.get(key)
        if existing is not None:
            return existing
        meters = _model_meters(self.client)
        attempt_multiplier = max(1, repair_attempts + 1)
        prompt_tokens = sum(estimate_message_tokens(message) for message in messages)
        reserved_tokens = (
            (prompt_tokens + self.max_output_tokens)
            * len(meters)
            * attempt_multiplier
        )
        reserved_cost = _conservative_cost(
            meters,
            input_tokens=prompt_tokens,
            output_tokens=self.max_output_tokens,
            attempt_multiplier=attempt_multiplier,
        )
        source_event_id = _request_event_id(
            self.conversation_id,
            turn_id,
            request_index,
        )
        reservation = self.store.reserve(
            idempotency_key=source_event_id,
            source_event_id=source_event_id,
            resource_kind=ResourceKind.MODEL,
            dimensions=self._dimensions(turn_id),
            budget=self.budget,
            model_calls=len(meters) * attempt_multiplier,
            tokens=reserved_tokens,
            cost=reserved_cost,
        )
        self._reservations[key] = reservation
        return reservation

    def commit_model_request(
        self,
        *,
        turn_id: str,
        request_index: int,
        purpose: str,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        fallback_attempts: object = None,
    ) -> tuple[UsageEntry, ...]:
        """Commit actual provider attempts and atomically close their allowance."""
        reservation = self._reservation(turn_id, request_index)
        attempts = _attempt_values(fallback_attempts)
        entries: list[UsageEntry] = []
        if attempts:
            for index, attempt in enumerate(attempts, start=1):
                entries.append(
                    self._entry(
                        source_event_id=f"{reservation.source_event_id}:attempt:{index}",
                        purpose=purpose,
                        turn_id=turn_id,
                        usage=_attempt_usage(attempt),
                        cost=_attempt_cost(attempt),
                        provider=_optional_text(getattr(attempt, "provider", None)),
                        model=_optional_text(getattr(attempt, "model", None)),
                        model_profile_id=_optional_text(
                            getattr(attempt, "model_profile_id", None)
                        ),
                        model_calls=1,
                        metadata={
                            "request_event_id": reservation.source_event_id,
                            "attempt": index,
                            "success": bool(getattr(attempt, "success", False)),
                            "error_code": _optional_text(
                                getattr(attempt, "error_code", None)
                            ),
                        },
                    )
                )
        else:
            provider, model, model_profile_id = _result_identity(
                self.client,
                cost,
            )
            entries.append(
                self._entry(
                    source_event_id=f"{reservation.source_event_id}:result",
                    purpose=purpose,
                    turn_id=turn_id,
                    usage=usage,
                    cost=cost,
                    provider=provider,
                    model=model,
                    model_profile_id=model_profile_id,
                    model_calls=1,
                    metadata={"request_event_id": reservation.source_event_id},
                )
            )
        committed = self.store.commit(reservation.id, entries)
        self._reservations.pop((turn_id, request_index), None)
        return committed

    def release_model_request(
        self,
        *,
        turn_id: str,
        request_index: int,
    ) -> BudgetReservation | None:
        """Release a request that failed before normalized accounting was available."""
        reservation = self._reservations.pop((turn_id, request_index), None)
        if reservation is None:
            return None
        return self.store.release(reservation.id)

    def _reservation(self, turn_id: str, request_index: int) -> BudgetReservation:
        key = (turn_id, request_index)
        reservation = self._reservations.get(key)
        if reservation is None:
            source_event_id = _request_event_id(
                self.conversation_id,
                turn_id,
                request_index,
            )
            reservation = self.store.reserve(
                idempotency_key=source_event_id,
                source_event_id=source_event_id,
                resource_kind=ResourceKind.MODEL,
                dimensions=self._dimensions(turn_id),
                budget=self.budget,
            )
            self._reservations[key] = reservation
        return reservation

    def _entry(
        self,
        *,
        source_event_id: str,
        purpose: str,
        turn_id: str,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        provider: str | None,
        model: str | None,
        model_profile_id: str | None,
        model_calls: int,
        metadata: dict[str, Any],
    ) -> UsageEntry:
        now = datetime.now(timezone.utc)
        exact_cost = _exact_cost(cost)
        units = {
            "model_calls": Decimal(model_calls),
            "input_tokens": Decimal(usage.input_tokens if usage is not None else 0),
            "output_tokens": Decimal(
                usage.output_tokens if usage is not None else 0
            ),
            "total_tokens": Decimal(usage.total_tokens if usage is not None else 0),
            "cached_input_tokens": Decimal(
                usage.cached_input_tokens if usage is not None else 0
            ),
            "reasoning_tokens": Decimal(
                usage.reasoning_tokens if usage is not None else 0
            ),
        }
        meter = _meter_for_profile(self.client, model_profile_id)
        return UsageEntry(
            id=str(uuid4()),
            resource_kind=ResourceKind.MODEL,
            source_event_id=source_event_id,
            dimensions=self._dimensions(turn_id),
            occurred_at=now,
            billing_period=now.strftime("%Y-%m"),
            purpose=purpose,
            units=units,
            cost=exact_cost,
            provider=provider or (meter.provider if meter is not None else None),
            model=model or (meter.model if meter is not None else None),
            credential_ref=meter.credential_ref if meter is not None else None,
            model_profile_id=model_profile_id
            or (meter.model_profile_id if meter is not None else None),
            usage_estimated=usage.estimated if usage is not None else False,
            trace_path=self.trace_path,
            metadata=metadata,
        )

    def _dimensions(self, turn_id: str) -> UsageDimensions:
        return replace(self.dimensions, turn_id=turn_id)


def _model_meters(client: LLMClient) -> tuple[ModelMeter, ...]:
    providers = getattr(client, "providers", None)
    clients = (
        tuple(providers)
        if isinstance(providers, (list, tuple)) and providers
        else (client,)
    )
    return tuple(
        ModelMeter(
            provider=_optional_text(getattr(candidate, "provider", None)),
            model=_optional_text(getattr(candidate, "model", None)),
            model_profile_id=_optional_text(
                getattr(candidate, "model_profile_id", None)
            ),
            credential_ref=_optional_text(
                getattr(candidate, "credential_ref", None)
            ),
        )
        for candidate in clients
    )


def _meter_for_profile(
    client: LLMClient,
    model_profile_id: str | None,
) -> ModelMeter | None:
    meters = _model_meters(client)
    if model_profile_id is not None:
        for meter in meters:
            if meter.model_profile_id == model_profile_id:
                return meter
    return meters[0] if len(meters) == 1 else None


def _conservative_cost(
    meters: tuple[ModelMeter, ...],
    *,
    input_tokens: int,
    output_tokens: int,
    attempt_multiplier: int,
) -> ExactCost:
    costs = [
        estimate_cost(
            meter.provider,
            meter.model,
            LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cache_miss_input_tokens=input_tokens,
                estimated=True,
                cache_split_estimated=True,
                source="budget_reservation",
            ),
        )
        for meter in meters
    ]
    if not costs or any(cost is None or cost.amount is None for cost in costs):
        return ExactCost(None, pricing_known=False, estimated=True)
    currencies = {cost.currency for cost in costs if cost is not None}
    if len(currencies) != 1:
        return ExactCost(None, pricing_known=False, estimated=True)
    amount = sum(
        (
            (cost.amount or Decimal(0)) * attempt_multiplier
            for cost in costs
            if cost is not None
        ),
        start=Decimal(0),
    )
    return ExactCost(
        amount,
        currency=next(iter(currencies)),
        pricing_known=True,
        estimated=True,
    )


def _exact_cost(cost: LLMCost | None) -> ExactCost:
    if cost is None:
        return ExactCost(None)
    return ExactCost(
        cost.amount,
        currency=cost.currency,
        pricing_known=cost.pricing_known,
        estimated=cost.estimated,
        reported=False,
    )


def _attempt_values(value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(value)


def _attempt_usage(attempt: object) -> LLMUsage | None:
    value = getattr(attempt, "usage", None)
    if isinstance(value, LLMUsage):
        return value
    if isinstance(value, dict):
        return usage_from_dict(value)
    return None


def _attempt_cost(attempt: object) -> LLMCost | None:
    value = getattr(attempt, "cost", None)
    if isinstance(value, LLMCost):
        return value
    if isinstance(value, dict):
        return cost_from_dict(value)
    return None


def _result_identity(
    client: LLMClient,
    cost: LLMCost | None,
) -> tuple[str | None, str | None, str | None]:
    return (
        _optional_text(cost.provider if cost is not None else None)
        or _optional_text(getattr(client, "provider", None)),
        _optional_text(cost.model if cost is not None else None)
        or _optional_text(getattr(client, "model", None)),
        _optional_text(getattr(client, "model_profile_id", None)),
    )


def _request_event_id(
    conversation_id: str,
    turn_id: str,
    request_index: int,
) -> str:
    return f"model:{conversation_id}:{turn_id}:{request_index}"


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["ModelMeter", "ModelUsageAccounting"]
