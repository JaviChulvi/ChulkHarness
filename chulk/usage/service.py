"""Model-transport integration for durable usage and budget reservations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
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
        additional_budgets: Iterable[RunBudget] = (),
        max_output_tokens: int,
        trace_path: Path | str | None = None,
        boundary_callback: Callable[[], object] | None = None,
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
        self.additional_budgets = tuple(additional_budgets)
        scopes = [budget.scope, *(item.scope for item in self.additional_budgets)]
        if len(scopes) != len(set(scopes)):
            raise ValueError("usage accounting budgets must use distinct scopes")
        self.max_output_tokens = max_output_tokens
        self.trace_path = str(trace_path) if trace_path is not None else None
        self.boundary_callback = boundary_callback
        self._reservations: dict[tuple[str, int], BudgetReservation] = {}
        self._constraint_reservations: dict[
            tuple[str, int],
            tuple[BudgetReservation, ...],
        ] = {}
        self._tool_reservations: dict[tuple[str, int, int], BudgetReservation] = {}
        self._tool_constraint_reservations: dict[
            tuple[str, int, int],
            tuple[BudgetReservation, ...],
        ] = {}
        self.reconcile_persisted_model_requests()
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
        if self.boundary_callback is not None:
            self.boundary_callback()
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
        try:
            constraints = self._reserve_constraints(
                source_event_id=source_event_id,
                resource_kind=ResourceKind.MODEL,
                dimensions=self._dimensions(turn_id),
                model_calls=len(meters) * attempt_multiplier,
                tokens=reserved_tokens,
                cost=reserved_cost,
            )
        except Exception:
            self.store.release(reservation.id)
            raise
        self._reservations[key] = reservation
        self._constraint_reservations[key] = constraints
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
        provider, model, model_profile_id = _result_identity(
            self.client,
            cost,
        )
        meter = _meter_for_profile(self.client, model_profile_id)
        occurred_at = self.store.checkpoint_model_response(
            reservation.id,
            request_index=request_index,
            purpose=purpose,
            usage=usage.to_dict() if usage is not None else None,
            cost=cost.to_dict() if cost is not None else None,
            fallback_attempts=tuple(
                _checkpoint_attempt(self.client, attempt)
                for attempt in attempts
            ),
            provider=provider,
            model=model,
            model_profile_id=model_profile_id,
            credential_ref=meter.credential_ref if meter is not None else None,
            trace_path=self.trace_path,
        )
        entries = self._response_entries(
            reservation=reservation,
            purpose=purpose,
            usage=usage,
            cost=cost,
            attempts=attempts,
            provider=provider,
            model=model,
            model_profile_id=model_profile_id,
            credential_ref=meter.credential_ref if meter is not None else None,
            occurred_at=occurred_at,
            trace_path=self.trace_path,
        )
        committed = self.store.commit(reservation.id, entries)
        key = (turn_id, request_index)
        for constraint in self._constraint_reservations.pop(key, ()):
            self.store.commit(constraint.id, entries)
        self._reservations.pop(key, None)
        return committed

    def reconcile_persisted_model_requests(self) -> tuple[UsageEntry, ...]:
        """Commit provider results checkpointed before an interrupted ledger write."""
        recovered: list[UsageEntry] = []
        while records := self.store.recoverable_model_usage(
            profile_id=self.dimensions.profile_id
        ):
            for record in records:
                reservation = record.reservation
                attempts = _attempt_values(record.fallback_attempts)
                entries = self._response_entries(
                    reservation=reservation,
                    purpose=record.purpose,
                    usage=usage_from_dict(
                        dict(record.usage) if record.usage is not None else None
                    ),
                    cost=cost_from_dict(
                        dict(record.cost) if record.cost is not None else None
                    ),
                    attempts=attempts,
                    provider=record.provider,
                    model=record.model,
                    model_profile_id=record.model_profile_id,
                    credential_ref=record.credential_ref,
                    occurred_at=record.occurred_at,
                    trace_path=record.trace_path,
                    recovered=True,
                )
                recovered.extend(self.store.commit(reservation.id, entries))
                for constraint in self.store.active_constraint_reservations(
                    reservation.source_event_id
                ):
                    self.store.commit(constraint.id, entries)
        return tuple(recovered)

    def _response_entries(
        self,
        *,
        reservation: BudgetReservation,
        purpose: str,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        attempts: tuple[object, ...],
        provider: str | None,
        model: str | None,
        model_profile_id: str | None,
        credential_ref: str | None,
        occurred_at: datetime | None,
        trace_path: str | None,
        recovered: bool = False,
    ) -> tuple[UsageEntry, ...]:
        entries: list[UsageEntry] = []
        if attempts:
            for index, attempt in enumerate(attempts, start=1):
                attempt_profile_id = _optional_text(
                    _attempt_value(attempt, "model_profile_id")
                )
                attempt_meter = _meter_for_profile(
                    self.client,
                    attempt_profile_id,
                )
                entries.append(
                    self._entry(
                        source_event_id=f"{reservation.source_event_id}:attempt:{index}",
                        purpose=purpose,
                        dimensions=reservation.dimensions,
                        usage=_attempt_usage(attempt),
                        cost=_attempt_cost(attempt),
                        provider=_optional_text(
                            _attempt_value(attempt, "provider")
                        ),
                        model=_optional_text(_attempt_value(attempt, "model")),
                        model_profile_id=attempt_profile_id,
                        credential_ref=_optional_text(
                            _attempt_value(attempt, "credential_ref")
                        )
                        or (
                            attempt_meter.credential_ref
                            if attempt_meter is not None
                            else None
                        ),
                        model_calls=1,
                        metadata={
                            "request_event_id": reservation.source_event_id,
                            "attempt": index,
                            "success": bool(
                                _attempt_value(attempt, "success")
                            ),
                            "error_code": _optional_text(
                                _attempt_value(attempt, "error_code")
                            ),
                            "recovered": recovered,
                        },
                        occurred_at=(
                            occurred_at + timedelta(microseconds=index - 1)
                            if occurred_at is not None
                            else None
                        ),
                        trace_path=trace_path,
                    )
                )
        else:
            entries.append(
                self._entry(
                    source_event_id=f"{reservation.source_event_id}:result",
                    purpose=purpose,
                    dimensions=reservation.dimensions,
                    usage=usage,
                    cost=cost,
                    provider=provider,
                    model=model,
                    model_profile_id=model_profile_id,
                    credential_ref=credential_ref,
                    model_calls=1,
                    metadata={
                        "request_event_id": reservation.source_event_id,
                        "recovered": recovered,
                    },
                    occurred_at=occurred_at,
                    trace_path=trace_path,
                )
            )
        return tuple(entries)

    def release_model_request(
        self,
        *,
        turn_id: str,
        request_index: int,
    ) -> BudgetReservation | None:
        """Release a request that failed before normalized accounting was available."""
        key = (turn_id, request_index)
        reservation = self._reservations.pop(key, None)
        if reservation is None:
            return None
        for constraint in self._constraint_reservations.pop(key, ()):
            self.store.release(constraint.id)
        return self.store.release(reservation.id)

    def reserve_tool_call(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        attempt: int,
        tool_name: str,
    ) -> BudgetReservation:
        """Reserve one concrete tool attempt before permission or execution."""
        key = (turn_id, tool_call_index, attempt)
        existing = self._tool_reservations.get(key)
        if existing is not None:
            return existing
        if self.boundary_callback is not None:
            self.boundary_callback()
        source_event_id = _tool_event_id(
            self.conversation_id,
            turn_id,
            tool_call_index,
            attempt,
        )
        reservation = self.store.reserve(
            idempotency_key=source_event_id,
            source_event_id=source_event_id,
            resource_kind=ResourceKind.TOOL,
            dimensions=self._dimensions(turn_id),
            budget=self.budget,
            tool_calls=1,
            cost=ExactCost(Decimal(0), pricing_known=True),
        )
        try:
            constraints = self._reserve_constraints(
                source_event_id=source_event_id,
                resource_kind=ResourceKind.TOOL,
                dimensions=self._dimensions(turn_id),
                tool_calls=1,
                cost=ExactCost(Decimal(0), pricing_known=True),
            )
        except Exception:
            self.store.release(reservation.id)
            raise
        self._tool_reservations[key] = reservation
        self._tool_constraint_reservations[key] = constraints
        return reservation

    def commit_tool_call(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        attempt: int,
        tool_name: str,
        success: bool,
        failure_kind: str | None,
    ) -> tuple[UsageEntry, ...]:
        """Commit one tool attempt against its pre-execution reservation."""
        key = (turn_id, tool_call_index, attempt)
        reservation = self._tool_reservations.get(key)
        if reservation is None:
            reservation = self.reserve_tool_call(
                turn_id=turn_id,
                tool_call_index=tool_call_index,
                attempt=attempt,
                tool_name=tool_name,
            )
        now = self.store.clock()
        if now.tzinfo is None:
            raise ValueError("usage store clock must return a timezone-aware datetime")
        now = now.astimezone(timezone.utc)
        entry = UsageEntry(
            id=str(uuid4()),
            resource_kind=ResourceKind.TOOL,
            source_event_id=f"{reservation.source_event_id}:result",
            dimensions=reservation.dimensions,
            occurred_at=now,
            billing_period=now.strftime("%Y-%m"),
            purpose="agent_tool",
            units={"tool_calls": Decimal(1)},
            cost=ExactCost(Decimal(0), pricing_known=True),
            tool_or_service=tool_name,
            trace_path=self.trace_path,
            metadata={
                "request_event_id": reservation.source_event_id,
                "tool_call_index": tool_call_index,
                "attempt": attempt,
                "success": success,
                "failure_kind": failure_kind,
            },
        )
        committed = self.store.commit(reservation.id, (entry,))
        for constraint in self._tool_constraint_reservations.pop(key, ()):
            self.store.commit(constraint.id, (entry,))
        self._tool_reservations.pop(key, None)
        return committed

    def release_tool_call(
        self,
        *,
        turn_id: str,
        tool_call_index: int,
        attempt: int,
    ) -> BudgetReservation | None:
        """Release an attempt that failed before a result could be observed."""
        key = (turn_id, tool_call_index, attempt)
        reservation = self._tool_reservations.pop(key, None)
        if reservation is None:
            return None
        for constraint in self._tool_constraint_reservations.pop(key, ()):
            self.store.release(constraint.id)
        return self.store.release(reservation.id)

    def _reserve_constraints(
        self,
        *,
        source_event_id: str,
        resource_kind: ResourceKind,
        dimensions: UsageDimensions,
        model_calls: int = 0,
        tool_calls: int = 0,
        tokens: int = 0,
        cost: ExactCost,
    ) -> tuple[BudgetReservation, ...]:
        reservations: list[BudgetReservation] = []
        try:
            for budget in self.additional_budgets:
                reservations.append(
                    self.store.reserve(
                        idempotency_key=(
                            f"{source_event_id}:constraint:{budget.scope.value}"
                        ),
                        source_event_id=source_event_id,
                        resource_kind=resource_kind,
                        dimensions=dimensions,
                        budget=budget,
                        model_calls=model_calls,
                        tool_calls=tool_calls,
                        tokens=tokens,
                        cost=cost,
                    )
                )
        except Exception:
            for reservation in reservations:
                self.store.release(reservation.id)
            raise
        return tuple(reservations)

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
        dimensions: UsageDimensions,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        provider: str | None,
        model: str | None,
        model_profile_id: str | None,
        credential_ref: str | None,
        model_calls: int,
        metadata: dict[str, Any],
        occurred_at: datetime | None = None,
        trace_path: str | None = None,
    ) -> UsageEntry:
        now = occurred_at or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("model usage timestamp must be timezone-aware")
        now = now.astimezone(timezone.utc)
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
            dimensions=dimensions,
            occurred_at=now,
            billing_period=now.strftime("%Y-%m"),
            purpose=purpose,
            units=units,
            cost=exact_cost,
            provider=provider or (meter.provider if meter is not None else None),
            model=model or (meter.model if meter is not None else None),
            credential_ref=credential_ref
            or (meter.credential_ref if meter is not None else None),
            model_profile_id=model_profile_id
            or (meter.model_profile_id if meter is not None else None),
            usage_estimated=usage.estimated if usage is not None else False,
            trace_path=trace_path,
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


def _attempt_value(attempt: object, name: str) -> object:
    if isinstance(attempt, Mapping):
        return attempt.get(name)
    return getattr(attempt, name, None)


def _attempt_usage(attempt: object) -> LLMUsage | None:
    value = _attempt_value(attempt, "usage")
    if isinstance(value, LLMUsage):
        return value
    if isinstance(value, dict):
        return usage_from_dict(value)
    return None


def _attempt_cost(attempt: object) -> LLMCost | None:
    value = _attempt_value(attempt, "cost")
    if isinstance(value, LLMCost):
        return value
    if isinstance(value, dict):
        return cost_from_dict(value)
    return None


def _checkpoint_attempt(
    client: LLMClient,
    attempt: object,
) -> dict[str, object]:
    model_profile_id = _optional_text(
        _attempt_value(attempt, "model_profile_id")
    )
    meter = _meter_for_profile(client, model_profile_id)
    usage = _attempt_usage(attempt)
    cost = _attempt_cost(attempt)
    return {
        "provider": _optional_text(_attempt_value(attempt, "provider")),
        "model": _optional_text(_attempt_value(attempt, "model")),
        "success": bool(_attempt_value(attempt, "success")),
        "error_code": _optional_text(_attempt_value(attempt, "error_code")),
        "model_profile_id": model_profile_id,
        "credential_ref": meter.credential_ref if meter is not None else None,
        "usage": usage.to_dict() if usage is not None else None,
        "cost": cost.to_dict() if cost is not None else None,
    }


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


def _tool_event_id(
    conversation_id: str,
    turn_id: str,
    tool_call_index: int,
    attempt: int,
) -> str:
    return (
        f"tool:{conversation_id}:{turn_id}:{tool_call_index}:"
        f"attempt:{attempt}"
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["ModelMeter", "ModelUsageAccounting"]
