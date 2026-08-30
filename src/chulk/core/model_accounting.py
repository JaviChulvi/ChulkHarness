"""Model-request accounting shared by synchronous and asynchronous transports."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from chulk.core.events import TraceEvent
from chulk.core.state import AgentState, TurnState
from chulk.hosting.async_utils import call_async_service
from chulk.llm import LLMCost, LLMUsage
from chulk.llm.usage import (
    aggregate_cost,
    aggregate_usage,
    cost_from_dict,
    usage_from_dict,
)
from chulk.usage import BudgetExceededError, ModelUsageAccounting


class ModelAccounting:
    """Records model reservations and usage without owning model transport I/O."""

    def __init__(
        self,
        *,
        state: AgentState,
        trace: Callable[[str, dict], None],
        usage_accounting: ModelUsageAccounting | None,
        async_usage_accounting: object | None = None,
    ) -> None:
        self.state = state
        self.trace = trace
        self.usage_accounting = usage_accounting
        self.async_usage_accounting = async_usage_accounting

    def record(
        self,
        turn: TurnState,
        *,
        request_index: int,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        fallback_attempts: object = None,
        purpose: str = "agent_action",
    ) -> tuple[dict | None, dict | None]:
        usage_payload = usage.to_dict() if usage is not None else None
        cost_payload = cost.to_dict() if cost is not None else None
        attempt_payloads = fallback_attempt_payloads(fallback_attempts)
        if self.usage_accounting is not None:
            entries = self.usage_accounting.commit_model_request(
                turn_id=turn.turn_id,
                request_index=request_index,
                purpose=purpose,
                usage=usage,
                cost=cost,
                fallback_attempts=fallback_attempts,
            )
            self._trace_committed(turn, request_index, entries)
        return self._record_report(
            turn,
            request_index=request_index,
            purpose=purpose,
            usage=usage_payload,
            cost=cost_payload,
            fallback_attempts=attempt_payloads,
        )

    async def record_async(
        self,
        turn: TurnState,
        *,
        request_index: int,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        fallback_attempts: object = None,
        purpose: str = "agent_action",
    ) -> tuple[dict | None, dict | None]:
        service = self.async_usage_accounting
        if service is None:
            return await asyncio.to_thread(
                self.record,
                turn,
                request_index=request_index,
                usage=usage,
                cost=cost,
                fallback_attempts=fallback_attempts,
                purpose=purpose,
            )
        usage_payload = usage.to_dict() if usage is not None else None
        cost_payload = cost.to_dict() if cost is not None else None
        attempt_payloads = fallback_attempt_payloads(fallback_attempts)
        entries = await call_async_service(
            service,
            "commit_model_request",
            turn_id=turn.turn_id,
            request_index=request_index,
            purpose=purpose,
            usage=usage,
            cost=cost,
            fallback_attempts=fallback_attempts,
        )
        self._trace_committed(turn, request_index, entries)
        return self._record_report(
            turn,
            request_index=request_index,
            purpose=purpose,
            usage=usage_payload,
            cost=cost_payload,
            fallback_attempts=attempt_payloads,
        )

    def reserve(
        self,
        turn: TurnState,
        *,
        request_index: int,
        messages: list[dict[str, str]],
        purpose: str,
        repair_attempts: int = 0,
    ) -> dict | None:
        if self.usage_accounting is None:
            return None
        try:
            reservation = self.usage_accounting.reserve_model_request(
                turn_id=turn.turn_id,
                request_index=request_index,
                messages=messages,
                purpose=purpose,
                repair_attempts=repair_attempts,
            )
        except BudgetExceededError as exc:
            self._record_exhausted(turn, request_index, exc)
            raise
        return self._record_reservation(turn, request_index, reservation)

    async def reserve_async(
        self,
        turn: TurnState,
        *,
        request_index: int,
        messages: list[dict[str, str]],
        purpose: str,
        repair_attempts: int = 0,
    ) -> dict | None:
        service = self.async_usage_accounting
        if service is None:
            return await asyncio.to_thread(
                self.reserve,
                turn,
                request_index=request_index,
                messages=messages,
                purpose=purpose,
                repair_attempts=repair_attempts,
            )
        try:
            reservation = await call_async_service(
                service,
                "reserve_model_request",
                turn_id=turn.turn_id,
                request_index=request_index,
                messages=messages,
                purpose=purpose,
                repair_attempts=repair_attempts,
            )
        except BudgetExceededError as exc:
            self._record_exhausted(turn, request_index, exc)
            raise
        return self._record_reservation(turn, request_index, reservation)

    def release(
        self,
        turn: TurnState,
        *,
        request_index: int,
        reason: str,
    ) -> dict | None:
        if self.usage_accounting is None:
            return None
        reservation = self.usage_accounting.release_model_request(
            turn_id=turn.turn_id,
            request_index=request_index,
        )
        return self._record_release(turn, request_index, reason, reservation)

    async def release_async(
        self,
        turn: TurnState,
        *,
        request_index: int,
        reason: str,
    ) -> dict | None:
        service = self.async_usage_accounting
        if service is None:
            return await asyncio.to_thread(
                self.release,
                turn,
                request_index=request_index,
                reason=reason,
            )
        reservation = await call_async_service(
            service,
            "release_model_request",
            turn_id=turn.turn_id,
            request_index=request_index,
        )
        return self._record_release(turn, request_index, reason, reservation)

    def _record_report(
        self,
        turn: TurnState,
        *,
        request_index: int,
        purpose: str,
        usage: dict | None,
        cost: dict | None,
        fallback_attempts: list[dict],
    ) -> tuple[dict | None, dict | None]:
        if usage is None and cost is None and not fallback_attempts:
            return None, None
        report = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "purpose": purpose,
            "usage": usage,
            "cost": cost,
        }
        if fallback_attempts:
            report["fallback_attempts"] = fallback_attempts
        turn.model_usage_reports.append(report)
        turn.model_usage_totals = aggregate_model_usage_reports(turn.model_usage_reports)
        self.state.last_usage_report = turn.model_usage_totals
        return usage, cost

    def _trace_committed(
        self, turn: TurnState, request_index: int, entries: Any
    ) -> None:
        self.trace(
            TraceEvent.BUDGET_COMMITTED,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "resource_kind": "model",
                "entry_ids": [entry.id for entry in entries],
                "source_event_ids": [entry.source_event_id for entry in entries],
            },
        )

    def _record_exhausted(
        self, turn: TurnState, request_index: int, exc: BudgetExceededError
    ) -> None:
        payload = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "resource_kind": "model",
            "scope": exc.scope.value,
            "dimension": exc.dimension,
            "limit": exc.limit,
            "committed": exc.committed,
            "reserved": exc.reserved,
            "requested": exc.requested,
            "message": str(exc),
        }
        turn.extension_metadata["budget_exhausted"] = payload
        self.trace(TraceEvent.BUDGET_EXHAUSTED, payload)

    def _record_reservation(
        self, turn: TurnState, request_index: int, reservation: Any
    ) -> dict:
        payload = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "resource_kind": "model",
            "reservation_id": reservation.id,
            "scope": reservation.budget.scope.value,
            "reserved_model_calls": reservation.reserved_model_calls,
            "reserved_tokens": reservation.reserved_tokens,
            "reserved_cost": reservation.reserved_cost.to_dict(),
            "expires_at": (
                reservation.expires_at.isoformat()
                if reservation.expires_at is not None
                else None
            ),
        }
        self.trace(TraceEvent.BUDGET_RESERVED, payload)
        return payload

    def _record_release(
        self,
        turn: TurnState,
        request_index: int,
        reason: str,
        reservation: Any | None,
    ) -> dict | None:
        if reservation is None:
            return None
        payload = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "resource_kind": "model",
            "reservation_id": reservation.id,
            "reason": reason,
        }
        self.trace(TraceEvent.BUDGET_RELEASED, payload)
        return payload


def fallback_attempt_payloads(fallback_attempts: object) -> list[dict]:
    if not fallback_attempts or not isinstance(fallback_attempts, list):
        return []
    return [
        attempt.to_dict()
        if hasattr(attempt, "to_dict")
        else dict(attempt)
        if isinstance(attempt, dict)
        else {"attempt": str(attempt)}
        for attempt in fallback_attempts
    ]


def aggregate_model_usage_reports(reports: list[dict]) -> dict:
    request_usages: list[LLMUsage | None] = []
    request_costs: list[LLMCost | None] = []
    failed_attempt_usages: list[LLMUsage | None] = []
    failed_attempt_costs: list[LLMCost | None] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        request_usages.append(usage_from_dict(report.get("usage")))
        request_costs.append(cost_from_dict(report.get("cost")))
        attempts = report.get("fallback_attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("success") is not False:
                continue
            failed_attempt_usages.append(usage_from_dict(attempt.get("usage")))
            failed_attempt_costs.append(cost_from_dict(attempt.get("cost")))
    usage = aggregate_usage(
        [*request_usages, *failed_attempt_usages], source="turn_total"
    )
    cost = aggregate_cost([*request_costs, *failed_attempt_costs])
    return {
        "request_count": len([report for report in reports if isinstance(report, dict)]),
        "usage": usage.to_dict() if usage is not None else None,
        "cost": cost.to_dict() if cost is not None else None,
    }
