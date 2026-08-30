from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from chulk.core.model_accounting import ModelAccounting
from chulk.core.state import AgentState, TurnState
from chulk.llm import LLMCost, LLMUsage


class _UsageService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def commit_model_request(self, **kwargs):
        self.calls.append(("commit", kwargs))
        return [SimpleNamespace(id="entry-1", source_event_id="source-1")]


class _AsyncUsageService(_UsageService):
    async def commit_model_request(self, **kwargs):
        return super().commit_model_request(**kwargs)


def _accounting(service: object, events: list[tuple[str, dict]]) -> ModelAccounting:
    return ModelAccounting(
        state=AgentState(),
        trace=lambda event, payload: events.append((event, payload)),
        usage_accounting=None,
        async_usage_accounting=service,
    )


def _usage() -> LLMUsage:
    return LLMUsage(input_tokens=12, output_tokens=3)


def _cost() -> LLMCost:
    return LLMCost(amount=Decimal("0.002"), pricing_known=True)


@pytest.mark.asyncio
async def test_sync_and_async_recording_share_report_and_trace_shape() -> None:
    sync_events: list[tuple[str, dict]] = []
    async_events: list[tuple[str, dict]] = []
    sync = ModelAccounting(
        state=AgentState(),
        trace=lambda event, payload: sync_events.append((event, payload)),
        usage_accounting=_UsageService(),  # type: ignore[arg-type]
    )
    async_accounting = _accounting(_AsyncUsageService(), async_events)
    sync_turn = TurnState("sync", turn_id="turn-1")
    async_turn = TurnState("async", turn_id="turn-1")

    sync_result = sync.record(
        sync_turn,
        request_index=0,
        usage=_usage(),
        cost=_cost(),
    )
    async_result = await async_accounting.record_async(
        async_turn,
        request_index=0,
        usage=_usage(),
        cost=_cost(),
    )

    assert async_result == sync_result
    assert async_turn.model_usage_reports == sync_turn.model_usage_reports
    assert async_turn.model_usage_totals == sync_turn.model_usage_totals
    assert async_events == sync_events
