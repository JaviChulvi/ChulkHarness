"""Hosted schedules submit definition-pinned durable runs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from chulk import ExecutionScope
from chulk.gateway import GatewayRunTarget
from chulk.runs import AsyncInMemoryRunStore, InMemoryRunStore
from chulk.scheduling import (
    AsyncHostedScheduleRunSubmitter,
    HostedScheduledOccurrence,
    HostedScheduleRunSubmitter,
)


def _occurrence(run_id: str) -> HostedScheduledOccurrence:
    scope = ExecutionScope(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        actor_id="scheduler",
        agent_id="digest-agent",
        agent_version="2.0.0",
        run_id=run_id,
        trigger_id="daily-summary",
    )
    return HostedScheduledOccurrence(
        schedule_id="daily-summary",
        target=GatewayRunTarget(
            scope=scope,
            definition_id="digest-agent",
            definition_version="2.0.0",
            definition_digest="sha256:definition",
        ),
        scheduled_for=datetime(2026, 7, 27, 8, tzinfo=timezone.utc),
        idempotency_key="schedule:daily-summary:2026-07-27",
    )


def test_hosted_schedule_submits_reference_without_a_stored_prompt() -> None:
    runs = InMemoryRunStore()
    submitter = HostedScheduleRunSubmitter(runs)
    occurrence = _occurrence("run-scheduled")

    assert submitter.submit(occurrence) == "run-scheduled"
    assert submitter.submit(occurrence) == "run-scheduled"
    record = runs.get(occurrence.target.scope, "run-scheduled")
    assert record.definition_digest == "sha256:definition"
    assert "prompt" not in record.metadata
    assert record.metadata["schedule_id"] == "daily-summary"


@pytest.mark.asyncio
async def test_async_hosted_schedule_uses_async_run_store() -> None:
    runs = AsyncInMemoryRunStore()
    submitter = AsyncHostedScheduleRunSubmitter(runs)
    occurrence = _occurrence("run-async-scheduled")

    assert await submitter.submit(occurrence) == "run-async-scheduled"
