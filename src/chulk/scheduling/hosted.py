"""Hosted scheduling adapters that submit immutable-definition durable runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib

from chulk.gateway import GatewayRunTarget
from chulk.runs import AsyncRunStore, RunStore, RunSubmission, StepDefinition


@dataclass(frozen=True, slots=True)
class HostedScheduledOccurrence:
    """One due occurrence selected by a host-owned scheduler."""

    schedule_id: str
    target: GatewayRunTarget
    scheduled_for: datetime
    idempotency_key: str

    def __post_init__(self) -> None:
        for name in ("schedule_id", "idempotency_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        if self.scheduled_for.tzinfo is None:
            raise ValueError("scheduled_for must include a timezone")
        object.__setattr__(
            self,
            "scheduled_for",
            self.scheduled_for.astimezone(timezone.utc),
        )


class HostedScheduleRunSubmitter:
    """Submit a due schedule as one idempotent durable run."""

    def __init__(
        self,
        runs: RunStore,
        *,
        steps: tuple[StepDefinition, ...] = (
            StepDefinition(id="agent", name="Scheduled agent turn"),
        ),
    ) -> None:
        self.runs = runs
        self.steps = steps

    def submit(self, occurrence: HostedScheduledOccurrence) -> str:
        record = self.runs.submit(
            occurrence.target.scope,
            _submission(occurrence, self.steps),
            actor="scheduler",
        )
        return record.id


class AsyncHostedScheduleRunSubmitter:
    """Native async hosted schedule submission adapter."""

    def __init__(
        self,
        runs: AsyncRunStore,
        *,
        steps: tuple[StepDefinition, ...] = (
            StepDefinition(id="agent", name="Scheduled agent turn"),
        ),
    ) -> None:
        self.runs = runs
        self.steps = steps

    async def submit(self, occurrence: HostedScheduledOccurrence) -> str:
        record = await self.runs.submit(
            occurrence.target.scope,
            _submission(occurrence, self.steps),
            actor="scheduler",
        )
        return record.id


def _submission(
    occurrence: HostedScheduledOccurrence,
    steps: tuple[StepDefinition, ...],
) -> RunSubmission:
    source_event_id = (
        f"schedule:{occurrence.schedule_id}:"
        f"{occurrence.scheduled_for.isoformat()}"
    )
    digest = hashlib.sha256(source_event_id.encode("utf-8")).hexdigest()
    return RunSubmission(
        idempotency_key=occurrence.idempotency_key,
        input_digest=f"sha256:{digest}",
        definition_digest=occurrence.target.definition_digest,
        steps=steps,
        source_event_id=source_event_id,
        correlation_id=occurrence.target.scope.run_id,
        metadata={
            "schedule_id": occurrence.schedule_id,
            "scheduled_for": occurrence.scheduled_for.isoformat(),
            "source_event_id": source_event_id,
        },
    )


__all__ = [
    "AsyncHostedScheduleRunSubmitter",
    "HostedScheduleRunSubmitter",
    "HostedScheduledOccurrence",
]
