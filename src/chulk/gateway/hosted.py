"""Durable-run adapters for hosted gateway ingress."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any

from chulk.gateway.models import InboundEnvelope
from chulk.gateway.stores import GatewayRunTarget
from chulk.runs import AsyncRunStore, RunStore, RunSubmission, StepDefinition


class DurableGatewayRunSubmitter:
    """Idempotently transfer one inbound event to the durable run owner."""

    def __init__(
        self,
        runs: RunStore,
        *,
        steps: tuple[StepDefinition, ...] = (
            StepDefinition(id="agent", name="Agent turn"),
        ),
    ) -> None:
        self.runs = runs
        self.steps = steps

    def submit(
        self,
        target: GatewayRunTarget,
        envelope: InboundEnvelope,
        *,
        idempotency_key: str,
    ) -> str:
        record = self.runs.submit(
            target.scope,
            _submission(target, envelope, idempotency_key, self.steps),
            actor="gateway",
        )
        return record.id


class AsyncDurableGatewayRunSubmitter:
    """Native async durable-run submission adapter."""

    def __init__(
        self,
        runs: AsyncRunStore,
        *,
        steps: tuple[StepDefinition, ...] = (
            StepDefinition(id="agent", name="Agent turn"),
        ),
    ) -> None:
        self.runs = runs
        self.steps = steps

    async def submit(
        self,
        target: GatewayRunTarget,
        envelope: InboundEnvelope,
        *,
        idempotency_key: str,
    ) -> str:
        record = await self.runs.submit(
            target.scope,
            _submission(target, envelope, idempotency_key, self.steps),
            actor="gateway",
        )
        return record.id


def _submission(
    target: GatewayRunTarget,
    envelope: InboundEnvelope,
    idempotency_key: str,
    steps: tuple[StepDefinition, ...],
) -> RunSubmission:
    return RunSubmission(
        idempotency_key=idempotency_key,
        input_digest=_envelope_digest(envelope),
        definition_digest=target.definition_digest,
        steps=steps,
        source_event_id=envelope.event_id,
        correlation_id=target.scope.run_id,
        metadata={
            "adapter": envelope.identity.adapter,
            "account_id": envelope.identity.account_id,
            "destination_id": envelope.destination_id,
            "thread_id": envelope.thread_id,
            "source_event_id": envelope.event_id,
        },
    )


def _envelope_digest(envelope: InboundEnvelope) -> str:
    payload = _portable(envelope)
    assert isinstance(payload, dict)
    payload.pop("received_at", None)
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _portable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _portable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _portable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = [
    "AsyncDurableGatewayRunSubmitter",
    "DurableGatewayRunSubmitter",
]
