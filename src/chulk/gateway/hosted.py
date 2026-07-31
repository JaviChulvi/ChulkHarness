"""Published-definition and durable-run adapters for hosted gateway ingress."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any

from chulk.authoring import (
    AgentDefinitionStore,
    AsyncAgentDefinitionStore,
    DefinitionStatus,
)
from chulk.gateway.models import InboundEnvelope
from chulk.gateway.stores import GatewayRunTarget
from chulk.hosting.scope import ExecutionScope
from chulk.runs import AsyncRunStore, RunStore, RunSubmission, StepDefinition


ScopeFactory = Callable[[Any, InboundEnvelope], ExecutionScope]
AsyncScopeFactory = Callable[
    [Any, InboundEnvelope],
    Awaitable[ExecutionScope],
]


class PublishedDefinitionGatewayResolver:
    """Resolve host authority to one active immutable definition revision."""

    def __init__(
        self,
        definitions: AgentDefinitionStore,
        scope_factory: ScopeFactory,
    ) -> None:
        self.definitions = definitions
        self.scope_factory = scope_factory

    def resolve(self, route: Any, envelope: InboundEnvelope) -> GatewayRunTarget:
        scope = self.scope_factory(route, envelope)
        if not isinstance(scope, ExecutionScope):
            raise TypeError("gateway scope factory must return ExecutionScope")
        record = self.definitions.get(
            scope,
            scope.agent_id,
            scope.agent_version,
        )
        if record.status is not DefinitionStatus.PUBLISHED:
            raise ValueError("gateway agent definition is not published")
        return GatewayRunTarget(
            scope=scope,
            definition_id=record.definition.agent_id,
            definition_version=record.definition.version,
            definition_digest=record.definition.digest,
        )


class AsyncPublishedDefinitionGatewayResolver:
    """Native async published-definition resolver."""

    def __init__(
        self,
        definitions: AsyncAgentDefinitionStore,
        scope_factory: AsyncScopeFactory,
    ) -> None:
        self.definitions = definitions
        self.scope_factory = scope_factory

    async def resolve(
        self,
        route: Any,
        envelope: InboundEnvelope,
    ) -> GatewayRunTarget:
        scope = await self.scope_factory(route, envelope)
        if not isinstance(scope, ExecutionScope):
            raise TypeError("gateway scope factory must return ExecutionScope")
        record = await self.definitions.get(
            scope,
            scope.agent_id,
            scope.agent_version,
        )
        if record.status is not DefinitionStatus.PUBLISHED:
            raise ValueError("gateway agent definition is not published")
        return GatewayRunTarget(
            scope=scope,
            definition_id=record.definition.agent_id,
            definition_version=record.definition.version,
            definition_digest=record.definition.digest,
        )


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
    "AsyncPublishedDefinitionGatewayResolver",
    "DurableGatewayRunSubmitter",
    "PublishedDefinitionGatewayResolver",
]
