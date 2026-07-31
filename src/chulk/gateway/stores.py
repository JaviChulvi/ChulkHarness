"""Host-implementable gateway storage and routing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from chulk.gateway.models import DeliveryReceipt, InboundEnvelope, OutboundEnvelope
from chulk.hosting.scope import ExecutionScope


@dataclass(frozen=True, slots=True)
class GatewayRunTarget:
    """Immutable hosted authority and published definition selected for an input."""

    scope: ExecutionScope
    definition_id: str
    definition_version: str
    definition_digest: str

    def __post_init__(self) -> None:
        for name in (
            "definition_id",
            "definition_version",
            "definition_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required")
        if self.scope.agent_id != self.definition_id:
            raise ValueError("execution scope agent_id must match the definition")
        if self.scope.agent_version != self.definition_version:
            raise ValueError("execution scope agent_version must match the definition")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.to_dict(),
            "definition_id": self.definition_id,
            "definition_version": self.definition_version,
            "definition_digest": self.definition_digest,
        }


@runtime_checkable
class GatewayRouter(Protocol):
    """Host routing boundary; implementations need not use SQLite."""

    def resolve(self, envelope: InboundEnvelope) -> Any | None: ...

    def consume_pairing(
        self,
        code: str,
        envelope: InboundEnvelope,
    ) -> Any | None: ...


@runtime_checkable
class AsyncGatewayRouter(Protocol):
    async def resolve(self, envelope: InboundEnvelope) -> Any | None: ...

    async def consume_pairing(
        self,
        code: str,
        envelope: InboundEnvelope,
    ) -> Any | None: ...


@runtime_checkable
class GatewayScopeResolver(Protocol):
    """Resolve authority and one exact published definition before submission."""

    def resolve(self, route: Any, envelope: InboundEnvelope) -> GatewayRunTarget: ...


@runtime_checkable
class AsyncGatewayScopeResolver(Protocol):
    async def resolve(
        self,
        route: Any,
        envelope: InboundEnvelope,
    ) -> GatewayRunTarget: ...


@runtime_checkable
class GatewayRunSubmitter(Protocol):
    """Idempotently submit one durable run before transport acknowledgement."""

    def submit(
        self,
        target: GatewayRunTarget,
        envelope: InboundEnvelope,
        *,
        idempotency_key: str,
    ) -> str: ...


@runtime_checkable
class AsyncGatewayRunSubmitter(Protocol):
    async def submit(
        self,
        target: GatewayRunTarget,
        envelope: InboundEnvelope,
        *,
        idempotency_key: str,
    ) -> str: ...


@runtime_checkable
class GatewayStore(Protocol):
    """Sync durable inbox/outbox boundary consumed by :class:`GatewayRuntime`."""

    def ingest(
        self,
        envelope: InboundEnvelope,
        *,
        profile_id: str,
        conversation_key: str | None = None,
        max_pending: int | None = None,
        run_target: GatewayRunTarget | None = None,
    ) -> Any: ...

    def ignore(
        self,
        envelope: InboundEnvelope,
        *,
        profile_id: str,
        reason: str,
        conversation_key: str | None = None,
    ) -> Any: ...

    def claim_execution(
        self,
        *,
        global_limit: int,
        profile_limit: int,
        profile_id: str | None = None,
        adapter_keys: tuple[tuple[str, str], ...] | None = None,
        queued_before: datetime | None = None,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> Any | None: ...

    def renew_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> bool: ...

    def complete_execution(
        self,
        inbox_id: str,
        execution_token: str,
        responses: tuple[OutboundEnvelope, ...],
    ) -> bool: ...

    def recover_expired_executions(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> tuple[Any, ...]: ...

    def get_inbox(self, inbox_id: str) -> Any | None: ...

    def request_cancellation(self, inbox_id: str) -> bool: ...

    def request_conversation_cancellation(
        self,
        *,
        profile_id: str,
        conversation_key: str,
        exclude_inbox_id: str,
    ) -> tuple[str, ...]: ...

    def mark_execution_cancelled(self, inbox_id: str, execution_token: str) -> bool: ...

    def quarantine_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        error: str,
    ) -> bool: ...

    def dead_letter_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        error: str,
    ) -> bool: ...

    def dead_letter_inbox(self, inbox_id: str, *, error: str) -> bool: ...

    def claim_delivery(
        self,
        *,
        adapter_keys: tuple[tuple[str, str], ...] | None = None,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> Any | None: ...

    def record_delivery(
        self,
        outbox_id: str,
        delivery_token: str,
        receipt: DeliveryReceipt,
        *,
        now: datetime | None = None,
    ) -> bool: ...

    def reconcile_delivery(
        self,
        outbox_id: str,
        receipt: DeliveryReceipt,
        *,
        now: datetime | None = None,
    ) -> bool: ...

    def list_reconciliation_required(
        self,
        *,
        limit: int = 100,
    ) -> tuple[Any, ...]: ...


@runtime_checkable
class AsyncGatewayStore(Protocol):
    """Native async equivalent of :class:`GatewayStore`."""

    async def ingest(
        self,
        envelope: InboundEnvelope,
        *,
        profile_id: str,
        conversation_key: str | None = None,
        max_pending: int | None = None,
        run_target: GatewayRunTarget | None = None,
    ) -> Any: ...

    async def ignore(
        self,
        envelope: InboundEnvelope,
        *,
        profile_id: str,
        reason: str,
        conversation_key: str | None = None,
    ) -> Any: ...

    async def claim_execution(
        self,
        *,
        global_limit: int,
        profile_limit: int,
        profile_id: str | None = None,
        adapter_keys: tuple[tuple[str, str], ...] | None = None,
        queued_before: datetime | None = None,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> Any | None: ...

    async def renew_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> bool: ...

    async def complete_execution(
        self,
        inbox_id: str,
        execution_token: str,
        responses: tuple[OutboundEnvelope, ...],
    ) -> bool: ...

    async def recover_expired_executions(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> tuple[Any, ...]: ...

    async def get_inbox(self, inbox_id: str) -> Any | None: ...

    async def request_cancellation(self, inbox_id: str) -> bool: ...

    async def request_conversation_cancellation(
        self,
        *,
        profile_id: str,
        conversation_key: str,
        exclude_inbox_id: str,
    ) -> tuple[str, ...]: ...

    async def mark_execution_cancelled(
        self,
        inbox_id: str,
        execution_token: str,
    ) -> bool: ...

    async def quarantine_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        error: str,
    ) -> bool: ...

    async def dead_letter_execution(
        self,
        inbox_id: str,
        execution_token: str,
        *,
        error: str,
    ) -> bool: ...

    async def dead_letter_inbox(self, inbox_id: str, *, error: str) -> bool: ...

    async def claim_delivery(
        self,
        *,
        adapter_keys: tuple[tuple[str, str], ...] | None = None,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> Any | None: ...

    async def record_delivery(
        self,
        outbox_id: str,
        delivery_token: str,
        receipt: DeliveryReceipt,
        *,
        now: datetime | None = None,
    ) -> bool: ...

    async def reconcile_delivery(
        self,
        outbox_id: str,
        receipt: DeliveryReceipt,
        *,
        now: datetime | None = None,
    ) -> bool: ...

    async def list_reconciliation_required(
        self,
        *,
        limit: int = 100,
    ) -> tuple[Any, ...]: ...


__all__ = [
    "AsyncGatewayRouter",
    "AsyncGatewayRunSubmitter",
    "AsyncGatewayScopeResolver",
    "AsyncGatewayStore",
    "GatewayRouter",
    "GatewayRunSubmitter",
    "GatewayRunTarget",
    "GatewayScopeResolver",
    "GatewayStore",
]
