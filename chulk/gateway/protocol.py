"""Protocol implemented by channel adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from chulk.gateway.models import DeliveryReceipt, InboundEnvelope, OutboundEnvelope


@runtime_checkable
class ChannelAdapter(Protocol):
    """Network transport boundary; adapters do not own agent orchestration."""

    @property
    def name(self) -> str:
        """Return the stable adapter type."""

    @property
    def account_id(self) -> str:
        """Return the configured account identity."""

    def receive(self) -> AsyncIterator[InboundEnvelope]:
        """Yield normalized inbound envelopes."""
        ...

    async def deliver(self, envelope: OutboundEnvelope) -> DeliveryReceipt:
        """Attempt one outbound delivery and return its checkpoint."""
        ...

    async def acknowledge(self, envelope: InboundEnvelope) -> None:
        """Acknowledge durable ingestion after the gateway owns the event."""
        ...

    async def close(self) -> None:
        """Release adapter resources."""
        ...


__all__ = ["ChannelAdapter"]
