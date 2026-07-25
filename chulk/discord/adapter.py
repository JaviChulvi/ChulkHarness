"""Discord messages normalized behind the shared gateway protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Protocol

from chulk.discord.client import DiscordMessage
from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    InboundEnvelope,
    OutboundEnvelope,
    TextPart,
    TrustLevel,
)


DISCORD_MESSAGE_LIMIT = 2_000


class DiscordTransport(Protocol):
    def receive(self) -> AsyncIterator[DiscordMessage]: ...

    async def send_message(self, channel_id: str, text: str) -> str: ...

    async def close(self) -> None: ...


class DiscordChannelAdapter:
    """Keep Discord-specific events outside agent orchestration."""

    name = "discord"

    def __init__(
        self,
        transport: DiscordTransport,
        *,
        account_id: str = "primary",
    ) -> None:
        account = account_id.strip()
        if not account or "\x00" in account:
            raise ValueError("Discord account_id is required")
        self.transport = transport
        self.account_id = account

    async def receive(self) -> AsyncIterator[InboundEnvelope]:
        async for message in self.transport.receive():
            if message.author_is_bot or not message.content.strip():
                continue
            yield self.normalize(message)

    def normalize(self, message: DiscordMessage) -> InboundEnvelope:
        return InboundEnvelope(
            event_id=message.message_id,
            idempotency_key=(
                f"discord:{self.account_id}:message:{message.message_id}"
            ),
            identity=ChannelIdentity(
                self.name,
                self.account_id,
                message.author_id,
            ),
            destination_id=message.channel_id,
            thread_id=message.thread_id,
            parts=(TextPart(message.content.strip()),),
            scope=(
                ChannelScope.GROUP
                if message.guild_id is not None
                else ChannelScope.DIRECT
            ),
            authentication=AuthenticationState.AUTHENTICATED,
            trust=TrustLevel.UNTRUSTED,
            extensions={
                "guild_id": message.guild_id,
                "message_id": message.message_id,
            },
        )

    async def deliver(self, envelope: OutboundEnvelope) -> DeliveryReceipt:
        text = envelope.text or ""
        if envelope.attachments:
            return DeliveryReceipt(
                envelope_id=envelope.envelope_id,
                state=DeliveryState.FAILED,
                attempt=1,
                error_code="unsupported_attachment",
                error_message="Discord attachment delivery is not configured",
            )
        if not text or len(text) > DISCORD_MESSAGE_LIMIT:
            return DeliveryReceipt(
                envelope_id=envelope.envelope_id,
                state=DeliveryState.FAILED,
                attempt=1,
                error_code="invalid_message_length",
                error_message="Discord delivery units must be 1 to 2000 characters",
            )
        message_id = await self.transport.send_message(
            envelope.target.thread_id or envelope.target.destination_id,
            text,
        )
        return DeliveryReceipt(
            envelope_id=envelope.envelope_id,
            state=DeliveryState.DELIVERED,
            attempt=1,
            adapter_message_id=message_id,
            checkpoint=str(envelope.sequence + 1),
        )

    async def acknowledge(self, _envelope: InboundEnvelope) -> None:
        return None

    async def close(self) -> None:
        await self.transport.close()


def split_discord_text(
    text: str,
    *,
    limit: int = DISCORD_MESSAGE_LIMIT,
) -> tuple[str, ...]:
    """Split text into retry-safe Discord delivery units."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not text:
        return ()
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining)
            break
        boundary = remaining.rfind("\n", 0, limit + 1)
        if boundary < max(1, limit // 2):
            boundary = remaining.rfind(" ", 0, limit + 1)
        if boundary < 1:
            boundary = limit
        part = remaining[:boundary].rstrip()
        if not part:
            part = remaining[:limit]
            boundary = limit
        parts.append(part)
        remaining = remaining[boundary:].lstrip()
    return tuple(parts)


def split_discord_envelope(
    envelope: OutboundEnvelope,
) -> tuple[OutboundEnvelope, ...]:
    """Expand one logical response into durable Discord-sized records."""
    parts = split_discord_text(envelope.text or "")
    return tuple(
        replace(
            envelope,
            envelope_id=f"{envelope.envelope_id}:{index}",
            text=part,
            sequence=index,
            final=index == len(parts) - 1,
        )
        for index, part in enumerate(parts)
    )


__all__ = [
    "DISCORD_MESSAGE_LIMIT",
    "DiscordChannelAdapter",
    "DiscordTransport",
    "split_discord_envelope",
    "split_discord_text",
]
