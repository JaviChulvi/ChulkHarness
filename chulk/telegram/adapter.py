"""Telegram transport normalized behind the shared channel adapter contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    InboundEnvelope,
    InboundPart,
    MediaPart,
    MediaReference,
    OutboundEnvelope,
    SQLiteGatewayLedger,
    TextPart,
    TrustLevel,
)
from chulk.telegram.client import TelegramClient, TelegramUpdate
from chulk.telegram.config import TelegramConfig


ADAPTER_LEASE_SECONDS = 120
ADAPTER_LEASE_RENEW_SECONDS = 40.0


class TelegramChannelAdapter:
    """Long-poll and deliver Telegram events without owning agent execution."""

    name = "telegram"

    def __init__(
        self,
        *,
        client: TelegramClient,
        config: TelegramConfig,
        ledger: SQLiteGatewayLedger,
        account_id: str = "primary",
        defer_cursor: bool = False,
    ) -> None:
        self.client = client
        self.config = config
        self.account_id = account_id
        self.ledger = ledger
        self.defer_cursor = defer_cursor
        self._instance_token: str | None = None
        self._pending_cursor: int | None = None
        self._renewal_task: asyncio.Task[None] | None = None

    async def receive(self) -> AsyncIterator[InboundEnvelope]:
        await self.start()
        while True:
            for envelope in await self.poll_once():
                yield envelope

    async def start(self) -> None:
        if self._instance_token is not None:
            return
        status = await asyncio.to_thread(
            self.ledger.start_adapter,
            self.name,
            self.account_id,
            lease_seconds=ADAPTER_LEASE_SECONDS,
        )
        assert status.instance_token is not None
        self._instance_token = status.instance_token
        self._renewal_task = asyncio.create_task(self._renew_lease(status.instance_token))

    async def poll_once(self) -> tuple[InboundEnvelope, ...]:
        await self.start()
        status = await asyncio.to_thread(
            self.ledger.adapter_status,
            self.name,
            self.account_id,
        )
        offset = int(status.cursor) if status is not None and status.cursor else None
        updates = await asyncio.to_thread(
            self.client.get_updates,
            offset=offset,
            timeout_seconds=self.config.poll_timeout_seconds,
        )
        envelopes = [self.normalize(update) for update in updates]
        envelopes.extend(
            self.unsupported(update_id, destination_id)
            for update_id, destination_id in getattr(
                self.client,
                "ignored_updates",
                (),
            )
        )
        return tuple(sorted(envelopes, key=lambda item: int(item.event_id)))

    def normalize(self, update: TelegramUpdate) -> InboundEnvelope:
        allowed = update.user_id in self.config.allowed_user_ids
        parts: list[InboundPart] = []
        if update.text.strip():
            parts.append(TextPart(update.text.strip()))
        if update.attachment is not None:
            parts.append(
                MediaPart(
                    MediaReference(
                        content_ref=f"telegram:{update.attachment.file_id}",
                        content_type=update.attachment.mime_type,
                        size_bytes=0,
                        file_name=update.attachment.file_name,
                    ),
                    caption=update.text.strip() or None,
                )
            )
        if not parts:
            parts.append(TextPart("Unsupported empty Telegram message"))
        return InboundEnvelope(
            event_id=str(update.update_id),
            idempotency_key=(
                f"telegram:{self.account_id}:update:{update.update_id}"
            ),
            identity=ChannelIdentity(
                self.name,
                self.account_id,
                str(update.user_id),
            ),
            destination_id=str(update.chat_id),
            parts=tuple(parts),
            scope=(
                ChannelScope.DIRECT
                if update.chat_type == "private"
                else ChannelScope.GROUP
            ),
            authentication=(
                AuthenticationState.AUTHENTICATED
                if allowed
                else AuthenticationState.UNAUTHENTICATED
            ),
            trust=TrustLevel.TRUSTED if allowed else TrustLevel.UNTRUSTED,
            extensions={
                "update_id": update.update_id,
                "chat_type": update.chat_type,
                "attachment": (
                    {
                        "file_id": update.attachment.file_id,
                        "kind": update.attachment.kind,
                        "mime_type": update.attachment.mime_type,
                        "file_name": update.attachment.file_name,
                    }
                    if update.attachment is not None
                    else None
                ),
            },
        )

    def unsupported(self, update_id: int, destination_id: str) -> InboundEnvelope:
        return InboundEnvelope(
            event_id=str(update_id),
            idempotency_key=f"telegram:{self.account_id}:update:{update_id}",
            identity=ChannelIdentity(self.name, self.account_id, "_unsupported"),
            destination_id=destination_id,
            parts=(TextPart("Unsupported Telegram update"),),
            scope=ChannelScope.DIRECT,
            authentication=AuthenticationState.UNKNOWN,
            trust=TrustLevel.UNTRUSTED,
            extensions={"update_id": update_id, "unsupported": True},
        )

    async def deliver(self, envelope: OutboundEnvelope) -> DeliveryReceipt:
        if envelope.attachments:
            return DeliveryReceipt(
                envelope_id=envelope.envelope_id,
                state=DeliveryState.FAILED,
                attempt=1,
                error_code="unsupported_attachment",
                error_message="Telegram attachment delivery is not configured",
            )
        assert envelope.text is not None
        await asyncio.to_thread(
            self.client.send_message,
            int(envelope.target.destination_id),
            envelope.text,
        )
        return DeliveryReceipt(
            envelope_id=envelope.envelope_id,
            state=DeliveryState.DELIVERED,
            attempt=1,
            checkpoint=str(envelope.sequence + 1),
        )

    async def acknowledge(self, envelope: InboundEnvelope) -> None:
        if self._instance_token is None:
            raise RuntimeError("Telegram adapter is not running")
        cursor = int(envelope.event_id) + 1
        if self.defer_cursor:
            self._pending_cursor = max(self._pending_cursor or 0, cursor)
            return
        await self._save_cursor(cursor)

    async def commit_acknowledgements(self) -> None:
        cursor = self._pending_cursor
        if cursor is None:
            return
        await self._save_cursor(cursor)
        self._pending_cursor = None

    async def commit_cursor(self, cursor: int) -> None:
        if cursor < 0:
            raise ValueError("cursor cannot be negative")
        await self._save_cursor(cursor)

    async def _save_cursor(self, cursor: int) -> None:
        if self._instance_token is None:
            raise RuntimeError("Telegram adapter is not running")
        saved = await asyncio.to_thread(
            self.ledger.save_cursor,
            self.name,
            self.account_id,
            str(cursor),
            instance_token=self._instance_token,
        )
        if not saved:
            raise RuntimeError("Telegram adapter lost its cursor lease")

    async def close(self) -> None:
        token = self._instance_token
        self._instance_token = None
        renewal = self._renewal_task
        self._renewal_task = None
        if renewal is not None:
            renewal.cancel()
            with suppress(asyncio.CancelledError):
                await renewal
        if token is not None:
            await asyncio.to_thread(
                self.ledger.stop_adapter,
                self.name,
                self.account_id,
                token,
            )

    async def _renew_lease(self, instance_token: str) -> None:
        while True:
            await asyncio.sleep(ADAPTER_LEASE_RENEW_SECONDS)
            renewed = await asyncio.to_thread(
                self.ledger.renew_adapter,
                self.name,
                self.account_id,
                instance_token,
                lease_seconds=ADAPTER_LEASE_SECONDS,
            )
            if not renewed:
                return


__all__ = ["TelegramChannelAdapter"]
