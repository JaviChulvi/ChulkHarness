"""Telegram normalization and shared adapter contract tests."""

from __future__ import annotations

import pytest

from chulk.gateway import (
    ChannelAdapter,
    DeliveryTarget,
    MediaReference,
    OutboundEnvelope,
    SQLiteGatewayLedger,
)
from chulk.telegram.adapter import TelegramChannelAdapter
from chulk.telegram.client import TelegramAttachment, TelegramUpdate
from chulk.telegram.config import TelegramConfig


class FakeClient:
    def __init__(self) -> None:
        self.next_offset = 8
        self.ignored_updates = ((6, "9"),)
        self.offsets: list[int | None] = []
        self.sent: list[tuple[int, str]] = []
        self.attachments: list[tuple[int, bytes, str, str, str | None]] = []

    def get_updates(self, *, offset: int | None, timeout_seconds: int):
        self.offsets.append(offset)
        assert timeout_seconds == 1
        return (
            TelegramUpdate(
                update_id=5,
                chat_id=9,
                user_id=7,
                text="caption",
                chat_type="private",
                attachment=TelegramAttachment(
                    file_id="file-1",
                    kind="document",
                    mime_type="application/pdf",
                ),
            ),
        )

    def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))

    def send_attachment(
        self,
        chat_id: int,
        data: bytes,
        *,
        mime_type: str,
        file_name: str,
        caption: str | None,
    ) -> None:
        self.attachments.append((chat_id, data, mime_type, file_name, caption))


@pytest.mark.asyncio
async def test_telegram_adapter_normalizes_auth_media_cursor_and_delivery(tmp_path) -> None:
    client = FakeClient()
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    adapter = TelegramChannelAdapter(
        client=client,  # type: ignore[arg-type]
        config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
            poll_timeout_seconds=1,
        ),
        ledger=ledger,
    )

    assert isinstance(adapter, ChannelAdapter)
    envelopes = await adapter.poll_once()
    assert [item.event_id for item in envelopes] == ["5", "6"]
    assert envelopes[0].identity.principal_id == "7"
    assert envelopes[0].parts[1].media.content_ref == "telegram:file-1"
    assert envelopes[1].extensions["unsupported"] is True

    await adapter.acknowledge(envelopes[0])
    assert ledger.adapter_status("telegram", "primary").cursor == "6"
    receipt = await adapter.deliver(
        OutboundEnvelope(
            profile_id="default",
            conversation_id="conversation",
            target=DeliveryTarget("telegram", "primary", "9"),
            text="answer",
        )
    )
    assert receipt.state.value == "delivered"
    assert client.sent == [(9, "answer")]
    await adapter.close()
    assert ledger.adapter_status("telegram", "primary").state == "stopped"


@pytest.mark.asyncio
async def test_telegram_attachment_delivery_requires_approval_and_resolves_content(
    tmp_path,
) -> None:
    client = FakeClient()
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    adapter = TelegramChannelAdapter(
        client=client,  # type: ignore[arg-type]
        config=TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
        ),
        ledger=ledger,
        content_resolver=lambda profile, ref, limit: b"payload",
    )
    base = {
        "profile_id": "default",
        "conversation_id": "conversation",
        "target": DeliveryTarget("telegram", "primary", "9"),
        "text": "generated",
        "attachments": (
            MediaReference(
                "content:" + "a" * 32,
                "application/pdf",
                7,
                "report.pdf",
            ),
        ),
    }

    denied = await adapter.deliver(OutboundEnvelope(**base))
    delivered = await adapter.deliver(
        OutboundEnvelope(
            **base,
            extensions={"media_delivery_approved": True},
        )
    )

    assert denied.error_code == "approval_required"
    assert delivered.state.value == "delivered"
    assert client.attachments == [
        (9, b"payload", "application/pdf", "report.pdf", "generated")
    ]
