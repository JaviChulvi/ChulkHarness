"""Tests for channel-neutral gateway models and adapter protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from chulk.gateway import (
    AuthenticationState,
    ChannelAdapter,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    DeliveryTarget,
    InboundEnvelope,
    MediaPart,
    MediaReference,
    OutboundEnvelope,
    ReactionPart,
    ReplyPart,
    TextPart,
    TrustLevel,
)


def test_inbound_envelope_carries_identity_security_and_external_content():
    media = MediaReference(
        content_ref="content:photo",
        content_type="image/png",
        size_bytes=128,
        sha256="abc",
    )
    envelope = InboundEnvelope(
        event_id="event-1",
        idempotency_key="telegram:account:update:1",
        identity=ChannelIdentity("telegram", "primary", "user-42"),
        destination_id="chat-9",
        thread_id="thread-2",
        parts=(
            TextPart("hello"),
            MediaPart(media),
            ReplyPart("event-0", "previous"),
            ReactionPart("👍", "event-0"),
        ),
        scope=ChannelScope.GROUP,
        authentication=AuthenticationState.AUTHENTICATED,
        trust=TrustLevel.TRUSTED,
        extensions={"update_id": 1},
    )

    assert envelope.identity.adapter == "telegram"
    assert envelope.parts[1].media.external_content
    assert envelope.external_content
    assert envelope.extensions["update_id"] == 1
    with pytest.raises(TypeError):
        envelope.extensions["update_id"] = 2  # type: ignore[index]


def test_outbound_delivery_supports_splitting_attachments_and_checkpoints():
    target = DeliveryTarget("telegram", "primary", "chat-9", thread_id="thread-2")
    attachment = MediaReference("content:report", "application/pdf", 500)
    envelope = OutboundEnvelope(
        profile_id="work",
        conversation_id="conversation-1",
        target=target,
        text="part one",
        attachments=(attachment,),
        checkpoint="rendered",
        sequence=0,
        final=False,
    )
    receipt = DeliveryReceipt(
        envelope_id=envelope.envelope_id,
        state=DeliveryState.RETRYABLE,
        attempt=1,
        checkpoint="rendered",
        retry_after_seconds=2,
        error_message="rate limited",
    )

    assert envelope.target.thread_id == "thread-2"
    assert envelope.sequence == 0
    assert not envelope.final
    assert receipt.state is DeliveryState.RETRYABLE
    assert receipt.checkpoint == envelope.checkpoint


def test_gateway_models_fail_closed_on_missing_parts_or_delivery_content():
    identity = ChannelIdentity("local", "cli", "owner")

    with pytest.raises(ValueError, match="at least one part"):
        InboundEnvelope(
            event_id="event",
            idempotency_key="key",
            identity=identity,
            destination_id="terminal",
            parts=(),
            scope=ChannelScope.LOCAL_OPERATOR,
            authentication=AuthenticationState.AUTHENTICATED,
            trust=TrustLevel.OWNER,
        )

    with pytest.raises(ValueError, match="text or an attachment"):
        OutboundEnvelope(
            profile_id="default",
            conversation_id="conversation",
            target=DeliveryTarget("local", "cli", "terminal"),
        )


class FakeAdapter:
    name = "fake"
    account_id = "test"

    async def receive(self) -> AsyncIterator[InboundEnvelope]:
        if False:
            yield

    async def deliver(self, envelope: OutboundEnvelope) -> DeliveryReceipt:
        return DeliveryReceipt(
            envelope_id=envelope.envelope_id,
            state=DeliveryState.DELIVERED,
            attempt=1,
        )

    async def acknowledge(self, envelope: InboundEnvelope) -> None:
        return None

    async def close(self) -> None:
        return None


def test_channel_adapter_protocol_is_runtime_checkable():
    assert isinstance(FakeAdapter(), ChannelAdapter)


def test_gateway_enums_normalize_strings_and_extensions_are_redacted():
    envelope = InboundEnvelope(
        event_id="event",
        idempotency_key="key",
        identity=ChannelIdentity("local", "cli", "owner"),
        destination_id="terminal",
        parts=(TextPart("hello"),),
        scope="local_operator",  # type: ignore[arg-type]
        authentication="authenticated",  # type: ignore[arg-type]
        trust="owner",  # type: ignore[arg-type]
        extensions={"authorization": "Bearer secret"},
    )

    assert envelope.scope is ChannelScope.LOCAL_OPERATOR
    assert envelope.authentication is AuthenticationState.AUTHENTICATED
    assert envelope.trust is TrustLevel.OWNER
    assert envelope.extensions["authorization"] == "[redacted]"
