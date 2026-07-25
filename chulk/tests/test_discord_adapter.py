"""Discord adapter, configuration, and gateway-boundary tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from chulk.discord import (
    DISCORD_MESSAGE_LIMIT,
    DiscordChannelAdapter,
    DiscordConfigError,
    DiscordMessage,
    load_discord_config,
    split_discord_envelope,
    split_discord_text,
)
from chulk.gateway import (
    ChannelScope,
    DeliveryState,
    DeliveryTarget,
    GatewayRuntime,
    OutboundEnvelope,
    SQLiteGatewayLedger,
    SQLiteGatewayRouter,
    TextPart,
)


class FakeDiscordTransport:
    def __init__(self, messages: tuple[DiscordMessage, ...] = ()) -> None:
        self.messages = messages
        self.sent: list[tuple[str, str]] = []
        self.closed = False

    async def receive(self) -> AsyncIterator[DiscordMessage]:
        for message in self.messages:
            yield message

    async def send_message(self, channel_id: str, text: str) -> str:
        self.sent.append((channel_id, text))
        return f"sent-{len(self.sent)}"

    async def close(self) -> None:
        self.closed = True


def _message(
    content: str,
    *,
    message_id: str = "1",
    guild_id: str | None = None,
    thread_id: str | None = None,
    author_is_bot: bool = False,
) -> DiscordMessage:
    return DiscordMessage(
        message_id=message_id,
        author_id="user-7",
        channel_id="channel-9",
        thread_id=thread_id,
        guild_id=guild_id,
        content=content,
        author_is_bot=author_is_bot,
    )


def test_discord_config_is_environment_only_and_validated(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHULK_DISCORD_BOT_TOKEN=file-token\n"
        "CHULK_DISCORD_ACCOUNT_ID=team-bot\n",
        encoding="utf-8",
    )

    config = load_discord_config(
        {"CHULK_DISCORD_BOT_TOKEN": "process-token"},
        env_file=env_file,
    )

    assert config.bot_token == "process-token"
    assert config.account_id == "team-bot"
    with pytest.raises(DiscordConfigError, match="BOT_TOKEN"):
        load_discord_config({})
    with pytest.raises(DiscordConfigError, match="MAX_PENDING"):
        load_discord_config(
            {
                "CHULK_DISCORD_BOT_TOKEN": "token",
                "CHULK_DISCORD_MAX_PENDING": "many",
            }
        )


@pytest.mark.asyncio
async def test_discord_adapter_normalizes_direct_group_and_thread_messages() -> None:
    transport = FakeDiscordTransport(
        (
            _message("ignore", message_id="1", author_is_bot=True),
            _message("  hello  ", message_id="2"),
            _message(
                "/status",
                message_id="3",
                guild_id="guild-4",
                thread_id="thread-5",
            ),
        )
    )
    adapter = DiscordChannelAdapter(transport, account_id="primary")

    received = [item async for item in adapter.receive()]

    assert len(received) == 2
    assert received[0].scope is ChannelScope.DIRECT
    assert received[0].identity.principal_id == "user-7"
    assert received[0].idempotency_key == "discord:primary:message:2"
    assert received[0].parts == (TextPart("hello"),)
    assert received[1].scope is ChannelScope.GROUP
    assert received[1].destination_id == "channel-9"
    assert received[1].thread_id == "thread-5"


@pytest.mark.asyncio
async def test_discord_deliveries_are_bounded_and_target_threads() -> None:
    transport = FakeDiscordTransport()
    adapter = DiscordChannelAdapter(transport)
    envelope = OutboundEnvelope(
        profile_id="default",
        conversation_id="conversation",
        target=DeliveryTarget(
            "discord",
            "primary",
            "channel-9",
            thread_id="thread-5",
        ),
        text="answer",
    )

    delivered = await adapter.deliver(envelope)
    oversized = await adapter.deliver(
        OutboundEnvelope(
            profile_id="default",
            conversation_id="conversation",
            target=envelope.target,
            text="x" * (DISCORD_MESSAGE_LIMIT + 1),
        )
    )

    assert transport.sent == [("thread-5", "answer")]
    assert delivered.state is DeliveryState.DELIVERED
    assert delivered.adapter_message_id == "sent-1"
    assert oversized.state is DeliveryState.FAILED
    assert oversized.error_code == "invalid_message_length"


def test_discord_responses_split_before_the_durable_outbox() -> None:
    text = ("line " * 500) + "\n" + ("tail " * 300)
    envelope = OutboundEnvelope(
        profile_id="default",
        conversation_id="conversation",
        target=DeliveryTarget("discord", "primary", "channel-9"),
        text=text,
        envelope_id="response",
    )

    text_parts = split_discord_text(text)
    records = split_discord_envelope(envelope)

    assert "".join("".join(text_parts).split()) == "".join(text.split())
    assert all(0 < len(part) <= DISCORD_MESSAGE_LIMIT for part in text_parts)
    assert tuple(item.text for item in records) == text_parts
    assert tuple(item.envelope_id for item in records) == tuple(
        f"response:{index}" for index in range(len(records))
    )
    assert records[-1].final
    assert all(not item.final for item in records[:-1])


@pytest.mark.asyncio
async def test_discord_direct_message_consumes_pairing_without_agent_execution(
    tmp_path: Path,
) -> None:
    transport = FakeDiscordTransport()
    adapter = DiscordChannelAdapter(transport)
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    router = SQLiteGatewayRouter(tmp_path / "control.sqlite")
    challenge = router.create_pairing(
        adapter="discord",
        account_id="primary",
        profile_id="default",
        ttl_seconds=600,
    )
    executed = False

    async def execute(_profile_id, _envelope):
        nonlocal executed
        executed = True
        return ()

    runtime = GatewayRuntime(
        ledger=ledger,
        router=router,
        adapters=(adapter,),
        executor=execute,
    )
    envelope = adapter.normalize(_message(challenge.code))

    assert await runtime.accept(adapter, envelope) is False
    assert executed is False
    route = router.resolve(adapter.normalize(_message("hello", message_id="2")))
    assert route is not None
    assert route.profile_id == "default"
    await runtime.close()


def test_discord_adapter_does_not_import_core_or_provider_modules() -> None:
    package = Path(__file__).resolve().parents[1] / "discord"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package.glob("*.py"))
        if path.name != "main.py"
    )

    assert "chulk.core" not in source
    assert "chulk.llm" not in source
