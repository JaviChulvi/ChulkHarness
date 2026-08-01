"""Narrow discord.py transport boundary with sanitized failures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import importlib
from typing import Any


class DiscordDependencyError(RuntimeError):
    """Raised when the optional Discord dependency is unavailable."""


class DiscordTransportError(RuntimeError):
    """Raised when Discord transport work fails without leaking credentials."""


@dataclass(frozen=True, slots=True)
class DiscordMessage:
    message_id: str
    author_id: str
    channel_id: str
    content: str
    guild_id: str | None = None
    thread_id: str | None = None
    author_is_bot: bool = False


class DiscordPyTransport:
    """Expose the small client surface consumed by the channel adapter."""

    def __init__(self, token: str, *, max_pending: int = 1_000) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._token = token
        self._client: Any = None
        self._runner: asyncio.Task[None] | None = None
        self._messages: asyncio.Queue[DiscordMessage] = asyncio.Queue(
            maxsize=max_pending
        )

    async def receive(self) -> AsyncIterator[DiscordMessage]:
        client = self._build_client()
        self._runner = asyncio.create_task(client.start(self._token, reconnect=True))
        try:
            while True:
                message_task = asyncio.create_task(self._messages.get())
                done, _ = await asyncio.wait(
                    (message_task, self._runner),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._runner in done:
                    message_task.cancel()
                    await asyncio.gather(message_task, return_exceptions=True)
                    error = self._runner.exception()
                    if error is not None:
                        raise DiscordTransportError(
                            "Discord client stopped unexpectedly"
                        ) from error
                    return
                yield message_task.result()
        finally:
            await self.close()

    async def send_message(self, channel_id: str, text: str) -> str:
        client = self._client
        if client is None:
            raise DiscordTransportError("Discord client is not connected")
        try:
            channel = client.get_channel(int(channel_id))
            if channel is None:
                channel = await client.fetch_channel(int(channel_id))
            sent = await channel.send(text)
        except Exception as exc:
            raise DiscordTransportError("Discord message delivery failed") from exc
        return str(sent.id)

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed():
            await client.close()
        runner = self._runner
        self._runner = None
        if runner is not None and runner is not asyncio.current_task():
            await asyncio.gather(runner, return_exceptions=True)

    def _build_client(self):
        try:
            discord = importlib.import_module("discord")
        except ImportError as exc:
            raise DiscordDependencyError(
                "Discord dependencies are unavailable; "
                "install chulkharness[discord]"
            ) from exc
        intents = discord.Intents.default()
        intents.message_content = True
        transport = self

        class Client(discord.Client):  # type: ignore[name-defined,misc]
            async def on_message(self, message) -> None:
                parent_id = getattr(message.channel, "parent_id", None)
                guild = getattr(message, "guild", None)
                normalized = DiscordMessage(
                    message_id=str(message.id),
                    author_id=str(message.author.id),
                    channel_id=str(parent_id or message.channel.id),
                    thread_id=(
                        str(message.channel.id)
                        if parent_id is not None
                        else None
                    ),
                    guild_id=str(guild.id) if guild is not None else None,
                    content=str(message.content or ""),
                    author_is_bot=bool(getattr(message.author, "bot", False)),
                )
                await transport._messages.put(normalized)

        self._client = Client(intents=intents)
        return self._client


__all__ = [
    "DiscordDependencyError",
    "DiscordMessage",
    "DiscordPyTransport",
    "DiscordTransportError",
]
