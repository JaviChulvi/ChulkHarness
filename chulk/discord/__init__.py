"""Optional Discord channel adapter."""

from chulk.discord.adapter import (
    DISCORD_MESSAGE_LIMIT,
    DiscordChannelAdapter,
    DiscordTransport,
    split_discord_envelope,
    split_discord_text,
)
from chulk.discord.client import (
    DiscordDependencyError,
    DiscordMessage,
    DiscordPyTransport,
    DiscordTransportError,
)
from chulk.discord.config import (
    DiscordConfig,
    DiscordConfigError,
    load_discord_config,
)


__all__ = [
    "DISCORD_MESSAGE_LIMIT",
    "DiscordChannelAdapter",
    "DiscordConfig",
    "DiscordConfigError",
    "DiscordDependencyError",
    "DiscordMessage",
    "DiscordPyTransport",
    "DiscordTransport",
    "DiscordTransportError",
    "load_discord_config",
    "split_discord_envelope",
    "split_discord_text",
]
