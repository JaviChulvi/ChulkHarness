"""Channel-neutral command names, parsing, and help text."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChannelCommand:
    """One normalized slash command received through a channel."""

    name: str
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class ChannelCommandSpec:
    """One command that keeps the same meaning across channel clients."""

    name: str
    description: str
    usage: str


SHARED_CHANNEL_COMMANDS: tuple[ChannelCommandSpec, ...] = (
    ChannelCommandSpec("new", "Start a new conversation", "/new"),
    ChannelCommandSpec("status", "Show the active runtime", "/status"),
    ChannelCommandSpec("stop", "Stop work in the current conversation", "/stop"),
    ChannelCommandSpec("model", "Show the active model profile", "/model"),
    ChannelCommandSpec("skills", "List available skills", "/skills"),
    ChannelCommandSpec("memory", "Show memory scope and status", "/memory"),
    ChannelCommandSpec("agents", "Show delegated agent activity", "/agents"),
    ChannelCommandSpec("jobs", "List scheduled work for this channel", "/jobs"),
)


def parse_channel_command(text: str) -> ChannelCommand | None:
    """Parse a slash command, including Telegram-style ``/name@bot`` forms."""
    normalized = text.strip()
    if not normalized.startswith("/"):
        return None
    raw_name, separator, arguments = normalized.partition(" ")
    name = raw_name[1:].split("@", 1)[0].strip().lower()
    if not name:
        return None
    return ChannelCommand(name=name, arguments=arguments.strip() if separator else "")


def shared_command_spec(name: str) -> ChannelCommandSpec | None:
    clean_name = name.removeprefix("/").strip().lower()
    return next(
        (item for item in SHARED_CHANNEL_COMMANDS if item.name == clean_name),
        None,
    )


def shared_command_help(
    *,
    additional: tuple[ChannelCommandSpec, ...] = (),
) -> str:
    """Render concise help from the shared registry."""
    specs = (*SHARED_CHANNEL_COMMANDS, *additional)
    lines = ["Send a message to talk with the Chulk agent.", ""]
    lines.extend(f"{item.usage} — {item.description}" for item in specs)
    lines.append("/help — Show this help")
    return "\n".join(lines)


__all__ = [
    "ChannelCommand",
    "ChannelCommandSpec",
    "SHARED_CHANNEL_COMMANDS",
    "parse_channel_command",
    "shared_command_help",
    "shared_command_spec",
]
