"""Telegram adapter for remotely interacting with a Chulk agent."""

from chulk.telegram.client import (
    TelegramAttachment,
    TelegramClient,
    TelegramError,
    TelegramUpdate,
)
from chulk.telegram.bot import TelegramAgentBot
from chulk.telegram.adapter import TelegramChannelAdapter
from chulk.telegram.config import TelegramConfig, TelegramConfigError, load_telegram_config

__all__ = [
    "TelegramClient",
    "TelegramAgentBot",
    "TelegramChannelAdapter",
    "TelegramAttachment",
    "TelegramConfig",
    "TelegramConfigError",
    "TelegramError",
    "TelegramUpdate",
    "load_telegram_config",
]
