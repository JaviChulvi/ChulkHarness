"""Command-line entry point for the Telegram adapter."""

from __future__ import annotations

import asyncio
import logging

from chulk.config import ConfigValueError, load_config
from chulk.telegram.bot import TelegramAgentBot
from chulk.telegram.client import TelegramClient
from chulk.telegram.config import TelegramConfigError, load_telegram_config


def main() -> int:
    """Load environment configuration and run the Telegram bot."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config()
        telegram_config = load_telegram_config(env_file=config.project_root / ".env")
    except (ConfigValueError, TelegramConfigError) as exc:
        logging.error("configuration error: %s", exc)
        return 2

    bot = TelegramAgentBot(
        config=config,
        telegram_config=telegram_config,
        client=TelegramClient(telegram_config.bot_token),
    )
    logging.info(
        "starting Telegram agent with provider=%s model=%s",
        config.llm_provider,
        config.model,
    )
    try:
        asyncio.run(bot.run_forever())
    except KeyboardInterrupt:
        logging.info("Telegram agent stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
