"""Command-line entry point for the Telegram adapter."""

from __future__ import annotations

import asyncio
import logging

from chulk.config import ConfigValueError, load_config
from chulk.llm.providers.gemini_media import GeminiMediaProcessor
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

    media_processor = None
    if config.llm_provider == "gemini":
        media_processor = GeminiMediaProcessor(
            model=config.model,
            api_key=config.gemini_api_key,
        )
    bot = TelegramAgentBot(
        config=config,
        telegram_config=telegram_config,
        client=TelegramClient(telegram_config.bot_token),
        media_processor=media_processor,
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
