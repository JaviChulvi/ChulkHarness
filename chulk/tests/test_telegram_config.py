from __future__ import annotations

from pathlib import Path

import pytest

from chulk.telegram.config import TelegramConfigError, load_telegram_config


def test_load_telegram_config_requires_token_and_allowlist() -> None:
    with pytest.raises(TelegramConfigError, match="BOT_TOKEN"):
        load_telegram_config({})
    with pytest.raises(TelegramConfigError, match="ALLOWED_USER_IDS"):
        load_telegram_config({"CHULK_TELEGRAM_BOT_TOKEN": "secret"})


def test_load_telegram_config_parses_environment_values() -> None:
    config = load_telegram_config(
        {
            "CHULK_TELEGRAM_BOT_TOKEN": " secret-token ",
            "CHULK_TELEGRAM_ALLOWED_USER_IDS": "123, 456,123",
            "CHULK_TELEGRAM_POLL_TIMEOUT_SECONDS": "20",
            "CHULK_TELEGRAM_RETRY_DELAY_SECONDS": "0.5",
            "CHULK_TAVILY_API_KEY": " tavily-secret ",
            "CHULK_WEB_SEARCH_MAX_RESULTS": "3",
            "CHULK_TELEGRAM_TIMEZONE": "Europe/Madrid",
            "CHULK_TELEGRAM_SCHEDULER_POLL_SECONDS": "2.5",
            "CHULK_TELEGRAM_SCHEDULING_ENABLED": "true",
        }
    )

    assert config.bot_token == "secret-token"
    assert config.allowed_user_ids == frozenset({123, 456})
    assert config.poll_timeout_seconds == 20
    assert config.retry_delay_seconds == 0.5
    assert config.tavily_api_key == "tavily-secret"
    assert config.web_search_max_results == 3
    assert config.timezone == "Europe/Madrid"
    assert config.scheduler_poll_seconds == 2.5
    assert config.scheduling_enabled is True
    assert config.max_attachment_bytes == 10 * 1024 * 1024


def test_load_telegram_config_reads_env_file_with_environment_precedence(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CHULK_TELEGRAM_BOT_TOKEN=file-token\n"
        "CHULK_TELEGRAM_ALLOWED_USER_IDS=123\n",
        encoding="utf-8",
    )

    config = load_telegram_config(
        {"CHULK_TELEGRAM_BOT_TOKEN": "process-token"},
        env_file=env_file,
    )

    assert config.bot_token == "process-token"
    assert config.allowed_user_ids == frozenset({123})


@pytest.mark.parametrize("value", ["abc", "1,-2", "1,2.5"])
def test_load_telegram_config_rejects_invalid_user_ids(value: str) -> None:
    with pytest.raises(TelegramConfigError, match="numeric ids"):
        load_telegram_config(
            {
                "CHULK_TELEGRAM_BOT_TOKEN": "secret",
                "CHULK_TELEGRAM_ALLOWED_USER_IDS": value,
            }
        )


def test_load_telegram_config_rejects_excessive_search_results() -> None:
    with pytest.raises(TelegramConfigError, match="WEB_SEARCH_MAX_RESULTS"):
        load_telegram_config(
            {
                "CHULK_TELEGRAM_BOT_TOKEN": "secret",
                "CHULK_TELEGRAM_ALLOWED_USER_IDS": "1",
                "CHULK_WEB_SEARCH_MAX_RESULTS": "11",
            }
        )


def test_load_telegram_config_rejects_unknown_timezone() -> None:
    with pytest.raises(TelegramConfigError, match="TIMEZONE"):
        load_telegram_config(
            {
                "CHULK_TELEGRAM_BOT_TOKEN": "secret",
                "CHULK_TELEGRAM_ALLOWED_USER_IDS": "1",
                "CHULK_TELEGRAM_TIMEZONE": "Mars/Olympus",
            }
        )


def test_scheduling_is_disabled_by_default_and_rejects_invalid_boolean() -> None:
    config = load_telegram_config(
        {
            "CHULK_TELEGRAM_BOT_TOKEN": "secret",
            "CHULK_TELEGRAM_ALLOWED_USER_IDS": "1",
        }
    )
    assert config.scheduling_enabled is False

    with pytest.raises(TelegramConfigError, match="SCHEDULING_ENABLED"):
        load_telegram_config(
            {
                "CHULK_TELEGRAM_BOT_TOKEN": "secret",
                "CHULK_TELEGRAM_ALLOWED_USER_IDS": "1",
                "CHULK_TELEGRAM_SCHEDULING_ENABLED": "sometimes",
            }
        )


def test_load_telegram_config_validates_attachment_bound() -> None:
    with pytest.raises(TelegramConfigError, match="CHULK_TELEGRAM_MAX_ATTACHMENT_BYTES"):
        load_telegram_config(
            {
                "CHULK_TELEGRAM_BOT_TOKEN": "secret",
                "CHULK_TELEGRAM_ALLOWED_USER_IDS": "1",
                "CHULK_TELEGRAM_MAX_ATTACHMENT_BYTES": str(21 * 1024 * 1024),
            }
        )
