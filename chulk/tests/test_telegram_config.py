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
        }
    )

    assert config.bot_token == "secret-token"
    assert config.allowed_user_ids == frozenset({123, 456})
    assert config.poll_timeout_seconds == 20
    assert config.retry_delay_seconds == 0.5


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
