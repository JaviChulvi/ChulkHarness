from __future__ import annotations

import logging

import pytest

import chulk.telegram.main as telegram_main
from chulk.config import load_config
from chulk.llm import LLMConfigurationError
from chulk.telegram.config import TelegramConfig


def test_media_configuration_failure_returns_sanitized_startup_exit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "gemini",
            "CHULK_MODEL": "gemini-test",
            "CHULK_GEMINI_API_KEY": "fake",
        }
    )
    monkeypatch.setattr(telegram_main, "load_config", lambda: config)
    monkeypatch.setattr(
        telegram_main,
        "load_telegram_config",
        lambda **_kwargs: TelegramConfig(
            bot_token="fake",
            allowed_user_ids=frozenset({7}),
        ),
    )

    def fail_media(**_kwargs: object) -> object:
        raise LLMConfigurationError("media package unavailable")

    monkeypatch.setattr(telegram_main, "GeminiMediaProcessor", fail_media)

    with caplog.at_level(logging.ERROR):
        result = telegram_main.main()

    assert result == 2
    assert "configuration error: media package unavailable" in caplog.text
