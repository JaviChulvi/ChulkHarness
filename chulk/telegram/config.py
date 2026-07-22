"""Environment-only configuration for the Telegram adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os


class TelegramConfigError(ValueError):
    """Raised when Telegram environment configuration is missing or invalid."""


@dataclass(frozen=True)
class TelegramConfig:
    """Validated Telegram bot configuration."""

    bot_token: str
    allowed_user_ids: frozenset[int]
    poll_timeout_seconds: int = 30
    retry_delay_seconds: float = 2.0


def load_telegram_config(
    environ: Mapping[str, str] | None = None,
) -> TelegramConfig:
    """Load Telegram credentials and policy exclusively from environment values."""
    env = os.environ if environ is None else environ
    token = (env.get("CHULK_TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise TelegramConfigError("CHULK_TELEGRAM_BOT_TOKEN is required")

    raw_user_ids = env.get("CHULK_TELEGRAM_ALLOWED_USER_IDS") or ""
    allowed_user_ids = _parse_user_ids(raw_user_ids)
    if not allowed_user_ids:
        raise TelegramConfigError(
            "CHULK_TELEGRAM_ALLOWED_USER_IDS must contain at least one numeric user id"
        )

    return TelegramConfig(
        bot_token=token,
        allowed_user_ids=allowed_user_ids,
        poll_timeout_seconds=_positive_int(
            env,
            "CHULK_TELEGRAM_POLL_TIMEOUT_SECONDS",
            30,
        ),
        retry_delay_seconds=_positive_float(
            env,
            "CHULK_TELEGRAM_RETRY_DELAY_SECONDS",
            2.0,
        ),
    )


def _parse_user_ids(value: str) -> frozenset[int]:
    parsed: set[int] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            user_id = int(item)
        except ValueError as exc:
            raise TelegramConfigError(
                "CHULK_TELEGRAM_ALLOWED_USER_IDS must be comma-separated numeric ids"
            ) from exc
        if user_id <= 0:
            raise TelegramConfigError(
                "CHULK_TELEGRAM_ALLOWED_USER_IDS must contain positive numeric ids"
            )
        parsed.add(user_id)
    return frozenset(parsed)


def _positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise TelegramConfigError(f"{key} must be an integer") from exc
    if parsed <= 0:
        raise TelegramConfigError(f"{key} must be greater than zero")
    return parsed


def _positive_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise TelegramConfigError(f"{key} must be a number") from exc
    if parsed <= 0:
        raise TelegramConfigError(f"{key} must be greater than zero")
    return parsed
