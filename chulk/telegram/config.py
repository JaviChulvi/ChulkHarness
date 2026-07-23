"""Environment-only configuration for the Telegram adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class TelegramConfigError(ValueError):
    """Raised when Telegram environment configuration is missing or invalid."""


@dataclass(frozen=True)
class TelegramConfig:
    """Validated Telegram bot configuration."""

    bot_token: str
    allowed_user_ids: frozenset[int]
    poll_timeout_seconds: int = 30
    retry_delay_seconds: float = 2.0
    tavily_api_key: str | None = None
    web_search_max_results: int = 5
    timezone: str = "UTC"
    scheduler_poll_seconds: float = 5.0
    scheduling_enabled: bool = False


def load_telegram_config(
    environ: Mapping[str, str] | None = None,
    *,
    env_file: Path | None = None,
) -> TelegramConfig:
    """Load Telegram credentials and policy exclusively from environment values."""
    process_env = dict(os.environ if environ is None else environ)
    env = {**_parse_env_file(env_file), **process_env}
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
        tavily_api_key=(env.get("CHULK_TAVILY_API_KEY") or "").strip() or None,
        web_search_max_results=_bounded_int(
            env,
            "CHULK_WEB_SEARCH_MAX_RESULTS",
            5,
            minimum=1,
            maximum=10,
        ),
        timezone=_timezone(env.get("CHULK_TELEGRAM_TIMEZONE") or "UTC"),
        scheduler_poll_seconds=_positive_float(
            env,
            "CHULK_TELEGRAM_SCHEDULER_POLL_SECONDS",
            5.0,
        ),
        scheduling_enabled=_boolean(
            env,
            "CHULK_TELEGRAM_SCHEDULING_ENABLED",
            False,
        ),
    )


def _timezone(value: str) -> str:
    clean = value.strip()
    try:
        ZoneInfo(clean)
    except ZoneInfoNotFoundError as exc:
        raise TelegramConfigError("CHULK_TELEGRAM_TIMEZONE must be a valid IANA timezone") from exc
    return clean


def _parse_env_file(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


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


def _boolean(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise TelegramConfigError(f"{key} must be true or false")


def _bounded_int(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = _positive_int(env, key, default)
    if not minimum <= value <= maximum:
        raise TelegramConfigError(f"{key} must be between {minimum} and {maximum}")
    return value
