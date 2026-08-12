"""Environment-only configuration for the optional Discord adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

from chulk.config import _parse_dotenv


class DiscordConfigError(ValueError):
    """Raised when Discord credentials or limits are invalid."""


@dataclass(frozen=True, slots=True)
class DiscordConfig:
    bot_token: str
    account_id: str = "primary"
    max_pending: int = 1_000

    def __post_init__(self) -> None:
        if not self.bot_token.strip():
            raise DiscordConfigError("Discord bot token is required")
        account = self.account_id.strip()
        if not account or "\x00" in account or len(account) > 128:
            raise DiscordConfigError("Discord account id is invalid")
        if (
            isinstance(self.max_pending, bool)
            or not isinstance(self.max_pending, int)
            or self.max_pending < 1
        ):
            raise DiscordConfigError("Discord max pending must be positive")
        object.__setattr__(self, "bot_token", self.bot_token.strip())
        object.__setattr__(self, "account_id", account)


def load_discord_config(
    environ: Mapping[str, str] | None = None,
    *,
    env_file: Path | None = None,
) -> DiscordConfig:
    process_env = dict(os.environ if environ is None else environ)
    env = {**_parse_dotenv(env_file), **process_env}
    token = (env.get("CHULK_DISCORD_BOT_TOKEN") or "").strip()
    if not token:
        raise DiscordConfigError("CHULK_DISCORD_BOT_TOKEN is required")
    raw_pending = (env.get("CHULK_DISCORD_MAX_PENDING") or "1000").strip()
    try:
        max_pending = int(raw_pending)
    except ValueError as exc:
        raise DiscordConfigError(
            "CHULK_DISCORD_MAX_PENDING must be an integer"
        ) from exc
    return DiscordConfig(
        bot_token=token,
        account_id=(env.get("CHULK_DISCORD_ACCOUNT_ID") or "primary"),
        max_pending=max_pending,
    )


__all__ = [
    "DiscordConfig",
    "DiscordConfigError",
    "load_discord_config",
]
