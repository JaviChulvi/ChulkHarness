"""Provider-neutral media processing contract for Telegram."""

from __future__ import annotations

from typing import Protocol

from chulk.telegram.client import TelegramAttachment


class TelegramMediaProcessor(Protocol):
    """Convert bounded media bytes into model-ready text."""

    def process(
        self,
        attachment: TelegramAttachment,
        data: bytes,
        *,
        instruction: str,
    ) -> str: ...


__all__ = ["TelegramMediaProcessor"]
