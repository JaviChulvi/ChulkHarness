"""Provider-neutral media processing contract for Telegram."""

from __future__ import annotations

from typing import Protocol

from chulk.core.context import TurnContextSection
from chulk.telegram.client import TelegramAttachment


class TelegramMediaError(RuntimeError):
    """Sanitized media failure safe to return to an authenticated user."""


def attachment_context(
    attachment: TelegramAttachment,
    extracted_text: str,
) -> TurnContextSection:
    """Wrap provider output as explicitly untrusted, turn-scoped evidence."""
    label = attachment.file_name or attachment.kind
    return TurnContextSection(
        id=f"telegram-attachment-{attachment.file_id[:16]}",
        title=f"Telegram attachment: {label}",
        source="telegram_attachment",
        content=extracted_text,
        metadata={
            "trusted": False,
            "attachment_kind": attachment.kind,
            "mime_type": attachment.mime_type,
        },
    )


class TelegramMediaProcessor(Protocol):
    """Convert bounded media bytes into model-ready text."""

    def process(
        self,
        attachment: TelegramAttachment,
        data: bytes,
        *,
        instruction: str,
    ) -> str: ...


__all__ = ["TelegramMediaError", "TelegramMediaProcessor", "attachment_context"]
