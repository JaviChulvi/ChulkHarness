"""Provider-neutral media processing contract for Telegram."""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePath
from typing import Protocol

from chulk.core.context import TurnContextSection
from chulk.media import (
    MediaItem,
    MediaKind,
    MediaTransformResult,
    ProcessorCapability,
    TransformKind,
)
from chulk.telegram.client import TelegramAttachment


class TelegramMediaError(RuntimeError):
    """Sanitized media failure safe to return to an authenticated user."""


SUPPORTED_MEDIA_MIME_TYPES = frozenset(
    {
        # Images, including the default iPhone camera formats.
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
        # Audio and voice notes, including common Apple containers.
        "audio/aac",
        "audio/aiff",
        "audio/flac",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "audio/ogg",
        "audio/wav",
        "audio/x-caf",
        # Short videos, including iPhone QuickTime/MOV.
        "video/mp4",
        "video/mov",
        "video/quicktime",
        "video/mpeg",
        "video/webm",
        "video/3gpp",
        # Documents that Gemini can consume inline as PDF or bounded text.
        "application/pdf",
        "application/json",
        "application/rtf",
        "application/xml",
        "text/calendar",
        "text/csv",
        "text/html",
        "text/markdown",
        "text/plain",
        "text/rtf",
        "text/vcard",
        "text/xml",
    }
)

_MIME_BY_EXTENSION = {
    ".aac": "audio/aac",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".caf": "audio/x-caf",
    ".csv": "text/csv",
    ".flac": "audio/flac",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".htm": "text/html",
    ".html": "text/html",
    ".ics": "text/calendar",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".json": "application/json",
    ".m4a": "audio/mp4",
    ".md": "text/markdown",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".ogg": "audio/ogg",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".rtf": "application/rtf",
    ".txt": "text/plain",
    ".vcf": "text/vcard",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".webp": "image/webp",
    ".xml": "application/xml",
}
_APPLE_IWORK_EXTENSIONS = frozenset({".key", ".numbers", ".pages"})


def validate_attachment(attachment: TelegramAttachment) -> TelegramAttachment:
    """Normalize known extensions and reject unsupported binary containers."""
    suffix = (
        PurePath(attachment.file_name).suffix.lower()
        if attachment.file_name is not None
        else ""
    )
    if suffix in _APPLE_IWORK_EXTENSIONS:
        product = {
            ".key": "Keynote",
            ".numbers": "Numbers",
            ".pages": "Pages",
        }[suffix]
        raise TelegramMediaError(
            f"{product} files are recognized but cannot be read safely yet. "
            "Export the file as PDF from your Mac or iPhone and send the PDF instead."
        )
    mime_type = attachment.mime_type.lower().split(";", 1)[0].strip()
    if mime_type in {"application/octet-stream", "binary/octet-stream", ""}:
        mime_type = _MIME_BY_EXTENSION.get(suffix, mime_type)
    if mime_type not in SUPPORTED_MEDIA_MIME_TYPES:
        raise TelegramMediaError(
            f"Unsupported attachment type: {suffix or mime_type or 'unknown'}. "
            "Send a PDF, text file, supported image, audio recording, or short video."
        )
    kind = attachment.kind
    if mime_type.startswith("image/"):
        kind = "image"
    elif mime_type.startswith("audio/"):
        kind = "audio"
    elif mime_type.startswith("video/"):
        kind = "video"
    else:
        kind = "document"
    return replace(attachment, kind=kind, mime_type=mime_type)


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


class TelegramMediaProcessorAdapter:
    """Expose the legacy channel processor through the shared media registry."""

    def __init__(
        self,
        processor: TelegramMediaProcessor,
        *,
        provider: str,
        max_bytes: int,
    ) -> None:
        self.processor = processor
        self.capability = ProcessorCapability(
            name=f"{provider}_telegram_media",
            transforms=frozenset(
                {
                    TransformKind.TRANSCRIPTION,
                    TransformKind.EXTRACTION,
                    TransformKind.UNDERSTANDING,
                }
            ),
            media_kinds=frozenset(MediaKind),
            max_bytes=max_bytes,
            provider=provider,
            network_access=True,
        )

    def process(
        self,
        item: MediaItem,
        data: bytes,
        *,
        instruction: str,
        transform: TransformKind,
    ) -> MediaTransformResult:
        attachment = TelegramAttachment(
            file_id=item.content_ref.id,
            kind=item.kind.value,
            mime_type=item.mime_type,
            file_name=item.file_name,
        )
        text = self.processor.process(
            attachment,
            data,
            instruction=instruction,
        )
        return MediaTransformResult(
            transform=transform,
            processor=self.capability.name,
            text=text,
        )


__all__ = [
    "SUPPORTED_MEDIA_MIME_TYPES",
    "TelegramMediaError",
    "TelegramMediaProcessor",
    "TelegramMediaProcessorAdapter",
    "attachment_context",
    "validate_attachment",
]
