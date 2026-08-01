from __future__ import annotations

import pytest

from chulk.telegram.client import TelegramAttachment
from chulk.telegram.media import TelegramMediaError, validate_attachment


@pytest.mark.parametrize(
    ("file_name", "reported_mime", "expected_mime", "expected_kind"),
    [
        ("IMG_1234.HEIC", "application/octet-stream", "image/heic", "image"),
        ("photo.heif", "application/octet-stream", "image/heif", "image"),
        ("Voice Memo.m4a", "application/octet-stream", "audio/mp4", "audio"),
        ("recording.caf", "application/octet-stream", "audio/x-caf", "audio"),
        ("Live Photo.mov", "application/octet-stream", "video/quicktime", "video"),
        ("contact.vcf", "application/octet-stream", "text/vcard", "document"),
        ("calendar.ics", "application/octet-stream", "text/calendar", "document"),
        ("notes.rtf", "application/octet-stream", "application/rtf", "document"),
        ("scan.pdf", "application/pdf", "application/pdf", "document"),
    ],
)
def test_validate_attachment_normalizes_common_apple_formats(
    file_name: str,
    reported_mime: str,
    expected_mime: str,
    expected_kind: str,
) -> None:
    attachment = validate_attachment(
        TelegramAttachment("id", "document", reported_mime, file_name)
    )

    assert attachment.mime_type == expected_mime
    assert attachment.kind == expected_kind


@pytest.mark.parametrize(
    ("file_name", "product"),
    [
        ("proposal.pages", "Pages"),
        ("budget.numbers", "Numbers"),
        ("slides.key", "Keynote"),
    ],
)
def test_validate_attachment_gives_iwork_export_guidance(
    file_name: str,
    product: str,
) -> None:
    with pytest.raises(TelegramMediaError, match=rf"{product}.*Export.*PDF"):
        validate_attachment(
            TelegramAttachment(
                "id",
                "document",
                "application/octet-stream",
                file_name,
            )
        )


def test_validate_attachment_rejects_unknown_binary_files() -> None:
    with pytest.raises(TelegramMediaError, match="Unsupported attachment type"):
        validate_attachment(
            TelegramAttachment(
                "id",
                "document",
                "application/octet-stream",
                "archive.zip",
            )
        )
