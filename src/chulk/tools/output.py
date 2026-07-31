"""Tool output preview and truncation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib


@dataclass(frozen=True)
class TextPreview:
    """Bounded text plus metadata about any omitted middle content."""

    text: str
    original_length: int
    returned_length: int
    truncated: bool
    omitted_length: int
    sha256: str

    def to_metadata(self) -> dict:
        return asdict(self)


def preview_text(text: str, max_chars: int) -> TextPreview:
    """Return a head/tail preview so both beginnings and endings survive truncation."""
    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero")

    original_length = len(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if original_length <= max_chars:
        return TextPreview(
            text=text,
            original_length=original_length,
            returned_length=original_length,
            truncated=False,
            omitted_length=0,
            sha256=digest,
        )

    marker = "\n[truncated output: middle omitted]\n"
    if max_chars <= len(marker) + 2:
        preview = text[:max_chars]
        omitted_length = original_length - len(preview)
    else:
        content_budget = max_chars - len(marker)
        head_chars = content_budget // 2
        tail_chars = content_budget - head_chars
        preview = text[:head_chars] + marker + text[-tail_chars:]
        omitted_length = original_length - head_chars - tail_chars

    return TextPreview(
        text=preview,
        original_length=original_length,
        returned_length=len(preview),
        truncated=True,
        omitted_length=max(0, omitted_length),
        sha256=digest,
    )
