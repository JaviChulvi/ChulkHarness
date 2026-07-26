"""Provider-neutral, persistence-safe media input models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias


class MediaKind(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class ContentTrust(StrEnum):
    OWNER = "owner"
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class RetentionPolicy(StrEnum):
    TURN = "turn"
    SESSION = "session"
    PROFILE = "profile"


@dataclass(frozen=True, slots=True)
class ContentRef:
    """Opaque identifier for bytes owned by one profile content store."""

    id: str

    def __post_init__(self) -> None:
        clean = self.id.strip()
        if not clean or len(clean) > 128 or "\x00" in clean:
            raise ValueError("content ref must be a non-empty opaque identifier")
        object.__setattr__(self, "id", clean)

    def __str__(self) -> str:
        return self.id


@dataclass(frozen=True, slots=True)
class MediaItem:
    """Safe metadata for content whose bytes remain outside prompts and traces."""

    kind: MediaKind
    mime_type: str
    byte_length: int
    content_ref: ContentRef
    sha256: str
    provenance: str
    trust: ContentTrust = ContentTrust.UNTRUSTED
    retention: RetentionPolicy = RetentionPolicy.SESSION
    file_name: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    expires_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MediaKind(self.kind))
        mime = _required(self.mime_type, "mime_type").lower().split(";", 1)[0]
        object.__setattr__(self, "mime_type", mime)
        if (
            isinstance(self.byte_length, bool)
            or not isinstance(self.byte_length, int)
            or self.byte_length < 0
        ):
            raise ValueError("byte_length must be a non-negative integer")
        digest = self.sha256.strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("sha256 must be a hexadecimal SHA-256 digest")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "provenance", _required(self.provenance, "provenance"))
        object.__setattr__(self, "trust", ContentTrust(self.trust))
        object.__setattr__(self, "retention", RetentionPolicy(self.retention))
        if self.file_name is not None:
            clean_name = self.file_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
            object.__setattr__(self, "file_name", clean_name[:255] or None)
        _parse_timestamp(self.created_at, "created_at")
        if self.expires_at is not None:
            _parse_timestamp(self.expires_at, "expires_at")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_safe_metadata(dict(self.metadata))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "mime_type": self.mime_type,
            "byte_length": self.byte_length,
            "content_ref": self.content_ref.id,
            "sha256": self.sha256,
            "provenance": self.provenance,
            "trust": self.trust.value,
            "retention": self.retention.value,
            "file_name": self.file_name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MediaItem":
        return cls(
            kind=MediaKind(str(value["kind"])),
            mime_type=str(value["mime_type"]),
            byte_length=int(value["byte_length"]),
            content_ref=ContentRef(str(value["content_ref"])),
            sha256=str(value["sha256"]),
            provenance=str(value["provenance"]),
            trust=ContentTrust(str(value.get("trust", ContentTrust.UNTRUSTED.value))),
            retention=RetentionPolicy(
                str(value.get("retention", RetentionPolicy.SESSION.value))
            ),
            file_name=(
                str(value["file_name"]) if value.get("file_name") is not None else None
            ),
            created_at=str(value["created_at"]),
            expires_at=(
                str(value["expires_at"])
                if value.get("expires_at") is not None
                else None
            ),
            metadata=(
                dict(value["metadata"])
                if isinstance(value.get("metadata"), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class TextInputPart:
    text: str
    external_content: bool = False
    kind: str = field(default="text", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _required(self.text, "text"))

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "character_length": len(self.text),
            "external_content": self.external_content,
        }


@dataclass(frozen=True, slots=True)
class MediaInputPart:
    media: MediaItem
    caption: str | None = None
    kind: str = field(default="media", init=False)

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.media.kind.value,
            "caption": self.caption.strip() if self.caption else None,
            "media": self.media.to_dict(),
        }


InputPart: TypeAlias = TextInputPart | MediaInputPart


@dataclass(frozen=True, slots=True)
class UserInput:
    """One typed user turn with a bounded text-only projection."""

    parts: tuple[InputPart, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("user input must contain at least one part")
        object.__setattr__(self, "parts", tuple(self.parts))

    @classmethod
    def text(cls, value: str) -> "UserInput":
        return cls((TextInputPart(value),))

    def textual_projection(self, *, max_chars: int = 12_000) -> str:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        blocks: list[str] = []
        for part in self.parts:
            if isinstance(part, TextInputPart):
                blocks.append(part.text)
                continue
            media = part.media
            label = media.file_name or media.kind.value
            descriptor = (
                f"[{media.kind.value}: {label}; {media.mime_type}; "
                f"{media.byte_length} bytes; content_ref={media.content_ref.id}]"
            )
            blocks.append(
                f"{part.caption.strip()}\n{descriptor}"
                if part.caption and part.caption.strip()
                else descriptor
            )
        projection = "\n\n".join(blocks).strip()
        if not projection:
            projection = "[typed input]"
        if len(projection) <= max_chars:
            return projection
        marker = "\n...[typed input projection truncated]"
        return projection[: max(0, max_chars - len(marker))] + marker

    def safe_metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(part.safe_metadata() for part in self.parts)

    @property
    def media_parts(self) -> tuple[MediaInputPart, ...]:
        return tuple(
            part for part in self.parts if isinstance(part, MediaInputPart)
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Provider-neutral request retaining typed parts beside text messages."""

    messages: tuple[Mapping[str, str], ...]
    user_input: UserInput | None = None
    purpose: str = "agent_action"

    def text_messages(self) -> list[dict[str, str]]:
        return [dict(message) for message in self.messages]


class UnsupportedMediaError(RuntimeError):
    """A typed media input has no allowed provider or processor path."""


def _required(value: str, field_name: str) -> str:
    clean = value.strip()
    if not clean or "\x00" in clean:
        raise ValueError(f"{field_name} cannot be empty")
    return clean


def _parse_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in value.items():
        clean_key = str(key).strip()
        if not clean_key:
            continue
        if item is None or isinstance(item, (str, int, float, bool)):
            safe[clean_key] = item
    return safe


__all__ = [
    "ContentRef",
    "ContentTrust",
    "InputPart",
    "MediaInputPart",
    "MediaItem",
    "MediaKind",
    "ModelRequest",
    "RetentionPolicy",
    "TextInputPart",
    "UnsupportedMediaError",
    "UserInput",
]
