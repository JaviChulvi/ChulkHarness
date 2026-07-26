"""Profile-owned bounded content storage with integrity and retention checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from chulk.media.models import (
    ContentRef,
    ContentTrust,
    MediaItem,
    MediaKind,
    RetentionPolicy,
)
from chulk.storage import initialize_sqlite_database, sqlite_connection


DEFAULT_MAX_CONTENT_BYTES = 25 * 1024 * 1024
DEFAULT_RETENTION_SECONDS = 7 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class MediaLimits:
    """Bounds applied before a processor receives potentially expensive media."""

    max_decompressed_bytes: int = 100 * 1024 * 1024
    max_duration_seconds: float = 15 * 60
    max_pages: int = 500

    def __post_init__(self) -> None:
        if self.max_decompressed_bytes < 1:
            raise ValueError("max_decompressed_bytes must be positive")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        if self.max_pages < 1:
            raise ValueError("max_pages must be positive")


class ContentNotFoundError(KeyError):
    pass


class ContentOwnershipError(PermissionError):
    pass


class ContentIntegrityError(RuntimeError):
    pass


class ContentLimitError(ValueError):
    pass


class ContentStore:
    """Store media bytes outside project files and text persistence."""

    def __init__(
        self,
        db_path: Path | str,
        content_dir: Path | str,
        *,
        profile_id: str,
        max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES,
        limits: MediaLimits | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.content_dir = Path(content_dir)
        self.profile_id = profile_id.strip()
        if not self.profile_id:
            raise ValueError("content store profile_id cannot be empty")
        if max_content_bytes < 1:
            raise ValueError("max_content_bytes must be positive")
        self.max_content_bytes = max_content_bytes
        self.limits = limits or MediaLimits()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        initialize_sqlite_database(self.db_path)
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def put(
        self,
        data: bytes,
        *,
        kind: MediaKind | str,
        mime_type: str,
        provenance: str,
        trust: ContentTrust | str = ContentTrust.UNTRUSTED,
        retention: RetentionPolicy | str = RetentionPolicy.SESSION,
        retention_seconds: int | None = DEFAULT_RETENTION_SECONDS,
        file_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> MediaItem:
        if not isinstance(data, bytes):
            raise TypeError("content data must be bytes")
        if len(data) > self.max_content_bytes:
            raise ContentLimitError(
                f"content exceeds the {self.max_content_bytes}-byte store limit"
            )
        normalized_kind = MediaKind(kind)
        normalized_mime = _normalize_mime(mime_type)
        _validate_kind_mime(normalized_kind, normalized_mime)
        _validate_signature(normalized_mime, data)
        if retention_seconds is not None and retention_seconds < 1:
            raise ValueError("retention_seconds must be positive or None")
        now = self._now()
        expires_at = (
            now + timedelta(seconds=retention_seconds)
            if retention_seconds is not None
            else None
        )
        content_id = uuid4().hex
        ref = ContentRef(f"content:{content_id}")
        digest = hashlib.sha256(data).hexdigest()
        safe_metadata = metadata or {}
        self._validate_expansion_metadata(safe_metadata)
        item = MediaItem(
            kind=normalized_kind,
            mime_type=normalized_mime,
            byte_length=len(data),
            content_ref=ref,
            sha256=digest,
            provenance=provenance,
            trust=ContentTrust(trust),
            retention=RetentionPolicy(retention),
            file_name=file_name,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat() if expires_at is not None else None,
            metadata=safe_metadata,
        )
        target = self._content_path(ref)
        temporary = target.with_suffix(".pending")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO content_items (
                        content_ref, profile_id, kind, mime_type, byte_length,
                        sha256, provenance, trust, retention, file_name,
                        created_at, expires_at, metadata_json, state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    """,
                    (
                        ref.id,
                        self.profile_id,
                        item.kind.value,
                        item.mime_type,
                        item.byte_length,
                        item.sha256,
                        item.provenance,
                        item.trust.value,
                        item.retention.value,
                        item.file_name,
                        item.created_at,
                        item.expires_at,
                        json.dumps(dict(item.metadata), sort_keys=True),
                    ),
                )
        except BaseException:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        return item

    def get(self, ref: ContentRef | str, *, profile_id: str | None = None) -> MediaItem:
        owner = (profile_id or self.profile_id).strip()
        row = self._row(ref)
        self._check_owner(row, owner)
        if str(row["state"]) != "active":
            raise ContentNotFoundError(str(ref))
        item = _item_from_row(row)
        if item.expires_at is not None and datetime.fromisoformat(item.expires_at) <= self._now():
            raise ContentNotFoundError(str(ref))
        return item

    def read(
        self,
        ref: ContentRef | str,
        *,
        profile_id: str | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        item = self.get(ref, profile_id=profile_id)
        limit = self.max_content_bytes if max_bytes is None else max_bytes
        if limit < 1:
            raise ValueError("max_bytes must be positive")
        if item.byte_length > limit:
            raise ContentLimitError(f"content exceeds the {limit}-byte read limit")
        path = self._content_path(item.content_ref)
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise ContentIntegrityError("content bytes are missing") from exc
        if len(data) != item.byte_length or hashlib.sha256(data).hexdigest() != item.sha256:
            raise ContentIntegrityError("content integrity verification failed")
        return data

    def delete(
        self,
        ref: ContentRef | str,
        *,
        profile_id: str | None = None,
    ) -> bool:
        owner = (profile_id or self.profile_id).strip()
        row = self._row(ref)
        self._check_owner(row, owner)
        if str(row["state"]) == "deleted":
            return False
        content_ref = ContentRef(str(row["content_ref"]))
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE content_items
                SET state = 'deleted', deleted_at = ?
                WHERE content_ref = ? AND profile_id = ? AND state = 'active'
                """,
                (self._now().isoformat(), content_ref.id, owner),
            )
        self._content_path(content_ref).unlink(missing_ok=True)
        return True

    def sweep_expired(self, *, limit: int = 500) -> tuple[str, ...]:
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        now = self._now().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT content_ref
                FROM content_items
                WHERE profile_id = ? AND state = 'active'
                  AND expires_at IS NOT NULL AND expires_at <= ?
                ORDER BY expires_at, content_ref
                LIMIT ?
                """,
                (self.profile_id, now, limit),
            ).fetchall()
        deleted: list[str] = []
        for row in rows:
            ref = str(row["content_ref"])
            if self.delete(ref):
                deleted.append(ref)
        return tuple(deleted)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS content_items (
                    content_ref TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    byte_length INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    trust TEXT NOT NULL,
                    retention TEXT NOT NULL,
                    file_name TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    deleted_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    state TEXT NOT NULL CHECK(state IN ('active', 'deleted'))
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS content_items_profile_expiry
                ON content_items(profile_id, state, expires_at)
                """
            )

    def _validate_expansion_metadata(self, metadata: dict[str, object]) -> None:
        decompressed = _optional_number(metadata.get("decompressed_bytes"))
        duration = _optional_number(metadata.get("duration_seconds"))
        pages = _optional_number(metadata.get("page_count"))
        if (
            decompressed is not None
            and decompressed > self.limits.max_decompressed_bytes
        ):
            raise ContentLimitError("content exceeds the decompressed byte limit")
        if duration is not None and duration > self.limits.max_duration_seconds:
            raise ContentLimitError("content exceeds the duration limit")
        if pages is not None and pages > self.limits.max_pages:
            raise ContentLimitError("content exceeds the page limit")

    def _row(self, ref: ContentRef | str) -> sqlite3.Row:
        normalized = ref if isinstance(ref, ContentRef) else ContentRef(str(ref))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM content_items WHERE content_ref = ?",
                (normalized.id,),
            ).fetchone()
        if row is None:
            raise ContentNotFoundError(normalized.id)
        return row

    @staticmethod
    def _check_owner(row: sqlite3.Row, profile_id: str) -> None:
        if str(row["profile_id"]) != profile_id:
            raise ContentOwnershipError("content is owned by another profile")

    def _content_path(self, ref: ContentRef) -> Path:
        content_id = ref.id.removeprefix("content:")
        if (
            len(content_id) != 32
            or any(char not in "0123456789abcdef" for char in content_id)
        ):
            raise ValueError("content ref is not owned by this store")
        return self.content_dir / content_id

    def _connect(self):
        return sqlite_connection(self.db_path)

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("content store clock must be timezone-aware")
        return value.astimezone(timezone.utc)


def _item_from_row(row: sqlite3.Row) -> MediaItem:
    try:
        metadata = json.loads(str(row["metadata_json"]))
    except (TypeError, ValueError):
        metadata = {}
    return MediaItem(
        kind=MediaKind(str(row["kind"])),
        mime_type=str(row["mime_type"]),
        byte_length=int(row["byte_length"]),
        content_ref=ContentRef(str(row["content_ref"])),
        sha256=str(row["sha256"]),
        provenance=str(row["provenance"]),
        trust=ContentTrust(str(row["trust"])),
        retention=RetentionPolicy(str(row["retention"])),
        file_name=str(row["file_name"]) if row["file_name"] is not None else None,
        created_at=str(row["created_at"]),
        expires_at=(
            str(row["expires_at"]) if row["expires_at"] is not None else None
        ),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _normalize_mime(value: str) -> str:
    clean = value.strip().lower().split(";", 1)[0]
    if "/" not in clean or "\x00" in clean:
        raise ValueError("mime_type must be a valid media type")
    return clean


def _validate_kind_mime(kind: MediaKind, mime_type: str) -> None:
    expected = {
        MediaKind.IMAGE: "image/",
        MediaKind.AUDIO: "audio/",
        MediaKind.VIDEO: "video/",
    }.get(kind)
    if expected is not None and not mime_type.startswith(expected):
        raise ValueError(f"{kind.value} content requires a {expected} MIME type")


def _validate_signature(mime_type: str, data: bytes) -> None:
    signatures = {
        "image/jpeg": (b"\xff\xd8\xff",),
        "image/png": (b"\x89PNG\r\n\x1a\n",),
        "image/webp": (b"RIFF",),
        "application/pdf": (b"%PDF-",),
    }
    expected = signatures.get(mime_type)
    if expected is not None and data and not any(data.startswith(item) for item in expected):
        raise ValueError("content signature does not match declared MIME type")
    if mime_type == "image/webp" and data and (
        len(data) < 12 or data[8:12] != b"WEBP"
    ):
        raise ValueError("content signature does not match declared MIME type")


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("media expansion metadata must use non-negative numbers")
    return float(value)


__all__ = [
    "ContentIntegrityError",
    "ContentLimitError",
    "ContentNotFoundError",
    "ContentOwnershipError",
    "ContentStore",
    "DEFAULT_MAX_CONTENT_BYTES",
    "DEFAULT_RETENTION_SECONDS",
    "MediaLimits",
]
