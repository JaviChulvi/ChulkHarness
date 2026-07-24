"""Opaque, conversation-owned trace artifact storage and bounded reads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Literal
from uuid import uuid4

from chulk.errors import ErrorDetails, TraceError
from chulk.storage.private_files import (
    PRIVATE_DIRECTORY_MODE,
    PRIVATE_FILE_MODE,
    prepare_private_directory,
    write_private_text,
)


ArtifactReadMode = Literal["slice", "head", "tail", "head_tail"]
DEFAULT_ARTIFACT_READ_BYTES = 8_192
MAX_ARTIFACT_READ_BYTES = 65_536
MAX_ARTIFACT_FILE_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_LINE_BYTES = 65_536
_ARTIFACT_ID_PATTERN = re.compile(r"art_[0-9a-f]{32}\Z")


class ArtifactAccessError(TraceError, ValueError):
    """Raised when an artifact reference cannot be resolved safely."""

    def __init__(self, message: str, *, artifact_id: str | None = None) -> None:
        super().__init__(
            message,
            details=ErrorDetails(extensions={"artifact_id": artifact_id}),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    conversation_id: str
    filename: str
    label: str
    char_count: int
    byte_count: int
    sha256: str
    created_at: str

    def reference(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "char_count": self.char_count,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactRead:
    artifact_id: str
    mode: ArtifactReadMode
    content: str
    byte_count: int
    total_byte_count: int
    sha256: str
    ranges: tuple[tuple[int, int], ...]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ranges"] = [list(item) for item in self.ranges]
        return payload


class TraceArtifactStore:
    """Write and resolve artifacts inside one conversation-owned directory."""

    def __init__(
        self,
        traces_dir: Path | str,
        conversation_id: str,
        *,
        create: bool = False,
    ) -> None:
        self.traces_dir = Path(traces_dir)
        self.conversation_id = _validate_component(conversation_id)
        self.artifacts_dir = self.traces_dir / f"{self.conversation_id}_artifacts"
        self.manifest_path = self.artifacts_dir / "manifest.jsonl"
        if create:
            prepare_private_directory(self.traces_dir)

    def write(self, label: str, content: str) -> ArtifactRecord:
        """Persist one artifact and append its private manifest record."""
        prepare_private_directory(self.artifacts_dir)
        artifact_id = f"art_{uuid4().hex}"
        filename = f"{artifact_id}.txt"
        encoded = content.encode("utf-8")
        record = ArtifactRecord(
            artifact_id=artifact_id,
            conversation_id=self.conversation_id,
            filename=filename,
            label=_safe_label(label),
            char_count=len(content),
            byte_count=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        write_private_text(
            self.artifacts_dir / filename,
            content,
            overwrite=False,
        )
        write_private_text(
            self.manifest_path,
            json.dumps(record.to_dict(), sort_keys=True) + "\n",
            append=True,
        )
        return record

    def read(
        self,
        artifact_id: str,
        *,
        mode: ArtifactReadMode = "head_tail",
        offset: int = 0,
        max_bytes: int = DEFAULT_ARTIFACT_READ_BYTES,
        max_artifact_bytes: int = MAX_ARTIFACT_FILE_BYTES,
    ) -> ArtifactRead:
        """Validate ownership and integrity, then return one bounded byte view."""
        clean_id = _validate_artifact_id(artifact_id)
        if mode not in {"slice", "head", "tail", "head_tail"}:
            raise ArtifactAccessError("Unsupported artifact read mode", artifact_id=clean_id)
        if offset < 0:
            raise ArtifactAccessError("Artifact offset cannot be negative", artifact_id=clean_id)
        if not 1 <= max_bytes <= MAX_ARTIFACT_READ_BYTES:
            raise ArtifactAccessError(
                f"Artifact max_bytes must be between 1 and {MAX_ARTIFACT_READ_BYTES}",
                artifact_id=clean_id,
            )
        if max_artifact_bytes < 1:
            raise ArtifactAccessError(
                "Artifact maximum file size must be positive",
                artifact_id=clean_id,
            )

        record = self._find_record(clean_id)
        expected_filename = f"{clean_id}.txt"
        if (
            record.conversation_id != self.conversation_id
            or record.filename != expected_filename
        ):
            raise ArtifactAccessError(
                "Artifact ownership metadata is invalid",
                artifact_id=clean_id,
            )
        path = self.artifacts_dir / expected_filename
        descriptor = _open_regular_read_only(path, artifact_id=clean_id)
        try:
            size = os.fstat(descriptor).st_size
            if size != record.byte_count:
                raise ArtifactAccessError(
                    "Artifact size does not match its recorded metadata",
                    artifact_id=clean_id,
                )
            if size > max_artifact_bytes:
                raise ArtifactAccessError(
                    "Artifact exceeds the configured integrity-read bound",
                    artifact_id=clean_id,
                )
            digest = _hash_descriptor(descriptor)
            if digest != record.sha256:
                raise ArtifactAccessError(
                    "Artifact hash does not match its recorded metadata",
                    artifact_id=clean_id,
                )
            content, ranges = _read_bounded(
                descriptor,
                size=size,
                mode=mode,
                offset=offset,
                max_bytes=max_bytes,
            )
        finally:
            os.close(descriptor)
        return ArtifactRead(
            artifact_id=clean_id,
            mode=mode,
            content=content.decode("utf-8", errors="replace"),
            byte_count=sum(end - start for start, end in ranges),
            total_byte_count=size,
            sha256=digest,
            ranges=ranges,
            truncated=sum(end - start for start, end in ranges) < size,
        )

    def inventory(
        self,
        *,
        max_artifact_bytes: int = MAX_ARTIFACT_FILE_BYTES,
    ) -> list[dict[str, Any]]:
        """Return recorded artifacts and integrity status without their content."""
        if not self.artifacts_dir.exists():
            return []
        records = self._load_manifest_records(allow_missing=True)
        inventory: list[dict[str, Any]] = []
        recorded_ids = {record.artifact_id for record in records}
        for record in records:
            entry = {
                key: value
                for key, value in record.to_dict().items()
                if key != "filename"
            }
            if record.conversation_id != self.conversation_id:
                entry.update(
                    integrity="owner_mismatch",
                    integrity_error="recorded conversation does not own this artifact",
                )
            elif record.filename != f"{record.artifact_id}.txt":
                entry.update(
                    integrity="invalid_manifest",
                    integrity_error="recorded filename is not id-derived",
                )
            else:
                integrity, error = self._inspect_integrity(
                    record,
                    max_artifact_bytes=max_artifact_bytes,
                )
                entry["integrity"] = integrity
                entry["integrity_error"] = error
            inventory.append(entry)

        for candidate in sorted(self.artifacts_dir.iterdir(), key=lambda item: item.name):
            match = re.fullmatch(r"(art_[0-9a-f]{32})\.txt", candidate.name)
            if match is None or match.group(1) in recorded_ids:
                continue
            mode = candidate.lstat().st_mode
            inventory.append(
                {
                    "artifact_id": match.group(1),
                    "conversation_id": self.conversation_id,
                    "label": None,
                    "char_count": None,
                    "byte_count": candidate.stat().st_size if stat.S_ISREG(mode) else None,
                    "sha256": None,
                    "created_at": None,
                    "integrity": "unrecorded",
                    "integrity_error": "artifact file has no manifest record",
                }
            )
        return inventory

    def _find_record(self, artifact_id: str) -> ArtifactRecord:
        records = self._load_manifest_records()
        matches = [
            record for record in records if record.artifact_id == artifact_id
        ]
        if len(matches) != 1:
            reason = "not recorded" if not matches else "recorded more than once"
            raise ArtifactAccessError(
                f"Artifact reference is {reason} for this conversation",
                artifact_id=artifact_id,
            )
        return matches[0]

    def _load_manifest_records(
        self,
        *,
        allow_missing: bool = False,
    ) -> list[ArtifactRecord]:
        _validate_artifact_directory(
            self.artifacts_dir,
            artifact_id="manifest",
        )
        if allow_missing and not self.manifest_path.exists():
            return []
        descriptor = _open_regular_read_only(
            self.manifest_path,
            artifact_id="manifest",
        )
        records: list[ArtifactRecord] = []
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                for raw_line in stream:
                    if len(raw_line.encode("utf-8")) > MAX_MANIFEST_LINE_BYTES:
                        raise ArtifactAccessError(
                            "Artifact manifest contains an oversized record",
                            artifact_id="manifest",
                        )
                    try:
                        value = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ArtifactAccessError(
                            "Artifact manifest contains invalid JSON",
                            artifact_id="manifest",
                        ) from exc
                    if not isinstance(value, dict):
                        raise ArtifactAccessError(
                            "Artifact manifest record is invalid",
                            artifact_id="manifest",
                        )
                    raw_artifact_id = value.get("artifact_id")
                    if not isinstance(raw_artifact_id, str):
                        raise ArtifactAccessError(
                            "Artifact manifest record is invalid",
                            artifact_id="manifest",
                        )
                    records.append(
                        _record_from_dict(
                            value,
                            artifact_id=raw_artifact_id,
                        )
                    )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        identities = [record.artifact_id for record in records]
        if len(set(identities)) != len(identities):
            raise ArtifactAccessError(
                "Artifact manifest contains duplicate ids",
                artifact_id="manifest",
            )
        return records

    def _inspect_integrity(
        self,
        record: ArtifactRecord,
        *,
        max_artifact_bytes: int,
    ) -> tuple[str, str | None]:
        path = self.artifacts_dir / record.filename
        try:
            descriptor = _open_regular_read_only(
                path,
                artifact_id=record.artifact_id,
            )
        except ArtifactAccessError as exc:
            status = "missing" if "missing" in str(exc) else "unsafe_target"
            return status, str(exc)
        try:
            size = os.fstat(descriptor).st_size
            if size != record.byte_count:
                return "size_mismatch", "file size differs from the manifest"
            if size > max_artifact_bytes:
                return "oversized", "file exceeds the integrity-read bound"
            if _hash_descriptor(descriptor) != record.sha256:
                return "hash_mismatch", "file hash differs from the manifest"
        finally:
            os.close(descriptor)
        return "valid", None


def _record_from_dict(value: dict[str, Any], *, artifact_id: str) -> ArtifactRecord:
    try:
        record = ArtifactRecord(
            artifact_id=str(value["artifact_id"]),
            conversation_id=str(value["conversation_id"]),
            filename=str(value["filename"]),
            label=str(value["label"]),
            char_count=int(value["char_count"]),
            byte_count=int(value["byte_count"]),
            sha256=str(value["sha256"]),
            created_at=str(value["created_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactAccessError(
            "Artifact manifest record is invalid",
            artifact_id=artifact_id,
        ) from exc
    if (
        record.artifact_id != artifact_id
        or record.char_count < 0
        or record.byte_count < 0
        or not re.fullmatch(r"[0-9a-f]{64}", record.sha256)
    ):
        raise ArtifactAccessError(
            "Artifact manifest record is invalid",
            artifact_id=artifact_id,
        )
    return record


def _open_regular_read_only(path: Path, *, artifact_id: str) -> int:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ArtifactAccessError(
            "Artifact data is missing",
            artifact_id=artifact_id,
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ArtifactAccessError(
            "Artifact target is not a regular file",
            artifact_id=artifact_id,
        )
    if os.name == "posix":
        os.chmod(path, PRIVATE_FILE_MODE, follow_symlinks=False)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactAccessError(
            "Artifact data could not be opened safely",
            artifact_id=artifact_id,
        ) from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ArtifactAccessError(
            "Artifact target is not a regular file",
            artifact_id=artifact_id,
        )
    return descriptor


def _validate_artifact_directory(path: Path, *, artifact_id: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ArtifactAccessError(
            "Artifact data is missing",
            artifact_id=artifact_id,
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ArtifactAccessError(
            "Artifact directory is not a regular directory",
            artifact_id=artifact_id,
        )
    if os.name == "posix":
        os.chmod(path, PRIVATE_DIRECTORY_MODE, follow_symlinks=False)


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 64 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _read_bounded(
    descriptor: int,
    *,
    size: int,
    mode: ArtifactReadMode,
    offset: int,
    max_bytes: int,
) -> tuple[bytes, tuple[tuple[int, int], ...]]:
    ranges: tuple[tuple[int, int], ...]
    if mode == "slice":
        start = min(offset, size)
        ranges = ((start, min(size, start + max_bytes)),)
    elif mode == "head":
        ranges = ((0, min(size, max_bytes)),)
    elif mode == "tail":
        ranges = ((max(0, size - max_bytes), size),)
    elif size <= max_bytes:
        ranges = ((0, size),)
    else:
        head_bytes = max_bytes // 2
        tail_bytes = max_bytes - head_bytes
        ranges = ((0, head_bytes), (size - tail_bytes, size))

    parts: list[bytes] = []
    for start, end in ranges:
        os.lseek(descriptor, start, os.SEEK_SET)
        parts.append(os.read(descriptor, end - start))
    return b"\n[... omitted bytes ...]\n".join(parts), ranges


def _validate_component(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("conversation_id must be a non-empty string")
    clean = value.strip()
    if clean in {".", ".."} or "/" in clean or "\\" in clean or "\x00" in clean:
        raise ValueError("conversation_id must be a safe filename component")
    return clean


def _validate_artifact_id(value: str) -> str:
    if not isinstance(value, str) or not _ARTIFACT_ID_PATTERN.fullmatch(value):
        raise ArtifactAccessError("Artifact id is invalid")
    return value


def _safe_label(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip()).strip(".-")
    return (clean or "artifact")[:128]


__all__ = [
    "ArtifactAccessError",
    "ArtifactRead",
    "ArtifactReadMode",
    "ArtifactRecord",
    "DEFAULT_ARTIFACT_READ_BYTES",
    "MAX_ARTIFACT_FILE_BYTES",
    "MAX_ARTIFACT_READ_BYTES",
    "TraceArtifactStore",
]
