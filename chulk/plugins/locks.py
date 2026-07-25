"""Owner-private exact plugin lock files."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from chulk.plugins.manifest import (
    PluginManifestError,
    plugin_manifest_from_mapping,
)
from chulk.plugins.models import (
    PLUGIN_LOCK_SCHEMA_VERSION,
    PluginLockEntry,
    PluginRegistrationStatus,
    PluginReview,
    PluginSourceKind,
)


_PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "profile_id", "plugins"}
)
_ENTRY_FIELDS = frozenset(
    {
        "name",
        "version",
        "digest",
        "source_kind",
        "source_path",
        "status",
        "manifest",
        "review",
        "installed_at",
    }
)
_REVIEW_FIELDS = frozenset(
    {
        "approved_by",
        "approved_at",
        "acknowledged_host_authority",
        "granted_capabilities",
    }
)
_MAX_LOCK_BYTES = 2_000_000


class PluginLockError(ValueError):
    """Raised when a plugin lock is invalid or unsafe."""


class PluginLockFile:
    """Read and atomically write one profile-owned plugin lock."""

    def __init__(self, path: Path | str, *, profile_id: str) -> None:
        clean_profile_id = profile_id.strip().lower()
        if not _PROFILE_ID_PATTERN.fullmatch(clean_profile_id):
            raise ValueError("invalid plugin lock profile id")
        self.path = Path(path)
        self.profile_id = clean_profile_id

    def read(self) -> dict[str, PluginLockEntry]:
        if not self.path.exists():
            return {}
        mode = self.path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise PluginLockError(
                f"plugin lock is not a regular file: {self.path}"
            )
        if self.path.stat().st_size > _MAX_LOCK_BYTES:
            raise PluginLockError(
                f"plugin lock exceeds {_MAX_LOCK_BYTES} bytes"
            )
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PluginLockError("plugin lock is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise PluginLockError("plugin lock must contain an object")
        if set(payload) != _TOP_LEVEL_FIELDS:
            raise PluginLockError(
                "plugin lock contains unsupported top-level fields"
            )
        if payload["schema_version"] != PLUGIN_LOCK_SCHEMA_VERSION:
            raise PluginLockError("unsupported plugin lock schema version")
        if payload["profile_id"] != self.profile_id:
            raise PluginLockError(
                "plugin lock profile does not match its owner"
            )
        raw_plugins = payload["plugins"]
        if not isinstance(raw_plugins, dict):
            raise PluginLockError("plugin lock plugins must be an object")
        entries: dict[str, PluginLockEntry] = {}
        for key, value in raw_plugins.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise PluginLockError(
                    "plugin lock entries must be named objects"
                )
            entry = _entry_from_dict(value)
            if entry.name != key:
                raise PluginLockError(
                    "plugin lock entry key does not match its name"
                )
            entries[key] = entry
        return entries

    def get(self, name: str) -> PluginLockEntry | None:
        return self.read().get(name)

    def update(self, entry: PluginLockEntry) -> None:
        entries = self.read()
        entries[entry.name] = entry
        self.write(entries)

    def write(self, entries: dict[str, PluginLockEntry]) -> None:
        for key, entry in entries.items():
            _validate_entry(entry)
            if key != entry.name:
                raise PluginLockError(
                    "plugin lock entry key does not match its name"
                )
        payload = {
            "schema_version": PLUGIN_LOCK_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "plugins": {
                name: entry.to_dict()
                for name, entry in sorted(entries.items())
            },
        }
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            mode = self.path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise PluginLockError(
                    f"plugin lock is not a regular file: {self.path}"
                )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                descriptor = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            if os.name == "posix":
                os.chmod(self.path, 0o600, follow_symlinks=False)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)


def _entry_from_dict(value: dict[str, Any]) -> PluginLockEntry:
    if set(value) != _ENTRY_FIELDS:
        raise PluginLockError("plugin lock entry has unsupported fields")
    raw_manifest = value["manifest"]
    raw_review = value["review"]
    if not isinstance(raw_manifest, dict) or not isinstance(raw_review, dict):
        raise PluginLockError(
            "plugin lock manifest and review must be objects"
        )
    if set(raw_review) != _REVIEW_FIELDS:
        raise PluginLockError("plugin lock review has unsupported fields")
    acknowledged = raw_review["acknowledged_host_authority"]
    capabilities = raw_review["granted_capabilities"]
    if not isinstance(acknowledged, bool) or not isinstance(
        capabilities,
        list,
    ):
        raise PluginLockError("plugin lock review fields are invalid")
    string_fields = (
        "name",
        "version",
        "digest",
        "source_kind",
        "source_path",
        "status",
        "installed_at",
    )
    if any(not isinstance(value[field], str) for field in string_fields):
        raise PluginLockError("plugin lock entry string fields are invalid")
    review_string_fields = ("approved_by", "approved_at")
    if any(
        not isinstance(raw_review[field], str)
        for field in review_string_fields
    ):
        raise PluginLockError("plugin lock review string fields are invalid")
    if any(not isinstance(item, str) for item in capabilities):
        raise PluginLockError(
            "plugin lock granted capabilities must be strings"
        )
    try:
        manifest = plugin_manifest_from_mapping(raw_manifest)
        entry = PluginLockEntry(
            name=value["name"],
            version=value["version"],
            digest=value["digest"],
            source_kind=PluginSourceKind(value["source_kind"]),
            source_path=Path(value["source_path"]),
            status=PluginRegistrationStatus(value["status"]),
            manifest=manifest,
            review=PluginReview(
                approved_by=raw_review["approved_by"],
                approved_at=raw_review["approved_at"],
                acknowledged_host_authority=acknowledged,
                granted_capabilities=tuple(capabilities),
            ),
            installed_at=value["installed_at"],
        )
    except (PluginManifestError, ValueError) as exc:
        raise PluginLockError(f"invalid plugin lock entry: {exc}") from exc
    _validate_entry(entry)
    return entry


def _validate_entry(entry: PluginLockEntry) -> None:
    if entry.name != entry.manifest.name:
        raise PluginLockError("plugin lock name does not match manifest")
    if entry.version != entry.manifest.version:
        raise PluginLockError("plugin lock version does not match manifest")
    if not _DIGEST_PATTERN.fullmatch(entry.digest):
        raise PluginLockError("plugin lock digest is invalid")
    if entry.source_kind is not PluginSourceKind.LOCAL_DIRECTORY:
        raise PluginLockError("unsupported plugin source kind")
    source_text = str(entry.source_path)
    if (
        not entry.source_path.is_absolute()
        or "\x00" in source_text
        or entry.source_path.resolve(strict=False) != entry.source_path
    ):
        raise PluginLockError(
            "local plugin source path must be canonical and absolute"
        )
    approved_by = entry.review.approved_by
    if (
        not approved_by.strip()
        or approved_by != approved_by.strip()
        or "\x00" in approved_by
        or len(approved_by) > 200
    ):
        raise PluginLockError("plugin approval identity is invalid")
    if not entry.review.acknowledged_host_authority:
        raise PluginLockError(
            "plugin review must acknowledge host-process authority"
        )
    granted = tuple(sorted(set(entry.review.granted_capabilities)))
    if granted != entry.review.granted_capabilities:
        raise PluginLockError(
            "granted plugin capabilities must be sorted and unique"
        )
    if not set(granted).issubset(entry.manifest.capabilities):
        raise PluginLockError(
            "granted capabilities are not declared by the plugin"
        )
    _timestamp(entry.review.approved_at, "approved_at")
    _timestamp(entry.installed_at, "installed_at")


def _timestamp(value: str, field_name: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PluginLockError(
            f"plugin lock {field_name} is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PluginLockError(
            f"plugin lock {field_name} must be timezone-aware"
        )


__all__ = ["PluginLockError", "PluginLockFile"]
