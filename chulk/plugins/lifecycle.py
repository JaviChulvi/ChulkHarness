"""Lifecycle diffs, recoverable lock history, and revocation state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from urllib.parse import urlsplit

from chulk.plugins.locks import PluginLockError, PluginLockFile
from chulk.plugins.models import (
    PluginAuthorityDiff,
    PluginLockEntry,
    PluginRevocation,
)


_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_MAX_STATE_BYTES = 2_000_000


class PluginLifecycleError(RuntimeError):
    """A reviewed lifecycle mutation could not be completed safely."""


@dataclass(frozen=True, slots=True)
class PluginRecoveryPoint:
    """One recoverable exact lock snapshot and optional database backup."""

    recovery_id: str
    entry: PluginLockEntry
    migration_backup: Path | None
    created_at: str


class PluginHistoryStore:
    """Owner-private append-only recovery metadata."""

    def __init__(
        self,
        runtime_dir: Path | str,
        *,
        profile_id: str,
    ) -> None:
        self.root = (
            Path(runtime_dir).expanduser().resolve()
            / "plugins"
            / "history"
        )
        self.profile_id = profile_id
        self.backup_root = (
            Path(runtime_dir).expanduser().resolve()
            / "plugins"
            / "backups"
        )

    def save(
        self,
        entry: PluginLockEntry,
        *,
        migration_backup: Path | None,
    ) -> PluginRecoveryPoint:
        timestamp = datetime.now(timezone.utc)
        recovery_id = (
            timestamp.strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + entry.digest.removeprefix("sha256:")[:12]
        )
        directory = self.root / entry.name / recovery_id
        directory.mkdir(parents=True, exist_ok=False)
        _owner_private(directory, directory=True)
        lock = PluginLockFile(
            directory / "entry.lock",
            profile_id=self.profile_id,
        )
        lock.write({entry.name: entry})
        backup_text = ""
        if migration_backup is not None:
            backup = migration_backup.resolve(strict=True)
            try:
                backup.relative_to(self.backup_root)
            except ValueError as exc:
                raise PluginLifecycleError(
                    "migration backup is outside plugin recovery storage"
                ) from exc
            backup_text = str(backup)
        created_at = timestamp.isoformat()
        _atomic_json(
            directory / "metadata.json",
            {
                "schema_version": 1,
                "recovery_id": recovery_id,
                "plugin_name": entry.name,
                "created_at": created_at,
                "migration_backup": backup_text,
            },
        )
        return PluginRecoveryPoint(
            recovery_id=recovery_id,
            entry=entry,
            migration_backup=(
                Path(backup_text) if backup_text else None
            ),
            created_at=created_at,
        )

    def latest(
        self,
        plugin_name: str,
        *,
        different_from: PluginLockEntry | None = None,
    ) -> PluginRecoveryPoint | None:
        plugin_dir = self.root / plugin_name
        if not plugin_dir.is_dir():
            return None
        for directory in sorted(plugin_dir.iterdir(), reverse=True):
            if not directory.is_dir() or directory.is_symlink():
                continue
            point = self._read(directory, plugin_name)
            if different_from is None or point.entry != different_from:
                return point
        return None

    def _read(
        self,
        directory: Path,
        plugin_name: str,
    ) -> PluginRecoveryPoint:
        metadata_path = directory / "metadata.json"
        payload = _read_json(metadata_path)
        expected = {
            "schema_version",
            "recovery_id",
            "plugin_name",
            "created_at",
            "migration_backup",
        }
        if set(payload) != expected or payload["schema_version"] != 1:
            raise PluginLifecycleError(
                "plugin recovery metadata is invalid"
            )
        if (
            payload["recovery_id"] != directory.name
            or payload["plugin_name"] != plugin_name
        ):
            raise PluginLifecycleError(
                "plugin recovery metadata identity mismatch"
            )
        try:
            entries = PluginLockFile(
                directory / "entry.lock",
                profile_id=self.profile_id,
            ).read()
        except PluginLockError as exc:
            raise PluginLifecycleError(str(exc)) from exc
        entry = entries.get(plugin_name)
        if entry is None or len(entries) != 1:
            raise PluginLifecycleError(
                "plugin recovery lock must contain exactly one plugin"
            )
        backup_value = payload["migration_backup"]
        if not isinstance(backup_value, str):
            raise PluginLifecycleError(
                "plugin recovery backup path is invalid"
            )
        backup: Path | None = None
        if backup_value:
            backup = Path(backup_value).resolve(strict=True)
            try:
                backup.relative_to(self.backup_root)
            except ValueError as exc:
                raise PluginLifecycleError(
                    "plugin recovery backup escaped its owner"
                ) from exc
        return PluginRecoveryPoint(
            recovery_id=directory.name,
            entry=entry,
            migration_backup=backup,
            created_at=str(payload["created_at"]),
        )


class PluginRevocationStore:
    """Fail-closed operator revocations keyed by exact package digest."""

    def __init__(self, runtime_dir: Path | str) -> None:
        self.path = (
            Path(runtime_dir).expanduser().resolve()
            / "plugins"
            / "revocations.json"
        )

    def list(self) -> tuple[PluginRevocation, ...]:
        if not self.path.exists():
            return ()
        payload = _read_json(self.path)
        if set(payload) != {"schema_version", "revocations"}:
            raise PluginLifecycleError(
                "plugin revocation state is invalid"
            )
        if payload["schema_version"] != 1:
            raise PluginLifecycleError(
                "unsupported plugin revocation schema"
            )
        raw = payload["revocations"]
        if not isinstance(raw, list):
            raise PluginLifecycleError(
                "plugin revocations must be a list"
            )
        records: list[PluginRevocation] = []
        for item in raw:
            if not isinstance(item, dict) or set(item) != {
                "digest",
                "plugin_name",
                "reason",
                "revoked_by",
                "revoked_at",
                "source_reference",
            }:
                raise PluginLifecycleError(
                    "plugin revocation record is invalid"
                )
            if any(not isinstance(value, str) for value in item.values()):
                raise PluginLifecycleError(
                    "plugin revocation values must be strings"
                )
            if not _DIGEST_PATTERN.fullmatch(item["digest"]):
                raise PluginLifecycleError(
                    "plugin revocation digest is invalid"
                )
            records.append(PluginRevocation(**item))
        return tuple(sorted(records, key=lambda item: item.digest))

    def revoke(
        self,
        entry: PluginLockEntry,
        *,
        reason: str,
        revoked_by: str,
    ) -> PluginRevocation:
        identity = _operator_identity(revoked_by)
        explanation = reason.strip()
        if not explanation or len(explanation) > 2_000:
            raise PluginLifecycleError(
                "revocation reason must contain 1-2000 characters"
            )
        existing = {item.digest: item for item in self.list()}
        if entry.digest in existing:
            return existing[entry.digest]
        record = PluginRevocation(
            digest=entry.digest,
            plugin_name=entry.name,
            reason=explanation,
            revoked_by=identity,
            revoked_at=datetime.now(timezone.utc).isoformat(),
            source_reference=entry.source_reference,
        )
        existing[record.digest] = record
        _atomic_json(
            self.path,
            {
                "schema_version": 1,
                "revocations": [
                    item.to_dict()
                    for item in sorted(
                        existing.values(),
                        key=lambda value: value.digest,
                    )
                ],
            },
        )
        return record

    def match(self, entry: PluginLockEntry) -> PluginRevocation | None:
        for record in self.list():
            if record.digest == entry.digest:
                return record
            if (
                record.source_reference
                and entry.source_reference == record.source_reference
                and record.plugin_name == entry.name
            ):
                return record
        return None


def plugin_authority_diff(
    current: object,
    candidate: object,
) -> PluginAuthorityDiff:
    """Return deterministic manifest authority and behavior changes."""
    return PluginAuthorityDiff(
        **_set_diffs(
            "capabilities",
            set(getattr(current, "capabilities")),
            set(getattr(candidate, "capabilities")),
        ),
        **_set_diffs(
            "secret_refs",
            set(getattr(current, "secret_refs")),
            set(getattr(candidate, "secret_refs")),
        ),
        **_set_diffs(
            "network_domains",
            set(getattr(current, "network_domains")),
            set(getattr(candidate, "network_domains")),
        ),
        **_set_diffs(
            "filesystem",
            {
                f"{item.path}:{item.access.value}"
                for item in getattr(current, "filesystem")
            },
            {
                f"{item.path}:{item.access.value}"
                for item in getattr(candidate, "filesystem")
            },
        ),
        **_set_diffs(
            "dependencies",
            _dependency_contracts(current),
            _dependency_contracts(candidate),
        ),
        **_set_diffs(
            "entry_points",
            {
                (
                    f"{item.category.value}:{item.name}={item.target}"
                    f"[{','.join(item.required_capabilities)}]"
                )
                for item in getattr(current, "entry_points")
            },
            {
                (
                    f"{item.category.value}:{item.name}={item.target}"
                    f"[{','.join(item.required_capabilities)}]"
                )
                for item in getattr(candidate, "entry_points")
            },
        ),
        **_set_diffs(
            "instructions",
            set(getattr(current, "instructions")),
            set(getattr(candidate, "instructions")),
        ),
    )


def validate_trusted_git_checkout(
    path: Path | str,
    *,
    repository_url: str,
    commit_sha: str,
    allowed_hosts: tuple[str, ...],
) -> Path:
    """Verify an existing checkout against an approved URL and exact commit."""
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise PluginLifecycleError(
            "trusted Git plugin source must be a regular directory"
        )
    clean_commit = commit_sha.strip().lower()
    if not _COMMIT_PATTERN.fullmatch(clean_commit):
        raise PluginLifecycleError(
            "trusted Git sources require an exact 40-character commit SHA"
        )
    parsed = urlsplit(repository_url)
    if parsed.scheme not in {"https", "ssh"} or not parsed.hostname:
        raise PluginLifecycleError(
            "trusted Git repository URL must use https or ssh"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PluginLifecycleError(
            "trusted Git repository URL cannot contain credentials or "
            "query/fragment data"
        )
    allowed = {host.lower() for host in allowed_hosts}
    if parsed.hostname.lower() not in allowed:
        raise PluginLifecycleError(
            "trusted Git repository host is not explicitly allowed"
        )
    actual_commit = _git(root, "rev-parse", "HEAD").lower()
    actual_remote = _git(root, "remote", "get-url", "origin")
    dirty = _git(root, "status", "--porcelain")
    if actual_commit != clean_commit:
        raise PluginLifecycleError(
            "trusted Git checkout does not match the reviewed commit"
        )
    if actual_remote != repository_url:
        raise PluginLifecycleError(
            "trusted Git checkout remote does not match the reviewed source"
        )
    if dirty:
        raise PluginLifecycleError(
            "trusted Git checkout must be clean before inspection"
        )
    return root


def disabled_entry(entry: PluginLockEntry) -> PluginLockEntry:
    """Create a disabled lock snapshot without mutating other authority."""
    from chulk.plugins.models import PluginRegistrationStatus

    return replace(
        entry,
        status=PluginRegistrationStatus.DISABLED,
    )


def _dependency_contracts(manifest: object) -> set[str]:
    values = {
        f"plugin:{item.name}{item.version_spec}:optional={item.optional}"
        for item in getattr(manifest, "dependencies")
    }
    values.update(
        f"python:{item.name}{item.version_spec}:optional={item.optional}"
        for item in getattr(manifest, "python_dependencies")
    )
    return values


def _set_diffs(
    field_name: str,
    current: set[str],
    candidate: set[str],
) -> dict[str, tuple[str, ...]]:
    return {
        f"added_{field_name}": tuple(sorted(candidate - current)),
        f"removed_{field_name}": tuple(sorted(current - candidate)),
    }


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PluginLifecycleError(
            f"trusted Git verification failed: {exc}"
        ) from exc
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise PluginLifecycleError(
            f"trusted Git verification failed: {message}"
        )
    return result.stdout.strip()


def _operator_identity(value: str) -> str:
    identity = value.strip()
    if (
        not identity
        or identity != value
        or "\x00" in identity
        or len(identity) > 200
    ):
        raise PluginLifecycleError("operator identity is invalid")
    return identity


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PluginLifecycleError(
            f"plugin lifecycle state is not a regular file: {path}"
        )
    if path.stat().st_size > _MAX_STATE_BYTES:
        raise PluginLifecycleError(
            f"plugin lifecycle state exceeds {_MAX_STATE_BYTES} bytes"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PluginLifecycleError(
            "plugin lifecycle state is invalid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise PluginLifecycleError(
            "plugin lifecycle state must contain an object"
        )
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    _owner_private(path.parent, directory=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _owner_private(path, directory=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _owner_private(path: Path, *, directory: bool) -> None:
    if os.name == "posix":
        os.chmod(path, 0o700 if directory else 0o600)


__all__ = [
    "PluginHistoryStore",
    "PluginLifecycleError",
    "PluginRecoveryPoint",
    "PluginRevocationStore",
    "disabled_entry",
    "plugin_authority_diff",
    "validate_trusted_git_checkout",
]
