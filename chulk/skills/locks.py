"""Credential-free project and profile-local skill lock files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from chulk.skills.lifecycle_models import SkillLifecycleStatus
from chulk.skills.manifest import SkillPackage


SKILL_LOCK_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SkillLockEntry:
    """One exact reviewed package identity recorded in a lock file."""

    name: str
    version: str
    digest: str
    source: str
    trust: str
    status: SkillLifecycleStatus
    revision_id: str
    required_tools: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    installed_at: str
    review_state: str = "approved"

    @classmethod
    def from_package(
        cls,
        package: SkillPackage,
        *,
        revision_id: str,
        status: SkillLifecycleStatus = SkillLifecycleStatus.ACTIVE,
        installed_at: str | None = None,
    ) -> SkillLockEntry:
        manifest = package.manifest
        return cls(
            name=manifest.name,
            version=manifest.version,
            digest=package.digest,
            source=manifest.source,
            trust=manifest.trust,
            status=status,
            revision_id=revision_id,
            required_tools=manifest.required_tools,
            required_capabilities=manifest.required_capabilities,
            installed_at=installed_at or datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "source": self.source,
            "trust": self.trust,
            "status": self.status.value,
            "revision_id": self.revision_id,
            "required_tools": list(self.required_tools),
            "required_capabilities": list(self.required_capabilities),
            "installed_at": self.installed_at,
            "review_state": self.review_state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SkillLockEntry:
        return cls(
            name=str(value["name"]),
            version=str(value["version"]),
            digest=str(value["digest"]),
            source=str(value["source"]),
            trust=str(value["trust"]),
            status=SkillLifecycleStatus(str(value["status"])),
            revision_id=str(value["revision_id"]),
            required_tools=tuple(
                str(item) for item in value.get("required_tools", [])
            ),
            required_capabilities=tuple(
                str(item) for item in value.get("required_capabilities", [])
            ),
            installed_at=str(value["installed_at"]),
            review_state=str(value.get("review_state", "approved")),
        )


class SkillLockFile:
    """Read and atomically update one deterministic skill lock."""

    def __init__(self, path: Path | str, *, scope: str, private: bool) -> None:
        if scope not in {"project", "profile"}:
            raise ValueError("skill lock scope must be project or profile")
        self.path = Path(path)
        self.scope = scope
        self.private = private

    def read(self) -> dict[str, SkillLockEntry]:
        if not self.path.exists():
            return {}
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError(f"skill lock is not a regular file: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("skill lock must contain a JSON object")
        if payload.get("schema_version") != SKILL_LOCK_SCHEMA_VERSION:
            raise ValueError("unsupported skill lock schema version")
        if payload.get("scope") != self.scope:
            raise ValueError("skill lock scope does not match its owner")
        skills = payload.get("skills")
        if not isinstance(skills, dict):
            raise ValueError("skill lock skills must be an object")
        return {
            str(name): SkillLockEntry.from_dict(value)
            for name, value in skills.items()
            if isinstance(value, dict)
        }

    def get(self, name: str) -> SkillLockEntry | None:
        return self.read().get(name)

    def update(self, entry: SkillLockEntry) -> None:
        entries = self.read()
        entries[entry.name] = entry
        self.write(entries)

    def remove(self, name: str) -> None:
        entries = self.read()
        entries.pop(name, None)
        self.write(entries)

    def write(self, entries: dict[str, SkillLockEntry]) -> None:
        payload = {
            "schema_version": SKILL_LOCK_SCHEMA_VERSION,
            "scope": self.scope,
            "skills": {
                name: entry.to_dict()
                for name, entry in sorted(entries.items())
            },
        }
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and (
            self.path.is_symlink() or not self.path.is_file()
        ):
            raise ValueError(f"skill lock is not a regular file: {self.path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            mode = 0o600 if self.private else 0o644
            if os.name == "posix":
                os.fchmod(descriptor, mode)
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
                os.chmod(self.path, mode, follow_symlinks=False)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def snapshot(self) -> bytes | None:
        if not self.path.exists():
            return None
        mode = self.path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(f"skill lock is not a regular file: {self.path}")
        return self.path.read_bytes()

    def restore(self, content: bytes | None) -> None:
        if content is None:
            self.path.unlink(missing_ok=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.restore.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            if os.name == "posix":
                os.chmod(
                    self.path,
                    0o600 if self.private else 0o644,
                    follow_symlinks=False,
                )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)


__all__ = [
    "SKILL_LOCK_SCHEMA_VERSION",
    "SkillLockEntry",
    "SkillLockFile",
]
