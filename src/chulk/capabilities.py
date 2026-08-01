"""Application intent and opt-in tool policy values for SDK agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast


class FileAccess(StrEnum):
    OFF = "off"
    READ = "read"
    WRITE = "write"


class MemoryMode(StrEnum):
    OFF = "off"
    READ_ONLY = "read-only"
    MANUAL = "manual"
    AUTOMATIC = "automatic"


@dataclass(frozen=True)
class Capabilities:
    """Tool categories an embedding application intends to expose."""

    files: FileAccess | str = FileAccess.READ
    shell: bool = False
    memory: MemoryMode | str = MemoryMode.READ_ONLY
    network: bool = False
    external_services: bool = False
    utilities: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", _file_access(self.files))
        object.__setattr__(self, "memory", _memory_mode(self.memory))

    @classmethod
    def none(cls) -> "Capabilities":
        return cls(files=FileAccess.OFF, memory=MemoryMode.OFF, utilities=False)

    @classmethod
    def read_only(cls) -> "Capabilities":
        return cls()

    @classmethod
    def coding(cls) -> "Capabilities":
        return cls(files=FileAccess.WRITE, shell=True, memory=MemoryMode.MANUAL)

    @classmethod
    def full(cls) -> "Capabilities":
        return cls(
            files=FileAccess.WRITE,
            shell=True,
            memory=MemoryMode.AUTOMATIC,
            network=True,
            external_services=True,
        )

    def with_memory(self, mode: MemoryMode | str) -> "Capabilities":
        return replace(self, memory=_memory_mode(mode))

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": cast(FileAccess, self.files).value,
            "shell": self.shell,
            "memory": cast(MemoryMode, self.memory).value,
            "network": self.network,
            "external_services": self.external_services,
            "utilities": self.utilities,
        }


@dataclass(frozen=True)
class ToolOutputPolicy:
    """Optional JSON-schema contract for a successful tool value."""

    schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", MappingProxyType(_copy_mapping(self.schema)))

    def to_dict(self) -> dict[str, Any]:
        return _plain_mapping(self.schema)


@dataclass(frozen=True)
class ToolRetryPolicy:
    """Bounded, opt-in retry behavior for one tool."""

    max_attempts: int = 1
    retryable_failure_kinds: tuple[str, ...] = ("environment_failure", "timeout")
    backoff_seconds: float = 0.0
    require_idempotent: bool = True

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        object.__setattr__(self, "retryable_failure_kinds", tuple(dict.fromkeys(self.retryable_failure_kinds)))


def _file_access(value: FileAccess | str) -> FileAccess:
    if isinstance(value, FileAccess):
        return value
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {"none": "off", "read-only": "read", "read-write": "write"}
    try:
        return FileAccess(aliases.get(normalized, normalized))
    except ValueError as exc:
        raise ValueError("files must be off, read, or write") from exc


def _memory_mode(value: MemoryMode | str) -> MemoryMode:
    if isinstance(value, MemoryMode):
        return value
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "read": "read-only",
        "manual-read-write": "manual",
        "auto": "automatic",
        "automatic-read-write": "automatic",
    }
    try:
        return MemoryMode(aliases.get(normalized, normalized))
    except ValueError as exc:
        raise ValueError("memory mode must be off, read-only, manual, or automatic") from exc


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[str(key)] = MappingProxyType(_copy_mapping(item))
        elif isinstance(item, (list, tuple)):
            result[str(key)] = tuple(item)
        else:
            result[str(key)] = item
    return result


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[str(key)] = _plain_mapping(item)
        elif isinstance(item, tuple):
            result[str(key)] = list(item)
        else:
            result[str(key)] = item
    return result


__all__ = [
    "Capabilities",
    "FileAccess",
    "MemoryMode",
    "ToolOutputPolicy",
    "ToolRetryPolicy",
]
