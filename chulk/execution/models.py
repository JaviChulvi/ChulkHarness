"""Immutable contracts shared by execution backends and tool adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class WorkspaceMode(str, Enum):
    """How a backend exposes the project workspace."""

    HOST = "host"
    TEMPORARY = "temporary"
    GIT_WORKTREE = "git_worktree"
    CONTAINER = "container"


class ContainmentStatus(str, Enum):
    """Containment assertion made by the selected backend."""

    UNCONTAINED = "uncontained"
    CONTAINED = "contained"


class NetworkPolicy(str, Enum):
    """Effective network access for an execution session."""

    HOST_INHERITED = "host_inherited"
    DENY = "deny"


class WorkspacePersistence(str, Enum):
    """Lifetime of changes made in the execution workspace."""

    PERSISTENT = "persistent"
    EPHEMERAL = "ephemeral"


class ChangeDisposition(str, Enum):
    """How successful workspace changes are delivered."""

    APPLY_DIRECTLY = "apply_directly"
    RETURN_CHANGE_SET = "return_change_set"
    DISCARD = "discard"


@dataclass(frozen=True)
class ExecutionPolicy:
    """Host-selected policy applied to one execution session."""

    name: str
    network: NetworkPolicy = NetworkPolicy.HOST_INHERITED
    persistence: WorkspacePersistence = WorkspacePersistence.PERSISTENT
    change_disposition: ChangeDisposition = ChangeDisposition.APPLY_DIRECTLY
    require_containment: bool = False


@dataclass(frozen=True)
class ExecutionWorkspace:
    """Opaque workspace identity and its externally useful properties."""

    workspace_id: str
    backend_name: str
    mode: WorkspaceMode
    containment: ContainmentStatus


@dataclass(frozen=True)
class ExecutionSessionRequest:
    """Host metadata used when opening a turn-scoped session."""

    conversation_id: str | None = None
    turn_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class FileReadRequest:
    path: str

    def to_arguments(self) -> dict[str, Any]:
        return {"path": self.path}


@dataclass(frozen=True)
class FileWriteRequest:
    path: str
    content: str
    overwrite: bool = False

    def to_arguments(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content, "overwrite": self.overwrite}


@dataclass(frozen=True)
class PatchApplyRequest:
    patch: str

    def to_arguments(self) -> dict[str, Any]:
        return {"patch": self.patch}


@dataclass(frozen=True)
class FileListRequest:
    path: str = "."
    pattern: str = "*"
    recursive: bool = False
    max_results: int = 100

    def to_arguments(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "pattern": self.pattern,
            "recursive": self.recursive,
            "max_results": self.max_results,
        }


@dataclass(frozen=True)
class FileSearchRequest:
    query: str
    path: str = "."
    pattern: str = "*"
    max_results: int = 100

    def to_arguments(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "path": self.path,
            "pattern": self.pattern,
            "max_results": self.max_results,
        }


@dataclass(frozen=True)
class CommandExecutionRequest:
    command: str
    timeout_seconds: int | None = None

    def to_arguments(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {"command": self.command}
        if self.timeout_seconds is not None:
            arguments["timeout_seconds"] = self.timeout_seconds
        return arguments


@dataclass(frozen=True)
class ChangeRecord:
    """One backend-reported workspace change."""

    path: str
    status: str
    sha256_before: str | None = None
    sha256_after: str | None = None
    mode_before: int | None = None
    mode_after: int | None = None
    content_after: str | None = None
    content_after_base64: str | None = None


@dataclass(frozen=True)
class ChangeSet:
    """Immutable collection of changes produced by an execution session."""

    changes: tuple[ChangeRecord, ...] = ()
    change_set_id: str = ""
    workspace_id: str = ""
    backend_name: str = ""
    patch: str = ""
    patch_truncated: bool = False
    total_bytes: int = 0


@dataclass(frozen=True)
class ChangeSetApproval:
    """Explicit host approval required before applying one change set."""

    change_set_id: str
    approved_by: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.change_set_id.strip():
            raise ValueError("change_set_id cannot be empty")
        if not self.approved_by.strip():
            raise ValueError("approved_by cannot be empty")


@dataclass(frozen=True)
class ChangeApplicationResult:
    """Outcome of a conflict-checked transactional host application."""

    success: bool
    change_set_id: str
    approved_by: str
    reason: str | None = None
    applied_paths: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    rollback_errors: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class ProcessHandle:
    """Opaque identity for a backend-owned long-running process."""

    process_id: str
    backend_name: str


@dataclass(frozen=True)
class ExecutionResult:
    """Backend-neutral operation result converted to a tool result at the edge."""

    success: bool
    observation: str
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    error: str | None = None
    failure_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    value: Any = None
    change_set: ChangeSet | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_tool_result(self, tool_name: str):
        """Convert at the tool boundary without coupling backend protocols to tools."""
        from chulk.tools.registry import ToolResult

        return ToolResult(
            tool_name=tool_name,
            success=self.success,
            observation=self.observation,
            stdout=self.stdout,
            stderr=self.stderr,
            exit_code=self.exit_code,
            error=self.error,
            failure_kind=self.failure_kind,
            metadata=dict(self.metadata),
            value=self.value,
        )
