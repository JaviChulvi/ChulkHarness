"""Host-owned policy objects for workspace materialization and execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import os


class UnsafePathAction(str, Enum):
    """Action taken when materialization encounters excluded unsafe content."""

    REJECT = "reject"
    SKIP = "skip"


@dataclass(frozen=True)
class EnvironmentPolicy:
    """Environment variables made available to workspace commands."""

    inherit_all: bool = False
    allowed_names: tuple[str, ...] = (
        "PATH",
        "SYSTEMROOT",
        "COMSPEC",
        "PATHEXT",
        "WINDIR",
        "TMP",
        "TEMP",
    )
    overrides: tuple[tuple[str, str], ...] = ()

    def build(self, source: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build a fresh process environment without mutating the host mapping."""
        available = source if source is not None else os.environ
        if self.inherit_all:
            result = dict(available)
        else:
            result = {
                name: available[name]
                for name in self.allowed_names
                if name in available
            }
        result.update(self.overrides)
        return result


@dataclass(frozen=True)
class SecretPolicy:
    """Explicit allowlist for secret-like environment names."""

    allowed_environment_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourcePolicy:
    """Bounds for workspace content, change sets, commands, and output."""

    max_files: int = 5_000
    max_file_bytes: int = 200_000
    max_total_bytes: int = 20_000_000
    max_change_files: int = 500
    max_patch_bytes: int = 1_000_000
    max_retained_change_sets: int = 100
    command_timeout_seconds: int = 10
    max_stdout_bytes: int = 8_000
    max_stderr_bytes: int = 4_000

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TransferPolicy:
    """Bounds reserved for backend uploads and downloads."""

    max_upload_bytes: int = 10_000_000
    max_download_bytes: int = 10_000_000

    def __post_init__(self) -> None:
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be a positive integer")
        if self.max_download_bytes < 1:
            raise ValueError("max_download_bytes must be a positive integer")


@dataclass(frozen=True)
class WorkspaceMaterializationPolicy:
    """Allowlist and unsafe-content behavior for temporary workspaces."""

    allowed_paths: tuple[str, ...] = (".",)
    ignored_names: tuple[str, ...] = (
        ".git",
        ".chulk",
        ".venv",
        ".conda",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "build",
        "dist",
        "node_modules",
        "traces",
        "chulkharness.egg-info",
        "chulk.egg-info",
    )
    ignored_action: UnsafePathAction = UnsafePathAction.SKIP
    secret_action: UnsafePathAction = UnsafePathAction.SKIP
    symlink_action: UnsafePathAction = UnsafePathAction.REJECT
    hardlink_action: UnsafePathAction = UnsafePathAction.REJECT
    special_file_action: UnsafePathAction = UnsafePathAction.REJECT
    large_file_action: UnsafePathAction = UnsafePathAction.REJECT

    def __post_init__(self) -> None:
        if not self.allowed_paths:
            raise ValueError("allowed_paths cannot be empty")
        if any(not path.strip() for path in self.allowed_paths):
            raise ValueError("allowed_paths cannot contain empty paths")
        for field_name in (
            "ignored_action",
            "secret_action",
            "symlink_action",
            "hardlink_action",
            "special_file_action",
            "large_file_action",
        ):
            object.__setattr__(
                self,
                field_name,
                UnsafePathAction(getattr(self, field_name)),
            )


@dataclass(frozen=True)
class GitWorktreePolicy:
    """Repository and cleanup policy for optional Git-worktree sessions."""

    require_clean: bool = True
    detached: bool = True
    branch_prefix: str = "chulk/workspace"
    remove_on_close: bool = True
    delete_branch_on_close: bool = True

    def __post_init__(self) -> None:
        prefix = self.branch_prefix.strip()
        if not prefix or prefix.startswith(("-", "/")) or prefix.endswith("/"):
            raise ValueError("branch_prefix must be a valid non-empty Git branch prefix")
        if any(character.isspace() for character in prefix):
            raise ValueError("branch_prefix cannot contain whitespace")
