"""Project-root-bound file tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.registry import Tool, ToolResult


IGNORED_DIRS = {
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
    "chulkharness.egg-info",
    "chulk.egg-info",
}
SENSITIVE_READ_DIRS = {".git", "traces"}
UNSAFE_WRITE_DIRS = {
    *IGNORED_DIRS,
    *SENSITIVE_READ_DIRS,
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "node_modules",
    "traces",
}
UNSAFE_SECRET_NAMES = {
    ".env",
    ".envrc",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "service-account.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
UNSAFE_SECRET_MARKERS = {"api_key", "apikey", "credential", "credentials", "password", "secret", "secrets", "token", "tokens"}
UNSAFE_KEY_MARKERS = {"key", "keys", "private_key", "private_keys", "secret_key", "ssh_key", "ssh_private_key"}
UNSAFE_SECRET_SUFFIXES = {"", ".env", ".ini", ".json", ".key", ".pem", ".p12", ".pfx", ".toml", ".txt", ".yaml", ".yml"}
UNSAFE_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
SENSITIVE_KEY_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
SAFE_ENV_TEMPLATE_SUFFIXES = {".example", ".sample", ".template"}
SQLITE_SIDECAR_SUFFIXES = {"journal", "shm", "wal"}
SENSITIVE_BACKUP_SUFFIXES = {".bak", ".backup", ".old", ".orig"}
MAX_TEXT_FILE_BYTES = 200_000
PATCH_REREAD_CONTEXT_LINES = 3
HUNK_HEADER_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")


@dataclass(frozen=True)
class FileReadPolicy:
    """Host-selected access policy for model-facing file read tools."""

    allow_sensitive_paths: bool = False


DEFAULT_FILE_READ_POLICY = FileReadPolicy()


@dataclass(frozen=True)
class PatchLine:
    """One parsed line inside a unified-diff hunk."""

    prefix: str
    text: str


@dataclass(frozen=True)
class PatchHunk:
    """One unified-diff hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[PatchLine]


@dataclass(frozen=True)
class FilePatch:
    """One file entry in a unified diff."""

    old_path: str | None
    new_path: str | None
    hunks: list[PatchHunk]


@dataclass(frozen=True)
class PendingPatchWrite:
    """One validated file write that can be committed atomically."""

    path: Path
    relative_path: str
    status: str
    old_text: str | None
    new_text: str


def read_file_tool(
    project_root: Path,
    *,
    read_policy: FileReadPolicy = DEFAULT_FILE_READ_POLICY,
) -> Tool:
    def invoke(arguments: dict[str, Any], context=None) -> ToolResult:
        session = getattr(context, "execution_session", None)
        if session is None:
            return read_file(arguments, project_root, read_policy=read_policy)
        from chulk.execution.models import FileReadRequest

        return session.read_file(FileReadRequest(path=arguments["path"])).to_tool_result(
            "read_file"
        )

    return Tool(
        name="read_file",
        description="Read a UTF-8 text file inside the project directory.",
        args_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                    "minLength": 1,
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        callable=invoke,
        accepts_context=True,
        run_in_executor=True,
        permission_level=ToolPermissionLevel.READ,
    )


def write_file_tool(project_root: Path) -> Tool:
    def invoke(arguments: dict[str, Any], context=None) -> ToolResult:
        session = getattr(context, "execution_session", None)
        if session is None:
            return write_file(arguments, project_root)
        from chulk.execution.models import FileWriteRequest

        return session.write_file(
            FileWriteRequest(
                path=arguments["path"],
                content=arguments["content"],
                overwrite=arguments.get("overwrite", False),
            )
        ).to_tool_result("write_file")

    return Tool(
        name="write_file",
        description=(
            "Create a UTF-8 text file inside the project directory. "
            "Prefer apply_patch for edits. Existing files require overwrite=true and must pass write-safety checks."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                    "minLength": 1,
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write.",
                    "maxLength": MAX_TEXT_FILE_BYTES,
                },
                "overwrite": {"type": "boolean", "description": "Set true to overwrite an existing file."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        callable=invoke,
        accepts_context=True,
        run_in_executor=True,
        permission_level=ToolPermissionLevel.WRITE,
    )


def apply_patch_tool(project_root: Path) -> Tool:
    def invoke(arguments: dict[str, Any], context=None) -> ToolResult:
        session = getattr(context, "execution_session", None)
        if session is None:
            return apply_patch(arguments, project_root)
        from chulk.execution.models import PatchApplyRequest

        return session.apply_patch(
            PatchApplyRequest(patch=arguments["patch"])
        ).to_tool_result("apply_patch")

    return Tool(
        name="apply_patch",
        description=(
            "Apply a unified diff inside the project directory. "
            "This is the preferred tool for modifying existing text files."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Unified diff text. Supports modifying files and creating new files.",
                    "minLength": 1,
                    "maxLength": MAX_TEXT_FILE_BYTES,
                }
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
        callable=invoke,
        accepts_context=True,
        run_in_executor=True,
        permission_level=ToolPermissionLevel.WRITE,
        metadata={"preferred_for": "file_edits"},
    )


def list_files_tool(
    project_root: Path,
    *,
    read_policy: FileReadPolicy = DEFAULT_FILE_READ_POLICY,
) -> Tool:
    def invoke(arguments: dict[str, Any], context=None) -> ToolResult:
        session = getattr(context, "execution_session", None)
        if session is None:
            return list_files(arguments, project_root, read_policy=read_policy)
        from chulk.execution.models import FileListRequest

        return session.list_files(
            FileListRequest(
                path=arguments.get("path", "."),
                pattern=arguments.get("pattern", "*"),
                recursive=arguments.get("recursive", False),
                max_results=arguments.get("max_results", 100),
            )
        ).to_tool_result("list_files")

    return Tool(
        name="list_files",
        description="List files inside the project directory.",
        args_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the project root.",
                    "minLength": 1,
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, for example *.py.",
                    "minLength": 1,
                },
                "recursive": {"type": "boolean", "description": "Whether to search recursively."},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of files to return.",
                    "minimum": 1,
                    "maximum": 500,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        callable=invoke,
        accepts_context=True,
        run_in_executor=True,
        permission_level=ToolPermissionLevel.READ,
    )


def search_files_tool(
    project_root: Path,
    *,
    read_policy: FileReadPolicy = DEFAULT_FILE_READ_POLICY,
) -> Tool:
    def invoke(arguments: dict[str, Any], context=None) -> ToolResult:
        session = getattr(context, "execution_session", None)
        if session is None:
            return search_files(arguments, project_root, read_policy=read_policy)
        from chulk.execution.models import FileSearchRequest

        return session.search_files(
            FileSearchRequest(
                query=arguments["query"],
                path=arguments.get("path", "."),
                pattern=arguments.get("pattern", "*"),
                max_results=arguments.get("max_results", 100),
            )
        ).to_tool_result("search_files")

    return Tool(
        name="search_files",
        description="Search text files inside the project directory.",
        args_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to search for.",
                    "minLength": 1,
                },
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the project root.",
                    "minLength": 1,
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, for example *.py.",
                    "minLength": 1,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return.",
                    "minimum": 1,
                    "maximum": 500,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        callable=invoke,
        accepts_context=True,
        run_in_executor=True,
        permission_level=ToolPermissionLevel.READ,
    )


def read_file(
    arguments: dict[str, Any],
    project_root: Path,
    *,
    read_policy: FileReadPolicy = DEFAULT_FILE_READ_POLICY,
) -> ToolResult:
    root = project_root.resolve()
    path = resolve_inside_root(root, arguments["path"])
    safety_error = safe_read_error(path, root, read_policy, requested_path=arguments["path"])
    if safety_error:
        return ToolResult("read_file", False, safety_error, error="sensitive_path")
    if not path.exists() or not path.is_file():
        return ToolResult("read_file", False, f"File not found: {_relative_path(path, root)}", error="not_found")
    if path.stat().st_size > MAX_TEXT_FILE_BYTES:
        return ToolResult("read_file", False, "File is too large to read safely.", error="file_too_large")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult("read_file", False, "File is not valid UTF-8 text.", error="not_text")
    return ToolResult("read_file", True, content, metadata={"path": _relative_path(path, root)})


def write_file(arguments: dict[str, Any], project_root: Path) -> ToolResult:
    path = resolve_inside_root(project_root, arguments["path"])
    content = arguments["content"]
    overwrite = arguments.get("overwrite", False)
    safety_error = safe_write_error(path, project_root)
    if safety_error:
        return ToolResult("write_file", False, safety_error, error="unsafe_path", metadata={"path": _relative_path(path, project_root)})
    if path.exists() and not overwrite:
        return ToolResult(
            "write_file",
            False,
            "File already exists. Use apply_patch for edits, or pass overwrite=true for a full-file replacement.",
            error="exists",
            metadata={"path": _relative_path(path, project_root)},
        )
    old_text = None
    if path.exists():
        if not path.is_file():
            return ToolResult("write_file", False, "Path exists but is not a file.", error="not_file")
        try:
            old_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult("write_file", False, "Existing file is not valid UTF-8 text.", error="not_text")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    old_hash = _sha256_text(old_text) if old_text is not None else None
    return ToolResult(
        "write_file",
        True,
        f"Wrote {len(content.encode('utf-8'))} bytes to {_relative_path(path, project_root)}.",
        metadata={
            "path": _relative_path(path, project_root),
            "status": "modified" if old_text is not None else "created",
            "sha256_before": old_hash,
            "sha256_after": _sha256_text(content),
        },
    )


def apply_patch(arguments: dict[str, Any], project_root: Path) -> ToolResult:
    """Apply a unified diff to files under the project root."""
    try:
        patches = parse_unified_diff(arguments["patch"])
        pending_writes = _prepare_patch_writes(patches, project_root)
    except PatchError as exc:
        return ToolResult("apply_patch", False, str(exc), error=exc.code, metadata=exc.metadata)

    try:
        _commit_patch_writes(pending_writes, project_root.resolve())
    except PatchCommitError as exc:
        return ToolResult(
            "apply_patch",
            False,
            str(exc),
            error=exc.code,
            metadata=exc.metadata,
        )

    changes: list[dict[str, Any]] = [
        {
            "path": pending.relative_path,
            "status": pending.status,
            "sha256_before": _sha256_text(pending.old_text) if pending.old_text is not None else None,
            "sha256_after": _sha256_text(pending.new_text),
        }
        for pending in pending_writes
    ]
    created_count = sum(1 for change in changes if change["status"] == "created")
    modified_count = sum(1 for change in changes if change["status"] == "modified")
    paths = [change["path"] for change in changes]
    return ToolResult(
        "apply_patch",
        True,
        f"Applied patch to {len(changes)} file(s): {', '.join(paths)}.",
        metadata={
            "paths": paths,
            "changes": changes,
            "changed_count": len(changes),
            "created_count": created_count,
            "modified_count": modified_count,
        },
    )


class PatchCommitError(RuntimeError):
    """Raised after a patch commit fails and rollback has been attempted."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.metadata = metadata


def _commit_patch_writes(
    pending_writes: list[PendingPatchWrite],
    project_root: Path,
) -> None:
    attempted: list[PendingPatchWrite] = []
    created_directories: set[Path] = set()
    try:
        for pending in pending_writes:
            created_directories.update(
                _create_patch_parent_directories(pending.path.parent, project_root)
            )
            attempted.append(pending)
            _write_patch_text(pending.path, pending.new_text)
    except BaseException as exc:
        rollback_errors = _rollback_patch_writes(attempted, created_directories)
        if rollback_errors:
            raise PatchCommitError(
                "Patch commit failed and rollback could not fully restore the workspace.",
                code="patch_rollback_failed",
                metadata={
                    "cause": type(exc).__name__,
                    "rollback_errors": rollback_errors,
                },
            ) from exc
        if isinstance(exc, Exception):
            raise PatchCommitError(
                "Patch commit failed; the workspace was restored.",
                code="patch_commit_failed",
                metadata={"cause": type(exc).__name__},
            ) from exc
        raise


def _create_patch_parent_directories(parent: Path, project_root: Path) -> tuple[Path, ...]:
    missing: list[Path] = []
    candidate = parent
    while candidate != project_root and not candidate.exists():
        missing.append(candidate)
        candidate = candidate.parent
    parent.mkdir(parents=True, exist_ok=True)
    return tuple(missing)


def _write_patch_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _restore_patch_write(pending: PendingPatchWrite) -> None:
    if pending.old_text is None:
        pending.path.unlink(missing_ok=True)
        return
    pending.path.write_text(pending.old_text, encoding="utf-8")


def _rollback_patch_writes(
    attempted: list[PendingPatchWrite],
    created_directories: set[Path],
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for pending in reversed(attempted):
        try:
            _restore_patch_write(pending)
        except BaseException as exc:
            errors.append(
                {
                    "path": pending.relative_path,
                    "error": type(exc).__name__,
                }
            )
    for directory in sorted(
        created_directories,
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except FileNotFoundError:
            continue
        except OSError as exc:
            if directory.exists():
                errors.append(
                    {
                        "path": str(directory),
                        "error": type(exc).__name__,
                    }
                )
    return errors


def list_files(
    arguments: dict[str, Any],
    project_root: Path,
    *,
    read_policy: FileReadPolicy = DEFAULT_FILE_READ_POLICY,
) -> ToolResult:
    root = project_root.resolve()
    requested_path = arguments.get("path", ".")
    directory = resolve_inside_root(root, requested_path)
    pattern = arguments.get("pattern", "*")
    recursive = arguments.get("recursive", False)
    max_results = min(arguments.get("max_results", 100), 500)

    safety_error = safe_read_error(directory, root, read_policy, requested_path=requested_path)
    if safety_error:
        return ToolResult("list_files", False, safety_error, error="sensitive_path")
    if not directory.exists() or not directory.is_dir():
        return ToolResult(
            "list_files",
            False,
            f"Directory not found: {_relative_path(directory, root)}",
            error="not_found",
        )

    iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
    results: list[str] = []
    for path in iterator:
        if (
            safe_read_error(path, root, read_policy) is not None
            or _is_ignored(path, root)
            or not path.is_file()
        ):
            continue
        results.append(_relative_path(path, root))
        if len(results) >= max_results:
            break
    return ToolResult("list_files", True, "\n".join(sorted(results)) or "No files found.")


def search_files(
    arguments: dict[str, Any],
    project_root: Path,
    *,
    read_policy: FileReadPolicy = DEFAULT_FILE_READ_POLICY,
) -> ToolResult:
    root = project_root.resolve()
    query = arguments["query"]
    requested_path = arguments.get("path", ".")
    directory = resolve_inside_root(root, requested_path)
    pattern = arguments.get("pattern", "*")
    max_results = min(arguments.get("max_results", 100), 500)
    safety_error = safe_read_error(directory, root, read_policy, requested_path=requested_path)
    if safety_error:
        return ToolResult("search_files", False, safety_error, error="sensitive_path")
    if not directory.exists() or not directory.is_dir():
        return ToolResult(
            "search_files",
            False,
            f"Directory not found: {_relative_path(directory, root)}",
            error="not_found",
        )

    if shutil.which("rg"):
        return _search_with_rg(root, directory, query, pattern, max_results, read_policy)
    return _search_with_python(root, directory, query, pattern, max_results, read_policy)


def _search_with_rg(
    project_root: Path,
    directory: Path,
    query: str,
    pattern: str,
    max_results: int,
    read_policy: FileReadPolicy,
) -> ToolResult:
    command = ["rg", "--json", "--line-number", "--color", "never", "--hidden", "--glob", pattern]
    if read_policy.allow_sensitive_paths:
        command.append("--no-ignore")
    for ignored_dir in sorted(IGNORED_DIRS):
        command.extend(["--glob", f"!{ignored_dir}/**", "--glob", f"!**/{ignored_dir}/**"])
    if not read_policy.allow_sensitive_paths:
        for sensitive_dir in sorted(SENSITIVE_READ_DIRS):
            command.extend(["--glob", f"!{sensitive_dir}/**", "--glob", f"!**/{sensitive_dir}/**"])
    command.extend(["--", query, str(directory)])

    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        return ToolResult("search_files", False, "Search failed.", error="search_failed")
    results: list[str] = []
    for raw_result in completed.stdout.splitlines():
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            return ToolResult("search_files", False, "Search returned invalid output.", error="search_failed")
        if result.get("type") != "match":
            continue
        data = result["data"]
        path_text = data["path"].get("text")
        line_text = data["lines"].get("text")
        if path_text is None or line_text is None:
            continue
        path = Path(path_text)
        if not path.is_absolute():
            path = project_root / path
        if safe_read_error(path, project_root, read_policy) is not None:
            continue
        relative_path = _relative_path(path, project_root)
        normalized_line = line_text.rstrip("\r\n")
        results.append(f"{relative_path}:{data['line_number']}:{normalized_line}")
        if len(results) >= max_results:
            break
    return ToolResult("search_files", True, "\n".join(results) or "No matches found.")


def _search_with_python(
    project_root: Path,
    directory: Path,
    query: str,
    pattern: str,
    max_results: int,
    read_policy: FileReadPolicy,
) -> ToolResult:
    results: list[str] = []
    for path in directory.rglob(pattern):
        if (
            safe_read_error(path, project_root, read_policy) is not None
            or _is_ignored(path, project_root)
            or not path.is_file()
            or path.stat().st_size > MAX_TEXT_FILE_BYTES
        ):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for index, line in enumerate(lines, start=1):
            if query in line:
                results.append(f"{_relative_path(path, project_root)}:{index}:{line}")
                if len(results) >= max_results:
                    return ToolResult("search_files", True, "\n".join(results))
    return ToolResult("search_files", True, "\n".join(results) or "No matches found.")


class PatchError(ValueError):
    """Raised when a unified diff cannot be safely applied."""

    def __init__(self, message: str, *, code: str = "patch_error", metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.metadata = metadata or {}


def parse_unified_diff(patch_text: str) -> list[FilePatch]:
    """Parse a small, strict subset of unified diff."""
    lines = patch_text.splitlines()
    patches: list[FilePatch] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if _is_unsupported_patch_header(line):
            raise PatchError(f"Unsupported patch operation: {line}", code="unsupported_patch_operation")
        if line.startswith(("diff --git ", "index ", "new file mode ", "old mode ", "new mode ")):
            index += 1
            continue
        if not line.startswith("--- "):
            index += 1
            continue

        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise PatchError("Patch file header is missing a +++ line.", code="patch_parse_error")

        old_path = _parse_patch_path(lines[index][4:])
        new_path = _parse_patch_path(lines[index + 1][4:])
        if new_path is None:
            raise PatchError("Deleting files is not supported by apply_patch v1.", code="unsupported_patch_operation")
        if old_path is not None and old_path != new_path:
            raise PatchError("Renaming files is not supported by apply_patch v1.", code="unsupported_patch_operation")

        index += 2
        hunks: list[PatchHunk] = []
        while index < len(lines):
            line = lines[index]
            if line.startswith("--- "):
                break
            if _is_unsupported_patch_header(line):
                raise PatchError(f"Unsupported patch operation: {line}", code="unsupported_patch_operation")
            if line.startswith(("diff --git ", "index ", "new file mode ", "old mode ", "new mode ")):
                index += 1
                continue
            if not line.startswith("@@ "):
                index += 1
                continue

            hunk, index = _parse_hunk(lines, index)
            hunks.append(hunk)

        if not hunks:
            raise PatchError("Patch file entry has no hunks.", code="patch_parse_error", metadata={"path": new_path})
        patches.append(FilePatch(old_path=old_path, new_path=new_path, hunks=hunks))

    if not patches:
        raise PatchError("Patch did not contain any unified-diff file entries.", code="patch_parse_error")
    return patches


def _parse_hunk(lines: list[str], start_index: int) -> tuple[PatchHunk, int]:
    header = lines[start_index]
    match = HUNK_HEADER_RE.match(header)
    if match is None:
        raise PatchError(f"Invalid hunk header: {header}", code="patch_parse_error")

    old_start = int(match.group("old_start"))
    old_count = int(match.group("old_count") or "1")
    new_start = int(match.group("new_start"))
    new_count = int(match.group("new_count") or "1")
    index = start_index + 1
    hunk_lines: list[PatchLine] = []
    while index < len(lines):
        line = lines[index]
        if line.startswith("@@ ") or line.startswith("diff --git ") or _looks_like_next_file_header(lines, index):
            break
        if line.startswith("\\"):
            index += 1
            continue
        if not line or line[0] not in {" ", "+", "-"}:
            raise PatchError(f"Invalid hunk line: {line}", code="patch_parse_error")
        hunk_lines.append(PatchLine(prefix=line[0], text=line[1:]))
        index += 1

    actual_old_count = sum(1 for line in hunk_lines if line.prefix in {" ", "-"})
    actual_new_count = sum(1 for line in hunk_lines if line.prefix in {" ", "+"})
    if actual_old_count != old_count or actual_new_count != new_count:
        raise PatchError(
            "Hunk line counts do not match the hunk header.",
            code="patch_parse_error",
            metadata={
                "header": header,
                "expected_old_count": old_count,
                "actual_old_count": actual_old_count,
                "expected_new_count": new_count,
                "actual_new_count": actual_new_count,
            },
        )

    return PatchHunk(old_start=old_start, old_count=old_count, new_start=new_start, new_count=new_count, lines=hunk_lines), index


def _prepare_patch_writes(patches: list[FilePatch], project_root: Path) -> list[PendingPatchWrite]:
    root = project_root.resolve()
    pending_writes: list[PendingPatchWrite] = []
    seen_paths: set[Path] = set()

    for patch in patches:
        target_path = resolve_inside_root(root, patch.new_path or "")
        relative_path = _relative_path(target_path, root)
        if target_path in seen_paths:
            raise PatchError("Patch touches the same file more than once.", code="patch_parse_error", metadata={"path": relative_path})
        seen_paths.add(target_path)

        safety_error = safe_write_error(target_path, root)
        if safety_error:
            raise PatchError(safety_error, code="unsafe_path", metadata={"path": relative_path})

        is_create = patch.old_path is None
        if is_create and target_path.exists():
            raise PatchError("Patch creates a file that already exists.", code="file_exists", metadata={"path": relative_path})
        if not is_create and (not target_path.exists() or not target_path.is_file()):
            raise PatchError("Patch modifies a file that does not exist.", code="not_found", metadata={"path": relative_path})

        old_text: str | None = None
        old_lines: list[str] = []
        if not is_create:
            if target_path.stat().st_size > MAX_TEXT_FILE_BYTES:
                raise PatchError("File is too large to patch safely.", code="file_too_large", metadata={"path": relative_path})
            try:
                old_text = target_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise PatchError("File is not valid UTF-8 text.", code="not_text", metadata={"path": relative_path}) from exc
            old_lines = old_text.splitlines()

        try:
            new_lines = _apply_file_patch(old_lines, patch)
        except PatchError as exc:
            metadata = {"path": relative_path, **exc.metadata}
            message = str(exc)
            reread_hint = metadata.get("reread_hint")
            if exc.code == "patch_context_mismatch" and isinstance(reread_hint, dict):
                reread_hint = {"path": relative_path, **reread_hint}
                metadata["reread_hint"] = reread_hint
                message = (
                    f"Patch {metadata['mismatch_type']} mismatch in {relative_path} "
                    f"at hunk {metadata['hunk_index']}, target line "
                    f"{metadata['target_line']}. Reread {relative_path} lines "
                    f"{reread_hint['start_line']}-{reread_hint['end_line']} and "
                    "rebuild this hunk from the current file text."
                )
            raise PatchError(message, code=exc.code, metadata=metadata) from exc

        new_text = "\n".join(new_lines)
        if new_lines:
            new_text += "\n"
        pending_writes.append(
            PendingPatchWrite(
                path=target_path,
                relative_path=relative_path,
                status="created" if is_create else "modified",
                old_text=old_text,
                new_text=new_text,
            )
        )

    return pending_writes


def _apply_file_patch(old_lines: list[str], patch: FilePatch) -> list[str]:
    output: list[str] = []
    cursor = 0
    for hunk_index, hunk in enumerate(patch.hunks, start=1):
        start_index = 0 if hunk.old_start == 0 else hunk.old_start - 1
        if start_index < cursor or start_index > len(old_lines):
            raise _patch_context_mismatch(
                mismatch_type="location",
                hunk_index=hunk_index,
                target_line=start_index + 1,
                file_line_count=len(old_lines),
            )
        output.extend(old_lines[cursor:start_index])
        old_index = start_index
        for line in hunk.lines:
            if line.prefix == " ":
                if old_index >= len(old_lines) or old_lines[old_index] != line.text:
                    raise _patch_context_mismatch(
                        mismatch_type="context",
                        hunk_index=hunk_index,
                        target_line=old_index + 1,
                        file_line_count=len(old_lines),
                    )
                output.append(line.text)
                old_index += 1
            elif line.prefix == "-":
                if old_index >= len(old_lines) or old_lines[old_index] != line.text:
                    raise _patch_context_mismatch(
                        mismatch_type="removal",
                        hunk_index=hunk_index,
                        target_line=old_index + 1,
                        file_line_count=len(old_lines),
                    )
                old_index += 1
            elif line.prefix == "+":
                output.append(line.text)
        cursor = old_index
    output.extend(old_lines[cursor:])
    return output


def _patch_context_mismatch(
    *,
    mismatch_type: str,
    hunk_index: int,
    target_line: int,
    file_line_count: int,
) -> PatchError:
    anchor_line = min(max(target_line, 1), max(file_line_count, 1))
    return PatchError(
        "Patch hunk did not match the current file.",
        code="patch_context_mismatch",
        metadata={
            "hunk_index": hunk_index,
            "target_line": target_line,
            "mismatch_type": mismatch_type,
            "file_line_count": file_line_count,
            "reread_hint": {
                "start_line": max(1, anchor_line - PATCH_REREAD_CONTEXT_LINES),
                "end_line": min(
                    max(file_line_count, 1),
                    anchor_line + PATCH_REREAD_CONTEXT_LINES,
                ),
            },
        },
    )


def _parse_patch_path(raw_path: str) -> str | None:
    path = raw_path.strip().split("\t", 1)[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    if not path:
        raise PatchError("Patch path is empty.", code="patch_parse_error")
    return path


def _is_unsupported_patch_header(line: str) -> bool:
    return line.startswith(("deleted file mode ", "rename from ", "rename to ", "copy from ", "copy to "))


def _looks_like_next_file_header(lines: list[str], index: int) -> bool:
    return index + 1 < len(lines) and lines[index].startswith("--- ") and lines[index + 1].startswith("+++ ")


def resolve_inside_root(project_root: Path, raw_path: str) -> Path:
    root = project_root.resolve()
    candidate = (root / raw_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path is outside the project root")
    return candidate


def safe_write_error(path: Path, project_root: Path) -> str | None:
    """Return a user-facing reason when a path should not be written by tools."""
    root = project_root.resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return "Path is outside the project root"

    parts = relative.parts
    lowered_parts = [part.lower() for part in parts]
    for part in lowered_parts:
        if part in UNSAFE_WRITE_DIRS:
            return f"Refusing to write inside unsafe directory: {part}"

    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in UNSAFE_SECRET_NAMES or name.startswith(".env."):
        return "Refusing to write secret or credential file"
    if _looks_like_sqlite_state(name):
        return "Refusing to write SQLite/database file"
    if suffix in UNSAFE_SECRET_SUFFIXES and _looks_secret_like_name(name):
        return "Refusing to write secret or credential file"
    return None


def safe_read_error(
    path: Path,
    project_root: Path,
    read_policy: FileReadPolicy = DEFAULT_FILE_READ_POLICY,
    *,
    requested_path: str | None = None,
) -> str | None:
    """Return a host-policy reason when model-facing file tools must not read a path."""
    root = project_root.resolve()
    try:
        resolved_relative = path.resolve().relative_to(root)
    except ValueError:
        return "Path is outside the project root"

    if read_policy.allow_sensitive_paths:
        return None

    if requested_path is not None and _is_sensitive_read_parts(Path(requested_path).parts):
        return "Refusing to read a sensitive file or runtime path."

    try:
        lexical_relative = path.relative_to(root)
    except ValueError:
        lexical_relative = resolved_relative
    if _is_sensitive_read_parts(lexical_relative.parts) or _is_sensitive_read_parts(resolved_relative.parts):
        return "Refusing to read a sensitive file or runtime path."
    return None


def _is_ignored(path: Path, project_root: Path) -> bool:
    relative_parts = path.relative_to(project_root).parts
    return any(part in IGNORED_DIRS for part in relative_parts)


def _is_sensitive_read_parts(parts: tuple[str, ...]) -> bool:
    lowered_parts = tuple(part.lower() for part in parts if part not in {"", "."})
    if not lowered_parts:
        return False
    if any(part in SENSITIVE_READ_DIRS for part in lowered_parts):
        return True

    for index, part in enumerate(lowered_parts):
        if part != ".chulk":
            continue
        runtime_parts = lowered_parts[index + 1 :]
        if not runtime_parts or runtime_parts[0] != "skills":
            return True

    return _is_sensitive_read_name(lowered_parts[-1])


def _is_sensitive_read_name(name: str) -> bool:
    candidate_names = [name]
    unwrapped_name = name
    while True:
        if unwrapped_name.endswith("~"):
            unwrapped_name = unwrapped_name[:-1]
        else:
            backup_suffix = next((suffix for suffix in SENSITIVE_BACKUP_SUFFIXES if unwrapped_name.endswith(suffix)), None)
            if backup_suffix is None:
                break
            unwrapped_name = unwrapped_name[: -len(backup_suffix)]
        candidate_names.append(unwrapped_name)

    return any(_is_base_sensitive_read_name(candidate_name) for candidate_name in candidate_names)


def _is_base_sensitive_read_name(name: str) -> bool:
    if _is_secret_env_name(name) or name in UNSAFE_SECRET_NAMES:
        return True
    if _looks_like_sqlite_state(name) or Path(name).suffix in SENSITIVE_KEY_SUFFIXES:
        return True
    suffix = Path(name).suffix
    return suffix in UNSAFE_SECRET_SUFFIXES and (_looks_secret_like_name(name) or _looks_key_like_name(name))


def _is_secret_env_name(name: str) -> bool:
    if name == ".env" or name == ".envrc":
        return True
    if not name.startswith(".env."):
        return False
    return not any(name.endswith(suffix) for suffix in SAFE_ENV_TEMPLATE_SUFFIXES)


def _looks_like_sqlite_state(name: str) -> bool:
    if any(name.endswith(suffix) for suffix in UNSAFE_SQLITE_SUFFIXES):
        return True
    database_name, separator, sidecar = name.rpartition("-")
    return bool(
        separator
        and sidecar in SQLITE_SIDECAR_SUFFIXES
        and any(database_name.endswith(suffix) for suffix in UNSAFE_SQLITE_SUFFIXES)
    )


def _looks_secret_like_name(name: str) -> bool:
    stem = name.rsplit(".", 1)[0]
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    parts = set(filter(None, normalized.split("_")))
    return (
        normalized in UNSAFE_SECRET_MARKERS
        or "api_key" in normalized
        or "apikey" in parts
        or bool(parts & UNSAFE_SECRET_MARKERS)
    )


def _looks_key_like_name(name: str) -> bool:
    stem = name.rsplit(".", 1)[0]
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return normalized in UNSAFE_KEY_MARKERS or "private_key" in normalized or "secret_key" in normalized


def _relative_path(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
