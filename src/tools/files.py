"""Project-root-bound file tools."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.tools.registry import Tool, ToolResult


IGNORED_DIRS = {".git", ".venv", ".conda", "__pycache__", ".pytest_cache", "chulkharness.egg-info"}
MAX_TEXT_FILE_BYTES = 200_000


def read_file_tool(project_root: Path) -> Tool:
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
        callable=lambda arguments: read_file(arguments, project_root),
    )


def write_file_tool(project_root: Path) -> Tool:
    return Tool(
        name="write_file",
        description="Write a UTF-8 text file inside the project directory. Existing files require overwrite=true.",
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
        callable=lambda arguments: write_file(arguments, project_root),
    )


def list_files_tool(project_root: Path) -> Tool:
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
        callable=lambda arguments: list_files(arguments, project_root),
    )


def search_files_tool(project_root: Path) -> Tool:
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
        callable=lambda arguments: search_files(arguments, project_root),
    )


def read_file(arguments: dict[str, Any], project_root: Path) -> ToolResult:
    path = _resolve_inside_root(project_root, arguments["path"])
    if not path.exists() or not path.is_file():
        return ToolResult("read_file", False, f"File not found: {path.relative_to(project_root)}", error="not_found")
    if path.stat().st_size > MAX_TEXT_FILE_BYTES:
        return ToolResult("read_file", False, "File is too large to read safely.", error="file_too_large")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult("read_file", False, "File is not valid UTF-8 text.", error="not_text")
    return ToolResult("read_file", True, content, metadata={"path": str(path.relative_to(project_root))})


def write_file(arguments: dict[str, Any], project_root: Path) -> ToolResult:
    path = _resolve_inside_root(project_root, arguments["path"])
    content = arguments["content"]
    overwrite = arguments.get("overwrite", False)
    if path.exists() and not overwrite:
        return ToolResult("write_file", False, "File already exists. Pass overwrite=true to replace it.", error="exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return ToolResult(
        "write_file",
        True,
        f"Wrote {len(content.encode('utf-8'))} bytes to {path.relative_to(project_root)}.",
        metadata={"path": str(path.relative_to(project_root))},
    )


def list_files(arguments: dict[str, Any], project_root: Path) -> ToolResult:
    directory = _resolve_inside_root(project_root, arguments.get("path", "."))
    pattern = arguments.get("pattern", "*")
    recursive = arguments.get("recursive", False)
    max_results = min(arguments.get("max_results", 100), 500)

    if not directory.exists() or not directory.is_dir():
        return ToolResult("list_files", False, f"Directory not found: {directory.relative_to(project_root)}", error="not_found")

    iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
    results: list[str] = []
    for path in iterator:
        if _is_ignored(path, project_root) or not path.is_file():
            continue
        results.append(str(path.relative_to(project_root)))
        if len(results) >= max_results:
            break
    return ToolResult("list_files", True, "\n".join(sorted(results)) or "No files found.")


def search_files(arguments: dict[str, Any], project_root: Path) -> ToolResult:
    query = arguments["query"]
    directory = _resolve_inside_root(project_root, arguments.get("path", "."))
    pattern = arguments.get("pattern", "*")
    max_results = min(arguments.get("max_results", 100), 500)
    if not directory.exists() or not directory.is_dir():
        return ToolResult("search_files", False, f"Directory not found: {directory.relative_to(project_root)}", error="not_found")

    if shutil.which("rg"):
        return _search_with_rg(project_root, directory, query, pattern, max_results)
    return _search_with_python(project_root, directory, query, pattern, max_results)


def _search_with_rg(project_root: Path, directory: Path, query: str, pattern: str, max_results: int) -> ToolResult:
    completed = subprocess.run(
        ["rg", "--line-number", "--color", "never", "--glob", pattern, "--", query, str(directory)],
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        return ToolResult("search_files", False, "Search failed.", stderr=completed.stderr, error="search_failed")
    lines = completed.stdout.splitlines()[:max_results]
    normalized = []
    for line in lines:
        normalized.append(line.replace(str(project_root) + "/", ""))
    return ToolResult("search_files", True, "\n".join(normalized) or "No matches found.")


def _search_with_python(project_root: Path, directory: Path, query: str, pattern: str, max_results: int) -> ToolResult:
    results: list[str] = []
    for path in directory.rglob(pattern):
        if _is_ignored(path, project_root) or not path.is_file() or path.stat().st_size > MAX_TEXT_FILE_BYTES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for index, line in enumerate(lines, start=1):
            if query in line:
                results.append(f"{path.relative_to(project_root)}:{index}:{line}")
                if len(results) >= max_results:
                    return ToolResult("search_files", True, "\n".join(results))
    return ToolResult("search_files", True, "\n".join(results) or "No matches found.")


def _resolve_inside_root(project_root: Path, raw_path: str) -> Path:
    root = project_root.resolve()
    candidate = (root / raw_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path is outside the project root")
    return candidate


def _is_ignored(path: Path, project_root: Path) -> bool:
    relative_parts = path.relative_to(project_root).parts
    return any(part in IGNORED_DIRS for part in relative_parts)
