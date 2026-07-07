"""Shell command tool placeholder.

This tool is dangerous and must enforce safety in Python before real use.
"""

from __future__ import annotations

from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.registry import Tool, ToolResult


DESTRUCTIVE_PATTERNS = [
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:"),
    re.compile(r">\s*/(?:etc|bin|sbin|usr|System|Library)\b"),
]

_SHELL_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;&|]")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_COMMAND_WRAPPERS = {"sudo", "command", "exec"}
_LONG_RECURSIVE_FLAGS = {"--recursive"}
_LONG_FORCE_FLAGS = {"--force"}


def shell_tool(project_root: Path, timeout_seconds: int = 10) -> Tool:
    """Create the shell command tool."""
    return Tool(
        name="run_cmd",
        description="Run a shell command in the project directory with timeout, output capture, and safety blocking.",
        args_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to run from the project root.",
                    "minLength": 1,
                    "maxLength": 4000,
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout in seconds.",
                    "minimum": 1,
                    "maximum": timeout_seconds,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        callable=lambda arguments: run_shell_command(arguments, project_root, timeout_seconds),
        timeout_seconds=timeout_seconds,
        run_in_executor=True,
        requires_confirmation=True,
        permission_level=ToolPermissionLevel.SHELL,
    )


def run_shell_command(
    arguments: dict[str, Any],
    project_root: Path | None = None,
    default_timeout_seconds: int = 10,
) -> ToolResult:
    """Run a shell command with basic local safety controls."""
    root = (project_root or Path.cwd()).resolve()
    command = arguments["command"]
    timeout_seconds = min(arguments.get("timeout_seconds", default_timeout_seconds), default_timeout_seconds)

    blocked_reason = _blocked_reason(command, root)
    if blocked_reason:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation=f"Blocked command: {blocked_reason}",
            error="blocked_command",
            metadata={"command": command, "cwd": str(root)},
        )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            tool_name="run_cmd",
            success=False,
            observation=f"Command timed out after {timeout_seconds} seconds.",
            stdout=_coerce_output_text(exc.stdout),
            stderr=_coerce_output_text(exc.stderr),
            error="timeout",
            metadata={
                "command": command,
                "cwd": str(root),
                "timeout_seconds": timeout_seconds,
                "stdout_length": len(exc.stdout or ""),
                "stderr_length": len(exc.stderr or ""),
            },
        )

    return ToolResult(
        tool_name="run_cmd",
        success=completed.returncode == 0,
        observation="Command completed." if completed.returncode == 0 else "Command failed.",
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        error=None if completed.returncode == 0 else "nonzero_exit",
        metadata={
            "command": command,
            "cwd": str(root),
            "timeout_seconds": timeout_seconds,
            "exit_code": completed.returncode,
            "stdout_length": len(completed.stdout),
            "stderr_length": len(completed.stderr),
        },
    )


def _blocked_reason(command: str, root: Path) -> str | None:
    lowered = command.strip().lower()
    for pattern in DESTRUCTIVE_PATTERNS:
        if pattern.search(lowered):
            return "command matches a destructive pattern"
    if _has_recursive_force_rm(lowered):
        return "command matches a destructive pattern"
    if _redirects_outside_root(command, root):
        return "command redirects output outside the project root"
    return None


def _has_recursive_force_rm(lowered_command: str) -> bool:
    """Detect `rm` invoked with both a recursive and a force flag, in any order."""
    for segment in _SHELL_SEGMENT_SPLIT.split(lowered_command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()

        idx = 0
        while idx < len(tokens) and (tokens[idx] in _COMMAND_WRAPPERS or _ENV_ASSIGNMENT.match(tokens[idx])):
            idx += 1
        if idx >= len(tokens):
            continue
        if tokens[idx].rsplit("/", 1)[-1] != "rm":
            continue

        has_recursive = False
        has_force = False
        for token in tokens[idx + 1 :]:
            if token == "--":
                break
            if token.startswith("--"):
                flag = token.split("=", 1)[0]
                if flag in _LONG_RECURSIVE_FLAGS:
                    has_recursive = True
                elif flag in _LONG_FORCE_FLAGS:
                    has_force = True
            elif token.startswith("-") and len(token) > 1:
                flags = token[1:]
                if "r" in flags:
                    has_recursive = True
                if "f" in flags:
                    has_force = True

        if has_recursive and has_force:
            return True
    return False


def _coerce_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _redirects_outside_root(command: str, root: Path) -> bool:
    for match in re.finditer(r"(?:\d?>{1,2}|&>)\s*([^\s;&|]+)", command):
        raw_path = match.group(1).strip("'\"")
        if raw_path.startswith("$"):
            return True
        candidate = (root / raw_path).resolve() if not raw_path.startswith("/") else Path(raw_path).resolve()
        if candidate != root and root not in candidate.parents:
            return True
    return False
