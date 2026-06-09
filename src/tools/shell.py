"""Shell command tool placeholder.

This tool is dangerous and must enforce safety in Python before real use.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Any

from src.tools.registry import Tool, ToolResult


MAX_OUTPUT_CHARS = 20_000
DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\s+-[^;&|]*r[^;&|]*f\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:"),
    re.compile(r">\s*/(?:etc|bin|sbin|usr|System|Library)\b"),
]


def shell_tool(project_root: Path, timeout_seconds: int = 10) -> Tool:
    """Create the shell command tool."""
    return Tool(
        name="run_cmd",
        description="Run a shell command in the project directory with timeout, output capture, and safety blocking.",
        args_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run from the project root."},
                "timeout_seconds": {"type": "integer", "description": "Optional timeout in seconds."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        callable=lambda arguments: run_shell_command(arguments, project_root, timeout_seconds),
        timeout_seconds=timeout_seconds,
        requires_confirmation=True,
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
            stdout=_truncate(exc.stdout or ""),
            stderr=_truncate(exc.stderr or ""),
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
        stdout=_truncate(completed.stdout),
        stderr=_truncate(completed.stderr),
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
    if _redirects_outside_root(command, root):
        return "command redirects output outside the project root"
    return None


def _redirects_outside_root(command: str, root: Path) -> bool:
    for match in re.finditer(r"(?:\d?>{1,2}|&>)\s*([^\s;&|]+)", command):
        raw_path = match.group(1).strip("'\"")
        if raw_path.startswith("$"):
            return True
        candidate = (root / raw_path).resolve() if not raw_path.startswith("/") else Path(raw_path).resolve()
        if candidate != root and root not in candidate.parents:
            return True
    return False


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"
