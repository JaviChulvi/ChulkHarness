"""Shell command tool placeholder.

This tool is dangerous and must enforce safety in Python before real use.
"""

from typing import Any

from agent.tool_registry import ToolResult


def run_shell_command(arguments: dict[str, Any]) -> ToolResult:
    """Run a shell command.

    Phase 2 will implement timeout, output capture, command blocking, and logging.
    """
    command = arguments.get("command", "")
    return ToolResult(
        tool_name="run_cmd",
        success=False,
        observation=f"Shell tool is not implemented yet. Requested command: {command!r}",
        error="not_implemented",
    )
