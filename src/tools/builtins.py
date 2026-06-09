"""Built-in tool registration."""

from __future__ import annotations

from pathlib import Path

from src.tools.calculator import calculator_tool
from src.tools.files import list_files_tool, read_file_tool, search_files_tool, write_file_tool
from src.tools.registry import ToolRegistry
from src.tools.shell import shell_tool


def create_default_tool_registry(project_root: Path, shell_timeout_seconds: int = 10) -> ToolRegistry:
    """Create the default tool registry for the agent runtime."""
    registry = ToolRegistry()
    registry.register(calculator_tool())
    registry.register(shell_tool(project_root, timeout_seconds=shell_timeout_seconds))
    registry.register(read_file_tool(project_root))
    registry.register(write_file_tool(project_root))
    registry.register(list_files_tool(project_root))
    registry.register(search_files_tool(project_root))
    return registry
