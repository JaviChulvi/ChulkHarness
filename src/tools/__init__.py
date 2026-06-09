"""Tool implementations and registry primitives."""

from src.tools.builtins import create_default_tool_registry
from src.tools.calculator import calculator_tool
from src.tools.files import list_files_tool, read_file_tool, search_files_tool, write_file_tool
from src.tools.registry import Tool, ToolRegistry, ToolResult
from src.tools.shell import run_shell_command, shell_tool

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "calculator_tool",
    "create_default_tool_registry",
    "list_files_tool",
    "read_file_tool",
    "run_shell_command",
    "search_files_tool",
    "shell_tool",
    "write_file_tool",
]
