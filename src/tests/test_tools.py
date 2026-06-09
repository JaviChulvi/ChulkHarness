"""Tests for built-in tools and registry behavior."""

from pathlib import Path
import sys

from src.tools import Tool, ToolRegistry, calculator_tool, create_default_tool_registry
from src.tools.files import list_files_tool, read_file_tool, search_files_tool, write_file_tool
from src.tools.registry import ToolResult
from src.tools.shell import run_shell_command


def test_registry_descriptions_include_registered_tool():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    description = registry.tool_descriptions_for_prompt()

    assert "calculator" in description
    assert "expression" in description


def test_registry_logs_tool_calls():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    registry.run("calculator", {"expression": "1 + 1"})

    assert registry.call_log == [
        {
            "tool_name": "calculator",
            "arguments": {"expression": "1 + 1"},
            "success": True,
            "error": None,
            "observation": "1 + 1 = 2",
        }
    ]


def test_registry_validates_required_arguments():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    result = registry.run("calculator", {})

    assert not result.success
    assert result.error == "invalid_arguments"
    assert "Missing required argument" in result.observation


def test_registry_catches_tool_exceptions():
    def broken_tool(_arguments):
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="broken",
            description="Broken test tool",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=broken_tool,
        )
    )

    result = registry.run("broken", {})

    assert not result.success
    assert result.observation == "Tool execution failed."
    assert result.error == "boom"


def test_registry_returns_safe_unknown_tool_error():
    registry = ToolRegistry()

    result = registry.run("missing", {})

    assert not result.success
    assert "Unknown tool" in result.observation


def test_calculator_evaluates_arithmetic():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    result = registry.run("calculator", {"expression": "(2 + 3) * 4"})

    assert result.success
    assert result.observation == "(2 + 3) * 4 = 20"


def test_calculator_rejects_non_arithmetic_expression():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    result = registry.run("calculator", {"expression": "__import__('os').system('echo nope')"})

    assert not result.success
    assert result.error == "invalid_expression"


def test_shell_command_success(tmp_path):
    result = run_shell_command({"command": "printf hello"}, tmp_path)

    assert result.success
    assert result.stdout == "hello"
    assert result.exit_code == 0
    assert result.metadata["cwd"] == str(tmp_path.resolve())
    assert result.metadata["stdout_length"] == 5


def test_shell_blocks_destructive_command(tmp_path):
    result = run_shell_command({"command": "rm -rf /"}, tmp_path)

    assert not result.success
    assert result.error == "blocked_command"


def test_shell_blocks_output_redirection_outside_root(tmp_path):
    result = run_shell_command({"command": "printf hello > ../outside.txt"}, tmp_path)

    assert not result.success
    assert result.error == "blocked_command"
    assert "outside the project root" in result.observation


def test_shell_command_timeout(tmp_path):
    command = f"{sys.executable} -c 'import time; time.sleep(2)'"

    result = run_shell_command({"command": command, "timeout_seconds": 1}, tmp_path, default_timeout_seconds=1)

    assert not result.success
    assert result.error == "timeout"


def test_file_tools_read_write_list_and_search(tmp_path):
    registry = ToolRegistry()
    registry.register(write_file_tool(tmp_path))
    registry.register(read_file_tool(tmp_path))
    registry.register(list_files_tool(tmp_path))
    registry.register(search_files_tool(tmp_path))

    write_result = registry.run("write_file", {"path": "notes/example.txt", "content": "hello chulk"})
    read_result = registry.run("read_file", {"path": "notes/example.txt"})
    list_result = registry.run("list_files", {"path": ".", "pattern": "*.txt", "recursive": True})
    search_result = registry.run("search_files", {"query": "chulk", "path": ".", "pattern": "*.txt"})

    assert write_result.success
    assert read_result.success
    assert read_result.observation == "hello chulk"
    assert "notes/example.txt" in list_result.observation
    assert "notes/example.txt" in search_result.observation


def test_file_tools_block_path_traversal(tmp_path):
    registry = ToolRegistry()
    registry.register(read_file_tool(tmp_path))

    result = registry.run("read_file", {"path": "../outside.txt"})

    assert not result.success
    assert "outside the project root" in (result.error or result.observation)


def test_default_tool_registry_contains_builtins(tmp_path):
    registry = create_default_tool_registry(tmp_path)
    names = {tool.name for tool in registry.list_tools()}

    assert {"calculator", "run_cmd", "read_file", "write_file", "list_files", "search_files"} <= names
