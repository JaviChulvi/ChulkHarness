"""Tool registration and execution primitives."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by a tool."""

    tool_name: str
    success: bool
    observation: str
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    """A callable action the agent may request."""

    name: str
    description: str
    args_schema: dict[str, Any]
    callable: Callable[[dict[str, Any]], ToolResult]
    requires_confirmation: bool = False
    timeout_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        return tool.callable(arguments)
