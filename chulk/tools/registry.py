"""Tool registration and execution primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from typing import Any

from chulk.tools.schema import ToolValidationError, ToolValidationIssue, validate_tool_arguments, validate_tool_schema


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

    def to_observation(self) -> str:
        """Return the safe observation text shown back to the model."""
        status = "success" if self.success else "error"
        parts = [f"Tool {self.tool_name} finished with {status}.", self.observation]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        if self.exit_code is not None:
            parts.append(f"exit_code: {self.exit_code}")
        if self.error:
            parts.append(f"error: {self.error}")
        return "\n".join(parts)


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
        self.call_log: list[dict[str, Any]] = []

    def register(self, tool: Tool) -> None:
        if not tool.name or not tool.name.replace("_", "").isalnum():
            raise ValueError("Tool names must be non-empty and contain only letters, numbers, and underscores")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        validate_tool_schema(tool.name, tool.args_schema)
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def tool_descriptions_for_prompt(self) -> str:
        """Return JSON tool descriptions suitable for prompt injection."""
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "arguments": tool.args_schema,
                "requires_confirmation": tool.requires_confirmation,
            }
            for tool in self.list_tools()
        ]
        return json.dumps(tools, indent=2, sort_keys=True)

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            tool = self.get(name)
        except KeyError as exc:
            available_tools = sorted(self._tools)
            result = ToolResult(
                tool_name=name,
                success=False,
                observation=_format_unknown_tool_observation(name, available_tools),
                error="unknown_tool",
                metadata={
                    "requested_tool_name": name,
                    "available_tools": available_tools,
                    "exception": str(exc),
                },
            )
            self._log_call(name, arguments, result)
            return result

        try:
            self._validate_arguments(tool, arguments)
            result = tool.callable(arguments)
            if not isinstance(result, ToolResult):
                result = ToolResult(tool_name=tool.name, success=True, observation=str(result))
        except ToolValidationError as exc:
            result = ToolResult(
                tool_name=tool.name,
                success=False,
                observation=_format_invalid_arguments_observation(tool, exc.issues),
                error="invalid_arguments",
                metadata={
                    "validation_errors": [issue.to_dict() for issue in exc.issues],
                    "args_schema": tool.args_schema,
                },
            )
        except Exception as exc:
            result = ToolResult(
                tool_name=tool.name,
                success=False,
                observation=(
                    f"Tool execution failed for {tool.name}: {exc}. "
                    "Retry only if corrected arguments or a safer alternative would change the outcome."
                ),
                error=str(exc),
                metadata={"exception_type": type(exc).__name__},
            )

        self._log_call(name, arguments, result)
        return result

    def _validate_arguments(self, tool: Tool, arguments: dict[str, Any]) -> None:
        validate_tool_arguments(tool.name, arguments, tool.args_schema or {})

    def _log_call(self, name: str, arguments: dict[str, Any], result: ToolResult) -> None:
        self.call_log.append(
            {
                "tool_name": name,
                "arguments": arguments,
                "success": result.success,
                "error": result.error,
                "observation": result.observation,
            }
        )


def _format_invalid_arguments_observation(tool: Tool, issues: list[ToolValidationIssue]) -> str:
    lines = [
        f"Tool call failed before execution because arguments for {tool.name} were invalid.",
        "Validation errors:",
    ]
    lines.extend(f"- {issue.to_prompt_line()}" for issue in issues)
    lines.append("Retry with arguments_json that matches this tool schema, or answer directly if no tool is needed.")
    return "\n".join(lines)


def _format_unknown_tool_observation(name: str, available_tools: list[str]) -> str:
    available = ", ".join(available_tools) if available_tools else "none"
    return (
        f"Unknown tool: {name}. Tool call failed before execution because {name} is not a registered tool. "
        f"Available tools: {available}. Retry with one of the available tool names, or answer directly if no tool is needed."
    )
