"""Tool registration and execution primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from typing import Any


class ToolValidationError(ValueError):
    """Raised when tool arguments do not match the declared schema."""


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
            result = ToolResult(
                tool_name=name,
                success=False,
                observation=f"Unknown tool: {name}",
                error=str(exc),
            )
            self._log_call(name, arguments, result)
            return result

        try:
            self._validate_arguments(tool, arguments)
            result = tool.callable(arguments)
            if not isinstance(result, ToolResult):
                result = ToolResult(tool_name=tool.name, success=True, observation=str(result))
        except ToolValidationError as exc:
            result = ToolResult(tool_name=tool.name, success=False, observation=str(exc), error="invalid_arguments")
        except Exception as exc:
            result = ToolResult(tool_name=tool.name, success=False, observation="Tool execution failed.", error=str(exc))

        self._log_call(name, arguments, result)
        return result

    def _validate_arguments(self, tool: Tool, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolValidationError("Tool arguments must be an object")

        schema = tool.args_schema or {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        additional_allowed = schema.get("additionalProperties", True)

        for field_name in required:
            if field_name not in arguments:
                raise ToolValidationError(f"Missing required argument: {field_name}")

        if not additional_allowed:
            unknown = sorted(set(arguments) - set(properties))
            if unknown:
                raise ToolValidationError(f"Unknown arguments: {', '.join(unknown)}")

        for field_name, value in arguments.items():
            if field_name not in properties:
                continue
            expected = properties[field_name].get("type")
            if expected and not _matches_json_type(value, expected):
                raise ToolValidationError(f"Argument {field_name} must be {expected}")

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


def _matches_json_type(value: Any, expected: str | list[str]) -> bool:
    expected_types = [expected] if isinstance(expected, str) else expected
    return any(_matches_single_json_type(value, item) for item in expected_types)


def _matches_single_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True
