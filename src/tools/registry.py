"""Tool registration and execution primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from typing import Any


class ToolValidationError(ValueError):
    """Raised when tool arguments do not match the declared schema."""

    def __init__(self, tool_name: str, issues: list["ToolValidationIssue"], args_schema: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.issues = issues
        self.args_schema = args_schema
        super().__init__(_format_validation_summary(tool_name, issues))


@dataclass(frozen=True)
class ToolValidationIssue:
    """One validation problem for model-provided tool arguments."""

    path: str
    message: str
    expected: str | None = None
    actual: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "path": self.path,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }

    def to_prompt_line(self) -> str:
        detail = f"{self.path}: {self.message}"
        if self.expected:
            detail += f" Expected: {self.expected}."
        if self.actual:
            detail += f" Got: {self.actual}."
        return detail


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
        schema = tool.args_schema or {}
        issues: list[ToolValidationIssue] = []

        if not isinstance(arguments, dict):
            issues.append(
                ToolValidationIssue(
                    path="$",
                    message="tool arguments must be a JSON object",
                    expected="object",
                    actual=_json_type_name(arguments),
                )
            )
        else:
            _validate_object("$", arguments, schema, issues)

        if issues:
            raise ToolValidationError(tool.name, issues, schema)

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


def _validate_object(path: str, value: dict[str, Any], schema: dict[str, Any], issues: list[ToolValidationIssue]) -> None:
    expected = schema.get("type")
    if expected and not _matches_json_type(value, expected):
        issues.append(
            ToolValidationIssue(
                path=path,
                message="value has the wrong type",
                expected=_format_expected_type(expected),
                actual=_json_type_name(value),
            )
        )
        return

    required = schema.get("required", [])
    properties = schema.get("properties", {})
    additional_allowed = schema.get("additionalProperties", True)

    for field_name in required:
        if field_name not in value:
            issues.append(ToolValidationIssue(path=_child_path(path, field_name), message="Missing required argument"))

    if not additional_allowed:
        for field_name in sorted(set(value) - set(properties)):
            issues.append(ToolValidationIssue(path=_child_path(path, field_name), message="Unknown argument"))

    for field_name, item in value.items():
        field_schema = properties.get(field_name)
        if field_schema is None:
            continue
        _validate_value(_child_path(path, field_name), item, field_schema, issues)


def _validate_value(path: str, value: Any, schema: dict[str, Any], issues: list[ToolValidationIssue]) -> None:
    expected = schema.get("type")
    if expected and not _matches_json_type(value, expected):
        issues.append(
            ToolValidationIssue(
                path=path,
                message="value has the wrong type",
                expected=_format_expected_type(expected),
                actual=_json_type_name(value),
            )
        )
        return

    if "enum" in schema and value not in schema["enum"]:
        issues.append(
            ToolValidationIssue(
                path=path,
                message="value is not one of the allowed options",
                expected=", ".join(str(item) for item in schema["enum"]),
                actual=repr(value),
            )
        )

    if isinstance(value, str):
        _validate_string(path, value, schema, issues)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        _validate_number(path, value, schema, issues)
    elif isinstance(value, list):
        _validate_array(path, value, schema, issues)
    elif isinstance(value, dict):
        _validate_object(path, value, schema, issues)


def _validate_string(path: str, value: str, schema: dict[str, Any], issues: list[ToolValidationIssue]) -> None:
    min_length = schema.get("minLength")
    max_length = schema.get("maxLength")
    if isinstance(min_length, int) and len(value) < min_length:
        issues.append(
            ToolValidationIssue(
                path=path,
                message="string is too short",
                expected=f"at least {min_length} characters",
                actual=f"{len(value)} characters",
            )
        )
    if isinstance(max_length, int) and len(value) > max_length:
        issues.append(
            ToolValidationIssue(
                path=path,
                message="string is too long",
                expected=f"at most {max_length} characters",
                actual=f"{len(value)} characters",
            )
        )


def _validate_number(path: str, value: int | float, schema: dict[str, Any], issues: list[ToolValidationIssue]) -> None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, int | float) and value < minimum:
        issues.append(
            ToolValidationIssue(path=path, message="number is too small", expected=f">= {minimum}", actual=str(value))
        )
    if isinstance(maximum, int | float) and value > maximum:
        issues.append(
            ToolValidationIssue(path=path, message="number is too large", expected=f"<= {maximum}", actual=str(value))
        )


def _validate_array(path: str, value: list[Any], schema: dict[str, Any], issues: list[ToolValidationIssue]) -> None:
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if isinstance(min_items, int) and len(value) < min_items:
        issues.append(
            ToolValidationIssue(
                path=path,
                message="array has too few items",
                expected=f"at least {min_items} items",
                actual=f"{len(value)} items",
            )
        )
    if isinstance(max_items, int) and len(value) > max_items:
        issues.append(
            ToolValidationIssue(
                path=path,
                message="array has too many items",
                expected=f"at most {max_items} items",
                actual=f"{len(value)} items",
            )
        )
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            _validate_value(f"{path}[{index}]", item, item_schema, issues)


def _child_path(parent: str, child: str) -> str:
    return child if parent == "$" else f"{parent}.{child}"


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _format_expected_type(expected: str | list[str]) -> str:
    if isinstance(expected, list):
        return " or ".join(expected)
    return expected


def _format_validation_summary(tool_name: str, issues: list[ToolValidationIssue]) -> str:
    issue_text = "; ".join(issue.to_prompt_line() for issue in issues)
    return f"Invalid arguments for tool {tool_name}: {issue_text}"


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
