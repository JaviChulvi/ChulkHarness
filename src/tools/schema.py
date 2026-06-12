"""JSON-schema subset validation for Chulk tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_JSON_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}


class ToolValidationError(ValueError):
    """Raised when tool arguments do not match the declared schema."""

    def __init__(self, tool_name: str, issues: list["ToolValidationIssue"], args_schema: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.issues = issues
        self.args_schema = args_schema
        super().__init__(format_validation_summary(tool_name, issues))


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


def validate_tool_schema(tool_name: str, schema: dict[str, Any]) -> None:
    """Validate a tool's declared argument schema at registration time."""
    if not isinstance(schema, dict):
        raise ValueError(f"Tool {tool_name} args_schema must be an object")
    issues: list[ToolValidationIssue] = []
    _validate_schema("$", schema, issues, require_object=True)
    if issues:
        issue_text = "; ".join(issue.to_prompt_line() for issue in issues)
        raise ValueError(f"Invalid schema for tool {tool_name}: {issue_text}")


def validate_tool_arguments(tool_name: str, arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate model-provided arguments against a tool schema."""
    issues: list[ToolValidationIssue] = []

    if not isinstance(arguments, dict):
        issues.append(
            ToolValidationIssue(
                path="$",
                message="tool arguments must be a JSON object",
                expected="object",
                actual=json_type_name(arguments),
            )
        )
    else:
        _validate_object("$", arguments, schema, issues)

    if issues:
        raise ToolValidationError(tool_name, issues, schema)


def format_validation_summary(tool_name: str, issues: list[ToolValidationIssue]) -> str:
    """Format validation issues for exceptions."""
    issue_text = "; ".join(issue.to_prompt_line() for issue in issues)
    return f"Invalid arguments for tool {tool_name}: {issue_text}"


def _validate_schema(
    path: str,
    schema: dict[str, Any],
    issues: list[ToolValidationIssue],
    *,
    require_object: bool = False,
) -> None:
    schema_type = schema.get("type")
    if require_object and schema_type != "object":
        issues.append(
            ToolValidationIssue(
                path=_child_path(path, "type"),
                message="root schema type must be object",
                expected="object",
                actual=repr(schema_type),
            )
        )
    if schema_type is not None:
        _validate_schema_type(_child_path(path, "type"), schema_type, issues)

    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, dict):
        issues.append(ToolValidationIssue(path=_child_path(path, "properties"), message="properties must be an object"))
    elif isinstance(properties, dict):
        for field_name, field_schema in properties.items():
            if not isinstance(field_name, str) or not field_name:
                issues.append(
                    ToolValidationIssue(
                        path=_child_path(path, "properties"),
                        message="property names must be strings",
                    )
                )
                continue
            if not isinstance(field_schema, dict):
                issues.append(
                    ToolValidationIssue(
                        path=_child_path(path, field_name),
                        message="property schema must be an object",
                    )
                )
                continue
            _validate_schema(_child_path(path, field_name), field_schema, issues)

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            issues.append(
                ToolValidationIssue(
                    path=_child_path(path, "required"),
                    message="required must be a string list",
                )
            )
        elif isinstance(properties, dict):
            for field_name in required:
                if field_name not in properties:
                    issues.append(
                        ToolValidationIssue(
                            path=_child_path(path, "required"),
                            message=f"required field {field_name} is not declared in properties",
                        )
                    )

    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        issues.append(
            ToolValidationIssue(
                path=_child_path(path, "additionalProperties"),
                message="additionalProperties must be boolean",
            )
        )

    enum = schema.get("enum")
    if enum is not None and not isinstance(enum, list):
        issues.append(ToolValidationIssue(path=_child_path(path, "enum"), message="enum must be a list"))

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, dict):
            issues.append(ToolValidationIssue(path=_child_path(path, "items"), message="items must be an object"))
        else:
            _validate_schema(_child_path(path, "items"), items, issues)


def _validate_schema_type(path: str, schema_type: str | list[str], issues: list[ToolValidationIssue]) -> None:
    schema_types = [schema_type] if isinstance(schema_type, str) else schema_type
    if not isinstance(schema_types, list) or not all(isinstance(item, str) for item in schema_types):
        issues.append(ToolValidationIssue(path=path, message="type must be a string or string list"))
        return
    unknown_types = sorted(set(schema_types) - SUPPORTED_JSON_TYPES)
    if unknown_types:
        issues.append(
            ToolValidationIssue(
                path=path,
                message="type contains unsupported JSON type",
                expected=", ".join(sorted(SUPPORTED_JSON_TYPES)),
                actual=", ".join(unknown_types),
            )
        )


def matches_json_type(value: Any, expected: str | list[str]) -> bool:
    """Return whether a value matches a JSON Schema type declaration."""
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
    if expected and not matches_json_type(value, expected):
        issues.append(
            ToolValidationIssue(
                path=path,
                message="value has the wrong type",
                expected=format_expected_type(expected),
                actual=json_type_name(value),
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
    if expected and not matches_json_type(value, expected):
        issues.append(
            ToolValidationIssue(
                path=path,
                message="value has the wrong type",
                expected=format_expected_type(expected),
                actual=json_type_name(value),
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


def json_type_name(value: Any) -> str:
    """Return the JSON type name for a Python value."""
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


def format_expected_type(expected: str | list[str]) -> str:
    """Format JSON type declarations for observations."""
    if isinstance(expected, list):
        return " or ".join(expected)
    return expected
