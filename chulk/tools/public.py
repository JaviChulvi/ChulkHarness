"""Public tool helpers for ergonomic agent construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
import inspect
import json
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.calculator import calculator_tool
from chulk.tools.files import apply_patch_tool, list_files_tool, read_file_tool, search_files_tool, write_file_tool
from chulk.tools.memory import (
    archive_memory_tool,
    compact_memories_tool,
    delete_memory_tool,
    export_memories_tool,
    import_memories_tool,
    list_memories_tool,
    restore_memory_tool,
    save_memory_tool,
    search_memory_tool,
    summarize_memories_tool,
    update_memory_tool,
)
from chulk.tools.registry import Tool, ToolExecutionContext, ToolFailureKind, ToolResult
from chulk.tools.shell import shell_tool


@dataclass(frozen=True)
class ToolRef:
    """Reference to a built-in tool that can be bound to runtime context."""

    name: str
    factory: Callable[[Any], Tool]

    def to_tool(self, context: Any) -> Tool:
        return self.factory(context)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    permission_level: ToolPermissionLevel | str = ToolPermissionLevel.READ,
    requires_confirmation: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Convert a Python callable into a Chulk tool."""

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or _description_from_callable(func)
        args_schema = _schema_from_callable(func)

        def invoke(arguments: dict[str, Any]) -> ToolResult:
            result = func(**arguments)
            if inspect.isawaitable(result):
                async def await_result():
                    resolved = await result
                    if isinstance(resolved, ToolResult):
                        return resolved
                    return ToolResult(
                        tool_name=tool_name,
                        success=True,
                        observation=_observation_from_result(resolved),
                        metadata={"result_type": type(resolved).__name__},
                    )

                return await_result()  # type: ignore[return-value]
            if isinstance(result, ToolResult):
                return result
            return ToolResult(
                tool_name=tool_name,
                success=True,
                observation=_observation_from_result(result),
                metadata={"result_type": type(result).__name__},
            )

        return Tool(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            callable=invoke,
            permission_level=permission_level,
            requires_confirmation=requires_confirmation,
        )

    if fn is None:
        return decorator
    return decorator(fn)


def default_software_engineer(*, include_memory: bool = True) -> list[ToolRef]:
    """Return the built-in tools used by the default coding-agent preset."""
    refs = [
        calculator,
        run_cmd,
        read_file,
        apply_patch,
        write_file,
        list_files,
        search_files,
    ]
    if include_memory:
        refs.extend(
            [
                save_memory,
                search_memory,
                list_memories,
                delete_memory,
                update_memory,
                summarize_memories,
                archive_memory,
                restore_memory,
                compact_memories,
                import_memories,
                export_memories,
            ]
        )
    return refs


def _memory_tool(factory: Callable[[Any], Tool]) -> Callable[[Any], Tool]:
    def bind(context: Any) -> Tool:
        if context.memory_store is None:
            raise ValueError("Memory tool requires a configured memory store")
        return factory(context)

    return bind


calculator = ToolRef("calculator", lambda _context: calculator_tool())
run_cmd = ToolRef("run_cmd", lambda context: shell_tool(context.project_root, timeout_seconds=context.shell_timeout_seconds))
read_file = ToolRef("read_file", lambda context: read_file_tool(context.project_root))
apply_patch = ToolRef("apply_patch", lambda context: apply_patch_tool(context.project_root))
write_file = ToolRef("write_file", lambda context: write_file_tool(context.project_root))
list_files = ToolRef("list_files", lambda context: list_files_tool(context.project_root))
search_files = ToolRef("search_files", lambda context: search_files_tool(context.project_root))
save_memory = ToolRef("save_memory", _memory_tool(lambda context: save_memory_tool(context.memory_store)))
search_memory = ToolRef("search_memory", _memory_tool(lambda context: search_memory_tool(context.memory_store)))
list_memories = ToolRef("list_memories", _memory_tool(lambda context: list_memories_tool(context.memory_store)))
delete_memory = ToolRef("delete_memory", _memory_tool(lambda context: delete_memory_tool(context.memory_store)))
update_memory = ToolRef("update_memory", _memory_tool(lambda context: update_memory_tool(context.memory_store)))
summarize_memories = ToolRef("summarize_memories", _memory_tool(lambda context: summarize_memories_tool(context.memory_store)))
archive_memory = ToolRef("archive_memory", _memory_tool(lambda context: archive_memory_tool(context.memory_store)))
restore_memory = ToolRef("restore_memory", _memory_tool(lambda context: restore_memory_tool(context.memory_store)))
compact_memories = ToolRef("compact_memories", _memory_tool(lambda context: compact_memories_tool(context.memory_store)))
import_memories = ToolRef(
    "import_memories",
    _memory_tool(lambda context: import_memories_tool(context.memory_store, context.project_root)),
)
export_memories = ToolRef(
    "export_memories",
    _memory_tool(lambda context: export_memories_tool(context.memory_store, context.project_root)),
)


def _description_from_callable(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func) or ""
    first_line = doc.splitlines()[0].strip() if doc else ""
    return first_line or f"Run {func.__name__}."


def _schema_from_callable(func: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, parameter in signature.parameters.items():
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            raise ValueError("@tool functions cannot use *args or **kwargs")
        annotation = hints.get(param_name, Any)
        schema = _json_schema_for_type(annotation)
        if parameter.default is not inspect.Parameter.empty:
            schema["default"] = _json_safe_value(parameter.default)
        else:
            required.append(param_name)
        properties[param_name] = schema
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _json_schema_for_type(annotation: Any) -> dict[str, Any]:
    annotation, description = _unwrap_annotated(annotation)
    schema = _json_schema_for_unwrapped_type(annotation)
    if description:
        schema["description"] = description
    return schema


def _json_schema_for_unwrapped_type(annotation: Any) -> dict[str, Any]:
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is type(None):
        return {"type": "null"}
    if annotation in {dict, dict[str, Any]}:
        return {"type": "object"}
    if annotation in {list, list[str]}:
        return {"type": "array"}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return _enum_schema(annotation)
    if is_dataclass(annotation):
        return _dataclass_schema(annotation)
    if hasattr(annotation, "model_json_schema"):
        schema = _pydantic_model_schema(annotation)
        return schema if schema is not None else {"type": "string"}

    origin = get_origin(annotation)
    if origin is Annotated:
        inner, description = _unwrap_annotated(annotation)
        schema = _json_schema_for_unwrapped_type(inner)
        if description:
            schema["description"] = description
        return schema
    if origin in {Union, UnionType}:
        return _union_schema(get_args(annotation))
    if origin is Literal:
        return _literal_schema(get_args(annotation))
    if origin in {list, tuple, set}:
        args = get_args(annotation)
        item_schema = _json_schema_for_type(args[0]) if args and args[0] is not Ellipsis else {}
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return _dict_schema(get_args(annotation))
    return {"type": "string"}


def _unwrap_annotated(annotation: Any) -> tuple[Any, str | None]:
    if get_origin(annotation) is not Annotated:
        return annotation, None
    args = get_args(annotation)
    description = next((item for item in args[1:] if isinstance(item, str) and item.strip()), None)
    return args[0], description.strip() if isinstance(description, str) else None


def _union_schema(args: tuple[Any, ...]) -> dict[str, Any]:
    schemas = [_json_schema_for_type(arg) for arg in args]
    null_schemas = [schema for schema in schemas if schema.get("type") == "null"]
    non_null_schemas = [schema for schema in schemas if schema.get("type") != "null"]
    if null_schemas and len(non_null_schemas) == 1:
        schema = dict(non_null_schemas[0])
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            schema["type"] = sorted({schema_type, "null"})
            return schema
        if isinstance(schema_type, list):
            schema["type"] = sorted({str(item) for item in schema_type} | {"null"})
            return schema

    simple_types: list[str] = []
    enum_values: list[Any] = []
    for schema in schemas:
        schema_type = schema.get("type")
        if isinstance(schema_type, str) and set(schema) <= {"type"}:
            simple_types.append(schema_type)
            continue
        if "enum" in schema and isinstance(schema.get("enum"), list):
            enum_values.extend(schema["enum"])
            type_value = schema.get("type")
            if isinstance(type_value, str):
                simple_types.append(type_value)
            elif isinstance(type_value, list):
                simple_types.extend(str(item) for item in type_value)
            continue
        return {"type": "string"}
    result: dict[str, Any] = {"type": sorted(set(simple_types))}
    if enum_values:
        result["enum"] = _dedupe_json_values(enum_values)
    return result


def _literal_schema(values: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "type": sorted({_json_type_for_value(value) for value in values}),
        "enum": list(values),
    }


def _enum_schema(enum_type: type[Enum]) -> dict[str, Any]:
    values = [item.value for item in enum_type]
    return {
        "type": sorted({_json_type_for_value(value) for value in values}),
        "enum": values,
    }


def _dataclass_schema(dataclass_type: type) -> dict[str, Any]:
    hints = get_type_hints(dataclass_type, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in fields(dataclass_type):
        properties[field.name] = _json_schema_for_type(hints.get(field.name, Any))
        if field.default is MISSING and field.default_factory is MISSING:
            required.append(field.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _pydantic_model_schema(model_type: type) -> dict[str, Any] | None:
    schema = _pydantic_fields_schema(model_type)
    if schema is not None:
        return schema
    try:
        raw_schema = model_type.model_json_schema()
    except Exception:
        return None
    return _enforced_schema_subset(raw_schema)


def _pydantic_fields_schema(model_type: type) -> dict[str, Any] | None:
    model_fields = getattr(model_type, "model_fields", None)
    if isinstance(model_fields, dict):
        return _pydantic_field_map_schema(model_fields)

    legacy_fields = getattr(model_type, "__fields__", None)
    if isinstance(legacy_fields, dict):
        return _pydantic_field_map_schema(legacy_fields)
    return None


def _pydantic_field_map_schema(field_map: dict[Any, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for raw_name, field_info in field_map.items():
        field_name = str(raw_name)
        properties[field_name] = _json_schema_for_type(_pydantic_field_annotation(field_info))
        if _pydantic_field_is_required(field_info):
            required.append(field_name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _pydantic_field_annotation(field_info: Any) -> Any:
    annotation = getattr(field_info, "annotation", None)
    if annotation is None:
        annotation = getattr(field_info, "outer_type_", None)
    if annotation is None:
        annotation = getattr(field_info, "type_", None)
    return annotation if annotation is not None else Any


def _pydantic_field_is_required(field_info: Any) -> bool:
    is_required = getattr(field_info, "is_required", None)
    if callable(is_required):
        return bool(is_required())
    return bool(getattr(field_info, "required", False))


def _enforced_schema_subset(schema: Any) -> dict[str, Any] | None:
    if not isinstance(schema, dict):
        return None
    unsupported = {"$defs", "$ref", "allOf", "anyOf", "definitions", "not", "oneOf", "patternProperties"}
    if any(key in schema for key in unsupported):
        return None

    result: dict[str, Any] = {}
    for key in (
        "type",
        "enum",
        "required",
        "description",
        "default",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    ):
        if key in schema:
            result[key] = schema[key]

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            return None
        normalized_properties: dict[str, Any] = {}
        for name, property_schema in properties.items():
            normalized = _enforced_schema_subset(property_schema)
            if normalized is None:
                return None
            normalized_properties[str(name)] = normalized
        result["properties"] = normalized_properties

    items = schema.get("items")
    if items is not None:
        normalized_items = _enforced_schema_subset(items)
        if normalized_items is None:
            return None
        result["items"] = normalized_items

    additional_properties = schema.get("additionalProperties")
    if isinstance(additional_properties, dict):
        normalized_additional = _enforced_schema_subset(additional_properties)
        if normalized_additional is None:
            return None
        result["additionalProperties"] = normalized_additional
    elif isinstance(additional_properties, bool):
        result["additionalProperties"] = additional_properties
    elif additional_properties is not None:
        return None

    return result or None


def _dict_schema(args: tuple[Any, ...]) -> dict[str, Any]:
    if len(args) < 2 or args[1] is Any:
        return {"type": "object"}
    return {"type": "object", "additionalProperties": _json_schema_for_type(args[1])}


def _json_type_for_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _dedupe_json_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _json_safe_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _observation_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, indent=2, sort_keys=True)
    except TypeError:
        return str(result)


__all__ = [
    "ToolRef",
    "apply_patch",
    "archive_memory",
    "calculator",
    "compact_memories",
    "default_software_engineer",
    "delete_memory",
    "export_memories",
    "import_memories",
    "list_files",
    "list_memories",
    "read_file",
    "restore_memory",
    "run_cmd",
    "save_memory",
    "search_files",
    "search_memory",
    "summarize_memories",
    "tool",
    "update_memory",
    "write_file",
]
