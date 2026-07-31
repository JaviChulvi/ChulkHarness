"""Public tool helpers for ergonomic agent construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields, is_dataclass
from enum import Enum
import inspect
import json
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints, overload

from chulk.capabilities import ToolOutputPolicy, ToolRetryPolicy
from chulk.tools.artifacts import (
    async_read_trace_artifact_tool,
    read_trace_artifact_tool,
)
from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.policy import ToolIdentity, ToolPolicy
from chulk.tools.calculator import calculator_tool
from chulk.tools.files import apply_patch_tool, list_files_tool, read_file_tool, search_files_tool, write_file_tool
from chulk.tools.processes import process_tools
from chulk.tools.memory import (
    async_archive_memory_tool,
    async_compact_memories_tool,
    async_delete_memory_tool,
    async_export_memories_tool,
    async_import_memories_tool,
    async_list_memories_tool,
    async_restore_memory_tool,
    async_save_memory_tool,
    async_search_memory_tool,
    async_summarize_memories_tool,
    async_update_memory_tool,
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
from chulk.tools.registry import Tool, ToolExecutionContext, ToolResult
from chulk.tools.sessions import (
    async_session_read_tool,
    async_session_search_tool,
    session_read_tool,
    session_search_tool,
)
from chulk.tools.shell import shell_tool


ToolContext = ToolExecutionContext


@dataclass(frozen=True)
class ToolRef:
    """Reference to a built-in tool that can be bound to runtime context."""

    name: str
    factory: Callable[[Any], Tool]
    async_factory: Callable[[Any], Tool] | None = None

    def to_tool(self, context: Any) -> Tool:
        return self.factory(context)

    def to_async_tool(self, context: Any) -> Tool:
        factory = self.async_factory or self.factory
        return factory(context)


@overload
def tool(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    permission_level: ToolPermissionLevel | str = ToolPermissionLevel.READ,
    requires_confirmation: bool = False,
    output_schema: dict[str, Any] | None = None,
    output_policy: ToolOutputPolicy | None = None,
    timeout_seconds: float | None = None,
    retry_policy: ToolRetryPolicy | None = None,
    idempotent: bool = False,
    identity: ToolIdentity | None = None,
    policy: ToolPolicy | None = None,
) -> Tool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    permission_level: ToolPermissionLevel | str = ToolPermissionLevel.READ,
    requires_confirmation: bool = False,
    output_schema: dict[str, Any] | None = None,
    output_policy: ToolOutputPolicy | None = None,
    timeout_seconds: float | None = None,
    retry_policy: ToolRetryPolicy | None = None,
    idempotent: bool = False,
    identity: ToolIdentity | None = None,
    policy: ToolPolicy | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    permission_level: ToolPermissionLevel | str = ToolPermissionLevel.READ,
    requires_confirmation: bool = False,
    output_schema: dict[str, Any] | None = None,
    output_policy: ToolOutputPolicy | None = None,
    timeout_seconds: float | None = None,
    retry_policy: ToolRetryPolicy | None = None,
    idempotent: bool = False,
    identity: ToolIdentity | None = None,
    policy: ToolPolicy | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Convert a Python callable into a Chulk tool."""

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or _description_from_callable(func)
        injected_parameter = _injected_parameter(func)
        args_schema = _schema_from_callable(func, injected_parameter=injected_parameter)

        async def invoke_async(
            arguments: dict[str, Any],
            context: ToolExecutionContext | None = None,
        ) -> ToolResult:
            result = await func(**_call_arguments(arguments, injected_parameter, context))
            return _coerce_tool_result(tool_name, result)

        def invoke(arguments: dict[str, Any], context: ToolExecutionContext | None = None) -> ToolResult:
            result = func(**_call_arguments(arguments, injected_parameter, context))
            if inspect.isawaitable(result):
                return _AwaitableToolResult(tool_name, result)  # type: ignore[return-value]
            return _coerce_tool_result(tool_name, result)

        return Tool(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            callable=invoke_async if inspect.iscoroutinefunction(func) else invoke,
            permission_level=permission_level,
            requires_confirmation=requires_confirmation,
            accepts_context=injected_parameter is not None,
            output_schema=output_policy.to_dict() if output_policy is not None else output_schema,
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy,
            idempotent=idempotent,
            identity=identity,
            policy=policy,
        )

    if fn is None:
        return decorator
    return decorator(fn)


def _coerce_tool_result(tool_name: str, result: Any) -> ToolResult:
    if isinstance(result, ToolResult):
        return result
    return ToolResult(
        tool_name=tool_name,
        success=True,
        observation=_observation_from_result(result),
        metadata={"result_type": type(result).__name__},
        value=_json_safe_value(result),
    )


class _AwaitableToolResult:
    def __init__(self, tool_name: str, awaitable: Any) -> None:
        self._tool_name = tool_name
        self._awaitable = awaitable
        self._runner: Any = None

    def __await__(self):
        if self._runner is None:
            self._runner = self._run()
        return self._runner.__await__()

    async def _run(self) -> ToolResult:
        try:
            resolved = await self._awaitable
            return _coerce_tool_result(self._tool_name, resolved)
        finally:
            self._close_inner()

    def close(self) -> None:
        close_runner = getattr(self._runner, "close", None)
        if callable(close_runner):
            close_runner()
        self._close_inner()

    def _close_inner(self) -> None:
        close_inner = getattr(self._awaitable, "close", None)
        if callable(close_inner):
            close_inner()


def default_software_engineer(*, include_memory: bool = True) -> list[ToolRef]:
    """Return the built-in tools used by the default coding-agent preset."""
    refs = [
        calculator,
        run_cmd,
        process_start,
        process_poll,
        process_logs,
        process_write,
        process_terminate,
        read_file,
        apply_patch,
        write_file,
        list_files,
        search_files,
        session_search,
        session_read,
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


def _artifact_tool(factory: Callable[[Any], Tool]) -> Callable[[Any], Tool]:
    def bind(context: Any) -> Tool:
        if context.artifact_store is None:
            raise ValueError("Trace artifact tool requires a bound artifact store")
        return factory(context)

    return bind


def _session_tool(factory: Callable[[Any], Tool]) -> Callable[[Any], Tool]:
    def bind(context: Any) -> Tool:
        if context.session_search_service is None:
            raise ValueError("Session tool requires a configured session search service")
        return factory(context)

    return bind


calculator = ToolRef("calculator", lambda _context: calculator_tool())
run_cmd = ToolRef(
    "run_cmd",
    lambda context: shell_tool(
        context.project_root,
        timeout_seconds=context.shell_timeout_seconds,
        stdout_limit_bytes=context.max_tool_stdout_bytes,
        stderr_limit_bytes=context.max_tool_stderr_bytes,
        execution_policy=context.shell_execution_policy,
        require_containment=context.require_shell_containment,
    ),
)


def _managed_process_tool(name: str) -> Tool:
    return next(tool for tool in process_tools() if tool.name == name)


process_start = ToolRef(
    "process_start",
    lambda _context: _managed_process_tool("process_start"),
)
process_poll = ToolRef(
    "process_poll",
    lambda _context: _managed_process_tool("process_poll"),
)
process_logs = ToolRef(
    "process_logs",
    lambda _context: _managed_process_tool("process_logs"),
)
process_write = ToolRef(
    "process_write",
    lambda _context: _managed_process_tool("process_write"),
)
process_terminate = ToolRef(
    "process_terminate",
    lambda _context: _managed_process_tool("process_terminate"),
)
read_file = ToolRef("read_file", lambda context: read_file_tool(context.project_root))
apply_patch = ToolRef("apply_patch", lambda context: apply_patch_tool(context.project_root))
write_file = ToolRef("write_file", lambda context: write_file_tool(context.project_root))
list_files = ToolRef("list_files", lambda context: list_files_tool(context.project_root))
search_files = ToolRef("search_files", lambda context: search_files_tool(context.project_root))
read_trace_artifact = ToolRef(
    "read_trace_artifact",
    _artifact_tool(
        lambda context: read_trace_artifact_tool(context.artifact_store)
    ),
    _artifact_tool(
        lambda context: async_read_trace_artifact_tool(
            context.artifact_store
        )
    ),
)
save_memory = ToolRef(
    "save_memory",
    _memory_tool(lambda context: save_memory_tool(context.memory_store)),
    _memory_tool(
        lambda context: async_save_memory_tool(context.memory_store)
    ),
)
search_memory = ToolRef(
    "search_memory",
    _memory_tool(lambda context: search_memory_tool(context.memory_store)),
    _memory_tool(
        lambda context: async_search_memory_tool(context.memory_store)
    ),
)
list_memories = ToolRef(
    "list_memories",
    _memory_tool(lambda context: list_memories_tool(context.memory_store)),
    _memory_tool(
        lambda context: async_list_memories_tool(context.memory_store)
    ),
)
delete_memory = ToolRef(
    "delete_memory",
    _memory_tool(lambda context: delete_memory_tool(context.memory_store)),
    _memory_tool(
        lambda context: async_delete_memory_tool(context.memory_store)
    ),
)
update_memory = ToolRef(
    "update_memory",
    _memory_tool(lambda context: update_memory_tool(context.memory_store)),
    _memory_tool(
        lambda context: async_update_memory_tool(context.memory_store)
    ),
)
summarize_memories = ToolRef(
    "summarize_memories",
    _memory_tool(
        lambda context: summarize_memories_tool(context.memory_store)
    ),
    _memory_tool(
        lambda context: async_summarize_memories_tool(
            context.memory_store
        )
    ),
)
archive_memory = ToolRef(
    "archive_memory",
    _memory_tool(lambda context: archive_memory_tool(context.memory_store)),
    _memory_tool(
        lambda context: async_archive_memory_tool(context.memory_store)
    ),
)
restore_memory = ToolRef(
    "restore_memory",
    _memory_tool(lambda context: restore_memory_tool(context.memory_store)),
    _memory_tool(
        lambda context: async_restore_memory_tool(context.memory_store)
    ),
)
compact_memories = ToolRef(
    "compact_memories",
    _memory_tool(lambda context: compact_memories_tool(context.memory_store)),
    _memory_tool(
        lambda context: async_compact_memories_tool(
            context.memory_store
        )
    ),
)
import_memories = ToolRef(
    "import_memories",
    _memory_tool(lambda context: import_memories_tool(context.memory_store, context.project_root)),
    _memory_tool(
        lambda context: async_import_memories_tool(
            context.memory_store,
            context.project_root,
        )
    ),
)
export_memories = ToolRef(
    "export_memories",
    _memory_tool(lambda context: export_memories_tool(context.memory_store, context.project_root)),
    _memory_tool(
        lambda context: async_export_memories_tool(
            context.memory_store,
            context.project_root,
        )
    ),
)
session_search = ToolRef(
    "session_search",
    _session_tool(
        lambda context: session_search_tool(context.session_search_service)
    ),
    _session_tool(
        lambda context: async_session_search_tool(
            context.session_search_service
        )
    ),
)
session_read = ToolRef(
    "session_read",
    _session_tool(
        lambda context: session_read_tool(context.session_search_service)
    ),
    _session_tool(
        lambda context: async_session_read_tool(
            context.session_search_service
        )
    ),
)


def _description_from_callable(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func) or ""
    first_line = doc.splitlines()[0].strip() if doc else ""
    return first_line or f"Run {func.__name__}."


def _schema_from_callable(
    func: Callable[..., Any],
    *,
    injected_parameter: str | None = None,
) -> dict[str, Any]:
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, parameter in signature.parameters.items():
        if param_name == injected_parameter:
            continue
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


def _injected_parameter(func: Callable[..., Any]) -> str | None:
    hints = get_type_hints(func, include_extras=True)
    injected = [name for name, annotation in hints.items() if name != "return" and _is_tool_context(annotation)]
    if len(injected) > 1:
        raise ValueError("@tool functions can declare at most one ToolContext parameter")
    return injected[0] if injected else None


def _is_tool_context(annotation: Any) -> bool:
    return annotation is ToolExecutionContext or get_origin(annotation) is ToolExecutionContext


def _call_arguments(
    arguments: dict[str, Any],
    injected_parameter: str | None,
    context: ToolExecutionContext | None,
) -> dict[str, Any]:
    if injected_parameter is None:
        return arguments
    if injected_parameter in arguments:
        raise ValueError(f"Injected parameter {injected_parameter} cannot be supplied by the model")
    if context is None:
        raise ValueError("Required tool dependencies and context were not provided by the host")
    context.require_deps()
    return {**arguments, injected_parameter: context}


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
    if annotation is dict or annotation == dict[str, Any]:
        return {"type": "object"}
    if annotation is list or annotation == list[str]:
        return {"type": "array"}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return _enum_schema(annotation)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _dataclass_schema(annotation)
    if isinstance(annotation, type) and callable(getattr(annotation, "model_json_schema", None)):
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
    model_json_schema = getattr(model_type, "model_json_schema", None)
    if not callable(model_json_schema):
        return None
    try:
        raw_schema = model_json_schema()
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
    "ToolContext",
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
    "read_trace_artifact",
    "restore_memory",
    "run_cmd",
    "save_memory",
    "search_files",
    "search_memory",
    "session_read",
    "session_search",
    "summarize_memories",
    "tool",
    "update_memory",
    "write_file",
]
