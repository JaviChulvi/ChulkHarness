"""Public tool helpers for ergonomic agent construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
import json
from typing import Any, get_args, get_origin, get_type_hints

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
from chulk.tools.registry import Tool, ToolResult
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
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Convert a Python callable into a Chulk tool."""

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or _description_from_callable(func)
        args_schema = _schema_from_callable(func)

        def invoke(arguments: dict[str, Any]) -> ToolResult:
            result = func(**arguments)
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
    hints = get_type_hints(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, parameter in signature.parameters.items():
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            raise ValueError("@tool functions cannot use *args or **kwargs")
        annotation = hints.get(param_name, Any)
        schema = _json_schema_for_type(annotation)
        if parameter.default is not inspect.Parameter.empty:
            schema["default"] = parameter.default
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
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation in {dict, dict[str, Any]}:
        return {"type": "object"}
    if annotation in {list, list[str]}:
        return {"type": "array"}
    origin = get_origin(annotation)
    if origin in {list, tuple, set}:
        args = get_args(annotation)
        item_schema = _json_schema_for_type(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}
    return {"type": "string"}


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
