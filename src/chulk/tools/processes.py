"""Tool adapters for backend-owned managed processes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from chulk.execution.models import (
    ProcessHandle,
    ProcessLogsRequest,
    ProcessPollRequest,
    ProcessStartRequest,
    ProcessTerminateRequest,
    ProcessWriteRequest,
)
from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.registry import Tool, ToolFailureKind, ToolResult


def process_tools() -> tuple[Tool, ...]:
    """Return the complete managed-process tool family."""
    return (
        _process_start_tool(),
        _handle_tool(
            name="process_poll",
            description="Poll a managed process owned by this conversation or child task.",
            operation=lambda session, arguments, handle: session.poll_process(
                ProcessPollRequest(handle)
            ),
        ),
        _handle_tool(
            name="process_logs",
            description="Read bounded managed-process output from an absolute cursor.",
            operation=lambda session, arguments, handle: session.read_process_logs(
                ProcessLogsRequest(
                    handle,
                    cursor=arguments.get("cursor", 0),
                    max_bytes=arguments.get("max_bytes"),
                )
            ),
            extra_properties={
                "cursor": {"type": "integer", "minimum": 0},
                "max_bytes": {"type": "integer", "minimum": 1},
            },
        ),
        _handle_tool(
            name="process_write",
            description="Write bounded UTF-8 data to an interactive managed process.",
            operation=lambda session, arguments, handle: session.write_process(
                ProcessWriteRequest(
                    handle,
                    data=arguments.get("data", ""),
                    close_stdin=arguments.get("close_stdin", False),
                )
            ),
            extra_properties={
                "data": {"type": "string"},
                "close_stdin": {"type": "boolean"},
            },
            requires_confirmation=True,
            permission_level=ToolPermissionLevel.SHELL,
        ),
        _handle_tool(
            name="process_terminate",
            description="Terminate a managed process owned by this conversation or child task.",
            operation=lambda session, arguments, handle: session.terminate_process(
                ProcessTerminateRequest(
                    handle,
                    grace_seconds=arguments.get("grace_seconds"),
                )
            ),
            extra_properties={
                "grace_seconds": {"type": "integer", "minimum": 1},
            },
            requires_confirmation=True,
            permission_level=ToolPermissionLevel.SHELL,
        ),
    )


def _process_start_tool() -> Tool:
    def invoke(arguments: dict[str, Any], context=None) -> ToolResult:
        session = getattr(context, "execution_session", None)
        if session is None:
            return _missing_session("process_start")
        result = session.start_process(
            ProcessStartRequest(
                command=arguments["command"],
                timeout_seconds=arguments.get("timeout_seconds"),
                interactive=arguments.get("interactive", False),
            )
        )
        return result.to_tool_result("process_start")

    return Tool(
        name="process_start",
        description=(
            "Start a bounded managed process owned by this conversation or child task."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1, "maxLength": 4000},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "interactive": {"type": "boolean"},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        callable=invoke,
        accepts_context=True,
        run_in_executor=True,
        requires_confirmation=True,
        permission_level=ToolPermissionLevel.SHELL,
    )


def _handle_tool(
    *,
    name: str,
    description: str,
    operation: Callable[[Any, dict[str, Any], ProcessHandle], Any],
    extra_properties: dict[str, Any] | None = None,
    requires_confirmation: bool = False,
    permission_level: ToolPermissionLevel = ToolPermissionLevel.READ,
) -> Tool:
    properties: dict[str, Any] = {
        "process_id": {"type": "string", "minLength": 1},
        "backend_name": {"type": "string", "minLength": 1},
        "workspace_id": {"type": "string", "minLength": 1},
    }
    properties.update(extra_properties or {})

    def invoke(arguments: dict[str, Any], context=None) -> ToolResult:
        session = getattr(context, "execution_session", None)
        if session is None:
            return _missing_session(name)
        handle = ProcessHandle(
            process_id=arguments["process_id"],
            backend_name=arguments["backend_name"],
            workspace_id=arguments["workspace_id"],
        )
        return operation(session, arguments, handle).to_tool_result(name)

    return Tool(
        name=name,
        description=description,
        args_schema={
            "type": "object",
            "properties": properties,
            "required": ["process_id", "backend_name", "workspace_id"],
            "additionalProperties": False,
        },
        callable=invoke,
        accepts_context=True,
        run_in_executor=True,
        requires_confirmation=requires_confirmation,
        permission_level=permission_level,
    )


def _missing_session(tool_name: str) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        success=False,
        observation="Managed process tools require an active execution session.",
        error="execution_session_required",
        failure_kind=ToolFailureKind.ENVIRONMENT,
    )
