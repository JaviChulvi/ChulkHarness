"""Tool registration and execution primitives."""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
import json
from typing import Any, Generic, TypeVar, cast

from chulk.capabilities import ToolRetryPolicy
from chulk.redaction import redact_data, redact_text
from chulk.tools.permissions import ToolPermissionLevel, normalize_permission_level
from chulk.tools.schema import (
    ToolValidationError,
    ToolValidationIssue,
    validate_tool_arguments,
    validate_tool_output,
    validate_tool_output_schema,
    validate_tool_schema,
)


class ToolFailureKind:
    """Stable tool failure categories for adapters and traces."""

    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    ASYNC_REQUIRED = "async_required"
    CANCELLED = "cancelled"
    ENVIRONMENT = "environment_failure"
    USER_BLOCKED = "user_blocked"
    INVALID_OUTPUT = "invalid_output"
    TIMEOUT = "timeout"


DepsT = TypeVar("DepsT")


@dataclass(frozen=True)
class ToolExecutionContext(Generic[DepsT]):
    """Host-owned request context passed through to tools without interpretation."""

    metadata: dict[str, Any] = field(default_factory=dict)
    deps: DepsT = field(default=None)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "has_dependencies": self.deps is not None}

    def require_deps(self) -> DepsT:
        """Return injected dependencies or fail before tool side effects begin."""
        if self.deps is None:
            raise ValueError("Required tool dependencies were not provided by the host")
        return cast(DepsT, self.deps)


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
    failure_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    value: Any = None

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


ToolReturn = ToolResult | Awaitable[ToolResult | Any] | Any
ToolCallable = Callable[[dict[str, Any]], ToolReturn] | Callable[[dict[str, Any], ToolExecutionContext | None], ToolReturn]


@dataclass(frozen=True)
class Tool:
    """A callable action the agent may request."""

    name: str
    description: str
    args_schema: dict[str, Any]
    callable: ToolCallable
    requires_confirmation: bool = False
    permission_level: ToolPermissionLevel | str = ToolPermissionLevel.READ
    timeout_seconds: float | None = None
    accepts_context: bool = False
    run_in_executor: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    retry_policy: ToolRetryPolicy | None = None
    idempotent: bool = False

    def normalized_permission_level(self) -> ToolPermissionLevel:
        return normalize_permission_level(self.permission_level)


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
        tool.normalized_permission_level()
        validate_tool_schema(tool.name, tool.args_schema)
        if tool.output_schema is not None:
            validate_tool_output_schema(tool.name, tool.output_schema)
        if tool.timeout_seconds is not None and tool.timeout_seconds <= 0:
            raise ValueError(f"Tool {tool.name} timeout_seconds must be greater than zero")
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
                "permission_level": tool.normalized_permission_level().value,
            }
            for tool in self.list_tools()
        ]
        return json.dumps(tools, indent=2, sort_keys=True)

    def run(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        try:
            tool = self.get(name)
        except KeyError as exc:
            available_tools = sorted(self._tools)
            result = ToolResult(
                tool_name=name,
                success=False,
                observation=_format_unknown_tool_observation(name, available_tools),
                error=ToolFailureKind.UNKNOWN_TOOL,
                failure_kind=ToolFailureKind.UNKNOWN_TOOL,
                metadata={
                    "requested_tool_name": name,
                    "available_tools": available_tools,
                    "exception": redact_text(str(exc)),
                },
            )
            self._log_call(name, arguments, result)
            return result

        try:
            self._validate_arguments(tool, arguments)
            result = self._call_tool_with_timeout(tool, arguments, context)
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                result = ToolResult(
                    tool_name=tool.name,
                    success=False,
                    observation=(
                        f"Tool {tool.name} returned an awaitable but was executed through the sync runner. "
                        "Use run_async or Agent.run_turn_async for async tools."
                    ),
                    error=ToolFailureKind.ASYNC_REQUIRED,
                    failure_kind=ToolFailureKind.ASYNC_REQUIRED,
                )
            else:
                result = self._validate_output(tool, self._coerce_result(tool, result))
        except ToolValidationError as exc:
            result = ToolResult(
                tool_name=tool.name,
                success=False,
                observation=_format_invalid_arguments_observation(tool, exc.issues),
                error=ToolFailureKind.INVALID_ARGUMENTS,
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
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
                    f"Tool execution failed for {tool.name}: {redact_text(str(exc))}. "
                    "Retry only if corrected arguments or a safer alternative would change the outcome."
                ),
                error=redact_text(str(exc)),
                failure_kind=ToolFailureKind.ENVIRONMENT,
                metadata=redact_data({"exception_type": type(exc).__name__}),
            )

        self._log_call(name, arguments, result)
        return result

    async def run_async(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        try:
            tool = self.get(name)
        except KeyError as exc:
            available_tools = sorted(self._tools)
            result = ToolResult(
                tool_name=name,
                success=False,
                observation=_format_unknown_tool_observation(name, available_tools),
                error=ToolFailureKind.UNKNOWN_TOOL,
                failure_kind=ToolFailureKind.UNKNOWN_TOOL,
                metadata={
                    "requested_tool_name": name,
                    "available_tools": available_tools,
                    "exception": redact_text(str(exc)),
                },
            )
            self._log_call(name, arguments, result)
            return result

        try:
            self._validate_arguments(tool, arguments)
            result = await self._call_tool_async_with_timeout(tool, arguments, context)
            result = self._validate_output(tool, self._coerce_result(tool, result))
        except ToolValidationError as exc:
            result = ToolResult(
                tool_name=tool.name,
                success=False,
                observation=_format_invalid_arguments_observation(tool, exc.issues),
                error=ToolFailureKind.INVALID_ARGUMENTS,
                failure_kind=ToolFailureKind.INVALID_ARGUMENTS,
                metadata={
                    "validation_errors": [issue.to_dict() for issue in exc.issues],
                    "args_schema": tool.args_schema,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure_kind = ToolFailureKind.CANCELLED if type(exc).__name__ == "CancelledError" else ToolFailureKind.ENVIRONMENT
            result = ToolResult(
                tool_name=tool.name,
                success=False,
                observation=(
                    f"Tool execution failed for {tool.name}: {redact_text(str(exc))}. "
                    "Retry only if corrected arguments or a safer alternative would change the outcome."
                ),
                error=failure_kind if failure_kind == ToolFailureKind.CANCELLED else redact_text(str(exc)),
                failure_kind=failure_kind,
                metadata=redact_data({"exception_type": type(exc).__name__}),
            )

        self._log_call(name, arguments, result)
        return result

    def _validate_arguments(self, tool: Tool, arguments: dict[str, Any]) -> None:
        validate_tool_arguments(tool.name, arguments, tool.args_schema or {})

    def _call_tool(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None,
    ) -> ToolReturn:
        if tool.accepts_context:
            return tool.callable(arguments, context)  # type: ignore[misc]
        return tool.callable(arguments)  # type: ignore[misc]

    def _coerce_result(self, tool: Tool, result: Any) -> ToolResult:
        if isinstance(result, ToolResult):
            return result
        return ToolResult(tool_name=tool.name, success=True, observation=str(result))

    def _call_tool_with_timeout(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None,
    ) -> ToolReturn:
        if tool.timeout_seconds is None:
            return self._call_tool(tool, arguments, context)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"chulk-{tool.name}")
        future = executor.submit(self._call_tool, tool, arguments, context)
        try:
            result = future.result(timeout=tool.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            return ToolResult(
                tool_name=tool.name,
                success=False,
                observation=f"Tool {tool.name} timed out after {tool.timeout_seconds:g} seconds.",
                error=ToolFailureKind.TIMEOUT,
                failure_kind=ToolFailureKind.TIMEOUT,
                metadata={"timeout_seconds": tool.timeout_seconds},
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return result

    async def _call_tool_async_with_timeout(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None,
    ) -> ToolReturn:
        async def execute() -> ToolReturn:
            if tool.run_in_executor:
                result = await asyncio.to_thread(self._call_tool, tool, arguments, context)
            else:
                result = self._call_tool(tool, arguments, context)
            if inspect.isawaitable(result):
                return await result
            return result

        try:
            if tool.timeout_seconds is None:
                return await execute()
            return await asyncio.wait_for(execute(), timeout=tool.timeout_seconds)
        except TimeoutError:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                observation=f"Tool {tool.name} timed out after {tool.timeout_seconds:g} seconds.",
                error=ToolFailureKind.TIMEOUT,
                failure_kind=ToolFailureKind.TIMEOUT,
                metadata={"timeout_seconds": tool.timeout_seconds},
            )

    def _validate_output(self, tool: Tool, result: ToolResult) -> ToolResult:
        if not result.success or tool.output_schema is None:
            return result
        try:
            validate_tool_output(tool.name, result.value, tool.output_schema)
        except ToolValidationError as exc:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                observation=f"Tool {tool.name} returned output that did not match its contract.",
                error=ToolFailureKind.INVALID_OUTPUT,
                failure_kind=ToolFailureKind.INVALID_OUTPUT,
                metadata={
                    "validation_errors": [issue.to_dict() for issue in exc.issues],
                    "output_schema": tool.output_schema,
                },
            )
        return replace(
            result,
            metadata={
                **result.metadata,
                "output_validated": True,
                "structured_output": result.value,
            },
        )

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
