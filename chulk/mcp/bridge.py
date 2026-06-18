"""MCP-to-Chulk tool bridge."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import asyncio
import re
from typing import Any, Protocol

from chulk.mcp.config import MCPServerConfig
from chulk.tools import Tool, ToolPermissionLevel, ToolResult


class MCPDependencyError(RuntimeError):
    """Raised when bridge MCP support needs the optional MCP SDK."""


class MCPClient(Protocol):
    """Small synchronous MCP client interface used by bridge tools."""

    def list_tools(self) -> list[object]:
        """Return remote MCP tool definitions."""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        """Call one remote MCP tool."""


@dataclass(frozen=True)
class MCPToolDefinition:
    """Normalized remote MCP tool definition."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


MCPClientFactory = Callable[[MCPServerConfig], MCPClient]


class StreamableHttpMCPClient:
    """Synchronous wrapper around the official MCP Python SDK streamable HTTP client."""

    def __init__(self, server: MCPServerConfig) -> None:
        self.server = server

    def list_tools(self) -> list[object]:
        return _run_async(self._list_tools())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        return _run_async(self._call_tool(name, arguments))

    async def _list_tools(self) -> list[object]:
        async with self._connect() as session:
            await session.initialize()
            result = await session.list_tools()
            return list(_value(result, "tools") or [])

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> object:
        async with self._connect() as session:
            await session.initialize()
            return await session.call_tool(name, arguments)

    def _connect(self):
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise MCPDependencyError("MCP bridge support requires: pip install -e '.[mcp]'") from exc

        headers = {}
        if self.server.authorization:
            headers["Authorization"] = f"Bearer {self.server.authorization}"
        return _StreamableHttpSessionContext(
            streamablehttp_client(self.server.server_url, headers=headers or None),
            ClientSession,
        )


class _StreamableHttpSessionContext:
    def __init__(self, client_context, session_type) -> None:
        self.client_context = client_context
        self.session_type = session_type
        self.session = None

    async def __aenter__(self):
        read_stream, write_stream, _ = await self.client_context.__aenter__()
        self.session = self.session_type(read_stream, write_stream)
        return await self.session.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self.session.__aexit__(exc_type, exc, tb)
        finally:
            await self.client_context.__aexit__(exc_type, exc, tb)


def create_mcp_bridge_tools(
    servers: Iterable[MCPServerConfig],
    *,
    client_factory: MCPClientFactory | None = None,
) -> list[Tool]:
    """Discover MCP tools and expose them as normal Chulk tools."""
    factory = client_factory or StreamableHttpMCPClient
    bridge_tools: list[Tool] = []
    used_names: set[str] = set()
    for server in servers:
        client = factory(server)
        allowed = set(server.allowed_tools)
        for raw_tool in client.list_tools():
            definition = normalize_mcp_tool_definition(raw_tool)
            if allowed and definition.name not in allowed:
                continue
            tool_name = mcp_bridge_tool_name(server.label, definition.name)
            if tool_name in used_names:
                raise ValueError(f"Duplicate MCP bridge tool name after normalization: {tool_name}")
            used_names.add(tool_name)
            bridge_tools.append(_bridge_tool(server, client, definition, tool_name))
    return bridge_tools


def normalize_mcp_tool_definition(raw_tool: object) -> MCPToolDefinition:
    """Return a normalized MCP tool definition from SDK objects or dicts."""
    name = _value(raw_tool, "name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("MCP tool definition did not include a name")
    description = _value(raw_tool, "description")
    input_schema = _value(raw_tool, "inputSchema") or _value(raw_tool, "input_schema") or {}
    if not isinstance(input_schema, dict) or not input_schema:
        input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    return MCPToolDefinition(
        name=name.strip(),
        description=description.strip() if isinstance(description, str) else "",
        input_schema=input_schema,
    )


def mcp_bridge_tool_name(server_label: str, tool_name: str) -> str:
    """Return a Chulk-safe bridge tool name."""
    normalized_tool = _safe_identifier(tool_name)
    return f"mcp_{_safe_identifier(server_label)}_{normalized_tool}"


def _bridge_tool(server: MCPServerConfig, client: MCPClient, definition: MCPToolDefinition, tool_name: str) -> Tool:
    def run(arguments: dict[str, Any]) -> ToolResult:
        try:
            raw_result = client.call_tool(definition.name, arguments)
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                observation=f"MCP tool {server.label}.{definition.name} failed: {exc}",
                error="mcp_call_failed",
                metadata={
                    "mcp_bridge": True,
                    "server_label": server.label,
                    "mcp_tool_name": definition.name,
                    "exception_type": type(exc).__name__,
                },
            )

        is_error = bool(_value(raw_result, "isError") or _value(raw_result, "is_error"))
        content = _format_mcp_content(_value(raw_result, "content"))
        observation = content or f"MCP tool {server.label}.{definition.name} returned no content."
        return ToolResult(
            tool_name=tool_name,
            success=not is_error,
            observation=observation,
            error="mcp_tool_error" if is_error else None,
            metadata={
                "mcp_bridge": True,
                "server_label": server.label,
                "mcp_tool_name": definition.name,
                "is_error": is_error,
            },
        )

    return Tool(
        name=tool_name,
        description=f"MCP {server.label}.{definition.name}: {definition.description or 'Remote MCP tool.'}",
        args_schema=definition.input_schema,
        callable=run,
        requires_confirmation=True,
        permission_level=ToolPermissionLevel.EXTERNAL_SERVICE,
        metadata={
            "mcp_bridge": True,
            "server_label": server.label,
            "mcp_tool_name": definition.name,
            "server_url": server.server_url,
        },
    )


def _format_mcp_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[str] = []
    for item in content:
        item_type = _value(item, "type")
        if item_type == "text":
            text = _value(item, "text")
            if isinstance(text, str):
                parts.append(text)
            continue
        if item_type == "resource_link":
            uri = _value(item, "uri")
            name = _value(item, "name")
            parts.append(f"resource_link: {name or uri}")
            continue
        parts.append(str(item))
    return "\n".join(part for part in parts if part)


def _safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "tool"
    if not cleaned[0].isalpha():
        cleaned = f"mcp_{cleaned}"
    return cleaned


def _value(source: object, key: str) -> object:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise MCPDependencyError("MCP bridge calls cannot run inside an existing event loop in the sync runtime")
