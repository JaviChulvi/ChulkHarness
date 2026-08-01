"""Read-only tools for exact prior-session evidence."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

from chulk.hosting.async_utils import call_async_service
from chulk.sessions import (
    MAX_SESSION_QUERY_CHARS,
    MAX_SESSION_SEARCH_LIMIT,
    MAX_SESSION_WINDOW_LIMIT,
    MAX_SESSION_WINDOW_RADIUS,
    SessionSearchService,
)
from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.registry import Tool, ToolResult


def session_search_tool(service: SessionSearchService) -> Tool:
    """Return a profile-bound exact session search tool."""
    return Tool(
        name="session_search",
        description=(
            "Search eligible user and assistant messages from profile-owned prior "
            "sessions. Results are bounded, redacted, and never saved as memory."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Plain search text, not FTS syntax.",
                    "minLength": 1,
                    "maxLength": MAX_SESSION_QUERY_CHARS,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum hits to return.",
                    "minimum": 1,
                    "maximum": MAX_SESSION_SEARCH_LIMIT,
                },
                "cursor": {
                    "type": "string",
                    "description": "Opaque cursor from a matching prior search.",
                    "minLength": 1,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        callable=lambda arguments: _search(arguments, service),
        run_in_executor=True,
        permission_level=ToolPermissionLevel.READ,
        idempotent=True,
    )


def session_read_tool(service: SessionSearchService) -> Tool:
    """Return a profile-bound bounded session-window tool."""
    return Tool(
        name="session_read",
        description=(
            "Read a bounded redacted user/assistant message window around an exact "
            "prior-session ordinal. Sensitive and external content is excluded."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "Conversation id returned by session_search.",
                    "minLength": 1,
                },
                "ordinal": {
                    "type": "integer",
                    "description": "Stable message ordinal to read around.",
                    "minimum": 1,
                },
                "before": {
                    "type": "integer",
                    "description": "Maximum ordinal radius before the anchor.",
                    "minimum": 0,
                    "maximum": MAX_SESSION_WINDOW_RADIUS,
                },
                "after": {
                    "type": "integer",
                    "description": "Maximum ordinal radius after the anchor.",
                    "minimum": 0,
                    "maximum": MAX_SESSION_WINDOW_RADIUS,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum visible messages to return.",
                    "minimum": 1,
                    "maximum": MAX_SESSION_WINDOW_LIMIT,
                },
                "cursor": {
                    "type": "string",
                    "description": "Opaque cursor from a matching prior window.",
                    "minLength": 1,
                },
            },
            "required": ["conversation_id", "ordinal"],
            "additionalProperties": False,
        },
        callable=lambda arguments: _read(arguments, service),
        run_in_executor=True,
        permission_level=ToolPermissionLevel.READ,
        idempotent=True,
    )


def async_session_search_tool(service: object) -> Tool:
    """Return the exact-search tool bound to a native async service."""

    async def invoke(arguments: dict[str, Any]) -> ToolResult:
        page = await call_async_service(
            service,
            "search",
            str(arguments["query"]),
            limit=arguments.get("limit", 10),
            cursor=arguments.get("cursor"),
        )
        return _search_result(page)

    return replace(
        session_search_tool(service),  # type: ignore[arg-type]
        callable=invoke,
        run_in_executor=False,
    )


def async_session_read_tool(service: object) -> Tool:
    """Return the bounded-window tool bound to a native async service."""

    async def invoke(arguments: dict[str, Any]) -> ToolResult:
        window = await call_async_service(
            service,
            "read_window",
            str(arguments["conversation_id"]),
            ordinal=arguments["ordinal"],
            before=arguments.get("before", 3),
            after=arguments.get("after", 3),
            limit=arguments.get("limit", 20),
            cursor=arguments.get("cursor"),
            include_sensitive=False,
        )
        return _read_result(window)

    return replace(
        session_read_tool(service),  # type: ignore[arg-type]
        callable=invoke,
        run_in_executor=False,
    )


def _search(
    arguments: dict[str, Any],
    service: SessionSearchService,
) -> ToolResult:
    page = service.search(
        str(arguments["query"]),
        limit=arguments.get("limit", 10),
        cursor=arguments.get("cursor"),
    )
    return _search_result(page)


def _search_result(page: Any) -> ToolResult:
    payload = page.to_dict()
    return ToolResult(
        tool_name="session_search",
        success=True,
        observation=json.dumps(payload, sort_keys=True),
        metadata={
            "message_ids": [hit.message_id for hit in page.hits],
            "conversation_ids": sorted(
                {hit.conversation_id for hit in page.hits}
            ),
            "next_cursor": page.next_cursor,
        },
        value=payload,
    )


def _read(
    arguments: dict[str, Any],
    service: SessionSearchService,
) -> ToolResult:
    window = service.read_window(
        str(arguments["conversation_id"]),
        ordinal=arguments["ordinal"],
        before=arguments.get("before", 3),
        after=arguments.get("after", 3),
        limit=arguments.get("limit", 20),
        cursor=arguments.get("cursor"),
        include_sensitive=False,
    )
    return _read_result(window)


def _read_result(window: Any) -> ToolResult:
    payload = window.to_dict()
    return ToolResult(
        tool_name="session_read",
        success=True,
        observation=json.dumps(payload, sort_keys=True),
        metadata={
            "conversation_id": window.conversation_id,
            "message_ids": [
                message.message_id for message in window.messages
            ],
            "next_cursor": window.next_cursor,
        },
        value=payload,
    )


__all__ = [
    "async_session_read_tool",
    "async_session_search_tool",
    "session_read_tool",
    "session_search_tool",
]
