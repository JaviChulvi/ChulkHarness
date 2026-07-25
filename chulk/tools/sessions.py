"""Read-only tools for exact prior-session evidence."""

from __future__ import annotations

import json
from typing import Any

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


def _search(
    arguments: dict[str, Any],
    service: SessionSearchService,
) -> ToolResult:
    page = service.search(
        str(arguments["query"]),
        limit=arguments.get("limit", 10),
        cursor=arguments.get("cursor"),
    )
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


__all__ = ["session_read_tool", "session_search_tool"]
