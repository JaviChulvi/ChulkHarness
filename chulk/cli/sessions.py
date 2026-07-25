"""Profile-scoped session search CLI commands."""

from __future__ import annotations

from collections.abc import Callable
import sqlite3

from chulk.cli.entrypoints import EXIT_OK, EXIT_RUNTIME_ERROR, json_text
from chulk.sessions import (
    AmbiguousSessionError,
    SessionNotFoundError,
    SessionSearchService,
    SQLiteSessionStore,
)


def run_session_command(
    command: str,
    *,
    store: SQLiteSessionStore,
    profile_id: str,
    query: str | None = None,
    conversation_id: str | None = None,
    ordinal: int | None = None,
    before: int = 3,
    after: int = 3,
    limit: int = 20,
    cursor: str | None = None,
    json_output: bool = False,
    output_func: Callable[[str], None] = print,
    error_func: Callable[[str], None] = print,
) -> int:
    """Run one read-only session query or deterministic index rebuild."""
    service = SessionSearchService(store, profile_id=profile_id)
    try:
        if command == "search":
            if query is None:
                raise ValueError("session search requires a query")
            page = service.search(query, limit=limit, cursor=cursor)
            payload = {
                "ok": True,
                "profile_id": profile_id,
                **page.to_dict(),
            }
            output_func(
                json_text(payload)
                if json_output
                else _format_search(page.to_dict())
            )
            return EXIT_OK
        if command == "read":
            if conversation_id is None or ordinal is None:
                raise ValueError(
                    "session read requires a conversation id and ordinal"
                )
            window = service.read_window(
                conversation_id,
                ordinal=ordinal,
                before=before,
                after=after,
                limit=limit,
                cursor=cursor,
                include_sensitive=False,
            )
            payload = {
                "ok": True,
                "profile_id": profile_id,
                **window.to_dict(),
            }
            output_func(
                json_text(payload)
                if json_output
                else _format_window(window.to_dict())
            )
            return EXIT_OK
        if command == "rebuild-index":
            indexed_messages = store.rebuild_search_index()
            payload = {
                "ok": True,
                "profile_id": profile_id,
                "fts_enabled": store.fts_enabled,
                "indexed_messages": indexed_messages,
            }
            output_func(
                json_text(payload)
                if json_output
                else _format_rebuild(payload)
            )
            return EXIT_OK
        raise ValueError(f"Unknown session command: {command}")
    except (
        AmbiguousSessionError,
        OSError,
        SessionNotFoundError,
        sqlite3.Error,
        ValueError,
    ) as exc:
        if json_output:
            output_func(
                json_text(
                    {
                        "ok": False,
                        "status": "session_error",
                        "error": str(exc),
                    }
                )
            )
        else:
            error_func(f"session error: {exc}")
        return EXIT_RUNTIME_ERROR


def _format_search(payload: dict[str, object]) -> str:
    hits = payload.get("hits")
    values = hits if isinstance(hits, list) else []
    lines = [f"Session search: {len(values)} hit(s)"]
    for value in values:
        if not isinstance(value, dict):
            continue
        lines.append(
            f"  {value.get('conversation_id')}:{value.get('ordinal')} "
            f"[{value.get('role')}] {value.get('snippet')}"
        )
    cursor = payload.get("next_cursor")
    if cursor:
        lines.append(f"  next cursor: {cursor}")
    return "\n".join(lines)


def _format_window(payload: dict[str, object]) -> str:
    messages = payload.get("messages")
    values = messages if isinstance(messages, list) else []
    lines = [
        f"Session {payload.get('conversation_id')} around "
        f"ordinal {payload.get('anchor_ordinal')}:"
    ]
    for value in values:
        if not isinstance(value, dict):
            continue
        lines.append(
            f"  {value.get('ordinal')} [{value.get('role')}] "
            f"{value.get('content')}"
        )
    cursor = payload.get("next_cursor")
    if cursor:
        lines.append(f"  next cursor: {cursor}")
    return "\n".join(lines)


def _format_rebuild(payload: dict[str, object]) -> str:
    if payload["fts_enabled"]:
        return (
            "Session search index rebuilt: "
            f"{payload['indexed_messages']} eligible message(s)."
        )
    return "Session search FTS5 unavailable; bounded fallback search remains active."


__all__ = ["run_session_command"]
