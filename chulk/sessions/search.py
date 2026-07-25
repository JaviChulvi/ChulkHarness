"""Profile-scoped exact session search and bounded window reads."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
import hashlib
import json
import re
import sqlite3
from typing import Any

from chulk.redaction import redact_text
from chulk.sessions.models import (
    SessionHit,
    SessionMessage,
    SessionSearchPage,
    SessionWindow,
)
from chulk.sessions.sqlite_store import (
    SQLiteSessionStore,
    SessionNotFoundError,
    _message_is_search_eligible,
    _safe_json_dict,
)


MAX_SESSION_QUERY_CHARS = 256
MAX_SESSION_QUERY_TERMS = 8
MAX_SESSION_SEARCH_LIMIT = 50
MAX_SESSION_WINDOW_LIMIT = 100
MAX_SESSION_WINDOW_RADIUS = 100
MAX_SESSION_SNIPPET_CHARS = 280
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)

SessionAuditCallback = Callable[[str, dict[str, Any]], None]
SessionRedactor = Callable[[str], str]


class SessionSearchService:
    """Read-only search facade bound to one profile runtime database."""

    def __init__(
        self,
        store: SQLiteSessionStore,
        *,
        profile_id: str = "default",
        redactor: SessionRedactor | None = None,
        audit_callback: SessionAuditCallback | None = None,
    ) -> None:
        clean_profile_id = profile_id.strip()
        if not clean_profile_id:
            raise ValueError("session search profile_id cannot be empty")
        self.store = store
        self.profile_id = clean_profile_id
        self.redactor = redactor or redact_text
        self.audit_callback = audit_callback

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        cursor: str | None = None,
    ) -> SessionSearchPage:
        clean_query, terms = _parse_query(query)
        clean_limit = _bounded_int(
            "session search limit",
            limit,
            maximum=MAX_SESSION_SEARCH_LIMIT,
        )
        shape = _shape_hash("search", self.profile_id, clean_query)
        offset = _decode_search_cursor(cursor, shape=shape) if cursor else 0
        candidates = self._search_candidates(
            terms,
            limit=min(10_000, max(100, (offset + clean_limit + 1) * 5)),
        )
        eligible = [
            row
            for row in candidates
            if _conversation_is_owned(
                _safe_json_dict(row["conversation_metadata"]),
                self.profile_id,
            )
            and _message_is_search_eligible(str(row["role"]), row["metadata"])
        ]
        page_rows = eligible[offset : offset + clean_limit + 1]
        has_more = len(page_rows) > clean_limit
        page_rows = page_rows[:clean_limit]
        hits = tuple(self._row_to_hit(row, terms) for row in page_rows)
        next_cursor = (
            _encode_cursor({"kind": "search", "shape": shape, "offset": offset + len(hits)})
            if has_more
            else None
        )
        self._audit(
            "session_search",
            {
                "profile_id": self.profile_id,
                "query_hash": hashlib.sha256(clean_query.encode()).hexdigest(),
                "limit": clean_limit,
                "message_ids": [hit.message_id for hit in hits],
                "conversation_ids": sorted(
                    {hit.conversation_id for hit in hits}
                ),
            },
        )
        return SessionSearchPage(clean_query, hits, next_cursor)

    def read_window(
        self,
        conversation_id: str,
        *,
        ordinal: int,
        before: int = 3,
        after: int = 3,
        limit: int = 20,
        cursor: str | None = None,
        include_sensitive: bool = False,
    ) -> SessionWindow:
        conversation = self.store.get_conversation(conversation_id)
        if not _conversation_is_owned(conversation.metadata, self.profile_id):
            raise SessionNotFoundError(
                f"No session found for id: {conversation_id}"
            )
        anchor = _bounded_int("session ordinal", ordinal, maximum=2_147_483_647)
        clean_before = _bounded_int(
            "session window before",
            before,
            maximum=MAX_SESSION_WINDOW_RADIUS,
            minimum=0,
        )
        clean_after = _bounded_int(
            "session window after",
            after,
            maximum=MAX_SESSION_WINDOW_RADIUS,
            minimum=0,
        )
        clean_limit = _bounded_int(
            "session window limit",
            limit,
            maximum=MAX_SESSION_WINDOW_LIMIT,
        )
        lower = max(1, anchor - clean_before)
        upper = anchor + clean_after
        shape = _shape_hash(
            "window",
            self.profile_id,
            conversation.id,
            str(anchor),
            str(clean_before),
            str(clean_after),
            str(include_sensitive),
        )
        next_ordinal = (
            _decode_window_cursor(
                cursor,
                shape=shape,
                conversation_id=conversation.id,
            )
            if cursor
            else lower
        )
        with self.store._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conversation_messages
                WHERE conversation_id = ?
                  AND ordinal >= ?
                  AND ordinal <= ?
                ORDER BY ordinal, id
                """,
                (conversation.id, next_ordinal, upper),
            ).fetchall()
        visible = [
            row
            for row in rows
            if _window_message_visible(
                str(row["role"]),
                row["metadata"],
                include_sensitive=include_sensitive,
            )
        ]
        page_rows = visible[: clean_limit + 1]
        has_more = len(page_rows) > clean_limit
        page_rows = page_rows[:clean_limit]
        messages = tuple(
            self._row_to_window_message(
                row,
                include_sensitive=include_sensitive,
            )
            for row in page_rows
        )
        next_cursor = (
            _encode_cursor(
                {
                    "kind": "window",
                    "shape": shape,
                    "conversation_id": conversation.id,
                    "next_ordinal": messages[-1].ordinal + 1,
                }
            )
            if has_more and messages
            else None
        )
        self._audit(
            "session_read",
            {
                "profile_id": self.profile_id,
                "conversation_id": conversation.id,
                "anchor_ordinal": anchor,
                "before": clean_before,
                "after": clean_after,
                "limit": clean_limit,
                "include_sensitive": include_sensitive,
                "message_ids": [message.message_id for message in messages],
            },
        )
        return SessionWindow(
            conversation_id=conversation.id,
            anchor_ordinal=anchor,
            messages=messages,
            next_cursor=next_cursor,
        )

    def _search_candidates(
        self,
        terms: tuple[str, ...],
        *,
        limit: int,
    ) -> list[sqlite3.Row]:
        if self.store.fts_enabled:
            query = " OR ".join(f'"{term}"*' for term in terms)
            try:
                with self.store._connect() as conn:
                    return conn.execute(
                        """
                        SELECT messages.*, conversations.trace_path,
                               conversations.metadata AS conversation_metadata,
                               bm25(session_messages_fts) AS search_rank
                        FROM session_messages_fts
                        JOIN conversation_messages AS messages
                          ON messages.id = session_messages_fts.message_id
                        JOIN conversations
                          ON conversations.id = messages.conversation_id
                        WHERE session_messages_fts MATCH ?
                        ORDER BY search_rank, messages.created_at DESC,
                                 messages.conversation_id, messages.ordinal,
                                 messages.id
                        LIMIT ?
                        """,
                        (query, limit),
                    ).fetchall()
            except sqlite3.OperationalError:
                pass
        with self.store._connect() as conn:
            rows = conn.execute(
                """
                SELECT messages.*, conversations.trace_path,
                       conversations.metadata AS conversation_metadata,
                       0.0 AS search_rank
                FROM conversation_messages AS messages
                JOIN conversations
                  ON conversations.id = messages.conversation_id
                ORDER BY messages.created_at DESC, messages.conversation_id,
                         messages.ordinal, messages.id
                LIMIT ?
                """,
                (10_000,),
            ).fetchall()
        return [
            row
            for row in rows
            if any(term in str(row["content"]).casefold() for term in terms)
        ][:limit]

    def _row_to_hit(
        self,
        row: sqlite3.Row,
        terms: tuple[str, ...],
    ) -> SessionHit:
        content = str(row["content"])
        snippet = _snippet(content, terms)
        if self.redactor is not None:
            snippet = self.redactor(snippet)
        return SessionHit(
            message_id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            ordinal=int(row["ordinal"]),
            role=str(row["role"]),
            snippet=snippet,
            created_at=str(row["created_at"]),
            turn_id=row["turn_id"],
            trace_path=row["trace_path"],
        )

    def _row_to_window_message(
        self,
        row: sqlite3.Row,
        *,
        include_sensitive: bool,
    ) -> SessionMessage:
        metadata = _safe_json_dict(row["metadata"])
        sensitive = _message_is_sensitive(metadata)
        content = str(row["content"])
        if not include_sensitive and self.redactor is not None:
            content = self.redactor(content)
        return SessionMessage(
            message_id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            ordinal=int(row["ordinal"]),
            role=str(row["role"]),
            content=content,
            created_at=str(row["created_at"]),
            turn_id=row["turn_id"],
            sensitive=sensitive,
        )

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        if self.audit_callback is not None:
            self.audit_callback(event, payload)


def _parse_query(value: str) -> tuple[str, tuple[str, ...]]:
    clean = " ".join(value.strip().split())
    if not clean:
        raise ValueError("session search query cannot be empty")
    if len(clean) > MAX_SESSION_QUERY_CHARS:
        raise ValueError(
            f"session search query cannot exceed {MAX_SESSION_QUERY_CHARS} characters"
        )
    terms = tuple(
        dict.fromkeys(
            match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(clean)
        )
    )
    if not terms:
        raise ValueError("session search query must contain searchable text")
    if len(terms) > MAX_SESSION_QUERY_TERMS:
        raise ValueError(
            f"session search query cannot exceed {MAX_SESSION_QUERY_TERMS} terms"
        )
    return clean, terms


def _snippet(content: str, terms: tuple[str, ...]) -> str:
    compact = " ".join(content.split())
    lowered = compact.casefold()
    positions = [lowered.find(term) for term in terms]
    matches = [position for position in positions if position >= 0]
    start = max(0, (min(matches) if matches else 0) - MAX_SESSION_SNIPPET_CHARS // 3)
    end = min(len(compact), start + MAX_SESSION_SNIPPET_CHARS)
    value = compact[start:end]
    if start:
        value = f"…{value}"
    if end < len(compact):
        value = f"{value}…"
    return value


def _window_message_visible(
    role: str,
    metadata: object,
    *,
    include_sensitive: bool,
) -> bool:
    if role not in {"user", "assistant"}:
        return False
    values = _safe_json_dict(metadata)
    if values.get("internal") is True or values.get("prompt_excluded") is True:
        return False
    if values.get("external_content") is True:
        return False
    if include_sensitive:
        return True
    return not _message_is_sensitive(values)


def _message_is_sensitive(metadata: dict[str, Any]) -> bool:
    return (
        any(
            metadata.get(key) is True
            for key in ("sensitive", "contains_secrets", "external_content")
        )
        or metadata.get("content_class") == "sensitive"
    )


def _conversation_is_owned(
    metadata: dict[str, Any],
    profile_id: str,
) -> bool:
    owner = metadata.get("profile_id")
    if owner is None:
        return profile_id == "default"
    return owner == profile_id


def _bounded_int(
    label: str,
    value: int,
    *,
    maximum: int,
    minimum: int = 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return value


def _shape_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def _encode_cursor(payload: dict[str, Any]) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _decode_cursor(value: str) -> dict[str, Any]:
    try:
        padding = "=" * (-len(value) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode((value + padding).encode()).decode()
        )
    except (
        binascii.Error,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("invalid session cursor") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid session cursor")
    return payload


def _decode_search_cursor(value: str, *, shape: str) -> int:
    payload = _decode_cursor(value)
    if payload.get("kind") != "search" or payload.get("shape") != shape:
        raise ValueError("session search cursor does not match the query")
    return _bounded_int(
        "session search cursor offset",
        _cursor_int(payload, "offset"),
        maximum=10_000,
        minimum=0,
    )


def _decode_window_cursor(
    value: str,
    *,
    shape: str,
    conversation_id: str,
) -> int:
    payload = _decode_cursor(value)
    if (
        payload.get("kind") != "window"
        or payload.get("shape") != shape
        or payload.get("conversation_id") != conversation_id
    ):
        raise ValueError("session window cursor does not match the query")
    return _bounded_int(
        "session window cursor ordinal",
        _cursor_int(payload, "next_ordinal"),
        maximum=2_147_483_647,
    )


def _cursor_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid session cursor")
    return value


__all__ = [
    "MAX_SESSION_QUERY_CHARS",
    "MAX_SESSION_QUERY_TERMS",
    "MAX_SESSION_SEARCH_LIMIT",
    "MAX_SESSION_SNIPPET_CHARS",
    "MAX_SESSION_WINDOW_LIMIT",
    "MAX_SESSION_WINDOW_RADIUS",
    "SessionSearchService",
]
