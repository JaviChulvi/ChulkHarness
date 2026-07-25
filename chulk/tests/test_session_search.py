"""Tests for profile-scoped exact session search."""

from __future__ import annotations

from pathlib import Path

import pytest

from chulk.sessions import (
    MAX_SESSION_QUERY_CHARS,
    SessionNotFoundError,
    SessionSearchService,
    SQLiteSessionStore,
)


def _conversation(
    store: SQLiteSessionStore,
    conversation_id: str,
    *,
    profile_id: str,
    trace_path: str | None = None,
) -> None:
    store.create_conversation(
        conversation_id,
        provider="test",
        model="mock",
        trace_path=trace_path,
        metadata={"profile_id": profile_id},
    )


def _message(
    store: SQLiteSessionStore,
    conversation_id: str,
    key: str,
    content: str,
    *,
    role: str = "user",
    metadata: dict[str, object] | None = None,
    created_at: str | None = None,
) -> None:
    store.save_message(
        conversation_id,
        role=role,
        content=content,
        turn_id=f"turn-{key}",
        message_key=key,
        metadata=metadata,
        created_at=created_at,
    )


def test_search_returns_bounded_ranked_provenance_and_pages(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(
        store,
        "conversation-alpha",
        profile_id="alpha",
        trace_path="traces/alpha.jsonl",
    )
    _message(
        store,
        "conversation-alpha",
        "both",
        "alpha beta evidence with token=super-secret-value",
        role="assistant",
        created_at="2026-07-25T10:00:00+00:00",
    )
    _message(
        store,
        "conversation-alpha",
        "alpha",
        "alpha only evidence",
        created_at="2026-07-25T11:00:00+00:00",
    )
    _message(
        store,
        "conversation-alpha",
        "beta",
        "beta only evidence",
        created_at="2026-07-25T12:00:00+00:00",
    )
    service = SessionSearchService(store, profile_id="alpha")

    first = service.search("alpha beta", limit=1)

    assert len(first.hits) == 1
    assert first.hits[0].conversation_id == "conversation-alpha"
    assert first.hits[0].role in {"user", "assistant"}
    assert first.hits[0].created_at
    assert first.hits[0].turn_id
    assert first.hits[0].trace_path == "traces/alpha.jsonl"
    assert "super-secret-value" not in first.hits[0].snippet
    assert first.next_cursor is not None

    second = service.search("alpha beta", limit=1, cursor=first.next_cursor)

    assert len(second.hits) == 1
    assert second.hits[0].message_id != first.hits[0].message_id
    assert second.next_cursor is not None
    with pytest.raises(ValueError, match="does not match"):
        service.search("different query", limit=1, cursor=first.next_cursor)


def test_search_escapes_fts_syntax_and_supports_unicode_fallback(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "conversation-alpha", profile_id="alpha")
    _message(
        store,
        "conversation-alpha",
        "unicode",
        "Revisión del café y búsqueda Unicode.",
    )
    service = SessionSearchService(store, profile_id="alpha")

    assert service.search('café OR "unterminated:*', limit=10).hits

    store.fts_enabled = False
    page = service.search("REVISIÓN", limit=10)

    assert [hit.conversation_id for hit in page.hits] == ["conversation-alpha"]


def test_search_validates_empty_query_term_count_and_limits(tmp_path: Path) -> None:
    service = SessionSearchService(
        SQLiteSessionStore(tmp_path / "store.sqlite"),
        profile_id="alpha",
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        service.search("   ")
    with pytest.raises(ValueError, match="characters"):
        service.search("x" * (MAX_SESSION_QUERY_CHARS + 1))
    with pytest.raises(ValueError, match="terms"):
        service.search("one two three four five six seven eight nine")
    with pytest.raises(ValueError, match="between 1 and 50"):
        service.search("query", limit=0)
    with pytest.raises(ValueError, match="integer"):
        service.search("query", limit=True)


def test_search_enforces_profile_ownership_and_content_exclusions(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "owned", profile_id="alpha")
    _conversation(store, "other", profile_id="beta")
    _conversation(store, "legacy", profile_id="default")
    _message(store, "owned", "visible", "needle visible evidence")
    _message(store, "other", "other", "needle other profile")
    _message(store, "legacy", "legacy", "needle legacy profile")
    excluded = [
        ("observation", "raw tool arguments", "observation", None),
        ("internal", "hidden prompt", "assistant", {"internal": True}),
        ("prompt", "hidden plan", "assistant", {"prompt_excluded": True}),
        ("sensitive", "sensitive value", "user", {"sensitive": True}),
        ("secret", "secret value", "user", {"contains_secrets": True}),
        ("external", "external value", "user", {"external_content": True}),
    ]
    for key, label, role, metadata in excluded:
        _message(
            store,
            "owned",
            key,
            f"needle {label}",
            role=role,
            metadata=metadata,
        )

    page = SessionSearchService(store, profile_id="alpha").search("needle")

    assert [hit.snippet for hit in page.hits] == ["needle visible evidence"]
    assert {hit.conversation_id for hit in page.hits} == {"owned"}
    assert [
        hit.conversation_id
        for hit in SessionSearchService(store, profile_id="default")
        .search("needle")
        .hits
    ] == ["legacy"]


def test_window_reads_paginate_without_crossing_query_shape(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "owned", profile_id="alpha")
    for ordinal in range(1, 8):
        _message(
            store,
            "owned",
            f"message-{ordinal}",
            f"message {ordinal}",
            role="assistant" if ordinal % 2 == 0 else "user",
        )
    service = SessionSearchService(store, profile_id="alpha")

    first = service.read_window(
        "owned",
        ordinal=4,
        before=3,
        after=3,
        limit=2,
    )
    second = service.read_window(
        "owned",
        ordinal=4,
        before=3,
        after=3,
        limit=2,
        cursor=first.next_cursor,
    )

    assert [message.ordinal for message in first.messages] == [1, 2]
    assert [message.ordinal for message in second.messages] == [3, 4]
    assert first.next_cursor is not None
    with pytest.raises(ValueError, match="does not match"):
        service.read_window(
            "owned",
            ordinal=5,
            before=3,
            after=3,
            limit=2,
            cursor=first.next_cursor,
        )


def test_window_sensitive_override_is_host_only_and_never_includes_external_content(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "owned", profile_id="alpha")
    _message(store, "owned", "normal", "normal token=plain-secret")
    _message(
        store,
        "owned",
        "sensitive",
        "sensitive token=host-secret",
        metadata={"sensitive": True},
    )
    _message(
        store,
        "owned",
        "external",
        "untrusted external text",
        metadata={"external_content": True},
    )
    _message(
        store,
        "owned",
        "internal",
        "internal prompt text",
        role="assistant",
        metadata={"internal": True},
    )
    service = SessionSearchService(store, profile_id="alpha")

    safe = service.read_window("owned", ordinal=2, before=2, after=2)
    trusted = service.read_window(
        "owned",
        ordinal=2,
        before=2,
        after=2,
        include_sensitive=True,
    )

    assert [message.ordinal for message in safe.messages] == [1]
    assert "plain-secret" not in safe.messages[0].content
    assert [message.ordinal for message in trusted.messages] == [1, 2]
    assert trusted.messages[1].content == "sensitive token=host-secret"
    assert trusted.messages[1].sensitive is True


def test_window_checks_profile_ownership_before_reading_content(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "other", profile_id="beta")
    _message(store, "other", "message", "private beta evidence")

    with pytest.raises(SessionNotFoundError, match="No session found"):
        SessionSearchService(store, profile_id="alpha").read_window(
            "other",
            ordinal=1,
        )


def test_search_audit_contains_only_query_metadata_and_selected_ids(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "owned", profile_id="alpha")
    _message(store, "owned", "message", "audit-only private phrase")
    events: list[tuple[str, dict[str, object]]] = []
    service = SessionSearchService(
        store,
        profile_id="alpha",
        audit_callback=lambda event, payload: events.append((event, payload)),
    )

    hit = service.search("private phrase").hits[0]
    service.read_window("owned", ordinal=hit.ordinal)

    assert [event for event, _payload in events] == [
        "session_search",
        "session_read",
    ]
    serialized = repr(events)
    assert "audit-only private phrase" not in serialized
    assert hit.message_id in serialized

