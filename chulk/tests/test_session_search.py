"""Tests for profile-scoped exact session search."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import chulk.sessions.sqlite_store as session_store_module
from chulk import Agent, AgentConfig, AsyncAgent
from chulk.cli.sessions import run_session_command
from chulk.llm import LLMClient
from chulk.main import main
from chulk.sessions import (
    MAX_SESSION_QUERY_CHARS,
    SessionNotFoundError,
    SessionSearchService,
    SQLiteSessionStore,
)
from chulk.tools import (
    ToolPermissionLevel,
    session_read_tool,
    session_search_tool,
)


class _UnusedLLM(LLMClient):
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("session read APIs must not call the model")


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


def test_message_updates_and_deletes_synchronize_search_transactionally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "owned", profile_id="alpha")
    _message(store, "owned", "message", "original searchable evidence")
    service = SessionSearchService(store, profile_id="alpha")
    message_id = service.search("original").hits[0].message_id

    assert store.update_message(
        "owned",
        message_id,
        content="revised searchable evidence",
    )
    assert service.search("original").hits == ()
    assert service.search("revised").hits[0].message_id == message_id

    original_replace = session_store_module._replace_session_fts

    def fail_index_update(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(
        session_store_module,
        "_replace_session_fts",
        fail_index_update,
    )
    with pytest.raises(RuntimeError, match="simulated index failure"):
        store.update_message(
            "owned",
            message_id,
            content="must roll back",
        )
    monkeypatch.setattr(
        session_store_module,
        "_replace_session_fts",
        original_replace,
    )

    assert service.search("revised").hits[0].message_id == message_id
    assert service.search("roll back").hits == ()
    assert store.update_message(
        "owned",
        message_id,
        metadata={"sensitive": True},
    )
    assert service.search("revised").hits == ()
    assert store.delete_message("owned", message_id)
    assert store.delete_message("owned", message_id) is False


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


def test_model_tools_are_read_only_redacted_and_cannot_request_sensitive_text(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "owned", profile_id="alpha")
    _message(
        store,
        "owned",
        "normal",
        "deploy token=model-secret",
    )
    _message(
        store,
        "owned",
        "sensitive",
        "deploy sensitive text",
        metadata={"sensitive": True},
    )
    service = SessionSearchService(store, profile_id="alpha")
    search_tool = session_search_tool(service)
    read_tool = session_read_tool(service)

    search_result = search_tool.callable({"query": "deploy"})
    read_result = read_tool.callable(
        {
            "conversation_id": "owned",
            "ordinal": 1,
            "before": 1,
            "after": 1,
        }
    )

    assert search_tool.permission_level is ToolPermissionLevel.READ
    assert read_tool.permission_level is ToolPermissionLevel.READ
    assert search_tool.idempotent is True
    assert read_tool.idempotent is True
    assert "include_sensitive" not in read_tool.args_schema["properties"]
    assert "model-secret" not in search_result.observation
    assert "model-secret" not in read_result.observation
    assert "sensitive text" not in read_result.observation


def test_public_sdk_exposes_profile_bound_session_search_and_trusted_read(
    tmp_path: Path,
) -> None:
    agent = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=_UnusedLLM(),
        tools=[],
        skills=[],
    )
    store = agent.runtime.session_store
    _conversation(store, "prior", profile_id="default")
    _message(store, "prior", "normal", "sdk searchable evidence")
    _message(
        store,
        "prior",
        "sensitive",
        "sdk sensitive evidence",
        metadata={"sensitive": True},
    )

    hit = agent.search_sessions("searchable").hits[0]
    safe = agent.read_session_window("prior", ordinal=hit.ordinal, after=2)
    trusted = agent.read_session_window(
        "prior",
        ordinal=hit.ordinal,
        after=2,
        include_sensitive=True,
    )

    assert hit.conversation_id == "prior"
    assert [message.ordinal for message in safe.messages] == [1]
    assert [message.ordinal for message in trusted.messages] == [1, 2]
    assert agent.trace_path is not None
    trace_events = [
        json.loads(line)
        for line in agent.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    search_event = next(
        event for event in trace_events if event["type"] == "session_search"
    )
    serialized_event = json.dumps(search_event)
    assert search_event["payload"]["message_ids"] == [hit.message_id]
    assert "query_hash" in search_event["payload"]
    assert "sdk searchable evidence" not in serialized_event
    assert "sdk sensitive evidence" not in serialized_event
    agent.close()


@pytest.mark.asyncio
async def test_async_sdk_session_search_matches_sync_semantics(
    tmp_path: Path,
) -> None:
    agent = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=_UnusedLLM(),
        tools=[],
        skills=[],
    )
    store = agent.runtime.session_store
    _conversation(store, "prior", profile_id="default")
    _message(store, "prior", "normal", "async searchable evidence")

    page = await agent.search_sessions("searchable")
    window = await agent.read_session_window(
        "prior",
        ordinal=page.hits[0].ordinal,
    )

    assert page.hits[0].conversation_id == "prior"
    assert window.messages[0].content == "async searchable evidence"
    await agent.close()


def test_default_runtime_registers_session_tools(tmp_path: Path) -> None:
    agent = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=_UnusedLLM(),
        skills=[],
    )

    tools = {
        tool.name: tool for tool in agent.tool_registry.list_tools()
    }

    assert tools["session_search"].permission_level is ToolPermissionLevel.READ
    assert tools["session_read"].permission_level is ToolPermissionLevel.READ
    agent.close()


def test_session_cli_search_read_and_rebuild_share_the_service(
    tmp_path: Path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "store.sqlite")
    _conversation(store, "prior", profile_id="alpha")
    _message(store, "prior", "normal", "cli searchable evidence")
    output: list[str] = []

    assert (
        run_session_command(
            "search",
            store=store,
            profile_id="alpha",
            query="searchable",
            json_output=True,
            output_func=output.append,
        )
        == 0
    )
    assert json.loads(output[-1])["hits"][0]["conversation_id"] == "prior"
    assert (
        run_session_command(
            "read",
            store=store,
            profile_id="alpha",
            conversation_id="prior",
            ordinal=1,
            output_func=output.append,
        )
        == 0
    )
    assert "cli searchable evidence" in output[-1]
    assert (
        run_session_command(
            "rebuild-index",
            store=store,
            profile_id="alpha",
            json_output=True,
            output_func=output.append,
        )
        == 0
    )
    assert json.loads(output[-1])["indexed_messages"] == 1


def test_main_routes_session_rebuild_without_constructing_a_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    output: list[str] = []

    exit_code = main(
        ["session", "rebuild-index", "--json"],
        output_func=output.append,
        error_func=output.append,
    )

    assert exit_code == 0
    payload = json.loads(output[-1])
    assert payload["ok"] is True
    assert isinstance(payload["fts_enabled"], bool)
