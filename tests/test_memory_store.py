"""Tests for the SQLite-backed long-term memory store."""

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from chulk.memory import (
    DEFAULT_MEMORY_NAMESPACE,
    MemoryRetentionPolicy,
    SQLiteMemoryStore,
    normalize_memory_namespace,
    select_memories_for_prompt,
    text_to_embedding,
)


def test_memory_namespace_normalization_is_opaque_and_compatible() -> None:
    assert normalize_memory_namespace(None) == DEFAULT_MEMORY_NAMESPACE
    assert normalize_memory_namespace(" Tenant:Workspace-1 ") == "tenant:workspace-1"
    for invalid in ("", "two words", "../escape", "üser", "x" * 129):
        with pytest.raises(ValueError, match="Memory namespace"):
            normalize_memory_namespace(invalid)


def test_sqlite_memory_store_saves_searches_lists_and_deletes(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    memory_id = store.save_memory(
        "Javier prefers direct repo-grounded implementation details.",
        tags=["preference", "workflow"],
        metadata={"source": "test"},
        importance=8,
    )

    search_results = store.search_memory("repo implementation")
    listed = store.list_memories()

    assert store.db_path.exists()
    assert search_results[0].id == memory_id
    assert search_results[0].tags == ["preference", "workflow"]
    assert search_results[0].metadata == {"source": "test"}
    assert search_results[0].importance == 8
    assert search_results[0].source == "manual"
    assert search_results[0].confidence == 1.0
    assert search_results[0].embedding
    assert search_results[0].namespace == DEFAULT_MEMORY_NAMESPACE
    assert listed[0].id == memory_id
    assert store.delete_memory(memory_id)
    assert store.search_memory("repo implementation") == []


def test_sqlite_memory_store_updates_memory(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    memory_id = store.save_memory("Old memory", tags=["project"], importance=2)

    updated = store.update_memory(
        memory_id,
        content="Updated memory about ChulkHarness",
        tags=["project", "chulk"],
        metadata={"repo": "chulk"},
        importance=5,
    )
    memory = store.get_memory(memory_id)

    assert updated
    assert memory is not None
    assert memory.content == "Updated memory about ChulkHarness"
    assert memory.tags == ["project", "chulk"]
    assert memory.metadata == {"repo": "chulk"}
    assert memory.importance == 5
    assert memory.updated_at >= memory.created_at


def test_sqlite_memory_store_fts_embedding_and_source_confidence(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    memory_id = store.save_memory(
        "ChulkHarness memory retrieval should use FTS before embeddings.",
        tags=["project", "search"],
        source="test",
        confidence=0.7,
        importance=6,
    )

    fts_results = store.search_memory("retrieval embeddings")
    vector_results = store.search_memory_by_embedding(text_to_embedding("memory retrieval"), limit=3)

    assert store.fts_enabled
    assert fts_results[0].id == memory_id
    assert vector_results[0].id == memory_id
    assert fts_results[0].source == "test"
    assert fts_results[0].confidence == 0.7


def test_sqlite_memory_store_summarizes_and_finds_profile_memories(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    profile_id = store.save_memory(
        "Prefer exact files, exact commands, and direct answers.",
        tags=["persona", "preference"],
        importance=9,
    )
    task_id = store.save_memory("ChulkHarness uses SQLite for durable memory.", tags=["project"], importance=4)

    profile, relevant = select_memories_for_prompt(store, "How does SQLite memory work?")
    summary = store.summarize_memories("SQLite")

    assert [memory.id for memory in profile] == [profile_id]
    assert [memory.id for memory in relevant] == [task_id]
    assert profile_id in store.summarize_memories("direct answers")
    assert "SQLite for durable memory" in summary


def test_profile_memories_keep_distinct_preferences_and_suppress_duplicates(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    first_id = store.save_memory("Prefer exact commands.", tags=["preference"], confidence=0.8)
    second_id = store.save_memory("Prefer concise summaries.", tags=["preference"], confidence=0.8)
    duplicate_id = store.save_memory("Prefer exact commands.", tags=["preference"], confidence=0.5, dedupe=False)

    profile_ids = [memory.id for memory in store.profile_memories(limit=5)]

    assert first_id in profile_ids
    assert second_id in profile_ids
    assert duplicate_id not in profile_ids


def test_sqlite_memory_store_deduplicates_archives_restores_and_compacts(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    first_id = store.save_memory("Prefer exact commands and tested changes.", tags=["preference"], confidence=0.6)
    duplicate_id = store.save_memory("Prefer exact commands and tested changes.", tags=["workflow"], confidence=0.9)

    assert duplicate_id == first_id
    merged = store.get_memory(first_id)
    assert merged is not None
    assert set(merged.tags) == {"preference", "workflow"}
    assert merged.confidence == 0.9

    second_id = store.save_memory(
        "Prefer exact commands and tested changes.",
        tags=["preference"],
        dedupe=False,
    )
    archived_count = store.compact_memories()

    assert archived_count == 1
    assert len(store.list_memories()) == 1
    assert store.archive_memory(first_id) or store.archive_memory(second_id)
    archived = store.list_memories(include_archived=True)
    assert any(memory.archived_at for memory in archived)
    archived_id = next(memory.id for memory in archived if memory.archived_at)
    assert store.restore_memory(archived_id)


def test_sqlite_memory_store_applies_archive_only_retention_by_namespace(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite", namespace="tenant:alpha")
    old_id = store.save_memory("old memory", importance=1)
    proposal_id = store.create_memory_proposal(
        "pending memory",
        evidence="user asked to remember it",
        conversation_id="conversation-1",
        turn_id="turn-1",
    )
    created_at = datetime.fromisoformat(store.get_memory(old_id).updated_at)

    retention_time = created_at + timedelta(days=31)
    archived = store.apply_retention(
        MemoryRetentionPolicy(max_age_days=30),
        now=retention_time,
    )

    assert archived == 1
    archived_memory = store.get_memory(old_id, include_archived=True)
    assert archived_memory is not None
    assert archived_memory.archived_at == retention_time.isoformat()
    proposal = store.get_memory_proposal(proposal_id)
    assert proposal is not None
    assert proposal.status == "pending"
    assert proposal.evidence == "user asked to remember it"


def test_sqlite_memory_store_applies_count_retention_only_in_namespace(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite", namespace="tenant:alpha")
    other = SQLiteMemoryStore(tmp_path / "memory.sqlite", namespace="tenant:beta")
    archive_id = store.save_memory("archive me", importance=1)
    keep_id = store.save_memory("keep me", importance=9)
    other_id = other.save_memory("other namespace memory", importance=1)

    assert store.apply_retention(MemoryRetentionPolicy(max_active_items=1)) == 1
    archived = store.get_memory(archive_id, include_archived=True)
    assert archived is not None and archived.archived_at is not None
    retained = store.get_memory(keep_id)
    assert retained is not None and retained.archived_at is None
    other_memory = other.get_memory(other_id)
    assert other_memory is not None and other_memory.archived_at is None


def test_combined_retention_counts_only_nonexpired_memories(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    expired_ids = [
        store.save_memory(f"expired memory record{index:03d}", importance=10)
        for index in range(20)
    ]
    fresh_ids = [
        store.save_memory(f"fresh memory record{index:03d}", importance=1)
        for index in range(90)
    ]
    latest_fresh = store.get_memory(fresh_ids[-1])
    assert latest_fresh is not None
    now = datetime.fromisoformat(latest_fresh.updated_at)
    expired_at = (now - timedelta(days=31)).isoformat()
    with sqlite3.connect(store.db_path) as conn:
        conn.executemany(
            "UPDATE memories SET updated_at = ? WHERE id = ?",
            [(expired_at, memory_id) for memory_id in expired_ids],
        )

    archived = store.apply_retention(
        MemoryRetentionPolicy(max_age_days=30, max_active_items=100),
        now=now,
    )

    assert archived == 20
    assert all(
        store.get_memory(memory_id, include_archived=True).archived_at == now.astimezone(timezone.utc).isoformat()
        for memory_id in expired_ids
    )
    assert all(store.get_memory(memory_id) is not None for memory_id in fresh_ids)
    assert len(store.list_memories(limit=100)) == 90


def test_memory_retention_policy_validates_limits_and_clock(tmp_path):
    with pytest.raises(ValueError, match="at least one limit"):
        MemoryRetentionPolicy()
    with pytest.raises(ValueError, match="max_age_days"):
        MemoryRetentionPolicy(max_age_days=0)
    with pytest.raises(ValueError, match="max_age_days"):
        MemoryRetentionPolicy(max_age_days="30")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_active_items"):
        MemoryRetentionPolicy(max_active_items=True)

    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.save_memory("memory")
    with pytest.raises(ValueError, match="timezone-aware"):
        store.apply_retention(
            MemoryRetentionPolicy(max_age_days=1),
            now=datetime(2026, 8, 29),
        )


def test_sqlite_memory_store_extracts_and_imports_exports_markdown(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    extracted_ids = store.extract_and_save_memories("Please remember that project alpha uses SQLite memory.")
    markdown = tmp_path / "MEMORY.md"
    markdown.write_text("- [persona, preference] Prefer concise implementation summaries.\n", encoding="utf-8")

    imported_ids = store.import_markdown(markdown)
    export_path = tmp_path / "exported.md"
    exported_count = store.export_markdown(export_path)

    assert extracted_ids
    assert imported_ids
    assert "project alpha uses SQLite memory" in store.get_memory(extracted_ids[0]).content
    assert "Prefer concise implementation summaries" in store.get_memory(imported_ids[0]).content
    assert exported_count >= 2
    assert "MEMORY" in export_path.read_text(encoding="utf-8")


def test_sqlite_memory_store_validates_inputs(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    for kwargs in [
        {"content": ""},
        {"content": "x", "importance": 0},
        {"content": "x", "importance": 11},
        {"content": "x", "confidence": -0.1},
        {"content": "x", "confidence": 1.1},
    ]:
        try:
            store.save_memory(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid memory arguments to fail: {kwargs}")
