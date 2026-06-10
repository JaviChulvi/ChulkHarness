"""Tests for the SQLite-backed long-term memory store."""

from src.memory import SQLiteMemoryStore, select_memories_for_prompt, text_to_embedding


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
