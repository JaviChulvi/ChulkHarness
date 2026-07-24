"""Regression tests for the shared SQLite reliability boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
from pathlib import Path
import sqlite3
import stat
from threading import Barrier, Event

import pytest

import chulk.memory.sqlite_store as memory_store_module
import chulk.storage.sqlite as sqlite_policy_module
from chulk.memory import SQLiteMemoryStore
from chulk.sessions import SQLiteSessionStore
from chulk.storage import (
    SQLITE_BUSY_TIMEOUT_MS,
    SQLITE_SCHEMA_VERSION,
    SQLITE_WAL_AUTOCHECKPOINT_PAGES,
    SQLiteMigration,
    SQLiteMigrationError,
    UnsupportedSQLiteSchemaVersionError,
    create_sqlite_backup,
    initialize_sqlite_database,
    sqlite_connection,
)
from chulk.storage.migrations import SQLITE_MIGRATIONS


def test_new_database_uses_shared_schema_and_explicit_connection_policy(tmp_path):
    path = tmp_path / "store.sqlite"
    store = SQLiteSessionStore(path)

    with store._connect() as conn:
        tables = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")
        }
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SQLITE_SCHEMA_VERSION
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == SQLITE_BUSY_TIMEOUT_MS
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == SQLITE_WAL_AUTOCHECKPOINT_PAGES

    assert {"memories", "memory_proposals", "conversations", "conversation_messages"} <= tables


def test_legacy_database_is_backed_up_migrated_and_deterministically_renumbered(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                importance INTEGER NOT NULL DEFAULT 1
            );
            INSERT INTO memories VALUES (
                'memory-1', 'legacy memory', '2026-01-01', '2026-01-01',
                '["Project", "project"]', '{}', 4
            );
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                trace_path TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO conversations VALUES (
                'conversation-1', NULL, 'test', 'mock', NULL, 'active',
                '2026-01-01', '2026-01-01', '{}'
            );
            CREATE TABLE conversation_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                turn_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                message_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            INSERT INTO conversation_messages VALUES (
                'later', 'conversation-1', NULL, 'assistant', 'later', 1,
                'later-key', '2026-01-02', '{}'
            );
            INSERT INTO conversation_messages VALUES (
                'earlier', 'conversation-1', NULL, 'user', 'earlier', 1,
                'earlier-key', '2026-01-01', '{}'
            );
            """
        )

    report = initialize_sqlite_database(path)

    assert report.from_version == 0
    assert report.to_version == SQLITE_SCHEMA_VERSION
    assert report.backup_path is not None
    assert report.backup_path.exists()
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        messages = conn.execute(
            "SELECT id, ordinal FROM conversation_messages ORDER BY ordinal"
        ).fetchall()
        tags = conn.execute("SELECT tag FROM memory_tags ORDER BY tag").fetchall()
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SQLITE_SCHEMA_VERSION
    with sqlite3.connect(report.backup_path) as backup:
        backup_ordinals = backup.execute(
            "SELECT ordinal FROM conversation_messages ORDER BY id"
        ).fetchall()
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 0

    assert [(row["id"], row["ordinal"]) for row in messages] == [("earlier", 1), ("later", 2)]
    assert [row["tag"] for row in tags] == ["project"]
    assert {"source", "confidence", "embedding", "archived_at"} <= columns
    assert backup_ordinals == [(1,), (1,)]


def test_memory_namespace_migration_preserves_v6_rows_and_rebuilds_fts(tmp_path):
    path = tmp_path / "legacy-memory-v6.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:6])
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            INSERT INTO memories (
                id, content, created_at, updated_at, tags, metadata, importance,
                source, confidence, embedding, archived_at, access_count,
                last_accessed_at
            )
            VALUES (
                'memory-1', 'legacy scoped memory', '2026-01-01', '2026-01-02',
                '["project"]', '{"origin":"v6"}', 7, 'manual', 0.8,
                '[0.1, 0.2]', NULL, 3, '2026-01-03'
            );
            INSERT INTO memory_tags (memory_id, tag)
            VALUES ('memory-1', 'project');
            INSERT INTO memory_proposals (
                id, content, tags, metadata, importance, source, confidence,
                evidence, conversation_id, turn_id, status, created_at,
                reviewed_at, accepted_memory_id
            )
            VALUES (
                'proposal-1', 'legacy proposal', '["workflow"]', '{}', 4,
                'manual_review', 0.9, 'evidence', 'conversation-1', 'turn-1',
                'pending', '2026-01-04', NULL, NULL
            );
            CREATE VIRTUAL TABLE memories_fts
            USING fts5(memory_id UNINDEXED, content, tags, metadata, source);
            INSERT INTO memories_fts
            VALUES ('stale-id', 'stale content', '', '{}', 'manual');
            """
        )

    store = SQLiteMemoryStore(path)
    memory = store.get_memory("memory-1")
    proposal = store.get_memory_proposal("proposal-1")

    assert memory is not None
    assert memory.namespace == "default"
    assert memory.content == "legacy scoped memory"
    assert memory.tags == ["project"]
    assert memory.metadata == {"origin": "v6"}
    assert memory.importance == 7
    assert memory.access_count == 3
    assert proposal is not None
    assert proposal.namespace == "default"
    assert proposal.content == "legacy proposal"
    assert proposal.to_dict()["namespace"] == "default"
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        memory_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(memories)")
        }
        proposal_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(memory_proposals)")
        }
        fts_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(memories_fts)")
        }
        indexes = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'index'"
            )
        }
        fts_rows = conn.execute(
            "SELECT memory_id, namespace FROM memories_fts"
        ).fetchall()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7

    backups = list(tmp_path.glob("legacy-memory-v6.sqlite.backup-v6-*.sqlite"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as backup:
        backup_columns = {
            row[1] for row in backup.execute("PRAGMA table_info(memories)")
        }
        backup_row = backup.execute(
            "SELECT content, tags, metadata, importance FROM memories WHERE id = 'memory-1'"
        ).fetchone()
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 6

    assert "namespace" in memory_columns
    assert "namespace" in proposal_columns
    assert "namespace" in fts_columns
    assert {
        "idx_memories_namespace_active",
        "idx_memory_proposals_namespace_status",
    } <= indexes
    assert [(row["memory_id"], row["namespace"]) for row in fts_rows] == [
        ("memory-1", "default")
    ]
    assert "namespace" not in backup_columns
    assert backup_row == (
        "legacy scoped memory",
        '["project"]',
        '{"origin":"v6"}',
        7,
    )


@pytest.mark.parametrize("store_type", [SQLiteMemoryStore, SQLiteSessionStore])
def test_store_rejects_a_database_from_a_newer_schema_version(tmp_path, store_type):
    path = tmp_path / f"{store_type.__name__}.sqlite"
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        conn.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO marker VALUES ('preserved')")
        conn.execute(f"PRAGMA user_version = {SQLITE_SCHEMA_VERSION + 1}")
    if os.name == "posix":
        os.chmod(path, 0o640)

    with pytest.raises(UnsupportedSQLiteSchemaVersionError) as exc_info:
        store_type(path)

    assert exc_info.value.found_version == SQLITE_SCHEMA_VERSION + 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone()[0] == "preserved"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SQLITE_SCHEMA_VERSION + 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()
    assert list(tmp_path.glob("*.backup-*.sqlite")) == []


@pytest.mark.skipif(os.name != "posix", reason="chmod race only applies on POSIX")
def test_permission_repair_tolerates_a_disappearing_wal_sidecar(monkeypatch, tmp_path):
    path = tmp_path / "sidecar-race.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE marker (value TEXT NOT NULL)")
    wal_path = Path(f"{path}-wal")
    wal_path.touch(mode=0o600)
    original_restrict = sqlite_policy_module._restrict_private_file

    def remove_before_chmod(candidate: Path) -> None:
        if candidate == wal_path:
            candidate.unlink()
        original_restrict(candidate)

    monkeypatch.setattr(sqlite_policy_module, "_restrict_private_file", remove_before_chmod)

    sqlite_policy_module._restrict_database_files(path)

    assert not wal_path.exists()


def test_journal_mode_policy_retries_transient_locked_errors(monkeypatch, tmp_path):
    path = tmp_path / "journal-mode-retry.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE marker (value TEXT NOT NULL)")
    original_set_journal_mode = sqlite_policy_module._set_journal_mode
    attempts = 0

    def transiently_locked(conn: sqlite3.Connection) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("database is locked")
        return original_set_journal_mode(conn)

    monkeypatch.setattr(sqlite_policy_module, "_set_journal_mode", transiently_locked)

    with sqlite_connection(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    assert attempts == 3


def test_failed_migration_rolls_back_and_retains_validated_private_backup(tmp_path):
    path = tmp_path / "failing.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE marker (value TEXT NOT NULL)")
        conn.execute("INSERT INTO marker VALUES ('before')")

    def fail_after_writes(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE partial_change (value TEXT NOT NULL)")
        conn.execute("UPDATE marker SET value = 'during'")
        raise RuntimeError("deliberate migration failure")

    with pytest.raises(SQLiteMigrationError) as exc_info:
        initialize_sqlite_database(
            path,
            migrations=(SQLiteMigration(1, "deliberate-failure", fail_after_writes),),
        )

    error = exc_info.value
    assert isinstance(error.__cause__, RuntimeError)
    assert error.backup_path is not None
    assert error.backup_path.exists()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM marker").fetchone()[0] == "before"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        assert conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE name = 'partial_change'"
        ).fetchone() is None
    with sqlite3.connect(error.backup_path) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert backup.execute("SELECT value FROM marker").fetchone()[0] == "before"
    if os.name == "posix":
        assert stat.S_IMODE(error.backup_path.stat().st_mode) == 0o600


def test_online_backup_includes_committed_wal_content(tmp_path):
    path = tmp_path / "live.sqlite"
    store = SQLiteSessionStore(path)
    store.create_conversation("conversation-1", provider="test", model="mock")

    with sqlite_connection(path) as anchor:
        store.save_message(
            "conversation-1",
            role="user",
            content="committed in WAL",
            message_key="message-1",
        )
        wal_path = Path(f"{path}-wal")
        assert wal_path.exists()
        assert wal_path.stat().st_size > 0
        backup_path = create_sqlite_backup(path)
        assert anchor.execute("SELECT count(*) FROM conversation_messages").fetchone()[0] == 1
        if os.name == "posix":
            for runtime_path in (path, wal_path, Path(f"{path}-shm"), backup_path):
                assert stat.S_IMODE(runtime_path.stat().st_mode) == 0o600

    with sqlite3.connect(backup_path) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert backup.execute("SELECT content FROM conversation_messages").fetchone()[0] == "committed in WAL"


def test_wal_reader_does_not_block_a_message_writer(tmp_path):
    path = tmp_path / "reader-writer.sqlite"
    store = SQLiteSessionStore(path)
    store.create_conversation("conversation-1", provider="test", model="mock")
    reader = sqlite3.connect(path)
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT count(*) FROM conversation_messages").fetchone()[0] == 0

        store.save_message(
            "conversation-1",
            role="user",
            content="writer completes",
            message_key="message-1",
        )

        assert reader.execute("SELECT count(*) FROM conversation_messages").fetchone()[0] == 0
        reader.commit()
        assert reader.execute("SELECT count(*) FROM conversation_messages").fetchone()[0] == 1
    finally:
        reader.close()


def test_contending_writer_waits_and_succeeds_after_lock_release(tmp_path):
    path = tmp_path / "writer-waits.sqlite"
    store = SQLiteSessionStore(path)
    store.create_conversation("conversation-1", provider="test", model="mock")
    lock = sqlite3.connect(path)
    lock.execute("BEGIN IMMEDIATE")
    started = Event()

    def write() -> None:
        started.set()
        store.save_message(
            "conversation-1",
            role="user",
            content="waited for lock",
            message_key="message-1",
        )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(write)
            assert started.wait(timeout=1)
            with pytest.raises(FutureTimeoutError):
                future.result(timeout=0.1)
            lock.commit()
            future.result(timeout=2)
    finally:
        if lock.in_transaction:
            lock.rollback()
        lock.close()

    assert [message.content for message in store.list_messages("conversation-1")] == ["waited for lock"]


def test_exhausted_busy_timeout_surfaces_locked_error_and_allows_retry(monkeypatch, tmp_path):
    path = tmp_path / "writer-timeout.sqlite"
    store = SQLiteSessionStore(path)
    store.create_conversation("conversation-1", provider="test", model="mock")
    monkeypatch.setattr(sqlite_policy_module, "SQLITE_BUSY_TIMEOUT_MS", 50)
    lock = sqlite3.connect(path)
    lock.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked") as exc_info:
            store.save_message(
                "conversation-1",
                role="user",
                content="retryable write",
                message_key="message-1",
            )
        assert exc_info.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert lock.execute("SELECT count(*) FROM conversation_messages").fetchone()[0] == 0
    finally:
        lock.rollback()
        lock.close()

    store.save_message(
        "conversation-1",
        role="user",
        content="retryable write",
        message_key="message-1",
    )
    assert [message.content for message in store.list_messages("conversation-1")] == ["retryable write"]


def test_concurrent_message_writers_receive_unique_contiguous_ordinals(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    store = SQLiteSessionStore(path)
    store.create_conversation("conversation-1", provider="test", model="mock")
    writer_count = 24
    race = Barrier(writer_count)

    def write_message(index: int) -> None:
        race.wait(timeout=5)
        store.save_message(
            "conversation-1",
            role="user",
            content=f"message {index}",
            message_key=f"message-{index}",
        )

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        list(executor.map(write_message, range(writer_count)))

    messages = store.list_messages("conversation-1", limit=writer_count)
    assert [message.ordinal for message in messages] == list(range(1, writer_count + 1))
    assert {message.content for message in messages} == {f"message {index}" for index in range(writer_count)}

    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO conversation_messages (
                    id, conversation_id, role, content, ordinal, message_key, created_at
                ) VALUES ('duplicate', 'conversation-1', 'user', 'duplicate', 1, 'duplicate', '2026-01-01')
                """
            )


def test_concurrent_initializers_and_cross_store_writers_share_one_database(tmp_path):
    path = tmp_path / "shared.sqlite"
    initializer_count = 8
    init_race = Barrier(initializer_count)

    def initialize(index: int) -> None:
        init_race.wait(timeout=5)
        store_type = SQLiteMemoryStore if index % 2 == 0 else SQLiteSessionStore
        store_type(path)

    with ThreadPoolExecutor(max_workers=initializer_count) as executor:
        list(executor.map(initialize, range(initializer_count)))

    memory_store = SQLiteMemoryStore(path)
    session_store = SQLiteSessionStore(path)
    session_store.create_conversation("conversation-1", provider="test", model="mock")
    writer_count = 12
    write_race = Barrier(writer_count * 2)

    def write_memory(index: int) -> None:
        write_race.wait(timeout=5)
        memory_store.save_memory(f"Concurrent durable fact {index}", dedupe=False)

    def write_message(index: int) -> None:
        write_race.wait(timeout=5)
        session_store.save_message(
            "conversation-1",
            role="user",
            content=f"message {index}",
            message_key=f"message-{index}",
        )

    with ThreadPoolExecutor(max_workers=writer_count * 2) as executor:
        futures = [executor.submit(write_memory, index) for index in range(writer_count)]
        futures.extend(executor.submit(write_message, index) for index in range(writer_count))
        for future in futures:
            future.result()

    assert len(memory_store.list_memories(limit=writer_count)) == writer_count
    assert [message.ordinal for message in session_store.list_messages("conversation-1")] == list(
        range(1, writer_count + 1)
    )
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SQLITE_SCHEMA_VERSION


def test_concurrent_partial_memory_updates_do_not_lose_unrelated_fields(monkeypatch, tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory-updates.sqlite")
    memory_id = store.save_memory("Shared memory", metadata={"initial": True}, importance=1)
    old_read_race = Barrier(2)
    original_get_memory = store.get_memory

    def synchronized_legacy_read(*args, **kwargs):
        memory = original_get_memory(*args, **kwargs)
        old_read_race.wait(timeout=5)
        return memory

    monkeypatch.setattr(store, "get_memory", synchronized_legacy_read)
    start = Barrier(2)

    def update_metadata() -> None:
        start.wait(timeout=5)
        assert store.update_memory(memory_id, metadata={"metadata": "updated"})

    def update_importance() -> None:
        start.wait(timeout=5)
        assert store.update_memory(memory_id, importance=9)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(update_metadata), executor.submit(update_importance)]
        for future in futures:
            future.result()

    memory = original_get_memory(memory_id)
    assert memory is not None
    assert memory.metadata == {"metadata": "updated"}
    assert memory.importance == 9


def test_optional_fts_is_not_part_of_the_required_schema_migration(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_store_module, "_ensure_fts", lambda _conn: False)

    store = SQLiteMemoryStore(tmp_path / "without-fts.sqlite")
    memory_id = store.save_memory("Memory still works without optional full-text search.")

    assert store.fts_enabled is False
    assert store.get_memory(memory_id) is not None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SQLITE_SCHEMA_VERSION
        assert conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE name = 'memories_fts'"
        ).fetchone() is None
