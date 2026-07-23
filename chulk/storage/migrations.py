"""Forward-only migrations for Chulk's shared SQLite database."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import sqlite3


@dataclass(frozen=True, slots=True)
class SQLiteMigration:
    """One ordered, database-wide SQLite schema migration."""

    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


def _migrate_to_shared_schema(conn: sqlite3.Connection) -> None:
    """Adopt legacy stores and create the complete shared schema."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            importance INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    _ensure_column(conn, "memories", "source", "TEXT NOT NULL DEFAULT 'manual'")
    _ensure_column(conn, "memories", "confidence", "REAL NOT NULL DEFAULT 1.0")
    _ensure_column(conn, "memories", "embedding", "TEXT")
    _ensure_column(conn, "memories", "archived_at", "TEXT")
    _ensure_column(conn, "memories", "access_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "memories", "last_accessed_at", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_archived_at ON memories(archived_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            PRIMARY KEY (memory_id, tag),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_proposals (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            importance INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            evidence TEXT,
            conversation_id TEXT,
            turn_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            accepted_memory_id TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_proposals_status_created "
        "ON memory_proposals(status, created_at)"
    )
    _backfill_memory_tags(conn)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            trace_path TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
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
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_lookup
        ON conversation_messages(conversation_id, ordinal)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_turns (
            turn_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            final_answer TEXT,
            model_request_count INTEGER NOT NULL DEFAULT 0,
            tool_call_count INTEGER NOT NULL DEFAULT 0,
            loaded_memory_ids TEXT NOT NULL DEFAULT '[]',
            loaded_skill_names TEXT NOT NULL DEFAULT '[]',
            errors TEXT NOT NULL DEFAULT '[]',
            active_plan TEXT,
            turn_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation
        ON conversation_turns(conversation_id, started_at)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_model_requests (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            turn_id TEXT,
            request_index INTEGER NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            prompt_char_count INTEGER NOT NULL DEFAULT 0,
            returned_prompt_char_count INTEGER NOT NULL DEFAULT 0,
            truncated INTEGER NOT NULL DEFAULT 0,
            loaded_memory_ids TEXT NOT NULL DEFAULT '[]',
            loaded_skill_names TEXT NOT NULL DEFAULT '[]',
            available_tool_names TEXT NOT NULL DEFAULT '[]',
            request_json TEXT NOT NULL,
            raw_response TEXT,
            usage_json TEXT,
            cost_json TEXT,
            created_at TEXT NOT NULL,
            response_created_at TEXT,
            UNIQUE (conversation_id, turn_id, request_index),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )
    _ensure_column(conn, "conversation_model_requests", "usage_json", "TEXT")
    _ensure_column(conn, "conversation_model_requests", "cost_json", "TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_tool_calls (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            resolved_tool_name TEXT,
            arguments TEXT NOT NULL DEFAULT '{}',
            iteration INTEGER NOT NULL,
            phase TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            success INTEGER,
            error TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            tool_call_json TEXT NOT NULL,
            UNIQUE (conversation_id, turn_id, phase, iteration),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_observations (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            content TEXT NOT NULL,
            output_metadata TEXT NOT NULL DEFAULT '{}',
            observation_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            content TEXT NOT NULL,
            source_message_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_summaries_lookup
        ON conversation_summaries(conversation_id, updated_at)
        """
    )


def _migrate_to_unique_message_ordinals(conn: sqlite3.Connection) -> None:
    """Normalize legacy message order and enforce one ordinal per conversation."""
    if not _table_exists(conn, "conversation_messages"):
        return

    conn.execute("DROP INDEX IF EXISTS uq_conversation_messages_ordinal")
    rows = conn.execute(
        """
        SELECT id, conversation_id
        FROM conversation_messages
        ORDER BY conversation_id, ordinal, created_at, rowid, id
        """
    ).fetchall()
    ordinals: dict[str, int] = {}
    for row in rows:
        conversation_id = str(row["conversation_id"])
        ordinal = ordinals.get(conversation_id, 0) + 1
        ordinals[conversation_id] = ordinal
        conn.execute(
            "UPDATE conversation_messages SET ordinal = ? WHERE id = ?",
            (ordinal, row["id"]),
        )
    conn.execute(
        """
        CREATE UNIQUE INDEX uq_conversation_messages_ordinal
        ON conversation_messages(conversation_id, ordinal)
        """
    )


def _migrate_to_adapter_cursors(conn: sqlite3.Connection) -> None:
    """Create durable cursors for polling-based external adapters."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adapter_cursors (
            adapter TEXT PRIMARY KEY,
            cursor INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _backfill_memory_tags(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM memory_tags")
    rows = conn.execute("SELECT id, tags FROM memories").fetchall()
    for row in rows:
        try:
            tags = json.loads(row["tags"])
        except (TypeError, json.JSONDecodeError):
            tags = []
        if not isinstance(tags, list):
            continue
        normalized: list[str] = []
        for value in tags:
            tag = str(value).strip().lower()
            if tag and tag not in normalized:
                normalized.append(tag)
        conn.executemany(
            "INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)",
            [(row["id"], tag) for tag in normalized],
        )


SQLITE_MIGRATIONS = (
    SQLiteMigration(1, "shared-memory-and-session-schema", _migrate_to_shared_schema),
    SQLiteMigration(2, "unique-message-ordinals", _migrate_to_unique_message_ordinals),
    SQLiteMigration(3, "adapter-cursors", _migrate_to_adapter_cursors),
)
SQLITE_SCHEMA_VERSION = SQLITE_MIGRATIONS[-1].version


__all__ = ["SQLITE_MIGRATIONS", "SQLITE_SCHEMA_VERSION", "SQLiteMigration"]
