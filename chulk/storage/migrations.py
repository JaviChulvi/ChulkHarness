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


def _migrate_to_scheduled_jobs(conn: sqlite3.Connection) -> None:
    """Create durable, destination-scoped scheduled jobs."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_jobs (
            id TEXT PRIMARY KEY,
            adapter TEXT NOT NULL,
            destination_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            next_run_at TEXT NOT NULL,
            interval_seconds INTEGER,
            status TEXT NOT NULL DEFAULT 'active',
            lease_until TEXT,
            last_run_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (interval_seconds IS NULL OR interval_seconds > 0),
            CHECK (status IN ('active', 'running', 'paused', 'completed', 'cancelled'))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_due
        ON scheduled_jobs(status, next_run_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_destination
        ON scheduled_jobs(adapter, destination_id, created_at)
        """
    )


def _migrate_to_claim_owned_scheduled_jobs(conn: sqlite3.Connection) -> None:
    """Add claim identity and a stable recurrence anchor to scheduled jobs."""
    _ensure_column(conn, "scheduled_jobs", "claim_token", "TEXT")
    _ensure_column(conn, "scheduled_jobs", "scheduled_for", "TEXT")
    conn.execute(
        """
        UPDATE scheduled_jobs
        SET scheduled_for = next_run_at
        WHERE scheduled_for IS NULL
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_claim
        ON scheduled_jobs(id, claim_token)
        """
    )


def _migrate_to_adapter_update_ledger(conn: sqlite3.Connection) -> None:
    """Create durable execution and response-delivery state for adapter updates."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adapter_updates (
            adapter TEXT NOT NULL,
            update_id INTEGER NOT NULL,
            destination_id TEXT NOT NULL,
            status TEXT NOT NULL,
            response_parts TEXT NOT NULL DEFAULT '[]',
            next_response_part INTEGER NOT NULL DEFAULT 0,
            execution_token TEXT,
            execution_lease_until TEXT,
            delivery_token TEXT,
            delivery_lease_until TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            executed_at TEXT,
            delivered_at TEXT,
            PRIMARY KEY (adapter, update_id),
            CHECK (update_id >= 0),
            CHECK (next_response_part >= 0),
            CHECK (
                status IN (
                    'processing', 'executed', 'delivering',
                    'delivered', 'ignored'
                )
            )
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_adapter_updates_outbox
        ON adapter_updates(adapter, status, updated_at)
        """
    )


def _migrate_to_memory_namespaces(conn: sqlite3.Connection) -> None:
    """Backfill the compatibility namespace for memories and proposals."""
    _ensure_column(conn, "memories", "namespace", "TEXT NOT NULL DEFAULT 'default'")
    _ensure_column(
        conn,
        "memory_proposals",
        "namespace",
        "TEXT NOT NULL DEFAULT 'default'",
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memories_namespace_active
        ON memories(namespace, archived_at, updated_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_proposals_namespace_status
        ON memory_proposals(namespace, status, created_at)
        """
    )


def _migrate_to_usage_ledger(conn: sqlite3.Connection) -> None:
    """Create the immutable metering ledger and crash-recoverable reservations."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_ledger (
            id TEXT PRIMARY KEY,
            resource_kind TEXT NOT NULL,
            source_event_id TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            tool_or_service TEXT,
            credential_ref TEXT,
            model_profile_id TEXT,
            profile_id TEXT NOT NULL,
            channel TEXT,
            conversation_id TEXT,
            turn_id TEXT,
            goal_id TEXT,
            job_id TEXT,
            child_task_id TEXT,
            purpose TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            billing_period TEXT NOT NULL,
            units_json TEXT NOT NULL DEFAULT '{}',
            model_calls INTEGER NOT NULL DEFAULT 0,
            tool_calls INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            cost_amount TEXT,
            currency TEXT NOT NULL,
            pricing_known INTEGER NOT NULL DEFAULT 0,
            cost_estimated INTEGER NOT NULL DEFAULT 0,
            cost_reported INTEGER NOT NULL DEFAULT 0,
            usage_estimated INTEGER NOT NULL DEFAULT 0,
            trace_path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE (resource_kind, source_event_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_usage_ledger_time
        ON usage_ledger(profile_id, occurred_at, id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_usage_ledger_dimensions
        ON usage_ledger(
            profile_id, conversation_id, turn_id, goal_id, job_id, child_task_id
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_reservations (
            id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            source_event_id TEXT NOT NULL,
            resource_kind TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            channel TEXT,
            conversation_id TEXT,
            turn_id TEXT,
            goal_id TEXT,
            job_id TEXT,
            child_task_id TEXT,
            budget_scope TEXT NOT NULL,
            budget_json TEXT NOT NULL,
            state TEXT NOT NULL,
            reserved_model_calls INTEGER NOT NULL DEFAULT 0,
            reserved_tool_calls INTEGER NOT NULL DEFAULT 0,
            reserved_tokens INTEGER NOT NULL DEFAULT 0,
            reserved_cost_amount TEXT,
            currency TEXT NOT NULL,
            pricing_known INTEGER NOT NULL DEFAULT 0,
            unknown_cost INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            ledger_entry_ids_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_usage_reservations_active_scope
        ON usage_reservations(
            state, profile_id, conversation_id, turn_id, goal_id, job_id,
            child_task_id, expires_at
        )
        """
    )


def _migrate_to_usage_recovery_checkpoints(conn: sqlite3.Connection) -> None:
    """Attach a private accounting checkpoint to persisted model requests."""
    _ensure_column(conn, "conversation_model_requests", "accounting_json", "TEXT")


def _migrate_to_session_search(conn: sqlite3.Connection) -> None:
    """Create the optional cross-session message index when FTS5 is available."""
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_search_window
        ON conversation_messages(conversation_id, ordinal, role)
        """
    )
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts
            USING fts5(
                message_id UNINDEXED,
                conversation_id UNINDEXED,
                role UNINDEXED,
                content
            )
            """
        )
    except sqlite3.OperationalError:
        # FTS5 is optional. SQLiteSessionStore activates the bounded scan
        # fallback and can rebuild the index when FTS later becomes available.
        return


def _migrate_to_skill_lifecycle(conn: sqlite3.Connection) -> None:
    """Create governed skill revisions, usage, and learning proposals."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_packages (
            profile_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            digest TEXT NOT NULL,
            source TEXT NOT NULL,
            trust TEXT NOT NULL,
            status TEXT NOT NULL,
            active_revision_id TEXT NOT NULL,
            view_count INTEGER NOT NULL DEFAULT 0,
            use_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            patch_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (profile_id, scope, name),
            CHECK (scope IN ('project', 'profile')),
            CHECK (status IN ('active', 'stale', 'archived', 'pinned')),
            CHECK (
                view_count >= 0 AND use_count >= 0
                AND success_count >= 0 AND patch_count >= 0
            )
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_package_revisions (
            id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            digest TEXT NOT NULL,
            source TEXT NOT NULL,
            trust TEXT NOT NULL,
            package_json TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            proposal_id TEXT,
            created_at TEXT NOT NULL,
            UNIQUE (profile_id, scope, name, digest),
            CHECK (scope IN ('project', 'profile'))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_skill_revisions_lookup
        ON skill_package_revisions(profile_id, scope, name, created_at)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_usage_events (
            id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            skill_version TEXT NOT NULL,
            skill_digest TEXT NOT NULL,
            kind TEXT NOT NULL,
            source_event_id TEXT NOT NULL,
            host_confirmed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE (profile_id, scope, skill_name, source_event_id, kind),
            CHECK (scope IN ('project', 'profile')),
            CHECK (kind IN ('view', 'use', 'success', 'patch')),
            CHECK (host_confirmed IN (0, 1))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learning_proposals (
            id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            target_name TEXT,
            rationale TEXT NOT NULL,
            evidence_turn_ids_json TEXT NOT NULL DEFAULT '[]',
            source_trace TEXT,
            content TEXT,
            diff TEXT,
            required_capabilities_json TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL,
            verification_steps_json TEXT NOT NULL DEFAULT '[]',
            reviewer_model TEXT,
            cost TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewed_by TEXT,
            applied_revision_id TEXT,
            accepted_memory_id TEXT,
            error TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            CHECK (
                kind IN (
                    'memory_create', 'memory_update', 'skill_create',
                    'skill_patch', 'skill_archive'
                )
            ),
            CHECK (status IN ('pending', 'approved', 'rejected', 'failed')),
            CHECK (confidence >= 0.0 AND confidence <= 1.0)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_learning_proposals_profile_status
        ON learning_proposals(profile_id, status, created_at)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learning_review_runs (
            id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            trigger TEXT NOT NULL,
            reviewer_model TEXT,
            status TEXT NOT NULL,
            proposal_count INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            cost_amount TEXT NOT NULL,
            currency TEXT NOT NULL DEFAULT 'USD',
            created_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT,
            CHECK (status IN ('reserved', 'completed', 'failed')),
            CHECK (proposal_count >= 0),
            CHECK (token_count >= 0)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_learning_review_runs_profile_created
        ON learning_review_runs(profile_id, created_at)
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
    SQLiteMigration(4, "scheduled-jobs", _migrate_to_scheduled_jobs),
    SQLiteMigration(5, "claim-owned-scheduled-jobs", _migrate_to_claim_owned_scheduled_jobs),
    SQLiteMigration(6, "adapter-update-ledger", _migrate_to_adapter_update_ledger),
    SQLiteMigration(7, "memory-namespaces", _migrate_to_memory_namespaces),
    SQLiteMigration(8, "usage-ledger-and-reservations", _migrate_to_usage_ledger),
    SQLiteMigration(
        9,
        "usage-recovery-checkpoints",
        _migrate_to_usage_recovery_checkpoints,
    ),
    SQLiteMigration(10, "profile-session-search", _migrate_to_session_search),
    SQLiteMigration(11, "skill-lifecycle-and-learning", _migrate_to_skill_lifecycle),
)
SQLITE_SCHEMA_VERSION = SQLITE_MIGRATIONS[-1].version


__all__ = ["SQLITE_MIGRATIONS", "SQLITE_SCHEMA_VERSION", "SQLiteMigration"]
