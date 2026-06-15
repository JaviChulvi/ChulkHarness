"""SQLite store for durable agent conversations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.core.state import ObservationRecord, Plan, PlanStep, ToolCallRecord, TurnState
from chulk.sessions.models import ConversationRecord, ConversationSummaryRecord, MessageRecord


class SessionNotFoundError(ValueError):
    """Raised when a requested conversation cannot be found."""


class AmbiguousSessionError(ValueError):
    """Raised when a conversation id prefix matches multiple sessions."""


class SQLiteSessionStore:
    """Durable store for conversations, turns, messages, and tool observations."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.initialize()

    def initialize(self) -> None:
        """Create session tables if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
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
                    created_at TEXT NOT NULL,
                    response_created_at TEXT,
                    UNIQUE (conversation_id, turn_id, request_index),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
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

    def create_conversation(
        self,
        conversation_id: str,
        *,
        provider: str,
        model: str,
        trace_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationRecord:
        """Create or refresh a conversation row."""
        now = _utc_now()
        clean_metadata = metadata or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, provider, model, trace_path, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    provider = excluded.provider,
                    model = excluded.model,
                    trace_path = excluded.trace_path,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    conversation_id,
                    provider,
                    model,
                    trace_path,
                    now,
                    now,
                    json.dumps(clean_metadata, sort_keys=True),
                ),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id_or_prefix: str) -> ConversationRecord:
        """Return a conversation by full id or unique prefix."""
        clean_id = conversation_id_or_prefix.strip()
        if not clean_id:
            raise SessionNotFoundError("Conversation id cannot be empty")

        with self._connect() as conn:
            exact = conn.execute(
                _conversation_select_sql("WHERE conversations.id = ?"),
                (clean_id,),
            ).fetchone()
            if exact is not None:
                return _row_to_conversation(exact)

            rows = conn.execute(
                _conversation_select_sql("WHERE conversations.id LIKE ? ORDER BY conversations.updated_at DESC"),
                (f"{clean_id}%",),
            ).fetchall()

        if not rows:
            raise SessionNotFoundError(f"No session found for id: {clean_id}")
        if len(rows) > 1:
            matches = ", ".join(row["id"][:8] for row in rows[:5])
            raise AmbiguousSessionError(f"Session id prefix is ambiguous: {clean_id} matches {matches}")
        return _row_to_conversation(rows[0])

    def list_conversations(self, limit: int = 20) -> list[ConversationRecord]:
        """Return recently updated conversations."""
        clean_limit = max(1, min(limit, 100))
        with self._connect() as conn:
            rows = conn.execute(
                _conversation_select_sql("ORDER BY conversations.updated_at DESC LIMIT ?"),
                (clean_limit,),
            ).fetchall()
        return [_row_to_conversation(row) for row in rows]

    def save_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        turn_id: str | None = None,
        message_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        """Persist a short-term conversation message if it has not been recorded already."""
        clean_content = content.strip()
        if not clean_content:
            return

        now = created_at or _utc_now()
        key = message_key or f"{conversation_id}:{uuid4()}"
        with self._connect() as conn:
            next_ordinal = _next_message_ordinal(conn, conversation_id)
            conn.execute(
                """
                INSERT OR IGNORE INTO conversation_messages (
                    id, conversation_id, turn_id, role, content, ordinal, message_key, created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    conversation_id,
                    turn_id,
                    role,
                    clean_content,
                    next_ordinal,
                    key,
                    now,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            _touch_conversation(conn, conversation_id, now)

    def list_messages(
        self,
        conversation_id: str,
        *,
        limit: int = 50,
        after_ordinal: int = 0,
    ) -> list[MessageRecord]:
        """Return persisted messages for a conversation in display order."""
        clean_limit = max(1, min(limit, 500))
        clean_after_ordinal = max(0, after_ordinal)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conversation_messages
                WHERE conversation_id = ?
                  AND ordinal > ?
                ORDER BY ordinal DESC
                LIMIT ?
                """,
                (conversation_id, clean_after_ordinal, clean_limit),
            ).fetchall()
        return [_row_to_message(row) for row in reversed(rows)]

    def load_recent_messages(
        self,
        conversation_id: str,
        limit: int,
        *,
        after_ordinal: int = 0,
    ) -> list[dict[str, str]]:
        """Return recent messages in the format expected by ConversationMemory."""
        return [
            {"role": message.role, "content": message.content}
            for message in self.list_messages(conversation_id, limit=limit, after_ordinal=after_ordinal)
        ]

    def save_conversation_summary(
        self,
        conversation_id: str,
        *,
        content: str,
        source_message_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist the latest compact summary for a conversation."""
        clean_content = content.strip()
        if not clean_content:
            return
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_summaries (
                    id, conversation_id, content, source_message_count, created_at, updated_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    conversation_id,
                    clean_content,
                    max(0, source_message_count),
                    now,
                    now,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            _touch_conversation(conn, conversation_id, now)

    def load_latest_summary(self, conversation_id: str) -> ConversationSummaryRecord | None:
        """Return the latest compact summary for a conversation, if any."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM conversation_summaries
                WHERE conversation_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        return _row_to_summary(row) if row is not None else None

    def save_turn_snapshot(self, conversation_id: str, turn: dict[str, Any]) -> None:
        """Upsert the latest inspectable turn snapshot."""
        turn_id = str(turn.get("turn_id", "")).strip()
        user_message = str(turn.get("user_message", "")).strip()
        if not turn_id or not user_message:
            return

        now = _utc_now()
        active_plan = turn.get("active_plan")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_turns (
                    turn_id, conversation_id, user_message, status, started_at, ended_at, final_answer,
                    model_request_count, tool_call_count, loaded_memory_ids, loaded_skill_names,
                    errors, active_plan, turn_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(turn_id) DO UPDATE SET
                    user_message = excluded.user_message,
                    status = excluded.status,
                    ended_at = excluded.ended_at,
                    final_answer = excluded.final_answer,
                    model_request_count = excluded.model_request_count,
                    tool_call_count = excluded.tool_call_count,
                    loaded_memory_ids = excluded.loaded_memory_ids,
                    loaded_skill_names = excluded.loaded_skill_names,
                    errors = excluded.errors,
                    active_plan = excluded.active_plan,
                    turn_json = excluded.turn_json,
                    updated_at = excluded.updated_at
                """,
                (
                    turn_id,
                    conversation_id,
                    user_message,
                    str(turn.get("status", "unknown")),
                    str(turn.get("started_at") or now),
                    turn.get("ended_at"),
                    turn.get("final_answer"),
                    int(turn.get("model_request_count") or 0),
                    int(turn.get("tool_call_count") or 0),
                    json.dumps(turn.get("loaded_memory_ids") or [], sort_keys=True),
                    json.dumps(turn.get("loaded_skill_names") or [], sort_keys=True),
                    json.dumps(turn.get("errors") or [], sort_keys=True),
                    json.dumps(active_plan, sort_keys=True) if active_plan else None,
                    json.dumps(turn, sort_keys=True),
                    now,
                ),
            )
            _set_conversation_status(conn, conversation_id, _conversation_status_from_turn(turn), now)

    def load_turns(self, conversation_id: str) -> list[TurnState]:
        """Load persisted turn snapshots as runtime TurnState objects."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT turn_json
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY started_at, updated_at
                """,
                (conversation_id,),
            ).fetchall()
        turns = []
        for row in rows:
            payload = _safe_json_dict(row["turn_json"])
            if payload:
                turns.append(_turn_from_dict(payload))
        return turns

    def save_model_request(self, conversation_id: str, payload: dict[str, Any]) -> None:
        """Persist one model request trace payload."""
        turn_id = payload.get("turn_id")
        request_index = int(payload.get("request_index") or 0)
        if request_index < 1:
            return
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_model_requests (
                    id, conversation_id, turn_id, request_index, message_count, prompt_char_count,
                    returned_prompt_char_count, truncated, loaded_memory_ids, loaded_skill_names,
                    available_tool_names, request_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id, turn_id, request_index) DO UPDATE SET
                    message_count = excluded.message_count,
                    prompt_char_count = excluded.prompt_char_count,
                    returned_prompt_char_count = excluded.returned_prompt_char_count,
                    truncated = excluded.truncated,
                    loaded_memory_ids = excluded.loaded_memory_ids,
                    loaded_skill_names = excluded.loaded_skill_names,
                    available_tool_names = excluded.available_tool_names,
                    request_json = excluded.request_json
                """,
                (
                    str(uuid4()),
                    conversation_id,
                    turn_id,
                    request_index,
                    int(payload.get("message_count") or 0),
                    int(payload.get("prompt_char_count") or 0),
                    int(payload.get("returned_prompt_char_count") or 0),
                    1 if payload.get("truncated") else 0,
                    json.dumps(payload.get("loaded_memory_ids") or [], sort_keys=True),
                    json.dumps(payload.get("loaded_skill_names") or [], sort_keys=True),
                    json.dumps(payload.get("available_tool_names") or [], sort_keys=True),
                    json.dumps(payload, sort_keys=True),
                    now,
                ),
            )

    def save_model_response(self, conversation_id: str, payload: dict[str, Any]) -> None:
        """Attach a raw model response to the matching request when possible."""
        turn_id = payload.get("turn_id")
        request_index = int(payload.get("request_index") or 0)
        if request_index < 1:
            return
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversation_model_requests
                SET raw_response = ?, response_created_at = ?
                WHERE conversation_id = ?
                  AND ((turn_id = ?) OR (turn_id IS NULL AND ? IS NULL))
                  AND request_index = ?
                """,
                (payload.get("content"), _utc_now(), conversation_id, turn_id, turn_id, request_index),
            )

    def save_tool_call(self, conversation_id: str, payload: dict[str, Any]) -> None:
        """Upsert a tool-call lifecycle record."""
        turn_id = str(payload.get("turn_id", "")).strip()
        iteration = int(payload.get("iteration") or 0)
        phase = str(payload.get("phase") or "execution")
        tool_name = str(payload.get("tool_name") or payload.get("resolved_tool_name") or "").strip()
        if not turn_id or not iteration or not tool_name:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_tool_calls (
                    id, conversation_id, turn_id, tool_name, resolved_tool_name, arguments, iteration,
                    phase, started_at, ended_at, success, error, metadata, tool_call_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id, turn_id, phase, iteration) DO UPDATE SET
                    tool_name = excluded.tool_name,
                    resolved_tool_name = excluded.resolved_tool_name,
                    arguments = excluded.arguments,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    success = excluded.success,
                    error = excluded.error,
                    metadata = excluded.metadata,
                    tool_call_json = excluded.tool_call_json
                """,
                (
                    str(uuid4()),
                    conversation_id,
                    turn_id,
                    tool_name,
                    payload.get("resolved_tool_name"),
                    json.dumps(payload.get("arguments") or {}, sort_keys=True),
                    iteration,
                    phase,
                    str(payload.get("started_at") or _utc_now()),
                    payload.get("ended_at"),
                    _optional_bool_to_int(payload.get("success")),
                    payload.get("error"),
                    json.dumps(payload.get("metadata") or {}, sort_keys=True),
                    json.dumps(payload, sort_keys=True),
                ),
            )

    def save_observation(
        self,
        conversation_id: str,
        *,
        turn_id: str,
        tool_name: str,
        content: str,
        output_metadata: dict[str, Any] | None = None,
        observation_key: str | None = None,
    ) -> None:
        """Persist one tool observation."""
        clean_content = content.strip()
        if not clean_content:
            return
        key = observation_key or f"{conversation_id}:{turn_id}:observation:{uuid4()}"
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO conversation_observations (
                    id, conversation_id, turn_id, tool_name, content, output_metadata, observation_key, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    conversation_id,
                    turn_id,
                    tool_name,
                    clean_content,
                    json.dumps(output_metadata or {}, sort_keys=True),
                    key,
                    now,
                ),
            )

    def update_conversation_status(self, conversation_id: str, status: str) -> None:
        """Update only the conversation status and timestamp."""
        with self._connect() as conn:
            _set_conversation_status(conn, conversation_id, status, _utc_now())

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _conversation_select_sql(suffix: str) -> str:
    return f"""
        SELECT conversations.*,
               COALESCE(turn_counts.turn_count, 0) AS turn_count
        FROM conversations
        LEFT JOIN (
            SELECT conversation_id, count(*) AS turn_count
            FROM conversation_turns
            GROUP BY conversation_id
        ) AS turn_counts ON turn_counts.conversation_id = conversations.id
        {suffix}
    """


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        id=row["id"],
        title=row["title"],
        provider=row["provider"],
        model=row["model"],
        trace_path=row["trace_path"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=_safe_json_dict(row["metadata"]),
        turn_count=int(row["turn_count"] or 0),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        id=row["id"],
        conversation_id=row["conversation_id"],
        turn_id=row["turn_id"],
        role=row["role"],
        content=row["content"],
        ordinal=int(row["ordinal"]),
        created_at=row["created_at"],
        metadata=_safe_json_dict(row["metadata"]),
    )


def _row_to_summary(row: sqlite3.Row) -> ConversationSummaryRecord:
    return ConversationSummaryRecord(
        id=row["id"],
        conversation_id=row["conversation_id"],
        content=row["content"],
        source_message_count=int(row["source_message_count"] or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=_safe_json_dict(row["metadata"]),
    )


def _turn_from_dict(payload: dict[str, Any]) -> TurnState:
    turn = TurnState(
        user_message=str(payload.get("user_message") or ""),
        turn_id=str(payload.get("turn_id") or uuid4()),
        started_at=str(payload.get("started_at") or _utc_now()),
        ended_at=payload.get("ended_at"),
        status=str(payload.get("status") or "completed"),
        model_request_count=int(payload.get("model_request_count") or 0),
        tool_call_count=int(payload.get("tool_call_count") or 0),
        available_tool_names=_safe_string_list(payload.get("available_tool_names")),
        loaded_memory_ids=_safe_string_list(payload.get("loaded_memory_ids")),
        extracted_memory_ids=_safe_string_list(payload.get("extracted_memory_ids")),
        loaded_skill_names=_safe_string_list(payload.get("loaded_skill_names")),
        errors=_safe_string_list(payload.get("errors")),
        final_answer=payload.get("final_answer"),
        active_plan=_plan_from_dict(payload.get("active_plan")),
        plan_approved=bool(payload.get("plan_approved")),
        planning_feedback_count=int(payload.get("planning_feedback_count") or 0),
        planning_tool_limit_feedback_sent=bool(payload.get("planning_tool_limit_feedback_sent")),
        reflection_count=int(payload.get("reflection_count") or 0),
        reflections=_safe_dict_list(payload.get("reflections")),
        context_reports=_safe_dict_list(payload.get("context_reports")),
    )
    turn.tool_calls = [_tool_call_from_dict(item) for item in _safe_dict_list(payload.get("tool_calls"))]
    turn.observations = [_observation_from_dict(item) for item in _safe_dict_list(payload.get("observations"))]
    return turn


def _plan_from_dict(payload: Any) -> Plan | None:
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    steps = []
    for index, item in enumerate(_safe_dict_list(payload.get("steps")), start=1):
        title = item.get("title")
        description = item.get("description")
        if not isinstance(title, str) or not isinstance(description, str):
            continue
        steps.append(
            PlanStep(
                id=str(item.get("id") or index),
                title=title,
                description=description,
                status=str(item.get("status") or "pending"),
            )
        )
    if not steps:
        return None
    plan = Plan(
        summary=summary,
        steps=steps,
        created_at=str(payload.get("created_at") or _utc_now()),
        approved_at=payload.get("approved_at"),
        rejected_at=payload.get("rejected_at"),
    )
    return plan


def _tool_call_from_dict(payload: dict[str, Any]) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=str(payload.get("tool_name") or ""),
        arguments=_safe_json_object(payload.get("arguments")),
        iteration=int(payload.get("iteration") or 0),
        phase=str(payload.get("phase") or "execution"),
        started_at=str(payload.get("started_at") or _utc_now()),
        ended_at=payload.get("ended_at"),
        resolved_tool_name=payload.get("resolved_tool_name"),
        success=payload.get("success"),
        error=payload.get("error"),
        metadata=_safe_json_object(payload.get("metadata")),
    )


def _observation_from_dict(payload: dict[str, Any]) -> ObservationRecord:
    return ObservationRecord(
        tool_name=str(payload.get("tool_name") or ""),
        content=str(payload.get("content") or ""),
        output_metadata=_safe_json_object(payload.get("output_metadata")),
        created_at=str(payload.get("created_at") or _utc_now()),
    )


def _next_message_ordinal(conn: sqlite3.Connection, conversation_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(ordinal), 0) + 1 AS next_ordinal FROM conversation_messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    return int(row["next_ordinal"])


def _touch_conversation(conn: sqlite3.Connection, conversation_id: str, updated_at: str) -> None:
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (updated_at, conversation_id))
    row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    if row is not None and not row["title"]:
        first_message = conn.execute(
            """
            SELECT content
            FROM conversation_messages
            WHERE conversation_id = ? AND role = 'user'
            ORDER BY ordinal
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()
        if first_message is not None:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (_title_from_message(first_message["content"]), conversation_id),
            )


def _set_conversation_status(conn: sqlite3.Connection, conversation_id: str, status: str, updated_at: str) -> None:
    conn.execute(
        "UPDATE conversations SET status = ?, updated_at = ? WHERE id = ?",
        (status, updated_at, conversation_id),
    )


def _conversation_status_from_turn(turn: dict[str, Any]) -> str:
    status = str(turn.get("status") or "active")
    if status == "waiting_for_approval":
        return "waiting_for_approval"
    if status == "failed":
        return "failed"
    if status == "plan_rejected":
        return "plan_rejected"
    if status == "completed":
        return "completed"
    return "active"


def _title_from_message(content: str, limit: int = 72) -> str:
    one_line = " ".join(content.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3].rstrip() + "..."


def _optional_bool_to_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _safe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_json_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
