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

from chulk.core.context import TurnContextSection
from chulk.core.state import ObservationRecord, Plan, PlanStep, PlanStepEvidence, ToolCallRecord, TurnState
from chulk.memory.store import select_recent_conversation_messages
from chulk.sessions.models import ConversationRecord, ConversationSummaryRecord, MessageRecord
from chulk.storage import initialize_sqlite_database, sqlite_connection


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
        """Create or migrate the shared memory and session database."""
        initialize_sqlite_database(self.db_path)

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

    def find_conversation_by_metadata(
        self,
        key: str,
        value: str | int,
    ) -> ConversationRecord | None:
        """Return the latest conversation whose metadata has an exact scalar value."""
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("Metadata key cannot be empty")
        with self._connect() as conn:
            rows = conn.execute(
                _conversation_select_sql("ORDER BY conversations.updated_at DESC")
            ).fetchall()
        for row in rows:
            record = _row_to_conversation(row)
            if record.metadata.get(clean_key) == value:
                return record
        return None

    def latest_conversation(self, *, require_turn: bool = True) -> ConversationRecord | None:
        """Return the most recently updated resumable conversation."""
        where = (
            "WHERE EXISTS (SELECT 1 FROM conversation_turns "
            "WHERE conversation_turns.conversation_id = conversations.id)"
            if require_turn
            else ""
        )
        with self._connect() as conn:
            row = conn.execute(
                _conversation_select_sql(f"{where} ORDER BY conversations.updated_at DESC LIMIT 1")
            ).fetchone()
        return _row_to_conversation(row) if row is not None else None

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
            conn.execute("BEGIN IMMEDIATE")
            _insert_message(
                conn,
                conversation_id,
                turn_id=turn_id,
                role=role,
                content=clean_content,
                message_key=key,
                metadata=metadata,
                created_at=now,
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

    def save_terminal_turn_bundle(
        self,
        conversation_id: str,
        *,
        turn_id: str,
        content: str,
        message_key: str,
        turn: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Atomically persist a terminal assistant message and its turn snapshot."""
        clean_content = content.strip()
        if (
            not clean_content
            or not message_key.strip()
            or not _valid_turn_snapshot(turn, expected_turn_id=turn_id)
        ):
            return False
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _insert_message(
                conn,
                conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=clean_content,
                message_key=message_key,
                metadata=metadata,
                created_at=now,
            )
            _save_turn_snapshot(conn, conversation_id, turn, now)
        return True

    def load_terminal_turn_message(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> dict[str, str] | None:
        """Return a terminal assistant message that may predate its turn snapshot."""
        message_keys = {
            f"{turn_id}:assistant:final": "final",
            f"{turn_id}:assistant:failed": "failed",
            f"{turn_id}:assistant:plan_rejected": "plan_rejected",
        }
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT content, message_key
                FROM conversation_messages
                WHERE conversation_id = ?
                  AND turn_id = ?
                  AND message_key IN (?, ?, ?)
                ORDER BY ordinal DESC
                LIMIT 1
                """,
                (conversation_id, turn_id, *message_keys),
            ).fetchone()
        if row is None:
            return None
        message_key = str(row["message_key"])
        kind = message_keys.get(message_key)
        if kind is None:
            return None
        return {"kind": kind, "content": str(row["content"])}

    def load_recent_messages(
        self,
        conversation_id: str,
        limit: int,
        *,
        after_ordinal: int = 0,
    ) -> list[dict[str, str]]:
        """Return recent messages in the format expected by ConversationMemory."""
        clean_limit = max(1, min(limit, 500))
        clean_after_ordinal = max(0, after_ordinal)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM conversation_messages
                WHERE conversation_id = ?
                  AND ordinal > ?
                ORDER BY ordinal
                """,
                (conversation_id, clean_after_ordinal),
            ).fetchall()
        messages = [
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in rows
            if not _message_is_prompt_excluded(row["metadata"])
        ]
        return select_recent_conversation_messages(
            messages,
            max_messages=clean_limit,
        )

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
            conn.execute("BEGIN IMMEDIATE")
            clean_source_message_count = max(0, source_message_count)
            clean_metadata = dict(metadata or {})
            clean_metadata["source_message_ordinal"] = _prompt_source_ordinal(
                conn,
                conversation_id,
                clean_source_message_count,
            )
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
                    clean_source_message_count,
                    now,
                    now,
                    json.dumps(clean_metadata, sort_keys=True),
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
        with self._connect() as conn:
            _save_turn_snapshot(conn, conversation_id, turn, now)

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
                SET raw_response = ?, usage_json = ?, cost_json = ?, response_created_at = ?
                WHERE conversation_id = ?
                  AND ((turn_id = ?) OR (turn_id IS NULL AND ? IS NULL))
                  AND request_index = ?
                """,
                (
                    payload.get("content"),
                    json.dumps(payload.get("usage"), sort_keys=True) if isinstance(payload.get("usage"), dict) else None,
                    json.dumps(payload.get("cost"), sort_keys=True) if isinstance(payload.get("cost"), dict) else None,
                    _utc_now(),
                    conversation_id,
                    turn_id,
                    turn_id,
                    request_index,
                ),
            )

    def load_uncheckpointed_hosted_mcp_requests(
        self,
        conversation_id: str,
        turn_id: str,
        *,
        checkpointed_request_count: int,
    ) -> list[dict[str, object]]:
        """Return hosted MCP requests newer than the durable turn checkpoint."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT request_index, request_json, created_at, response_created_at
                FROM conversation_model_requests
                WHERE conversation_id = ?
                  AND turn_id = ?
                  AND request_index > ?
                ORDER BY request_index
                """,
                (conversation_id, turn_id, max(0, checkpointed_request_count)),
            ).fetchall()

        requests: list[dict[str, object]] = []
        for row in rows:
            payload = _safe_json_dict(row["request_json"])
            if payload.get("hosted_mcp_enabled") is not True:
                continue
            requests.append(
                {
                    "request_index": int(row["request_index"]),
                    "server_labels": payload.get("hosted_mcp_server_labels") or [],
                    "created_at": str(row["created_at"]),
                    "response_recorded": row["response_created_at"] is not None,
                }
            )
        return requests

    def save_tool_call(self, conversation_id: str, payload: dict[str, Any]) -> None:
        """Upsert a tool-call lifecycle record and any matching intent checkpoint."""
        turn_id = str(payload.get("turn_id", "")).strip()
        iteration = int(payload.get("iteration") or 0)
        phase = str(payload.get("phase") or "execution")
        tool_name = str(payload.get("tool_name") or payload.get("resolved_tool_name") or "").strip()
        if not turn_id or not iteration or not tool_name:
            return
        turn = payload.get("turn")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
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
                    str(payload.get("started_at") or now),
                    payload.get("ended_at"),
                    _optional_bool_to_int(payload.get("success")),
                    payload.get("error"),
                    json.dumps(payload.get("metadata") or {}, sort_keys=True),
                    json.dumps(payload, sort_keys=True),
                ),
            )
            if (
                isinstance(turn, dict)
                and _valid_turn_snapshot(turn, expected_turn_id=turn_id)
            ):
                _save_turn_snapshot(conn, conversation_id, turn, now)

    def load_tool_calls_without_observations(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> list[dict[str, object]]:
        """Return persisted tool calls without a matching durable observation."""
        with self._connect() as conn:
            call_rows = conn.execute(
                """
                SELECT tool_name, arguments, iteration, phase, started_at,
                       ended_at, success
                FROM conversation_tool_calls
                WHERE conversation_id = ?
                  AND turn_id = ?
                ORDER BY iteration
                """,
                (conversation_id, turn_id),
            ).fetchall()
            observation_rows = conn.execute(
                """
                SELECT tool_name, output_metadata
                FROM conversation_observations
                WHERE conversation_id = ?
                  AND turn_id = ?
                ORDER BY observation_key
                """,
                (conversation_id, turn_id),
            ).fetchall()

        calls = [
            {
                "tool_name": str(row["tool_name"]),
                "arguments": _safe_json_dict(row["arguments"]),
                "iteration": int(row["iteration"]),
                "phase": str(row["phase"]),
                "started_at": str(row["started_at"]),
                "ended_at": row["ended_at"],
                "success": (
                    None if row["success"] is None else bool(row["success"])
                ),
            }
            for row in call_rows
        ]
        observed_identities: set[tuple[str, int]] = set()
        legacy_observation_tools: list[str] = []
        for row in observation_rows:
            metadata = _safe_json_dict(row["output_metadata"])
            if metadata.get("synthetic") is True:
                continue
            identity = metadata.get("tool_call_identity")
            if not isinstance(identity, dict):
                legacy_observation_tools.append(str(row["tool_name"]))
                continue
            phase = identity.get("phase")
            iteration = identity.get("iteration")
            if (
                isinstance(phase, str)
                and phase
                and isinstance(iteration, int)
                and not isinstance(iteration, bool)
                and iteration > 0
            ):
                observed_identities.add((phase, iteration))
            else:
                legacy_observation_tools.append(str(row["tool_name"]))

        unmatched = [
            call
            for call in calls
            if (str(call["phase"]), int(call["iteration"]))
            not in observed_identities
        ]
        for observed_tool_name in legacy_observation_tools:
            matching_index = next(
                (
                    index
                    for index, call in enumerate(unmatched)
                    if call["tool_name"] == observed_tool_name
                ),
                None,
            )
            if matching_index is not None:
                unmatched.pop(matching_index)
        return unmatched

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

    def save_tool_observation_bundle(
        self,
        conversation_id: str,
        *,
        turn_id: str,
        observation_index: int,
        tool_name: str,
        content: str,
        output_metadata: dict[str, Any] | None = None,
        action_context: str | None = None,
        turn: dict[str, Any] | None = None,
    ) -> None:
        """Atomically persist one tool action, observation, and turn checkpoint."""
        if (
            isinstance(observation_index, bool)
            or not isinstance(observation_index, int)
            or observation_index < 1
        ):
            raise ValueError("observation_index must be a positive integer")
        clean_content = content.strip()
        if not clean_content:
            return

        now = _utc_now()
        observation_key = f"{turn_id}:observation:{observation_index}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if isinstance(action_context, str) and action_context.strip():
                _insert_message(
                    conn,
                    conversation_id,
                    turn_id=turn_id,
                    role="assistant",
                    content=action_context.strip(),
                    message_key=f"{turn_id}:tool_action:{observation_index}",
                    metadata={
                        "tool_name": tool_name,
                        "internal": True,
                        "event": "tool_observation",
                        "observation_index": observation_index,
                    },
                    created_at=now,
                )
            observation_insert = conn.execute(
                """
                INSERT OR IGNORE INTO conversation_observations (
                    id, conversation_id, turn_id, tool_name, content, output_metadata,
                    observation_key, created_at
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
                    observation_key,
                    now,
                ),
            )
            _insert_message(
                conn,
                conversation_id,
                turn_id=turn_id,
                role="observation",
                content=clean_content,
                message_key=observation_key,
                metadata={
                    "tool_name": tool_name,
                    "observation_index": observation_index,
                },
                created_at=now,
            )
            if (
                observation_insert.rowcount == 1
                and isinstance(turn, dict)
                and _valid_turn_snapshot(turn, expected_turn_id=turn_id)
            ):
                _save_turn_snapshot(conn, conversation_id, turn, now)
            else:
                _touch_conversation(conn, conversation_id, now)

    def max_observation_index(self, conversation_id: str, turn_id: str) -> int:
        """Return the largest recorder sequence already stored for one turn."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT observation_key
                FROM conversation_observations
                WHERE conversation_id = ? AND turn_id = ?
                """,
                (conversation_id, turn_id),
            ).fetchall()
        indexes = []
        for row in rows:
            suffix = str(row["observation_key"] or "").rsplit(":", 1)[-1]
            if suffix.isdigit():
                indexes.append(int(suffix))
        return max(indexes, default=0)

    def update_conversation_status(self, conversation_id: str, status: str) -> None:
        """Update only the conversation status and timestamp."""
        with self._connect() as conn:
            _set_conversation_status(conn, conversation_id, status, _utc_now())

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with sqlite_connection(self.db_path) as conn:
            yield conn


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
        context_sections=[_context_section_from_dict(item) for item in _safe_dict_list(payload.get("context_sections"))],
        prompt_profile=payload.get("prompt_profile"),
        locale=payload.get("locale"),
        extension_metadata=_safe_json_object(payload.get("extension_metadata")),
        tool_context_metadata=_safe_json_object(payload.get("tool_context_metadata")),
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
        model_usage_reports=_safe_dict_list(payload.get("model_usage_reports")),
        model_usage_totals=_safe_json_dict(payload.get("model_usage_totals")),
        plan_execution_feedback_count=int(payload.get("plan_execution_feedback_count") or 0),
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
                depends_on=_safe_string_list(item.get("depends_on")),
                acceptance_criteria=_safe_string_list(item.get("acceptance_criteria")),
                retry_limit=int(item.get("retry_limit") or 0),
                evidence=[_plan_step_evidence_from_dict(record) for record in _safe_dict_list(item.get("evidence"))],
                started_at=item.get("started_at"),
                completed_at=item.get("completed_at"),
                blocked_at=item.get("blocked_at"),
                blocked_reason=item.get("blocked_reason"),
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


def _context_section_from_dict(payload: dict[str, Any]) -> TurnContextSection:
    return TurnContextSection(
        id=str(payload.get("id") or ""),
        title=payload.get("title"),
        source=payload.get("source"),
        content=str(payload.get("content") or ""),
        metadata=_safe_json_object(payload.get("metadata")),
    )


def _tool_call_from_dict(payload: dict[str, Any]) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=str(payload.get("tool_name") or ""),
        arguments=_safe_json_object(payload.get("arguments")),
        iteration=int(payload.get("iteration") or 0),
        phase=str(payload.get("phase") or "execution"),
        plan_step_id=payload.get("plan_step_id"),
        started_at=str(payload.get("started_at") or _utc_now()),
        ended_at=payload.get("ended_at"),
        resolved_tool_name=payload.get("resolved_tool_name"),
        success=payload.get("success"),
        error=payload.get("error"),
        failure_kind=payload.get("failure_kind"),
        metadata=_safe_json_object(payload.get("metadata")),
    )


def _plan_step_evidence_from_dict(payload: dict[str, Any]) -> PlanStepEvidence:
    return PlanStepEvidence(
        content=str(payload.get("content") or ""),
        tool_name=payload.get("tool_name"),
        tool_call_iteration=payload.get("tool_call_iteration"),
        created_at=str(payload.get("created_at") or _utc_now()),
        metadata=_safe_json_object(payload.get("metadata")),
    )


def _observation_from_dict(payload: dict[str, Any]) -> ObservationRecord:
    return ObservationRecord(
        tool_name=str(payload.get("tool_name") or ""),
        content=str(payload.get("content") or ""),
        output_metadata=_safe_json_object(payload.get("output_metadata")),
        created_at=str(payload.get("created_at") or _utc_now()),
    )


def _insert_message(
    conn: sqlite3.Connection,
    conversation_id: str,
    *,
    turn_id: str | None,
    role: str,
    content: str,
    message_key: str,
    metadata: dict[str, Any] | None,
    created_at: str,
) -> None:
    """Insert one idempotent message using the caller's transaction."""
    next_ordinal = _next_message_ordinal(conn, conversation_id)
    conn.execute(
        """
        INSERT INTO conversation_messages (
            id, conversation_id, turn_id, role, content, ordinal, message_key, created_at, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_key) DO NOTHING
        """,
        (
            str(uuid4()),
            conversation_id,
            turn_id,
            role,
            content,
            next_ordinal,
            message_key,
            created_at,
            json.dumps(metadata or {}, sort_keys=True),
        ),
    )


def _valid_turn_snapshot(turn: dict[str, Any], *, expected_turn_id: str) -> bool:
    return (
        str(turn.get("turn_id", "")).strip() == expected_turn_id
        and bool(str(turn.get("user_message", "")).strip())
    )


def _save_turn_snapshot(
    conn: sqlite3.Connection,
    conversation_id: str,
    turn: dict[str, Any],
    updated_at: str,
) -> None:
    """Upsert one turn snapshot using the caller's transaction."""
    turn_id = str(turn.get("turn_id", "")).strip()
    user_message = str(turn.get("user_message", "")).strip()
    if not turn_id or not user_message:
        return
    active_plan = turn.get("active_plan")
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
            str(turn.get("started_at") or updated_at),
            turn.get("ended_at"),
            turn.get("final_answer"),
            int(turn.get("model_request_count") or 0),
            int(turn.get("tool_call_count") or 0),
            json.dumps(turn.get("loaded_memory_ids") or [], sort_keys=True),
            json.dumps(turn.get("loaded_skill_names") or [], sort_keys=True),
            json.dumps(turn.get("errors") or [], sort_keys=True),
            json.dumps(active_plan, sort_keys=True) if active_plan else None,
            json.dumps(turn, sort_keys=True),
            updated_at,
        ),
    )
    _set_conversation_status(
        conn,
        conversation_id,
        _conversation_status_from_turn(turn),
        updated_at,
    )


def _next_message_ordinal(conn: sqlite3.Connection, conversation_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(ordinal), 0) + 1 AS next_ordinal FROM conversation_messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    return int(row["next_ordinal"])


def _prompt_source_ordinal(
    conn: sqlite3.Connection,
    conversation_id: str,
    source_message_count: int,
) -> int:
    """Map a logical prompt-history count to its durable message ordinal."""
    if source_message_count <= 0:
        return 0
    rows = conn.execute(
        """
        SELECT ordinal, metadata
        FROM conversation_messages
        WHERE conversation_id = ?
        ORDER BY ordinal
        """,
        (conversation_id,),
    ).fetchall()
    prompt_ordinals = [
        int(row["ordinal"])
        for row in rows
        if not _message_is_prompt_excluded(row["metadata"])
    ]
    if not prompt_ordinals:
        return 0
    index = min(source_message_count, len(prompt_ordinals)) - 1
    return prompt_ordinals[index]


def _message_is_prompt_excluded(metadata: Any) -> bool:
    return _safe_json_dict(metadata).get("prompt_excluded") is True


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
    if status == "cancelled":
        return "cancelled"
    if status == "blocked":
        return "blocked"
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
