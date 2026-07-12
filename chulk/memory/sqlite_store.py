"""SQLite-backed long-term memory store."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from chulk.memory.constants import PROFILE_MEMORY_TAGS
from chulk.memory.extraction import extract_memory_candidates
from chulk.memory.markdown import parse_markdown_memory_line as _parse_markdown_memory_line
from chulk.memory.models import MemoryExtractionCandidate, MemoryProposalRecord, MemoryRecord
from chulk.memory.security import ensure_memory_payload_safe
from chulk.memory.retrieval import (
    choose_memory_to_keep as _choose_memory_to_keep,
    content_similarity as _content_similarity,
    cosine_similarity as _cosine_similarity,
    merge_ranked_memories as _merge_ranked_memories,
    merge_tags as _merge_tags,
    normalize_confidence as _normalize_confidence,
    normalize_content as _normalize_content,
    normalize_embedding as _normalize_embedding,
    normalize_importance as _normalize_importance,
    normalize_limit as _normalize_limit,
    normalize_source as _normalize_source,
    normalize_tags as _normalize_tags,
    resolve_profile_conflicts as _resolve_profile_conflicts,
    safe_json_dict as _safe_json_dict,
    safe_json_float_list as _safe_json_float_list,
    safe_json_list as _safe_json_list,
    score_memory as _score_memory,
    text_to_embedding,
    tokenize as _tokenize,
)


class SQLiteMemoryStore:
    """Small SQLite store for durable user, project, and preference memories."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.fts_enabled = False
        self.initialize()

    def initialize(self) -> None:
        """Create the memory database and schema if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
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
            self.fts_enabled = _ensure_fts(conn)
            if self.fts_enabled:
                _backfill_memory_fts(conn)

    def save_memory(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: int = 1,
        source: str = "manual",
        confidence: float = 1.0,
        embedding: list[float] | None = None,
        dedupe: bool = True,
    ) -> str:
        """Save a new long-term memory and return its id."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty")

        clean_tags = _normalize_tags(tags or [])
        clean_metadata = metadata or {}
        clean_importance = _normalize_importance(importance)
        clean_source = _normalize_source(source)
        clean_confidence = _normalize_confidence(confidence)
        ensure_memory_payload_safe(
            content=clean_content,
            tags=clean_tags,
            metadata=clean_metadata,
            source=clean_source,
        )
        clean_embedding = _normalize_embedding(embedding) or text_to_embedding(clean_content)

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._save_memory_in_connection(
                conn,
                content=clean_content,
                tags=clean_tags,
                metadata=clean_metadata,
                importance=clean_importance,
                source=clean_source,
                confidence=clean_confidence,
                embedding=clean_embedding,
                dedupe=dedupe,
            )

    def _save_memory_in_connection(
        self,
        conn: sqlite3.Connection,
        *,
        content: str,
        tags: list[str],
        metadata: dict[str, Any],
        importance: int,
        source: str,
        confidence: float,
        embedding: list[float],
        dedupe: bool = True,
    ) -> str:
        """Persist one normalized memory inside the caller's transaction."""
        ensure_memory_payload_safe(content=content, tags=tags, metadata=metadata, source=source)
        if dedupe:
            duplicate = _find_duplicate_memory_in_connection(conn, content)
            if duplicate is not None:
                next_tags = _merge_tags(duplicate.tags, tags)
                next_metadata = {**duplicate.metadata, **metadata}
                next_source = duplicate.source if duplicate.source != "manual" else source
                next_embedding = duplicate.embedding or embedding
                ensure_memory_payload_safe(
                    content=duplicate.content,
                    tags=next_tags,
                    metadata=next_metadata,
                    source=next_source,
                )
                conn.execute(
                    """
                    UPDATE memories
                    SET updated_at = ?, tags = ?, metadata = ?, importance = ?,
                        source = ?, confidence = ?, embedding = ?, archived_at = NULL
                    WHERE id = ?
                    """,
                    (
                        _utc_now(),
                        json.dumps(next_tags, sort_keys=True),
                        json.dumps(next_metadata, sort_keys=True),
                        max(duplicate.importance, importance),
                        next_source,
                        max(duplicate.confidence, confidence),
                        json.dumps(next_embedding),
                        duplicate.id,
                    ),
                )
                _replace_memory_tags(conn, duplicate.id, next_tags)
                _replace_memory_fts(
                    conn,
                    enabled=self.fts_enabled,
                    memory_id=duplicate.id,
                    content=duplicate.content,
                    tags=next_tags,
                    metadata=next_metadata,
                    source=next_source,
                )
                return duplicate.id

        memory_id = str(uuid4())
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO memories (
                id, content, created_at, updated_at, tags, metadata, importance,
                source, confidence, embedding, archived_at, access_count, last_accessed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, NULL)
            """,
            (
                memory_id,
                content,
                now,
                now,
                json.dumps(tags, sort_keys=True),
                json.dumps(metadata, sort_keys=True),
                importance,
                source,
                confidence,
                json.dumps(embedding),
            ),
        )
        _replace_memory_tags(conn, memory_id, tags)
        _replace_memory_fts(
            conn,
            enabled=self.fts_enabled,
            memory_id=memory_id,
            content=content,
            tags=tags,
            metadata=metadata,
            source=source,
        )
        return memory_id

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: int | None = None,
        source: str | None = None,
        confidence: float | None = None,
        embedding: list[float] | None = None,
        archived_at: str | None = None,
    ) -> bool:
        """Update an existing memory. Returns False when the id is unknown."""
        existing = self.get_memory(memory_id, include_archived=True)
        if existing is None:
            return False

        next_content = existing.content if content is None else content.strip()
        if not next_content:
            raise ValueError("Memory content cannot be empty")

        next_tags = existing.tags if tags is None else _normalize_tags(tags)
        next_metadata = existing.metadata if metadata is None else metadata
        next_importance = existing.importance if importance is None else _normalize_importance(importance)
        next_source = existing.source if source is None else _normalize_source(source)
        next_confidence = existing.confidence if confidence is None else _normalize_confidence(confidence)
        next_embedding = existing.embedding if embedding is None else _normalize_embedding(embedding)
        next_archived_at = existing.archived_at if archived_at is None else archived_at
        if next_embedding is None:
            next_embedding = text_to_embedding(next_content)
        ensure_memory_payload_safe(
            content=next_content,
            tags=next_tags,
            metadata=next_metadata,
            source=next_source,
        )

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET content = ?, updated_at = ?, tags = ?, metadata = ?, importance = ?,
                    source = ?, confidence = ?, embedding = ?, archived_at = ?
                WHERE id = ?
                """,
                (
                    next_content,
                    _utc_now(),
                    json.dumps(next_tags, sort_keys=True),
                    json.dumps(next_metadata, sort_keys=True),
                    next_importance,
                    next_source,
                    next_confidence,
                    json.dumps(next_embedding),
                    next_archived_at,
                    memory_id,
                ),
            )
            _replace_memory_tags(conn, memory_id, next_tags)
            _replace_memory_fts(
                conn,
                enabled=self.fts_enabled,
                memory_id=memory_id,
                content=next_content,
                tags=next_tags,
                metadata=next_metadata,
                source=next_source,
            )
        return cursor.rowcount > 0

    def restore_memory(self, memory_id: str) -> bool:
        """Restore an archived memory."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET archived_at = NULL, updated_at = ? WHERE id = ? AND archived_at IS NOT NULL",
                (_utc_now(), memory_id),
            )
        return cursor.rowcount > 0

    def get_memory(self, memory_id: str, *, include_archived: bool = False) -> MemoryRecord | None:
        """Return one memory by id."""
        archived_filter = "" if include_archived else " AND archived_at IS NULL"
        with self._connect() as conn:
            row = conn.execute(f"SELECT * FROM memories WHERE id = ?{archived_filter}", (memory_id,)).fetchone()
        return _row_to_memory(row) if row else None

    def search_memory(
        self,
        query: str,
        limit: int = 5,
        *,
        embedding: list[float] | None = None,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        """Search memories with FTS when available and optional vector reranking."""
        clean_query = query.strip()
        clean_limit = _normalize_limit(limit)
        clean_embedding = _normalize_embedding(embedding)
        if clean_embedding is None and clean_query:
            clean_embedding = text_to_embedding(clean_query)
        if not clean_query and clean_embedding is None:
            return self.list_memories(limit=clean_limit, include_archived=include_archived)

        text_matches = self._search_memory_fts(clean_query, clean_limit * 5, include_archived=include_archived)
        if not text_matches and clean_query:
            text_matches = self._search_memory_like(clean_query, clean_limit * 5, include_archived=include_archived)
        if clean_embedding is None:
            return text_matches[:clean_limit]

        vector_matches = self.search_memory_by_embedding(
            clean_embedding,
            limit=clean_limit * 5,
            include_archived=include_archived,
        )
        merged = _merge_ranked_memories(text_matches, vector_matches)
        scored = []
        terms = _tokenize(clean_query)
        for memory in merged:
            lexical_score = _score_memory(memory, terms) if terms else 0
            vector_score = _cosine_similarity(clean_embedding, memory.embedding or [])
            scored.append((lexical_score + vector_score + memory.importance + memory.confidence, memory))
        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        results = [memory for _, memory in scored[:clean_limit]]
        self._mark_accessed([memory.id for memory in results])
        return results

    def search_memory_by_embedding(
        self,
        embedding: list[float],
        limit: int = 5,
        *,
        include_archived: bool = False,
    ) -> list[MemoryRecord]:
        """Search memories by vector similarity using stored embedding values."""
        clean_limit = _normalize_limit(limit)
        clean_embedding = _normalize_embedding(embedding)
        if clean_embedding is None:
            return []
        archived_filter = "" if include_archived else "WHERE archived_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                {archived_filter}
                ORDER BY importance DESC, confidence DESC, updated_at DESC
                LIMIT 1000
                """
            ).fetchall()
        records = [_row_to_memory(row) for row in rows]
        scored = [
            (_cosine_similarity(clean_embedding, memory.embedding or []), memory)
            for memory in records
            if memory.embedding
        ]
        scored = [(score, memory) for score, memory in scored if score > 0]
        scored.sort(key=lambda item: (item[0], item[1].importance, item[1].updated_at), reverse=True)
        results = [memory for _, memory in scored[:clean_limit]]
        self._mark_accessed([memory.id for memory in results])
        return results

    def search_by_tags(self, tags: list[str], limit: int = 5, *, include_archived: bool = False) -> list[MemoryRecord]:
        """Return memories matching any tag."""
        clean_tags = _normalize_tags(tags)
        clean_limit = _normalize_limit(limit)
        if not clean_tags:
            return []

        archived_filter = "" if include_archived else "AND memories.archived_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT memories.*
                FROM memories
                JOIN memory_tags ON memory_tags.memory_id = memories.id
                WHERE memory_tags.tag IN ({",".join("?" for _ in clean_tags)})
                {archived_filter}
                ORDER BY memories.confidence DESC, memories.importance DESC, memories.updated_at DESC
                LIMIT ?
                """,
                (*clean_tags, clean_limit),
            ).fetchall()

        results = [_row_to_memory(row) for row in rows]
        self._mark_accessed([memory.id for memory in results])
        return results

    def profile_memories(self, limit: int = 5) -> list[MemoryRecord]:
        """Return persona/preference/workflow memories for prompt shaping."""
        memories = self.search_by_tags(sorted(PROFILE_MEMORY_TAGS), limit=max(limit * 3, limit))
        return _resolve_profile_conflicts(memories)[:limit]

    def list_memories(self, limit: int = 50, *, include_archived: bool = False) -> list[MemoryRecord]:
        """List newest memories first, weighted by importance."""
        clean_limit = _normalize_limit(limit)
        archived_filter = "" if include_archived else "WHERE archived_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                {archived_filter}
                ORDER BY importance DESC, confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (clean_limit,),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by id."""
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
            _delete_memory_fts(conn, enabled=self.fts_enabled, memory_id=memory_id)
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def archive_memory(self, memory_id: str) -> bool:
        """Archive a memory without deleting it."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET archived_at = ?, updated_at = ? WHERE id = ? AND archived_at IS NULL",
                (_utc_now(), _utc_now(), memory_id),
            )
        return cursor.rowcount > 0

    def archive_memories_older_than(self, days: int) -> int:
        """Archive memories whose updated_at timestamp is older than the cutoff."""
        if days < 1:
            raise ValueError("days must be greater than zero")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET archived_at = ?, updated_at = ?
                WHERE updated_at < ? AND archived_at IS NULL
                """,
                (_utc_now(), _utc_now(), cutoff),
            )
        return cursor.rowcount

    def decay_importance(self, *, days_since_accessed: int = 90, amount: int = 1) -> int:
        """Lower importance for stale memories."""
        if days_since_accessed < 1:
            raise ValueError("days_since_accessed must be greater than zero")
        if amount < 1:
            raise ValueError("amount must be greater than zero")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_since_accessed)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET importance = max(1, importance - ?), updated_at = ?
                WHERE archived_at IS NULL
                  AND (last_accessed_at IS NULL OR last_accessed_at < ?)
                """,
                (amount, _utc_now(), cutoff),
            )
        return cursor.rowcount

    def compact_memories(self) -> int:
        """Archive near-duplicate active memories, keeping the strongest record."""
        memories = self.list_memories(limit=100, include_archived=False)
        archived = 0
        for index, memory in enumerate(memories):
            if memory.archived_at:
                continue
            for other in memories[index + 1 :]:
                if other.archived_at:
                    continue
                if _content_similarity(memory.content, other.content) < 0.86:
                    continue
                keep, archive = _choose_memory_to_keep(memory, other)
                if self.archive_memory(archive.id):
                    archived += 1
                memory = keep
        return archived

    def find_duplicate_memory(self, content: str, *, threshold: float = 0.90) -> MemoryRecord | None:
        """Return a likely duplicate active memory, if one exists."""
        clean_content = content.strip()
        if not clean_content:
            return None
        normalized = _normalize_content(clean_content)
        candidates = self.search_memory(clean_content, limit=20)
        for candidate in candidates:
            if _normalize_content(candidate.content) == normalized:
                return candidate
            if _content_similarity(candidate.content, clean_content) >= threshold:
                return candidate
        return None

    def extract_memory_candidates(self, text: str) -> list[MemoryExtractionCandidate]:
        """Extract explicit user-requested memories from a user message."""
        return extract_memory_candidates(text)

    def extract_and_save_memories(self, text: str) -> list[str]:
        """Extract explicit memories from text and save them."""
        memory_ids = []
        for candidate in self.extract_memory_candidates(text):
            memory_ids.append(
                self.save_memory(
                    candidate.content,
                    tags=candidate.tags,
                    metadata=candidate.metadata,
                    importance=candidate.importance,
                    source=candidate.source,
                    confidence=candidate.confidence,
                )
            )
        return memory_ids

    def create_memory_proposal(
        self,
        content: str,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: int = 1,
        source: str = "manual_review",
        confidence: float = 1.0,
        evidence: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> str:
        """Persist one candidate without making it available to retrieval."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory proposal content cannot be empty")
        clean_tags = _normalize_tags(tags or [])
        clean_metadata = metadata or {}
        clean_source = _normalize_source(source)
        ensure_memory_payload_safe(
            content=clean_content,
            tags=clean_tags,
            metadata=clean_metadata,
            source=clean_source,
            evidence=evidence,
            conversation_id=conversation_id,
            turn_id=turn_id,
        )
        proposal_id = str(uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_proposals (
                    id, content, tags, metadata, importance, source, confidence,
                    evidence, conversation_id, turn_id, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    proposal_id,
                    clean_content,
                    json.dumps(clean_tags, sort_keys=True),
                    json.dumps(clean_metadata, sort_keys=True),
                    _normalize_importance(importance),
                    clean_source,
                    _normalize_confidence(confidence),
                    evidence,
                    conversation_id,
                    turn_id,
                    _utc_now(),
                ),
            )
        return proposal_id

    def list_memory_proposals(self, *, status: str | None = "pending") -> list[MemoryProposalRecord]:
        """List durable proposals, newest first."""
        if status is not None and status not in {"pending", "approved", "rejected"}:
            raise ValueError("proposal status must be pending, approved, rejected, or None")
        query = "SELECT * FROM memory_proposals"
        parameters: tuple[object, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        return [_row_to_memory_proposal(row) for row in rows]

    def get_memory_proposal(self, proposal_id: str) -> MemoryProposalRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memory_proposals WHERE id = ?", (proposal_id,)).fetchone()
        return _row_to_memory_proposal(row) if row is not None else None

    def approve_memory_proposal(self, proposal_id: str) -> MemoryProposalRecord:
        """Accept one pending proposal and persist it as a retrievable memory."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM memory_proposals WHERE id = ?", (proposal_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory proposal: {proposal_id}")
            proposal = _row_to_memory_proposal(row)
            if proposal.status != "pending":
                return proposal
            ensure_memory_payload_safe(
                content=proposal.content,
                tags=proposal.tags,
                metadata=proposal.metadata,
                source=proposal.source,
                evidence=proposal.evidence,
                conversation_id=proposal.conversation_id,
                turn_id=proposal.turn_id,
            )
            memory_id = self._save_memory_in_connection(
                conn,
                content=proposal.content,
                tags=_normalize_tags(proposal.tags),
                metadata={**proposal.metadata, "proposal_id": proposal.id},
                importance=_normalize_importance(proposal.importance),
                source=_normalize_source(proposal.source),
                confidence=_normalize_confidence(proposal.confidence),
                embedding=text_to_embedding(proposal.content),
            )
            conn.execute(
                """
                UPDATE memory_proposals
                SET status = 'approved', reviewed_at = ?, accepted_memory_id = ?
                WHERE id = ? AND status = 'pending'
                """,
                (_utc_now(), memory_id, proposal_id),
            )
            approved_row = conn.execute(
                "SELECT * FROM memory_proposals WHERE id = ?",
                (proposal_id,),
            ).fetchone()
            assert approved_row is not None
            return _row_to_memory_proposal(approved_row)

    def reject_memory_proposal(self, proposal_id: str) -> MemoryProposalRecord:
        """Reject one pending proposal without creating a memory."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM memory_proposals WHERE id = ?", (proposal_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory proposal: {proposal_id}")
            proposal = _row_to_memory_proposal(row)
            if proposal.status != "pending":
                return proposal
            conn.execute(
                """
                UPDATE memory_proposals
                SET status = 'rejected', reviewed_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (_utc_now(), proposal_id),
            )
            rejected_row = conn.execute(
                "SELECT * FROM memory_proposals WHERE id = ?",
                (proposal_id,),
            ).fetchone()
            assert rejected_row is not None
            return _row_to_memory_proposal(rejected_row)

    def import_markdown(self, path: Path | str) -> list[str]:
        """Import simple bullet memories from a Markdown file."""
        markdown_path = Path(path)
        if not markdown_path.exists():
            raise FileNotFoundError(markdown_path)
        parsed_memories: list[tuple[str, list[str]]] = []
        for line in markdown_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_markdown_memory_line(line)
            if parsed is None:
                continue
            content, tags = parsed
            ensure_memory_payload_safe(
                content=content,
                tags=tags,
                metadata={"path": str(markdown_path)},
                source="memory_md",
            )
            parsed_memories.append((content, tags))

        memory_ids = []
        for content, tags in parsed_memories:
            memory_ids.append(
                self.save_memory(
                    content,
                    tags=tags,
                    source="memory_md",
                    confidence=0.8,
                    metadata={"path": str(markdown_path)},
                )
            )
        return memory_ids

    def export_markdown(self, path: Path | str, *, include_archived: bool = False) -> int:
        """Export memories to a human-readable Markdown file."""
        markdown_path = Path(path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        memories = self.list_memories(limit=100, include_archived=include_archived)
        lines = ["# MEMORY", "", "Human-readable export from ChulkHarness SQLite memory.", ""]
        for memory in memories:
            tag_text = ", ".join(memory.tags) if memory.tags else "untagged"
            lines.append(f"- [{tag_text}] {memory.content}")
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return len(memories)

    def summarize_memories(self, query: str | None = None, limit: int = 10) -> str:
        """Return a compact text summary of stored memories."""
        memories = self.search_memory(query, limit=limit) if query else self.list_memories(limit=limit)
        if not memories:
            return "No memories found."
        lines = []
        for memory in memories:
            tag_text = ", ".join(memory.tags) if memory.tags else "untagged"
            lines.append(
                f"- [{memory.id}] ({tag_text}, source {memory.source}, confidence {memory.confidence:.2f}, "
                f"importance {memory.importance}) {memory.content}"
            )
        return "\n".join(lines)

    def _search_memory_fts(self, query: str, limit: int, *, include_archived: bool) -> list[MemoryRecord]:
        if not self.fts_enabled:
            return []
        terms = _tokenize(query)
        if not terms:
            return []
        clean_limit = _normalize_limit(limit)
        fts_query = " OR ".join(f"{term}*" for term in terms)
        archived_filter = "" if include_archived else "AND memories.archived_at IS NULL"
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT memories.*
                    FROM memories_fts
                    JOIN memories ON memories.id = memories_fts.memory_id
                    WHERE memories_fts MATCH ?
                    {archived_filter}
                    ORDER BY bm25(memories_fts), memories.importance DESC,
                             memories.confidence DESC, memories.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_query, clean_limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        results = [_row_to_memory(row) for row in rows]
        self._mark_accessed([memory.id for memory in results])
        return results

    def _search_memory_like(self, query: str, limit: int, *, include_archived: bool) -> list[MemoryRecord]:
        terms = _tokenize(query)
        if not terms:
            return []
        clean_limit = _normalize_limit(limit)
        like_values = [f"%{term}%" for term in terms]
        clauses = " OR ".join(["lower(content || ' ' || tags || ' ' || metadata || ' ' || source) LIKE ?"] * len(like_values))
        archived_filter = "" if include_archived else "AND archived_at IS NULL"
        sql = f"SELECT * FROM memories WHERE ({clauses}) {archived_filter}"

        with self._connect() as conn:
            rows = conn.execute(sql, like_values).fetchall()

        scored = [(_score_memory(_row_to_memory(row), terms), _row_to_memory(row)) for row in rows]
        scored = [(score, memory) for score, memory in scored if score > 0]
        scored.sort(key=lambda item: (item[0], item[1].importance, item[1].confidence, item[1].updated_at), reverse=True)
        results = [memory for _, memory in scored[:clean_limit]]
        self._mark_accessed([memory.id for memory in results])
        return results

    def _mark_accessed(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE id = ?
                """,
                [(_utc_now(), memory_id) for memory_id in memory_ids],
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            _annotate_memory_error(exc, "connect")
            raise
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            _annotate_memory_error(exc, "transaction")
            raise
        finally:
            conn.close()


def select_memories_for_prompt(
    memory_store: SQLiteMemoryStore,
    user_message: str,
    *,
    relevant_limit: int = 5,
    profile_limit: int = 5,
) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
    """Select profile memories and query-relevant memories for one turn."""
    profile = memory_store.profile_memories(limit=profile_limit)
    relevant = memory_store.search_memory(user_message, limit=relevant_limit)
    profile_ids = {memory.id for memory in profile}
    relevant = [memory for memory in relevant if memory.id not in profile_ids]
    return profile, relevant


def _annotate_memory_error(exc: sqlite3.Error, operation: str) -> None:
    """Attach non-sensitive store context for the public boundary mapper."""
    try:
        exc.memory_operation = operation  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass


def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        content=row["content"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        tags=_safe_json_list(row["tags"]),
        metadata=_safe_json_dict(row["metadata"]),
        importance=row["importance"],
        source=row["source"],
        confidence=row["confidence"],
        embedding=_safe_json_float_list(row["embedding"]),
        archived_at=row["archived_at"],
        access_count=row["access_count"],
        last_accessed_at=row["last_accessed_at"],
    )


def _row_to_memory_proposal(row: sqlite3.Row) -> MemoryProposalRecord:
    return MemoryProposalRecord(
        id=row["id"],
        content=row["content"],
        tags=_safe_json_list(row["tags"]),
        metadata=_safe_json_dict(row["metadata"]),
        importance=row["importance"],
        source=row["source"],
        confidence=row["confidence"],
        evidence=row["evidence"],
        conversation_id=row["conversation_id"],
        turn_id=row["turn_id"],
        status=row["status"],
        created_at=row["created_at"],
        reviewed_at=row["reviewed_at"],
        accepted_memory_id=row["accepted_memory_id"],
    )


def _find_duplicate_memory_in_connection(
    conn: sqlite3.Connection,
    content: str,
    *,
    threshold: float = 0.90,
) -> MemoryRecord | None:
    """Find an active duplicate without leaving the current transaction."""
    normalized = _normalize_content(content)
    rows = conn.execute(
        """
        SELECT * FROM memories
        WHERE archived_at IS NULL
        ORDER BY importance DESC, confidence DESC, updated_at DESC
        LIMIT 1000
        """
    ).fetchall()
    for row in rows:
        candidate = _row_to_memory(row)
        if _normalize_content(candidate.content) == normalized:
            return candidate
        if _content_similarity(candidate.content, content) >= threshold:
            return candidate
    return None


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(memory_id UNINDEXED, content, tags, metadata, source)
            """
        )
    except sqlite3.OperationalError:
        return False
    return True


def _replace_memory_tags(conn: sqlite3.Connection, memory_id: str, tags: list[str]) -> None:
    conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
    conn.executemany(
        "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
        [(memory_id, tag) for tag in tags],
    )


def _replace_memory_fts(
    conn: sqlite3.Connection,
    *,
    enabled: bool,
    memory_id: str,
    content: str,
    tags: list[str],
    metadata: dict[str, Any],
    source: str,
) -> None:
    if not enabled:
        return
    _delete_memory_fts(conn, enabled=enabled, memory_id=memory_id)
    conn.execute(
        """
        INSERT INTO memories_fts (memory_id, content, tags, metadata, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (memory_id, content, " ".join(tags), json.dumps(metadata, sort_keys=True), source),
    )


def _delete_memory_fts(conn: sqlite3.Connection, *, enabled: bool, memory_id: str) -> None:
    if enabled:
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))


def _backfill_memory_tags(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, tags FROM memories").fetchall()
    for row in rows:
        _replace_memory_tags(conn, row["id"], _normalize_tags(_safe_json_list(row["tags"])))


def _backfill_memory_fts(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT count(*) AS count FROM memories_fts").fetchone()["count"]
    if count:
        return
    rows = conn.execute("SELECT * FROM memories").fetchall()
    for row in rows:
        _replace_memory_fts(
            conn,
            enabled=True,
            memory_id=row["id"],
            content=row["content"],
            tags=_normalize_tags(_safe_json_list(row["tags"])),
            metadata=_safe_json_dict(row["metadata"]),
            source=row["source"],
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
