"""SQLite-backed long-term memory store."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any
from uuid import uuid4


PROFILE_MEMORY_TAGS = {"persona", "preference", "style", "workflow"}
DEFAULT_EMBEDDING_DIMENSIONS = 64
SEARCH_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "into",
    "the",
    "this",
    "that",
    "what",
    "when",
    "where",
    "with",
    "work",
    "works",
}


@dataclass(frozen=True)
class MemoryRecord:
    """A durable memory record stored in SQLite."""

    id: str
    content: str
    created_at: str
    updated_at: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: int = 1
    source: str = "manual"
    confidence: float = 1.0
    embedding: list[float] | None = None
    archived_at: str | None = None
    access_count: int = 0
    last_accessed_at: str | None = None


@dataclass(frozen=True)
class MemoryExtractionCandidate:
    """A candidate durable memory extracted from a user message."""

    content: str
    tags: list[str]
    source: str = "auto_extracted"
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: int = 5


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
        clean_embedding = _normalize_embedding(embedding) or text_to_embedding(clean_content)

        if dedupe:
            duplicate = self.find_duplicate_memory(clean_content)
            if duplicate is not None:
                self.update_memory(
                    duplicate.id,
                    tags=_merge_tags(duplicate.tags, clean_tags),
                    metadata={**duplicate.metadata, **clean_metadata},
                    importance=max(duplicate.importance, clean_importance),
                    source=duplicate.source if duplicate.source != "manual" else clean_source,
                    confidence=max(duplicate.confidence, clean_confidence),
                    embedding=duplicate.embedding or clean_embedding,
                    archived_at=None,
                )
                return duplicate.id

        memory_id = str(uuid4())
        now = _utc_now()

        with self._connect() as conn:
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
                    clean_content,
                    now,
                    now,
                    json.dumps(clean_tags, sort_keys=True),
                    json.dumps(clean_metadata, sort_keys=True),
                    clean_importance,
                    clean_source,
                    clean_confidence,
                    json.dumps(clean_embedding),
                ),
            )
            _replace_memory_tags(conn, memory_id, clean_tags)
            _replace_memory_fts(
                conn,
                enabled=self.fts_enabled,
                memory_id=memory_id,
                content=clean_content,
                tags=clean_tags,
                metadata=clean_metadata,
                source=clean_source,
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

    def import_markdown(self, path: Path | str) -> list[str]:
        """Import simple bullet memories from a Markdown file."""
        markdown_path = Path(path)
        if not markdown_path.exists():
            raise FileNotFoundError(markdown_path)
        memory_ids = []
        for line in markdown_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_markdown_memory_line(line)
            if parsed is None:
                continue
            content, tags = parsed
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
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


def extract_memory_candidates(text: str) -> list[MemoryExtractionCandidate]:
    """Extract explicit memories from text without hidden inference."""
    clean_text = text.strip()
    if not clean_text:
        return []

    patterns = [
        (r"\bremember that (?P<content>.+)", ["explicit", "user"]),
        (r"\bplease remember (?P<content>.+)", ["explicit", "user"]),
        (r"\bmy preference is (?P<content>.+)", ["preference", "user"]),
        (r"\bi prefer (?P<content>.+)", ["preference", "user"]),
    ]
    candidates: list[MemoryExtractionCandidate] = []
    seen_content: set[str] = set()
    for pattern, tags in patterns:
        match = re.search(pattern, clean_text, flags=re.IGNORECASE)
        if not match:
            continue
        content = _strip_sentence(match.group("content"))
        if not content:
            continue
        if "preference" in tags and not content.lower().startswith("user prefers"):
            content = f"User prefers {content}"
        normalized = _normalize_content(content)
        if normalized in seen_content:
            continue
        seen_content.add(normalized)
        candidates.append(
            MemoryExtractionCandidate(
                content=content,
                tags=tags,
                metadata={"extracted_from": "user_message"},
                importance=7 if "preference" in tags else 5,
                confidence=0.9,
            )
        )
    return candidates


def text_to_embedding(text: str, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> list[float]:
    """Create a small deterministic lexical embedding for local vector search."""
    vector = [0.0] * dimensions
    terms = _tokenize(text)
    if not terms:
        return vector
    for term in terms:
        index = sum(ord(char) for char in term) % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


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


def _normalize_tags(tags: list[str]) -> list[str]:
    clean_tags = []
    for tag in tags:
        clean = str(tag).strip().lower()
        if clean and clean not in clean_tags:
            clean_tags.append(clean)
    return clean_tags


def _merge_tags(left: list[str], right: list[str]) -> list[str]:
    return _normalize_tags([*left, *right])


def _normalize_importance(importance: int) -> int:
    if not isinstance(importance, int) or isinstance(importance, bool):
        raise ValueError("Memory importance must be an integer")
    if importance < 1 or importance > 10:
        raise ValueError("Memory importance must be between 1 and 10")
    return importance


def _normalize_limit(limit: int) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("Memory limit must be an integer")
    if limit < 1:
        raise ValueError("Memory limit must be greater than zero")
    return min(limit, 100)


def _normalize_source(source: str) -> str:
    clean_source = str(source).strip().lower().replace(" ", "_")
    return clean_source or "manual"


def _normalize_confidence(confidence: float) -> float:
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise ValueError("Memory confidence must be a number")
    if confidence < 0 or confidence > 1:
        raise ValueError("Memory confidence must be between 0 and 1")
    return float(confidence)


def _normalize_embedding(embedding: list[float] | None) -> list[float] | None:
    if embedding is None:
        return None
    if not isinstance(embedding, list):
        raise ValueError("Memory embedding must be a list of numbers")
    clean_embedding: list[float] = []
    for value in embedding:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError("Memory embedding values must be numbers")
        clean_embedding.append(float(value))
    return clean_embedding


def _safe_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _safe_json_float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    parsed = _safe_json_list(value)
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None


def _safe_json_dict(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tokenize(text: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [term for term in terms if len(term) > 2 and term not in SEARCH_STOPWORDS]


def _score_memory(memory: MemoryRecord, terms: list[str]) -> int:
    haystack = " ".join(
        [
            memory.content,
            " ".join(memory.tags),
            json.dumps(memory.metadata, sort_keys=True),
            memory.source,
        ]
    ).lower()
    term_score = sum(haystack.count(term) for term in terms)
    return term_score + memory.importance


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _merge_ranked_memories(left: list[MemoryRecord], right: list[MemoryRecord]) -> list[MemoryRecord]:
    merged: dict[str, MemoryRecord] = {}
    for memory in [*left, *right]:
        merged[memory.id] = memory
    return list(merged.values())


def _normalize_content(content: str) -> str:
    return " ".join(_tokenize(content))


def _content_similarity(left: str, right: str) -> float:
    left_terms = set(_tokenize(left))
    right_terms = set(_tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _choose_memory_to_keep(left: MemoryRecord, right: MemoryRecord) -> tuple[MemoryRecord, MemoryRecord]:
    left_score = (left.confidence, left.importance, left.updated_at)
    right_score = (right.confidence, right.importance, right.updated_at)
    return (left, right) if left_score >= right_score else (right, left)


def _resolve_profile_conflicts(memories: list[MemoryRecord]) -> list[MemoryRecord]:
    ordered = sorted(memories, key=lambda item: (item.confidence, item.importance, item.updated_at), reverse=True)
    selected: list[MemoryRecord] = []
    for memory in ordered:
        is_near_duplicate = any(
            set(memory.tags) & set(existing.tags) & PROFILE_MEMORY_TAGS
            and _content_similarity(memory.content, existing.content) >= 0.86
            for existing in selected
        )
        if not is_near_duplicate:
            selected.append(memory)
    return selected


def _parse_markdown_memory_line(line: str) -> tuple[str, list[str]] | None:
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None
    body = stripped[2:].strip()
    tags: list[str] = []
    tag_match = re.match(r"\[(?P<tags>[^\]]+)\]\s+(?P<content>.+)", body)
    if tag_match:
        tags = [tag.strip() for tag in tag_match.group("tags").split(",")]
        body = tag_match.group("content").strip()
    if not body:
        return None
    return body, _normalize_tags(tags)


def _strip_sentence(text: str) -> str:
    return text.strip().strip(".! ")
