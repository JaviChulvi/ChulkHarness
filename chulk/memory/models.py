"""Durable memory data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass(frozen=True)
class MemoryProposalRecord:
    """A durable memory candidate awaiting or recording host review."""

    id: str
    content: str
    tags: list[str]
    metadata: dict[str, Any]
    importance: int
    source: str
    confidence: float
    evidence: str | None
    conversation_id: str | None
    turn_id: str | None
    status: str
    created_at: str
    reviewed_at: str | None = None
    accepted_memory_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "importance": self.importance,
            "source": self.source,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "status": self.status,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "accepted_memory_id": self.accepted_memory_id,
        }
