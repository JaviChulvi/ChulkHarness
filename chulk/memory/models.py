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
