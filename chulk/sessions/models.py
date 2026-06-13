"""Session persistence data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConversationRecord:
    """A persisted agent conversation."""

    id: str
    created_at: str
    updated_at: str
    provider: str
    model: str
    trace_path: str | None = None
    title: str | None = None
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0


@dataclass(frozen=True)
class MessageRecord:
    """A persisted short-term conversation message."""

    id: str
    conversation_id: str
    turn_id: str | None
    role: str
    content: str
    ordinal: int
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
