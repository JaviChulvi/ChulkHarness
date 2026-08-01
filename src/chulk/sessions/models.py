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


@dataclass(frozen=True)
class ConversationSummaryRecord:
    """A compact task-local summary for older messages in a conversation."""

    id: str
    conversation_id: str
    content: str
    source_message_count: int
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionHit:
    """One bounded exact-message match from a profile-owned conversation."""

    message_id: str
    conversation_id: str
    ordinal: int
    role: str
    snippet: str
    created_at: str
    turn_id: str | None = None
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "ordinal": self.ordinal,
            "role": self.role,
            "snippet": self.snippet,
            "created_at": self.created_at,
            "turn_id": self.turn_id,
            "trace_path": self.trace_path,
        }


@dataclass(frozen=True, slots=True)
class SessionSearchPage:
    """One bounded page of deterministic cross-session hits."""

    query: str
    hits: tuple[SessionHit, ...]
    next_cursor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [hit.to_dict() for hit in self.hits],
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True, slots=True)
class SessionMessage:
    """One eligible message returned in a bounded session window."""

    message_id: str
    conversation_id: str
    ordinal: int
    role: str
    content: str
    created_at: str
    turn_id: str | None = None
    sensitive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "ordinal": self.ordinal,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "turn_id": self.turn_id,
            "sensitive": self.sensitive,
        }


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """A bounded page around a stable conversation ordinal."""

    conversation_id: str
    anchor_ordinal: int
    messages: tuple[SessionMessage, ...]
    next_cursor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "anchor_ordinal": self.anchor_ordinal,
            "messages": [message.to_dict() for message in self.messages],
            "next_cursor": self.next_cursor,
        }
