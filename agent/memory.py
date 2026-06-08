"""Short-term and long-term memory primitives."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Memory:
    """A durable memory record."""

    id: str
    content: str
    created_at: str
    tags: list[str]
    metadata: dict[str, Any]


class ConversationMemory:
    """In-memory short-term conversation state."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def recent(self, limit: int = 20) -> list[dict[str, str]]:
        return self.messages[-limit:]


def new_memory(content: str, tags: list[str] | None = None, metadata: dict[str, Any] | None = None) -> Memory:
    """Create a memory object before SQLite persistence is implemented."""
    return Memory(
        id=str(uuid4()),
        content=content,
        created_at=datetime.now(timezone.utc).isoformat(),
        tags=tags or [],
        metadata=metadata or {},
    )
