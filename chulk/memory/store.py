"""Short-term conversation memory primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Memory:
    """A simple memory value object used before persistence."""

    id: str
    content: str
    created_at: str
    tags: list[str]
    metadata: dict[str, Any]


class ConversationMemory:
    """In-memory short-term conversation state."""

    def __init__(self, max_messages: int = 20) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be greater than zero")
        self.max_messages = max_messages
        self.messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant", "tool", "observation"}:
            raise ValueError(f"Unsupported message role: {role}")
        self.messages.append({"role": role, "content": content})
        self.trim_to_limit()

    def add_user_message(self, content: str) -> None:
        self.add("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add("assistant", content)

    def add_observation(self, content: str) -> None:
        self.add("observation", content)

    def trim_to_limit(self) -> None:
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def recent(self, limit: int | None = None) -> list[dict[str, str]]:
        if limit is None:
            limit = self.max_messages
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
