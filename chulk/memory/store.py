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
        self.conversation_summary: str | None = None
        self.summary_message_count = 0
        self._total_message_count = 0
        self._pending_summary_messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant", "tool", "observation"}:
            raise ValueError(f"Unsupported message role: {role}")
        self.messages.append({"role": role, "content": content})
        self._total_message_count += 1
        self.trim_to_limit()

    def add_user_message(self, content: str) -> None:
        self.add("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add("assistant", content)

    def add_observation(self, content: str) -> None:
        self.add("observation", content)

    def trim_to_limit(self) -> None:
        if len(self.messages) > self.max_messages:
            dropped_count = len(self.messages) - self.max_messages
            self._pending_summary_messages.extend(self.messages[:dropped_count])
            self.messages = self.messages[-self.max_messages :]

    def recent(self, limit: int | None = None) -> list[dict[str, str]]:
        if limit is None:
            limit = self.max_messages
        return self.messages[-limit:]

    def replace(
        self,
        messages: list[dict[str, str]],
        *,
        conversation_summary: str | None = None,
        summary_message_count: int = 0,
    ) -> None:
        """Replace runtime history from a persisted session."""
        self.messages = [
            {"role": str(message.get("role") or ""), "content": str(message.get("content") or "")}
            for message in messages
            if str(message.get("role") or "") in {"system", "user", "assistant", "tool", "observation"}
        ]
        self.conversation_summary = conversation_summary.strip() if conversation_summary and conversation_summary.strip() else None
        self.summary_message_count = max(0, summary_message_count)
        self._total_message_count = self.summary_message_count + len(self.messages)
        self._pending_summary_messages = []
        self.trim_to_limit()

    def consume_pending_summary_messages(self) -> list[dict[str, str]]:
        """Return messages dropped by the raw history limit since the last compaction."""
        messages = list(self._pending_summary_messages)
        self._pending_summary_messages = []
        return messages

    def remove_messages(self, messages_to_remove: list[dict[str, str]]) -> int:
        """Remove exact message objects from raw history after summarizing them."""
        remove_ids = {id(message) for message in messages_to_remove}
        before_count = len(self.messages)
        self.messages = [message for message in self.messages if id(message) not in remove_ids]
        return before_count - len(self.messages)

    def update_conversation_summary(self, content: str, *, summarized_message_count: int) -> None:
        """Store the rolling compact summary and its raw-message coverage."""
        clean_content = content.strip()
        if not clean_content:
            return
        self.conversation_summary = clean_content
        self.summary_message_count = min(
            self._total_message_count,
            self.summary_message_count + max(0, summarized_message_count),
        )


def new_memory(content: str, tags: list[str] | None = None, metadata: dict[str, Any] | None = None) -> Memory:
    """Create a memory object before SQLite persistence is implemented."""
    return Memory(
        id=str(uuid4()),
        content=content,
        created_at=datetime.now(timezone.utc).isoformat(),
        tags=tags or [],
        metadata=metadata or {},
    )
