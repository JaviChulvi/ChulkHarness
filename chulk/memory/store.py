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
        selected = select_recent_conversation_messages(
            self.messages,
            max_messages=self.max_messages,
        )
        if len(selected) == len(self.messages):
            return
        dropped_count = len(self.messages) - len(selected)
        self._pending_summary_messages.extend(self.messages[:dropped_count])
        self.messages = selected

    def recent(self, limit: int | None = None) -> list[dict[str, str]]:
        if limit is None:
            limit = self.max_messages
        return select_recent_conversation_messages(
            self.messages,
            max_messages=limit,
        )

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


def select_recent_conversation_messages(
    messages: list[dict[str, str]],
    *,
    max_messages: int,
) -> list[dict[str, str]]:
    """Select a recent suffix without splitting the active turn or a tool result."""
    if max_messages < 1:
        raise ValueError("max_messages must be greater than zero")
    if not messages:
        return []

    blocks = _conversation_message_blocks(messages)
    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        None,
    )

    if latest_user_index is None:
        selected_start = blocks[-1][0]
        selected_count = blocks[-1][1] - blocks[-1][0]
        earlier_blocks = reversed(blocks[:-1])
    else:
        protected_block_index = next(
            index
            for index, (start, end) in enumerate(blocks)
            if start <= latest_user_index < end
        )
        selected_start = blocks[protected_block_index][0]
        selected_count = len(messages) - selected_start
        earlier_blocks = reversed(blocks[:protected_block_index])

    for start, end in earlier_blocks:
        block_size = end - start
        if selected_count + block_size > max_messages:
            break
        selected_start = start
        selected_count += block_size

    return list(messages[selected_start:])


def _conversation_message_blocks(
    messages: list[dict[str, str]],
) -> list[tuple[int, int]]:
    """Return half-open message ranges, pairing executed actions with observations."""
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        next_index = index + 1
        if (
            _is_executed_tool_action(messages[index])
            and next_index < len(messages)
            and messages[next_index].get("role") == "observation"
        ):
            next_index += 1
        blocks.append((index, next_index))
        index = next_index
    return blocks


def _is_executed_tool_action(message: dict[str, str]) -> bool:
    return (
        message.get("role") == "assistant"
        and message.get("content", "").lstrip().startswith("<executed_tool_action>")
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
