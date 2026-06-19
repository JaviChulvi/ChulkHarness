"""Context accounting and prompt-budget helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any


ESTIMATED_CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4


@dataclass(frozen=True)
class ContextBudget:
    """Approximate prompt budget used before a provider request."""

    max_prompt_tokens: int = 0
    response_reserve_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.max_prompt_tokens < 0:
            raise ValueError("max_prompt_tokens cannot be negative")
        if self.response_reserve_tokens < 0:
            raise ValueError("response_reserve_tokens cannot be negative")

    @property
    def enabled(self) -> bool:
        return self.max_prompt_tokens > 0

    @property
    def input_token_budget(self) -> int | None:
        if not self.enabled:
            return None
        return max(0, self.max_prompt_tokens - self.response_reserve_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "context_window_tokens": self.max_prompt_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "response_reserve_tokens": self.response_reserve_tokens,
            "input_token_budget": self.input_token_budget,
        }


@dataclass(frozen=True)
class TurnContextSection:
    """Host-provided context for one turn, such as retrieved sources."""

    id: str
    content: str
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ContextSection:
    """One named chunk of context sent to the model."""

    name: str
    label: str
    char_count: int
    estimated_tokens: int
    item_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        name: str,
        label: str,
        text: str,
        *,
        item_count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextSection":
        return cls(
            name=name,
            label=label,
            char_count=len(text),
            estimated_tokens=estimate_tokens(text),
            item_count=item_count,
            metadata=metadata or {},
        )

    @classmethod
    def from_messages(
        cls,
        name: str,
        label: str,
        messages: list[dict[str, str]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextSection":
        char_count = sum(len(_message_content(message)) for message in messages)
        estimated_tokens = sum(estimate_message_tokens(message) for message in messages)
        return cls(
            name=name,
            label=label,
            char_count=char_count,
            estimated_tokens=estimated_tokens,
            item_count=len(messages),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "item_count": self.item_count,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ContextReport:
    """Inspectable prompt context report for one model request."""

    sections: list[ContextSection]
    budget: ContextBudget = field(default_factory=ContextBudget)
    message_estimated_tokens: int | None = None
    included_message_count: int = 0
    omitted_message_count: int = 0
    omitted_observation_count: int = 0

    @property
    def total_char_count(self) -> int:
        return sum(section.char_count for section in self.sections)

    @property
    def estimated_tokens(self) -> int:
        if self.message_estimated_tokens is not None:
            return self.message_estimated_tokens
        return self.section_estimated_tokens

    @property
    def section_estimated_tokens(self) -> int:
        return sum(section.estimated_tokens for section in self.sections)

    @property
    def over_budget_tokens(self) -> int:
        input_budget = self.budget.input_token_budget
        if input_budget is None:
            return 0
        return max(0, self.estimated_tokens - input_budget)

    @property
    def trimmed(self) -> bool:
        return self.omitted_message_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_char_count": self.total_char_count,
            "estimated_tokens": self.estimated_tokens,
            "section_estimated_tokens": self.section_estimated_tokens,
            "budget": self.budget.to_dict(),
            "over_budget_tokens": self.over_budget_tokens,
            "trimmed": self.trimmed,
            "included_message_count": self.included_message_count,
            "omitted_message_count": self.omitted_message_count,
            "omitted_observation_count": self.omitted_observation_count,
            "sections": [section.to_dict() for section in self.sections],
        }


@dataclass(frozen=True)
class AgentPrompt:
    """Messages plus accounting metadata for one model request."""

    messages: list[dict[str, str]]
    context_report: ContextReport
    omitted_messages: list[dict[str, str]] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Return a deterministic, provider-agnostic token estimate."""
    if not text:
        return 0
    return max(1, ceil(len(text) / ESTIMATED_CHARS_PER_TOKEN))


def estimate_message_tokens(message: dict[str, str]) -> int:
    """Estimate token cost for a chat message including a small role overhead."""
    return (
        MESSAGE_OVERHEAD_TOKENS
        + estimate_tokens(_message_role(message))
        + estimate_tokens(_message_content(message))
    )


def select_messages_for_budget(
    *,
    system_message: dict[str, str],
    history_messages: list[dict[str, str]],
    budget: ContextBudget,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return history messages that fit the prompt budget plus omitted messages."""
    if not budget.enabled:
        return list(history_messages), []

    input_budget = budget.input_token_budget
    if input_budget is None:
        return list(history_messages), []

    selected = list(history_messages)
    while _messages_token_count([system_message, *selected]) > input_budget:
        drop_range = _oldest_droppable_history_block(selected)
        if drop_range is None:
            break
        del selected[drop_range]

    selected_ids = {id(message) for message in selected}
    omitted = [message for message in history_messages if id(message) not in selected_ids]
    return selected, omitted


def build_context_report(
    *,
    system_sections: list[ContextSection],
    history_messages: list[dict[str, str]],
    omitted_messages: list[dict[str, str]],
    budget: ContextBudget,
    sent_messages: list[dict[str, str]],
) -> ContextReport:
    """Build a per-section context report for prompt visibility."""
    non_observation_messages = [
        message for message in history_messages if _message_role(message) != "observation"
    ]
    observation_messages = [
        message for message in history_messages if _message_role(message) == "observation"
    ]
    sections = [
        *system_sections,
        ContextSection.from_messages(
            "history",
            "Conversation history",
            non_observation_messages,
            metadata={"roles": _role_counts(non_observation_messages)},
        ),
        ContextSection.from_messages(
            "observations",
            "Tool observations",
            observation_messages,
            metadata={"roles": _role_counts(observation_messages)},
        ),
    ]
    omitted_observations = sum(1 for message in omitted_messages if _message_role(message) == "observation")
    return ContextReport(
        sections=sections,
        budget=budget,
        message_estimated_tokens=_messages_token_count(sent_messages),
        included_message_count=len(history_messages),
        omitted_message_count=len(omitted_messages),
        omitted_observation_count=omitted_observations,
    )


def _messages_token_count(messages: list[dict[str, str]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def _oldest_droppable_history_block(messages: list[dict[str, str]]) -> slice | None:
    """Return the oldest complete pre-current-turn message block to omit."""
    latest_user_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if _message_role(messages[index]) == "user":
            latest_user_index = index
            break

    drop_limit = latest_user_index if latest_user_index is not None else len(messages)
    if drop_limit <= 0:
        return None

    end = drop_limit
    for index in range(1, drop_limit):
        if _message_role(messages[index]) == "user":
            end = index
            break
    return slice(0, end)


def _role_counts(messages: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in messages:
        role = _message_role(message)
        counts[role] = counts.get(role, 0) + 1
    return counts


def _message_role(message: dict[str, str]) -> str:
    return str(message.get("role", ""))


def _message_content(message: dict[str, str]) -> str:
    return str(message.get("content", ""))
