"""Planning-phase policy."""

from __future__ import annotations


READ_ONLY_PLANNING_TOOL_NAMES = frozenset(
    {
        "calculator",
        "list_files",
        "read_file",
        "search_files",
        "search_memory",
        "list_memories",
        "summarize_memories",
    }
)

SUBSTANTIVE_RECONNAISSANCE_TOOL_NAMES = frozenset({"read_file", "search_files", "search_memory", "summarize_memories"})

RECONNAISSANCE_STEP_TERMS = frozenset(
    {
        "analyze",
        "check",
        "examine",
        "explore",
        "inspect",
        "list",
        "look",
        "open",
        "read",
        "review",
        "search",
        "see",
        "understand",
    }
)

IMPLEMENTATION_STEP_TERMS = frozenset(
    {
        "add",
        "build",
        "create",
        "define",
        "document",
        "extend",
        "implement",
        "introduce",
        "modify",
        "refactor",
        "register",
        "test",
        "update",
        "validate",
        "wire",
    }
)


def format_read_only_planning_tools() -> str:
    """Return read-only planning tool names for prompt text."""
    return ", ".join(sorted(READ_ONLY_PLANNING_TOOL_NAMES))


def plan_step_looks_like_reconnaissance(title: str, description: str) -> bool:
    """Return True when a plan step is mostly about looking around."""
    words = _words(f"{title} {description}")
    if not words:
        return False
    if words & IMPLEMENTATION_STEP_TERMS:
        return False
    return bool(words & RECONNAISSANCE_STEP_TERMS)


def plan_looks_like_reconnaissance(steps: list[tuple[str, str]]) -> bool:
    """Return True when too much of a proposed plan is still discovery work."""
    if not steps:
        return False
    reconnaissance_steps = sum(1 for title, description in steps if plan_step_looks_like_reconnaissance(title, description))
    return reconnaissance_steps / len(steps) >= 0.5


def _words(text: str) -> set[str]:
    return {"".join(character for character in word.lower() if character.isalnum()) for word in text.split()}
