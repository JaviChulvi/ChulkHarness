"""Provider message-shape helpers."""

from __future__ import annotations


def split_instructions(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Split system/developer instructions from conversational input."""
    instruction_parts: list[str] = []
    response_input: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role in {"system", "developer"}:
            instruction_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
            content = f"{message.get('role', 'unknown')}: {content}"
        response_input.append({"role": role, "content": content})

    return "\n\n".join(part for part in instruction_parts if part), response_input


def chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize messages for OpenAI-compatible chat-completions providers."""
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
            content = f"{message.get('role', 'unknown')}: {content}"
        normalized_messages.append({"role": role, "content": content})
    return normalized_messages
