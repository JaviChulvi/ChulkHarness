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


def local_chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize messages for local chat templates with basic user/assistant support."""
    instruction_parts: list[str] = []
    conversation: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role in {"system", "developer"}:
            if content:
                instruction_parts.append(content)
            continue
        if role == "assistant":
            _append_chat_message(conversation, "assistant", content)
            continue
        if role == "user":
            _append_chat_message(conversation, "user", content)
            continue
        prefix = role or "message"
        _append_chat_message(conversation, "user", f"{prefix}: {content}")

    instructions = "\n\n".join(part for part in instruction_parts if part)
    if instructions:
        instruction_block = f"Instructions:\n{instructions}"
        last_user_index = _last_user_message_index(conversation)
        if last_user_index is None:
            conversation.insert(0, {"role": "user", "content": instruction_block})
        else:
            original_content = conversation[last_user_index]["content"]
            conversation[last_user_index]["content"] = "\n\n".join(
                [
                    instruction_block,
                    "User message:",
                    original_content,
                ]
            )

    if _last_user_message_index(conversation) is None:
        conversation.insert(0, {"role": "user", "content": "Continue from the available context."})
    if conversation[-1]["role"] != "user":
        conversation.append({"role": "user", "content": "Continue."})

    return conversation


def _append_chat_message(messages: list[dict[str, str]], role: str, content: str) -> None:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] = "\n\n".join([messages[-1]["content"], content])
        return
    messages.append({"role": role, "content": content})


def _last_user_message_index(messages: list[dict[str, str]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            return index
    return None
