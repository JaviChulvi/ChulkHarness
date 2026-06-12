"""Trace payload formatting helpers."""

from __future__ import annotations

from src.core.actions import FinalAnswerAction, PlanAction, ToolCallAction


def format_action_trace(action: FinalAnswerAction | PlanAction | ToolCallAction) -> dict:
    """Return a trace-safe payload for a parsed action."""
    if isinstance(action, FinalAnswerAction):
        return {"type": action.type}
    if isinstance(action, PlanAction):
        return {"type": action.type, "plan": action.plan.to_dict()}
    return {"type": action.type, "tool_name": action.tool_name, "arguments": action.arguments}


def format_model_request_trace(
    messages: list[dict[str, str]],
    *,
    max_prompt_chars: int,
    request_index: int,
    turn_id: str | None = None,
    loaded_memory_ids: list[str],
    loaded_skill_names: list[str],
    available_tool_names: list[str],
    context_report: dict | None = None,
    max_output_tokens: int | None = None,
) -> dict:
    """Return a bounded, redaction-ready trace payload for one model request."""
    prompt_char_count = sum(len(str(message.get("content", ""))) for message in messages)
    remaining_chars = max_prompt_chars
    traced_messages = []
    returned_char_count = 0

    for message in messages:
        content = str(message.get("content", ""))
        content_char_count = len(content)
        returned_content = content[:remaining_chars] if remaining_chars > 0 else ""
        remaining_chars = max(0, remaining_chars - len(returned_content))
        returned_char_count += len(returned_content)
        traced_messages.append(
            {
                "role": str(message.get("role", "")),
                "content": returned_content,
                "content_char_count": content_char_count,
                "returned_content_char_count": len(returned_content),
                "truncated": len(returned_content) < content_char_count,
            }
        )

    return {
        "turn_id": turn_id,
        "request_index": request_index,
        "messages": traced_messages,
        "message_count": len(messages),
        "prompt_char_count": prompt_char_count,
        "returned_prompt_char_count": returned_char_count,
        "max_prompt_chars": max_prompt_chars,
        "truncated": returned_char_count < prompt_char_count,
        "loaded_memory_ids": loaded_memory_ids,
        "loaded_skill_names": loaded_skill_names,
        "available_tool_names": available_tool_names,
        "context_report": context_report or {},
        "max_output_tokens": max_output_tokens,
    }
