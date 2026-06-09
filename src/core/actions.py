"""Model action parsing for direct answers and tool calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal


class ActionParseError(ValueError):
    """Raised when a model response cannot be parsed into an agent action."""


@dataclass(frozen=True)
class FinalAnswerAction:
    """A direct answer from the model."""

    type: Literal["final_answer"]
    content: str


@dataclass(frozen=True)
class ToolCallAction:
    """A request from the model to call a tool."""

    type: Literal["tool_call"]
    tool_name: str
    arguments: dict[str, Any]


AgentAction = FinalAnswerAction | ToolCallAction


def parse_model_response(raw_response: str | dict[str, Any]) -> AgentAction:
    """Parse a model response into a final answer or tool call."""
    payload = _coerce_json_object(raw_response)
    action_type = payload.get("type")

    if action_type == "final_answer":
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ActionParseError("final_answer.content must be a non-empty string")
        return FinalAnswerAction(type="final_answer", content=content)

    if action_type == "tool_call":
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ActionParseError("tool_call.tool_name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ActionParseError("tool_call.arguments must be an object")
        return ToolCallAction(type="tool_call", tool_name=tool_name, arguments=arguments)

    raise ActionParseError("model response type must be final_answer or tool_call")


def _coerce_json_object(raw_response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        raise ActionParseError("model response must be a JSON object string")

    text = raw_response.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ActionParseError("model response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ActionParseError("model response JSON must be an object")
    return payload
