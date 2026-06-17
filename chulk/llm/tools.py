"""Provider-native tool-call helpers for LLM clients."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from chulk.core.prompts import JSON_ACTION_PROMPT


PLAN_TOOL_NAME = "chulk_propose_plan"
PLAN_STEP_UPDATE_TOOL_NAME = "chulk_plan_step_update"


def provider_action_tools(tools: list[object] | None) -> list[dict[str, Any]]:
    """Return provider-neutral native tool declarations for Chulk actions."""
    declarations = [_tool_declaration(tool) for tool in tools or []]
    declarations.append(_plan_tool_declaration())
    declarations.append(_plan_step_update_tool_declaration())
    return declarations


def openai_response_tools(tools: list[object] | None) -> list[dict[str, Any]]:
    """Return Responses API tool declarations."""
    return [
        {
            "type": "function",
            "name": declaration["name"],
            "description": declaration["description"],
            "parameters": declaration["parameters"],
        }
        for declaration in provider_action_tools(tools)
    ]


def chat_completion_tools(tools: list[object] | None) -> list[dict[str, Any]]:
    """Return OpenAI-compatible chat-completion tool declarations."""
    return [
        {
            "type": "function",
            "function": {
                "name": declaration["name"],
                "description": declaration["description"],
                "parameters": declaration["parameters"],
            },
        }
        for declaration in provider_action_tools(tools)
    ]


def native_tool_action_payload(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize a native provider function call into Chulk action JSON."""
    if name == PLAN_TOOL_NAME:
        return {
            "type": "plan",
            "content": None,
            "tool_name": None,
            "arguments_json": "{}",
            "plan": arguments,
            "step_update_json": "{}",
        }
    if name == PLAN_STEP_UPDATE_TOOL_NAME:
        return {
            "type": "plan_step_update",
            "content": None,
            "tool_name": None,
            "arguments_json": "{}",
            "plan_json": "{}",
            "step_update": arguments,
        }
    return {
        "type": "tool_call",
        "content": None,
        "tool_name": name,
        "arguments": arguments,
        "plan_json": "{}",
        "step_update_json": "{}",
    }


def native_final_answer_payload(content: str) -> dict[str, Any]:
    """Normalize provider text into a Chulk final-answer action."""
    return {
        "type": "final_answer",
        "content": content,
        "tool_name": None,
        "arguments_json": "{}",
        "plan_json": "{}",
        "step_update_json": "{}",
    }


def parse_native_arguments(raw_arguments: object) -> dict[str, Any]:
    """Parse native tool arguments from provider-specific transports."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if raw_arguments in (None, ""):
        return {}
    if not isinstance(raw_arguments, str):
        raise ValueError("native tool arguments must be a JSON object or JSON object string")
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("native tool arguments were not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("native tool arguments must decode to a JSON object")
    return parsed


def action_payload_json(payload: dict[str, Any]) -> str:
    """Serialize a normalized action payload for the shared action parser."""
    return json.dumps(payload, sort_keys=True)


def with_json_action_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return messages with the JSON action protocol appended for fallback calls."""
    if not messages:
        return [{"role": "system", "content": JSON_ACTION_PROMPT}]
    first = messages[0]
    if first.get("role") == "system":
        return [
            {
                **first,
                "content": "\n\n".join([first.get("content", ""), JSON_ACTION_PROMPT]).strip(),
            },
            *messages[1:],
        ]
    return [{"role": "system", "content": JSON_ACTION_PROMPT}, *messages]


def public_value(value: object) -> Any:
    """Return JSON-safe provider metadata for traces."""
    if isinstance(value, dict):
        return {str(key): public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        try:
            return public_value(value.model_dump())
        except Exception:
            return str(value)
    result: dict[str, Any] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if callable(item):
            continue
        if isinstance(item, (str, int, float, bool, type(None), dict, list)):
            result[key] = public_value(item)
    return result or str(value)


def _tool_declaration(tool: object) -> dict[str, Any]:
    return {
        "name": str(getattr(tool, "name")),
        "description": str(getattr(tool, "description", "")),
        "parameters": deepcopy(getattr(tool, "args_schema", {}) or {}),
    }


def _plan_tool_declaration() -> dict[str, Any]:
    return {
        "name": PLAN_TOOL_NAME,
        "description": "Propose the approval plan requested by the user before executing implementation work.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "blocked"],
                            },
                            "depends_on": {"type": "array", "items": {"type": "string"}},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                            "retry_limit": {"type": "integer", "minimum": 0},
                        },
                        "required": [
                            "id",
                            "title",
                            "description",
                            "status",
                            "depends_on",
                            "acceptance_criteria",
                            "retry_limit",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["summary", "steps"],
            "additionalProperties": False,
        },
    }


def _plan_step_update_tool_declaration() -> dict[str, Any]:
    return {
        "name": PLAN_STEP_UPDATE_TOOL_NAME,
        "description": "Mark the current approved plan step completed or blocked with evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_id": {"type": "string"},
                "status": {"type": "string", "enum": ["completed", "blocked"]},
                "evidence": {"type": "string"},
                "reason": {"type": ["string", "null"]},
            },
            "required": ["step_id", "status", "evidence", "reason"],
            "additionalProperties": False,
        },
    }
