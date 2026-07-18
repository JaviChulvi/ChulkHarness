"""Provider-native tool-call helpers for LLM clients."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from chulk.core.prompts import JSON_ACTION_PROMPT, format_tools_for_prompt
from chulk.tools.registry import tool_descriptions_for_prompt


PLAN_TOOL_NAME = "chulk_propose_plan"
PLAN_STEP_UPDATE_TOOL_NAME = "chulk_plan_step_update"
JSON_FALLBACK_WRAPPER = "chulk_action_fallback"


@dataclass(frozen=True)
class PlanningToolAvailability:
    """Planning actions exposed through a provider's native tool transport."""

    propose_plan: bool = False
    update_plan_step: bool = False

    @property
    def enabled(self) -> bool:
        """Return whether at least one planning declaration is requested."""
        return self.propose_plan or self.update_plan_step


def provider_action_tools(
    tools: Iterable[object] | None,
    *,
    planning_tools: PlanningToolAvailability | None = None,
) -> list[dict[str, Any]]:
    """Return provider-neutral native tool declarations for Chulk actions."""
    declarations = [_tool_declaration(tool) for tool in tools or []]
    availability = planning_tools or PlanningToolAvailability()
    if availability.propose_plan:
        declarations.append(_plan_tool_declaration())
    if availability.update_plan_step:
        declarations.append(_plan_step_update_tool_declaration())
    return declarations


def openai_response_tools(
    tools: list[object] | None,
    *,
    hosted_mcp_servers: list[object] | tuple[object, ...] | None = None,
    planning_tools: PlanningToolAvailability | None = None,
) -> list[dict[str, Any]]:
    """Return Responses API tool declarations."""
    hosted_mcp_servers = hosted_mcp_servers or []
    chulk_tools = _non_bridge_tools(tools) if hosted_mcp_servers else tools
    declarations = [
        {
            "type": "function",
            "name": declaration["name"],
            "description": declaration["description"],
            "parameters": declaration["parameters"],
        }
        for declaration in provider_action_tools(
            chulk_tools,
            planning_tools=planning_tools,
        )
    ]
    declarations.extend(_hosted_mcp_tool(server) for server in hosted_mcp_servers)
    return declarations


def chat_completion_tools(
    tools: list[object] | None,
    *,
    planning_tools: PlanningToolAvailability | None = None,
) -> list[dict[str, Any]]:
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
        for declaration in provider_action_tools(
            tools,
            planning_tools=planning_tools,
        )
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


def with_json_action_prompt(
    messages: list[dict[str, str]],
    *,
    tools: list[object] | None = None,
) -> list[dict[str, str]]:
    """Return fallback messages with one JSON protocol and a full safe tool catalog."""
    tool_catalog = format_tools_for_prompt(tool_descriptions_for_prompt(tools or []))
    if not messages:
        return [{"role": "system", "content": _json_fallback_prompt("", tool_catalog)}]
    first = messages[0]
    if first.get("role") == "system":
        return [
            {
                **first,
                "content": _json_fallback_prompt(first.get("content", ""), tool_catalog),
            },
            *messages[1:],
        ]
    return [
        {"role": "system", "content": _json_fallback_prompt("", tool_catalog)},
        *messages,
    ]


def _json_fallback_prompt(content: str, tool_catalog: str) -> str:
    if _is_chulk_prompt(content):
        updated, tools_replaced = _replace_xml_section(content, "tools", tool_catalog)
        updated, protocol_replaced = _replace_xml_section(
            updated,
            "action_protocol",
            JSON_ACTION_PROMPT,
        )
        if tools_replaced and protocol_replaced:
            return updated.strip()

    updated_wrapper = _replace_json_fallback_wrapper(content, tool_catalog)
    if updated_wrapper is not None:
        return updated_wrapper.strip()

    wrapper_content = "\n".join(
        [
            _xml_section("tools", tool_catalog),
            _xml_section("action_protocol", JSON_ACTION_PROMPT),
        ]
    )
    wrapper = _xml_section(JSON_FALLBACK_WRAPPER, wrapper_content)
    return "\n\n".join(part for part in [content.strip(), wrapper] if part).strip()


def _is_chulk_prompt(content: str) -> bool:
    stripped = content.strip()
    return stripped.startswith("<chulk_prompt>") and stripped.endswith("</chulk_prompt>")


def _replace_json_fallback_wrapper(content: str, tool_catalog: str) -> str | None:
    start_tag = f"<{JSON_FALLBACK_WRAPPER}>"
    end_tag = f"</{JSON_FALLBACK_WRAPPER}>"
    before, separator, remainder = content.partition(start_tag)
    if not separator:
        return None
    body, end_separator, after = remainder.partition(end_tag)
    if not end_separator:
        return None
    updated, tools_replaced = _replace_xml_section(body, "tools", tool_catalog)
    updated, protocol_replaced = _replace_xml_section(
        updated,
        "action_protocol",
        JSON_ACTION_PROMPT,
    )
    if not tools_replaced or not protocol_replaced:
        return None
    return f"{before}{start_tag}{updated}{end_tag}{after}"


def _replace_xml_section(content: str, name: str, replacement: str) -> tuple[str, bool]:
    start_tag = f"<{name}>"
    end_tag = f"</{name}>"
    before, separator, remainder = content.partition(start_tag)
    if not separator:
        return content, False
    _current, end_separator, after = remainder.partition(end_tag)
    if not end_separator:
        return content, False
    section = _xml_section(name, replacement)
    return f"{before}{section}{after}", True


def _xml_section(name: str, content: str) -> str:
    return f"<{name}>\n{content.strip()}\n</{name}>"


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


def _non_bridge_tools(tools: list[object] | None) -> list[object]:
    return [
        tool
        for tool in tools or []
        if not isinstance(getattr(tool, "metadata", None), dict) or not getattr(tool, "metadata", {}).get("mcp_bridge")
    ]


def _hosted_mcp_tool(server: object) -> dict[str, Any]:
    if hasattr(server, "to_openai_tool"):
        tool = server.to_openai_tool()  # type: ignore[no-any-return, attr-defined]
    elif isinstance(server, dict):
        tool = dict(server)
    else:
        raise TypeError(f"Unsupported hosted MCP server definition: {server!r}")
    if tool.get("type") != "mcp":
        raise ValueError("Hosted MCP tool definitions must have type 'mcp'")
    return tool


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
