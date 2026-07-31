"""Model action parsing for direct answers, plans, and tool calls."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any, Literal

from chulk.core.state import Plan, PlanStep


ACTION_PAYLOAD_FIELDS = frozenset(
    {
        "type",
        "content",
        "tool_name",
        "arguments",
        "arguments_json",
        "plan",
        "plan_json",
        "step_update",
        "step_update_json",
    }
)


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


@dataclass(frozen=True)
class PlanAction:
    """A plan proposed by the model before executing a turn."""

    type: Literal["plan"]
    plan: Plan


@dataclass(frozen=True)
class PlanStepUpdateAction:
    """A model assertion that a plan step is complete or blocked."""

    type: Literal["plan_step_update"]
    step_id: str
    status: Literal["completed", "blocked"]
    evidence: str
    reason: str | None = None


AgentAction = FinalAnswerAction | ToolCallAction | PlanAction | PlanStepUpdateAction

STRICT_AGENT_ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": ["final_answer", "tool_call", "plan", "plan_step_update"],
            "description": "Whether the assistant is answering directly, proposing a plan, updating a plan step, or requesting a tool call.",
        },
        "content": {
            "type": ["string", "null"],
            "description": "Final user-facing answer when type is final_answer; otherwise null.",
        },
        "tool_name": {
            "type": ["string", "null"],
            "description": "Tool name when type is tool_call; otherwise null.",
        },
        "arguments_json": {
            "type": "string",
            "description": (
                "Tool arguments encoded as a JSON object string when type is tool_call; "
                "use {} when type is final_answer, plan, or plan_step_update."
            ),
        },
        "plan_json": {
            "type": "string",
            "description": (
                "Plan encoded as a JSON object string when type is plan; "
                "use {} when type is final_answer, tool_call, or plan_step_update."
            ),
        },
        "step_update_json": {
            "type": "string",
            "description": (
                "Plan step update encoded as a JSON object string when type is plan_step_update; "
                "use {} for all other action types."
            ),
        },
    },
    "required": ["type", "content", "tool_name", "arguments_json", "plan_json", "step_update_json"],
    "additionalProperties": False,
}


def action_json_schema_for(
    allowed_actions: Iterable[str],
) -> dict[str, Any]:
    """Return the strict shared schema narrowed to the legal action types."""
    legal_actions = tuple(
        dict.fromkeys(str(action).strip() for action in allowed_actions)
    )
    supported = tuple(STRICT_AGENT_ACTION_JSON_SCHEMA["properties"]["type"]["enum"])
    if not legal_actions or any(action not in supported for action in legal_actions):
        raise ValueError("allowed_actions must contain supported action types")
    schema = deepcopy(STRICT_AGENT_ACTION_JSON_SCHEMA)
    schema["properties"]["type"]["enum"] = list(legal_actions)
    return schema


def parse_model_response(raw_response: str | dict[str, Any]) -> AgentAction:
    """Parse a model response into a final answer or tool call."""
    payload = _coerce_json_object(raw_response)
    _validate_action_transports(payload)
    action_type = payload.get("type")

    if action_type == "final_answer":
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ActionParseError("final_answer.content must be a non-empty string")
        _reject_irrelevant_fields(
            payload,
            action_type="final_answer",
            reject_tool=True,
            reject_plan=True,
            reject_step_update=True,
        )
        return FinalAnswerAction(type="final_answer", content=content)

    if action_type == "tool_call":
        _reject_nonempty_content(payload, action_type="tool_call")
        _reject_irrelevant_fields(
            payload,
            action_type="tool_call",
            reject_plan=True,
            reject_step_update=True,
        )
        tool_name = payload.get("tool_name")
        arguments = _coerce_tool_arguments(payload)
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ActionParseError("tool_call.tool_name must be a non-empty string")
        return ToolCallAction(type="tool_call", tool_name=tool_name, arguments=arguments)

    if action_type == "plan":
        _reject_nonempty_content(payload, action_type="plan")
        _reject_irrelevant_fields(
            payload,
            action_type="plan",
            reject_tool=True,
            reject_step_update=True,
        )
        return PlanAction(type="plan", plan=_coerce_plan(payload))

    if action_type == "plan_step_update":
        _reject_nonempty_content(payload, action_type="plan_step_update")
        _reject_irrelevant_fields(
            payload,
            action_type="plan_step_update",
            reject_tool=True,
            reject_plan=True,
        )
        return _coerce_plan_step_update(payload)

    raise ActionParseError("model response type must be final_answer, tool_call, plan, or plan_step_update")


def _has_tool_call_fields(payload: dict[str, Any]) -> bool:
    tool_name = payload.get("tool_name")
    if tool_name is not None:
        return True

    arguments = payload.get("arguments")
    if arguments not in (None, {}):
        return True

    raw_arguments_json = payload.get("arguments_json")
    if raw_arguments_json is None:
        return False
    return bool(_decode_json_object_field(payload, "arguments_json"))


def _has_step_update_fields(payload: dict[str, Any]) -> bool:
    step_update = payload.get("step_update")
    if step_update not in (None, {}):
        return True

    raw_step_update_json = payload.get("step_update_json")
    if raw_step_update_json is None:
        return False
    return bool(_decode_json_object_field(payload, "step_update_json"))


def _has_plan_fields(payload: dict[str, Any]) -> bool:
    plan = payload.get("plan")
    if plan not in (None, {}):
        return True

    raw_plan_json = payload.get("plan_json")
    if raw_plan_json is None:
        return False
    return bool(_decode_json_object_field(payload, "plan_json"))


def _reject_nonempty_content(payload: dict[str, Any], *, action_type: str) -> None:
    content = payload.get("content")
    if content not in (None, ""):
        raise ActionParseError(f"{action_type} must not include content")


def _reject_irrelevant_fields(
    payload: dict[str, Any],
    *,
    action_type: str,
    reject_tool: bool = False,
    reject_plan: bool = False,
    reject_step_update: bool = False,
) -> None:
    field_groups = []
    if reject_tool and _has_tool_call_fields(payload):
        field_groups.append("tool call")
    if reject_plan and _has_plan_fields(payload):
        field_groups.append("plan")
    if reject_step_update and _has_step_update_fields(payload):
        field_groups.append("plan step update")
    if field_groups:
        raise ActionParseError(
            f"{action_type} must not include {' or '.join(field_groups)} fields"
        )


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


def _validate_action_transports(payload: dict[str, Any]) -> None:
    unknown_fields = sorted(set(payload) - ACTION_PAYLOAD_FIELDS)
    if unknown_fields:
        raise ActionParseError(
            "model response contains unsupported fields: "
            + ", ".join(unknown_fields)
        )

    content = payload.get("content")
    if content is not None and not isinstance(content, str):
        raise ActionParseError("content must be a string or null")
    tool_name = payload.get("tool_name")
    if tool_name is not None and not isinstance(tool_name, str):
        raise ActionParseError("tool_name must be a string or null")

    for alias, json_field in (
        ("arguments", "arguments_json"),
        ("plan", "plan_json"),
        ("step_update", "step_update_json"),
    ):
        alias_present = alias in payload
        if alias_present and not isinstance(payload[alias], dict):
            raise ActionParseError(f"{alias} must be an object")
        if json_field not in payload:
            continue
        decoded = _decode_json_object_field(payload, json_field)
        if alias_present and decoded:
            raise ActionParseError(
                f"model response must not combine {alias} with non-empty {json_field}"
            )


def _decode_json_object_field(
    payload: dict[str, Any],
    field_name: str,
) -> dict[str, Any]:
    raw_value = payload.get(field_name)
    if not isinstance(raw_value, str):
        raise ActionParseError(f"{field_name} must be a JSON object string")
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ActionParseError(
            f"{field_name} must contain a JSON object"
        ) from exc
    if not isinstance(decoded, dict):
        raise ActionParseError(f"{field_name} must contain a JSON object")
    return decoded


def _coerce_tool_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider-specific argument transports into a dict."""
    if "arguments" in payload:
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ActionParseError("tool_call.arguments must be an object")
        return arguments

    if "arguments_json" not in payload:
        return {}
    return _decode_json_object_field(payload, "arguments_json")


def _coerce_plan_step_update(payload: dict[str, Any]) -> PlanStepUpdateAction:
    """Normalize provider-specific plan step update transports."""
    if "step_update" in payload:
        update_payload = payload.get("step_update")
    else:
        update_payload = (
            _decode_json_object_field(payload, "step_update_json")
            if "step_update_json" in payload
            else {}
        )

    if not isinstance(update_payload, dict):
        raise ActionParseError("plan_step_update payload must be an object")
    unknown_fields = sorted(
        set(update_payload) - {"step_id", "status", "evidence", "reason"}
    )
    if unknown_fields:
        raise ActionParseError(
            "plan_step_update contains unsupported fields: "
            + ", ".join(unknown_fields)
        )

    step_id = update_payload.get("step_id")
    status = update_payload.get("status")
    evidence = update_payload.get("evidence")
    reason = update_payload.get("reason")

    if not isinstance(step_id, str) or not step_id.strip():
        raise ActionParseError("plan_step_update.step_id must be a non-empty string")
    if status not in {"completed", "blocked"}:
        raise ActionParseError("plan_step_update.status must be completed or blocked")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ActionParseError("plan_step_update.evidence must be a non-empty string")
    if reason is not None and not isinstance(reason, str):
        raise ActionParseError("plan_step_update.reason must be a string or null")
    if status == "blocked" and (not isinstance(reason, str) or not reason.strip()):
        raise ActionParseError("plan_step_update.reason must be a non-empty string when status is blocked")

    return PlanStepUpdateAction(
        type="plan_step_update",
        step_id=step_id.strip(),
        status=status,
        evidence=evidence.strip(),
        reason=reason.strip() if isinstance(reason, str) and reason.strip() else None,
    )


def _coerce_plan(payload: dict[str, Any]) -> Plan:
    """Normalize provider-specific plan transports into a Plan object."""
    if "plan" in payload:
        plan_payload = payload.get("plan")
    else:
        plan_payload = (
            _decode_json_object_field(payload, "plan_json")
            if "plan_json" in payload
            else {}
        )

    if not isinstance(plan_payload, dict):
        raise ActionParseError("plan payload must be an object")
    unknown_plan_fields = sorted(set(plan_payload) - {"summary", "steps"})
    if unknown_plan_fields:
        raise ActionParseError(
            "plan contains unsupported fields: " + ", ".join(unknown_plan_fields)
        )

    summary = plan_payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ActionParseError("plan.summary must be a non-empty string")

    raw_steps = plan_payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ActionParseError("plan.steps must be a non-empty list")

    steps: list[PlanStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            raise ActionParseError("each plan step must be an object")

        unknown_step_fields = sorted(
            set(raw_step)
            - {
                "id",
                "title",
                "description",
                "status",
                "depends_on",
                "acceptance_criteria",
                "retry_limit",
            }
        )
        if unknown_step_fields:
            raise ActionParseError(
                "plan step contains unsupported fields: "
                + ", ".join(unknown_step_fields)
            )

        step_id = raw_step.get("id")
        title = raw_step.get("title")
        description = raw_step.get("description")
        status = raw_step.get("status", "pending")
        depends_on = _coerce_string_list(raw_step.get("depends_on", []), field_name="plan step depends_on")
        acceptance_criteria = _coerce_string_list(
            raw_step.get("acceptance_criteria", []),
            field_name="plan step acceptance_criteria",
        )
        retry_limit = _coerce_retry_limit(raw_step.get("retry_limit", 0))

        if not isinstance(step_id, str) or not step_id.strip():
            step_id = str(index)
        if not isinstance(title, str) or not title.strip():
            raise ActionParseError("plan step title must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ActionParseError("plan step description must be a non-empty string")
        if status != "pending":
            raise ActionParseError("proposed plan step status must be pending")

        steps.append(
            PlanStep(
                id=step_id.strip(),
                title=title.strip(),
                description=description.strip(),
                status=status,
                depends_on=depends_on,
                acceptance_criteria=acceptance_criteria,
                retry_limit=retry_limit,
            )
        )

    try:
        return Plan(summary=summary.strip(), steps=steps)
    except ValueError as exc:
        raise ActionParseError(str(exc)) from exc


def _coerce_string_list(value: Any, *, field_name: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ActionParseError(f"{field_name} must be a list of strings")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ActionParseError(f"{field_name} must contain only non-empty strings")
        clean_item = item.strip()
        if clean_item not in result:
            result.append(clean_item)
    return result


def _coerce_retry_limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionParseError("plan step retry_limit must be an integer")
    if value < 0:
        raise ActionParseError("plan step retry_limit cannot be negative")
    return value
