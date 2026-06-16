"""Model action parsing for direct answers, plans, and tool calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

from chulk.core.state import PLAN_STEP_STATUSES, Plan, PlanStep


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


def parse_model_response(raw_response: str | dict[str, Any]) -> AgentAction:
    """Parse a model response into a final answer or tool call."""
    payload = _coerce_json_object(raw_response)
    action_type = payload.get("type")

    if action_type == "final_answer":
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ActionParseError("final_answer.content must be a non-empty string")
        if _has_tool_call_fields(payload) or _has_step_update_fields(payload):
            raise ActionParseError("final_answer must not include tool call fields or plan step update fields")
        return FinalAnswerAction(type="final_answer", content=content)

    if action_type == "tool_call":
        tool_name = payload.get("tool_name")
        arguments = _coerce_tool_arguments(payload)
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ActionParseError("tool_call.tool_name must be a non-empty string")
        return ToolCallAction(type="tool_call", tool_name=tool_name, arguments=arguments)

    if action_type == "plan":
        return PlanAction(type="plan", plan=_coerce_plan(payload))

    if action_type == "plan_step_update":
        return _coerce_plan_step_update(payload)

    raise ActionParseError("model response type must be final_answer, tool_call, plan, or plan_step_update")


def _has_tool_call_fields(payload: dict[str, Any]) -> bool:
    tool_name = payload.get("tool_name")
    if isinstance(tool_name, str) and tool_name.strip():
        return True

    arguments = payload.get("arguments")
    if arguments not in (None, {}):
        return True

    raw_arguments_json = payload.get("arguments_json")
    if raw_arguments_json in (None, "", "{}"):
        return False
    if not isinstance(raw_arguments_json, str):
        return True

    try:
        arguments = json.loads(raw_arguments_json)
    except json.JSONDecodeError:
        return True
    return bool(arguments)


def _has_step_update_fields(payload: dict[str, Any]) -> bool:
    raw_step_update_json = payload.get("step_update_json")
    if raw_step_update_json in (None, "", "{}"):
        return False
    if not isinstance(raw_step_update_json, str):
        return True

    try:
        step_update = json.loads(raw_step_update_json)
    except json.JSONDecodeError:
        return True
    return bool(step_update)


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


def _coerce_tool_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider-specific argument transports into a dict."""
    if "arguments" in payload:
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise ActionParseError("tool_call.arguments must be an object")
        return arguments

    raw_arguments_json = payload.get("arguments_json", "{}")
    if not isinstance(raw_arguments_json, str):
        raise ActionParseError("tool_call.arguments_json must be a string")

    try:
        arguments = json.loads(raw_arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise ActionParseError("tool_call.arguments_json must contain a JSON object") from exc

    if not isinstance(arguments, dict):
        raise ActionParseError("tool_call.arguments_json must contain a JSON object")
    return arguments


def _coerce_plan_step_update(payload: dict[str, Any]) -> PlanStepUpdateAction:
    """Normalize provider-specific plan step update transports."""
    if "step_update" in payload:
        update_payload = payload.get("step_update")
    else:
        raw_step_update_json = payload.get("step_update_json", "{}")
        if not isinstance(raw_step_update_json, str):
            raise ActionParseError("plan_step_update.step_update_json must be a string")
        try:
            update_payload = json.loads(raw_step_update_json or "{}")
        except json.JSONDecodeError as exc:
            raise ActionParseError("plan_step_update.step_update_json must contain a JSON object") from exc

    if not isinstance(update_payload, dict):
        raise ActionParseError("plan_step_update payload must be an object")

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
        raw_plan_json = payload.get("plan_json", "{}")
        if not isinstance(raw_plan_json, str):
            raise ActionParseError("plan.plan_json must be a string")
        try:
            plan_payload = json.loads(raw_plan_json or "{}")
        except json.JSONDecodeError as exc:
            raise ActionParseError("plan.plan_json must contain a JSON object") from exc

    if not isinstance(plan_payload, dict):
        raise ActionParseError("plan payload must be an object")

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
        if not isinstance(status, str) or status not in PLAN_STEP_STATUSES:
            allowed = ", ".join(sorted(PLAN_STEP_STATUSES))
            raise ActionParseError(f"plan step status must be one of: {allowed}")

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
    return 0
