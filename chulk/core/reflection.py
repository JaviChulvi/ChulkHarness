"""Final-answer reflection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from chulk.core.prompts import REFLECTION_PROMPT
from chulk.core.state import Plan, TurnState


MAX_REFLECTION_FIELD_CHARS = 6000
MAX_REFLECTION_STEP_DESCRIPTION_CHARS = 600
MAX_REFLECTION_CRITERION_CHARS = 400
MAX_REFLECTION_CRITERIA_PER_STEP = 8


class ReflectionParseError(ValueError):
    """Raised when a reflection response cannot be parsed."""


@dataclass(frozen=True)
class ReflectionResult:
    """A bounded review of a proposed final answer."""

    approved: bool
    reason: str
    feedback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "feedback": self.feedback,
        }


def build_reflection_messages(turn: TurnState, proposed_answer: str) -> list[dict[str, str]]:
    """Build the reviewer prompt for a proposed final answer."""
    sections = [
        "User request:",
        _truncate(turn.user_message),
        "",
        "Proposed final answer:",
        _truncate(proposed_answer),
        "",
        "Turn evidence:",
        _format_turn_evidence(turn),
    ]
    return [
        {"role": "system", "content": REFLECTION_PROMPT},
        {"role": "user", "content": "\n".join(sections)},
    ]


def parse_reflection_response(raw_response: str | dict[str, Any]) -> ReflectionResult:
    """Parse a reviewer response into a normalized reflection result."""
    payload = _coerce_json_object(raw_response)
    approved = payload.get("approved")
    if type(approved) is not bool:
        raise ReflectionParseError("reflection.approved must be a boolean")

    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ReflectionParseError("reflection.reason must be a non-empty string")

    raw_feedback = payload.get("feedback")
    feedback = raw_feedback.strip() if isinstance(raw_feedback, str) else None
    if not approved and not feedback:
        raise ReflectionParseError("reflection.feedback must be a non-empty string when approved is false")

    return ReflectionResult(approved=approved, reason=reason.strip(), feedback=feedback)


def _format_turn_evidence(turn: TurnState) -> str:
    lines: list[str] = []
    if turn.active_plan is not None:
        lines.extend(["Active plan:", _format_plan_reference(turn.active_plan)])
    else:
        lines.append("Active plan: none")

    if turn.tool_calls:
        lines.append("Tool calls:")
        for call in turn.tool_calls:
            status = "pending" if call.success is None else ("success" if call.success else "failed")
            error = f"; error={call.error}" if call.error else ""
            lines.append(f"- {call.phase} #{call.iteration} {call.tool_name}: {status}{error}")
    else:
        lines.append("Tool calls: none")

    if turn.observations:
        lines.append("Observations:")
        for observation in turn.observations:
            lines.append(f"- {observation.tool_name}: {_truncate(observation.content, max_chars=1200)}")
    else:
        lines.append("Observations: none")

    if turn.errors:
        lines.append("Errors:")
        lines.extend(f"- {_truncate(error, max_chars=1200)}" for error in turn.errors)
    else:
        lines.append("Errors: none")

    return _truncate("\n".join(lines))


def _format_plan_reference(plan: Plan) -> str:
    """Summarize plan state without copying observation evidence into reflection."""
    lines = [
        f"- status: {plan.status()}",
        f"- summary: {_truncate(plan.summary, max_chars=600)}",
        "- steps:",
    ]
    for step in plan.steps:
        evidence_sources = sorted(
            {record.tool_name or "plan_step_update" for record in step.evidence}
        )
        evidence_reference = f"; evidence_records={len(step.evidence)}"
        if evidence_sources:
            evidence_reference += "; evidence_sources=" + ",".join(evidence_sources)
        lines.append(
            f"  - [{step.status}] {_truncate(step.id, max_chars=160)}: "
            f"{_truncate(step.title, max_chars=400)}{evidence_reference}"
        )
        lines.append(
            "    description: "
            + _truncate(
                step.description,
                max_chars=MAX_REFLECTION_STEP_DESCRIPTION_CHARS,
            )
        )
        criteria = step.acceptance_criteria[:MAX_REFLECTION_CRITERIA_PER_STEP]
        if criteria:
            lines.append("    acceptance_criteria:")
            lines.extend(
                "      - "
                + _truncate(
                    criterion,
                    max_chars=MAX_REFLECTION_CRITERION_CHARS,
                )
                for criterion in criteria
            )
            omitted_count = len(step.acceptance_criteria) - len(criteria)
            if omitted_count:
                lines.append(f"      - ... [{omitted_count} more criteria omitted]")
        else:
            lines.append("    acceptance_criteria: none")
    active_step = plan.active_step()
    lines.append(f"- current_step: {active_step.id if active_step else 'none'}")
    return "\n".join(lines)


def _coerce_json_object(raw_response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        raise ReflectionParseError("reflection response must be a JSON object string")

    text = raw_response.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReflectionParseError("reflection response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ReflectionParseError("reflection response JSON must be an object")
    return payload


def _truncate(text: str, max_chars: int = MAX_REFLECTION_FIELD_CHARS) -> str:
    clean_text = text.strip()
    if len(clean_text) <= max_chars:
        return clean_text
    return clean_text[:max_chars].rstrip() + "\n... [truncated]"
