"""Final-answer reflection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from chulk.core.prompts import REFLECTION_PROMPT
from chulk.core.state import TurnState


MAX_REFLECTION_FIELD_CHARS = 6000


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
        lines.extend(["Active plan:", turn.active_plan.to_prompt()])
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
