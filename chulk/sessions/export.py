"""Human-readable and machine-readable session transcript export."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from chulk.core.state import TurnState
from chulk.sessions.models import ConversationRecord, ConversationSummaryRecord
from chulk.sessions.sqlite_store import SQLiteSessionStore


EXPORT_FORMATS = ("md", "json")
DEFAULT_EXPORT_FORMAT = "md"


def normalize_export_format(value: str | None) -> str:
    """Validate and normalize a requested transcript export format."""
    fmt = (value or DEFAULT_EXPORT_FORMAT).strip().lower()
    if fmt not in EXPORT_FORMATS:
        supported = ", ".join(EXPORT_FORMATS)
        raise ValueError(f"Unsupported export format: {fmt!r} (expected one of: {supported})")
    return fmt


def default_export_path(runtime_dir: Path, conversation_id: str, fmt: str) -> Path:
    """Return the default `.chulk/exports/<id>-<timestamp>.<fmt>` export path."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(runtime_dir) / "exports" / f"{conversation_id}-{timestamp}.{fmt}"


def export_session(
    store: SQLiteSessionStore,
    session_id: str,
    *,
    runtime_dir: Path,
    format: str = DEFAULT_EXPORT_FORMAT,
    output_path: Path | str | None = None,
) -> Path:
    """Render a session transcript and write it to disk. Returns the written path.

    ``session_id`` may be a full conversation id or a unique prefix; lookup
    failures raise the same ``SessionNotFoundError`` / ``AmbiguousSessionError``
    used by session resume.
    """
    fmt = normalize_export_format(format)
    record = store.get_conversation(session_id)
    summary = store.load_latest_summary(record.id)
    turns = store.load_turns(record.id)

    if fmt == "json":
        content = json.dumps(render_json_transcript(record, turns, summary), indent=2, sort_keys=True)
    else:
        content = render_markdown_transcript(record, turns, summary)

    path = Path(output_path) if output_path is not None else default_export_path(runtime_dir, record.id, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def render_json_transcript(
    record: ConversationRecord,
    turns: list[TurnState],
    summary: ConversationSummaryRecord | None,
) -> dict[str, Any]:
    """Return a structured `{"session": {...}, "messages": [...]}` transcript."""
    messages: list[dict[str, Any]] = []
    for turn in turns:
        messages.extend(_turn_events(turn))
    return {
        "session": _session_header(record, summary),
        "messages": messages,
    }


def render_markdown_transcript(
    record: ConversationRecord,
    turns: list[TurnState],
    summary: ConversationSummaryRecord | None,
) -> str:
    """Return a self-contained Markdown transcript for one session."""
    lines = [
        f"# Session {record.id}",
        "",
        f"- **Title:** {record.title or '(untitled)'}",
        f"- **Status:** {record.status}",
        f"- **Provider / Model:** {record.provider} / {record.model}",
        f"- **Created:** {record.created_at}",
        f"- **Updated:** {record.updated_at}",
        f"- **Turns:** {record.turn_count}",
        "",
    ]
    if summary is not None:
        lines.extend(["## Summary", "", summary.content, ""])

    lines.append("## Transcript")
    lines.append("")
    if not turns:
        lines.append("_No turns recorded._")
        return "\n".join(lines).rstrip() + "\n"

    for index, turn in enumerate(turns, start=1):
        lines.append(f"### Turn {index} — {turn.status}")
        lines.append("")
        lines.append(f"**User:** {turn.user_message}")
        lines.append("")
        for tool_call, observation in zip(turn.tool_calls, turn.observations):
            tool_name = tool_call.resolved_tool_name or tool_call.tool_name
            outcome = "ok" if tool_call.success else "failed"
            lines.append(f"**Tool call:** `{tool_name}` ({outcome})")
            lines.append(f"- arguments: `{json.dumps(tool_call.arguments, sort_keys=True)}`")
            lines.append("- result:")
            lines.extend(_fenced_block(observation.content, indent="  "))
            if tool_call.error:
                lines.append(f"- error: {tool_call.error}")
            lines.append("")
        if turn.active_plan is not None:
            lines.append(f"**Plan proposed:** {turn.active_plan.summary}")
            for step in turn.active_plan.steps:
                lines.append(f"  - [{step.status}] {step.title}: {step.description}")
            lines.append("")
        if turn.errors:
            lines.append("**Errors:**")
            lines.extend(f"- {error}" for error in turn.errors)
            lines.append("")
        if turn.final_answer:
            lines.append(f"**Assistant:** {turn.final_answer}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _session_header(record: ConversationRecord, summary: ConversationSummaryRecord | None) -> dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "status": record.status,
        "provider": record.provider,
        "model": record.model,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "turn_count": record.turn_count,
        "summary": (
            {"content": summary.content, "source_message_count": summary.source_message_count}
            if summary is not None
            else None
        ),
    }


def _turn_events(turn: TurnState) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if turn.user_message:
        events.append({"turn_id": turn.turn_id, "role": "user", "content": turn.user_message})
    for tool_call, observation in zip(turn.tool_calls, turn.observations):
        events.append(
            {
                "turn_id": turn.turn_id,
                "role": "tool_call",
                "tool_name": tool_call.resolved_tool_name or tool_call.tool_name,
                "arguments": tool_call.arguments,
                "success": tool_call.success,
                "error": tool_call.error,
                "result": observation.content,
            }
        )
    if turn.active_plan is not None:
        events.append(
            {
                "turn_id": turn.turn_id,
                "role": "plan",
                "summary": turn.active_plan.summary,
                "steps": [
                    {"title": step.title, "status": step.status, "description": step.description}
                    for step in turn.active_plan.steps
                ],
            }
        )
    if turn.final_answer:
        events.append({"turn_id": turn.turn_id, "role": "assistant", "content": turn.final_answer})
    return events


def _fenced_block(content: str, *, indent: str) -> list[str]:
    body_lines = content.splitlines() or [""]
    return [f"{indent}```", *[f"{indent}{line}" for line in body_lines], f"{indent}```"]
