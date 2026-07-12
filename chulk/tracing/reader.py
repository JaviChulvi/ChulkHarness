"""Read, normalize, replay, and export Chulk JSONL traces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from html import escape
import json
from pathlib import Path
from string import Template
from typing import Any

from chulk.errors import ErrorDetails, TraceError
from chulk.tracing.logger import TRACE_SCHEMA_VERSION


LEGACY_TRACE_SCHEMA_VERSION = 0
SUPPORTED_TRACE_SCHEMA_VERSIONS = frozenset({LEGACY_TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION})
_FAILURE_EVENT_TYPES = {"turn_failed", "tool_call_failed", "model_stream_failed"}


class TraceFormatError(TraceError, ValueError):
    """Raised when a trace cannot be parsed safely."""

    def __init__(self, message: str, *, trace_path: Path | str | None = None) -> None:
        super().__init__(
            message,
            details=ErrorDetails(trace_path=str(trace_path) if trace_path is not None else None),
        )


@dataclass(frozen=True)
class TraceRecord:
    """One normalized event and its source line."""

    schema_version: int
    conversation_id: str
    type: str
    payload: dict[str, Any]
    timestamp: str
    line_number: int
    turn_id: str | None = None

    @property
    def created_at(self) -> str:
        """Backward-compatible alias for the v1 ``timestamp`` field."""
        return self.timestamp

    def to_dict(self) -> dict[str, Any]:
        event: dict[str, Any] = {
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "line_number": self.line_number,
        }
        if self.turn_id is not None:
            event["turn_id"] = self.turn_id
        return event


@dataclass(frozen=True)
class Trace:
    """Inspectable, replay-friendly view of one JSONL trace."""

    path: Path
    events: tuple[TraceRecord, ...]

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "Trace":
        trace_path = Path(path).expanduser().resolve()
        if not trace_path.exists():
            raise TraceFormatError(f"Trace file does not exist: {trace_path}", trace_path=trace_path)
        if not trace_path.is_file():
            raise TraceFormatError(f"Trace path is not a file: {trace_path}", trace_path=trace_path)

        try:
            trace_text = trace_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise TraceFormatError(
                f"Trace file is not valid UTF-8: {trace_path}",
                trace_path=trace_path,
            ) from exc

        events: list[TraceRecord] = []
        active_turn_id: str | None = None
        versioned_conversation_id: str | None = None
        for line_number, raw_line in enumerate(trace_text.splitlines(), start=1):
            if not raw_line.strip():
                continue
            value = _parse_json_line(raw_line, line_number=line_number, trace_path=trace_path)
            schema_version = _schema_version(value, line_number=line_number, trace_path=trace_path)
            event_type = _event_type(value, line_number=line_number, trace_path=trace_path)
            payload = _event_payload(value, line_number=line_number, trace_path=trace_path)
            timestamp = _event_timestamp(
                value,
                schema_version=schema_version,
                line_number=line_number,
                trace_path=trace_path,
            )
            conversation_id = _event_conversation_id(
                value,
                payload,
                schema_version=schema_version,
                default=trace_path.stem,
                line_number=line_number,
                trace_path=trace_path,
            )
            if schema_version > LEGACY_TRACE_SCHEMA_VERSION:
                if versioned_conversation_id is None:
                    versioned_conversation_id = conversation_id
                elif conversation_id != versioned_conversation_id:
                    raise TraceFormatError(
                        f"Trace line {line_number} changes conversation_id from "
                        f"{versioned_conversation_id!r} to {conversation_id!r}",
                        trace_path=trace_path,
                    )

            turn_id = _event_turn_id(
                value,
                payload,
                schema_version=schema_version,
                active_turn_id=active_turn_id,
                line_number=line_number,
                trace_path=trace_path,
            )
            if event_type == "turn_started" and turn_id is not None:
                active_turn_id = turn_id
            events.append(
                TraceRecord(
                    schema_version=schema_version,
                    conversation_id=conversation_id,
                    type=event_type,
                    payload=payload,
                    timestamp=timestamp,
                    line_number=line_number,
                    turn_id=turn_id,
                )
            )
            if event_type == "turn_finished" and turn_id == active_turn_id:
                active_turn_id = None

        if not events:
            raise TraceFormatError(f"Trace file contains no events: {trace_path}", trace_path=trace_path)
        return cls(path=trace_path, events=tuple(events))

    def summary(self) -> dict[str, Any]:
        counts = Counter(event.type for event in self.events)
        final_answer = None
        conversation_id = _trace_conversation_id(self.path, self.events)
        usage: dict[str, Any] | None = None
        for event in self.events:
            if event.type == "final_answer" and isinstance(event.payload.get("content"), str):
                final_answer = event.payload["content"]
            if event.type == "turn_finished":
                turn = event.payload.get("turn")
                if isinstance(turn, dict) and isinstance(turn.get("model_usage_totals"), dict):
                    usage = turn["model_usage_totals"]
        schema_versions = sorted({event.schema_version for event in self.events})
        return {
            "path": str(self.path),
            "conversation_id": conversation_id,
            "schema_versions": schema_versions,
            "legacy_event_count": sum(
                event.schema_version == LEGACY_TRACE_SCHEMA_VERSION for event in self.events
            ),
            "event_count": len(self.events),
            "session_count": counts.get("session_started", 0),
            "turn_count": counts.get("turn_started", 0),
            "started_at": self.events[0].timestamp,
            "ended_at": self.events[-1].timestamp,
            "duration_ms": _elapsed_ms(self.events[0].timestamp, self.events[-1].timestamp),
            "event_types": dict(sorted(counts.items())),
            "failure_count": sum(counts.get(event_type, 0) for event_type in _FAILURE_EVENT_TYPES),
            "final_answer": final_answer,
            "usage": usage,
        }

    def replay(self) -> dict[str, Any]:
        """Deterministically reconstruct a trace without executing runtime work."""
        turns: list[dict[str, Any]] = []
        turns_by_id: dict[str, dict[str, Any]] = {}
        sessions: list[dict[str, Any]] = []
        active_session: dict[str, Any] | None = None

        for event in self.events:
            if event.type == "session_started":
                active_session = {
                    "started_at": event.timestamp,
                    "ended_at": None,
                    "duration_ms": None,
                    "resumed": bool(event.payload.get("resumed", False)),
                    "explicit": True,
                }
                sessions.append(active_session)
            elif event.type == "session_finished":
                if active_session is None:
                    active_session = {
                        "started_at": None,
                        "ended_at": event.timestamp,
                        "duration_ms": _number_or_none(event.payload.get("duration_ms")),
                        "resumed": False,
                        "explicit": True,
                    }
                    sessions.append(active_session)
                else:
                    active_session["ended_at"] = event.timestamp
                    active_session["duration_ms"] = _number_or_none(
                        event.payload.get("duration_ms")
                    )
                    active_session = None

            turn = _replay_turn_for_event(event, turns=turns, turns_by_id=turns_by_id)
            if turn is None:
                continue
            _apply_replay_event(turn, event)

        if not sessions:
            sessions.append(
                {
                    "started_at": self.events[0].timestamp,
                    "ended_at": self.events[-1].timestamp,
                    "duration_ms": _elapsed_ms(
                        self.events[0].timestamp,
                        self.events[-1].timestamp,
                    ),
                    "resumed": False,
                    "explicit": False,
                }
            )

        for turn in turns:
            if turn["status"] == "unknown":
                if turn["failures"]:
                    turn["status"] = "failed"
                elif turn["final_answer"] is not None:
                    turn["status"] = "completed"

        return {
            "path": str(self.path),
            "conversation_id": _trace_conversation_id(self.path, self.events),
            "mode": "read_only",
            "executed": False,
            "schema_versions": sorted({event.schema_version for event in self.events}),
            "event_count": len(self.events),
            "session_count": len(sessions),
            "turn_count": len(turns),
            "failure_count": sum(len(turn["failures"]) for turn in turns),
            "sessions": sessions,
            "turns": turns,
        }

    def to_html(self) -> str:
        """Return a self-contained, escaped HTML trace report."""
        summary = self.summary()
        event_rows = []
        for event in self.events:
            payload = escape(json.dumps(event.payload, indent=2, sort_keys=True, ensure_ascii=False))
            event_rows.append(
                "<details class='event'>"
                f"<summary><span>{escape(event.type)}</span><time>{escape(event.timestamp)}</time></summary>"
                f"<pre>{payload}</pre>"
                "</details>"
            )
        final_answer = summary.get("final_answer") or "No final answer recorded."
        template = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chulk trace report</title>
  <style>
    :root { color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    body { max-width: 1100px; margin: 0 auto; padding: 32px 20px; background: #0b100c; color: #e8f5ea; }
    h1, h2 { color: #3fff51; }
    .notice { color: #a9b8ac; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    .card, .event { border: 1px solid #315438; border-radius: 8px; background: #111a13; }
    .card { padding: 14px; }
    .card strong { display: block; margin-top: 6px; overflow-wrap: anywhere; }
    .event { margin: 10px 0; }
    summary { display: flex; justify-content: space-between; gap: 16px; padding: 12px; cursor: pointer; }
    time { color: #a9b8ac; font-size: 0.85rem; }
    pre { margin: 0; padding: 14px; overflow: auto; border-top: 1px solid #315438; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Chulk trace</h1>
  <p class="notice">Trace exports may contain sensitive runtime data. Handle this file accordingly.</p>
  <section class="grid">
    <div class="card">Conversation<strong>$conversation</strong></div>
    <div class="card">Schema versions<strong>$schema_versions</strong></div>
    <div class="card">Events<strong>$event_count</strong></div>
    <div class="card">Turns<strong>$turn_count</strong></div>
    <div class="card">Failures<strong>$failure_count</strong></div>
  </section>
  <h2>Final answer</h2>
  <pre>$final_answer</pre>
  <h2>Events</h2>
  $event_rows
</body>
</html>
""")
        return template.substitute(
            conversation=escape(str(summary["conversation_id"])),
            schema_versions=escape(", ".join(str(item) for item in summary["schema_versions"])),
            event_count=str(summary["event_count"]),
            turn_count=str(summary["turn_count"]),
            failure_count=str(summary["failure_count"]),
            final_answer=escape(str(final_answer)),
            event_rows="\n".join(event_rows),
        )


def _parse_json_line(raw_line: str, *, line_number: int, trace_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise TraceFormatError(
            f"Invalid JSON on trace line {line_number}: {exc.msg}",
            trace_path=trace_path,
        ) from exc
    if not isinstance(value, dict):
        raise TraceFormatError(
            f"Trace line {line_number} must contain a JSON object",
            trace_path=trace_path,
        )
    return value


def _schema_version(value: dict[str, Any], *, line_number: int, trace_path: Path) -> int:
    raw_version = value.get("schema_version", LEGACY_TRACE_SCHEMA_VERSION)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise TraceFormatError(
            f"Trace line {line_number} has a non-integer schema_version",
            trace_path=trace_path,
        )
    if raw_version not in SUPPORTED_TRACE_SCHEMA_VERSIONS:
        raise TraceFormatError(
            f"Trace line {line_number} uses unsupported schema_version {raw_version}",
            trace_path=trace_path,
        )
    return raw_version


def _event_type(value: dict[str, Any], *, line_number: int, trace_path: Path) -> str:
    event_type = value.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise TraceFormatError(
            f"Trace line {line_number} is missing a string event type",
            trace_path=trace_path,
        )
    return event_type


def _event_payload(
    value: dict[str, Any],
    *,
    line_number: int,
    trace_path: Path,
) -> dict[str, Any]:
    payload = value.get("payload", {})
    if not isinstance(payload, dict):
        raise TraceFormatError(
            f"Trace line {line_number} has a non-object payload",
            trace_path=trace_path,
        )
    return payload


def _event_timestamp(
    value: dict[str, Any],
    *,
    schema_version: int,
    line_number: int,
    trace_path: Path,
) -> str:
    field_name = "created_at" if schema_version == LEGACY_TRACE_SCHEMA_VERSION else "timestamp"
    timestamp = value.get(field_name)
    if not isinstance(timestamp, str) or not timestamp:
        raise TraceFormatError(
            f"Trace line {line_number} is missing {field_name}",
            trace_path=trace_path,
        )
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TraceFormatError(
            f"Trace line {line_number} has an invalid {field_name}",
            trace_path=trace_path,
        ) from exc
    return timestamp


def _event_conversation_id(
    value: dict[str, Any],
    payload: dict[str, Any],
    *,
    schema_version: int,
    default: str,
    line_number: int,
    trace_path: Path,
) -> str:
    if schema_version > LEGACY_TRACE_SCHEMA_VERSION:
        conversation_id = value.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise TraceFormatError(
                f"Trace line {line_number} is missing conversation_id",
                trace_path=trace_path,
            )
        return conversation_id
    return _payload_conversation_id(payload) or default


def _event_turn_id(
    value: dict[str, Any],
    payload: dict[str, Any],
    *,
    schema_version: int,
    active_turn_id: str | None,
    line_number: int,
    trace_path: Path,
) -> str | None:
    if schema_version > LEGACY_TRACE_SCHEMA_VERSION and "turn_id" in value:
        raw_turn_id = value["turn_id"]
        if raw_turn_id is not None and (not isinstance(raw_turn_id, str) or not raw_turn_id):
            raise TraceFormatError(
                f"Trace line {line_number} has an invalid turn_id",
                trace_path=trace_path,
            )
        if isinstance(raw_turn_id, str):
            return raw_turn_id
    return _payload_turn_id(payload) or active_turn_id


def _payload_conversation_id(payload: dict[str, Any]) -> str | None:
    direct = payload.get("conversation_id")
    if isinstance(direct, str) and direct:
        return direct
    agent_state = payload.get("agent_state")
    if isinstance(agent_state, dict):
        nested = agent_state.get("conversation_id")
        if isinstance(nested, str) and nested:
            return nested
    return None


def _payload_turn_id(payload: dict[str, Any]) -> str | None:
    direct = payload.get("turn_id")
    if isinstance(direct, str) and direct:
        return direct
    turn = payload.get("turn")
    if isinstance(turn, dict):
        nested = turn.get("turn_id")
        if isinstance(nested, str) and nested:
            return nested
    return None


def _trace_conversation_id(path: Path, events: tuple[TraceRecord, ...]) -> str:
    for event in events:
        if event.schema_version > LEGACY_TRACE_SCHEMA_VERSION:
            return event.conversation_id
    for event in reversed(events):
        inferred = _payload_conversation_id(event.payload)
        if inferred is not None:
            return inferred
    return path.stem


def _replay_turn_for_event(
    event: TraceRecord,
    *,
    turns: list[dict[str, Any]],
    turns_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if event.turn_id is None:
        return None
    turn = turns_by_id.get(event.turn_id)
    if turn is not None:
        return turn
    turn = {
        "turn_id": event.turn_id,
        "status": "unknown",
        "user_message": None,
        "final_answer": None,
        "started_at": event.timestamp if event.type == "turn_started" else None,
        "ended_at": None,
        "duration_ms": None,
        "model_request_count": 0,
        "tool_calls": [],
        "failures": [],
    }
    turns.append(turn)
    turns_by_id[event.turn_id] = turn
    return turn


def _apply_replay_event(turn: dict[str, Any], event: TraceRecord) -> None:
    payload = event.payload
    snapshot = payload.get("turn")
    if event.type == "turn_started":
        turn["started_at"] = event.timestamp
        if isinstance(snapshot, dict):
            _apply_turn_snapshot(turn, snapshot)
    elif event.type == "user_message" and isinstance(payload.get("content"), str):
        turn["user_message"] = payload["content"]
    elif event.type == "model_request_started":
        turn["model_request_count"] += 1
    elif event.type == "tool_call_started":
        turn["tool_calls"].append(_tool_summary(payload, status="started"))
    elif event.type in {"tool_call_completed", "tool_call_failed"}:
        status = "completed" if event.type == "tool_call_completed" else "failed"
        replacement = _tool_summary(payload, status=status)
        match = _matching_started_tool(turn["tool_calls"], replacement)
        if match is None:
            turn["tool_calls"].append(replacement)
        else:
            match.update(replacement)
        if event.type == "tool_call_failed":
            turn["failures"].append(str(payload.get("error") or event.type))
    elif event.type == "final_answer" and isinstance(payload.get("content"), str):
        turn["final_answer"] = payload["content"]
    elif event.type in _FAILURE_EVENT_TYPES:
        message = payload.get("message") or payload.get("error") or event.type
        turn["failures"].append(str(message))
    elif event.type == "turn_finished":
        turn["ended_at"] = event.timestamp
        if isinstance(snapshot, dict):
            _apply_turn_snapshot(turn, snapshot)
        timing = payload.get("timing")
        if isinstance(timing, dict):
            turn["duration_ms"] = _number_or_none(timing.get("duration_ms"))
        if turn["duration_ms"] is None and turn["started_at"] is not None:
            turn["duration_ms"] = _elapsed_ms(turn["started_at"], event.timestamp)


def _apply_turn_snapshot(turn: dict[str, Any], snapshot: dict[str, Any]) -> None:
    if isinstance(snapshot.get("status"), str):
        turn["status"] = snapshot["status"]
    if isinstance(snapshot.get("user_message"), str):
        turn["user_message"] = snapshot["user_message"]
    if isinstance(snapshot.get("final_answer"), str):
        turn["final_answer"] = snapshot["final_answer"]
    model_request_count = snapshot.get("model_request_count")
    if isinstance(model_request_count, int) and not isinstance(model_request_count, bool):
        turn["model_request_count"] = model_request_count
    tool_calls = snapshot.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        turn["tool_calls"] = [
            _tool_summary(item, status=_tool_snapshot_status(item))
            for item in tool_calls
            if isinstance(item, dict)
        ]
    errors = snapshot.get("errors")
    if isinstance(errors, list):
        for item in errors:
            message = str(item)
            if message not in turn["failures"]:
                turn["failures"].append(message)


def _tool_summary(payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    summary = {
        "tool_name": payload.get("tool_name") if isinstance(payload.get("tool_name"), str) else None,
        "iteration": payload.get("iteration") if isinstance(payload.get("iteration"), int) else None,
        "status": status,
        "success": payload.get("success") if isinstance(payload.get("success"), bool) else None,
        "error": payload.get("error") if isinstance(payload.get("error"), str) else None,
        "failure_kind": (
            payload.get("failure_kind") if isinstance(payload.get("failure_kind"), str) else None
        ),
    }
    if summary["success"] is None and status in {"completed", "failed"}:
        summary["success"] = status == "completed"
    return summary


def _tool_snapshot_status(payload: dict[str, Any]) -> str:
    success = payload.get("success")
    if success is True:
        return "completed"
    if success is False:
        return "failed"
    return "started"


def _matching_started_tool(
    tool_calls: list[dict[str, Any]],
    replacement: dict[str, Any],
) -> dict[str, Any] | None:
    for tool_call in reversed(tool_calls):
        if tool_call.get("status") != "started":
            continue
        if tool_call.get("tool_name") != replacement.get("tool_name"):
            continue
        if (
            replacement.get("iteration") is not None
            and tool_call.get("iteration") != replacement.get("iteration")
        ):
            continue
        return tool_call
    return None


def _number_or_none(value: object) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _elapsed_ms(started_at: str, ended_at: str) -> float | None:
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        return round(max(0.0, (end - start).total_seconds() * 1000), 3)
    except (TypeError, ValueError):
        return None


__all__ = [
    "LEGACY_TRACE_SCHEMA_VERSION",
    "SUPPORTED_TRACE_SCHEMA_VERSIONS",
    "Trace",
    "TraceFormatError",
    "TraceRecord",
]
