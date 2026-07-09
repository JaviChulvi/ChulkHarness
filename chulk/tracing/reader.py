"""Read and export Chulk JSONL traces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from string import Template
from typing import Any


class TraceFormatError(ValueError):
    """Raised when a trace cannot be parsed safely."""


@dataclass(frozen=True)
class TraceRecord:
    """One parsed event and its source line."""

    type: str
    payload: dict[str, Any]
    created_at: str
    line_number: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "payload": self.payload,
            "created_at": self.created_at,
            "line_number": self.line_number,
        }


@dataclass(frozen=True)
class Trace:
    """Inspectable, replay-friendly view of one JSONL trace."""

    path: Path
    events: tuple[TraceRecord, ...]

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "Trace":
        trace_path = Path(path).expanduser().resolve()
        if not trace_path.exists():
            raise TraceFormatError(f"Trace file does not exist: {trace_path}")
        if not trace_path.is_file():
            raise TraceFormatError(f"Trace path is not a file: {trace_path}")

        try:
            trace_text = trace_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise TraceFormatError(f"Trace file is not valid UTF-8: {trace_path}") from exc

        events: list[TraceRecord] = []
        for line_number, raw_line in enumerate(trace_text.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TraceFormatError(f"Invalid JSON on trace line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise TraceFormatError(f"Trace line {line_number} must contain a JSON object")
            event_type = value.get("type")
            payload = value.get("payload", {})
            created_at = value.get("created_at")
            if not isinstance(event_type, str) or not event_type:
                raise TraceFormatError(f"Trace line {line_number} is missing a string event type")
            if not isinstance(payload, dict):
                raise TraceFormatError(f"Trace line {line_number} has a non-object payload")
            if not isinstance(created_at, str):
                raise TraceFormatError(f"Trace line {line_number} is missing created_at")
            events.append(TraceRecord(event_type, payload, created_at, line_number))
        if not events:
            raise TraceFormatError(f"Trace file contains no events: {trace_path}")
        return cls(path=trace_path, events=tuple(events))

    def summary(self) -> dict[str, Any]:
        counts = Counter(event.type for event in self.events)
        final_answer = None
        conversation_id = self.path.stem
        usage: dict[str, Any] | None = None
        for event in self.events:
            if event.type == "final_answer" and isinstance(event.payload.get("content"), str):
                final_answer = event.payload["content"]
            agent_state = event.payload.get("agent_state")
            if isinstance(agent_state, dict) and isinstance(agent_state.get("conversation_id"), str):
                conversation_id = agent_state["conversation_id"]
            if event.type == "turn_finished":
                turn = event.payload.get("turn")
                if isinstance(turn, dict) and isinstance(turn.get("model_usage_totals"), dict):
                    usage = turn["model_usage_totals"]
        failure_types = {"turn_failed", "tool_call_failed", "model_stream_failed"}
        return {
            "path": str(self.path),
            "conversation_id": conversation_id,
            "event_count": len(self.events),
            "turn_count": counts.get("turn_started", 0),
            "started_at": self.events[0].created_at,
            "ended_at": self.events[-1].created_at,
            "event_types": dict(sorted(counts.items())),
            "failure_count": sum(counts.get(event_type, 0) for event_type in failure_types),
            "final_answer": final_answer,
            "usage": usage,
        }

    def to_html(self) -> str:
        """Return a self-contained, escaped HTML trace report."""
        summary = self.summary()
        event_rows = []
        for event in self.events:
            payload = escape(json.dumps(event.payload, indent=2, sort_keys=True, ensure_ascii=False))
            event_rows.append(
                "<details class='event'>"
                f"<summary><span>{escape(event.type)}</span><time>{escape(event.created_at)}</time></summary>"
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
            event_count=str(summary["event_count"]),
            turn_count=str(summary["turn_count"]),
            failure_count=str(summary["failure_count"]),
            final_answer=escape(str(final_answer)),
            event_rows="\n".join(event_rows),
        )


__all__ = ["Trace", "TraceFormatError", "TraceRecord"]
