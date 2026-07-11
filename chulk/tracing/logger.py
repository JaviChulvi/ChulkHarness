"""Structured tracing utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from chulk.redaction import redact_data


@dataclass(frozen=True)
class TraceEvent:
    """A single structured trace event."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JSONLTraceLogger:
    """Append-only internal JSONL trace logger, independent of public events."""

    def __init__(
        self,
        traces_dir: Path | str,
        conversation_id: str,
        *,
        defer_until_event: str | None = None,
    ) -> None:
        self.traces_dir = Path(traces_dir)
        self.conversation_id = conversation_id
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.traces_dir / f"{conversation_id}.jsonl"
        self.artifacts_dir = self.traces_dir / f"{conversation_id}_artifacts"
        self._defer_until_event = defer_until_event
        self._active = defer_until_event is None
        self._pending_events: list[dict[str, Any]] = []

    def log(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append a trace event."""
        safe_payload = redact_data(payload or {})
        event = TraceEvent(type=event_type, payload=safe_payload).to_dict()
        if not self._active:
            if event_type != self._defer_until_event:
                self._pending_events.append(event)
                return
            self._active = True
            for pending_event in self._pending_events:
                self._append_event(pending_event)
            self._pending_events.clear()
        self._append_event(event)

    def activate(self) -> None:
        """Flush deferred startup events when runtime work begins."""
        if self._active:
            return
        self._active = True
        for event in self._pending_events:
            self._append_event(event)
        self._pending_events.clear()

    def _append_event(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event, sort_keys=True) + "\n")

    def write_artifact(self, name: str, content: str) -> dict[str, Any]:
        """Persist full trace-adjacent content that is too large for model context."""
        self.activate()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_artifact_name(name)
        path = self.artifacts_dir / f"{safe_name}-{uuid4().hex}.txt"
        path.write_text(content, encoding="utf-8")
        return {
            "path": str(path),
            "char_count": len(content),
            "byte_count": len(content.encode("utf-8")),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }


def _safe_artifact_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip())
    safe = safe.strip(".-")
    return safe or "artifact"
