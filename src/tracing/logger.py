"""Structured tracing utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    """A single structured trace event."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JSONLTraceLogger:
    """Append-only JSONL trace logger."""

    def __init__(self, traces_dir: Path | str, conversation_id: str) -> None:
        self.traces_dir = Path(traces_dir)
        self.conversation_id = conversation_id
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.traces_dir / f"{conversation_id}.jsonl"

    def log(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append a trace event."""
        safe_payload = _redact(payload or {})
        event = TraceEvent(type=event_type, payload=safe_payload).to_dict()
        with self.path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event, sort_keys=True) + "\n")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_secret(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _redact_secret(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in {"api_key", "token", "secret", "password"}):
        return "[redacted]"
    return _redact(value)
