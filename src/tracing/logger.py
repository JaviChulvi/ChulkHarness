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
        self.artifacts_dir = self.traces_dir / f"{conversation_id}_artifacts"

    def log(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append a trace event."""
        safe_payload = _redact(payload or {})
        event = TraceEvent(type=event_type, payload=safe_payload).to_dict()
        with self.path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event, sort_keys=True) + "\n")

    def write_artifact(self, name: str, content: str) -> dict[str, Any]:
        """Persist full trace-adjacent content that is too large for model context."""
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


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_secret(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_secret(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in {"api_key", "token", "secret", "password"}):
        return "[redacted]"
    return _redact(value)


def _redact_text(text: str) -> str:
    """Redact obvious secret values inside free-form trace text."""
    redacted = re.sub(
        r"(?i)\b([a-z0-9_-]*(?:api[_-]?key|token|secret|password)[a-z0-9_-]*)\s*([:=])\s*['\"]?[^'\"\s]+",
        lambda match: f"{match.group(1)}{match.group(2)} [redacted]",
        text,
    )
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[redacted]", redacted)
    return redacted


def _safe_artifact_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip())
    safe = safe.strip(".-")
    return safe or "artifact"
