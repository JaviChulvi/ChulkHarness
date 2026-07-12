"""Structured tracing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from threading import RLock
from time import monotonic
from typing import Any
from uuid import uuid4

from chulk.redaction import redact_data


TRACE_SCHEMA_VERSION = 1
SESSION_STARTED = "session_started"
SESSION_FINISHED = "session_finished"
TURN_STARTED = "turn_started"
TURN_FINISHED = "turn_finished"


@dataclass(frozen=True)
class TraceEvent:
    """One versioned internal trace envelope."""

    type: str
    conversation_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        event: dict[str, Any] = {
            "schema_version": self.schema_version,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "payload": self.payload,
        }
        if self.turn_id is not None:
            event["turn_id"] = self.turn_id
        return event


class JSONLTraceLogger:
    """Append-only internal JSONL trace logger, independent of public events."""

    def __init__(
        self,
        traces_dir: Path | str,
        conversation_id: str,
        *,
        defer_until_event: str | None = None,
    ) -> None:
        clean_conversation_id = _validate_conversation_id(conversation_id)
        self.traces_dir = Path(traces_dir)
        self.conversation_id = clean_conversation_id
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.traces_dir / f"{clean_conversation_id}.jsonl"
        self.artifacts_dir = self.traces_dir / f"{clean_conversation_id}_artifacts"
        resumed = self.path.exists()
        self._defer_until_event = defer_until_event
        self._active = defer_until_event is None
        self._pending_events: list[dict[str, Any]] = []
        self._current_turn_id: str | None = None
        self._turn_started_at: dict[str, float] = {}
        self._started_at = monotonic()
        self._event_count = 0
        self._closed = False
        self._lock = RLock()
        self.log(SESSION_STARTED, {"resumed": resumed})

    def log(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> None:
        """Append a versioned, redacted trace event."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Trace logger is closed")
            if not isinstance(event_type, str) or not event_type.strip():
                raise ValueError("trace event type must be a non-empty string")
            if payload is not None and not isinstance(payload, dict):
                raise TypeError("trace event payload must be a dictionary")

            safe_payload = redact_data(payload or {})
            event_turn_id = _event_turn_id(safe_payload, turn_id) or self._current_turn_id
            if event_type == TURN_STARTED and event_turn_id is not None:
                self._current_turn_id = event_turn_id
                self._turn_started_at[event_turn_id] = monotonic()
            if event_type == TURN_FINISHED and event_turn_id is not None:
                safe_payload = _add_turn_timing(
                    safe_payload,
                    self._turn_started_at.pop(event_turn_id, None),
                )

            event = TraceEvent(
                type=event_type,
                conversation_id=self.conversation_id,
                turn_id=event_turn_id,
                payload=safe_payload,
            ).to_dict()
            self._record_event(event_type, event)

            if event_type == TURN_FINISHED and event_turn_id == self._current_turn_id:
                self._current_turn_id = None

    def activate(self) -> None:
        """Flush deferred startup events when runtime work begins."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Trace logger is closed")
            if self._active:
                return
            self._active = True
            for event in self._pending_events:
                self._append_event(event)
            self._pending_events.clear()

    def close(self) -> None:
        """Record the end of an active trace session exactly once."""
        with self._lock:
            if self._closed:
                return
            if self._active:
                payload = {
                    "duration_ms": _duration_ms(self._started_at),
                    "event_count": self._event_count + 1,
                }
                event = TraceEvent(
                    type=SESSION_FINISHED,
                    conversation_id=self.conversation_id,
                    payload=payload,
                ).to_dict()
                self._append_event(event)
            else:
                # Preserve lazy runtime behavior: constructing and closing an unused
                # agent must not create a trace file.
                self._pending_events.clear()
            self._closed = True

    def _record_event(self, event_type: str, event: dict[str, Any]) -> None:
        if not self._active:
            if event_type != self._defer_until_event:
                self._pending_events.append(event)
                return
            self._active = True
            for pending_event in self._pending_events:
                self._append_event(pending_event)
            self._pending_events.clear()
        self._append_event(event)

    def _append_event(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event, sort_keys=True) + "\n")
        self._event_count += 1

    def write_artifact(self, name: str, content: str) -> dict[str, Any]:
        """Persist full trace-adjacent content that is too large for model context."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Trace logger is closed")
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


def _validate_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string")
    clean = conversation_id.strip()
    if clean in {".", ".."} or "/" in clean or "\\" in clean or "\x00" in clean:
        raise ValueError("conversation_id must be a safe filename component")
    return clean


def _event_turn_id(payload: dict[str, Any], explicit_turn_id: str | None) -> str | None:
    if explicit_turn_id is not None:
        if not isinstance(explicit_turn_id, str) or not explicit_turn_id:
            raise ValueError("turn_id must be a non-empty string when provided")
        return explicit_turn_id
    turn_id = payload.get("turn_id")
    if isinstance(turn_id, str) and turn_id:
        return turn_id
    turn = payload.get("turn")
    if isinstance(turn, dict):
        nested_turn_id = turn.get("turn_id")
        if isinstance(nested_turn_id, str) and nested_turn_id:
            return nested_turn_id
    return None


def _add_turn_timing(payload: dict[str, Any], started_at: float | None) -> dict[str, Any]:
    if started_at is None:
        return payload
    timing = payload.get("timing")
    safe_timing = dict(timing) if isinstance(timing, dict) else {}
    safe_timing.setdefault("duration_ms", _duration_ms(started_at))
    return {**payload, "timing": safe_timing}


def _duration_ms(started_at: float) -> float:
    return round(max(0.0, (monotonic() - started_at) * 1000), 3)


def _safe_artifact_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip())
    safe = safe.strip(".-")
    return safe or "artifact"


__all__ = ["JSONLTraceLogger", "TRACE_SCHEMA_VERSION", "TraceEvent"]
