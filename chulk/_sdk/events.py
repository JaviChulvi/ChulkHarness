"""Public event compatibility dispatch for SDK facades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from chulk.core import Agent as CoreAgent, TraceEvent


@dataclass(frozen=True)
class AgentEvent:
    """One public event emitted during an SDK agent run."""

    type: str
    payload: dict

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": dict(self.payload)}


EventCallback = Callable[[AgentEvent], None]
DeltaCallback = Callable[[str], None]


class EventDispatcher:
    """Route runtime events to constructor and per-run SDK callbacks."""

    def __init__(self, runtime: CoreAgent, *, on_event: EventCallback | None = None) -> None:
        self._base_event_callback = runtime.event_callback
        self._on_event = on_event
        self.active_on_event: EventCallback | None = None
        self.active_on_delta: DeltaCallback | None = None
        runtime.event_callback = self.dispatch

    def dispatch(self, event_type: str, payload: dict) -> None:
        if self._base_event_callback is not None:
            self._base_event_callback(event_type, payload)
        event = AgentEvent(event_type, payload)
        if self._on_event is not None:
            self._on_event(event)
        if self.active_on_event is not None:
            self.active_on_event(event)
        if event_type == TraceEvent.MODEL_STREAM_DELTA and self.active_on_delta is not None:
            text = payload.get("text")
            if isinstance(text, str) and text:
                self.active_on_delta(text)


__all__ = ["AgentEvent", "DeltaCallback", "EventCallback", "EventDispatcher"]
