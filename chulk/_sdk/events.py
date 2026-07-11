"""Projection from internal trace activity to the stable public event catalog."""

from __future__ import annotations

from typing import Any, Callable

from chulk._sdk.results import run_result_from_runtime
from chulk.core import Agent as CoreAgent, TraceEvent
from chulk.events import (
    AgentEvent,
    EventName,
    ModelDeltaPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    PermissionPayload,
    PlanPayload,
    ResourcesLoadedPayload,
    RunCompletedPayload,
    RunFailedPayload,
    RunStartedPayload,
    ToolCallPayload,
)


EventCallback = Callable[[AgentEvent], None]
DeltaCallback = Callable[[str], None]


class EventDispatcher:
    """Route internal events to trace consumers and curated SDK callbacks."""

    def __init__(self, runtime: CoreAgent, *, on_event: EventCallback | None = None) -> None:
        self.runtime = runtime
        self._base_event_callback = runtime.event_callback
        self._on_event = on_event
        self.active_on_event: EventCallback | None = None
        self.active_on_delta: DeltaCallback | None = None
        runtime.event_callback = self.dispatch

    def dispatch(self, event_type: str, payload: dict) -> None:
        if self._base_event_callback is not None:
            self._base_event_callback(event_type, payload)
        event = project_event(self.runtime, event_type, payload)
        if event is None:
            return
        if self._on_event is not None:
            self._on_event(event)
        if self.active_on_event is not None:
            self.active_on_event(event)
        if event.name == EventName.MODEL_DELTA.value and self.active_on_delta is not None:
            text = event.payload.text if isinstance(event.payload, ModelDeltaPayload) else None
            if text:
                self.active_on_delta(text)


def project_event(runtime: CoreAgent, event_type: str, payload: dict[str, Any]) -> AgentEvent | None:
    """Project one explicitly supported internal event, excluding all others."""
    conversation_id = runtime.state.conversation_id
    turn_id = _turn_id(runtime, payload)
    extensions = {"internal_event": event_type}

    if event_type == TraceEvent.TURN_STARTED:
        turn = payload.get("turn") if isinstance(payload.get("turn"), dict) else {}
        return _event(EventName.RUN_STARTED, conversation_id, turn_id, RunStartedPayload(str(turn.get("user_message") or "")), extensions)
    if event_type == TraceEvent.MODEL_REQUEST_STARTED:
        return _event(
            EventName.MODEL_REQUEST_STARTED,
            conversation_id,
            turn_id,
            ModelRequestPayload(payload.get("request_index"), payload.get("purpose")),
            extensions,
        )
    if event_type == TraceEvent.MODEL_STREAM_DELTA:
        text = payload.get("text")
        if isinstance(text, str) and text:
            return _event(EventName.MODEL_DELTA, conversation_id, turn_id, ModelDeltaPayload(text), extensions)
        return None
    if event_type == TraceEvent.MODEL_RESPONSE:
        return _event(
            EventName.MODEL_RESPONSE_COMPLETED,
            conversation_id,
            turn_id,
            ModelResponsePayload(
                request_index=payload.get("request_index"),
                content=payload.get("content") if isinstance(payload.get("content"), str) else None,
                usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
                cost=payload.get("cost") if isinstance(payload.get("cost"), dict) else None,
            ),
            extensions,
        )
    if event_type in {TraceEvent.TOOL_CALL_STARTED, TraceEvent.TOOL_CALL_COMPLETED, TraceEvent.TOOL_CALL_FAILED}:
        name = {
            TraceEvent.TOOL_CALL_STARTED: EventName.TOOL_CALL_STARTED,
            TraceEvent.TOOL_CALL_COMPLETED: EventName.TOOL_CALL_COMPLETED,
            TraceEvent.TOOL_CALL_FAILED: EventName.TOOL_CALL_FAILED,
        }[event_type]
        return _event(
            name,
            conversation_id,
            turn_id,
            ToolCallPayload(
                tool_name=str(payload.get("tool_name") or "unknown"),
                success=payload.get("success") if isinstance(payload.get("success"), bool) else None,
                failure_kind=payload.get("failure_kind"),
                error=payload.get("error"),
            ),
            extensions,
        )
    if event_type in {TraceEvent.TOOL_PERMISSION_REQUESTED, TraceEvent.MCP_APPROVAL_REQUESTED}:
        request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        return _event(
            EventName.PERMISSION_REQUESTED,
            conversation_id,
            turn_id,
            PermissionPayload(
                tool_name=str(request.get("tool_name") or "unknown"),
                reason=request.get("reason"),
                policy_name=request.get("policy_name"),
            ),
            extensions,
        )
    if event_type in {TraceEvent.TOOL_PERMISSION_DECIDED, TraceEvent.MCP_APPROVAL_DECIDED}:
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        return _event(
            EventName.PERMISSION_RESOLVED,
            conversation_id,
            turn_id,
            PermissionPayload(
                tool_name=str(decision.get("tool_name") or "unknown"),
                decision=decision.get("decision"),
                reason=decision.get("reason"),
                policy_name=decision.get("policy_name"),
            ),
            extensions,
        )
    if event_type == TraceEvent.MEMORY_SEARCH_COMPLETED:
        return _event(
            EventName.MEMORY_LOADED,
            conversation_id,
            turn_id,
            ResourcesLoadedPayload(tuple(str(item) for item in payload.get("loaded_memory_ids") or ())),
            extensions,
        )
    if event_type == TraceEvent.SKILL_SELECTION_COMPLETED:
        return _event(
            EventName.SKILL_LOADED,
            conversation_id,
            turn_id,
            ResourcesLoadedPayload(tuple(str(item) for item in payload.get("loaded_skill_names") or ())),
            extensions,
        )
    if event_type in {TraceEvent.PLAN_CREATED, TraceEvent.PLAN_APPROVED}:
        name = EventName.PLAN_CREATED if event_type == TraceEvent.PLAN_CREATED else EventName.PLAN_APPROVED
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        return _event(name, conversation_id, turn_id, PlanPayload(plan), extensions)
    if event_type == TraceEvent.TURN_FINISHED:
        result = run_result_from_runtime(runtime)
        if result.status in {"failed", "blocked"}:
            message = result.errors[-1] if result.errors else result.content or "The run failed."
            return _event(
                EventName.RUN_FAILED,
                conversation_id,
                turn_id,
                RunFailedPayload({"category": "run", "message": message, "result": result.to_dict()}),
                extensions,
            )
        return _event(
            EventName.RUN_COMPLETED,
            conversation_id,
            turn_id,
            RunCompletedPayload(result),
            extensions,
        )
    return None


def terminal_event(result: Any) -> AgentEvent:
    """Create the exact in-band terminal event for a generator run result."""
    if getattr(result, "status", None) in {"failed", "blocked"}:
        errors = getattr(result, "errors", ())
        message = errors[-1] if errors else getattr(result, "content", "The run failed.")
        payload = RunFailedPayload({"category": "run", "message": message, "result": result.to_dict()})
        name = EventName.RUN_FAILED
    else:
        payload = RunCompletedPayload(result)
        name = EventName.RUN_COMPLETED
    return _event(name, result.conversation_id, result.turn_id, payload, {"source": "run_events"})


def failure_event(error: Any, *, conversation_id: str, turn_id: str | None) -> AgentEvent:
    payload = error.to_dict() if hasattr(error, "to_dict") else {"category": "run", "message": str(error)}
    return _event(EventName.RUN_FAILED, conversation_id, turn_id, RunFailedPayload(payload), {"source": "run_events"})


def _event(
    name: EventName,
    conversation_id: str,
    turn_id: str | None,
    payload: Any,
    extensions: dict[str, Any],
) -> AgentEvent:
    return AgentEvent(
        name=name.value,
        conversation_id=conversation_id,
        turn_id=turn_id,
        payload=payload,
        extensions=extensions,
    )


def _turn_id(runtime: CoreAgent, payload: dict[str, Any]) -> str | None:
    turn_id = payload.get("turn_id")
    if isinstance(turn_id, str):
        return turn_id
    turn = payload.get("turn")
    if isinstance(turn, dict) and isinstance(turn.get("turn_id"), str):
        return turn["turn_id"]
    return runtime.state.current_turn_id


__all__ = [
    "AgentEvent",
    "DeltaCallback",
    "EventCallback",
    "EventDispatcher",
    "failure_event",
    "project_event",
    "terminal_event",
]
