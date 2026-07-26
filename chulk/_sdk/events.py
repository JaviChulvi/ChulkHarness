"""Projection from internal trace activity to the stable public event catalog."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from chulk._sdk.results import cost_snapshot, plan_snapshot, run_result_from_runtime, usage_snapshot
from chulk.core import Agent as CoreAgent, TraceEvent
from chulk.events import (
    AgentEvent,
    BudgetPayload,
    EventName,
    LearningProposalChangedPayload,
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
    extensions: dict[str, Any] = {"internal_event": event_type}
    execution_scope = getattr(runtime, "execution_scope", None)
    if execution_scope is not None:
        extensions["execution_scope"] = execution_scope.to_dict()
        extensions["execution_scope_key"] = execution_scope.key

    if event_type == TraceEvent.TURN_STARTED:
        turn_value = payload.get("turn")
        turn = turn_value if isinstance(turn_value, dict) else {}
        return _event(
            EventName.RUN_STARTED,
            conversation_id,
            turn_id,
            RunStartedPayload(str(turn.get("user_message") or "")),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type == TraceEvent.MODEL_REQUEST_STARTED:
        return _event(
            EventName.MODEL_REQUEST_STARTED,
            conversation_id,
            turn_id,
            ModelRequestPayload(payload.get("request_index"), payload.get("purpose")),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type == TraceEvent.MODEL_STREAM_DELTA:
        text = payload.get("text")
        if isinstance(text, str) and text:
            return _event(
                EventName.MODEL_DELTA,
                conversation_id,
                turn_id,
                ModelDeltaPayload(text),
                extensions,
                profile_id=runtime.profile_id,
            )
        return None
    if event_type == TraceEvent.MODEL_RESPONSE:
        return _event(
            EventName.MODEL_RESPONSE_COMPLETED,
            conversation_id,
            turn_id,
            ModelResponsePayload(
                request_index=payload.get("request_index"),
                content=payload.get("content") if isinstance(payload.get("content"), str) else None,
                usage=usage_snapshot(payload.get("usage")),
                cost=cost_snapshot(payload.get("cost")),
            ),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type == TraceEvent.LEARNING_PROPOSAL_CHANGED:
        return _event(
            EventName.LEARNING_PROPOSAL_CHANGED,
            conversation_id,
            turn_id,
            LearningProposalChangedPayload(
                proposal_id=str(payload.get("proposal_id") or ""),
                kind=str(payload.get("kind") or "unknown"),
                status=str(payload.get("status") or "unknown"),
                action=str(payload.get("action") or "changed"),
                target_name=(
                    payload.get("target_name")
                    if isinstance(payload.get("target_name"), str)
                    else None
                ),
                extensions={
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "proposal_id",
                        "kind",
                        "status",
                        "action",
                        "target_name",
                    }
                },
            ),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type in {
        TraceEvent.BUDGET_RESERVED,
        TraceEvent.BUDGET_COMMITTED,
        TraceEvent.BUDGET_RELEASED,
        TraceEvent.BUDGET_EXHAUSTED,
    }:
        name = {
            TraceEvent.BUDGET_RESERVED: EventName.BUDGET_RESERVED,
            TraceEvent.BUDGET_COMMITTED: EventName.BUDGET_COMMITTED,
            TraceEvent.BUDGET_RELEASED: EventName.BUDGET_RELEASED,
            TraceEvent.BUDGET_EXHAUSTED: EventName.BUDGET_EXHAUSTED,
        }[event_type]
        return _event(
            name,
            conversation_id,
            turn_id,
            BudgetPayload(
                resource_kind=str(payload.get("resource_kind") or "unknown"),
                scope=payload.get("scope")
                if isinstance(payload.get("scope"), str)
                else None,
                reservation_id=payload.get("reservation_id")
                if isinstance(payload.get("reservation_id"), str)
                else None,
                dimension=payload.get("dimension")
                if isinstance(payload.get("dimension"), str)
                else None,
                message=payload.get("message")
                if isinstance(payload.get("message"), str)
                else None,
                extensions={
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "resource_kind",
                        "scope",
                        "reservation_id",
                        "dimension",
                        "message",
                    }
                },
            ),
            extensions,
            profile_id=runtime.profile_id,
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
                extensions={
                    "tool_identity": (
                        payload.get("metadata", {}).get("tool_identity")
                        if isinstance(payload.get("metadata"), dict)
                        else None
                    ),
                    "tool_identity_digest": (
                        payload.get("metadata", {}).get(
                            "tool_identity_digest"
                        )
                        if isinstance(payload.get("metadata"), dict)
                        else None
                    ),
                    "tool_policy": (
                        payload.get("metadata", {}).get("tool_policy")
                        if isinstance(payload.get("metadata"), dict)
                        else None
                    ),
                    "tool_policy_digest": (
                        payload.get("metadata", {}).get(
                            "tool_policy_digest"
                        )
                        if isinstance(payload.get("metadata"), dict)
                        else None
                    ),
                },
            ),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type == TraceEvent.TOOL_AUTHORIZATION_REQUESTED:
        return _event(
            EventName.PERMISSION_REQUESTED,
            conversation_id,
            turn_id,
            PermissionPayload(
                tool_name=str(payload.get("tool_name") or "unknown"),
                reason="host authorization required",
                policy_name="host",
                extensions={
                    key: payload.get(key)
                    for key in (
                        "tool_identity",
                        "tool_identity_digest",
                        "tool_policy",
                        "tool_policy_digest",
                        "arguments_digest",
                    )
                },
            ),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type == TraceEvent.TOOL_AUTHORIZATION_DECIDED:
        return _event(
            EventName.PERMISSION_RESOLVED,
            conversation_id,
            turn_id,
            PermissionPayload(
                tool_name=str(payload.get("tool_name") or "unknown"),
                decision=payload.get("decision")
                if isinstance(payload.get("decision"), str)
                else None,
                reason=payload.get("reason")
                if isinstance(payload.get("reason"), str)
                else None,
                policy_name="host",
                extensions={
                    key: payload.get(key)
                    for key in (
                        "tool_identity",
                        "tool_identity_digest",
                        "tool_policy",
                        "tool_policy_digest",
                        "arguments_digest",
                    )
                },
            ),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type in {TraceEvent.TOOL_PERMISSION_REQUESTED, TraceEvent.MCP_APPROVAL_REQUESTED}:
        request_value = payload.get("request")
        request = request_value if isinstance(request_value, dict) else {}
        return _event(
            EventName.PERMISSION_REQUESTED,
            conversation_id,
            turn_id,
            PermissionPayload(
                tool_name=str(request.get("tool_name") or "unknown"),
                reason=request.get("reason"),
                policy_name=request.get("policy_name"),
                extensions={
                    "tool_identity": request.get("tool_identity"),
                    "tool_policy": request.get("tool_policy"),
                    "arguments_digest": request.get("arguments_digest"),
                },
            ),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type in {TraceEvent.TOOL_PERMISSION_DECIDED, TraceEvent.MCP_APPROVAL_DECIDED}:
        decision_value = payload.get("decision")
        decision = decision_value if isinstance(decision_value, dict) else {}
        return _event(
            EventName.PERMISSION_RESOLVED,
            conversation_id,
            turn_id,
            PermissionPayload(
                tool_name=str(decision.get("tool_name") or "unknown"),
                decision=decision.get("decision"),
                reason=decision.get("reason"),
                policy_name=decision.get("policy_name"),
                extensions={
                    "tool_identity": decision.get("tool_identity"),
                    "tool_policy": decision.get("tool_policy"),
                    "arguments_digest": decision.get("arguments_digest"),
                },
            ),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type == TraceEvent.MEMORY_SEARCH_COMPLETED:
        return _event(
            EventName.MEMORY_LOADED,
            conversation_id,
            turn_id,
            ResourcesLoadedPayload(tuple(str(item) for item in payload.get("loaded_memory_ids") or ())),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type == TraceEvent.SKILL_SELECTION_COMPLETED:
        return _event(
            EventName.SKILL_LOADED,
            conversation_id,
            turn_id,
            ResourcesLoadedPayload(tuple(str(item) for item in payload.get("loaded_skill_names") or ())),
            extensions,
            profile_id=runtime.profile_id,
        )
    if event_type in {TraceEvent.PLAN_CREATED, TraceEvent.PLAN_APPROVED}:
        name = EventName.PLAN_CREATED if event_type == TraceEvent.PLAN_CREATED else EventName.PLAN_APPROVED
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
        snapshot = plan_snapshot(plan)
        if snapshot is not None:
            return _event(
                name,
                conversation_id,
                turn_id,
                PlanPayload(snapshot),
                extensions,
                profile_id=runtime.profile_id,
            )
        return None
    if event_type == TraceEvent.TURN_FINISHED:
        result = run_result_from_runtime(runtime)
        if result.status in {"failed", "blocked", "cancelled"}:
            message = result.errors[-1] if result.errors else result.content or "The run failed."
            return _event(
                EventName.RUN_FAILED,
                conversation_id,
                turn_id,
                RunFailedPayload(_run_failure_payload(result, message)),
                extensions,
                profile_id=runtime.profile_id,
            )
        return _event(
            EventName.RUN_COMPLETED,
            conversation_id,
            turn_id,
            RunCompletedPayload(result),
            extensions,
            profile_id=runtime.profile_id,
        )
    return None


def terminal_event(result: Any, *, profile_id: str | None = None) -> AgentEvent:
    """Create the exact in-band terminal event for a generator run result."""
    payload: RunFailedPayload | RunCompletedPayload
    if getattr(result, "status", None) in {"failed", "blocked", "cancelled"}:
        errors = getattr(result, "errors", ())
        message = errors[-1] if errors else getattr(result, "content", "The run failed.")
        payload = RunFailedPayload(_run_failure_payload(result, message))
        name = EventName.RUN_FAILED
    else:
        payload = RunCompletedPayload(result)
        name = EventName.RUN_COMPLETED
    return _event(
        name,
        result.conversation_id,
        result.turn_id,
        payload,
        {"source": "run_events"},
        profile_id=profile_id,
    )


def failure_event(
    error: Any,
    *,
    conversation_id: str,
    turn_id: str | None,
    profile_id: str | None = None,
) -> AgentEvent:
    payload = error.to_dict() if hasattr(error, "to_dict") else {"category": "run", "message": str(error)}
    return _event(
        EventName.RUN_FAILED,
        conversation_id,
        turn_id,
        RunFailedPayload(payload),
        {"source": "run_events"},
        profile_id=profile_id,
    )


def _event(
    name: EventName,
    conversation_id: str,
    turn_id: str | None,
    payload: Any,
    extensions: dict[str, Any],
    *,
    profile_id: str | None = None,
) -> AgentEvent:
    return AgentEvent(
        name=name.value,
        conversation_id=conversation_id,
        turn_id=turn_id,
        profile_id=profile_id,
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


def _run_failure_payload(result: Any, message: str) -> dict[str, Any]:
    extension_metadata = getattr(result, "extension_metadata", {})
    budget = (
        extension_metadata.get("budget_exhausted")
        if isinstance(extension_metadata, Mapping)
        else None
    )
    if isinstance(budget, Mapping):
        return {
            "category": "budget_exhausted",
            "message": message,
            "details": budget,
            "result": result.to_dict(),
        }
    return {
        "category": "run",
        "message": message,
        "result": result.to_dict(),
    }


__all__ = [
    "AgentEvent",
    "DeltaCallback",
    "EventCallback",
    "EventDispatcher",
    "failure_event",
    "project_event",
    "terminal_event",
]
