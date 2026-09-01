"""Trace event names shared by the agent and CLI."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from chulk.redaction import redact_data


@dataclass(frozen=True)
class AgentEvent:
    """Typed event wrapper for host adapters."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


class RuntimeEventDispatcher:
    """Own trace callbacks, redaction, and the hosted public-event sink."""

    def __init__(
        self,
        *,
        trace_logger: Any = None,
        event_callback: Callable[[str, dict], None] | None = None,
        event_sink: Callable[[AgentEvent], None] | None = None,
        audit_callback: Callable[[str, dict], None] | None = None,
        public_event_sink: Any = None,
        redaction_callback: Callable[[str, str, dict], str] | None = None,
        redaction_fail_closed: bool = False,
    ) -> None:
        self.trace_logger = trace_logger
        self.event_callback = event_callback
        self.event_sink = event_sink
        self.audit_callback = audit_callback
        self.public_event_sink = public_event_sink
        self.redaction_callback = redaction_callback
        self.redaction_fail_closed = redaction_fail_closed

    def emit(self, event_type: str, payload: dict | None = None) -> None:
        safe_payload = self._redact_payload(event_type, dict(payload or {}))
        if self.trace_logger is not None:
            self.trace_logger.log(event_type, safe_payload)
        if self.event_callback is not None:
            self.event_callback(event_type, safe_payload)
        if self.audit_callback is not None:
            self.audit_callback(event_type, safe_payload)
        if self.event_sink is not None:
            self.event_sink(AgentEvent(event_type, safe_payload))

    def redact_text(
        self,
        event_type: str,
        text: str,
        metadata: dict,
    ) -> tuple[str, dict]:
        callback = self.redaction_callback
        if callback is None:
            return text, {"redacted": False}
        try:
            redacted = callback(event_type, text, metadata)
        except Exception as exc:
            if self.redaction_fail_closed:
                return "[redaction failed]", {
                    "redacted": True,
                    "redaction_error": str(exc),
                    "fail_closed": True,
                }
            return text, {
                "redacted": False,
                "redaction_error": str(exc),
                "fail_closed": False,
            }
        if not isinstance(redacted, str):
            redacted = str(redacted)
        return redacted, {"redacted": redacted != text}

    @contextmanager
    def capture(
        self,
        callback: Callable[[str, dict], None],
    ) -> Iterator[None]:
        """Temporarily chain one trace callback and restore it reliably."""
        previous = self.event_callback

        def dispatch(event_type: str, payload: dict) -> None:
            callback(event_type, payload)
            if previous is not None:
                previous(event_type, payload)

        self.event_callback = dispatch
        try:
            yield
        finally:
            self.event_callback = previous

    def set_event_callback(
        self,
        callback: Callable[[str, dict], None] | None,
    ) -> Callable[[str, dict], None] | None:
        previous = self.event_callback
        self.event_callback = callback
        return previous

    def set_public_sink(self, sink: Any) -> Any:
        previous = self.public_event_sink
        self.public_event_sink = sink
        return previous

    def clear_callbacks(self) -> None:
        self.event_callback = None
        self.event_sink = None
        self.audit_callback = None

    def _redact_payload(self, event_type: str, payload: dict) -> dict:
        baseline: dict = redact_data(payload)
        if self.redaction_callback is None:
            return baseline

        redacted_any = baseline != payload
        error: str | None = None

        def redact_value(value: object, path: str) -> object:
            nonlocal redacted_any, error
            if isinstance(value, str):
                redacted, metadata = self.redact_text(
                    event_type,
                    value,
                    {"path": path},
                )
                redacted_any = redacted_any or bool(metadata.get("redacted"))
                if metadata.get("redaction_error"):
                    error = str(metadata["redaction_error"])
                    redacted_any = redacted_any or bool(metadata.get("fail_closed"))
                return redacted
            if isinstance(value, dict):
                return {
                    key: redact_value(item, f"{path}.{key}")
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [
                    redact_value(item, f"{path}[{index}]")
                    for index, item in enumerate(value)
                ]
            return value

        redacted_payload = {
            key: redact_value(item, f"payload.{key}")
            for key, item in baseline.items()
        }
        final_payload: dict = redact_data(redacted_payload)
        redacted_any = redacted_any or final_payload != redacted_payload
        if redacted_any:
            final_payload["_redacted"] = True
        if error is not None:
            final_payload["_redaction_error"] = error
        return final_payload


class TraceEvent:
    """Internal trace/progress constants; not a public compatibility catalog."""

    SESSION_STARTED = "session_started"
    SESSION_FINISHED = "session_finished"
    TURN_STARTED = "turn_started"
    MODEL_PROFILE_SELECTED = "model_profile_selected"
    USER_MESSAGE = "user_message"
    MEDIA_INPUT_PREPARED = "media_input_prepared"
    MEDIA_TRANSFORMED = "media_transformed"
    TURN_CONTEXT_SELECTED = "turn_context_selected"
    HOST_RESOURCE_AVAILABLE = "host_resource_available"
    APPLICATION_EVENT = "application_event"
    MEMORY_EXTRACTION_COMPLETED = "memory_extraction_completed"
    MEMORY_SEARCH_STARTED = "memory_search_started"
    MEMORY_SEARCH_COMPLETED = "memory_search_completed"
    SKILL_SELECTION_STARTED = "skill_selection_started"
    SKILL_SELECTION_COMPLETED = "skill_selection_completed"
    LEARNING_PROPOSAL_CHANGED = "learning_proposal_changed"
    CONTEXT_SUMMARY_CREATED = "context_summary_created"
    CONTEXT_BUDGET_REJECTED = "context_budget_rejected"
    MODEL_REQUEST_STARTED = "model_request_started"
    BUDGET_RESERVED = "budget_reserved"
    BUDGET_COMMITTED = "budget_committed"
    BUDGET_RELEASED = "budget_released"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LLM_FALLBACK_ATTEMPTS = "llm_fallback_attempts"
    MODEL_RESPONSE = "model_response"
    PARSED_ACTION = "parsed_action"
    MODEL_RESPONSE_PARSED = "model_response_parsed"
    MODEL_STREAM_STARTED = "model_stream_started"
    MODEL_STREAM_DELTA = "model_stream_delta"
    MODEL_STREAM_COMPLETED = "model_stream_completed"
    MODEL_STREAM_FAILED = "model_stream_failed"
    TOOL_PERMISSION_REQUESTED = "tool_permission_requested"
    TOOL_PERMISSION_DECIDED = "tool_permission_decided"
    TOOL_AUTHORIZATION_REQUESTED = "tool_authorization_requested"
    TOOL_AUTHORIZATION_DECIDED = "tool_authorization_decided"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_ATTEMPT = "tool_call_attempt"
    TOOL_CALL = "tool_call"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    TOOL_OBSERVATION = "tool_observation"
    MCP_CONFIG_LOADED = "mcp_config_loaded"
    MCP_TOOL_DISCOVERY_COMPLETED = "mcp_tool_discovery_completed"
    MCP_APPROVAL_REQUESTED = "mcp_approval_requested"
    MCP_APPROVAL_DECIDED = "mcp_approval_decided"
    PLAN_CREATED = "plan_created"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"
    PLAN_REVISION_REQUESTED = "plan_revision_requested"
    PLAN_STEP_STARTED = "plan_step_started"
    PLAN_STEP_COMPLETED = "plan_step_completed"
    PLAN_STEP_BLOCKED = "plan_step_blocked"
    REFLECTION_STARTED = "reflection_started"
    REFLECTION_COMPLETED = "reflection_completed"
    REFLECTION_REVISION_REQUESTED = "reflection_revision_requested"
    REFLECTION_FAILED = "reflection_failed"
    FINAL_ANSWER = "final_answer"
    TURN_FAILED = "turn_failed"
    TURN_FINISHED = "turn_finished"
