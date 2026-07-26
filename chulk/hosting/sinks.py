"""Reference host sinks with bounded, redacted delivery semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from chulk.events import AgentEvent
from chulk.hosting.scope import ExecutionScope
from chulk.redaction import redact_data


class SinkDeliveryError(RuntimeError):
    """Raised when a fail-closed host sink cannot accept an event."""


class InMemoryEventSink:
    """Scope-owned public event sink used by local mode and contract tests."""

    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        if event.execution_scope is None:
            raise ValueError("hosted public event is missing execution_scope")
        self.scope.assert_resumable(event.execution_scope)
        self.events.append(event)


class CallbackEventSink:
    """Invoke an app callback with explicit fail-closed behavior."""

    def __init__(
        self,
        callback: Callable[[AgentEvent], None],
        *,
        fail_closed: bool = True,
    ) -> None:
        self.callback = callback
        self.fail_closed = fail_closed
        self.failures: list[str] = []

    def emit(self, event: AgentEvent) -> None:
        try:
            self.callback(event)
        except Exception as exc:
            self.failures.append(type(exc).__name__)
            if self.fail_closed:
                raise SinkDeliveryError(
                    "public event sink rejected a redacted event"
                ) from exc


class InMemoryAuditSink:
    """Scope-owned audit sink that rejects unsafe fields before storage."""

    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.events: list[dict[str, Any]] = []

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        scope: ExecutionScope,
    ) -> None:
        self.scope.assert_same_authority(scope)
        safe = safe_audit_payload(payload)
        self.events.append(
            {
                "type": event_type,
                "payload": safe,
                "scope": scope.to_dict(),
            }
        )


def safe_audit_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return redacted metadata and reject raw secret-bearing fields."""
    forbidden = {
        "api_key",
        "authorization",
        "credential",
        "credentials",
        "password",
        "prompt",
        "raw_arguments",
        "secret",
        "token",
    }

    def inspect(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).casefold() in forbidden:
                    raise ValueError(
                        f"audit payload cannot contain field {key!r}"
                    )
                inspect(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                inspect(nested)

    inspect(value)
    redacted = redact_data(dict(value))
    if not isinstance(redacted, dict):
        raise ValueError("audit payload must remain an object after redaction")
    return redacted


__all__ = [
    "CallbackEventSink",
    "InMemoryAuditSink",
    "InMemoryEventSink",
    "SinkDeliveryError",
    "safe_audit_payload",
]
