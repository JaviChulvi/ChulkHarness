"""Reference host sinks with bounded, redacted delivery semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from chulk.events import AgentEvent
from chulk.hosting.async_utils import call_async_service
from chulk.hosting.scope import ExecutionScope
from chulk.hosting.services import AsyncAuditSink, AsyncTraceSink
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


class AsyncInMemoryEventSink:
    """Native async scope-owned public event sink."""

    def __init__(self, scope: ExecutionScope) -> None:
        self.scope = scope
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        if event.execution_scope is None:
            raise ValueError("hosted public event is missing execution_scope")
        self.scope.assert_resumable(event.execution_scope)
        self.events.append(event)


class BufferedAsyncEventSink:
    """Sync-facing event buffer drained through a host's async sink."""

    def __init__(self, sink: object) -> None:
        self.sink = sink
        self._pending: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self._pending.append(event)

    async def flush(self) -> None:
        while self._pending:
            event = self._pending[0]
            await call_async_service(self.sink, "emit", event)
            self._pending.pop(0)


class BufferedAsyncTraceSink:
    """Preserve trace ordering while the synchronous core emits into an async sink.

    The async hosted facade drains this journal at every public await boundary.
    It deliberately never invokes a host's synchronous compatibility method.
    """

    def __init__(self, sink: object) -> None:
        self.sink = cast(AsyncTraceSink, sink)
        self.path = getattr(sink, "path", None)
        self.artifact_store = getattr(sink, "artifact_store", None)
        self._pending: list[tuple[str, dict[str, Any] | None, str | None]] = []

    def activate(self) -> None:
        """Activation is represented by the next queued trace record."""

    def log(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> None:
        self._pending.append((event_type, payload, turn_id))

    def write_artifact(self, name: str, content: str) -> dict[str, Any] | None:
        raise RuntimeError(
            "native async trace artifacts require the async artifact execution path"
        )

    def close(self) -> None:
        """The owning async facade closes the native resource."""

    async def flush(self) -> None:
        while self._pending:
            event_type, payload, turn_id = self._pending[0]
            await call_async_service(
                self.sink,
                "log",
                event_type,
                payload,
                turn_id=turn_id,
            )
            self._pending.pop(0)


class BufferedAsyncAuditSink:
    """Queue redacted audit records for native async host sinks."""

    def __init__(self, sink: object) -> None:
        self.sink = cast(AsyncAuditSink, sink)
        self._pending: list[tuple[str, dict[str, Any], ExecutionScope]] = []

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        scope: ExecutionScope,
    ) -> None:
        self._pending.append((event_type, payload, scope))

    async def flush(self) -> None:
        while self._pending:
            event_type, payload, scope = self._pending[0]
            await call_async_service(
                self.sink,
                "record",
                event_type,
                payload,
                scope=scope,
            )
            self._pending.pop(0)


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
    "BufferedAsyncAuditSink",
    "AsyncInMemoryEventSink",
    "BufferedAsyncEventSink",
    "BufferedAsyncTraceSink",
    "CallbackEventSink",
    "InMemoryAuditSink",
    "InMemoryEventSink",
    "SinkDeliveryError",
    "safe_audit_payload",
]
