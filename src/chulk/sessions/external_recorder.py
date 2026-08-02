"""Recovery-only recorders for externally owned conversation transcripts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from chulk.core.events import TraceEvent
from chulk.hosting.async_utils import call_async_service
from chulk.hosting.scope import ExecutionScope
from chulk.hosting.transcripts import (
    AsyncExecutionJournal,
    AsyncTranscriptProjectionSink,
    ExecutionJournal,
    TranscriptProjection,
    TranscriptProjectionSink,
)
from chulk.resources import HostResource


_TERMINAL_EVENTS = frozenset(
    {
        TraceEvent.FINAL_ANSWER,
        TraceEvent.TURN_FAILED,
        TraceEvent.PLAN_REJECTED,
    }
)


class ExternalTranscriptRecorder:
    """Persist recovery evidence without writing conversation or message rows."""

    def __init__(
        self,
        journal: ExecutionJournal,
        projections: TranscriptProjectionSink,
        conversation_id: str,
        scope: ExecutionScope,
    ) -> None:
        self.journal = journal
        self.projections = projections
        self.conversation_id = conversation_id
        self.scope = scope
        self.journal.bind_scope(conversation_id, scope)

    def callback(self, event_type: str, payload: dict[str, Any]) -> None:
        turn = payload.get("turn")
        if isinstance(turn, dict):
            if event_type in _TERMINAL_EVENTS:
                projection = _terminal_projection(
                    self.conversation_id,
                    event_type,
                    payload,
                    turn,
                )
                if projection is not None:
                    self.projections.emit(projection)
            self.journal.save_turn_snapshot(
                self.conversation_id,
                _recovery_turn(turn),
            )
        self.journal.append_event(
            self.conversation_id,
            event_type,
            _recovery_event(payload),
        )


class AsyncExternalTranscriptRecorder:
    """Queue recovery evidence for native-async journal and projection sinks."""

    def __init__(
        self,
        journal: AsyncExecutionJournal,
        projections: AsyncTranscriptProjectionSink,
        conversation_id: str,
        scope: ExecutionScope,
    ) -> None:
        self.journal = journal
        self.projections = projections
        self.conversation_id = conversation_id
        self.scope = scope
        self._pending: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        await call_async_service(
            self.journal,
            "bind_scope",
            self.conversation_id,
            self.scope,
        )

    def callback(self, event_type: str, payload: dict[str, Any]) -> None:
        self._pending.append((event_type, deepcopy(payload)))

    async def flush(self) -> None:
        while self._pending:
            event_type, payload = self._pending[0]
            turn = payload.get("turn")
            if isinstance(turn, dict):
                if event_type in _TERMINAL_EVENTS:
                    projection = _terminal_projection(
                        self.conversation_id,
                        event_type,
                        payload,
                        turn,
                    )
                    if projection is not None:
                        await call_async_service(
                            self.projections,
                            "emit",
                            projection,
                        )
                await call_async_service(
                    self.journal,
                    "save_turn_snapshot",
                    self.conversation_id,
                    _recovery_turn(turn),
                )
            await call_async_service(
                self.journal,
                "append_event",
                self.conversation_id,
                event_type,
                _recovery_event(payload),
            )
            self._pending.pop(0)


def _recovery_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Strip host-owned transcript text while retaining recovery state."""
    result = deepcopy(turn)
    result["user_message"] = "[externally owned transcript]"
    result["final_answer"] = None
    result["context_sections"] = [
        {
            **section,
            "content": "[content omitted]",
        }
        for section in result.get("context_sections", [])
        if isinstance(section, dict)
    ]
    result["observations"] = [
        {
            **observation,
            "content": "[recovery evidence omitted]",
        }
        for observation in result.get("observations", [])
        if isinstance(observation, dict)
    ]
    return result


def _recovery_event(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "turn_id",
        "status",
        "tool_name",
        "iteration",
        "request_index",
        "observation_index",
        "failure_kind",
    ):
        value = payload.get(name)
        if isinstance(value, (str, int, bool)) or value is None:
            result[name] = value
    turn = payload.get("turn")
    if isinstance(turn, dict):
        result["turn"] = _recovery_turn(turn)
    return result


def _terminal_projection(
    conversation_id: str,
    event_type: str,
    payload: dict[str, Any],
    turn: dict[str, Any],
) -> TranscriptProjection | None:
    turn_id = str(turn.get("turn_id") or payload.get("turn_id") or "")
    metadata = turn.get("extension_metadata")
    transcript = metadata.get("external_transcript") if isinstance(metadata, dict) else None
    digest = transcript.get("digest") if isinstance(transcript, dict) else None
    if not turn_id or not isinstance(digest, str):
        return None
    content = str(
        payload.get("content")
        or payload.get("message")
        or turn.get("final_answer")
        or ""
    )
    resources = tuple(
        HostResource.from_dict(item)
        for item in turn.get("resources", [])
        if isinstance(item, dict)
    )
    extensions = {}
    delivery = metadata.get("final_answer_delivery") if isinstance(metadata, dict) else None
    if isinstance(delivery, dict):
        extensions["final_answer_delivery"] = deepcopy(delivery)
    return TranscriptProjection(
        idempotency_key=f"{conversation_id}:{turn_id}:assistant:terminal",
        conversation_id=conversation_id,
        turn_id=turn_id,
        input_transcript_digest=digest,
        status=str(turn.get("status") or event_type),
        content=content,
        resources=resources,
        extensions=extensions,
    )


__all__ = [
    "AsyncExternalTranscriptRecorder",
    "ExternalTranscriptRecorder",
]
