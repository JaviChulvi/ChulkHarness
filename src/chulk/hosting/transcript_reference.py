"""In-memory reference services for external-transcript hosted contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from chulk.hosting.scope import ExecutionScope
from chulk.hosting.transcripts import TranscriptProjection


class InMemoryExecutionJournal:
    """Reference recovery journal that never stores business transcript rows."""

    def __init__(self) -> None:
        self._scopes: dict[str, ExecutionScope] = {}
        self._turns: dict[str, dict[str, dict[str, Any]]] = {}
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def bind_scope(self, conversation_id: str, scope: ExecutionScope) -> None:
        existing = self._scopes.get(conversation_id)
        if existing is not None:
            scope.assert_resumable(existing)
        self._scopes[conversation_id] = scope

    def load_scope(self, conversation_id: str) -> ExecutionScope | None:
        return self._scopes.get(conversation_id)

    def load_turns(self, conversation_id: str) -> list[Any]:
        from chulk.sessions.sqlite_store import _turn_from_dict

        values = self._turns.get(conversation_id, {})
        return [_turn_from_dict(deepcopy(item)) for item in values.values()]

    def save_turn_snapshot(
        self,
        conversation_id: str,
        turn: dict[str, Any],
    ) -> None:
        turn_id = str(turn.get("turn_id") or "")
        if not turn_id:
            raise ValueError("execution journal turn snapshot requires turn_id")
        self._turns.setdefault(conversation_id, {})[turn_id] = deepcopy(turn)

    def append_event(
        self,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.events.append((conversation_id, event_type, deepcopy(payload)))


class AsyncInMemoryExecutionJournal:
    """Native-async wrapper over the deterministic in-memory journal."""

    def __init__(self) -> None:
        self._journal = InMemoryExecutionJournal()

    @property
    def events(self) -> list[tuple[str, str, dict[str, Any]]]:
        return self._journal.events

    async def bind_scope(
        self,
        conversation_id: str,
        scope: ExecutionScope,
    ) -> None:
        self._journal.bind_scope(conversation_id, scope)

    async def load_scope(self, conversation_id: str) -> ExecutionScope | None:
        return self._journal.load_scope(conversation_id)

    async def load_turns(self, conversation_id: str) -> list[Any]:
        return self._journal.load_turns(conversation_id)

    async def save_turn_snapshot(
        self,
        conversation_id: str,
        turn: dict[str, Any],
    ) -> None:
        self._journal.save_turn_snapshot(conversation_id, turn)

    async def append_event(
        self,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self._journal.append_event(conversation_id, event_type, payload)


class InMemoryTranscriptProjectionSink:
    """Idempotent reference sink keyed by the terminal projection identity."""

    def __init__(self) -> None:
        self.projections: list[TranscriptProjection] = []
        self._keys: set[str] = set()

    def emit(self, projection: TranscriptProjection) -> None:
        if projection.idempotency_key in self._keys:
            return
        self._keys.add(projection.idempotency_key)
        self.projections.append(projection)


class AsyncInMemoryTranscriptProjectionSink:
    """Native-async idempotent projection reference sink."""

    def __init__(self) -> None:
        self._sink = InMemoryTranscriptProjectionSink()

    @property
    def projections(self) -> list[TranscriptProjection]:
        return self._sink.projections

    async def emit(self, projection: TranscriptProjection) -> None:
        self._sink.emit(projection)


__all__ = [
    "AsyncInMemoryExecutionJournal",
    "AsyncInMemoryTranscriptProjectionSink",
    "InMemoryExecutionJournal",
    "InMemoryTranscriptProjectionSink",
]
