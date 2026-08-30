"""Turn-scoped tool context lifecycle ownership."""

from __future__ import annotations

from collections.abc import Callable

from chulk.core.state import TurnState
from chulk.tools.registry import (
    ToolContextLifecycle,
    ToolExecutionContext,
)


class ToolContextRuntime:
    """Create, cache, and release request-scoped tool contexts."""

    def __init__(
        self,
        *,
        conversation_id: Callable[[], str],
        default_context: ToolExecutionContext | None = None,
        lifecycle: ToolContextLifecycle | None = None,
    ) -> None:
        self._conversation_id = conversation_id
        self.default_context = default_context
        self.lifecycle = lifecycle
        self._contexts: dict[str, ToolExecutionContext | None] = {}

    def prepare(
        self,
        turn: TurnState,
        context: ToolExecutionContext | None,
    ) -> ToolExecutionContext:
        prepared = self._build(turn, context)
        if self.lifecycle is not None and prepared.execution_session is None:
            prepared = self.lifecycle.open(prepared)
        self._contexts[turn.turn_id] = prepared
        return prepared

    async def prepare_async(
        self,
        turn: TurnState,
        context: ToolExecutionContext | None,
    ) -> ToolExecutionContext:
        prepared = self._build(turn, context)
        if self.lifecycle is not None and prepared.execution_session is None:
            prepared = await self.lifecycle.open_async(prepared)
        self._contexts[turn.turn_id] = prepared
        return prepared

    def get(self, turn: TurnState) -> ToolExecutionContext | None:
        if turn.turn_id in self._contexts:
            return self._contexts[turn.turn_id]
        if not turn.tool_context_metadata and self.default_context is None and self.lifecycle is None:
            return None
        return self.prepare(turn, self.default_context)

    async def get_async(self, turn: TurnState) -> ToolExecutionContext | None:
        if turn.turn_id in self._contexts:
            return self._contexts[turn.turn_id]
        if not turn.tool_context_metadata and self.default_context is None and self.lifecycle is None:
            return None
        return await self.prepare_async(turn, self.default_context)

    def release(self, turn: TurnState) -> None:
        context = self._contexts.pop(turn.turn_id, None)
        if context is not None and self.lifecycle is not None:
            self.lifecycle.close(context)

    async def release_async(self, turn: TurnState) -> None:
        context = self._contexts.pop(turn.turn_id, None)
        if context is not None and self.lifecycle is not None:
            await self.lifecycle.aclose(context)

    def close(self) -> list[Exception]:
        failures: list[Exception] = []
        for context in tuple(self._contexts.values()):
            if context is None or self.lifecycle is None:
                continue
            try:
                self.lifecycle.close(context)
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        self._contexts.clear()
        return failures

    async def aclose(self) -> list[Exception]:
        failures: list[Exception] = []
        for turn_id, context in tuple(self._contexts.items()):
            if context is not None and self.lifecycle is not None:
                try:
                    await self.lifecycle.aclose(context)
                except Exception as exc:  # pragma: no cover - defensive aggregation
                    failures.append(exc)
            self._contexts.pop(turn_id, None)
        return failures

    def _build(
        self,
        turn: TurnState,
        context: ToolExecutionContext | None,
    ) -> ToolExecutionContext:
        default = context or self.default_context
        return ToolExecutionContext(
            metadata={
                **(default.metadata if default is not None else {}),
                **turn.tool_context_metadata,
                "conversation_id": self._conversation_id(),
                "turn_id": turn.turn_id,
            },
            deps=default.deps if default is not None else None,
            execution_session=default.execution_session if default is not None else None,
            scope=default.scope if default is not None else None,
            credentials=default.credentials if default is not None else {},
            effect_key=default.effect_key if default is not None else None,
        )
