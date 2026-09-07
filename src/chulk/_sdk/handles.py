"""Synchronous and asynchronous compatibility handles for SDK agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chulk._sdk.events import DeltaCallback, EventCallback, EventDispatcher
from chulk._sdk.results import (
    PlanResult,
    RunResult,
    plan_result_from_runtime,
    run_result_from_runtime,
)
from chulk.core import Agent as CoreAgent
from chulk.core.context import TurnContextSection
from chulk.media import UserInput
from chulk.results import (
    RunStatus,
)
from chulk.tools import ToolExecutionContext
from chulk.tracing.artifacts import (
    ArtifactReadMode,
    DEFAULT_ARTIFACT_READ_BYTES,
)


class AgentHandle:
    """Provisional handle implementation shared with the synchronous facade."""

    def __init__(self, runtime: CoreAgent, *, on_event: EventCallback | None = None) -> None:
        self._runtime = runtime
        self._events = EventDispatcher(runtime, on_event=on_event)
        self._closed = False

    @property
    def runtime(self) -> CoreAgent:
        return self._runtime

    @runtime.setter
    def runtime(self, value: CoreAgent) -> None:
        self._runtime = value

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def state(self):
        return self.runtime.state

    @property
    def conversation_id(self) -> str:
        return self.runtime.state.conversation_id

    @property
    def trace_path(self) -> Path | None:
        trace_logger = getattr(self.runtime, "trace_logger", None)
        return getattr(trace_logger, "path", None)

    @property
    def tool_registry(self):
        return self.runtime.catalog.active_registry

    @property
    def skill_registry(self):
        return self.runtime.skill_context.registry

    def run(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> str:
        """Run one normal agent turn, optionally receiving streamed answer deltas."""
        return self.run_result(
            message,
            on_delta=on_delta,
            on_event=on_event,
            context_sections=context_sections,
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata,
            tool_context=tool_context,
        ).content

    def run_result(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> RunResult:
        """Run one normal agent turn and return structured SDK metadata."""
        self._ensure_open()
        with self._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = self.runtime.run_turn(
                message,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            )
        return run_result_from_runtime(self.runtime, content)

    def run_input(
        self,
        user_input: UserInput,
        **kwargs: Any,
    ) -> str:
        """Run one typed text/media turn."""
        return self.run_input_result(user_input, **kwargs).content

    def run_input_result(
        self,
        user_input: UserInput,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> RunResult:
        """Run typed input and return structured SDK metadata."""
        self._ensure_open()
        with self._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = self.runtime.run_input(
                user_input,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            )
        return run_result_from_runtime(self.runtime, content)

    def __call__(self, message: str) -> str:
        return self.run(message)

    def plan(self, message: str) -> str:
        """Run one planned turn that pauses for approval before mutation."""
        return self.plan_result(message).content

    def plan_result(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> PlanResult:
        """Run one planned turn and return the created plan snapshot."""
        self._ensure_open()
        with self._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = self.runtime.run_planned_turn(message)
        return plan_result_from_runtime(self.runtime, content)

    def approve(self) -> str:
        """Approve and continue a pending plan."""
        return self.approve_result().content

    def approve_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        """Approve a pending plan or continue a restored approved plan."""
        self._ensure_open()
        has_plan_to_run = (
            self.runtime.has_pending_plan() or self.runtime.has_resumable_plan()
        )
        with self._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = self.runtime.approve_plan()
        if not has_plan_to_run:
            return self._no_pending_plan_result(content)
        return run_result_from_runtime(self.runtime, content)

    def reject(self) -> str:
        """Reject a pending plan or cancel a restored approved plan."""
        return self.reject_result().content

    def reject_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        """Reject or cancel the active plan and return its terminal result."""
        self._ensure_open()
        has_plan_to_cancel = (
            self.runtime.has_pending_plan() or self.runtime.has_resumable_plan()
        )
        with self._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = self.runtime.reject_plan()
        if not has_plan_to_cancel:
            return self._no_pending_plan_result(content)
        return run_result_from_runtime(self.runtime, content)

    def close(self) -> None:
        """Close owned runtime resources exactly once."""
        if self._closed:
            return
        self._closed = True
        self.runtime.close()

    def cancel(self) -> bool:
        """Cooperatively cancel the active synchronous turn, if any."""
        self._ensure_open()
        return self.runtime.cancel_active_turn()

    def read_artifact(
        self,
        artifact_id: str,
        *,
        mode: ArtifactReadMode = "head_tail",
        offset: int = 0,
        max_bytes: int = DEFAULT_ARTIFACT_READ_BYTES,
    ) -> dict[str, Any]:
        """Return a bounded artifact view owned by this conversation."""
        logger = self.runtime.trace_logger
        if logger is None:
            raise RuntimeError("Trace artifacts are unavailable")
        return logger.read_artifact(
            artifact_id,
            mode=mode,
            offset=offset,
            max_bytes=max_bytes,
        ).to_dict()

    async def aclose(self) -> None:
        """Close owned runtime resources exactly once from an async host."""
        if self._closed:
            return
        try:
            await self.runtime.aclose()
        finally:
            if self.runtime.closed:
                self._closed = True

    def __enter__(self) -> "AgentHandle":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Agent is closed")

    def _no_pending_plan_result(self, content: str) -> RunResult:
        return RunResult(
            content=content,
            status=RunStatus.NO_PENDING_PLAN,
            turn_id=None,
            conversation_id=self.conversation_id,
            trace_path=self.trace_path,
        )


class AsyncAgentHandle:
    """Provisional async compatibility handle backed by the synchronous runtime."""

    def __init__(self, handle: AgentHandle) -> None:
        self.handle = handle

    @property
    def runtime(self) -> CoreAgent:
        return self.handle.runtime

    @property
    def state(self):
        return self.handle.state

    @property
    def conversation_id(self) -> str:
        return self.handle.conversation_id

    @property
    def trace_path(self) -> Path | None:
        return self.handle.trace_path

    @property
    def tool_registry(self):
        return self.handle.tool_registry

    @property
    def skill_registry(self):
        return self.handle.skill_registry

    @property
    def closed(self) -> bool:
        return self.handle.closed

    async def run(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> str:
        return (
            await self.run_result(
                message,
                on_delta=on_delta,
                on_event=on_event,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            )
        ).content

    async def run_result(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> RunResult:
        self.handle._ensure_open()
        with self.handle._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = await self.runtime.run_turn_async(
                message,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            )
        return run_result_from_runtime(self.runtime, content)

    async def run_input(
        self,
        user_input: UserInput,
        **kwargs: Any,
    ) -> str:
        return (await self.run_input_result(user_input, **kwargs)).content

    async def run_input_result(
        self,
        user_input: UserInput,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> RunResult:
        self.handle._ensure_open()
        with self.handle._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = await self.runtime.run_input_async(
                user_input,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            )
        return run_result_from_runtime(self.runtime, content)

    async def plan(self, message: str) -> str:
        return (await self.plan_result(message)).content

    async def plan_result(
        self,
        message: str,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> PlanResult:
        self.handle._ensure_open()
        with self.handle._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = await self.runtime.run_planned_turn_async(message)
        return plan_result_from_runtime(self.runtime, content)

    async def approve(self) -> str:
        return (await self.approve_result()).content

    async def approve_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        self.handle._ensure_open()
        has_plan_to_run = (
            self.runtime.has_pending_plan() or self.runtime.has_resumable_plan()
        )
        with self.handle._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = await self.runtime.approve_plan_async()
        if not has_plan_to_run:
            return self.handle._no_pending_plan_result(content)
        return run_result_from_runtime(self.runtime, content)

    async def reject(self) -> str:
        return (await self.reject_result()).content

    async def reject_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        self.handle._ensure_open()
        has_plan_to_cancel = (
            self.runtime.has_pending_plan() or self.runtime.has_resumable_plan()
        )
        with self.handle._events.callbacks(on_delta=on_delta, on_event=on_event):
            content = await self.runtime.reject_plan_async()
        if not has_plan_to_cancel:
            return self.handle._no_pending_plan_result(content)
        return run_result_from_runtime(self.runtime, content)

    async def close(self) -> None:
        await self.handle.aclose()

    async def __aenter__(self) -> "AsyncAgentHandle":
        self.handle._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
