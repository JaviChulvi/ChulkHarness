"""Public Agent facades and provisional compatibility handles."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Iterable, Iterator
from contextlib import suppress
from pathlib import Path
import threading
from typing import Any, Callable, TypeVar

from chulk.capabilities import Capabilities, MemoryMode
from chulk._sdk.config import AgentConfig, AgentPreset, coerce_config, ensure_chat_kwargs
from chulk._sdk.error_mapping import map_public_error
from chulk._sdk.event_channel import RunEventChannel, RunGate
from chulk._sdk.events import DeltaCallback, EventCallback, EventDispatcher, failure_event, terminal_event
from chulk._sdk.results import (
    PlanResult,
    RunResult,
    memory_proposal_snapshot,
    plan_result_from_runtime,
    run_result_from_runtime,
)
from chulk.config import Config
from chulk.core import Agent as CoreAgent
from chulk.core.context import TurnContextSection
from chulk.llm import LLMClient
from chulk.events import AgentEvent, EventName
from chulk.mcp import MCPServerConfig
from chulk.results import MemoryProposal, RunStatus
from chulk.runtime import create_agent as create_runtime_agent
from chulk.tools import ShellExecutionPolicy, ToolExecutionContext
from chulk.tools.permissions import PermissionDecision, PermissionDecisionRecord, PermissionRequest


PermissionCallback = Callable[[PermissionRequest, PermissionDecisionRecord], PermissionDecision | bool]
T = TypeVar("T")


class AgentHandle:
    """Provisional compatibility handle behind the public synchronous facade."""

    def __init__(self, runtime: CoreAgent, *, on_event: EventCallback | None = None) -> None:
        self.runtime = runtime
        self._events = EventDispatcher(runtime, on_event=on_event)
        self._closed = False

    @property
    def _active_on_event(self) -> EventCallback | None:
        return self._events.active_on_event

    @_active_on_event.setter
    def _active_on_event(self, value: EventCallback | None) -> None:
        self._events.active_on_event = value

    @property
    def _active_on_delta(self) -> DeltaCallback | None:
        return self._events.active_on_delta

    @_active_on_delta.setter
    def _active_on_delta(self, value: DeltaCallback | None) -> None:
        self._events.active_on_delta = value

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
        return self.runtime.tool_registry

    @property
    def skill_registry(self):
        return self.runtime.skill_registry

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
        content = self._with_callbacks(
            lambda: self.runtime.run_turn(
                message,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            ),
            on_delta=on_delta,
            on_event=on_event,
        )
        return self._run_result(content)

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
        content = self._with_callbacks(
            lambda: self.runtime.run_planned_turn(message),
            on_delta=on_delta,
            on_event=on_event,
        )
        return self._plan_result(content)

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
        content = self._with_callbacks(lambda: self.runtime.approve_plan(), on_delta=on_delta, on_event=on_event)
        if not has_plan_to_run:
            return self._no_pending_plan_result(content)
        return self._run_result(content)

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
        content = self._with_callbacks(lambda: self.runtime.reject_plan(), on_delta=on_delta, on_event=on_event)
        if not has_plan_to_cancel:
            return self._no_pending_plan_result(content)
        return self._run_result(content)

    def close(self) -> None:
        """Close owned runtime resources exactly once."""
        if self._closed:
            return
        self._closed = True
        self.runtime.close()

    def __enter__(self) -> "AgentHandle":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Agent is closed")

    def _with_callbacks(
        self,
        call: Callable[[], str],
        *,
        on_delta: DeltaCallback | None,
        on_event: EventCallback | None,
    ) -> str:
        previous_on_delta = self._events.active_on_delta
        previous_on_event = self._events.active_on_event
        self._events.active_on_delta = on_delta
        self._events.active_on_event = on_event
        try:
            return call()
        finally:
            self._events.active_on_delta = previous_on_delta
            self._events.active_on_event = previous_on_event

    def _last_turn(self):
        return self.runtime.state.turns[-1] if self.runtime.state.turns else None

    def _run_result(self, content: str) -> RunResult:
        return run_result_from_runtime(self.runtime, content)

    def _no_pending_plan_result(self, content: str) -> RunResult:
        return RunResult(
            content=content,
            status=RunStatus.NO_PENDING_PLAN,
            turn_id=None,
            conversation_id=self.conversation_id,
            trace_path=self.trace_path,
        )

    def _plan_result(self, content: str) -> PlanResult:
        return plan_result_from_runtime(self.runtime, content)


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
        previous_on_delta = self.handle._active_on_delta
        previous_on_event = self.handle._active_on_event
        self.handle._active_on_delta = on_delta
        self.handle._active_on_event = on_event
        try:
            content = await self.runtime.run_turn_async(
                message,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
            )
        finally:
            self.handle._active_on_delta = previous_on_delta
            self.handle._active_on_event = previous_on_event
        return self.handle._run_result(content)

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
        previous_on_delta = self.handle._active_on_delta
        previous_on_event = self.handle._active_on_event
        self.handle._active_on_delta = on_delta
        self.handle._active_on_event = on_event
        try:
            content = await self.runtime.run_planned_turn_async(message)
        finally:
            self.handle._active_on_delta = previous_on_delta
            self.handle._active_on_event = previous_on_event
        return self.handle._plan_result(content)

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
        previous_on_delta = self.handle._active_on_delta
        previous_on_event = self.handle._active_on_event
        self.handle._active_on_delta = on_delta
        self.handle._active_on_event = on_event
        try:
            content = await self.runtime.approve_plan_async()
        finally:
            self.handle._active_on_delta = previous_on_delta
            self.handle._active_on_event = previous_on_event
        if not has_plan_to_run:
            return self.handle._no_pending_plan_result(content)
        return self.handle._run_result(content)

    async def reject(self) -> str:
        self.handle._ensure_open()
        return await asyncio.to_thread(self.handle.reject)

    async def reject_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        self.handle._ensure_open()
        return await asyncio.to_thread(self.handle.reject_result, on_delta=on_delta, on_event=on_event)

    async def close(self) -> None:
        self.handle.close()

    async def __aenter__(self) -> "AsyncAgentHandle":
        self.handle._ensure_open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


class Agent:
    """Public synchronous Chulk SDK facade."""

    def __init__(
        self,
        *,
        config: Config | AgentConfig | None = None,
        preset: AgentPreset | None = None,
        llm: LLMClient | Any | None = None,
        tools: Iterable[object] | None = None,
        skills: object | Iterable[object] | None = None,
        system_prompt: str | None = None,
        conversation_id: str | None = None,
        permission_callback: PermissionCallback | None = None,
        on_event: EventCallback | None = None,
        mcp: Iterable[MCPServerConfig] | None = None,
        redaction_callback: Callable[[str, str, dict], str] | None = None,
        redaction_fail_closed: bool = False,
        capabilities: Capabilities | None = None,
        memory_mode: MemoryMode | str | None = None,
        deps: object | None = None,
        shell_execution_policy: ShellExecutionPolicy | None = None,
        require_shell_containment: bool = False,
    ) -> None:
        selected_capabilities = _selected_capabilities(config, capabilities, memory_mode)
        try:
            self._handle = _build_handle(
                config=config,
                preset=preset,
                llm=llm,
                tools=tools,
                skills=skills,
                system_prompt=system_prompt,
                conversation_id=conversation_id,
                permission_callback=permission_callback,
                on_event=on_event,
                mcp=mcp,
                redaction_callback=redaction_callback,
                redaction_fail_closed=redaction_fail_closed,
                capabilities=selected_capabilities,
                deps=deps,
                shell_execution_policy=shell_execution_policy,
                require_shell_containment=require_shell_containment,
            )
        except Exception as exc:
            mapped = map_public_error(exc, config=config, operation="construct")
            if mapped is exc:
                raise
            raise mapped from exc
        self._run_gate = RunGate()
        self._capabilities = selected_capabilities
        self._deps = deps

    @property
    def runtime(self) -> CoreAgent:
        return self._handle.runtime

    @property
    def state(self):
        return self._handle.state

    @property
    def conversation_id(self) -> str:
        return self._handle.conversation_id

    @property
    def trace_path(self) -> Path | None:
        return self._handle.trace_path

    @property
    def tool_registry(self):
        return self._handle.tool_registry

    @property
    def skill_registry(self):
        return self._handle.skill_registry

    @property
    def closed(self) -> bool:
        return self._handle.closed

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def run(self, message: str, **kwargs: Any) -> str:
        options = self._run_options(kwargs)
        return self._invoke("run", lambda: self._handle.run(message, **options), serialized=True)

    def run_result(self, message: str, **kwargs: Any) -> RunResult:
        options = self._run_options(kwargs)
        return self._invoke("run_result", lambda: self._handle.run_result(message, **options), serialized=True)

    def __call__(self, message: str) -> str:
        return self.run(message)

    def plan(self, message: str) -> str:
        return self._invoke("plan", lambda: self._handle.plan(message), serialized=True)

    def plan_result(self, message: str, **kwargs: Any) -> PlanResult:
        return self._invoke("plan_result", lambda: self._handle.plan_result(message, **kwargs), serialized=True)

    def approve(self) -> str:
        return self._invoke("approve", self._handle.approve, serialized=True)

    def approve_result(self, **kwargs: Any) -> RunResult:
        return self._invoke("approve_result", lambda: self._handle.approve_result(**kwargs), serialized=True)

    def reject(self) -> str:
        return self._invoke("reject", self._handle.reject, serialized=True)

    def reject_result(self, **kwargs: Any) -> RunResult:
        return self._invoke("reject_result", lambda: self._handle.reject_result(**kwargs), serialized=True)

    def close(self) -> None:
        self._invoke("close", self._handle.close, serialized=True)

    def list_memory_proposals(self) -> tuple[MemoryProposal, ...]:
        """Return pending manual-memory proposals as immutable snapshots."""
        def operation() -> tuple[MemoryProposal, ...]:
            policy = self.runtime.memory_policy
            if policy is None:
                return ()
            return tuple(memory_proposal_snapshot(item) for item in policy.list_pending())

        return self._invoke("list_memory_proposals", operation)

    def approve_memory_proposal(self, proposal_id: str) -> MemoryProposal:
        """Approve one pending memory proposal."""
        def operation() -> MemoryProposal:
            policy = self.runtime.memory_policy
            if policy is None:
                raise RuntimeError("Memory is not configured")
            return memory_proposal_snapshot(policy.approve(proposal_id))

        return self._invoke("approve_memory_proposal", operation)

    def reject_memory_proposal(self, proposal_id: str) -> MemoryProposal:
        """Reject one pending memory proposal."""
        def operation() -> MemoryProposal:
            policy = self.runtime.memory_policy
            if policy is None:
                raise RuntimeError("Memory is not configured")
            return memory_proposal_snapshot(policy.reject(proposal_id))

        return self._invoke("reject_memory_proposal", operation)

    def run_events(self, message: str, **kwargs: Any) -> Iterator[AgentEvent]:
        """Yield one run's ordered public events, including its terminal result."""
        channel = RunEventChannel()
        caller_on_event = kwargs.pop("on_event", None)
        attempted_turn_id: str | None = None

        def on_event(event: AgentEvent) -> None:
            nonlocal attempted_turn_id
            if event.name == EventName.RUN_STARTED.value:
                attempted_turn_id = event.turn_id
            if event.name not in {EventName.RUN_COMPLETED.value, EventName.RUN_FAILED.value}:
                channel.publish(event)
            if caller_on_event is not None:
                caller_on_event(event)

        def work() -> None:
            try:
                result = self.run_result(message, on_event=on_event, **kwargs)
            except Exception as exc:
                terminalized = _terminalized_failure_event(self.runtime, attempted_turn_id)
                event = terminalized or failure_event(
                    exc, conversation_id=self.conversation_id, turn_id=attempted_turn_id
                )
                channel.finish(event)
                if terminalized is None:
                    _notify_event_callback_safely(caller_on_event, event)
            else:
                channel.finish(terminal_event(result))

        worker = threading.Thread(target=work, name="chulk-run-events")
        worker.start()
        yield from channel.iterate(worker)

    def __enter__(self) -> "Agent":
        self._invoke("enter", self._handle._ensure_open)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _invoke(self, operation: str, call: Callable[[], T], *, serialized: bool = False) -> T:
        try:
            if serialized:
                with self._run_gate.hold():
                    return call()
            return call()
        except Exception as exc:
            mapped = map_public_error(exc, runtime=self.runtime, operation=operation)
            if mapped is exc:
                raise
            raise mapped from exc

    def _run_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        options = dict(kwargs)
        deps = options.pop("deps", None)
        if deps is not None:
            if options.get("tool_context") is not None:
                raise ValueError("Pass either deps or tool_context, not both")
            options["tool_context"] = ToolExecutionContext(deps=deps)
        return options


class AsyncAgent:
    """Public asynchronous facade backed by Chulk's compatibility runtime."""

    def __init__(self, **kwargs: Any) -> None:
        self._agent = Agent(**kwargs)
        self._handle = AsyncAgentHandle(self._agent._handle)
        self._async_run_gate = asyncio.Lock()

    @property
    def runtime(self) -> CoreAgent:
        return self._handle.runtime

    @property
    def state(self):
        return self._handle.state

    @property
    def conversation_id(self) -> str:
        return self._handle.conversation_id

    @property
    def trace_path(self) -> Path | None:
        return self._handle.trace_path

    @property
    def tool_registry(self):
        return self._handle.tool_registry

    @property
    def skill_registry(self):
        return self._handle.skill_registry

    @property
    def closed(self) -> bool:
        return self._handle.closed

    @property
    def capabilities(self) -> Capabilities:
        return self._agent.capabilities

    async def run(self, message: str, **kwargs: Any) -> str:
        options = self._agent._run_options(kwargs)
        return await self._invoke_async("run", lambda: self._handle.run(message, **options), serialized=True)

    async def run_result(self, message: str, **kwargs: Any) -> RunResult:
        options = self._agent._run_options(kwargs)
        return await self._invoke_async(
            "run_result",
            lambda: self._handle.run_result(message, **options),
            serialized=True,
        )

    async def plan(self, message: str) -> str:
        return await self._invoke_async("plan", lambda: self._handle.plan(message), serialized=True)

    async def plan_result(self, message: str, **kwargs: Any) -> PlanResult:
        return await self._invoke_async(
            "plan_result",
            lambda: self._handle.plan_result(message, **kwargs),
            serialized=True,
        )

    async def approve(self) -> str:
        return await self._invoke_async("approve", self._handle.approve, serialized=True)

    async def approve_result(self, **kwargs: Any) -> RunResult:
        return await self._invoke_async(
            "approve_result",
            lambda: self._handle.approve_result(**kwargs),
            serialized=True,
        )

    async def reject(self) -> str:
        return await self._invoke_async("reject", self._handle.reject, serialized=True)

    async def reject_result(self, **kwargs: Any) -> RunResult:
        return await self._invoke_async(
            "reject_result",
            lambda: self._handle.reject_result(**kwargs),
            serialized=True,
        )

    async def close(self) -> None:
        await self._invoke_async("close", self._handle.close, serialized=True)

    async def list_memory_proposals(self) -> tuple[MemoryProposal, ...]:
        return await asyncio.to_thread(self._agent.list_memory_proposals)

    async def approve_memory_proposal(self, proposal_id: str) -> MemoryProposal:
        return await asyncio.to_thread(self._agent.approve_memory_proposal, proposal_id)

    async def reject_memory_proposal(self, proposal_id: str) -> MemoryProposal:
        return await asyncio.to_thread(self._agent.reject_memory_proposal, proposal_id)

    async def run_events_async(self, message: str, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        """Asynchronously yield one run's ordered public events and terminal result."""
        channel = RunEventChannel()
        caller_on_event = kwargs.pop("on_event", None)
        attempted_turn_id: str | None = None

        def on_event(event: AgentEvent) -> None:
            nonlocal attempted_turn_id
            if event.name == EventName.RUN_STARTED.value:
                attempted_turn_id = event.turn_id
            if event.name not in {EventName.RUN_COMPLETED.value, EventName.RUN_FAILED.value}:
                channel.publish(event)
            if caller_on_event is not None:
                caller_on_event(event)

        async def work() -> None:
            try:
                result = await self.run_result(message, on_event=on_event, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                terminalized = _terminalized_failure_event(self.runtime, attempted_turn_id)
                event = terminalized or failure_event(
                    exc, conversation_id=self.conversation_id, turn_id=attempted_turn_id
                )
                channel.finish(event)
                if terminalized is None:
                    _notify_event_callback_safely(caller_on_event, event)
            else:
                channel.finish(terminal_event(result))

        worker = asyncio.create_task(work())
        try:
            while True:
                item = await asyncio.to_thread(channel.get)
                if not isinstance(item, AgentEvent):
                    break
                yield item
        finally:
            channel.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def __aenter__(self) -> "AsyncAgent":
        self._agent._invoke("enter", self._agent._handle._ensure_open)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _invoke_async(
        self,
        operation: str,
        call: Callable[[], Awaitable[T]],
        *,
        serialized: bool = False,
    ) -> T:
        try:
            if serialized:
                async with self._async_run_gate:
                    return await call()
            return await call()
        except Exception as exc:
            mapped = map_public_error(exc, runtime=self.runtime, operation=operation)
            if mapped is exc:
                raise
            raise mapped from exc


def _notify_event_callback_safely(callback: EventCallback | None, event: AgentEvent) -> None:
    """Best-effort delivery after a callback itself caused run failure."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        return


def _terminalized_failure_event(runtime: CoreAgent, attempted_turn_id: str | None) -> AgentEvent | None:
    if attempted_turn_id is None:
        return None
    result = run_result_from_runtime(runtime)
    if (
        result.turn_id == attempted_turn_id
        and result.status in {RunStatus.FAILED, RunStatus.BLOCKED, RunStatus.CANCELLED}
    ):
        return terminal_event(result)
    return None


def _build_handle(
    *,
    config: Config | AgentConfig | None = None,
    preset: AgentPreset | None = None,
    llm: LLMClient | Any | None = None,
    tools: Iterable[object] | None = None,
    skills: object | Iterable[object] | None = None,
    system_prompt: str | None = None,
    conversation_id: str | None = None,
    permission_callback: PermissionCallback | None = None,
    on_event: EventCallback | None = None,
    mcp: Iterable[MCPServerConfig] | None = None,
    redaction_callback: Callable[[str, str, dict], str] | None = None,
    redaction_fail_closed: bool = False,
    capabilities: Capabilities | None = None,
    deps: object | None = None,
    shell_execution_policy: ShellExecutionPolicy | None = None,
    require_shell_containment: bool = False,
) -> AgentHandle:
    runtime_config = coerce_config(config)
    selected_tools = tools if tools is not None else (preset.tools if preset is not None else None)
    selected_skills = skills if skills is not None else (preset.skills if preset is not None else None)
    selected_prompt = system_prompt or (preset.system_prompt if preset is not None else None)
    runtime = create_runtime_agent(
        runtime_config,
        conversation_id=conversation_id,
        llm_client=llm,
        tool_specs=selected_tools,
        skill_specs=selected_skills,
        system_prompt=selected_prompt,
        permission_callback=permission_callback,
        mcp_servers=tuple(mcp) if mcp is not None else None,
        redaction_callback=redaction_callback,
        redaction_fail_closed=redaction_fail_closed,
        capabilities=capabilities,
        deps=deps,
        shell_execution_policy=shell_execution_policy,
        require_shell_containment=require_shell_containment,
    )
    return AgentHandle(runtime, on_event=on_event)


def _selected_capabilities(
    config: Config | AgentConfig | None,
    capabilities: Capabilities | None,
    memory_mode: MemoryMode | str | None,
) -> Capabilities:
    selected = capabilities
    if selected is None and isinstance(config, AgentConfig):
        selected = config.resolved_capabilities()
    if selected is None:
        selected = Capabilities.read_only()
    if memory_mode is not None:
        selected = selected.with_memory(memory_mode)
    return selected


def agent(**kwargs: Any) -> Agent:
    """Create the public synchronous Agent facade."""
    return Agent(**kwargs)


def chat_agent(**kwargs: Any) -> Agent:
    """Create a plain chat Agent with no tools or skills configured."""
    ensure_chat_kwargs(kwargs)
    return Agent(tools=[], skills=[], **kwargs)


def ChatAgent(**kwargs: Any) -> Agent:
    """Compatibility constructor for a plain chat Agent."""
    return chat_agent(**kwargs)


def async_agent(**kwargs: Any) -> AsyncAgent:
    """Create the public asynchronous Agent facade."""
    return AsyncAgent(**kwargs)


def async_chat_agent(**kwargs: Any) -> AsyncAgent:
    """Create an async plain chat Agent with no tools or skills configured."""
    ensure_chat_kwargs(kwargs)
    return AsyncAgent(tools=[], skills=[], **kwargs)


def AsyncChatAgent(**kwargs: Any) -> AsyncAgent:
    """Compatibility constructor for an async plain chat Agent."""
    return async_chat_agent(**kwargs)


__all__ = [
    "Agent",
    "AgentHandle",
    "AsyncAgent",
    "AsyncAgentHandle",
    "AsyncChatAgent",
    "ChatAgent",
    "agent",
    "async_agent",
    "async_chat_agent",
    "chat_agent",
]
