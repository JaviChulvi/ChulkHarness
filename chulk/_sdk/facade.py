"""Public Agent facades and provisional compatibility handles."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Iterable, Iterator
from contextlib import suppress
from pathlib import Path
import threading
from typing import Any, Callable, TypeVar, cast
from uuid import uuid4

from chulk.capabilities import Capabilities, MemoryMode
from chulk._sdk.config import AgentConfig, AgentPreset, coerce_config, ensure_chat_kwargs
from chulk._sdk.error_mapping import map_public_error
from chulk._sdk.event_channel import RunEventChannel, RunGate
from chulk._sdk.events import DeltaCallback, EventCallback, EventDispatcher, failure_event, terminal_event
from chulk._sdk.results import (
    PlanResult,
    RunResult,
    governed_skill_snapshot,
    governed_skill_revision_snapshot,
    learning_proposal_snapshot,
    memory_proposal_snapshot,
    plan_result_from_runtime,
    run_result_from_runtime,
)
from chulk.config import Config
from chulk.core import Agent as CoreAgent
from chulk.core.context import TurnContextSection
from chulk.llm import LLMClient
from chulk.events import AgentEvent, EventName
from chulk.execution import ExecutionBackend
from chulk.goals import GoalExecutionContext
from chulk.hosting import (
    AsyncRuntimeServices,
    AsyncServiceBinding,
    ExecutionScope,
    RuntimeServices,
)
from chulk.hosting.async_utils import call_async_service
from chulk.hosting.services import ResolvedRuntimeServices
from chulk.mcp import MCPServerConfig
from chulk.media import ContentStore, MediaProcessorRegistry, UserInput
from chulk.plugins import (
    LoadedPluginEntryPoint,
    LocalPluginRegistry,
    PluginAuditReport,
    PluginCategory,
    PluginInspection,
    PluginLifecycleReceipt,
    PluginLockEntry,
    PluginUpdatePlan,
)
from chulk.results import (
    GovernedSkill,
    GovernedSkillRevision,
    LearningProposal,
    LearningReview,
    MemoryProposal,
    RunStatus,
)
from chulk.skills import (
    LearningProposalStatus,
    LearningReviewPolicy,
    LearningReviewQuota,
    SkillScope,
    SkillUsageKind,
)
from chulk.runtime import (
    create_agent as create_runtime_agent,
    create_async_hosted_agent,
)
from chulk.sessions import (
    SessionSearchPage,
    SessionSearchService,
    SessionWindow,
)
from chulk.tools import ShellExecutionPolicy, ToolExecutionContext
from chulk.tools.permissions import PermissionDecision, PermissionDecisionRecord, PermissionRequest
from chulk.tracing.artifacts import (
    ArtifactReadMode,
    DEFAULT_ARTIFACT_READ_BYTES,
)
from chulk.usage import (
    RunBudget,
    UsageAggregate,
    UsageDimensions,
    UsageGroupBy,
    UsageLedger,
    UsagePage,
)


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
        content = self._with_callbacks(
            lambda: self.runtime.run_input(
                user_input,
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
        self._closed = True
        await self.runtime.aclose()

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
        previous_on_delta = self.handle._active_on_delta
        previous_on_event = self.handle._active_on_event
        self.handle._active_on_delta = on_delta
        self.handle._active_on_event = on_event
        try:
            content = await self.runtime.run_input_async(
                user_input,
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
        return (await self.reject_result()).content

    async def reject_result(
        self,
        *,
        on_delta: DeltaCallback | None = None,
        on_event: EventCallback | None = None,
    ) -> RunResult:
        self.handle._ensure_open()
        has_plan_to_cancel = (
            self.runtime.has_pending_plan()
            or self.runtime.has_resumable_plan()
        )
        previous_on_delta = self.handle._active_on_delta
        previous_on_event = self.handle._active_on_event
        self.handle._active_on_delta = on_delta
        self.handle._active_on_event = on_event
        try:
            content = await self.runtime.reject_plan_async()
        finally:
            self.handle._active_on_delta = previous_on_delta
            self.handle._active_on_event = previous_on_event
        if not has_plan_to_cancel:
            return self.handle._no_pending_plan_result(content)
        return self.handle._run_result(content)

    async def close(self) -> None:
        await self.handle.aclose()

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
        conversation_metadata: dict[str, object] | None = None,
        runtime_metadata: dict[str, object] | None = None,
        permission_callback: PermissionCallback | None = None,
        on_event: EventCallback | None = None,
        mcp: Iterable[MCPServerConfig] | None = None,
        redaction_callback: Callable[[str, str, dict], str] | None = None,
        redaction_fail_closed: bool = False,
        capabilities: Capabilities | None = None,
        memory_mode: MemoryMode | str | None = None,
        memory_namespace: str | None = None,
        deps: object | None = None,
        shell_execution_policy: ShellExecutionPolicy | None = None,
        require_shell_containment: bool = False,
        execution_backend: ExecutionBackend | None = None,
        run_budget: RunBudget | None = None,
        usage_dimensions: UsageDimensions | None = None,
        learning_review_policy: LearningReviewPolicy | None = None,
        learning_review_quota: LearningReviewQuota | None = None,
        automatic_learning_approval: bool = False,
        plugin_registry: LocalPluginRegistry | None = None,
        goal_execution: GoalExecutionContext | None = None,
        content_store: ContentStore | None = None,
        media_processors: MediaProcessorRegistry | None = None,
        services: RuntimeServices | None = None,
        execution_scope: ExecutionScope | None = None,
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
                conversation_metadata=conversation_metadata,
                runtime_metadata=runtime_metadata,
                permission_callback=permission_callback,
                on_event=on_event,
                mcp=mcp,
                redaction_callback=redaction_callback,
                redaction_fail_closed=redaction_fail_closed,
                capabilities=selected_capabilities,
                memory_namespace=memory_namespace,
                deps=deps,
                shell_execution_policy=shell_execution_policy,
                require_shell_containment=require_shell_containment,
                execution_backend=execution_backend,
                run_budget=run_budget,
                usage_dimensions=usage_dimensions,
                learning_review_policy=learning_review_policy,
                learning_review_quota=learning_review_quota,
                automatic_learning_approval=automatic_learning_approval,
                plugin_registry=plugin_registry,
                goal_execution=goal_execution,
                content_store=content_store,
                media_processors=media_processors,
                services=services,
                execution_scope=execution_scope,
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
    def execution_scope(self) -> ExecutionScope:
        return cast(ExecutionScope, self.runtime.execution_scope)

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

    def run_input(self, user_input: UserInput, **kwargs: Any) -> str:
        options = self._run_options(kwargs)
        return self._invoke(
            "run_input",
            lambda: self._handle.run_input(user_input, **options),
            serialized=True,
        )

    def run_input_result(self, user_input: UserInput, **kwargs: Any) -> RunResult:
        options = self._run_options(kwargs)
        return self._invoke(
            "run_input_result",
            lambda: self._handle.run_input_result(user_input, **options),
            serialized=True,
        )

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

    def list_learning_proposals(
        self,
        *,
        status: str | None = "pending",
        limit: int = 100,
    ) -> tuple[LearningProposal, ...]:
        """Return the unified memory and skill review queue."""
        def operation() -> tuple[LearningProposal, ...]:
            service = self.runtime.learning_proposals
            if service is None:
                return ()
            normalized = (
                None if status is None else LearningProposalStatus(status)
            )
            return tuple(
                learning_proposal_snapshot(item)
                for item in service.list(status=normalized, limit=limit)
            )

        return self._invoke("list_learning_proposals", operation)

    def get_learning_proposal(
        self,
        proposal_id: str,
    ) -> LearningProposal:
        """Return one proposal from the unified review queue."""
        def operation() -> LearningProposal:
            service = self.runtime.learning_proposals
            if service is None:
                raise RuntimeError("learning proposals are not configured")
            return learning_proposal_snapshot(service.get(proposal_id))

        return self._invoke("get_learning_proposal", operation)

    def approve_learning_proposal(
        self,
        proposal_id: str,
        *,
        approved_by: str = "sdk-host",
    ) -> LearningProposal:
        """Apply one explicitly approved learning proposal."""
        def operation() -> LearningProposal:
            service = self.runtime.learning_proposals
            if service is None:
                raise RuntimeError("learning proposals are not configured")
            return learning_proposal_snapshot(
                service.approve(
                    proposal_id,
                    approved_by=approved_by,
                )
            )

        return self._invoke("approve_learning_proposal", operation)

    def review_learning(
        self,
        *,
        trigger: str = "manual",
        turn_id: str | None = None,
        host_confirmed_success: bool = False,
    ) -> LearningReview:
        """Run the restricted reviewer against one finished turn."""
        def operation() -> LearningReview:
            outcome = self.runtime.review_learning(
                trigger=trigger,
                turn_id=turn_id,
                host_confirmed_success=host_confirmed_success,
            )
            service = self.runtime.learning_proposals
            assert service is not None
            return LearningReview(
                skipped=outcome.skipped,
                rationale=outcome.rationale,
                proposals=tuple(
                    learning_proposal_snapshot(service.get(proposal_id))
                    for proposal_id in outcome.proposal_ids
                ),
                review_run_id=outcome.review_run_id,
            )

        return self._invoke("review_learning", operation, serialized=True)

    def reject_learning_proposal(
        self,
        proposal_id: str,
        *,
        rejected_by: str = "sdk-host",
    ) -> LearningProposal:
        """Reject one learning proposal without applying it."""
        def operation() -> LearningProposal:
            service = self.runtime.learning_proposals
            if service is None:
                raise RuntimeError("learning proposals are not configured")
            return learning_proposal_snapshot(
                service.reject(
                    proposal_id,
                    rejected_by=rejected_by,
                )
            )

        return self._invoke("reject_learning_proposal", operation)

    def list_governed_skills(
        self,
        *,
        scope: str | None = None,
    ) -> tuple[GovernedSkill, ...]:
        """List governed skills and record an explicit host view."""
        def operation() -> tuple[GovernedSkill, ...]:
            store = self.runtime.skill_lifecycle_store
            if store is None:
                return ()
            event_id = f"sdk-view:{uuid4()}"
            records = store.list_skills(scope=scope)
            viewed = tuple(
                store.record_usage(
                    name=record.name,
                    scope=record.scope,
                    version=record.version,
                    digest=record.digest,
                    kind=SkillUsageKind.VIEW,
                    source_event_id=event_id,
                )
                for record in records
            )
            return tuple(governed_skill_snapshot(item) for item in viewed)

        return self._invoke("list_governed_skills", operation)

    def rollback_skill(
        self,
        revision_id: str,
        *,
        scope: str = "project",
        approved_by: str = "sdk-host",
    ) -> GovernedSkill:
        """Restore one immutable skill revision through the host boundary."""
        def operation() -> GovernedSkill:
            lifecycle = self.runtime.skill_lifecycle
            if lifecycle is None:
                raise RuntimeError("skill lifecycle is not configured")
            if scope not in {"project", "profile"}:
                raise ValueError("scope must be project or profile")
            return governed_skill_snapshot(
                lifecycle.rollback(
                    revision_id,
                    scope=cast(SkillScope, scope),
                    approved_by=approved_by,
                )
            )

        return self._invoke("rollback_skill", operation)

    def list_skill_revisions(
        self,
        name: str,
        *,
        scope: str = "project",
        limit: int = 100,
    ) -> tuple[GovernedSkillRevision, ...]:
        """List immutable revision identities available for rollback."""
        def operation() -> tuple[GovernedSkillRevision, ...]:
            store = self.runtime.skill_lifecycle_store
            if store is None:
                return ()
            return tuple(
                governed_skill_revision_snapshot(item)
                for item in store.list_revisions(
                    name,
                    scope=scope,
                    limit=limit,
                )
            )

        return self._invoke("list_skill_revisions", operation)

    def confirm_skill_success(
        self,
        *,
        turn_id: str | None = None,
    ) -> tuple[GovernedSkill, ...]:
        """Record host-confirmed success for skills used by a completed run."""
        return self._invoke(
            "confirm_skill_success",
            lambda: tuple(
                governed_skill_snapshot(item)
                for item in self.runtime.confirm_skill_success(turn_id=turn_id)
            ),
        )

    def inspect_plugin(self, path: Path | str) -> PluginInspection:
        """Inspect a local plugin package without importing its code."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "inspect_plugin",
            lambda: registry.inspect(path),
        )

    def register_local_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLockEntry:
        """Register one exact local package through an explicit host action."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "register_local_plugin",
            lambda: registry.register_local(
                path,
                approved_by=approved_by,
                acknowledge_host_authority=acknowledge_host_authority,
                granted_capabilities=granted_capabilities,
            ),
            serialized=True,
        )

    def install_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLifecycleReceipt:
        """Quarantine and install an exact directory or prebuilt wheel."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "install_plugin",
            lambda: registry.install(
                path,
                approved_by=approved_by,
                acknowledge_host_authority=acknowledge_host_authority,
                granted_capabilities=granted_capabilities,
            ),
            serialized=True,
        )

    def plan_plugin_update(
        self,
        path: Path | str,
    ) -> PluginUpdatePlan:
        """Return a static update and authority diff without enabling it."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "plan_plugin_update",
            lambda: registry.plan_update(path),
        )

    def update_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] | None = None,
        approve_authority_changes: bool = False,
    ) -> PluginLifecycleReceipt:
        """Apply one reviewed update with rollback boundaries."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "update_plugin",
            lambda: registry.update(
                path,
                approved_by=approved_by,
                acknowledge_host_authority=acknowledge_host_authority,
                granted_capabilities=granted_capabilities,
                approve_authority_changes=approve_authority_changes,
            ),
            serialized=True,
        )

    def uninstall_plugin(
        self,
        plugin_name: str,
        *,
        approved_by: str,
    ) -> PluginLifecycleReceipt:
        """Disable a plugin while retaining exact recovery metadata."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "uninstall_plugin",
            lambda: registry.uninstall(
                plugin_name,
                approved_by=approved_by,
            ),
            serialized=True,
        )

    def rollback_plugin(
        self,
        plugin_name: str,
        *,
        approved_by: str,
    ) -> PluginLifecycleReceipt:
        """Restore the newest valid plugin recovery point."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "rollback_plugin",
            lambda: registry.rollback(
                plugin_name,
                approved_by=approved_by,
            ),
            serialized=True,
        )

    def revoke_plugin(
        self,
        plugin_name: str,
        *,
        reason: str,
        revoked_by: str,
    ) -> PluginLifecycleReceipt:
        """Revoke one exact installed digest and fail closed at startup."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "revoke_plugin",
            lambda: registry.revoke(
                plugin_name,
                reason=reason,
                revoked_by=revoked_by,
            ),
            serialized=True,
        )

    def list_plugins(self) -> tuple[PluginLockEntry, ...]:
        """List reviewed plugin registrations without importing them."""
        registry = self.runtime.plugin_registry
        if registry is None:
            return ()
        return self._invoke("list_plugins", registry.list)

    def audit_plugins(self) -> PluginAuditReport:
        """Recheck exact plugin identities without importing plugin code."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke("audit_plugins", registry.audit)

    def load_plugin_entry_point(
        self,
        plugin_name: str,
        category: PluginCategory | str,
        entry_name: str,
        *,
        available_capabilities: tuple[str, ...] = (),
    ) -> LoadedPluginEntryPoint:
        """Import an exact reviewed factory through the host SDK boundary."""
        registry = self.runtime.plugin_registry
        if registry is None:
            raise RuntimeError("plugin registry is not configured")
        return self._invoke(
            "load_plugin_entry_point",
            lambda: registry.load_entry_point(
                plugin_name,
                category,
                entry_name,
                available_capabilities=available_capabilities,
            ),
            serialized=True,
        )

    @property
    def usage_ledger(self) -> UsageLedger:
        """Return a query facade bound to this runtime's profile database."""
        accounting = self.runtime.usage_accounting
        if accounting is None:  # pragma: no cover - runtime assembly always supplies it
            raise RuntimeError("Usage accounting is not configured")
        return UsageLedger(
            accounting.store.db_path,
            profile_id=self.runtime.profile_id,
        )

    def query_usage(self, **kwargs: Any) -> UsagePage:
        """Query profile-owned durable usage without prompt or credential data."""
        return self._invoke(
            "query_usage",
            lambda: self.usage_ledger.query(**kwargs),
        )

    def group_usage(
        self,
        group_by: UsageGroupBy,
        **kwargs: Any,
    ) -> tuple[UsageAggregate, ...]:
        """Return exact grouped totals from the profile-owned usage ledger."""
        return self._invoke(
            "group_usage",
            lambda: self.usage_ledger.group(group_by, **kwargs),
        )

    @property
    def session_search(self) -> SessionSearchService:
        """Return the exact-search service bound to this runtime profile."""
        service = getattr(self.runtime, "session_search_service", None)
        if not isinstance(service, SessionSearchService):
            raise RuntimeError("Session search is not configured")
        return service

    def search_sessions(
        self,
        query: str,
        *,
        limit: int = 10,
        cursor: str | None = None,
    ) -> SessionSearchPage:
        """Search eligible profile-owned prior-session messages."""
        return self._invoke(
            "search_sessions",
            lambda: self.session_search.search(
                query,
                limit=limit,
                cursor=cursor,
            ),
        )

    def read_session_window(
        self,
        conversation_id: str,
        *,
        ordinal: int,
        before: int = 3,
        after: int = 3,
        limit: int = 20,
        cursor: str | None = None,
        include_sensitive: bool = False,
    ) -> SessionWindow:
        """Read a bounded session window, with an explicit trusted-host override."""
        return self._invoke(
            "read_session_window",
            lambda: self.session_search.read_window(
                conversation_id,
                ordinal=ordinal,
                before=before,
                after=after,
                limit=limit,
                cursor=cursor,
                include_sensitive=include_sensitive,
            ),
        )

    def read_artifact(
        self,
        artifact_id: str,
        *,
        mode: ArtifactReadMode = "head_tail",
        offset: int = 0,
        max_bytes: int = DEFAULT_ARTIFACT_READ_BYTES,
    ) -> dict[str, Any]:
        """Read one bounded artifact view through the agent ownership boundary."""
        return self._invoke(
            "read_artifact",
            lambda: self._handle.read_artifact(
                artifact_id,
                mode=mode,
                offset=offset,
                max_bytes=max_bytes,
            ),
        )

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
        last_event_id: str | None = None

        def on_event(event: AgentEvent) -> None:
            nonlocal attempted_turn_id, last_event_id
            if event.name == EventName.RUN_STARTED.value:
                attempted_turn_id = event.turn_id
            if event.name not in {EventName.RUN_COMPLETED.value, EventName.RUN_FAILED.value}:
                channel.publish(event)
                last_event_id = event.event_id
            if caller_on_event is not None:
                caller_on_event(event)

        def work() -> None:
            try:
                result = self.run_result(message, on_event=on_event, **kwargs)
            except Exception as exc:
                terminalized = _terminalized_failure_event(
                    self.runtime,
                    attempted_turn_id,
                    causation_id=last_event_id,
                )
                event = terminalized or failure_event(
                    exc,
                    conversation_id=self.conversation_id,
                    turn_id=attempted_turn_id,
                    profile_id=self.runtime.profile_id,
                    execution_scope=self.runtime.execution_scope,
                    causation_id=last_event_id,
                )
                channel.finish(event)
                if terminalized is None:
                    _notify_event_callback_safely(caller_on_event, event)
            else:
                channel.finish(
                    terminal_event(
                        result,
                        profile_id=self.runtime.profile_id,
                        execution_scope=self.runtime.execution_scope,
                        causation_id=last_event_id,
                    )
                )

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
    def execution_scope(self) -> ExecutionScope:
        return cast(ExecutionScope, self.runtime.execution_scope)

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

    async def run_input(self, user_input: UserInput, **kwargs: Any) -> str:
        options = self._agent._run_options(kwargs)
        return await self._invoke_async(
            "run_input",
            lambda: self._handle.run_input(user_input, **options),
            serialized=True,
        )

    async def run_input_result(
        self,
        user_input: UserInput,
        **kwargs: Any,
    ) -> RunResult:
        options = self._agent._run_options(kwargs)
        return await self._invoke_async(
            "run_input_result",
            lambda: self._handle.run_input_result(user_input, **options),
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

    async def list_learning_proposals(
        self,
        *,
        status: str | None = "pending",
        limit: int = 100,
    ) -> tuple[LearningProposal, ...]:
        return await asyncio.to_thread(
            self._agent.list_learning_proposals,
            status=status,
            limit=limit,
        )

    async def get_learning_proposal(
        self,
        proposal_id: str,
    ) -> LearningProposal:
        return await asyncio.to_thread(
            self._agent.get_learning_proposal,
            proposal_id,
        )

    async def approve_learning_proposal(
        self,
        proposal_id: str,
        *,
        approved_by: str = "sdk-host",
    ) -> LearningProposal:
        return await asyncio.to_thread(
            self._agent.approve_learning_proposal,
            proposal_id,
            approved_by=approved_by,
        )

    async def review_learning(
        self,
        *,
        trigger: str = "manual",
        turn_id: str | None = None,
        host_confirmed_success: bool = False,
    ) -> LearningReview:
        return await asyncio.to_thread(
            self._agent.review_learning,
            trigger=trigger,
            turn_id=turn_id,
            host_confirmed_success=host_confirmed_success,
        )

    async def reject_learning_proposal(
        self,
        proposal_id: str,
        *,
        rejected_by: str = "sdk-host",
    ) -> LearningProposal:
        return await asyncio.to_thread(
            self._agent.reject_learning_proposal,
            proposal_id,
            rejected_by=rejected_by,
        )

    async def list_governed_skills(
        self,
        *,
        scope: str | None = None,
    ) -> tuple[GovernedSkill, ...]:
        return await asyncio.to_thread(
            self._agent.list_governed_skills,
            scope=scope,
        )

    async def rollback_skill(
        self,
        revision_id: str,
        *,
        scope: str = "project",
        approved_by: str = "sdk-host",
    ) -> GovernedSkill:
        return await asyncio.to_thread(
            self._agent.rollback_skill,
            revision_id,
            scope=scope,
            approved_by=approved_by,
        )

    async def list_skill_revisions(
        self,
        name: str,
        *,
        scope: str = "project",
        limit: int = 100,
    ) -> tuple[GovernedSkillRevision, ...]:
        return await asyncio.to_thread(
            self._agent.list_skill_revisions,
            name,
            scope=scope,
            limit=limit,
        )

    async def confirm_skill_success(
        self,
        *,
        turn_id: str | None = None,
    ) -> tuple[GovernedSkill, ...]:
        return await asyncio.to_thread(
            self._agent.confirm_skill_success,
            turn_id=turn_id,
        )

    async def inspect_plugin(
        self,
        path: Path | str,
    ) -> PluginInspection:
        return await asyncio.to_thread(self._agent.inspect_plugin, path)

    async def register_local_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLockEntry:
        return await asyncio.to_thread(
            self._agent.register_local_plugin,
            path,
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=granted_capabilities,
        )

    async def install_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLifecycleReceipt:
        return await asyncio.to_thread(
            self._agent.install_plugin,
            path,
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=granted_capabilities,
        )

    async def plan_plugin_update(
        self,
        path: Path | str,
    ) -> PluginUpdatePlan:
        return await asyncio.to_thread(
            self._agent.plan_plugin_update,
            path,
        )

    async def update_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] | None = None,
        approve_authority_changes: bool = False,
    ) -> PluginLifecycleReceipt:
        return await asyncio.to_thread(
            self._agent.update_plugin,
            path,
            approved_by=approved_by,
            acknowledge_host_authority=acknowledge_host_authority,
            granted_capabilities=granted_capabilities,
            approve_authority_changes=approve_authority_changes,
        )

    async def uninstall_plugin(
        self,
        plugin_name: str,
        *,
        approved_by: str,
    ) -> PluginLifecycleReceipt:
        return await asyncio.to_thread(
            self._agent.uninstall_plugin,
            plugin_name,
            approved_by=approved_by,
        )

    async def rollback_plugin(
        self,
        plugin_name: str,
        *,
        approved_by: str,
    ) -> PluginLifecycleReceipt:
        return await asyncio.to_thread(
            self._agent.rollback_plugin,
            plugin_name,
            approved_by=approved_by,
        )

    async def revoke_plugin(
        self,
        plugin_name: str,
        *,
        reason: str,
        revoked_by: str,
    ) -> PluginLifecycleReceipt:
        return await asyncio.to_thread(
            self._agent.revoke_plugin,
            plugin_name,
            reason=reason,
            revoked_by=revoked_by,
        )

    async def list_plugins(self) -> tuple[PluginLockEntry, ...]:
        return await asyncio.to_thread(self._agent.list_plugins)

    async def audit_plugins(self) -> PluginAuditReport:
        return await asyncio.to_thread(self._agent.audit_plugins)

    async def load_plugin_entry_point(
        self,
        plugin_name: str,
        category: PluginCategory | str,
        entry_name: str,
        *,
        available_capabilities: tuple[str, ...] = (),
    ) -> LoadedPluginEntryPoint:
        return await asyncio.to_thread(
            self._agent.load_plugin_entry_point,
            plugin_name,
            category,
            entry_name,
            available_capabilities=available_capabilities,
        )

    @property
    def usage_ledger(self) -> UsageLedger:
        return self._agent.usage_ledger

    @property
    def session_search(self) -> SessionSearchService:
        return self._agent.session_search

    async def query_usage(self, **kwargs: Any) -> UsagePage:
        return await asyncio.to_thread(self._agent.query_usage, **kwargs)

    async def group_usage(
        self,
        group_by: UsageGroupBy,
        **kwargs: Any,
    ) -> tuple[UsageAggregate, ...]:
        return await asyncio.to_thread(
            self._agent.group_usage,
            group_by,
            **kwargs,
        )

    async def search_sessions(
        self,
        query: str,
        *,
        limit: int = 10,
        cursor: str | None = None,
    ) -> SessionSearchPage:
        return await asyncio.to_thread(
            self._agent.search_sessions,
            query,
            limit=limit,
            cursor=cursor,
        )

    async def read_session_window(
        self,
        conversation_id: str,
        *,
        ordinal: int,
        before: int = 3,
        after: int = 3,
        limit: int = 20,
        cursor: str | None = None,
        include_sensitive: bool = False,
    ) -> SessionWindow:
        return await asyncio.to_thread(
            self._agent.read_session_window,
            conversation_id,
            ordinal=ordinal,
            before=before,
            after=after,
            limit=limit,
            cursor=cursor,
            include_sensitive=include_sensitive,
        )

    async def read_artifact(
        self,
        artifact_id: str,
        *,
        mode: ArtifactReadMode = "head_tail",
        offset: int = 0,
        max_bytes: int = DEFAULT_ARTIFACT_READ_BYTES,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._agent.read_artifact,
            artifact_id,
            mode=mode,
            offset=offset,
            max_bytes=max_bytes,
        )

    async def approve_memory_proposal(self, proposal_id: str) -> MemoryProposal:
        return await asyncio.to_thread(self._agent.approve_memory_proposal, proposal_id)

    async def reject_memory_proposal(self, proposal_id: str) -> MemoryProposal:
        return await asyncio.to_thread(self._agent.reject_memory_proposal, proposal_id)

    async def run_events_async(self, message: str, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        """Asynchronously yield one run's ordered public events and terminal result."""
        channel = RunEventChannel()
        caller_on_event = kwargs.pop("on_event", None)
        attempted_turn_id: str | None = None
        last_event_id: str | None = None

        def on_event(event: AgentEvent) -> None:
            nonlocal attempted_turn_id, last_event_id
            if event.name == EventName.RUN_STARTED.value:
                attempted_turn_id = event.turn_id
            if event.name not in {EventName.RUN_COMPLETED.value, EventName.RUN_FAILED.value}:
                channel.publish(event)
                last_event_id = event.event_id
            if caller_on_event is not None:
                caller_on_event(event)

        async def work() -> None:
            try:
                result = await self.run_result(message, on_event=on_event, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                terminalized = _terminalized_failure_event(
                    self.runtime,
                    attempted_turn_id,
                    causation_id=last_event_id,
                )
                event = terminalized or failure_event(
                    exc,
                    conversation_id=self.conversation_id,
                    turn_id=attempted_turn_id,
                    profile_id=self.runtime.profile_id,
                    execution_scope=self.runtime.execution_scope,
                    causation_id=last_event_id,
                )
                channel.finish(event)
                if terminalized is None:
                    _notify_event_callback_safely(caller_on_event, event)
            else:
                channel.finish(
                    terminal_event(
                        result,
                        profile_id=self.runtime.profile_id,
                        execution_scope=self.runtime.execution_scope,
                        causation_id=last_event_id,
                    )
                )

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
        finally:
            event_buffer = getattr(self.runtime, "async_event_buffer", None)
            if event_buffer is not None:
                await event_buffer.flush()


def _notify_event_callback_safely(callback: EventCallback | None, event: AgentEvent) -> None:
    """Best-effort delivery after a callback itself caused run failure."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        return


def _terminalized_failure_event(
    runtime: CoreAgent,
    attempted_turn_id: str | None,
    *,
    causation_id: str | None = None,
) -> AgentEvent | None:
    if attempted_turn_id is None:
        return None
    result = run_result_from_runtime(runtime)
    if (
        result.turn_id == attempted_turn_id
        and result.status in {RunStatus.FAILED, RunStatus.BLOCKED, RunStatus.CANCELLED}
    ):
        return terminal_event(
            result,
            profile_id=runtime.profile_id,
            execution_scope=runtime.execution_scope,
            causation_id=causation_id,
        )
    return None


class HostedRuntime(Agent):
    """Synchronous SDK facade that requires a complete hosted boundary."""

    def __init__(
        self,
        *,
        services: RuntimeServices,
        execution_scope: ExecutionScope,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            services=services,
            execution_scope=execution_scope,
            **kwargs,
        )


class AsyncHostedRuntime(AsyncAgent):
    """Asynchronous SDK facade that requires async-hosted service contracts."""

    def __init__(
        self,
        *,
        services: AsyncRuntimeServices,
        execution_scope: ExecutionScope,
        **kwargs: Any,
    ) -> None:
        if any(
            isinstance(getattr(services, name), AsyncServiceBinding)
            for name in services.__dataclass_fields__
        ):
            raise ValueError(
                "native async service bindings require "
                "await AsyncHostedRuntime.create(...)"
            )
        sync_boundary = services.as_sync_services()
        super().__init__(
            services=sync_boundary,
            execution_scope=execution_scope,
            **kwargs,
        )
        from chulk.hosting.sinks import BufferedAsyncEventSink

        sink = self.runtime.public_event_sink
        event_buffer = BufferedAsyncEventSink(sink)
        self.runtime.public_event_sink = event_buffer
        self.runtime.async_event_buffer = event_buffer
        self._async_owned_services: ResolvedRuntimeServices | None = None
        self._async_host_flushables: tuple[object, ...] = ()

    @classmethod
    async def create(
        cls,
        *,
        services: AsyncRuntimeServices,
        execution_scope: ExecutionScope,
        **kwargs: Any,
    ) -> "AsyncHostedRuntime":
        """Build a native async hosted runtime without a sync service bridge."""

        config_arg = kwargs.get("config")
        preset = kwargs.get("preset")
        capabilities = _selected_capabilities(
            config_arg,
            kwargs.get("capabilities"),
            kwargs.get("memory_mode"),
        )
        conflicts = [
            name
            for name in (
                "execution_backend",
                "plugin_registry",
                "content_store",
                "media_processors",
                "memory_namespace",
            )
            if kwargs.get(name) is not None
        ]
        if conflicts:
            raise ValueError(
                "hosted services cannot be combined with individual runtime "
                "injections: " + ", ".join(conflicts)
            )
        selected_tools = kwargs.get("tools")
        if selected_tools is None and preset is not None:
            selected_tools = preset.tools
        selected_skills = kwargs.get("skills")
        if selected_skills is None and preset is not None:
            selected_skills = preset.skills
        selected_prompt = kwargs.get("system_prompt")
        if selected_prompt is None and preset is not None:
            selected_prompt = preset.system_prompt
        try:
            core, resolved = await create_async_hosted_agent(
                coerce_config(config_arg),
                services=services,
                execution_scope=execution_scope,
                conversation_id=kwargs.get("conversation_id"),
                conversation_metadata=kwargs.get("conversation_metadata"),
                runtime_metadata=kwargs.get("runtime_metadata"),
                llm_client=kwargs.get("llm"),
                tool_specs=selected_tools,
                skill_specs=selected_skills,
                system_prompt=selected_prompt,
                permission_callback=kwargs.get("permission_callback"),
                mcp_servers=(
                    tuple(kwargs["mcp"])
                    if kwargs.get("mcp") is not None
                    else None
                ),
                redaction_callback=kwargs.get("redaction_callback"),
                redaction_fail_closed=bool(
                    kwargs.get("redaction_fail_closed", False)
                ),
                capabilities=capabilities,
                deps=kwargs.get("deps"),
                shell_execution_policy=kwargs.get(
                    "shell_execution_policy"
                ),
                require_shell_containment=bool(
                    kwargs.get("require_shell_containment", False)
                ),
                run_budget=kwargs.get("run_budget"),
                usage_dimensions=kwargs.get("usage_dimensions"),
                goal_execution=kwargs.get("goal_execution"),
            )
        except Exception as exc:
            mapped = map_public_error(
                exc,
                config=config_arg,
                operation="construct",
            )
            if mapped is exc:
                raise
            raise mapped from exc

        handle = AgentHandle(core, on_event=kwargs.get("on_event"))
        sync_facade = Agent.__new__(Agent)
        sync_facade._handle = handle
        sync_facade._run_gate = RunGate()
        sync_facade._capabilities = capabilities
        sync_facade._deps = kwargs.get("deps")

        runtime = cls.__new__(cls)
        runtime._agent = sync_facade
        runtime._handle = AsyncAgentHandle(handle)
        runtime._async_run_gate = asyncio.Lock()
        runtime._async_owned_services = resolved
        runtime._async_host_flushables = ()
        return runtime

    async def _invoke_async(
        self,
        operation: str,
        call: Callable[[], Awaitable[T]],
        *,
        serialized: bool = False,
    ) -> T:
        del serialized
        async with self._async_run_gate:
            try:
                return await super()._invoke_async(
                    operation,
                    call,
                    serialized=False,
                )
            finally:
                for flushable in self._async_host_flushables:
                    flush = getattr(flushable, "flush")
                    await flush()

    def _resolved_async_services(self) -> ResolvedRuntimeServices:
        resolved = self._async_owned_services
        if resolved is None:
            raise RuntimeError("Agent is closed")
        return resolved

    @property
    def usage_ledger(self) -> Any:
        """Return the native async usage service bound to this runtime."""
        return self._resolved_async_services().usage

    @property
    def session_search(self) -> Any:
        """Return the native async session-search service for this runtime."""
        return self._resolved_async_services().sessions.search

    async def _call_hosted_service(
        self,
        operation: str,
        service_name: str,
        method_name: str,
        /,
        *args: Any,
        serialized: bool = False,
        **kwargs: Any,
    ) -> Any:
        async def call() -> Any:
            service = getattr(
                self._resolved_async_services(),
                service_name,
            )
            return await call_async_service(
                service,
                method_name,
                *args,
                **kwargs,
            )

        return await self._invoke_async(
            operation,
            call,
            serialized=serialized,
        )

    async def list_memory_proposals(self) -> tuple[MemoryProposal, ...]:
        async def operation() -> tuple[MemoryProposal, ...]:
            policy = self.runtime.async_memory_policy
            if policy is None:
                return ()
            return tuple(
                memory_proposal_snapshot(item)
                for item in await policy.list_pending()
            )

        return await self._invoke_async("list_memory_proposals", operation)

    async def approve_memory_proposal(
        self,
        proposal_id: str,
    ) -> MemoryProposal:
        async def operation() -> MemoryProposal:
            policy = self.runtime.async_memory_policy
            if policy is None:
                raise RuntimeError("Memory is not configured")
            return memory_proposal_snapshot(await policy.approve(proposal_id))

        return await self._invoke_async("approve_memory_proposal", operation)

    async def reject_memory_proposal(
        self,
        proposal_id: str,
    ) -> MemoryProposal:
        async def operation() -> MemoryProposal:
            policy = self.runtime.async_memory_policy
            if policy is None:
                raise RuntimeError("Memory is not configured")
            return memory_proposal_snapshot(await policy.reject(proposal_id))

        return await self._invoke_async("reject_memory_proposal", operation)

    async def list_learning_proposals(
        self,
        *,
        status: str | None = "pending",
        limit: int = 100,
    ) -> tuple[LearningProposal, ...]:
        async def operation() -> tuple[LearningProposal, ...]:
            service = self._resolved_async_services().skills.learning_proposals
            if service is None:
                return ()
            normalized = (
                None if status is None else LearningProposalStatus(status)
            )
            records = await call_async_service(
                service,
                "list",
                status=normalized,
                limit=limit,
            )
            return tuple(
                learning_proposal_snapshot(item) for item in records
            )

        return await self._invoke_async(
            "list_learning_proposals",
            operation,
        )

    async def get_learning_proposal(
        self,
        proposal_id: str,
    ) -> LearningProposal:
        async def operation() -> LearningProposal:
            service = self._resolved_async_services().skills.learning_proposals
            if service is None:
                raise RuntimeError("learning proposals are not configured")
            record = await call_async_service(service, "get", proposal_id)
            return learning_proposal_snapshot(record)

        return await self._invoke_async("get_learning_proposal", operation)

    async def approve_learning_proposal(
        self,
        proposal_id: str,
        *,
        approved_by: str = "sdk-host",
    ) -> LearningProposal:
        async def operation() -> LearningProposal:
            service = self._resolved_async_services().skills.learning_proposals
            if service is None:
                raise RuntimeError("learning proposals are not configured")
            record = await call_async_service(
                service,
                "approve",
                proposal_id,
                approved_by=approved_by,
            )
            return learning_proposal_snapshot(record)

        return await self._invoke_async(
            "approve_learning_proposal",
            operation,
        )

    async def review_learning(
        self,
        *,
        trigger: str = "manual",
        turn_id: str | None = None,
        host_confirmed_success: bool = False,
    ) -> LearningReview:
        async def operation() -> LearningReview:
            outcome = await self.runtime.review_learning_async(
                trigger=trigger,
                turn_id=turn_id,
                host_confirmed_success=host_confirmed_success,
            )
            service = self._resolved_async_services().skills.learning_proposals
            if service is None:
                raise RuntimeError("learning proposals are not configured")
            proposals = []
            for proposal_id in outcome.proposal_ids:
                proposals.append(
                    learning_proposal_snapshot(
                        await call_async_service(
                            service,
                            "get",
                            proposal_id,
                        )
                    )
                )
            return LearningReview(
                skipped=outcome.skipped,
                rationale=outcome.rationale,
                proposals=tuple(proposals),
                review_run_id=outcome.review_run_id,
            )

        return await self._invoke_async(
            "review_learning",
            operation,
            serialized=True,
        )

    async def reject_learning_proposal(
        self,
        proposal_id: str,
        *,
        rejected_by: str = "sdk-host",
    ) -> LearningProposal:
        async def operation() -> LearningProposal:
            service = self._resolved_async_services().skills.learning_proposals
            if service is None:
                raise RuntimeError("learning proposals are not configured")
            record = await call_async_service(
                service,
                "reject",
                proposal_id,
                rejected_by=rejected_by,
            )
            return learning_proposal_snapshot(record)

        return await self._invoke_async(
            "reject_learning_proposal",
            operation,
        )

    async def list_governed_skills(
        self,
        *,
        scope: str | None = None,
    ) -> tuple[GovernedSkill, ...]:
        async def operation() -> tuple[GovernedSkill, ...]:
            store = self._resolved_async_services().skills.lifecycle_store
            if store is None:
                return ()
            records = await call_async_service(
                store,
                "list_skills",
                scope=scope,
            )
            event_id = f"sdk-view:{uuid4()}"
            viewed = []
            for record in records:
                viewed.append(
                    await call_async_service(
                        store,
                        "record_usage",
                        name=record.name,
                        scope=record.scope,
                        version=record.version,
                        digest=record.digest,
                        kind=SkillUsageKind.VIEW,
                        source_event_id=event_id,
                    )
                )
            return tuple(
                governed_skill_snapshot(item) for item in viewed
            )

        return await self._invoke_async("list_governed_skills", operation)

    async def rollback_skill(
        self,
        revision_id: str,
        *,
        scope: str = "project",
        approved_by: str = "sdk-host",
    ) -> GovernedSkill:
        async def operation() -> GovernedSkill:
            lifecycle = self._resolved_async_services().skills.lifecycle
            if lifecycle is None:
                raise RuntimeError("skill lifecycle is not configured")
            if scope not in {"project", "profile"}:
                raise ValueError("scope must be project or profile")
            record = await call_async_service(
                lifecycle,
                "rollback",
                revision_id,
                scope=cast(SkillScope, scope),
                approved_by=approved_by,
            )
            return governed_skill_snapshot(record)

        return await self._invoke_async("rollback_skill", operation)

    async def list_skill_revisions(
        self,
        name: str,
        *,
        scope: str = "project",
        limit: int = 100,
    ) -> tuple[GovernedSkillRevision, ...]:
        async def operation() -> tuple[GovernedSkillRevision, ...]:
            store = self._resolved_async_services().skills.lifecycle_store
            if store is None:
                return ()
            records = await call_async_service(
                store,
                "list_revisions",
                name,
                scope=scope,
                limit=limit,
            )
            return tuple(
                governed_skill_revision_snapshot(item)
                for item in records
            )

        return await self._invoke_async("list_skill_revisions", operation)

    async def confirm_skill_success(
        self,
        *,
        turn_id: str | None = None,
    ) -> tuple[GovernedSkill, ...]:
        async def operation() -> tuple[GovernedSkill, ...]:
            records = await self.runtime.confirm_skill_success_async(
                turn_id=turn_id
            )
            return tuple(
                governed_skill_snapshot(item) for item in records
            )

        return await self._invoke_async("confirm_skill_success", operation)

    async def inspect_plugin(
        self,
        path: Path | str,
    ) -> PluginInspection:
        return cast(
            PluginInspection,
            await self._call_hosted_service(
                "inspect_plugin",
                "plugins",
                "inspect",
                path,
            ),
        )

    async def register_local_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLockEntry:
        return cast(
            PluginLockEntry,
            await self._call_hosted_service(
            "register_local_plugin",
                "plugins",
                "register_local",
                path,
                approved_by=approved_by,
                acknowledge_host_authority=acknowledge_host_authority,
                granted_capabilities=granted_capabilities,
                serialized=True,
            ),
        )

    async def install_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] = (),
    ) -> PluginLifecycleReceipt:
        return cast(
            PluginLifecycleReceipt,
            await self._call_hosted_service(
                "install_plugin",
                "plugins",
                "install",
                path,
                approved_by=approved_by,
                acknowledge_host_authority=acknowledge_host_authority,
                granted_capabilities=granted_capabilities,
                serialized=True,
            ),
        )

    async def plan_plugin_update(
        self,
        path: Path | str,
    ) -> PluginUpdatePlan:
        return cast(
            PluginUpdatePlan,
            await self._call_hosted_service(
                "plan_plugin_update",
                "plugins",
                "plan_update",
                path,
            ),
        )

    async def update_plugin(
        self,
        path: Path | str,
        *,
        approved_by: str,
        acknowledge_host_authority: bool,
        granted_capabilities: tuple[str, ...] | None = None,
        approve_authority_changes: bool = False,
    ) -> PluginLifecycleReceipt:
        return cast(
            PluginLifecycleReceipt,
            await self._call_hosted_service(
                "update_plugin",
                "plugins",
                "update",
                path,
                approved_by=approved_by,
                acknowledge_host_authority=acknowledge_host_authority,
                granted_capabilities=granted_capabilities,
                approve_authority_changes=approve_authority_changes,
                serialized=True,
            ),
        )

    async def uninstall_plugin(
        self,
        plugin_name: str,
        *,
        approved_by: str,
    ) -> PluginLifecycleReceipt:
        return cast(
            PluginLifecycleReceipt,
            await self._call_hosted_service(
                "uninstall_plugin",
                "plugins",
                "uninstall",
                plugin_name,
                approved_by=approved_by,
                serialized=True,
            ),
        )

    async def rollback_plugin(
        self,
        plugin_name: str,
        *,
        approved_by: str,
    ) -> PluginLifecycleReceipt:
        return cast(
            PluginLifecycleReceipt,
            await self._call_hosted_service(
                "rollback_plugin",
                "plugins",
                "rollback",
                plugin_name,
                approved_by=approved_by,
                serialized=True,
            ),
        )

    async def revoke_plugin(
        self,
        plugin_name: str,
        *,
        reason: str,
        revoked_by: str,
    ) -> PluginLifecycleReceipt:
        return cast(
            PluginLifecycleReceipt,
            await self._call_hosted_service(
                "revoke_plugin",
                "plugins",
                "revoke",
                plugin_name,
                reason=reason,
                revoked_by=revoked_by,
                serialized=True,
            ),
        )

    async def list_plugins(self) -> tuple[PluginLockEntry, ...]:
        records = await self._call_hosted_service(
            "list_plugins",
            "plugins",
            "list",
        )
        return tuple(records)

    async def audit_plugins(self) -> PluginAuditReport:
        return cast(
            PluginAuditReport,
            await self._call_hosted_service(
                "audit_plugins",
                "plugins",
                "audit",
            ),
        )

    async def load_plugin_entry_point(
        self,
        plugin_name: str,
        category: PluginCategory | str,
        entry_name: str,
        *,
        available_capabilities: tuple[str, ...] = (),
    ) -> LoadedPluginEntryPoint:
        return cast(
            LoadedPluginEntryPoint,
            await self._call_hosted_service(
                "load_plugin_entry_point",
                "plugins",
                "load_entry_point",
                plugin_name,
                category,
                entry_name,
                available_capabilities=available_capabilities,
                serialized=True,
            ),
        )

    async def query_usage(self, **kwargs: Any) -> UsagePage:
        return cast(
            UsagePage,
            await self._call_hosted_service(
                "query_usage",
                "usage",
                "query",
                **kwargs,
            ),
        )

    async def group_usage(
        self,
        group_by: UsageGroupBy,
        **kwargs: Any,
    ) -> tuple[UsageAggregate, ...]:
        records = await self._call_hosted_service(
            "group_usage",
            "usage",
            "group",
            group_by,
            **kwargs,
        )
        return tuple(records)

    async def search_sessions(
        self,
        query: str,
        *,
        limit: int = 10,
        cursor: str | None = None,
    ) -> SessionSearchPage:
        async def operation() -> SessionSearchPage:
            search = self._resolved_async_services().sessions.search
            return cast(
                SessionSearchPage,
                await call_async_service(
                    search,
                    "search",
                    query,
                    limit=limit,
                    cursor=cursor,
                ),
            )

        return await self._invoke_async("search_sessions", operation)

    async def read_session_window(
        self,
        conversation_id: str,
        *,
        ordinal: int,
        before: int = 3,
        after: int = 3,
        limit: int = 20,
        cursor: str | None = None,
        include_sensitive: bool = False,
    ) -> SessionWindow:
        async def operation() -> SessionWindow:
            search = self._resolved_async_services().sessions.search
            return cast(
                SessionWindow,
                await call_async_service(
                    search,
                    "read_window",
                    conversation_id,
                    ordinal=ordinal,
                    before=before,
                    after=after,
                    limit=limit,
                    cursor=cursor,
                    include_sensitive=include_sensitive,
                ),
            )

        return await self._invoke_async("read_session_window", operation)

    async def read_artifact(
        self,
        artifact_id: str,
        *,
        mode: ArtifactReadMode = "head_tail",
        offset: int = 0,
        max_bytes: int = DEFAULT_ARTIFACT_READ_BYTES,
    ) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            record = await call_async_service(
                self._resolved_async_services().artifacts,
                "read",
                artifact_id,
                mode=mode,
                offset=offset,
                max_bytes=max_bytes,
            )
            if isinstance(record, dict):
                return dict(record)
            to_dict = getattr(record, "to_dict", None)
            if callable(to_dict):
                return cast(dict[str, Any], to_dict())
            raise TypeError("async artifact store returned an unsupported read")

        return await self._invoke_async("read_artifact", operation)

    async def close(self) -> None:
        async def operation() -> None:
            owned = self._async_owned_services
            self._async_owned_services = None
            failure: BaseException | None = None
            operations: list[Callable[[], Awaitable[object]]] = [
                self.runtime._flush_async_services,
                self._handle.close,
            ]
            if owned is not None:
                operations.append(owned.aclose_owned)
            for close_operation in operations:
                try:
                    await close_operation()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note(
                            "async close also failed with "
                            f"{type(exc).__name__}: {exc}"
                        )
            if failure is not None:
                raise failure

        await self._invoke_async("close", operation, serialized=True)


def _build_handle(
    *,
    config: Config | AgentConfig | None = None,
    preset: AgentPreset | None = None,
    llm: LLMClient | Any | None = None,
    tools: Iterable[object] | None = None,
    skills: object | Iterable[object] | None = None,
    system_prompt: str | None = None,
    conversation_id: str | None = None,
    conversation_metadata: dict[str, object] | None = None,
    runtime_metadata: dict[str, object] | None = None,
    permission_callback: PermissionCallback | None = None,
    on_event: EventCallback | None = None,
    mcp: Iterable[MCPServerConfig] | None = None,
    redaction_callback: Callable[[str, str, dict], str] | None = None,
    redaction_fail_closed: bool = False,
    capabilities: Capabilities | None = None,
    memory_namespace: str | None = None,
    deps: object | None = None,
    shell_execution_policy: ShellExecutionPolicy | None = None,
    require_shell_containment: bool = False,
    execution_backend: ExecutionBackend | None = None,
    run_budget: RunBudget | None = None,
    usage_dimensions: UsageDimensions | None = None,
    learning_review_policy: LearningReviewPolicy | None = None,
    learning_review_quota: LearningReviewQuota | None = None,
    automatic_learning_approval: bool = False,
    plugin_registry: LocalPluginRegistry | None = None,
    goal_execution: GoalExecutionContext | None = None,
    content_store: ContentStore | None = None,
    media_processors: MediaProcessorRegistry | None = None,
    services: RuntimeServices | None = None,
    execution_scope: ExecutionScope | None = None,
) -> AgentHandle:
    runtime_config = coerce_config(config)
    selected_tools = tools if tools is not None else (preset.tools if preset is not None else None)
    selected_skills = skills if skills is not None else (preset.skills if preset is not None else None)
    selected_prompt = system_prompt or (preset.system_prompt if preset is not None else None)
    runtime = create_runtime_agent(
        runtime_config,
        conversation_id=conversation_id,
        conversation_metadata=conversation_metadata,
        runtime_metadata=runtime_metadata,
        llm_client=llm,
        tool_specs=selected_tools,
        skill_specs=selected_skills,
        system_prompt=selected_prompt,
        permission_callback=permission_callback,
        mcp_servers=tuple(mcp) if mcp is not None else None,
        redaction_callback=redaction_callback,
        redaction_fail_closed=redaction_fail_closed,
        capabilities=capabilities,
        memory_namespace=_selected_memory_namespace(config, memory_namespace),
        deps=deps,
        shell_execution_policy=shell_execution_policy,
        require_shell_containment=require_shell_containment,
        execution_backend=execution_backend,
        run_budget=run_budget,
        usage_dimensions=usage_dimensions,
        learning_review_policy=learning_review_policy,
        learning_review_quota=learning_review_quota,
        automatic_learning_approval=automatic_learning_approval,
        plugin_registry=plugin_registry,
        goal_execution=goal_execution,
        content_store=content_store,
        media_processors=media_processors,
        services=services,
        execution_scope=execution_scope,
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


def _selected_memory_namespace(
    config: Config | AgentConfig | None,
    memory_namespace: str | None,
) -> str | None:
    if memory_namespace is not None:
        return memory_namespace
    if isinstance(config, AgentConfig):
        return config.memory_namespace
    return None


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
