"""Public asynchronous SDK agent facade."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from chulk.capabilities import Capabilities
from chulk._sdk.agent import (
    Agent,
    _notify_event_callback_safely,
    _terminalized_failure_event,
)
from chulk._sdk.error_mapping import map_public_error
from chulk._sdk.event_channel import RunEventChannel
from chulk._sdk.events import failure_event, terminal_event
from chulk._sdk.handles import AsyncAgentHandle
from chulk._sdk.results import (
    PlanResult,
    RunResult,
)
from chulk.events import AgentEvent, EventName
from chulk.hosting import (
    ExecutionScope,
)
from chulk.media import UserInput
from chulk.plugins import (
    LoadedPluginEntryPoint,
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
)
from chulk.sessions import (
    SessionSearchPage,
    SessionSearchService,
    SessionWindow,
)
from chulk.tracing.artifacts import (
    ArtifactReadMode,
    DEFAULT_ARTIFACT_READ_BYTES,
)
from chulk.usage import (
    UsageAggregate,
    UsageGroupBy,
    UsageLedger,
    UsagePage,
)

T = TypeVar("T")


class AsyncAgent(AsyncAgentHandle):
    """Public asynchronous facade backed by Chulk's compatibility runtime."""

    @classmethod
    def from_directory(cls, path: Path | str, **kwargs: Any) -> "AsyncAgent":
        """Create a local async SDK agent from an agent directory."""
        from chulk.authoring.directory import AgentDirectory

        source = AgentDirectory.load(path)
        tools = tuple(kwargs.get("tools") or ())
        available = {str(getattr(tool, "name", "")): tool for tool in tools}
        missing = set(source.tools) - set(available)
        if missing:
            raise ValueError(
                "AsyncAgent.from_directory requires the declared tools: "
                + ", ".join(sorted(missing))
            )
        kwargs["tools"] = tuple(available[name] for name in source.tools)
        if "system_prompt" in kwargs:
            raise ValueError("AsyncAgent.from_directory owns system_prompt through instructions.md")
        return cls(system_prompt=source.instructions, **kwargs)

    def __init__(self, **kwargs: Any) -> None:
        self._initialize_agent(Agent(**kwargs))

    def _initialize_agent(self, agent: Agent) -> None:
        self._agent = agent
        super().__init__(agent)
        self._async_run_gate = asyncio.Lock()

    @property
    def execution_scope(self) -> ExecutionScope:
        return cast(ExecutionScope, self.runtime.execution_scope)

    @property
    def capabilities(self) -> Capabilities:
        return self._agent.capabilities

    async def run(self, message: str, **kwargs: Any) -> str:
        options = self._agent._run_options(kwargs)
        result = await self._invoke_async(
            "run",
            lambda: AsyncAgentHandle.run_result(self, message, **options),
            serialized=True,
        )
        return result.content

    async def run_result(self, message: str, **kwargs: Any) -> RunResult:
        options = self._agent._run_options(kwargs)
        return await self._invoke_async(
            "run_result",
            lambda: AsyncAgentHandle.run_result(self, message, **options),
            serialized=True,
        )

    async def run_input(self, user_input: UserInput, **kwargs: Any) -> str:
        options = self._agent._run_options(kwargs)
        result = await self._invoke_async(
            "run_input",
            lambda: AsyncAgentHandle.run_input_result(self, user_input, **options),
            serialized=True,
        )
        return result.content

    async def run_input_result(
        self,
        user_input: UserInput,
        **kwargs: Any,
    ) -> RunResult:
        options = self._agent._run_options(kwargs)
        return await self._invoke_async(
            "run_input_result",
            lambda: AsyncAgentHandle.run_input_result(self, user_input, **options),
            serialized=True,
        )

    async def plan(self, message: str) -> str:
        result = await self._invoke_async(
            "plan",
            lambda: AsyncAgentHandle.plan_result(self, message),
            serialized=True,
        )
        return result.content

    async def plan_result(self, message: str, **kwargs: Any) -> PlanResult:
        return await self._invoke_async(
            "plan_result",
            lambda: AsyncAgentHandle.plan_result(self, message, **kwargs),
            serialized=True,
        )

    async def approve(self) -> str:
        result = await self._invoke_async(
            "approve",
            lambda: AsyncAgentHandle.approve_result(self),
            serialized=True,
        )
        return result.content

    async def approve_result(self, **kwargs: Any) -> RunResult:
        return await self._invoke_async(
            "approve_result",
            lambda: AsyncAgentHandle.approve_result(self, **kwargs),
            serialized=True,
        )

    async def reject(self) -> str:
        result = await self._invoke_async(
            "reject",
            lambda: AsyncAgentHandle.reject_result(self),
            serialized=True,
        )
        return result.content

    async def reject_result(self, **kwargs: Any) -> RunResult:
        return await self._invoke_async(
            "reject_result",
            lambda: AsyncAgentHandle.reject_result(self, **kwargs),
            serialized=True,
        )

    async def close(self) -> None:
        await self._invoke_async("close", lambda: AsyncAgentHandle.close(self), serialized=True)

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
            if (
                not worker.done()
                and getattr(self.runtime, "final_answer_streaming", None)
                == "incremental"
            ):
                worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    async def __aenter__(self) -> "AsyncAgent":
        self._agent._invoke("enter", self._agent._ensure_open)
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
