"""Hosted synchronous and asynchronous SDK runtimes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from pathlib import Path
from typing import Any, Callable, TypeVar, cast
from uuid import uuid4

from chulk._sdk.agent import Agent
from chulk._sdk.async_agent import AsyncAgent
from chulk._sdk.config import AgentConfig, AgentPreset, coerce_config
from chulk._sdk.construction import PermissionCallback, _selected_capabilities
from chulk._sdk.error_mapping import map_public_error
from chulk._sdk.event_channel import RunGate
from chulk._sdk.events import EventCallback
from chulk._sdk.handles import AgentHandle, AsyncAgentHandle
from chulk._sdk.results import (
    governed_skill_snapshot,
    governed_skill_revision_snapshot,
    learning_proposal_snapshot,
    memory_proposal_snapshot,
)
from chulk.hosting import (
    AsyncRuntimeServices,
    AsyncServiceBinding,
    AsyncTranscriptResolver,
    ExecutionScope,
    HostedServiceManifest,
    RuntimeServices,
)
from chulk.hosting.async_utils import call_async_service
from chulk.hosting.services import ResolvedRuntimeServices
from chulk.hosting.tool_catalog import AsyncToolCatalogResolver
from chulk.config import Config
from chulk.capabilities import Capabilities, MemoryMode
from chulk.core.plan_execution import AsyncPlanStepVerifier, PlanStepVerifier
from chulk.goals import GoalExecutionContext
from chulk.llm import LLMClient
from chulk.mcp import MCPServerConfig
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
from chulk.streaming import (
    AsyncIncrementalOutputPolicy,
    FinalAnswerStreamingMode,
    IncrementalOutputPolicy,
    OutputPolicyFailureMode,
)
from chulk.runtime import create_async_hosted_agent
from chulk._runtime.request import AgentAssemblyRequest
from chulk.sessions import SessionSearchPage, SessionWindow
from chulk.skills import (
    LearningProposalStatus,
    SkillScope,
    SkillUsageKind,
)
from chulk.tracing.artifacts import (
    ArtifactReadMode,
    DEFAULT_ARTIFACT_READ_BYTES,
)
from chulk.tools import ShellExecutionPolicy
from chulk.usage import (
    RunBudget,
    UsageAggregate,
    UsageDimensions,
    UsageGroupBy,
    UsagePage,
)


T = TypeVar("T")


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

    @property
    def service_manifest(self) -> HostedServiceManifest:
        """Return the resolved hosted capability and service manifest."""
        return cast(HostedServiceManifest, self.runtime.resolved_services.manifest)


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

        sink = self.runtime.events.public_event_sink
        event_buffer = BufferedAsyncEventSink(sink)
        self.runtime.events.set_public_sink(event_buffer)
        self.runtime.async_event_buffer = event_buffer
        self._async_owned_services: ResolvedRuntimeServices | None = None
        self._async_host_flushables: tuple[object, ...] = ()

    @classmethod
    async def create(
        cls,
        *,
        services: AsyncRuntimeServices,
        execution_scope: ExecutionScope,
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
        plan_step_verifier: PlanStepVerifier | None = None,
        async_plan_step_verifier: AsyncPlanStepVerifier | None = None,
        on_event: EventCallback | None = None,
        mcp: Iterable[MCPServerConfig] | None = None,
        redaction_callback: Callable[[str, str, dict], str] | None = None,
        redaction_fail_closed: bool = False,
        final_answer_streaming: FinalAnswerStreamingMode
        | str = FinalAnswerStreamingMode.VALIDATED,
        output_policy: IncrementalOutputPolicy | None = None,
        async_output_policy: AsyncIncrementalOutputPolicy | None = None,
        output_policy_failure_mode: OutputPolicyFailureMode
        | str = OutputPolicyFailureMode.CLOSED,
        capabilities: Capabilities | None = None,
        memory_mode: MemoryMode | str | None = None,
        deps: object | None = None,
        shell_execution_policy: ShellExecutionPolicy | None = None,
        require_shell_containment: bool = False,
        run_budget: RunBudget | None = None,
        usage_dimensions: UsageDimensions | None = None,
        goal_execution: GoalExecutionContext | None = None,
        async_transcript_resolver: AsyncTranscriptResolver | None = None,
        transcript_timeout_seconds: float | None = None,
        async_tool_catalog_resolver: AsyncToolCatalogResolver | None = None,
        tool_catalog_timeout_seconds: float | None = None,
    ) -> "AsyncHostedRuntime":
        """Build a native async hosted runtime without a sync service bridge."""

        capabilities = _selected_capabilities(
            config,
            capabilities,
            memory_mode,
        )
        selected_tools = tools
        if selected_tools is None and preset is not None:
            selected_tools = preset.tools
        selected_skills = skills
        if selected_skills is None and preset is not None:
            selected_skills = preset.skills
        selected_prompt = system_prompt
        if selected_prompt is None and preset is not None:
            selected_prompt = preset.system_prompt
        try:
            core, resolved = await create_async_hosted_agent(
                AgentAssemblyRequest(
                    config=coerce_config(config),
                services=services,
                execution_scope=execution_scope,
                conversation_id=conversation_id,
                conversation_metadata=conversation_metadata,
                runtime_metadata=runtime_metadata,
                llm_client=llm,
                tool_specs=selected_tools,
                skill_specs=selected_skills,
                system_prompt=selected_prompt,
                permission_callback=permission_callback,
                plan_step_verifier=plan_step_verifier,
                async_plan_step_verifier=async_plan_step_verifier,
                mcp_servers=tuple(mcp) if mcp is not None else None,
                redaction_callback=redaction_callback,
                redaction_fail_closed=redaction_fail_closed,
                final_answer_streaming=final_answer_streaming,
                output_policy=output_policy,
                async_output_policy=async_output_policy,
                output_policy_failure_mode=output_policy_failure_mode,
                capabilities=capabilities,
                deps=deps,
                shell_execution_policy=shell_execution_policy,
                require_shell_containment=require_shell_containment,
                run_budget=run_budget,
                usage_dimensions=usage_dimensions,
                goal_execution=goal_execution,
                async_transcript_resolver=async_transcript_resolver,
                transcript_timeout_seconds=transcript_timeout_seconds,
                async_tool_catalog_resolver=async_tool_catalog_resolver,
                    tool_catalog_timeout_seconds=tool_catalog_timeout_seconds,
                )
            )
        except Exception as exc:
            mapped = map_public_error(
                exc,
                config=config,
                operation="construct",
            )
            if mapped is exc:
                raise
            raise mapped from exc

        handle = AgentHandle(core, on_event=on_event)
        sync_facade = Agent.__new__(Agent)
        sync_facade._handle = handle
        sync_facade._run_gate = RunGate()
        sync_facade._capabilities = capabilities
        sync_facade._deps = deps

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
    def service_manifest(self) -> HostedServiceManifest:
        """Return the resolved hosted capability and service manifest."""
        if self._uses_sync_compatibility():
            return cast(
                HostedServiceManifest,
                self.runtime.resolved_services.manifest,
            )
        return self._resolved_async_services().manifest

    def _uses_sync_compatibility(self) -> bool:
        return self._async_owned_services is None and not self.closed

    @property
    def usage_ledger(self) -> Any:
        """Return the native async usage service bound to this runtime."""
        if self._uses_sync_compatibility():
            service = self.runtime._model_accounting.usage_accounting
            if service is None:
                raise RuntimeError("Usage accounting is not configured")
            return service
        return self._resolved_async_services().usage

    @property
    def session_search(self) -> Any:
        """Return the native async session-search service for this runtime."""
        if self._uses_sync_compatibility():
            service = self.runtime.resolved_services.sessions.search
            if service is None:
                raise RuntimeError("Session search is not configured")
            return service
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
        if self._uses_sync_compatibility():
            if service_name == "usage":
                async def call_sync_service() -> Any:
                    return await call_async_service(
                        self.usage_ledger,
                        method_name,
                        *args,
                        **kwargs,
                    )

                return await self._invoke_async(
                    operation,
                    call_sync_service,
                    serialized=serialized,
                )
            return await self._call_sync_facade(
                operation,
                *args,
                serialized=serialized,
                **kwargs,
            )

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

    async def _call_sync_facade(
        self,
        operation: str,
        /,
        *args: Any,
        serialized: bool = False,
        **kwargs: Any,
    ) -> Any:
        async def call() -> Any:
            return await call_async_service(
                self._agent,
                operation,
                *args,
                **kwargs,
            )

        return await self._invoke_async(
            operation,
            call,
            serialized=serialized,
        )

    async def _call_sync_session_search(
        self,
        operation: str,
        method_name: str,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async def call() -> Any:
            return await call_async_service(
                self.session_search,
                method_name,
                *args,
                **kwargs,
            )

        return await self._invoke_async(operation, call)

    async def list_memory_proposals(self) -> tuple[MemoryProposal, ...]:
        if self._uses_sync_compatibility():
            return cast(
                tuple[MemoryProposal, ...],
                await self._call_sync_facade("list_memory_proposals"),
            )

        async def operation() -> tuple[MemoryProposal, ...]:
            policy = self.runtime.memory_context.async_policy
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
        if self._uses_sync_compatibility():
            return cast(
                MemoryProposal,
                await self._call_sync_facade(
                    "approve_memory_proposal",
                    proposal_id,
                ),
            )

        async def operation() -> MemoryProposal:
            policy = self.runtime.memory_context.async_policy
            if policy is None:
                raise RuntimeError("Memory is not configured")
            return memory_proposal_snapshot(await policy.approve(proposal_id))

        return await self._invoke_async("approve_memory_proposal", operation)

    async def reject_memory_proposal(
        self,
        proposal_id: str,
    ) -> MemoryProposal:
        if self._uses_sync_compatibility():
            return cast(
                MemoryProposal,
                await self._call_sync_facade(
                    "reject_memory_proposal",
                    proposal_id,
                ),
            )

        async def operation() -> MemoryProposal:
            policy = self.runtime.memory_context.async_policy
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
        if self._uses_sync_compatibility():
            return cast(
                tuple[LearningProposal, ...],
                await self._call_sync_facade(
                    "list_learning_proposals",
                    status=status,
                    limit=limit,
                ),
            )

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
        if self._uses_sync_compatibility():
            return cast(
                LearningProposal,
                await self._call_sync_facade(
                    "get_learning_proposal",
                    proposal_id,
                ),
            )

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
        if self._uses_sync_compatibility():
            return cast(
                LearningProposal,
                await self._call_sync_facade(
                    "approve_learning_proposal",
                    proposal_id,
                    approved_by=approved_by,
                ),
            )

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
        if self._uses_sync_compatibility():
            return cast(
                LearningReview,
                await self._call_sync_facade(
                    "review_learning",
                    trigger=trigger,
                    turn_id=turn_id,
                    host_confirmed_success=host_confirmed_success,
                    serialized=True,
                ),
            )

        async def operation() -> LearningReview:
            outcome = await self.runtime.learning.review_async(
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
        if self._uses_sync_compatibility():
            return cast(
                LearningProposal,
                await self._call_sync_facade(
                    "reject_learning_proposal",
                    proposal_id,
                    rejected_by=rejected_by,
                ),
            )

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
        if self._uses_sync_compatibility():
            return cast(
                tuple[GovernedSkill, ...],
                await self._call_sync_facade(
                    "list_governed_skills",
                    scope=scope,
                ),
            )

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
                skill = governed_skill_snapshot(record)
                viewed.append(
                    await call_async_service(
                        store,
                        "record_usage",
                        name=skill.name,
                        scope=skill.scope,
                        version=skill.version,
                        digest=skill.digest,
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
        if self._uses_sync_compatibility():
            return cast(
                GovernedSkill,
                await self._call_sync_facade(
                    "rollback_skill",
                    revision_id,
                    scope=scope,
                    approved_by=approved_by,
                ),
            )

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
        if self._uses_sync_compatibility():
            return cast(
                tuple[GovernedSkillRevision, ...],
                await self._call_sync_facade(
                    "list_skill_revisions",
                    name,
                    scope=scope,
                    limit=limit,
                ),
            )

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
        if self._uses_sync_compatibility():
            return cast(
                tuple[GovernedSkill, ...],
                await self._call_sync_facade(
                    "confirm_skill_success",
                    turn_id=turn_id,
                ),
            )

        async def operation() -> tuple[GovernedSkill, ...]:
            records = await self.runtime.skill_context.confirm_success_async(
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
        if self._uses_sync_compatibility():
            return cast(
                SessionSearchPage,
                await self._call_sync_session_search(
                    "search_sessions",
                    "search",
                    query,
                    limit=limit,
                    cursor=cursor,
                ),
            )

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
        if self._uses_sync_compatibility():
            return cast(
                SessionWindow,
                await self._call_sync_session_search(
                    "read_session_window",
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
        if self._uses_sync_compatibility():
            async def sync_operation() -> dict[str, Any]:
                trace = self.runtime.trace_logger
                store = getattr(trace, "artifact_store", None)
                if store is None:
                    raise RuntimeError("Trace artifacts are unavailable")
                record = await call_async_service(
                    store,
                    "read",
                    artifact_id,
                    mode=mode,
                    offset=offset,
                    max_bytes=max_bytes,
                )
                return _artifact_read_payload(record)

            return await self._invoke_async("read_artifact", sync_operation)

        async def operation() -> dict[str, Any]:
            record = await call_async_service(
                self._resolved_async_services().artifacts,
                "read",
                artifact_id,
                mode=mode,
                offset=offset,
                max_bytes=max_bytes,
            )
            return _artifact_read_payload(record)

        return await self._invoke_async("read_artifact", operation)

    async def close(self) -> None:
        async def operation() -> None:
            owned = self._async_owned_services
            failure: BaseException | None = None
            operations: list[Callable[[], Awaitable[object]]] = [
                self.runtime.resources.flush,
                self._handle.close,
            ]
            if owned is not None:
                operations.append(owned.aclose_owned)
            for close_operation in operations:
                try:
                    await close_operation()
                except asyncio.CancelledError as exc:
                    if failure is not None:
                        exc.add_note(
                            "async close previously failed with "
                            f"{type(failure).__name__}: {failure}"
                        )
                    failure = exc
                    break
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note(
                            "async close also failed with "
                            f"{type(exc).__name__}: {exc}"
                        )
            if not isinstance(failure, asyncio.CancelledError):
                self._async_owned_services = None
            if failure is not None:
                raise failure

        await self._invoke_async("close", operation, serialized=True)


def _artifact_read_payload(record: object) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    to_dict = getattr(record, "to_dict", None)
    if callable(to_dict):
        return cast(dict[str, Any], to_dict())
    raise TypeError("async artifact store returned an unsupported read")
