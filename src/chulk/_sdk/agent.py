"""Public synchronous SDK agent facade."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import threading
from typing import Any, Callable, TypeVar, cast
from uuid import uuid4

from chulk.capabilities import Capabilities, MemoryMode
from chulk._sdk.config import AgentConfig, AgentPreset
from chulk._sdk.construction import (
    PermissionCallback,
    _build_handle,
    _selected_capabilities,
)
from chulk._sdk.error_mapping import map_public_error
from chulk._sdk.event_channel import RunEventChannel, RunGate
from chulk._sdk.events import EventCallback, failure_event, terminal_event
from chulk._sdk.results import (
    PlanResult,
    RunResult,
    governed_skill_snapshot,
    governed_skill_revision_snapshot,
    learning_proposal_snapshot,
    memory_proposal_snapshot,
    run_result_from_runtime,
)
from chulk.config import Config
from chulk.core import Agent as CoreAgent
from chulk.llm import LLMClient
from chulk.events import AgentEvent, EventName
from chulk.execution import ExecutionBackend
from chulk.goals import GoalExecutionContext
from chulk.hosting import (
    AsyncTranscriptResolver,
    ExecutionScope,
    RuntimeServices,
    TranscriptResolver,
)
from chulk.hosting.tool_catalog import (
    AsyncToolCatalogResolver,
    ToolCatalogResolver,
)
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
from chulk.sessions import (
    SessionSearchPage,
    SessionSearchService,
    SessionWindow,
)
from chulk.tools import ShellExecutionPolicy, ToolExecutionContext
from chulk.streaming import (
    AsyncIncrementalOutputPolicy,
    FinalAnswerStreamingMode,
    IncrementalOutputPolicy,
    OutputPolicyFailureMode,
)
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

T = TypeVar("T")


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
        final_answer_streaming: FinalAnswerStreamingMode | str = FinalAnswerStreamingMode.VALIDATED,
        output_policy: IncrementalOutputPolicy | None = None,
        async_output_policy: AsyncIncrementalOutputPolicy | None = None,
        output_policy_failure_mode: OutputPolicyFailureMode | str = OutputPolicyFailureMode.CLOSED,
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
        transcript_resolver: TranscriptResolver | None = None,
        async_transcript_resolver: AsyncTranscriptResolver | None = None,
        transcript_timeout_seconds: float | None = None,
        tool_catalog_resolver: ToolCatalogResolver | None = None,
        async_tool_catalog_resolver: AsyncToolCatalogResolver | None = None,
        tool_catalog_timeout_seconds: float | None = None,
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
                final_answer_streaming=final_answer_streaming,
                output_policy=output_policy,
                async_output_policy=async_output_policy,
                output_policy_failure_mode=output_policy_failure_mode,
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
                transcript_resolver=transcript_resolver,
                async_transcript_resolver=async_transcript_resolver,
                transcript_timeout_seconds=transcript_timeout_seconds,
                tool_catalog_resolver=tool_catalog_resolver,
                async_tool_catalog_resolver=async_tool_catalog_resolver,
                tool_catalog_timeout_seconds=tool_catalog_timeout_seconds,
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

    def cancel(self) -> bool:
        """Cooperatively cancel the active synchronous turn, if any."""
        return self._invoke("cancel", self._handle.cancel)

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
