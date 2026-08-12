"""Configured agent assembly behind the public runtime facade."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
import warnings
from typing import Any, cast

from chulk._version import __version__
from chulk.capabilities import Capabilities, MemoryMode
from chulk.config import Config
from chulk.core import Agent, AgentState
from chulk.core.context import ContextBudget
from chulk.core.events import AgentEvent, TraceEvent
from chulk.core.prompts import BASE_SYSTEM_PROMPT
from chulk.execution import (
    ExecutionBackend,
    ExecutionContextLifecycle,
    HostExecutionBackend,
)
from chulk.goals.runtime import GoalExecutionContext
from chulk.hosting import (
    AsyncExternalTranscriptSessionRuntimeServices,
    AsyncRuntimeServices,
    AsyncTranscriptResolver,
    DisabledHostedService,
    ExecutionScope,
    ExternalTranscriptSessionRuntimeServices,
    RuntimeServices,
    SessionRuntimeServices,
    SkillRuntimeServices,
    TranscriptResolver,
)
from chulk.hosting.async_utils import call_async_service, close_async_resource
from chulk.hosting.services import ResolvedRuntimeServices
from chulk.hosting.sinks import (
    BufferedAsyncAuditSink,
    BufferedAsyncEventSink,
    BufferedAsyncTraceSink,
)
from chulk.hosting.tool_catalog import (
    AsyncToolCatalogResolver,
    ToolCatalogResolver,
)
from chulk.llm import LLMClient
from chulk.llm.lifecycle import close_resources
from chulk.mcp import MCPServerConfig
from chulk.media import ContentStore, LocalTextExtractor, MediaProcessorRegistry
from chulk.memory import (
    AsyncMemoryPolicy,
    ConversationMemory,
    MemoryPolicy,
    SQLiteMemoryStore,
)
from chulk.plugins import LocalPluginRegistry
from chulk.sessions import (
    AsyncExternalTranscriptRecorder,
    AsyncSessionRecorder,
    ExternalTranscriptRecorder,
    SessionSearchService,
    SQLiteSessionStore,
    SessionRecorder,
)
from chulk.skills import (
    LearningProposalService,
    LearningReviewCoordinator,
    LearningReviewPolicy,
    LearningReviewQuota,
    RestrictedLearningReviewer,
    SQLiteSkillLifecycleStore,
    SkillLifecycleManager,
    SkillRegistry,
)
from chulk.tools import (
    ShellExecutionPolicy,
    Tool,
    ToolExecutionContext,
)
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    permission_policy_for_profile,
)
from chulk.tools.policy import ToolPolicyHooks
from chulk.tracing import JSONLTraceLogger
from chulk.tracing.artifacts import TraceArtifactStore
from chulk.usage import (
    ModelUsageAccounting,
    RunBudget,
    SQLiteUsageStore,
    UsageDimensions,
)
from chulk.streaming import (
    AsyncIncrementalOutputPolicy,
    FinalAnswerStreamingMode,
    IncrementalOutputPolicy,
    OutputPolicyFailureMode,
)
from chulk._runtime.request import (
    client_model_capabilities,
    default_llm_client_factory,
)
from chulk._runtime.services import resolve_runtime_services
from chulk._runtime.sessions import (
    block_unresolved_tool_intent,
    create_agent_state,
    create_agent_state_async,
    create_external_agent_state,
    create_external_agent_state_async,
    session_result_redactor,
    summary_source_ordinal,
)
from chulk._runtime.skills import (
    SkillSpecResolution,
    resolve_skill_specs,
    resolve_skill_specs_async,
    skill_capability_names,
)
from chulk._runtime.tools import (
    MCPBridgeRequired,
    create_tool_registry,
)


def assemble_agent(
    config: Config,
    llm_client_factory: Callable[[Config], LLMClient] | None = None,
    *,
    conversation_id: str | None = None,
    conversation_metadata: dict[str, object] | None = None,
    llm_client: LLMClient | None = None,
    tool_specs: Iterable[object] | None = None,
    skill_specs: object | Iterable[object] | None = None,
    system_prompt: str | None = None,
    permission_callback: Callable[
        [PermissionRequest, PermissionDecisionRecord],
        PermissionDecision | bool,
    ]
    | None = None,
    mcp_servers: Iterable[MCPServerConfig] | None = None,
    event_sink: Callable[[AgentEvent], None] | None = None,
    redaction_callback: Callable[[str, str, dict], str] | None = None,
    redaction_fail_closed: bool = False,
    final_answer_streaming: FinalAnswerStreamingMode | str = FinalAnswerStreamingMode.VALIDATED,
    output_policy: IncrementalOutputPolicy | None = None,
    async_output_policy: AsyncIncrementalOutputPolicy | None = None,
    output_policy_failure_mode: OutputPolicyFailureMode | str = OutputPolicyFailureMode.CLOSED,
    capabilities: Capabilities | None = None,
    deps: object | None = None,
    shell_execution_policy: ShellExecutionPolicy | None = None,
    require_shell_containment: bool = False,
    execution_backend: ExecutionBackend | None = None,
    memory_namespace: str | None = None,
    profile_id: str | None = None,
    allowed_skill_names: Iterable[str] | None = None,
    runtime_metadata: dict | None = None,
    run_budget: RunBudget | None = None,
    additional_run_budgets: Iterable[RunBudget] = (),
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
    agent_factory: Callable[..., Agent],
    bridge_tool_factory: Callable[[Iterable[MCPServerConfig]], Iterable[Tool]],
    mcp_bridge_required: MCPBridgeRequired,
    mcp_provider_path: Callable[..., str],
    unresolved_tool_handler: Callable[..., None] = block_unresolved_tool_intent,
) -> Agent:
    """Create the configured Chulk agent runtime."""
    if llm_client is not None and llm_client_factory is not None:
        raise ValueError("Pass either llm_client or llm_client_factory, not both")
    service_resolution = resolve_runtime_services(
        services,
        execution_scope,
        conversation_id=conversation_id,
        tool_specs=tool_specs,
        skill_specs=skill_specs,
        execution_backend=execution_backend,
        plugin_registry=plugin_registry,
        content_store=content_store,
        media_processors=media_processors,
        memory_namespace=memory_namespace,
    )
    hosted_state = service_resolution.hosted_state
    resolved_services = service_resolution.services
    conversation_id = service_resolution.conversation_id
    execution_scope = service_resolution.execution_scope
    session_store: Any
    session_recorder: Any
    session_search_service: Any
    external_sessions = False

    if llm_client_factory is None:
        llm_client_factory = default_llm_client_factory
    effective_profile_id = profile_id or config.profile_id
    goal_snapshot = (
        goal_execution.assert_boundary()
        if goal_execution is not None
        else None
    )
    if goal_snapshot is not None and goal_snapshot.profile_id != effective_profile_id:
        raise ValueError("goal execution profile does not match runtime profile")
    if (
        goal_snapshot is not None
        and run_budget is not None
        and run_budget != goal_snapshot.budget
    ):
        raise ValueError("run_budget does not match the claimed goal budget")
    if (
        goal_snapshot is not None
        and usage_dimensions is not None
        and usage_dimensions.goal_id not in {None, goal_snapshot.id}
    ):
        raise ValueError("usage dimensions do not match the claimed goal")
    plugins_enabled = (
        resolved_services is None
        or resolved_services.is_enabled("plugins")
    )
    selected_plugin_registry = (
        resolved_services.plugins
        if resolved_services is not None and plugins_enabled
        else plugin_registry
        or LocalPluginRegistry(
            config.runtime_dir,
            profile_id=effective_profile_id,
        )
    ) if plugins_enabled else None
    if selected_plugin_registry is not None:
        plugin_profile_id = getattr(
            selected_plugin_registry,
            "profile_id",
            effective_profile_id,
        )
        if plugin_profile_id != effective_profile_id:
            raise ValueError(
                "plugin registry profile does not match the runtime profile"
            )
        plugin_audit_report = selected_plugin_registry.verify_startup()
    else:
        plugin_audit_report = None
    effective_conversation_metadata = dict(conversation_metadata or {})
    metadata_profile_id = effective_conversation_metadata.get("profile_id")
    if metadata_profile_id is not None and metadata_profile_id != effective_profile_id:
        raise ValueError(
            "conversation metadata profile_id does not match the runtime profile"
        )
    effective_conversation_metadata["profile_id"] = effective_profile_id
    memory_enabled = (
        resolved_services is None
        or resolved_services.is_enabled("memory")
    )
    memory_store = (
        resolved_services.memory
        if resolved_services is not None and memory_enabled
        else SQLiteMemoryStore(
            config.store_path,
            namespace=memory_namespace,
        )
    ) if memory_enabled else None
    selected_capabilities = capabilities or Capabilities.full()
    if not memory_enabled and selected_capabilities.memory != MemoryMode.OFF:
        raise ValueError(
            "memory capability requires the hosted memory service"
        )
    memory_policy = (
        MemoryPolicy(memory_store, selected_capabilities.memory)
        if memory_store is not None
        else None
    )
    if resolved_services is not None:
        external_sessions = isinstance(
            resolved_services.sessions,
            ExternalTranscriptSessionRuntimeServices,
        )
        if not isinstance(
            resolved_services.sessions,
            (SessionRuntimeServices, ExternalTranscriptSessionRuntimeServices),
        ):
            raise TypeError(
                "hosted sessions service must be SessionRuntimeServices or "
                "ExternalTranscriptSessionRuntimeServices"
            )
        if external_sessions and transcript_resolver is None:
            raise ValueError(
                "external transcript sessions require transcript_resolver"
            )
        if external_sessions and async_transcript_resolver is not None:
            raise ValueError(
                "synchronous hosted runs cannot use async_transcript_resolver"
            )
        if not external_sessions and (
            transcript_resolver is not None
            or async_transcript_resolver is not None
        ):
            raise ValueError(
                "transcript resolvers require external transcript sessions"
            )
        skills_enabled = resolved_services.is_enabled("skills")
        if skills_enabled and not isinstance(
            resolved_services.skills,
            SkillRuntimeServices,
        ):
            raise TypeError("hosted skills service must be SkillRuntimeServices")
        if not skills_enabled and skill_specs:
            raise ValueError(
                "skill specifications require the hosted skills service"
            )
        if external_sessions:
            external_service = cast(
                ExternalTranscriptSessionRuntimeServices,
                resolved_services.sessions,
            )
            session_store = external_service.journal
            session_search_service = DisabledHostedService(
                "external_transcript_search"
            )
        else:
            session_service = cast(
                SessionRuntimeServices,
                resolved_services.sessions,
            )
            session_store = session_service.store
            session_search_service = session_service.search
        selected_content_store = (
            resolved_services.content
            if resolved_services.is_enabled("content")
            else None
        )
        selected_media_processors = (
            resolved_services.media
            if resolved_services.is_enabled("media")
            else None
        )
        skill_registry = (
            resolved_services.skills.registry if skills_enabled else None
        )
        skill_lifecycle_store = (
            resolved_services.skills.lifecycle_store if skills_enabled else None
        )
        skill_lifecycle = (
            resolved_services.skills.lifecycle if skills_enabled else None
        )
        learning_proposals = (
            resolved_services.skills.learning_proposals if skills_enabled else None
        )
        learning_reviewer = (
            resolved_services.skills.learning_reviewer if skills_enabled else None
        )
        if external_sessions:
            state = hosted_state or create_external_agent_state(
                session_store,
                cast(str, conversation_id),
                execution_scope=cast(ExecutionScope, execution_scope),
            )
        else:
            state = hosted_state or create_agent_state(
                session_store,
                conversation_id,
                execution_scope=execution_scope,
                unresolved_tool_handler=unresolved_tool_handler,
            )
        trace_logger = resolved_services.traces
    else:
        session_store = SQLiteSessionStore(config.store_path)
        selected_content_store = content_store or ContentStore(
            config.store_path,
            config.runtime_dir / "content",
            profile_id=effective_profile_id,
        )
        selected_content_store.sweep_expired()
        selected_media_processors = media_processors or MediaProcessorRegistry(
            (LocalTextExtractor(),)
        )
        profile_skills_dir = config.runtime_dir / "profile-skills"
        registry_skill_dirs = tuple(
            dict.fromkeys((*config.skills_dirs, profile_skills_dir))
        )
        skill_registry = SkillRegistry(
            config.skills_dir,
            skills_dirs=registry_skill_dirs,
            max_skills=config.max_skills_per_turn,
            max_content_chars=config.max_skill_content_chars,
        )
        state = create_agent_state(
            session_store,
            conversation_id,
            unresolved_tool_handler=unresolved_tool_handler,
        )
        trace_logger = JSONLTraceLogger(
            config.traces_dir,
            state.conversation_id,
            defer_until_event=(
                TraceEvent.TURN_STARTED
                if conversation_id is None
                else None
            ),
        )
        skill_lifecycle_store = SQLiteSkillLifecycleStore(
            config.store_path,
            profile_id=effective_profile_id,
        )
        skill_lifecycle = SkillLifecycleManager(
            skill_lifecycle_store,
            project_skills_dir=config.skills_dir,
            profile_skills_dir=profile_skills_dir,
            project_lock_path=config.skills_dir.parent / "skills.lock",
            profile_lock_path=config.runtime_dir / "profile-skills.lock",
            registry=skill_registry,
        )
        skill_lifecycle.register_existing(scope="project")
        skill_lifecycle.register_existing(scope="profile")
        learning_proposals = LearningProposalService(
            memory_store=cast(SQLiteMemoryStore, memory_store),
            lifecycle_store=skill_lifecycle_store,
            lifecycle_manager=skill_lifecycle,
            automatic_approval_enabled=automatic_learning_approval,
        )
    if execution_scope is None:
        execution_scope = ExecutionScope.local(
            agent_id=f"profile:{effective_profile_id}",
            agent_version=__version__,
            conversation_id=state.conversation_id,
            profile_id=effective_profile_id,
        )
    effective_conversation_metadata["execution_scope"] = execution_scope.to_dict()
    effective_conversation_metadata["execution_scope_key"] = execution_scope.key

    def audit_session_read(
        event_type: str,
        payload: dict,
    ) -> None:
        trace_logger.activate()
        trace_logger.log(event_type, payload)
        if resolved_services is not None:
            resolved_services.audit.record(
                event_type,
                payload,
                scope=execution_scope,
            )

    def hosted_audit(event_type: str, payload: dict) -> None:
        if resolved_services is None:
            return
        resolved_services.audit.record(
            event_type,
            payload,
            scope=execution_scope,
        )

    if resolved_services is None:
        session_search_service = SessionSearchService(
            session_store,
            profile_id=effective_profile_id,
            redactor=session_result_redactor(
                redaction_callback,
                fail_closed=redaction_fail_closed,
            ),
            audit_callback=audit_session_read,
        )
    if skill_registry is not None:
        skill_registry.load_metadata()
        skill_resolution = resolve_skill_specs(skill_registry, skill_specs)
    else:
        skill_resolution = SkillSpecResolution(pinned_skill_names=[], warnings=[])
    if allowed_skill_names is not None and skill_registry is not None:
        allowed = tuple(allowed_skill_names)
        skill_registry.restrict_to(
            [
                *allowed,
                *(
                    name
                    for name in skill_resolution.pinned_skill_names
                    if name in allowed
                ),
            ]
        )
        skill_resolution = SkillSpecResolution(
            pinned_skill_names=[
                name for name in skill_resolution.pinned_skill_names if name in allowed
            ],
            warnings=skill_resolution.warnings,
        )
    for warning_payload in skill_resolution.warnings:
        warnings.warn(warning_payload["message"], UserWarning, stacklevel=2)
        trace_logger.log("skill_config_warning", warning_payload)
    if skill_resolution.warnings:
        trace_logger.activate()
    conversation_memory = ConversationMemory(max_messages=config.history_limit)
    if conversation_id is not None and not (
        resolved_services is not None and external_sessions
    ):
        latest_summary = session_store.load_latest_summary(state.conversation_id)
        recent_messages = session_store.load_recent_messages(
            state.conversation_id,
            config.history_limit,
            after_ordinal=summary_source_ordinal(latest_summary),
        )
        conversation_memory.replace(
            recent_messages,
            conversation_summary=latest_summary.content
            if latest_summary is not None
            else None,
            summary_message_count=latest_summary.source_message_count
            if latest_summary is not None
            else 0,
        )
        state.messages = conversation_memory.recent()
        state.conversation_summary = conversation_memory.conversation_summary
    if resolved_services is not None and external_sessions:
        session_recorder = ExternalTranscriptRecorder(
            session_store,
            cast(
                ExternalTranscriptSessionRuntimeServices,
                resolved_services.sessions,
            ).projections,
            state.conversation_id,
            execution_scope,
        )
    else:
        session_recorder = SessionRecorder(
            session_store,
            state.conversation_id,
            provider=config.llm_provider,
            model=config.model,
            trace_path=trace_logger.path,
            lazy=conversation_id is None and not conversation_metadata,
            metadata=effective_conversation_metadata,
        )
    client_is_owned = llm_client is None
    client = llm_client if llm_client is not None else llm_client_factory(config)
    if hasattr(client, "bind_config"):
        client = client.bind_config(config)  # type: ignore[assignment, attr-defined]
    if resolved_services is None:
        learning_reviewer = LearningReviewCoordinator(
            reviewer=RestrictedLearningReviewer(client),
            proposal_service=cast(LearningProposalService, learning_proposals),
            lifecycle_store=cast(
                SQLiteSkillLifecycleStore,
                skill_lifecycle_store,
            ),
            policy=learning_review_policy,
            quota=learning_review_quota,
            automatic_approval=automatic_learning_approval,
            granted_capabilities=tuple(
                sorted(skill_capability_names(selected_capabilities))
            ),
        )
    selection_result = getattr(client, "selection_result", None)
    effective_runtime_metadata = dict(runtime_metadata or {})
    if selection_result is not None and hasattr(selection_result, "to_dict"):
        effective_runtime_metadata["model_selection"] = selection_result.to_dict()
    model_capabilities = client_model_capabilities(client, config)
    context_budget = ContextBudget(
        max_prompt_tokens=model_capabilities.context_window_tokens,
        response_reserve_tokens=model_capabilities.default_response_reserve_tokens,
        max_input_tokens=model_capabilities.max_input_tokens,
    )
    base_usage_dimensions = usage_dimensions or UsageDimensions(
        profile_id=effective_profile_id,
        channel=(
            str(effective_conversation_metadata["channel"])
            if isinstance(effective_conversation_metadata.get("channel"), str)
            else None
        ),
    )
    if goal_snapshot is not None:
        base_usage_dimensions = replace(
            base_usage_dimensions,
            goal_id=goal_snapshot.id,
        )
    if base_usage_dimensions.profile_id != effective_profile_id:
        raise ValueError(
            "usage dimensions profile_id does not match the runtime profile"
        )
    if (
        base_usage_dimensions.conversation_id is not None
        and base_usage_dimensions.conversation_id != state.conversation_id
    ):
        raise ValueError(
            "usage dimensions conversation_id does not match the runtime conversation"
        )
    usage_accounting = (
        resolved_services.usage
        if resolved_services is not None
        else ModelUsageAccounting(
            SQLiteUsageStore(config.store_path),
            client=client,
            dimensions=replace(
                base_usage_dimensions,
                conversation_id=state.conversation_id,
            ),
            budget=(
                goal_snapshot.budget
                if goal_snapshot is not None
                else run_budget or RunBudget()
            ),
            additional_budgets=additional_run_budgets,
            max_output_tokens=(
                model_capabilities.max_output_tokens
                or model_capabilities.default_response_reserve_tokens
            ),
            trace_path=trace_logger.path,
            boundary_callback=(
                goal_execution.assert_boundary
                if goal_execution is not None
                else None
            ),
        )
    )
    configured_mcp_servers = (
        tuple(mcp_servers) if mcp_servers is not None else config.mcp_servers
    )
    active_mcp_servers = (
        configured_mcp_servers if selected_capabilities.external_services else ()
    )
    backend_is_owned = resolved_services is None and execution_backend is None
    selected_execution_backend = (
        resolved_services.execution
        if resolved_services is not None
        else execution_backend
        or HostExecutionBackend(
            config.project_root,
            shell_timeout_seconds=config.shell_timeout_seconds,
            max_stdout_bytes=config.max_tool_stdout_chars,
            max_stderr_bytes=config.max_tool_stderr_chars,
            shell_execution_policy=shell_execution_policy,
            require_shell_containment=require_shell_containment,
        )
    )
    execution_lifecycle = ExecutionContextLifecycle(selected_execution_backend)
    tool_registry, mcp_bridge_tool_names = create_tool_registry(
        config,
        memory_store,
        tool_specs,
        active_mcp_servers,
        llm_client=client,
        capabilities=selected_capabilities,
        memory_policy=memory_policy,
        session_search_service=session_search_service,
        deps=deps,
        shell_execution_policy=shell_execution_policy,
        require_shell_containment=require_shell_containment,
        artifact_store=(
            resolved_services.artifacts
            if (
                resolved_services is not None
                and resolved_services.is_enabled("artifacts")
            )
            else trace_logger.artifact_store
            if resolved_services is None
            else None
        ),
        bridge_tool_factory=bridge_tool_factory,
        mcp_bridge_required=mcp_bridge_required,
    )
    available_tool_names = {tool.name for tool in tool_registry.list_tools()}
    skill_capabilities = skill_capability_names(selected_capabilities)
    if available_tool_names:
        skill_capabilities.add("tools")
    if skill_registry is not None:
        skill_registry.configure_environment(
            available_tools=available_tool_names,
            capabilities=skill_capabilities,
        )
    if active_mcp_servers:
        trace_logger.log(
            "mcp_config_loaded",
            {
                "config_path": str(config.mcp_config_path),
                "servers": [
                    server.to_dict()
                    if hasattr(server, "to_dict")
                    else {"server": str(server)}
                    for server in active_mcp_servers
                ],
                "provider_path": mcp_provider_path(
                    config,
                    active_mcp_servers,
                    llm_client=client,
                ),
            },
        )
        trace_logger.log(
            "mcp_tool_discovery_completed",
            {
                "bridge_tool_names": mcp_bridge_tool_names,
                "bridge_required": mcp_bridge_required(
                    config,
                    active_mcp_servers,
                    llm_client=client,
                ),
            },
        )
    owned_resources: list[object] = [client] if client_is_owned else []
    if resolved_services is not None:
        owned_resources.extend(resolved_services.owned_resources)
    if backend_is_owned:
        owned_resources.append(selected_execution_backend)
    try:
        agent = agent_factory(
            client,
            state=state,
            memory=conversation_memory,
            memory_store=memory_store,
            memory_policy=memory_policy,
            skill_registry=skill_registry,
            trace_logger=trace_logger,
            tool_registry=tool_registry,
            max_tool_calls_per_turn=config.max_tool_calls_per_turn,
            max_skills_per_turn=config.max_skills_per_turn,
            max_skill_content_chars=config.max_skill_content_chars,
            trace_max_prompt_chars=config.trace_max_prompt_chars,
            max_observation_chars=config.max_observation_chars,
            max_tool_stdout_chars=config.max_tool_stdout_chars,
            max_tool_stderr_chars=config.max_tool_stderr_chars,
            max_reflection_attempts=config.max_reflection_attempts,
            permission_policy=permission_policy_for_profile(config.permission_profile),
            permission_callback=permission_callback,
            context_budget=context_budget,
            max_model_output_tokens=(
                model_capabilities.max_output_tokens
                or model_capabilities.default_response_reserve_tokens
            ),
            event_callback=session_recorder.callback,
            event_sink=event_sink,
            audit_callback=(
                hosted_audit if resolved_services is not None else None
            ),
            redaction_callback=redaction_callback,
            redaction_fail_closed=redaction_fail_closed,
            final_answer_streaming=final_answer_streaming,
            output_policy=output_policy,
            async_output_policy=async_output_policy,
            output_policy_failure_mode=output_policy_failure_mode,
            transcript_resolver=transcript_resolver,
            async_transcript_resolver=async_transcript_resolver,
            transcript_timeout_seconds=transcript_timeout_seconds,
            pinned_skill_names=skill_resolution.pinned_skill_names,
            system_prompt=system_prompt or BASE_SYSTEM_PROMPT,
            mcp_servers=active_mcp_servers,
            mcp_bridge_tool_names=mcp_bridge_tool_names,
            owned_resources=owned_resources,
            default_tool_context=ToolExecutionContext(deps=deps)
            if deps is not None
            else None,
            runtime_metadata=effective_runtime_metadata,
            tool_context_lifecycle=execution_lifecycle,
            profile_id=effective_profile_id,
            usage_accounting=usage_accounting,
            skill_lifecycle_store=skill_lifecycle_store,
            skill_lifecycle=skill_lifecycle,
            learning_proposals=learning_proposals,
            learning_reviewer=learning_reviewer,
            plugin_registry=(
                selected_plugin_registry
                if selected_plugin_registry is not None
                else resolved_services.plugins
                if resolved_services is not None
                else None
            ),
            plugin_audit_report=plugin_audit_report,
            goal_execution=goal_execution,
            content_store=selected_content_store,
            media_processors=selected_media_processors,
            execution_scope=execution_scope,
            tool_policy_hooks=(
                cast(ToolPolicyHooks, resolved_services.tool_policy)
                if resolved_services is not None
                else None
            ),
            tool_catalog_resolver=tool_catalog_resolver,
            async_tool_catalog_resolver=async_tool_catalog_resolver,
            tool_catalog_timeout_seconds=tool_catalog_timeout_seconds,
            close_trace_logger=resolved_services is None,
        )
    except Exception:
        for resource in reversed(owned_resources):
            close_resources((resource,))
        raise
    agent.session_store = session_store
    agent.session_recorder = session_recorder
    agent.session_search_service = session_search_service
    agent.run_store = (
        resolved_services.runs
        if resolved_services is not None
        else None
    )
    agent.approval_store = (
        resolved_services.approvals
        if resolved_services is not None
        else None
    )
    agent.public_event_sink = (
        resolved_services.events
        if resolved_services is not None
        else None
    )
    agent.hosted_service_manifest = (
        resolved_services.manifest
        if resolved_services is not None
        else None
    )
    if learning_proposals is not None:
        learning_proposals.event_callback = agent._trace
    return agent


async def assemble_async_hosted_agent(
    config: Config,
    *,
    services: AsyncRuntimeServices,
    execution_scope: ExecutionScope,
    conversation_id: str | None = None,
    conversation_metadata: dict[str, object] | None = None,
    runtime_metadata: dict | None = None,
    llm_client: LLMClient | None = None,
    tool_specs: Iterable[object] | None = None,
    skill_specs: object | Iterable[object] | None = None,
    system_prompt: str | None = None,
    permission_callback: Callable[
        [PermissionRequest, PermissionDecisionRecord],
        PermissionDecision | bool,
    ]
    | None = None,
    mcp_servers: Iterable[MCPServerConfig] | None = None,
    redaction_callback: Callable[[str, str, dict], str] | None = None,
    redaction_fail_closed: bool = False,
    final_answer_streaming: FinalAnswerStreamingMode | str = FinalAnswerStreamingMode.VALIDATED,
    output_policy: IncrementalOutputPolicy | None = None,
    async_output_policy: AsyncIncrementalOutputPolicy | None = None,
    output_policy_failure_mode: OutputPolicyFailureMode | str = OutputPolicyFailureMode.CLOSED,
    capabilities: Capabilities | None = None,
    deps: object | None = None,
    shell_execution_policy: ShellExecutionPolicy | None = None,
    require_shell_containment: bool = False,
    run_budget: RunBudget | None = None,
    usage_dimensions: UsageDimensions | None = None,
    goal_execution: GoalExecutionContext | None = None,
    profile_id: str | None = None,
    async_transcript_resolver: AsyncTranscriptResolver | None = None,
    transcript_timeout_seconds: float | None = None,
    async_tool_catalog_resolver: AsyncToolCatalogResolver | None = None,
    tool_catalog_timeout_seconds: float | None = None,
    agent_factory: Callable[..., Agent],
    bridge_tool_factory: Callable[[Iterable[MCPServerConfig]], Iterable[Tool]],
    mcp_bridge_required: MCPBridgeRequired,
) -> tuple[Agent, ResolvedRuntimeServices]:
    """Assemble a hosted agent without invoking async services synchronously."""
    if tool_specs is None:
        raise ValueError("hosted runtime requires an explicit tools collection")
    if skill_specs is None:
        raise ValueError("hosted runtime requires an explicit skills collection")
    requested_conversation_id = conversation_id or execution_scope.conversation_id
    if (
        conversation_id is not None
        and execution_scope.conversation_id not in {None, conversation_id}
    ):
        raise ValueError(
            "execution scope conversation_id does not match conversation_id"
        )
    hosted_state: AgentState | None = None
    if requested_conversation_id is None:
        hosted_state = AgentState()
        requested_conversation_id = hosted_state.conversation_id
    execution_scope = execution_scope.with_conversation(requested_conversation_id)
    load_conversation_id = (
        requested_conversation_id if hosted_state is None else None
    )
    resolved = await services.resolve_async(execution_scope)
    session_store: Any
    session_recorder: Any
    session_search_service: Any
    external_sessions = False
    client: LLMClient | None = None
    client_is_owned = False
    try:
        external_sessions = isinstance(
            resolved.sessions,
            AsyncExternalTranscriptSessionRuntimeServices,
        )
        if not isinstance(
            resolved.sessions,
            (SessionRuntimeServices, AsyncExternalTranscriptSessionRuntimeServices),
        ):
            raise TypeError(
                "hosted sessions service must be SessionRuntimeServices or "
                "AsyncExternalTranscriptSessionRuntimeServices"
            )
        if external_sessions and async_transcript_resolver is None:
            raise ValueError(
                "external transcript sessions require async_transcript_resolver"
            )
        if not external_sessions and async_transcript_resolver is not None:
            raise ValueError(
                "async_transcript_resolver requires external transcript sessions"
            )
        effective_profile_id = profile_id or config.profile_id
        goal_snapshot = (
            goal_execution.assert_boundary()
            if goal_execution is not None
            else None
        )
        if (
            goal_snapshot is not None
            and goal_snapshot.profile_id != effective_profile_id
        ):
            raise ValueError(
                "goal execution profile does not match runtime profile"
            )
        if (
            goal_snapshot is not None
            and run_budget is not None
            and run_budget != goal_snapshot.budget
        ):
            raise ValueError(
                "run_budget does not match the claimed goal budget"
            )
        if (
            goal_snapshot is not None
            and usage_dimensions is not None
            and usage_dimensions.goal_id not in {None, goal_snapshot.id}
        ):
            raise ValueError(
                "usage dimensions do not match the claimed goal"
            )

        plugins_enabled = resolved.is_enabled("plugins")
        plugin_registry = resolved.plugins if plugins_enabled else None
        if plugin_registry is not None:
            plugin_profile_id = getattr(
                plugin_registry,
                "profile_id",
                effective_profile_id,
            )
            if plugin_profile_id != effective_profile_id:
                raise ValueError(
                    "plugin registry profile does not match the runtime profile"
                )
            plugin_audit_report = await call_async_service(
                plugin_registry,
                "verify_startup",
            )
        else:
            plugin_audit_report = None

        effective_metadata = dict(conversation_metadata or {})
        metadata_profile_id = effective_metadata.get("profile_id")
        if (
            metadata_profile_id is not None
            and metadata_profile_id != effective_profile_id
        ):
            raise ValueError(
                "conversation metadata profile_id does not match the "
                "runtime profile"
            )
        effective_metadata["profile_id"] = effective_profile_id
        effective_metadata["execution_scope"] = execution_scope.to_dict()
        effective_metadata["execution_scope_key"] = execution_scope.key

        if external_sessions:
            external_service = cast(
                AsyncExternalTranscriptSessionRuntimeServices,
                resolved.sessions,
            )
            session_store = external_service.journal
            session_search_service = DisabledHostedService(
                "external_transcript_search"
            )
        else:
            session_service = cast(SessionRuntimeServices, resolved.sessions)
            session_store = session_service.store
            session_search_service = session_service.search
        skills_enabled = resolved.is_enabled("skills")
        if skills_enabled and not isinstance(
            resolved.skills,
            SkillRuntimeServices,
        ):
            raise TypeError(
                "hosted skills service must be SkillRuntimeServices"
            )
        if not skills_enabled and skill_specs:
            raise ValueError(
                "skill specifications require the hosted skills service"
            )
        skill_registry = (
            resolved.skills.registry if skills_enabled else None
        )
        memory_enabled = resolved.is_enabled("memory")
        memory_store = resolved.memory if memory_enabled else None
        selected_capabilities = capabilities or Capabilities.full()
        if not memory_enabled and selected_capabilities.memory != MemoryMode.OFF:
            raise ValueError(
                "memory capability requires the hosted memory service"
            )
        memory_policy = (
            AsyncMemoryPolicy(
                memory_store,
                selected_capabilities.memory,
            )
            if memory_store is not None
            else None
        )
        if external_sessions:
            state = hosted_state or await create_external_agent_state_async(
                session_store,
                requested_conversation_id,
                execution_scope=execution_scope,
            )
        else:
            state = hosted_state or await create_agent_state_async(
                session_store,
                load_conversation_id,
                execution_scope=execution_scope,
            )

        trace_logger = BufferedAsyncTraceSink(resolved.traces)
        audit_sink = BufferedAsyncAuditSink(resolved.audit)
        event_sink = BufferedAsyncEventSink(resolved.events)

        if skill_registry is not None:
            await call_async_service(skill_registry, "load_metadata")
            skill_resolution = await resolve_skill_specs_async(
                skill_registry,
                skill_specs,
            )
        else:
            skill_resolution = SkillSpecResolution(
                pinned_skill_names=[],
                warnings=[],
            )
        for warning_payload in skill_resolution.warnings:
            warnings.warn(
                warning_payload["message"],
                UserWarning,
                stacklevel=2,
            )
            trace_logger.log("skill_config_warning", warning_payload)

        conversation_memory = ConversationMemory(
            max_messages=config.history_limit
        )
        if load_conversation_id is not None and not external_sessions:
            latest_summary = await call_async_service(
                session_store,
                "load_latest_summary",
                state.conversation_id,
            )
            recent_messages = await call_async_service(
                session_store,
                "load_recent_messages",
                state.conversation_id,
                config.history_limit,
                after_ordinal=summary_source_ordinal(latest_summary),
            )
            conversation_memory.replace(
                recent_messages,
                conversation_summary=(
                    latest_summary.content
                    if latest_summary is not None
                    else None
                ),
                summary_message_count=(
                    latest_summary.source_message_count
                    if latest_summary is not None
                    else 0
                ),
            )
            state.messages = conversation_memory.recent()
            state.conversation_summary = conversation_memory.conversation_summary
        if external_sessions:
            session_recorder = AsyncExternalTranscriptRecorder(
                session_store,
                cast(
                    AsyncExternalTranscriptSessionRuntimeServices,
                    resolved.sessions,
                ).projections,
                state.conversation_id,
                execution_scope,
            )
        else:
            session_recorder = AsyncSessionRecorder(
                session_store,
                state.conversation_id,
                provider=config.llm_provider,
                model=config.model,
                trace_path=trace_logger.path,
                lazy=(load_conversation_id is None and not conversation_metadata),
                metadata=effective_metadata,
            )
        await session_recorder.initialize()

        client_is_owned = llm_client is None
        client = (
            llm_client
            if llm_client is not None
            else default_llm_client_factory(config)
        )
        if hasattr(client, "bind_config"):
            client = client.bind_config(config)  # type: ignore[assignment, attr-defined]
        selection_result = getattr(client, "selection_result", None)
        effective_runtime_metadata = dict(runtime_metadata or {})
        if selection_result is not None and hasattr(selection_result, "to_dict"):
            effective_runtime_metadata["model_selection"] = (
                selection_result.to_dict()
            )
        model_capabilities = client_model_capabilities(client, config)
        context_budget = ContextBudget(
            max_prompt_tokens=model_capabilities.context_window_tokens,
            response_reserve_tokens=(
                model_capabilities.default_response_reserve_tokens
            ),
            max_input_tokens=model_capabilities.max_input_tokens,
        )

        configured_mcp_servers = (
            tuple(mcp_servers)
            if mcp_servers is not None
            else config.mcp_servers
        )
        active_mcp_servers = (
            configured_mcp_servers
            if selected_capabilities.external_services
            else ()
        )
        execution_lifecycle = ExecutionContextLifecycle(resolved.execution)
        tool_registry, mcp_bridge_tool_names = create_tool_registry(
            config,
            cast(SQLiteMemoryStore | None, memory_store),
            tool_specs,
            active_mcp_servers,
            llm_client=client,
            capabilities=selected_capabilities,
            memory_policy=cast(MemoryPolicy | None, memory_policy),
            session_search_service=cast(
                SessionSearchService,
                session_search_service,
            ),
            deps=deps,
            shell_execution_policy=shell_execution_policy,
            require_shell_containment=require_shell_containment,
            artifact_store=(
                cast(TraceArtifactStore, resolved.artifacts)
                if resolved.is_enabled("artifacts")
                else None
            ),
            bridge_tool_factory=bridge_tool_factory,
            mcp_bridge_required=mcp_bridge_required,
            async_services=True,
        )
        available_tool_names = {
            tool.name for tool in tool_registry.list_tools()
        }
        skill_capabilities = skill_capability_names(selected_capabilities)
        if available_tool_names:
            skill_capabilities.add("tools")
        if skill_registry is not None:
            await call_async_service(
                skill_registry,
                "configure_environment",
                available_tools=available_tool_names,
                capabilities=skill_capabilities,
            )

        owned_resources: list[object] = [client] if client_is_owned else []

        def hosted_audit(event_type: str, payload: dict) -> None:
            audit_sink.record(
                event_type,
                payload,
                scope=execution_scope,
            )

        agent = agent_factory(
            client,
            state=state,
            memory=conversation_memory,
            memory_store=None,
            memory_policy=None,
            skill_registry=cast(SkillRegistry | None, skill_registry),
            trace_logger=cast(JSONLTraceLogger, trace_logger),
            tool_registry=tool_registry,
            max_tool_calls_per_turn=config.max_tool_calls_per_turn,
            max_skills_per_turn=config.max_skills_per_turn,
            max_skill_content_chars=config.max_skill_content_chars,
            trace_max_prompt_chars=config.trace_max_prompt_chars,
            max_observation_chars=config.max_observation_chars,
            max_tool_stdout_chars=config.max_tool_stdout_chars,
            max_tool_stderr_chars=config.max_tool_stderr_chars,
            max_reflection_attempts=config.max_reflection_attempts,
            permission_policy=permission_policy_for_profile(
                config.permission_profile
            ),
            permission_callback=permission_callback,
            context_budget=context_budget,
            max_model_output_tokens=(
                model_capabilities.max_output_tokens
                or model_capabilities.default_response_reserve_tokens
            ),
            event_callback=session_recorder.callback,
            audit_callback=hosted_audit,
            redaction_callback=redaction_callback,
            redaction_fail_closed=redaction_fail_closed,
            final_answer_streaming=final_answer_streaming,
            output_policy=output_policy,
            async_output_policy=async_output_policy,
            output_policy_failure_mode=output_policy_failure_mode,
            async_transcript_resolver=async_transcript_resolver,
            transcript_timeout_seconds=transcript_timeout_seconds,
            pinned_skill_names=skill_resolution.pinned_skill_names,
            system_prompt=system_prompt or BASE_SYSTEM_PROMPT,
            mcp_servers=active_mcp_servers,
            mcp_bridge_tool_names=mcp_bridge_tool_names,
            owned_resources=owned_resources,
            default_tool_context=(
                ToolExecutionContext(deps=deps)
                if deps is not None
                else None
            ),
            runtime_metadata=effective_runtime_metadata,
            tool_context_lifecycle=execution_lifecycle,
            profile_id=effective_profile_id,
            usage_accounting=None,
            skill_lifecycle_store=(
                resolved.skills.lifecycle_store if skills_enabled else None
            ),
            skill_lifecycle=(
                resolved.skills.lifecycle if skills_enabled else None
            ),
            learning_proposals=(
                resolved.skills.learning_proposals if skills_enabled else None
            ),
            learning_reviewer=(
                resolved.skills.learning_reviewer if skills_enabled else None
            ),
            plugin_registry=cast(
                LocalPluginRegistry,
                plugin_registry if plugin_registry is not None else resolved.plugins,
            ),
            plugin_audit_report=plugin_audit_report,
            goal_execution=goal_execution,
            content_store=(
                cast(ContentStore, resolved.content)
                if resolved.is_enabled("content")
                else None
            ),
            media_processors=(
                cast(MediaProcessorRegistry, resolved.media)
                if resolved.is_enabled("media")
                else None
            ),
            execution_scope=execution_scope,
            tool_policy_hooks=cast(ToolPolicyHooks, resolved.tool_policy),
            async_tool_catalog_resolver=async_tool_catalog_resolver,
            tool_catalog_timeout_seconds=tool_catalog_timeout_seconds,
            close_trace_logger=False,
            restore_plan_context=False,
        )
        agent.memory_store = cast(SQLiteMemoryStore | None, memory_store)
        agent.memory_policy = cast(MemoryPolicy | None, memory_policy)
        agent.async_memory_store = memory_store
        agent.async_memory_policy = memory_policy
        agent.async_skill_registry = skill_registry
        agent.async_usage_accounting = resolved.usage
        agent.async_artifact_store = (
            resolved.artifacts if resolved.is_enabled("artifacts") else None
        )
        agent.async_content_store = (
            resolved.content if resolved.is_enabled("content") else None
        )
        agent.async_media_processors = (
            resolved.media if resolved.is_enabled("media") else None
        )
        agent.async_flushables = (
            trace_logger,
            session_recorder,
            audit_sink,
            event_sink,
        )
        agent.session_store = session_store
        agent.session_recorder = session_recorder
        agent.session_search_service = session_search_service
        agent.run_store = resolved.runs
        agent.approval_store = resolved.approvals
        agent.public_event_sink = event_sink
        agent.hosted_service_manifest = resolved.manifest
        agent._refresh_action_runtime()
        await agent.restore_plan_turn_context_async()
        await agent._flush_async_services()
        return agent, resolved
    except BaseException as exc:
        cleanup_resources: list[object] = []
        if client_is_owned and client is not None:
            cleanup_resources.append(client)
        cleanup_resources.append(resolved)
        for resource in cleanup_resources:
            try:
                if isinstance(resource, ResolvedRuntimeServices):
                    await resource.aclose_owned()
                else:
                    await close_async_resource(resource)
            except BaseException as cleanup_error:
                exc.add_note(
                    "async hosted construction cleanup also failed with "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise
