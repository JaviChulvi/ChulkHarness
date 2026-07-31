"""Configured agent assembly behind the public runtime facade."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
import warnings
from typing import cast

from chulk._version import __version__
from chulk.capabilities import Capabilities
from chulk.config import Config
from chulk.core import Agent
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
    ExecutionScope,
    RuntimeServices,
    SessionRuntimeServices,
    SkillRuntimeServices,
)
from chulk.llm import LLMClient
from chulk.llm.lifecycle import close_resources
from chulk.mcp import MCPServerConfig
from chulk.media import ContentStore, LocalTextExtractor, MediaProcessorRegistry
from chulk.memory import ConversationMemory, MemoryPolicy, SQLiteMemoryStore
from chulk.plugins import LocalPluginRegistry
from chulk.sessions import (
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
from chulk.usage import (
    ModelUsageAccounting,
    RunBudget,
    SQLiteUsageStore,
    UsageDimensions,
)
from chulk._runtime.request import (
    client_model_capabilities,
    default_llm_client_factory,
)
from chulk._runtime.services import resolve_runtime_services
from chulk._runtime.sessions import (
    block_unresolved_tool_intent,
    create_agent_state,
    session_result_redactor,
    summary_source_ordinal,
)
from chulk._runtime.skills import (
    SkillSpecResolution,
    resolve_skill_specs,
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
    selected_plugin_registry = (
        resolved_services.plugins
        if resolved_services is not None
        else plugin_registry
        or LocalPluginRegistry(
            config.runtime_dir,
            profile_id=effective_profile_id,
        )
    )
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
    effective_conversation_metadata = dict(conversation_metadata or {})
    metadata_profile_id = effective_conversation_metadata.get("profile_id")
    if metadata_profile_id is not None and metadata_profile_id != effective_profile_id:
        raise ValueError(
            "conversation metadata profile_id does not match the runtime profile"
        )
    effective_conversation_metadata["profile_id"] = effective_profile_id
    if execution_scope is not None:
        effective_conversation_metadata["execution_scope"] = (
            execution_scope.to_dict()
        )
        effective_conversation_metadata["execution_scope_key"] = (
            execution_scope.key
        )
    memory_store = (
        resolved_services.memory
        if resolved_services is not None
        else SQLiteMemoryStore(
            config.store_path,
            namespace=memory_namespace,
        )
    )
    selected_capabilities = capabilities or Capabilities.full()
    memory_policy = MemoryPolicy(memory_store, selected_capabilities.memory)
    if resolved_services is not None:
        if not isinstance(resolved_services.sessions, SessionRuntimeServices):
            raise TypeError(
                "hosted sessions service must be SessionRuntimeServices"
            )
        if not isinstance(resolved_services.skills, SkillRuntimeServices):
            raise TypeError("hosted skills service must be SkillRuntimeServices")
        session_store = resolved_services.sessions.store
        session_search_service = resolved_services.sessions.search
        selected_content_store = resolved_services.content
        selected_media_processors = resolved_services.media
        skill_registry = resolved_services.skills.registry
        skill_lifecycle_store = resolved_services.skills.lifecycle_store
        skill_lifecycle = resolved_services.skills.lifecycle
        learning_proposals = resolved_services.skills.learning_proposals
        learning_reviewer = resolved_services.skills.learning_reviewer
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
            memory_store=memory_store,
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
    skill_registry.load_metadata()
    skill_resolution = resolve_skill_specs(skill_registry, skill_specs)
    if allowed_skill_names is not None:
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
    if conversation_id is not None:
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
            proposal_service=learning_proposals,
            lifecycle_store=skill_lifecycle_store,
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
            if resolved_services is not None
            else trace_logger.artifact_store
        ),
        bridge_tool_factory=bridge_tool_factory,
        mcp_bridge_required=mcp_bridge_required,
    )
    available_tool_names = {tool.name for tool in tool_registry.list_tools()}
    skill_capabilities = skill_capability_names(selected_capabilities)
    if available_tool_names:
        skill_capabilities.add("tools")
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
            plugin_registry=selected_plugin_registry,
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
    if learning_proposals is not None:
        learning_proposals.event_callback = agent._trace
    return agent
