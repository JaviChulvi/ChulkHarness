"""Runtime assembly for configured Chulk agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
import warnings
from typing import Protocol, cast

from chulk.capabilities import Capabilities
from chulk._version import __version__
from chulk.config import Config
from chulk.core import Agent, AgentState, TurnState
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
from chulk.llm import (
    LLMClient,
    LLMModelCapabilities,
    create_llm_client,
    provider_capabilities,
    provider_connection_from_config,
)
from chulk.llm.lifecycle import close_resources
from chulk.llm.capabilities import (
    client_requires_mcp_bridge,
    client_supports_hosted_mcp_tools,
    client_supports_native_tool_calling,
    resolve_runtime_model_capabilities,
)
from chulk.mcp import MCPServerConfig, create_mcp_bridge_tools
from chulk.media import ContentStore, LocalTextExtractor, MediaProcessorRegistry
from chulk.memory import ConversationMemory, MemoryPolicy, SQLiteMemoryStore
from chulk.plugins import LocalPluginRegistry
from chulk.redaction import redact_text
from chulk.sessions import (
    ConversationSummaryRecord,
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
    SkillAllowlistRef,
    SkillDirectoryRef,
    SkillLifecycleManager,
    SkillPinRef,
    SkillRef,
    SkillRegistry,
)
from chulk.tools import (
    ShellExecutionPolicy,
    Tool,
    ToolExecutionContext,
    ToolRegistry,
    create_default_tool_registry,
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


class LLMClientFactory(Protocol):
    """Factory used by tests and the CLI to inject an LLM client."""

    def __call__(self, config: Config) -> LLMClient:
        """Return an LLM client for the given runtime config."""


@dataclass(frozen=True)
class RuntimeToolContext:
    """Context required to bind project-scoped tool references."""

    project_root: Path
    shell_timeout_seconds: int
    max_tool_stdout_bytes: int
    max_tool_stderr_bytes: int
    shell_execution_policy: ShellExecutionPolicy | None = None
    require_shell_containment: bool = False
    memory_store: SQLiteMemoryStore | None = None
    session_search_service: SessionSearchService | None = None
    artifact_store: TraceArtifactStore | None = None
    deps: object | None = None


@dataclass(frozen=True)
class SkillSpecResolution:
    """Resolved SDK skill configuration for one agent runtime."""

    pinned_skill_names: list[str]
    warnings: list[dict[str, str]]


@dataclass(frozen=True)
class MCPRoute:
    """Effective MCP transport across the clients that may handle a request."""

    provider_path: str
    bridge_required: bool


def create_agent(
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
) -> Agent:
    """Create the configured Chulk agent runtime."""
    if llm_client is not None and llm_client_factory is not None:
        raise ValueError("Pass either llm_client or llm_client_factory, not both")
    hosted_state: AgentState | None = None
    resolved_services = None
    if services is not None:
        conflicts = [
            name
            for name, value in (
                ("execution_backend", execution_backend),
                ("plugin_registry", plugin_registry),
                ("content_store", content_store),
                ("media_processors", media_processors),
                ("memory_namespace", memory_namespace),
            )
            if value is not None
        ]
        if conflicts:
            raise ValueError(
                "hosted services cannot be combined with individual runtime "
                "injections: " + ", ".join(conflicts)
            )
        if tool_specs is None:
            raise ValueError(
                "hosted runtime requires an explicit tools collection"
            )
        if skill_specs is None:
            raise ValueError(
                "hosted runtime requires an explicit skills collection"
            )
        if execution_scope is None:
            raise ValueError("hosted runtime requires an ExecutionScope")
        requested_conversation_id = (
            conversation_id or execution_scope.conversation_id
        )
        if (
            conversation_id is not None
            and execution_scope.conversation_id not in {None, conversation_id}
        ):
            raise ValueError(
                "execution scope conversation_id does not match conversation_id"
            )
        if requested_conversation_id is None:
            hosted_state = AgentState()
            requested_conversation_id = hosted_state.conversation_id
        execution_scope = execution_scope.with_conversation(
            requested_conversation_id
        )
        conversation_id = (
            requested_conversation_id
            if hosted_state is None
            else None
        )
        resolved_services = services.resolve(execution_scope)
    elif execution_scope is not None:
        raise ValueError("execution_scope is only accepted with hosted services")

    if llm_client_factory is None:
        llm_client_factory = _default_llm_client_factory
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
        state = hosted_state or _create_agent_state(
            session_store,
            conversation_id,
            execution_scope=execution_scope,
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
        state = _create_agent_state(session_store, conversation_id)
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
            redactor=_session_result_redactor(
                redaction_callback,
                fail_closed=redaction_fail_closed,
            ),
            audit_callback=audit_session_read,
        )
    skill_registry.load_metadata()
    skill_resolution = _resolve_skill_specs(skill_registry, skill_specs)
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
            after_ordinal=_summary_source_ordinal(latest_summary),
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
                sorted(_skill_capability_names(selected_capabilities))
            ),
        )
    selection_result = getattr(client, "selection_result", None)
    effective_runtime_metadata = dict(runtime_metadata or {})
    if selection_result is not None and hasattr(selection_result, "to_dict"):
        effective_runtime_metadata["model_selection"] = selection_result.to_dict()
    model_capabilities = _client_model_capabilities(client, config)
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
    tool_registry, mcp_bridge_tool_names = _create_tool_registry(
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
    )
    available_tool_names = {tool.name for tool in tool_registry.list_tools()}
    skill_capabilities = _skill_capability_names(selected_capabilities)
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
                "provider_path": _mcp_provider_path(
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
                "bridge_required": _mcp_bridge_required(
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
        agent = Agent(
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


def _session_result_redactor(
    callback: Callable[[str, str, dict], str] | None,
    *,
    fail_closed: bool,
) -> Callable[[str], str]:
    """Compose baseline secret redaction with an optional host policy."""

    def redact(value: str) -> str:
        safe_value = redact_text(value)
        if callback is None:
            return safe_value
        try:
            custom_value = callback(
                "session_search_result",
                safe_value,
                {"source": "session_search"},
            )
        except Exception:
            return "[redaction failed]" if fail_closed else safe_value
        return redact_text(str(custom_value))

    return redact


def _summary_source_ordinal(summary: ConversationSummaryRecord | None) -> int:
    """Return the durable ordinal covered by a logical conversation summary."""
    if summary is None:
        return 0
    source_ordinal = summary.metadata.get("source_message_ordinal")
    if (
        isinstance(source_ordinal, int)
        and not isinstance(source_ordinal, bool)
        and source_ordinal >= 0
    ):
        return source_ordinal
    return summary.source_message_count


def _create_agent_state(
    session_store: SQLiteSessionStore,
    conversation_id: str | None,
    *,
    execution_scope: ExecutionScope | None = None,
) -> AgentState:
    """Create fresh state or rebuild state for an existing conversation."""
    if conversation_id is None:
        return AgentState()

    conversation = session_store.get_conversation(conversation_id)
    if execution_scope is not None:
        raw_scope = conversation.metadata.get("execution_scope")
        if not isinstance(raw_scope, dict):
            raise ValueError(
                "hosted conversation has no persisted execution scope"
            )
        execution_scope.assert_resumable(
            ExecutionScope.from_dict(raw_scope)
        )
    state = AgentState(conversation_id=conversation.id)
    state.turns = session_store.load_turns(conversation.id)
    if not state.turns:
        return state

    latest_turn = state.turns[-1]
    _reconcile_terminal_turn_message(
        session_store,
        conversation.id,
        conversation.status,
        latest_turn,
    )
    _reconcile_blocked_plan_turn(
        session_store,
        conversation.id,
        latest_turn,
    )
    if latest_turn.status == "in_progress":
        hosted_requests = session_store.load_uncheckpointed_hosted_mcp_requests(
            conversation.id,
            latest_turn.turn_id,
            checkpointed_request_count=latest_turn.model_request_count,
        )
        if hosted_requests:
            _block_uncertain_hosted_mcp_request(
                session_store,
                conversation.id,
                latest_turn,
                hosted_requests,
            )
        else:
            unreconciled_calls = [
                record.to_dict()
                for record in latest_turn.tool_calls
                if record.success is None or record.ended_at is None
            ]
            if not unreconciled_calls:
                unreconciled_calls = session_store.load_tool_calls_without_observations(
                    conversation.id,
                    latest_turn.turn_id,
                )
            if unreconciled_calls:
                _block_unresolved_tool_intent(
                    session_store,
                    conversation.id,
                    latest_turn,
                    unreconciled_calls,
                )
    state.current_turn_id = latest_turn.turn_id
    state.loaded_memory_ids = list(latest_turn.loaded_memory_ids)
    state.extracted_memory_ids = list(latest_turn.extracted_memory_ids)
    state.loaded_skill_names = list(latest_turn.loaded_skill_names)
    state.available_tool_names = list(latest_turn.available_tool_names)
    state.errors = [error for turn in state.turns for error in turn.errors]
    state.final_answer = latest_turn.final_answer
    if latest_turn.context_reports:
        state.last_context_report = latest_turn.context_reports[-1]
    if latest_turn.model_usage_totals:
        state.last_usage_report = latest_turn.model_usage_totals
    if (
        latest_turn.status == "waiting_for_approval"
        and latest_turn.active_plan is not None
        and not latest_turn.plan_approved
    ):
        state.active_plan = latest_turn.active_plan
        state.pending_plan_turn_id = latest_turn.turn_id
    elif latest_turn.can_continue_approved_plan():
        state.active_plan = latest_turn.active_plan
    return state


def _reconcile_terminal_turn_message(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    conversation_status: str,
    turn: TurnState,
) -> None:
    """Terminalize a legacy turn whose terminal message preceded its snapshot."""
    if turn.status not in {"in_progress", "waiting_for_approval"}:
        return
    terminal_message = session_store.load_terminal_turn_message(
        conversation_id,
        turn.turn_id,
    )
    if terminal_message is None:
        return
    content = terminal_message["content"]
    kind = terminal_message["kind"]
    if kind == "final":
        turn.complete(content)
    elif kind == "plan_rejected":
        turn.reject_plan(content)
    elif kind == "failed":
        plan_status = (
            turn.active_plan.status() if turn.active_plan is not None else None
        )
        if conversation_status == "cancelled":
            turn.cancel(content)
        elif conversation_status == "blocked" or plan_status == "blocked":
            turn.block(content)
        else:
            turn.fail(content)
    else:  # pragma: no cover - constrained by SQLiteSessionStore
        return
    session_store.save_turn_snapshot(conversation_id, turn.to_dict())


def _reconcile_blocked_plan_turn(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
) -> None:
    """Terminalize a checkpoint that already contains a blocked plan step."""
    plan = turn.active_plan
    if turn.status != "in_progress" or plan is None or plan.status() != "blocked":
        return
    blocked_step = next(
        (step for step in plan.steps if step.status == "blocked"),
        None,
    )
    if blocked_step is None:  # pragma: no cover - Plan.status enforces this
        return
    reason = blocked_step.blocked_reason or "Step blocked."
    message = f"Plan step blocked: {blocked_step.title}. {reason}"
    turn.block(message)
    _save_recovery_terminal(
        session_store,
        conversation_id,
        turn,
        message,
        message_key_suffix="blocked_plan_checkpoint",
        metadata={"recovery": "blocked_plan_checkpoint"},
    )


def _block_uncertain_hosted_mcp_request(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
    hosted_requests: list[dict[str, object]],
) -> None:
    """Fail closed when a hosted provider request lacks a durable checkpoint."""
    latest = hosted_requests[-1]
    raw_request_index = latest.get("request_index")
    request_index = (
        raw_request_index
        if isinstance(raw_request_index, int)
        and not isinstance(raw_request_index, bool)
        else 0
    )
    reason = (
        "Turn execution stopped after restart because hosted MCP request "
        f"{request_index} may have executed a remote operation without a durable "
        "checkpoint. Chulk will not replay it automatically; inspect remote state "
        "before retrying."
    )
    plan = turn.active_plan
    active_step = plan.active_step() if plan is not None else None
    if active_step is not None:
        active_step.block(reason)
    turn.block(reason)
    _save_recovery_terminal(
        session_store,
        conversation_id,
        turn,
        reason,
        message_key_suffix="uncertain_hosted_mcp",
        metadata={
            "recovery": "uncertain_hosted_mcp",
            "request_index": request_index,
        },
    )


def _block_unresolved_tool_intent(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
    unresolved_calls: list[dict[str, object]],
) -> None:
    """Fail closed when execution stopped after intent but before a result."""
    latest = unresolved_calls[-1]
    tool_name = str(latest.get("tool_name") or "tool")
    raw_iteration = latest.get("iteration")
    iteration = (
        raw_iteration
        if isinstance(raw_iteration, int) and not isinstance(raw_iteration, bool)
        else 0
    )
    reason = (
        "Turn execution stopped after restart because "
        f"tool call {tool_name} (iteration {iteration}) has no matching persisted observation. "
        "Chulk will not replay it automatically; inspect external state before retrying."
    )
    plan = turn.active_plan
    active_step = plan.active_step() if plan is not None else None
    if active_step is not None:
        active_step.block(reason)
    turn.block(reason)
    _save_recovery_terminal(
        session_store,
        conversation_id,
        turn,
        reason,
        message_key_suffix="unresolved_tool_intent",
        metadata={"recovery": "unresolved_tool_intent"},
    )


def _save_recovery_terminal(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
    content: str,
    *,
    message_key_suffix: str,
    metadata: dict[str, object],
) -> None:
    saved = session_store.save_terminal_turn_bundle(
        conversation_id,
        turn_id=turn.turn_id,
        content=content,
        message_key=f"{turn.turn_id}:assistant:{message_key_suffix}",
        turn=turn.to_dict(),
        metadata=metadata,
    )
    if not saved:  # pragma: no cover - TurnState guarantees a valid payload
        raise RuntimeError("Failed to persist terminal recovery state")


def _default_llm_client_factory(config: Config) -> LLMClient:
    return create_llm_client(
        provider=config.llm_provider,
        model=config.model,
        connection=provider_connection_from_config(config.llm_provider, config),
        local_context_window_tokens=config.local_context_window_tokens,
        timeout_seconds=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )


def _client_model_capabilities(
    client: LLMClient, config: Config
) -> LLMModelCapabilities:
    capabilities = getattr(client, "model_capabilities", None)
    if isinstance(capabilities, LLMModelCapabilities):
        return capabilities
    return resolve_runtime_model_capabilities(
        config.llm_provider,
        config.model,
        local_context_window_tokens=config.local_context_window_tokens,
    )


def _create_tool_registry(
    config: Config,
    memory_store: SQLiteMemoryStore,
    tool_specs: Iterable[object] | None,
    mcp_servers: Iterable[MCPServerConfig],
    *,
    llm_client: LLMClient | None = None,
    capabilities: Capabilities,
    memory_policy: MemoryPolicy,
    session_search_service: SessionSearchService,
    deps: object | None,
    shell_execution_policy: ShellExecutionPolicy | None,
    require_shell_containment: bool,
    artifact_store: TraceArtifactStore,
) -> tuple[ToolRegistry, list[str]]:
    if tool_specs is None:
        registry = create_default_tool_registry(
            config.project_root,
            config.shell_timeout_seconds,
            memory_store=memory_store,
            capabilities=capabilities,
            memory_policy=memory_policy,
            session_search_service=session_search_service,
            max_tool_stdout_bytes=config.max_tool_stdout_chars,
            max_tool_stderr_bytes=config.max_tool_stderr_chars,
            shell_execution_policy=shell_execution_policy,
            require_shell_containment=require_shell_containment,
        )
        return _register_mcp_bridge_tools(
            config,
            registry,
            mcp_servers,
            llm_client=llm_client,
        )

    context = RuntimeToolContext(
        project_root=config.project_root,
        shell_timeout_seconds=config.shell_timeout_seconds,
        max_tool_stdout_bytes=config.max_tool_stdout_chars,
        max_tool_stderr_bytes=config.max_tool_stderr_chars,
        shell_execution_policy=shell_execution_policy,
        require_shell_containment=require_shell_containment,
        memory_store=memory_store,
        session_search_service=session_search_service,
        artifact_store=artifact_store,
        deps=deps,
    )
    registry = ToolRegistry()
    for spec in tool_specs:
        tool = _resolve_tool_spec(spec, context)
        registry.register(tool)
    return _register_mcp_bridge_tools(
        config,
        registry,
        mcp_servers,
        llm_client=llm_client,
    )


def _register_mcp_bridge_tools(
    config: Config,
    registry: ToolRegistry,
    mcp_servers: Iterable[MCPServerConfig],
    *,
    llm_client: LLMClient | None = None,
) -> tuple[ToolRegistry, list[str]]:
    servers = tuple(mcp_servers)
    if not servers or not _mcp_bridge_required(
        config,
        servers,
        llm_client=llm_client,
    ):
        return registry, []
    bridge_tools = create_mcp_bridge_tools(servers)
    bridge_tool_names: list[str] = []
    for tool in bridge_tools:
        registry.register(tool)
        bridge_tool_names.append(tool.name)
    return registry, bridge_tool_names


def _mcp_bridge_required(
    config: Config,
    mcp_servers: Iterable[object],
    *,
    llm_client: LLMClient | None = None,
) -> bool:
    return resolve_mcp_route(
        config,
        mcp_servers,
        llm_client=llm_client,
    ).bridge_required


def _mcp_provider_path(
    config: Config,
    mcp_servers: Iterable[object],
    *,
    llm_client: LLMClient | None = None,
) -> str:
    return resolve_mcp_route(
        config,
        mcp_servers,
        llm_client=llm_client,
    ).provider_path


def resolve_mcp_route(
    config: Config,
    mcp_servers: Iterable[object],
    *,
    llm_client: LLMClient | None = None,
) -> MCPRoute:
    """Resolve one MCP route from the effective bound client path when available."""
    if not tuple(mcp_servers):
        return MCPRoute(provider_path="none", bridge_required=False)

    if llm_client is not None:
        native_protocol = client_supports_native_tool_calling(llm_client)
        has_hosted = client_supports_hosted_mcp_tools(llm_client)
        has_bridge = client_requires_mcp_bridge(llm_client)
    else:
        provider_names = [
            config.llm_provider,
            *(provider.provider for provider in config.llm_fallback_providers),
        ]
        native_support = [
            _supports_native_tool_calling(provider) for provider in provider_names
        ]
        hosted_support = [_supports_hosted_mcp(provider) for provider in provider_names]
        native_protocol = all(native_support)
        has_hosted = any(hosted_support)
        has_bridge = any(not item for item in hosted_support)

    if not native_protocol:
        return MCPRoute(provider_path="bridge", bridge_required=True)

    if has_hosted and has_bridge:
        route_path = "hosted+bridge"
    else:
        route_path = "hosted" if has_hosted else "bridge"
    return MCPRoute(provider_path=route_path, bridge_required=has_bridge)


def _supports_hosted_mcp(provider: str) -> bool:
    return provider_capabilities(provider).supports_hosted_mcp_tools


def _supports_native_tool_calling(provider: str) -> bool:
    return provider_capabilities(provider).supports_native_tool_calling


def _skill_capability_names(capabilities: Capabilities) -> set[str]:
    """Flatten runtime capabilities into manifest-facing capability names."""
    values = capabilities.to_dict()
    names = {
        name
        for name in ("shell", "network", "external_services", "utilities")
        if values[name] is True
    }
    file_access = str(values["files"])
    if file_access in {"read", "write"}:
        names.update({"files", "files:read"})
    if file_access == "write":
        names.add("files:write")
    memory_mode = str(values["memory"])
    if memory_mode != "off":
        names.update({"memory", "memory:read"})
    if memory_mode in {"manual", "automatic"}:
        names.add("memory:write")
    if memory_mode == "automatic":
        names.add("memory:automatic")
    return names


def _resolve_tool_spec(spec: object, context: RuntimeToolContext) -> Tool:
    if isinstance(spec, Tool):
        return spec
    if hasattr(spec, "to_tool"):
        return spec.to_tool(context)  # type: ignore[no-any-return, attr-defined]
    raise TypeError(f"Unsupported tool spec: {spec!r}")


def _resolve_skill_specs(
    registry: SkillRegistry, skill_specs: object | Iterable[object] | None
) -> SkillSpecResolution:
    specs = _coerce_skill_specs(skill_specs)
    if specs is None:
        return SkillSpecResolution(pinned_skill_names=[], warnings=[])
    if not specs:
        registry.clear()
        return SkillSpecResolution(pinned_skill_names=[], warnings=[])

    allowlist_requests: list[str] = []
    pin_requests: list[str] = []
    warning_payloads: list[dict[str, str]] = []
    has_allowlist = False

    for spec in specs:
        if isinstance(spec, SkillAllowlistRef):
            has_allowlist = True
            allowlist_requests.extend(spec.names)
            continue
        if isinstance(spec, SkillPinRef):
            pin_requests.extend(spec.names)
            continue
        if isinstance(spec, SkillDirectoryRef):
            spec.register(registry)
            continue
        if isinstance(spec, SkillRef):
            if spec.skill_path is not None:
                skill = registry.register_path(spec.skill_path)
                pin_requests.append(skill.name)
                continue
            if spec.name is not None:
                pin_requests.append(spec.name)
                continue
            raise ValueError("SkillRef must include name or skill_path")
        if hasattr(spec, "register"):
            pinned_name = spec.register(registry)  # type: ignore[attr-defined]
            if pinned_name:
                pin_requests.append(str(pinned_name))
            continue
        if isinstance(spec, str):
            pin_requests.append(spec)
            continue
        raise TypeError(f"Unsupported skill spec: {spec!r}")

    allowlisted_names = _resolve_existing_skill_names(
        registry,
        allowlist_requests,
        kind="allowlist",
        warning_payloads=warning_payloads,
    )
    pinned_skill_names = _resolve_existing_skill_names(
        registry,
        pin_requests,
        kind="pin",
        warning_payloads=warning_payloads,
    )

    if has_allowlist:
        registry.restrict_to([*allowlisted_names, *pinned_skill_names])

    return SkillSpecResolution(
        pinned_skill_names=pinned_skill_names, warnings=warning_payloads
    )


def _coerce_skill_specs(
    skill_specs: object | Iterable[object] | None,
) -> list[object] | None:
    if skill_specs is None:
        return None
    if isinstance(
        skill_specs, (str, SkillAllowlistRef, SkillDirectoryRef, SkillPinRef, SkillRef)
    ):
        return [skill_specs]
    try:
        return list(cast(Iterable[object], skill_specs))
    except TypeError:
        return [skill_specs]


def _resolve_existing_skill_names(
    registry: SkillRegistry,
    names: Iterable[str],
    *,
    kind: str,
    warning_payloads: list[dict[str, str]],
) -> list[str]:
    resolved_names: list[str] = []
    for requested_name in names:
        skill = registry.get_skill(requested_name)
        if skill is None:
            _append_missing_skill_warning(kind, requested_name, warning_payloads)
            continue
        if skill.name not in resolved_names:
            resolved_names.append(skill.name)
    return resolved_names


def _append_missing_skill_warning(
    kind: str, requested_name: str, warnings_list: list[dict[str, str]]
) -> None:
    if any(
        payload["kind"] == kind and payload["name"] == requested_name
        for payload in warnings_list
    ):
        return
    warnings_list.append(
        {
            "kind": kind,
            "name": requested_name,
            "message": f"Skill '{requested_name}' requested by {kind} configuration is not registered; skipping.",
        }
    )
