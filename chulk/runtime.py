"""Runtime assembly for configured Chulk agents."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Protocol, cast

from chulk.capabilities import Capabilities
from chulk.config import Config
from chulk.core import Agent, AgentState, TurnState
from chulk.core.context import ContextBudget
from chulk.core.events import AgentEvent, TraceEvent
from chulk.core.prompts import BASE_SYSTEM_PROMPT
from chulk.llm import (
    LLMClient,
    LLMModelCapabilities,
    create_llm_client,
    provider_capabilities,
    provider_connection_from_config,
)
from chulk.llm.capabilities import (
    client_requires_mcp_bridge,
    client_supports_hosted_mcp_tools,
    client_supports_native_tool_calling,
    resolve_runtime_model_capabilities,
)
from chulk.mcp import MCPServerConfig, create_mcp_bridge_tools
from chulk.memory import ConversationMemory, MemoryPolicy, SQLiteMemoryStore
from chulk.sessions import ConversationSummaryRecord, SQLiteSessionStore, SessionRecorder
from chulk.skills import SkillAllowlistRef, SkillDirectoryRef, SkillPinRef, SkillRef, SkillRegistry
from chulk.tools import ShellExecutionPolicy, Tool, ToolExecutionContext, ToolRegistry, create_default_tool_registry
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    permission_policy_for_profile,
)
from chulk.tracing import JSONLTraceLogger


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
) -> Agent:
    """Create the configured Chulk agent runtime."""
    if llm_client is not None and llm_client_factory is not None:
        raise ValueError("Pass either llm_client or llm_client_factory, not both")

    if llm_client_factory is None:
        llm_client_factory = _default_llm_client_factory
    memory_store = SQLiteMemoryStore(config.store_path)
    selected_capabilities = capabilities or Capabilities.full()
    memory_policy = MemoryPolicy(memory_store, selected_capabilities.memory)
    session_store = SQLiteSessionStore(config.store_path)
    skill_registry = SkillRegistry(
        config.skills_dir,
        skills_dirs=config.skills_dirs,
        max_skills=config.max_skills_per_turn,
        max_content_chars=config.max_skill_content_chars,
    )
    state = _create_agent_state(session_store, conversation_id)
    trace_logger = JSONLTraceLogger(
        config.traces_dir,
        state.conversation_id,
        defer_until_event=TraceEvent.TURN_STARTED if conversation_id is None else None,
    )
    skill_registry.load_metadata()
    skill_resolution = _resolve_skill_specs(skill_registry, skill_specs)
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
            conversation_summary=latest_summary.content if latest_summary is not None else None,
            summary_message_count=latest_summary.source_message_count if latest_summary is not None else 0,
        )
        state.messages = conversation_memory.recent()
        state.conversation_summary = conversation_memory.conversation_summary
    session_recorder = SessionRecorder(
        session_store,
        state.conversation_id,
        provider=config.llm_provider,
        model=config.model,
        trace_path=trace_logger.path,
        lazy=conversation_id is None,
    )
    client_is_owned = llm_client is None
    client = llm_client if llm_client is not None else llm_client_factory(config)
    if hasattr(client, "bind_config"):
        client = client.bind_config(config)  # type: ignore[assignment, attr-defined]
    model_capabilities = _client_model_capabilities(client, config)
    context_budget = ContextBudget(
        max_prompt_tokens=model_capabilities.context_window_tokens,
        response_reserve_tokens=model_capabilities.default_response_reserve_tokens,
        max_input_tokens=model_capabilities.max_input_tokens,
    )
    configured_mcp_servers = tuple(mcp_servers) if mcp_servers is not None else config.mcp_servers
    active_mcp_servers = configured_mcp_servers if selected_capabilities.external_services else ()
    tool_registry, mcp_bridge_tool_names = _create_tool_registry(
        config,
        memory_store,
        tool_specs,
        active_mcp_servers,
        llm_client=client,
        capabilities=selected_capabilities,
        memory_policy=memory_policy,
        deps=deps,
        shell_execution_policy=shell_execution_policy,
        require_shell_containment=require_shell_containment,
    )
    if active_mcp_servers:
        trace_logger.log(
            "mcp_config_loaded",
            {
                "config_path": str(config.mcp_config_path),
                "servers": [
                    server.to_dict() if hasattr(server, "to_dict") else {"server": str(server)}
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
    owned_resources = [client] if client_is_owned else []
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
            event_callback=session_recorder.callback,
            event_sink=event_sink,
            redaction_callback=redaction_callback,
            redaction_fail_closed=redaction_fail_closed,
            pinned_skill_names=skill_resolution.pinned_skill_names,
            system_prompt=system_prompt or BASE_SYSTEM_PROMPT,
            mcp_servers=active_mcp_servers,
            mcp_bridge_tool_names=mcp_bridge_tool_names,
            owned_resources=owned_resources,
            default_tool_context=ToolExecutionContext(deps=deps) if deps is not None else None,
        )
    except Exception:
        for resource in reversed(owned_resources):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        raise
    agent.session_store = session_store
    agent.session_recorder = session_recorder
    return agent


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


def _create_agent_state(session_store: SQLiteSessionStore, conversation_id: str | None) -> AgentState:
    """Create fresh state or rebuild state for an existing conversation."""
    if conversation_id is None:
        return AgentState()

    conversation = session_store.get_conversation(conversation_id)
    state = AgentState(conversation_id=conversation.id)
    state.turns = session_store.load_turns(conversation.id)
    if not state.turns:
        return state

    latest_turn = state.turns[-1]
    if latest_turn.status == "in_progress":
        unresolved_calls = [
            record.to_dict()
            for record in latest_turn.tool_calls
            if record.success is None or record.ended_at is None
        ]
        if not unresolved_calls:
            unresolved_calls = session_store.load_unresolved_tool_calls(
                conversation.id,
                latest_turn.turn_id,
            )
        if unresolved_calls:
            _block_unresolved_tool_intent(
                session_store,
                conversation.id,
                latest_turn,
                unresolved_calls,
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
        f"tool call {tool_name} (iteration {iteration}) started without a recorded result. "
        "Chulk will not replay it automatically; inspect external state before retrying."
    )
    plan = turn.active_plan
    active_step = plan.active_step() if plan is not None else None
    if active_step is not None:
        active_step.block(reason)
    turn.block(reason)
    session_store.save_turn_snapshot(conversation_id, turn.to_dict())
    session_store.save_message(
        conversation_id,
        turn_id=turn.turn_id,
        role="assistant",
        content=reason,
        message_key=f"{turn.turn_id}:assistant:unresolved_tool_intent",
        metadata={"recovery": "unresolved_tool_intent"},
    )


def _default_llm_client_factory(config: Config) -> LLMClient:
    return create_llm_client(
        provider=config.llm_provider,
        model=config.model,
        connection=provider_connection_from_config(config.llm_provider, config),
        local_context_window_tokens=config.local_context_window_tokens,
        timeout_seconds=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )


def _client_model_capabilities(client: LLMClient, config: Config) -> LLMModelCapabilities:
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
    deps: object | None,
    shell_execution_policy: ShellExecutionPolicy | None,
    require_shell_containment: bool,
) -> tuple[ToolRegistry, list[str]]:
    if tool_specs is None:
        registry = create_default_tool_registry(
            config.project_root,
            config.shell_timeout_seconds,
            memory_store=memory_store,
            capabilities=capabilities,
            memory_policy=memory_policy,
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
        native_support = [_supports_native_tool_calling(provider) for provider in provider_names]
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


def _resolve_tool_spec(spec: object, context: RuntimeToolContext) -> Tool:
    if isinstance(spec, Tool):
        return spec
    if hasattr(spec, "to_tool"):
        return spec.to_tool(context)  # type: ignore[no-any-return, attr-defined]
    raise TypeError(f"Unsupported tool spec: {spec!r}")


def _resolve_skill_specs(registry: SkillRegistry, skill_specs: object | Iterable[object] | None) -> SkillSpecResolution:
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

    return SkillSpecResolution(pinned_skill_names=pinned_skill_names, warnings=warning_payloads)


def _coerce_skill_specs(skill_specs: object | Iterable[object] | None) -> list[object] | None:
    if skill_specs is None:
        return None
    if isinstance(skill_specs, (str, SkillAllowlistRef, SkillDirectoryRef, SkillPinRef, SkillRef)):
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


def _append_missing_skill_warning(kind: str, requested_name: str, warnings_list: list[dict[str, str]]) -> None:
    if any(payload["kind"] == kind and payload["name"] == requested_name for payload in warnings_list):
        return
    warnings_list.append(
        {
            "kind": kind,
            "name": requested_name,
            "message": f"Skill '{requested_name}' requested by {kind} configuration is not registered; skipping.",
        }
    )
