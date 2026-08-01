"""Compatibility entrypoint for configured Chulk agent runtimes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from chulk._runtime.assembly import (
    assemble_agent,
    assemble_async_hosted_agent,
)
from chulk._runtime.sessions import block_unresolved_tool_intent
from chulk._runtime.skills import SkillSpecResolution as SkillSpecResolution
from chulk._runtime.tools import RuntimeToolContext as RuntimeToolContext
from chulk._version import __version__ as __version__
from chulk.capabilities import Capabilities
from chulk.config import Config
from chulk.core import Agent, TurnState
from chulk.core.events import AgentEvent
from chulk.execution import ExecutionBackend
from chulk.goals.runtime import GoalExecutionContext
from chulk.hosting import AsyncRuntimeServices, ExecutionScope, RuntimeServices
from chulk.hosting.services import ResolvedRuntimeServices
from chulk.llm import LLMClient, provider_capabilities
from chulk.llm.capabilities import (
    client_requires_mcp_bridge,
    client_supports_hosted_mcp_tools,
    client_supports_native_tool_calling,
)
from chulk.mcp import MCPServerConfig, create_mcp_bridge_tools
from chulk.media import ContentStore, MediaProcessorRegistry
from chulk.plugins import LocalPluginRegistry
from chulk.sessions import SQLiteSessionStore
from chulk.skills import LearningReviewPolicy, LearningReviewQuota
from chulk.tools import ShellExecutionPolicy
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
)
from chulk.usage import RunBudget, UsageDimensions


class LLMClientFactory(Protocol):
    """Factory used by tests and the CLI to inject an LLM client."""

    def __call__(self, config: Config) -> LLMClient:
        """Return an LLM client for the given runtime config."""


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
    return assemble_agent(
        config,
        llm_client_factory,
        conversation_id=conversation_id,
        conversation_metadata=conversation_metadata,
        llm_client=llm_client,
        tool_specs=tool_specs,
        skill_specs=skill_specs,
        system_prompt=system_prompt,
        permission_callback=permission_callback,
        mcp_servers=mcp_servers,
        event_sink=event_sink,
        redaction_callback=redaction_callback,
        redaction_fail_closed=redaction_fail_closed,
        capabilities=capabilities,
        deps=deps,
        shell_execution_policy=shell_execution_policy,
        require_shell_containment=require_shell_containment,
        execution_backend=execution_backend,
        memory_namespace=memory_namespace,
        profile_id=profile_id,
        allowed_skill_names=allowed_skill_names,
        runtime_metadata=runtime_metadata,
        run_budget=run_budget,
        additional_run_budgets=additional_run_budgets,
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
        agent_factory=Agent,
        bridge_tool_factory=create_mcp_bridge_tools,
        mcp_bridge_required=_mcp_bridge_required,
        mcp_provider_path=_mcp_provider_path,
        unresolved_tool_handler=_block_unresolved_tool_intent,
    )


async def create_async_hosted_agent(
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
    capabilities: Capabilities | None = None,
    deps: object | None = None,
    shell_execution_policy: ShellExecutionPolicy | None = None,
    require_shell_containment: bool = False,
    run_budget: RunBudget | None = None,
    usage_dimensions: UsageDimensions | None = None,
    goal_execution: GoalExecutionContext | None = None,
    profile_id: str | None = None,
) -> tuple[Agent, ResolvedRuntimeServices]:
    """Create a hosted agent through native async service contracts."""
    return await assemble_async_hosted_agent(
        config,
        services=services,
        execution_scope=execution_scope,
        conversation_id=conversation_id,
        conversation_metadata=conversation_metadata,
        runtime_metadata=runtime_metadata,
        llm_client=llm_client,
        tool_specs=tool_specs,
        skill_specs=skill_specs,
        system_prompt=system_prompt,
        permission_callback=permission_callback,
        mcp_servers=mcp_servers,
        redaction_callback=redaction_callback,
        redaction_fail_closed=redaction_fail_closed,
        capabilities=capabilities,
        deps=deps,
        shell_execution_policy=shell_execution_policy,
        require_shell_containment=require_shell_containment,
        run_budget=run_budget,
        usage_dimensions=usage_dimensions,
        goal_execution=goal_execution,
        profile_id=profile_id,
        agent_factory=Agent,
        bridge_tool_factory=create_mcp_bridge_tools,
        mcp_bridge_required=_mcp_bridge_required,
    )


def _block_unresolved_tool_intent(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
    unresolved_calls: list[dict[str, object]],
) -> None:
    """Preserve the runtime recovery seam at its historical import path."""
    block_unresolved_tool_intent(
        session_store,
        conversation_id,
        turn,
        unresolved_calls,
    )


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
