"""Shared construction helpers for public SDK agents."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

from chulk.capabilities import Capabilities, MemoryMode
from chulk._sdk.config import AgentConfig, AgentPreset, coerce_config
from chulk._sdk.events import EventCallback
from chulk._sdk.handles import AgentHandle
from chulk.config import Config
from chulk.llm import LLMClient
from chulk.execution import ExecutionBackend
from chulk.goals import GoalExecutionContext
from chulk.hosting import (
    AsyncTranscriptResolver,
    ExecutionScope,
    RuntimeServices,
    TranscriptResolver,
)
from chulk.mcp import MCPServerConfig
from chulk.media import ContentStore, MediaProcessorRegistry
from chulk.plugins import (
    LocalPluginRegistry,
)
from chulk.skills import (
    LearningReviewPolicy,
    LearningReviewQuota,
)
from chulk.runtime import create_agent as create_runtime_agent
from chulk.tools import ShellExecutionPolicy
from chulk.tools.permissions import PermissionDecision, PermissionDecisionRecord, PermissionRequest
from chulk.usage import (
    RunBudget,
    UsageDimensions,
)
from chulk.streaming import (
    AsyncIncrementalOutputPolicy,
    FinalAnswerStreamingMode,
    IncrementalOutputPolicy,
    OutputPolicyFailureMode,
)


PermissionCallback = Callable[[PermissionRequest, PermissionDecisionRecord], PermissionDecision | bool]


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
    final_answer_streaming: FinalAnswerStreamingMode | str = FinalAnswerStreamingMode.VALIDATED,
    output_policy: IncrementalOutputPolicy | None = None,
    async_output_policy: AsyncIncrementalOutputPolicy | None = None,
    output_policy_failure_mode: OutputPolicyFailureMode | str = OutputPolicyFailureMode.CLOSED,
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
    transcript_resolver: TranscriptResolver | None = None,
    async_transcript_resolver: AsyncTranscriptResolver | None = None,
    transcript_timeout_seconds: float | None = None,
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
        final_answer_streaming=final_answer_streaming,
        output_policy=output_policy,
        async_output_policy=async_output_policy,
        output_policy_failure_mode=output_policy_failure_mode,
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
        transcript_resolver=transcript_resolver,
        async_transcript_resolver=async_transcript_resolver,
        transcript_timeout_seconds=transcript_timeout_seconds,
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
