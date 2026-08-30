"""Model client construction for runtime assembly."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chulk.config import Config
from chulk.llm import (
    LLMClient,
    LLMModelCapabilities,
)
from chulk.llm.capabilities import resolve_runtime_model_capabilities
from chulk.llm.factory import _bind_llm_client
from chulk.streaming import (
    FinalAnswerStreamingMode,
    OutputPolicyFailureMode,
)

if TYPE_CHECKING:
    from chulk.capabilities import Capabilities
    from chulk.core.events import AgentEvent
    from chulk.core.plan_execution import AsyncPlanStepVerifier, PlanStepVerifier
    from chulk.execution import ExecutionBackend
    from chulk.goals.runtime import GoalExecutionContext
    from chulk.hosting import (
        AsyncRuntimeServices,
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
    from chulk.media import ContentStore, MediaProcessorRegistry
    from chulk.plugins import LocalPluginRegistry
    from chulk.skills import LearningReviewPolicy, LearningReviewQuota
    from chulk.streaming import (
        AsyncIncrementalOutputPolicy,
        IncrementalOutputPolicy,
    )
    from chulk.tools import ShellExecutionPolicy
    from chulk.tools.permissions import (
        PermissionDecision,
        PermissionDecisionRecord,
        PermissionRequest,
    )
    from chulk.usage import RunBudget, UsageDimensions


@dataclass(frozen=True, slots=True)
class AgentAssemblyRequest:
    """Private normalized input for configured sync and hosted assembly."""

    config: Config
    llm_client_factory: Callable[[Config], LLMClient] | None = None
    conversation_id: str | None = None
    conversation_metadata: dict[str, object] | None = None
    llm_client: LLMClient | None = None
    tool_specs: Iterable[object] | None = None
    skill_specs: object | Iterable[object] | None = None
    system_prompt: str | None = None
    permission_callback: Callable[
        [PermissionRequest, PermissionDecisionRecord],
        PermissionDecision | bool,
    ] | None = None
    plan_step_verifier: PlanStepVerifier | None = None
    async_plan_step_verifier: AsyncPlanStepVerifier | None = None
    mcp_servers: Iterable[MCPServerConfig] | None = None
    event_sink: Callable[[AgentEvent], None] | None = None
    redaction_callback: Callable[[str, str, dict], str] | None = None
    redaction_fail_closed: bool = False
    final_answer_streaming: FinalAnswerStreamingMode | str = (
        FinalAnswerStreamingMode.VALIDATED
    )
    output_policy: IncrementalOutputPolicy | None = None
    async_output_policy: AsyncIncrementalOutputPolicy | None = None
    output_policy_failure_mode: OutputPolicyFailureMode | str = (
        OutputPolicyFailureMode.CLOSED
    )
    capabilities: Capabilities | None = None
    deps: object | None = None
    shell_execution_policy: ShellExecutionPolicy | None = None
    require_shell_containment: bool = False
    execution_backend: ExecutionBackend | None = None
    memory_namespace: str | None = None
    profile_id: str | None = None
    allowed_skill_names: Iterable[str] | None = None
    runtime_metadata: dict | None = None
    run_budget: RunBudget | None = None
    additional_run_budgets: Iterable[RunBudget] = ()
    usage_dimensions: UsageDimensions | None = None
    learning_review_policy: LearningReviewPolicy | None = None
    learning_review_quota: LearningReviewQuota | None = None
    automatic_learning_approval: bool = False
    plugin_registry: LocalPluginRegistry | None = None
    goal_execution: GoalExecutionContext | None = None
    content_store: ContentStore | None = None
    media_processors: MediaProcessorRegistry | None = None
    services: RuntimeServices | AsyncRuntimeServices | None = None
    execution_scope: ExecutionScope | None = None
    transcript_resolver: TranscriptResolver | None = None
    async_transcript_resolver: AsyncTranscriptResolver | None = None
    transcript_timeout_seconds: float | None = None
    tool_catalog_resolver: ToolCatalogResolver | None = None
    async_tool_catalog_resolver: AsyncToolCatalogResolver | None = None
    tool_catalog_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.llm_client is not None and self.llm_client_factory is not None:
            raise ValueError("Pass either llm_client or llm_client_factory, not both")


def default_llm_client_factory(config: Config) -> LLMClient:
    """Create the configured model client."""
    return _bind_llm_client(
        config,
        provider=config.llm_provider,
        model=config.model,
        local_context_window_tokens=config.local_context_window_tokens,
        timeout_seconds=config.llm_timeout_seconds,
        max_retries=config.llm_max_retries,
    )


def client_model_capabilities(
    client: LLMClient,
    config: Config,
) -> LLMModelCapabilities:
    """Return bound-client capabilities or resolve the configured fallback."""
    capabilities = getattr(client, "model_capabilities", None)
    if isinstance(capabilities, LLMModelCapabilities):
        return capabilities
    return resolve_runtime_model_capabilities(
        config.llm_provider,
        config.model,
        local_context_window_tokens=config.local_context_window_tokens,
    )
