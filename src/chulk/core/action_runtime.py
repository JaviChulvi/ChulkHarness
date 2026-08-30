"""Narrow service port consumed by action-loop transport drivers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from chulk.core.model_transport import ModelTransport
from chulk.core.tool_execution import ToolExecutor
from chulk.core.turn_effects import TurnEffects


class ActionLoopPort(Protocol):
    """The complete, explicit dependency surface of the action loop."""

    model: ModelTransport
    tools: ToolExecutor
    effects: TurnEffects
    async_flush: Callable[[], Awaitable[None]] | None

    def is_cancelled(self) -> bool: ...


@dataclass
class ActionLoopRuntime:
    """Concrete runtime assembled by Agent from focused services."""

    model: ModelTransport
    tools: ToolExecutor
    effects: TurnEffects
    async_flush: Callable[[], Awaitable[None]] | None = None
    cancelled: Callable[[], bool] = field(default_factory=lambda: _never_cancelled)

    def is_cancelled(self) -> bool:
        return self.cancelled()


@dataclass(slots=True)
class AgentRuntimeComponents:
    """Single internal input containing every dependency resolved by assembly."""

    llm_client: Any
    state: Any = None
    memory: Any = None
    memory_store: Any = None
    memory_policy: Any = None
    skill_registry: Any = None
    tool_registry: Any = None
    trace_logger: Any = None
    system_prompt: Any = None
    max_tool_calls_per_turn: int = 5
    max_json_repair_attempts: int = 2
    max_skills_per_turn: int = 3
    max_skill_content_chars: int = 4000
    trace_max_prompt_chars: int = 50000
    max_observation_chars: int = 12000
    max_tool_stdout_chars: int = 8000
    max_tool_stderr_chars: int = 4000
    max_reflection_attempts: int = 0
    permission_policy: Any = None
    permission_callback: Any = None
    plan_step_verifier: Any = None
    async_plan_step_verifier: Any = None
    context_budget: Any = None
    max_model_output_tokens: int | None = None
    stream_idle_timeout_seconds: float | None = 60.0
    event_callback: Any = None
    event_sink: Any = None
    audit_callback: Any = None
    public_event_sink: Any = None
    event_dispatcher: Any = None
    redaction_callback: Any = None
    redaction_fail_closed: bool = False
    final_answer_streaming: Any = "validated_final_answer"
    output_policy: Any = None
    async_output_policy: Any = None
    output_policy_failure_mode: Any = "fail_closed"
    pinned_skill_names: Any = None
    mcp_servers: Any = None
    mcp_bridge_tool_names: Any = None
    owned_resources: Any = None
    default_tool_context: Any = None
    runtime_metadata: Any = None
    tool_context_lifecycle: Any = None
    profile_id: str = "default"
    usage_accounting: Any = None
    skill_lifecycle_store: Any = None
    skill_lifecycle: Any = None
    learning_proposals: Any = None
    learning_reviewer: Any = None
    plugin_registry: Any = None
    plugin_audit_report: Any = None
    goal_execution: Any = None
    content_store: Any = None
    media_processors: Any = None
    execution_scope: Any = None
    tool_policy_hooks: Any = None
    transcript_resolver: Any = None
    async_transcript_resolver: Any = None
    transcript_timeout_seconds: float | None = None
    tool_catalog_resolver: Any = None
    async_tool_catalog_resolver: Any = None
    tool_catalog_timeout_seconds: float | None = None
    close_trace_logger: bool = True
    restore_plan_context: bool = True
    async_memory_store: Any = None
    async_memory_policy: Any = None
    async_skill_registry: Any = None
    async_usage_accounting: Any = None
    async_artifact_store: Any = None
    async_content_store: Any = None
    async_media_processors: Any = None
    async_flushables: tuple[object, ...] = ()
    session_store: Any = None
    session_recorder: Any = None
    session_search_service: Any = None
    run_store: Any = None
    approval_store: Any = None
    hosted_service_manifest: Any = None
    async_event_buffer: Any = None
    resource_lifecycle: Any = None
    resolved_services: Any = None


class AgentTurnCancelled(RuntimeError):
    """Raised internally when a host cooperatively cancels a synchronous turn."""


def _never_cancelled() -> bool:
    return False
