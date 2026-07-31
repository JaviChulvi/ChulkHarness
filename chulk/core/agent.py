"""Public agent composition, lifecycle, memory selection, and resource ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from chulk.core.action_loop import run_action_loop, run_action_loop_async
from chulk.core.action_runtime import ActionLoopRuntime
from chulk.core.context import ContextBudget, TurnContextSection
from chulk.core.events import AgentEvent, TraceEvent
from chulk.core.model_transport import ModelTransport
from chulk.core.plan_execution import PlanExecution
from chulk.core.planning import read_only_planning_tool_names
from chulk.core.prompts import BASE_SYSTEM_PROMPT
from chulk.core.state import AgentState, TurnState
from chulk.core.signals import DurableApprovalPaused
from chulk.core.tool_execution import ToolExecutor
from chulk.core.turn_effects import TurnEffects
from chulk.llm import LLMCost, LLMClient, LLMUsage
from chulk.llm.capabilities import client_requires_mcp_bridge
from chulk.llm.lifecycle import aclose_resources, close_resources
from chulk.llm.usage import (
    aggregate_cost,
    aggregate_usage,
    cost_from_dict,
    usage_from_dict,
)
from chulk.goals.runtime import GoalExecutionContext
from chulk.hosting import ExecutionScope
from chulk.hosting.async_utils import call_async_service
from chulk.mcp import MCPServerConfig
from chulk.memory.constants import PROFILE_MEMORY_TAGS
from chulk.media import (
    ContentStore,
    MediaInputPart,
    MediaProcessorRegistry,
    TextInputPart,
    UserInput,
)
from chulk.memory import (
    AsyncMemoryPolicy,
    ConversationMemory,
    MemoryPolicy,
    MemoryRecord,
    SQLiteMemoryStore,
    route_memory_candidates,
    select_memories_for_prompt,
)
from chulk.memory.extraction import extract_memory_candidates
from chulk.skills import (
    LearningProposalService,
    LearningReviewContext,
    LearningReviewCoordinator,
    LearningReviewOutcome,
    LearningReviewTrigger,
    SQLiteSkillLifecycleStore,
    SkillLifecycleManager,
    SkillLifecycleRecord,
    SkillRegistry,
    SkillSelection,
    SkillUsageKind,
)
from chulk.tools import ToolRegistry
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    ToolPermissionPolicy,
)
from chulk.tools.registry import ToolContextLifecycle, ToolExecutionContext
from chulk.tools.policy import ToolPolicyHooks
from chulk.tracing import JSONLTraceLogger
from chulk.redaction import redact_text
from chulk.usage import BudgetExceededError, ModelUsageAccounting

if TYPE_CHECKING:
    from chulk.plugins.models import PluginAuditReport
    from chulk.plugins.registry import LocalPluginRegistry


class Agent:
    """Coordinates model calls, memory retrieval, skill loading, and tools."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        state: AgentState | None = None,
        memory: ConversationMemory | None = None,
        memory_store: SQLiteMemoryStore | None = None,
        memory_policy: MemoryPolicy | None = None,
        skill_registry: SkillRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        trace_logger: JSONLTraceLogger | None = None,
        system_prompt: str = BASE_SYSTEM_PROMPT,
        max_tool_calls_per_turn: int = 5,
        max_json_repair_attempts: int = 2,
        max_skills_per_turn: int = 3,
        max_skill_content_chars: int = 4000,
        trace_max_prompt_chars: int = 50000,
        max_observation_chars: int = 12000,
        max_tool_stdout_chars: int = 8000,
        max_tool_stderr_chars: int = 4000,
        max_reflection_attempts: int = 0,
        permission_policy: ToolPermissionPolicy | None = None,
        permission_callback: Callable[
            [PermissionRequest, PermissionDecisionRecord], PermissionDecision | bool
        ]
        | None = None,
        context_budget: ContextBudget | None = None,
        max_model_output_tokens: int | None = None,
        event_callback: Callable[[str, dict], None] | None = None,
        event_sink: Callable[[AgentEvent], None] | None = None,
        audit_callback: Callable[[str, dict], None] | None = None,
        redaction_callback: Callable[[str, str, dict], str] | None = None,
        redaction_fail_closed: bool = False,
        pinned_skill_names: list[str] | None = None,
        mcp_servers: list[MCPServerConfig] | tuple[MCPServerConfig, ...] | None = None,
        mcp_bridge_tool_names: list[str] | None = None,
        owned_resources: list[object] | tuple[object, ...] | None = None,
        default_tool_context: ToolExecutionContext | None = None,
        runtime_metadata: dict | None = None,
        tool_context_lifecycle: ToolContextLifecycle | None = None,
        profile_id: str = "default",
        usage_accounting: ModelUsageAccounting | None = None,
        skill_lifecycle_store: SQLiteSkillLifecycleStore | None = None,
        skill_lifecycle: SkillLifecycleManager | None = None,
        learning_proposals: LearningProposalService | None = None,
        learning_reviewer: LearningReviewCoordinator | None = None,
        plugin_registry: LocalPluginRegistry | None = None,
        plugin_audit_report: PluginAuditReport | None = None,
        goal_execution: GoalExecutionContext | None = None,
        content_store: ContentStore | None = None,
        media_processors: MediaProcessorRegistry | None = None,
        execution_scope: ExecutionScope | None = None,
        tool_policy_hooks: ToolPolicyHooks | None = None,
        close_trace_logger: bool = True,
        restore_plan_context: bool = True,
    ) -> None:
        if max_json_repair_attempts < 0:
            raise ValueError("max_json_repair_attempts cannot be negative")
        if max_skills_per_turn < 1:
            raise ValueError("max_skills_per_turn must be greater than zero")
        if max_skill_content_chars < 1:
            raise ValueError("max_skill_content_chars must be greater than zero")
        if trace_max_prompt_chars < 1:
            raise ValueError("trace_max_prompt_chars must be greater than zero")
        if max_observation_chars < 1:
            raise ValueError("max_observation_chars must be greater than zero")
        if max_tool_stdout_chars < 1:
            raise ValueError("max_tool_stdout_chars must be greater than zero")
        if max_tool_stderr_chars < 1:
            raise ValueError("max_tool_stderr_chars must be greater than zero")
        if max_reflection_attempts < 0:
            raise ValueError("max_reflection_attempts cannot be negative")
        self.context_budget = context_budget or ContextBudget()
        self.max_model_output_tokens = max_model_output_tokens
        if (
            self.max_model_output_tokens is not None
            and self.max_model_output_tokens < 1
        ):
            raise ValueError("max_model_output_tokens must be greater than zero")
        self.profile_id = profile_id
        self.llm_client = llm_client
        self.state = state or AgentState()
        self.memory = memory or ConversationMemory()
        self.memory_store = memory_store
        self.memory_policy = memory_policy or (
            MemoryPolicy(memory_store, "automatic")
            if memory_store is not None
            else None
        )
        self.skill_registry = skill_registry
        self.tool_registry = tool_registry or ToolRegistry()
        self.trace_logger = trace_logger
        self.system_prompt = system_prompt
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.max_json_repair_attempts = max_json_repair_attempts
        self.max_skills_per_turn = max_skills_per_turn
        self.max_skill_content_chars = max_skill_content_chars
        self.trace_max_prompt_chars = trace_max_prompt_chars
        self.max_observation_chars = max_observation_chars
        self.max_tool_stdout_chars = max_tool_stdout_chars
        self.max_tool_stderr_chars = max_tool_stderr_chars
        self.max_reflection_attempts = max_reflection_attempts
        self.permission_policy = permission_policy or ToolPermissionPolicy()
        self.permission_callback = permission_callback
        self.event_callback = event_callback
        self.event_sink = event_sink
        self.audit_callback = audit_callback
        self.redaction_callback = redaction_callback
        self.redaction_fail_closed = redaction_fail_closed
        self.pinned_skill_names = pinned_skill_names or []
        self.mcp_servers = tuple(mcp_servers or ())
        self.mcp_bridge_tool_names = list(mcp_bridge_tool_names or [])
        self._owned_resources = list(owned_resources or [])
        self._closed = False
        self._tool_contexts: dict[str, ToolExecutionContext | None] = {}
        self.default_tool_context = default_tool_context
        self.runtime_metadata = deepcopy(runtime_metadata or {})
        self.execution_scope = execution_scope
        self._close_trace_logger = close_trace_logger
        self.usage_accounting = usage_accounting
        self.skill_lifecycle_store = skill_lifecycle_store
        self.skill_lifecycle = skill_lifecycle
        self.learning_proposals = learning_proposals
        self.learning_reviewer = learning_reviewer
        self.plugin_registry = plugin_registry
        self.plugin_audit_report = plugin_audit_report
        self.goal_execution = goal_execution
        self.content_store = content_store
        self.media_processors = media_processors or MediaProcessorRegistry()
        self.tool_context_lifecycle = tool_context_lifecycle
        self.async_memory_store: object | None = None
        self.async_memory_policy: AsyncMemoryPolicy | None = None
        self.async_skill_registry: object | None = None
        self.async_usage_accounting: object | None = None
        self.async_artifact_store: object | None = None
        self.async_content_store: object | None = None
        self.async_media_processors: object | None = None
        self.async_flushables: tuple[object, ...] = ()
        self._profile_memories: list[MemoryRecord] = []
        self._relevant_memories: list[MemoryRecord] = []
        self._selected_skills: list[SkillSelection] = []
        if restore_plan_context:
            self._restore_plan_turn_context()
        self.state.conversation_summary = self.memory.conversation_summary
        self._tool_executor = ToolExecutor(
            registry=self.tool_registry,
            permission_policy=self.permission_policy,
            permission_callback=self.permission_callback,
            trace=self._trace,
            get_context=self._tool_context_for_turn,
            usage_accounting=self.usage_accounting,
            goal_execution=self.goal_execution,
            execution_scope=self.execution_scope,
            policy_hooks=tool_policy_hooks,
            get_context_async=self._tool_context_for_turn_async,
            async_usage_accounting=self.async_usage_accounting,
        )
        self._plan_execution = PlanExecution(
            state=self.state,
            memory=self.memory,
            trace=self._trace,
        )
        self._turn_effects = TurnEffects(
            state=self.state,
            memory=self.memory,
            llm_client=self.llm_client,
            plan=self._plan_execution,
            trace=self._trace,
            redact_text=self._redact_text,
            artifact_writer=self._write_tool_output_artifact,
            planning_tool_names=lambda: read_only_planning_tool_names(
                self.tool_registry.list_tools()
            ),
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            max_reflection_attempts=self.max_reflection_attempts,
            max_observation_chars=self.max_observation_chars,
            max_tool_stdout_chars=self.max_tool_stdout_chars,
            max_tool_stderr_chars=self.max_tool_stderr_chars,
            async_artifact_writer=self._write_tool_output_artifact_async,
        )
        self._model_transport = ModelTransport(
            llm_client=self.llm_client,
            state=self.state,
            memory=self.memory,
            tool_registry=self.tool_registry,
            system_prompt=self.system_prompt,
            context_budget=self.context_budget,
            skill_registry=self.skill_registry,
            get_profile_memories=lambda: self._profile_memories,
            get_relevant_memories=lambda: self._relevant_memories,
            get_selected_skills=lambda: self._selected_skills,
            trace=self._trace,
            record_accounting=self._record_model_accounting,
            reserve_accounting=self._reserve_model_accounting,
            release_accounting=self._release_model_accounting,
            resolve_mcp_approval=self._tool_executor.resolve_hosted_mcp_approval,
            mcp_servers=self.mcp_servers,
            max_skill_content_chars=self.max_skill_content_chars,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            max_json_repair_attempts=self.max_json_repair_attempts,
            max_reflection_attempts=self.max_reflection_attempts,
            trace_max_prompt_chars=self.trace_max_prompt_chars,
            max_output_tokens=self.max_model_output_tokens,
            record_accounting_async=self._record_model_accounting_async,
            reserve_accounting_async=self._reserve_model_accounting_async,
            release_accounting_async=self._release_model_accounting_async,
            flush_async=self._flush_async_services,
        )
        self._action_runtime = ActionLoopRuntime(
            model=self._model_transport,
            tools=self._tool_executor,
            effects=self._turn_effects,
            async_flush=self._flush_async_services,
        )

    @property
    def closed(self) -> bool:
        """Return whether the runtime has been finalized."""
        return self._closed

    def close(self) -> None:
        """Finalize owned closeable resources exactly once."""
        if self._closed:
            return
        self._closed = True
        failures: list[Exception] = []
        for context in tuple(self._tool_contexts.values()):
            if context is None or self.tool_context_lifecycle is None:
                continue
            try:
                self.tool_context_lifecycle.close(context)
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        self._tool_contexts.clear()
        for resource in reversed(self._owned_resources):
            try:
                close_resources((resource,))
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        if self.trace_logger is not None and self._close_trace_logger:
            try:
                self.trace_logger.close()
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        self.event_callback = None
        self.event_sink = None
        self.audit_callback = None
        if failures:
            raise RuntimeError(
                f"Failed to close {len(failures)} owned agent resource(s)"
            ) from failures[0]

    async def aclose(self) -> None:
        """Finalize owned closeable resources exactly once from an async host."""
        if self._closed:
            return
        self._closed = True
        failures: list[Exception] = []
        for context in tuple(self._tool_contexts.values()):
            if context is None or self.tool_context_lifecycle is None:
                continue
            try:
                await self.tool_context_lifecycle.aclose(context)
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        self._tool_contexts.clear()
        for resource in reversed(self._owned_resources):
            try:
                await aclose_resources((resource,))
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        if self.trace_logger is not None and self._close_trace_logger:
            try:
                self.trace_logger.close()
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        self.event_callback = None
        self.event_sink = None
        self.audit_callback = None
        if failures:
            raise RuntimeError(
                f"Failed to close {len(failures)} owned agent resource(s)"
            ) from failures[0]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Agent is closed")

    def run_turn(
        self,
        user_message: str,
        *,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> str:
        """Run one user turn and return the assistant response."""
        self._ensure_open()
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")
        return self._run_user_turn(
            clean_message,
            require_plan=False,
            context_sections=context_sections,
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata,
            tool_context=tool_context,
        )

    def run_input(
        self,
        user_input: UserInput,
        *,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> str:
        """Run one typed text/media turn through explicit media transforms."""
        self._ensure_open()
        blocked = self._new_turn_block_message()
        if blocked is not None:
            return blocked
        turn_id = str(uuid4())
        projection, prepared, media_sections, media_events = (
            self._prepare_user_input(user_input, turn_id=turn_id)
        )
        return self._run_user_turn(
            projection,
            require_plan=False,
            context_sections=[*(context_sections or []), *media_sections],
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata,
            tool_context=tool_context,
            input_parts=list(user_input.safe_metadata()),
            model_input=prepared,
            media_events=media_events,
            turn_id=turn_id,
        )

    async def run_turn_async(
        self,
        user_message: str,
        *,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> str:
        """Run one user turn and await async tools in the current event loop."""
        self._ensure_open()
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")
        return await self._run_user_turn_async(
            clean_message,
            require_plan=False,
            context_sections=context_sections,
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata,
            tool_context=tool_context,
        )

    async def run_input_async(
        self,
        user_input: UserInput,
        *,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
    ) -> str:
        """Run a typed input without blocking the caller during transforms."""
        self._ensure_open()
        blocked = self._new_turn_block_message()
        if blocked is not None:
            return blocked
        turn_id = str(uuid4())
        projection, prepared, media_sections, media_events = (
            await self._prepare_user_input_async(
            user_input,
            turn_id=turn_id,
        )
        )
        return await self._run_user_turn_async(
            projection,
            require_plan=False,
            context_sections=[*(context_sections or []), *media_sections],
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata,
            tool_context=tool_context,
            input_parts=list(user_input.safe_metadata()),
            model_input=prepared,
            media_events=media_events,
            turn_id=turn_id,
        )

    def run_planned_turn(self, user_message: str) -> str:
        """Run one user turn that must propose a plan before tool execution."""
        self._ensure_open()
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")
        return self._run_user_turn(clean_message, require_plan=True)

    async def run_planned_turn_async(self, user_message: str) -> str:
        """Run one planned turn through async model and tool transports."""
        self._ensure_open()
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")
        return await self._run_user_turn_async(clean_message, require_plan=True)

    def _run_user_turn(
        self,
        clean_message: str,
        *,
        require_plan: bool,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
        input_parts: list[dict] | None = None,
        model_input: UserInput | None = None,
        media_events: list[dict] | None = None,
        turn_id: str | None = None,
    ) -> str:
        """Start a user turn and run it until it completes or waits for approval."""
        self._refresh_action_runtime()
        turn: TurnState | None = None
        previous_turn_count = len(self.state.turns)
        try:
            turn_or_response = self._start_user_turn(
                clean_message,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
                input_parts=input_parts,
                model_input=model_input,
                media_events=media_events,
                turn_id=turn_id,
            )
            if isinstance(turn_or_response, str):
                return turn_or_response
            turn = turn_or_response
            result = self._run_action_loop(turn, require_plan=require_plan)
        except DurableApprovalPaused:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                turn.status = "waiting_for_approval"
                self._release_tool_context(turn)
            raise
        except BaseException as exc:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                self._terminalize_exception(turn, exc)
                self._release_tool_context(turn)
            raise
        if turn.status != "waiting_for_approval":
            self._release_tool_context(turn)
        return result

    async def _run_user_turn_async(
        self,
        clean_message: str,
        *,
        require_plan: bool,
        context_sections: list[TurnContextSection | dict | str] | None = None,
        prompt_profile: str | None = None,
        locale: str | None = None,
        extension_metadata: dict | None = None,
        tool_context: ToolExecutionContext | dict | None = None,
        input_parts: list[dict] | None = None,
        model_input: UserInput | None = None,
        media_events: list[dict] | None = None,
        turn_id: str | None = None,
    ) -> str:
        """Start a user turn and run it with async tool execution."""
        self._refresh_action_runtime()
        turn: TurnState | None = None
        previous_turn_count = len(self.state.turns)
        try:
            turn_or_response = await self._start_user_turn_async(
                clean_message,
                context_sections=context_sections,
                prompt_profile=prompt_profile,
                locale=locale,
                extension_metadata=extension_metadata,
                tool_context=tool_context,
                input_parts=input_parts,
                model_input=model_input,
                media_events=media_events,
                turn_id=turn_id,
            )
            if isinstance(turn_or_response, str):
                return turn_or_response
            turn = turn_or_response
            result = await self._run_action_loop_async(turn, require_plan=require_plan)
        except DurableApprovalPaused as exc:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                turn.status = "waiting_for_approval"
                await self._flush_async_services_after_error(exc)
                await _await_cleanup_after_error(
                    self._release_tool_context_async(turn),
                    exc,
                )
            raise
        except BaseException as exc:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                self._terminalize_exception(turn, exc)
                await self._flush_async_services_after_error(exc)
                await _await_cleanup_after_error(
                    self._release_tool_context_async(turn),
                    exc,
                )
            raise
        if turn.status != "waiting_for_approval":
            await self._release_tool_context_async(turn)
        await self._flush_async_services()
        return result

    async def _start_user_turn_async(
        self,
        clean_message: str,
        *,
        context_sections: list[TurnContextSection | dict | str] | None,
        prompt_profile: str | None,
        locale: str | None,
        extension_metadata: dict | None,
        tool_context: ToolExecutionContext | dict | None,
        input_parts: list[dict] | None = None,
        model_input: UserInput | None = None,
        media_events: list[dict] | None = None,
        turn_id: str | None = None,
    ) -> TurnState | str:
        """Create a turn while awaiting hosted memory, skills, and execution."""

        blocked = self._new_turn_block_message()
        if blocked is not None:
            return blocked

        turn_context_sections = _coerce_turn_context_sections(context_sections)
        execution_context = (
            _coerce_tool_execution_context(tool_context)
            or self.default_tool_context
        )
        turn = TurnState(
            user_message=clean_message,
            turn_id=turn_id or str(uuid4()),
            available_tool_names=[
                tool.name for tool in self.tool_registry.list_tools()
            ],
            context_sections=turn_context_sections,
            prompt_profile=prompt_profile,
            locale=locale,
            input_parts=deepcopy(input_parts or []),
            extension_metadata={
                **deepcopy(extension_metadata or {}),
                **deepcopy(self.runtime_metadata),
            },
            tool_context_metadata=(
                execution_context.metadata if execution_context else {}
            ),
        )
        turn.model_input = model_input
        if execution_context is None:
            execution_context = ToolExecutionContext()
        execution_context = ToolExecutionContext(
            metadata={
                **execution_context.metadata,
                "conversation_id": self.state.conversation_id,
                "turn_id": turn.turn_id,
            },
            deps=execution_context.deps,
            execution_session=execution_context.execution_session,
        )
        if (
            self.tool_context_lifecycle is not None
            and execution_context.execution_session is None
        ):
            execution_context = await self.tool_context_lifecycle.open_async(
                execution_context
            )
        self._tool_contexts[turn.turn_id] = execution_context
        self.state.current_turn_id = turn.turn_id
        self.state.available_tool_names = turn.available_tool_names
        self.state.turns.append(turn)
        self._trace(TraceEvent.TURN_STARTED, {"turn": turn.to_dict()})
        if input_parts:
            self._trace(
                TraceEvent.MEDIA_INPUT_PREPARED,
                {
                    "turn_id": turn.turn_id,
                    "parts": deepcopy(input_parts),
                },
            )
        for media_event in media_events or []:
            self._trace(
                TraceEvent.MEDIA_TRANSFORMED,
                {"turn_id": turn.turn_id, **deepcopy(media_event)},
            )
        model_selection = turn.extension_metadata.get("model_selection")
        if isinstance(model_selection, dict):
            self._trace(
                TraceEvent.MODEL_PROFILE_SELECTED,
                {"turn_id": turn.turn_id, **model_selection},
            )
        if turn_context_sections or prompt_profile or locale:
            self._trace(
                TraceEvent.TURN_CONTEXT_SELECTED,
                {
                    "turn_id": turn.turn_id,
                    "context_section_ids": [
                        section.id for section in turn_context_sections
                    ],
                    "context_sections": [
                        section.to_dict() for section in turn_context_sections
                    ],
                    "prompt_profile": prompt_profile,
                    "locale": locale,
                },
            )

        await self._extract_long_term_memories_async(clean_message)
        await self._select_long_term_memories_async(clean_message)
        await self._select_skills_async(clean_message)
        turn.extracted_memory_ids = list(self.state.extracted_memory_ids)
        turn.loaded_memory_ids = list(self.state.loaded_memory_ids)
        turn.loaded_skill_names = list(self.state.loaded_skill_names)
        self.memory.add_user_message(clean_message)
        self._trace(
            TraceEvent.USER_MESSAGE,
            {"turn_id": turn.turn_id, "content": clean_message},
        )
        await self._flush_async_services()
        return turn

    def _start_user_turn(
        self,
        clean_message: str,
        *,
        context_sections: list[TurnContextSection | dict | str] | None,
        prompt_profile: str | None,
        locale: str | None,
        extension_metadata: dict | None,
        tool_context: ToolExecutionContext | dict | None,
        input_parts: list[dict] | None = None,
        model_input: UserInput | None = None,
        media_events: list[dict] | None = None,
        turn_id: str | None = None,
    ) -> TurnState | str:
        """Create and trace a user turn before model/tool execution."""
        blocked = self._new_turn_block_message()
        if blocked is not None:
            return blocked

        turn_context_sections = _coerce_turn_context_sections(context_sections)
        execution_context = (
            _coerce_tool_execution_context(tool_context) or self.default_tool_context
        )
        turn = TurnState(
            user_message=clean_message,
            turn_id=turn_id or str(uuid4()),
            available_tool_names=[
                tool.name for tool in self.tool_registry.list_tools()
            ],
            context_sections=turn_context_sections,
            prompt_profile=prompt_profile,
            locale=locale,
            input_parts=deepcopy(input_parts or []),
            extension_metadata={
                **deepcopy(extension_metadata or {}),
                **deepcopy(self.runtime_metadata),
            },
            tool_context_metadata=execution_context.metadata
            if execution_context
            else {},
        )
        turn.model_input = model_input
        if execution_context is None:
            execution_context = ToolExecutionContext()
        execution_context = ToolExecutionContext(
            metadata={
                **execution_context.metadata,
                "conversation_id": self.state.conversation_id,
                "turn_id": turn.turn_id,
            },
            deps=execution_context.deps,
            execution_session=execution_context.execution_session,
        )
        if (
            self.tool_context_lifecycle is not None
            and execution_context.execution_session is None
        ):
            execution_context = self.tool_context_lifecycle.open(execution_context)
        self.state.current_turn_id = turn.turn_id
        self.state.available_tool_names = turn.available_tool_names
        self.state.turns.append(turn)
        self._trace(TraceEvent.TURN_STARTED, {"turn": turn.to_dict()})
        if input_parts:
            self._trace(
                TraceEvent.MEDIA_INPUT_PREPARED,
                {
                    "turn_id": turn.turn_id,
                    "parts": deepcopy(input_parts),
                },
            )
        for media_event in media_events or []:
            self._trace(
                TraceEvent.MEDIA_TRANSFORMED,
                {"turn_id": turn.turn_id, **deepcopy(media_event)},
            )
        model_selection = turn.extension_metadata.get("model_selection")
        if isinstance(model_selection, dict):
            self._trace(
                TraceEvent.MODEL_PROFILE_SELECTED,
                {"turn_id": turn.turn_id, **model_selection},
            )
        if turn_context_sections or prompt_profile or locale:
            self._trace(
                TraceEvent.TURN_CONTEXT_SELECTED,
                {
                    "turn_id": turn.turn_id,
                    "context_section_ids": [
                        section.id for section in turn_context_sections
                    ],
                    "context_sections": [
                        section.to_dict() for section in turn_context_sections
                    ],
                    "prompt_profile": prompt_profile,
                    "locale": locale,
                },
            )

        self._extract_long_term_memories(clean_message)
        self._select_long_term_memories(clean_message)
        self._select_skills(clean_message)
        turn.extracted_memory_ids = list(self.state.extracted_memory_ids)
        turn.loaded_memory_ids = list(self.state.loaded_memory_ids)
        turn.loaded_skill_names = list(self.state.loaded_skill_names)
        self.memory.add_user_message(clean_message)
        self._trace(
            TraceEvent.USER_MESSAGE, {"turn_id": turn.turn_id, "content": clean_message}
        )

        self._tool_contexts[turn.turn_id] = execution_context
        return turn

    def _new_turn_block_message(self) -> str | None:
        if self.has_pending_plan():
            return (
                "A plan is waiting for approval. Use /approve to execute it or "
                "/reject to cancel it."
            )
        if self.has_resumable_plan():
            return (
                "An approved plan is waiting to continue. Use /approve to resume it or "
                "/reject to cancel it before starting a new turn."
            )
        return None

    def _prepare_user_input(
        self,
        user_input: UserInput,
        *,
        turn_id: str,
    ) -> tuple[
        str,
        UserInput,
        list[TurnContextSection],
        list[dict],
    ]:
        """Verify ownership and convert non-native media into bounded text."""
        projection = user_input.textual_projection(
            max_chars=self.max_observation_chars
        )
        prepared_parts: list[TextInputPart] = []
        sections: list[TurnContextSection] = []
        events: list[dict] = []
        for operation_index, part in enumerate(user_input.parts, start=1):
            if isinstance(part, TextInputPart):
                prepared_parts.append(part)
                continue
            if not isinstance(part, MediaInputPart):  # pragma: no cover - closed union
                raise TypeError("unsupported user input part")
            if self.content_store is None:
                raise RuntimeError(
                    "Typed media input requires a profile-owned ContentStore."
                )
            item = self.content_store.get(
                part.media.content_ref,
                profile_id=self.profile_id,
            )
            if item.sha256 != part.media.sha256:
                raise ValueError("typed media metadata does not match stored content")
            processor, transform = self.media_processors.choose(item)
            capability = processor.capability
            usage_reservation = None
            if self.usage_accounting is not None:
                usage_reservation = self.usage_accounting.reserve_media_transform(
                    turn_id=turn_id,
                    operation_index=operation_index,
                    content_ref=item.content_ref.id,
                    processor=capability.name,
                    provider=capability.provider,
                    pricing_per_unit=capability.pricing_per_unit,
                    network_access=capability.network_access,
                )
            try:
                result = self.media_processors.transform(
                    self.content_store,
                    item,
                    instruction=part.caption or projection,
                    selection=(processor, transform),
                )
            except BaseException:
                if self.usage_accounting is not None and usage_reservation is not None:
                    self.usage_accounting.release_media_transform(usage_reservation)
                raise
            if result.text is None:
                raise RuntimeError(
                    "Native media transforms require a model client media adapter."
                )
            bounded_text = result.text[: self.max_observation_chars]
            sections.append(
                TurnContextSection(
                    id=f"media-{item.content_ref.id.removeprefix('content:')[:16]}",
                    title=f"Media transform: {item.file_name or item.kind.value}",
                    source="media_processor",
                    content=bounded_text,
                    metadata={
                        "trusted": item.trust.value != "untrusted",
                        "external_content": True,
                        "content_ref": item.content_ref.id,
                        "mime_type": item.mime_type,
                        "processor": result.processor,
                        "transform": result.transform.value,
                        "truncated": len(result.text) > len(bounded_text),
                    },
                    persist_content=False,
                )
            )
            events.append(
                {
                    "content_ref": item.content_ref.id,
                    "media_kind": item.kind.value,
                    "mime_type": item.mime_type,
                    "processor": result.processor,
                    "transform": result.transform.value,
                    "output_characters": len(bounded_text),
                    "units": result.units,
                }
            )
            if self.usage_accounting is not None and usage_reservation is not None:
                self.usage_accounting.commit_media_transform(
                    usage_reservation,
                    content_ref=item.content_ref.id,
                    processor=result.processor,
                    provider=str(result.metadata.get("provider") or "local"),
                    byte_length=item.byte_length,
                    units=result.units,
                    unit_name=str(result.metadata.get("unit_name") or "request"),
                    network_access=bool(result.metadata.get("network_access")),
                    retains_data=bool(result.metadata.get("retains_data")),
                )
        if not prepared_parts:
            prepared_parts.append(TextInputPart(projection, external_content=True))
        return projection, UserInput(tuple(prepared_parts)), sections, events

    async def _prepare_user_input_async(
        self,
        user_input: UserInput,
        *,
        turn_id: str,
    ) -> tuple[
        str,
        UserInput,
        list[TurnContextSection],
        list[dict],
    ]:
        content_store = self.async_content_store
        processors = self.async_media_processors
        if content_store is None or processors is None:
            return await asyncio.to_thread(
                self._prepare_user_input,
                user_input,
                turn_id=turn_id,
            )
        projection = user_input.textual_projection(
            max_chars=self.max_observation_chars
        )
        prepared_parts: list[TextInputPart] = []
        sections: list[TurnContextSection] = []
        events: list[dict] = []
        for operation_index, part in enumerate(user_input.parts, start=1):
            if isinstance(part, TextInputPart):
                prepared_parts.append(part)
                continue
            if not isinstance(part, MediaInputPart):
                raise TypeError("unsupported user input part")
            item = await call_async_service(
                content_store,
                "get",
                part.media.content_ref,
                profile_id=self.profile_id,
            )
            if item.sha256 != part.media.sha256:
                raise ValueError(
                    "typed media metadata does not match stored content"
                )
            processor, transform = await call_async_service(
                processors,
                "choose",
                item,
            )
            capability = processor.capability
            usage_reservation = None
            if self.async_usage_accounting is not None:
                usage_reservation = await call_async_service(
                    self.async_usage_accounting,
                    "reserve_media_transform",
                    turn_id=turn_id,
                    operation_index=operation_index,
                    content_ref=item.content_ref.id,
                    processor=capability.name,
                    provider=capability.provider,
                    pricing_per_unit=capability.pricing_per_unit,
                    network_access=capability.network_access,
                )
            try:
                result = await call_async_service(
                    processors,
                    "transform",
                    content_store,
                    item,
                    instruction=part.caption or projection,
                    selection=(processor, transform),
                )
                if result.text is None:
                    raise RuntimeError(
                        "Native media transforms require a model client media "
                        "adapter."
                    )
                bounded_text = result.text[: self.max_observation_chars]
                sections.append(
                    TurnContextSection(
                        id=(
                            "media-"
                            + item.content_ref.id.removeprefix("content:")[:16]
                        ),
                        title=(
                            f"Media transform: "
                            f"{item.file_name or item.kind.value}"
                        ),
                        source="media_processor",
                        content=bounded_text,
                        metadata={
                            "trusted": item.trust.value != "untrusted",
                            "external_content": True,
                            "content_ref": item.content_ref.id,
                            "mime_type": item.mime_type,
                            "processor": result.processor,
                            "transform": result.transform.value,
                            "truncated": len(result.text) > len(bounded_text),
                        },
                        persist_content=False,
                    )
                )
                events.append(
                    {
                        "content_ref": item.content_ref.id,
                        "media_kind": item.kind.value,
                        "mime_type": item.mime_type,
                        "processor": result.processor,
                        "transform": result.transform.value,
                        "output_characters": len(bounded_text),
                        "units": result.units,
                    }
                )
                if (
                    self.async_usage_accounting is not None
                    and usage_reservation is not None
                ):
                    await call_async_service(
                        self.async_usage_accounting,
                        "commit_media_transform",
                        usage_reservation,
                        content_ref=item.content_ref.id,
                        processor=result.processor,
                        provider=str(
                            result.metadata.get("provider") or "local"
                        ),
                        byte_length=item.byte_length,
                        units=result.units,
                        unit_name=str(
                            result.metadata.get("unit_name") or "request"
                        ),
                        network_access=bool(
                            result.metadata.get("network_access")
                        ),
                        retains_data=bool(
                            result.metadata.get("retains_data")
                        ),
                    )
                    usage_reservation = None
            except BaseException as exc:
                if (
                    self.async_usage_accounting is not None
                    and usage_reservation is not None
                ):
                    await _await_cleanup_after_error(
                        call_async_service(
                            self.async_usage_accounting,
                            "release_media_transform",
                            usage_reservation,
                        ),
                        exc,
                    )
                raise
        if not prepared_parts:
            prepared_parts.append(
                TextInputPart(projection, external_content=True)
            )
        return projection, UserInput(tuple(prepared_parts)), sections, events

    def has_pending_plan(self) -> bool:
        """Return True when a turn is paused on a plan awaiting approval."""
        turn = self._pending_plan_turn()
        return bool(turn and turn.active_plan and not turn.plan_approved)

    def has_resumable_plan(self) -> bool:
        """Return True when an approved durable plan can continue after restart."""
        return self._resumable_plan_turn() is not None

    def approve_plan(self) -> str:
        """Approve the pending plan and continue the paused turn."""
        self._ensure_open()
        self._refresh_action_runtime()
        turn = self._pending_plan_turn() or self._resumable_plan_turn()
        try:
            turn_or_response = self._prepare_plan_execution()
            if isinstance(turn_or_response, str):
                return turn_or_response
            turn = turn_or_response
            return self._run_action_loop(turn, require_plan=False)
        except BaseException as exc:
            if turn is not None:
                self._terminalize_exception(turn, exc)
            raise
        finally:
            if turn is not None:
                self._release_tool_context(turn)

    async def approve_plan_async(self) -> str:
        """Approve the pending plan and continue it with async tool execution."""
        self._ensure_open()
        self._refresh_action_runtime()
        turn = self._pending_plan_turn() or self._resumable_plan_turn()
        try:
            turn_or_response = self._prepare_plan_execution()
            if isinstance(turn_or_response, str):
                result = turn_or_response
            else:
                turn = turn_or_response
                await self._flush_async_services()
                result = await self._run_action_loop_async(
                    turn,
                    require_plan=False,
                )
        except BaseException as exc:
            if turn is not None:
                self._terminalize_exception(turn, exc)
                await self._flush_async_services_after_error(exc)
                await _await_cleanup_after_error(
                    self._release_tool_context_async(turn),
                    exc,
                )
            raise
        if turn is not None:
            await self._release_tool_context_async(turn)
        return result

    def _prepare_plan_execution(self) -> TurnState | str:
        """Approve a pending plan or continue an already-approved durable plan."""
        turn = self._pending_plan_turn()
        if turn is None:
            resumed_turn = self._resumable_plan_turn()
            if resumed_turn is None or resumed_turn.active_plan is None:
                return "No plan is waiting for approval."
            self.state.current_turn_id = resumed_turn.turn_id
            self.state.active_plan = resumed_turn.active_plan
            self.state.messages = self.memory.recent()
            return resumed_turn
        if turn.active_plan is None:
            return "No plan is waiting for approval."

        turn.approve_plan()
        self.state.pending_plan_turn_id = None
        self.state.active_plan = turn.active_plan
        self.state.messages = self.memory.recent()
        self._trace(
            TraceEvent.PLAN_APPROVED,
            {
                "turn_id": turn.turn_id,
                "plan": turn.active_plan.to_dict(),
                "turn": turn.to_dict(),
            },
        )
        return turn

    def reject_plan(self) -> str:
        """Reject a pending plan or cancel a restored approved plan."""
        self._ensure_open()
        self._refresh_action_runtime()
        turn = self._pending_plan_turn()
        resumed = False
        if turn is None:
            turn = self._resumable_plan_turn()
            resumed = turn is not None
        if turn is None or turn.active_plan is None:
            return "No plan is waiting for approval."

        try:
            if resumed:
                message = (
                    "Approved plan cancelled. No further steps will run; any work already "
                    "completed was not rolled back."
                )
                turn.cancel(message)
                event_type = TraceEvent.TURN_FAILED
            else:
                message = "Plan rejected. No tools were run."
                turn.reject_plan(message)
                event_type = TraceEvent.PLAN_REJECTED
            self._plan_execution.clear(turn)
            self.state.final_answer = message
            self.memory.add_assistant_message(message)
            self.state.messages = self.memory.recent()
            self._trace(
                event_type,
                {
                    "turn_id": turn.turn_id,
                    "plan": turn.active_plan.to_dict(),
                    "message": message,
                    "status": turn.status,
                    "turn": turn.to_dict(),
                },
            )
            self._trace(
                TraceEvent.TURN_FINISHED, self._turn_effects.state_snapshot(turn)
            )
            return message
        finally:
            self._release_tool_context(turn)

    async def reject_plan_async(self) -> str:
        """Reject or cancel a plan without entering a synchronous host path."""

        self._ensure_open()
        self._refresh_action_runtime()
        turn = self._pending_plan_turn()
        resumed = False
        if turn is None:
            turn = self._resumable_plan_turn()
            resumed = turn is not None
        if turn is None or turn.active_plan is None:
            return "No plan is waiting for approval."
        try:
            if resumed:
                message = (
                    "Approved plan cancelled. No further steps will run; any "
                    "work already completed was not rolled back."
                )
                turn.cancel(message)
                event_type = TraceEvent.TURN_FAILED
            else:
                message = "Plan rejected. No tools were run."
                turn.reject_plan(message)
                event_type = TraceEvent.PLAN_REJECTED
            self._plan_execution.clear(turn)
            self.state.final_answer = message
            self.memory.add_assistant_message(message)
            self.state.messages = self.memory.recent()
            self._trace(
                event_type,
                {
                    "turn_id": turn.turn_id,
                    "plan": turn.active_plan.to_dict(),
                    "message": message,
                    "status": turn.status,
                    "turn": turn.to_dict(),
                },
            )
            self._trace(
                TraceEvent.TURN_FINISHED,
                self._turn_effects.state_snapshot(turn),
            )
            await self._flush_async_services()
        except BaseException as exc:
            await _await_cleanup_after_error(
                self._release_tool_context_async(turn),
                exc,
            )
            raise
        await self._release_tool_context_async(turn)
        return message

    def describe_plan_status(self) -> str:
        """Return a compact pending-plan status for the CLI."""
        if self.state.active_plan is None:
            return "No active plan. Use /plan <request> to create one."

        status = self.state.active_plan.status()
        return "\n".join(
            [
                f"Active plan is {status}.",
                self.state.active_plan.to_user_text(),
            ]
        )

    def _run_action_loop(self, turn: TurnState, *, require_plan: bool) -> str:
        """Run model/tool iterations until the turn pauses or completes."""
        self._refresh_action_runtime()
        return run_action_loop(self._action_runtime, turn, require_plan=require_plan)

    async def _run_action_loop_async(
        self, turn: TurnState, *, require_plan: bool
    ) -> str:
        """Run model/tool iterations, awaiting async tool calls."""
        self._refresh_action_runtime()
        return await run_action_loop_async(
            self._action_runtime,
            turn,
            require_plan=require_plan,
        )

    def _refresh_action_runtime(self) -> None:
        """Reflect mutable public runtime configuration in focused services."""
        self._validate_mcp_route()
        model = self._model_transport
        model.llm_client = self.llm_client
        model.state = self.state
        model.memory = self.memory
        model.tool_registry = self.tool_registry
        model.system_prompt = self.system_prompt
        model.context_budget = self.context_budget
        model.skill_registry = self.skill_registry
        model.mcp_servers = tuple(self.mcp_servers)
        model.max_skill_content_chars = self.max_skill_content_chars
        model.max_tool_calls_per_turn = self.max_tool_calls_per_turn
        model.max_json_repair_attempts = self.max_json_repair_attempts
        model.max_reflection_attempts = self.max_reflection_attempts
        model.trace_max_prompt_chars = self.trace_max_prompt_chars
        model.flush_async = self._flush_async_services

        tools = self._tool_executor
        tools.registry = self.tool_registry
        tools.permission_policy = self.permission_policy
        tools.permission_callback = self.permission_callback
        tools.usage_accounting = self.usage_accounting
        tools.async_usage_accounting = self.async_usage_accounting
        tools.goal_execution = self.goal_execution

        self._plan_execution.state = self.state
        self._plan_execution.memory = self.memory

        effects = self._turn_effects
        effects.state = self.state
        effects.memory = self.memory
        effects.llm_client = self.llm_client
        effects.max_tool_calls_per_turn = self.max_tool_calls_per_turn
        effects.max_reflection_attempts = self.max_reflection_attempts
        effects.max_observation_chars = self.max_observation_chars
        effects.max_tool_stdout_chars = self.max_tool_stdout_chars
        effects.max_tool_stderr_chars = self.max_tool_stderr_chars
        effects.async_artifact_writer = self._write_tool_output_artifact_async
        self._action_runtime.async_flush = self._flush_async_services

    def _restore_plan_turn_context(self) -> None:
        """Restore context that shaped a pending or resumable approved plan."""
        turn = self._pending_plan_turn() or self._resumable_plan_turn()
        if turn is None:
            return

        if self.skill_registry is not None:
            for name in turn.loaded_skill_names:
                skill = self.skill_registry.get_skill(name, visible_only=True)
                if skill is None:
                    continue
                self.skill_registry.load_content(skill.name)
                self._selected_skills.append(
                    SkillSelection(
                        skill=skill,
                        score=10_000,
                        matched_keywords=["restored_pending_plan"],
                    )
                )

        if (
            self.memory_store is not None
            and self.memory_policy is not None
            and self.memory_policy.retrieval_enabled
        ):
            for memory_id in turn.loaded_memory_ids:
                memory = self.memory_store.get_memory(
                    memory_id,
                    include_archived=True,
                )
                if memory is None:
                    continue
                if set(memory.tags) & PROFILE_MEMORY_TAGS:
                    self._profile_memories.append(memory)
                else:
                    self._relevant_memories.append(memory)
        elif (
            self.memory_policy is not None and not self.memory_policy.retrieval_enabled
        ):
            self.state.loaded_memory_ids = []
            turn.loaded_memory_ids = []

    async def restore_plan_turn_context_async(self) -> None:
        """Restore persisted plan context through native async services."""

        turn = self._pending_plan_turn() or self._resumable_plan_turn()
        if turn is None:
            return
        registry = self.async_skill_registry
        if registry is not None:
            for name in turn.loaded_skill_names:
                skill = await call_async_service(
                    registry,
                    "get_skill",
                    name,
                    visible_only=True,
                )
                if skill is None:
                    continue
                loaded_content = await call_async_service(
                    registry,
                    "load_content",
                    skill.name,
                )
                if getattr(skill, "loaded_content", None) is None:
                    skill.loaded_content = loaded_content
                self._selected_skills.append(
                    SkillSelection(
                        skill=skill,
                        score=10_000,
                        matched_keywords=["restored_pending_plan"],
                    )
                )

        store = self.async_memory_store
        policy = self.async_memory_policy
        if store is not None and policy is not None and policy.retrieval_enabled:
            for memory_id in turn.loaded_memory_ids:
                memory = await call_async_service(
                    store,
                    "get_memory",
                    memory_id,
                    include_archived=True,
                )
                if memory is None:
                    continue
                if set(memory.tags) & PROFILE_MEMORY_TAGS:
                    self._profile_memories.append(memory)
                else:
                    self._relevant_memories.append(memory)
        elif policy is not None and not policy.retrieval_enabled:
            self.state.loaded_memory_ids = []
            turn.loaded_memory_ids = []

    def _validate_mcp_route(self) -> None:
        """Fail closed when mutable runtime state lacks a required MCP bridge."""
        if not self.mcp_servers or not client_requires_mcp_bridge(self.llm_client):
            return
        registered_names = {tool.name for tool in self.tool_registry.list_tools()}
        bridge_names = set(self.mcp_bridge_tool_names)
        if bridge_names and bridge_names.issubset(registered_names):
            return
        raise RuntimeError(
            "The current LLM client requires MCP bridge tools, but this agent was "
            "not assembled with them. Rebuild the agent for the replacement client."
        )

    def _pending_plan_turn(self) -> TurnState | None:
        pending_turn_id = self.state.pending_plan_turn_id
        if pending_turn_id is None:
            return None
        for turn in self.state.turns:
            if turn.turn_id == pending_turn_id:
                return turn
        return None

    def _resumable_plan_turn(self) -> TurnState | None:
        if self.state.active_plan is None or not self.state.turns:
            return None
        turn = self.state.turns[-1]
        if (
            turn.active_plan is self.state.active_plan
            and turn.can_continue_approved_plan()
        ):
            return turn
        return None

    def _extract_long_term_memories(self, user_message: str) -> None:
        """Save explicit user-requested memories before retrieval."""
        self.state.extracted_memory_ids = []
        if self.memory_store is None or self.memory_policy is None:
            return
        result = route_memory_candidates(
            user_message,
            self.memory_policy,
            conversation_id=self.state.conversation_id,
            turn_id=self.state.current_turn_id,
        )
        self.state.extracted_memory_ids = list(result.accepted_memory_ids)
        if result.accepted_memory_ids or result.proposal_ids:
            self._trace(
                TraceEvent.MEMORY_EXTRACTION_COMPLETED,
                {
                    "turn_id": self.state.current_turn_id,
                    "memory_ids": list(result.accepted_memory_ids),
                    "proposal_ids": list(result.proposal_ids),
                    "memory_mode": self.memory_policy.mode.value,
                    "memory_namespace": self.memory_store.namespace,
                },
            )
        for proposal_id in result.proposal_ids:
            self._trace(
                TraceEvent.LEARNING_PROPOSAL_CHANGED,
                {
                    "turn_id": self.state.current_turn_id,
                    "proposal_id": proposal_id,
                    "kind": "memory_create",
                    "status": "pending",
                    "action": "created",
                    "target_name": None,
                },
            )

    async def _extract_long_term_memories_async(
        self,
        user_message: str,
    ) -> None:
        self.state.extracted_memory_ids = []
        policy = self.async_memory_policy
        if policy is None:
            await asyncio.to_thread(
                self._extract_long_term_memories,
                user_message,
            )
            return
        result = await policy.handle_candidates(
            extract_memory_candidates(user_message),
            conversation_id=self.state.conversation_id,
            turn_id=self.state.current_turn_id,
            evidence=user_message,
        )
        self.state.extracted_memory_ids = list(result.accepted_memory_ids)
        namespace = getattr(self.async_memory_store, "namespace", None)
        if result.accepted_memory_ids or result.proposal_ids:
            self._trace(
                TraceEvent.MEMORY_EXTRACTION_COMPLETED,
                {
                    "turn_id": self.state.current_turn_id,
                    "memory_ids": list(result.accepted_memory_ids),
                    "proposal_ids": list(result.proposal_ids),
                    "memory_mode": policy.mode.value,
                    "memory_namespace": namespace,
                },
            )
        for proposal_id in result.proposal_ids:
            self._trace(
                TraceEvent.LEARNING_PROPOSAL_CHANGED,
                {
                    "turn_id": self.state.current_turn_id,
                    "proposal_id": proposal_id,
                    "kind": "memory_create",
                    "status": "pending",
                    "action": "created",
                    "target_name": None,
                },
            )

    def _select_long_term_memories(self, user_message: str) -> None:
        """Select durable memories that should shape this turn."""
        self._profile_memories = []
        self._relevant_memories = []
        self.state.loaded_memory_ids = []

        if (
            self.memory_store is None
            or self.memory_policy is None
            or not self.memory_policy.retrieval_enabled
        ):
            return

        self._trace(
            TraceEvent.MEMORY_SEARCH_STARTED,
            {
                "turn_id": self.state.current_turn_id,
                "query": user_message,
                "memory_namespace": self.memory_store.namespace,
            },
        )
        profile, relevant = select_memories_for_prompt(
            self.memory_store,
            user_message,
        )
        self._profile_memories = profile
        self._relevant_memories = relevant
        self.state.loaded_memory_ids = [
            memory.id for memory in [*profile, *relevant]
        ]
        self._trace(
            TraceEvent.MEMORY_SEARCH_COMPLETED,
            {
                "turn_id": self.state.current_turn_id,
                "profile_memory_ids": [memory.id for memory in profile],
                "relevant_memory_ids": [memory.id for memory in relevant],
                "loaded_memory_ids": self.state.loaded_memory_ids,
                "memory_namespace": self.memory_store.namespace,
            },
        )

    async def _select_long_term_memories_async(
        self,
        user_message: str,
    ) -> None:
        self._profile_memories = []
        self._relevant_memories = []
        self.state.loaded_memory_ids = []
        store = self.async_memory_store
        policy = self.async_memory_policy
        if store is None or policy is None:
            await asyncio.to_thread(
                self._select_long_term_memories,
                user_message,
            )
            return
        if not policy.retrieval_enabled:
            return
        namespace = getattr(store, "namespace", None)
        self._trace(
            TraceEvent.MEMORY_SEARCH_STARTED,
            {
                "turn_id": self.state.current_turn_id,
                "query": user_message,
                "memory_namespace": namespace,
            },
        )
        profile = await call_async_service(
            store,
            "profile_memories",
            limit=5,
        )
        relevant = await call_async_service(
            store,
            "search_memory",
            user_message,
            limit=5,
        )
        profile_ids = {memory.id for memory in profile}
        relevant = [
            memory for memory in relevant if memory.id not in profile_ids
        ]
        self._profile_memories = list(profile)
        self._relevant_memories = list(relevant)
        self.state.loaded_memory_ids = [
            memory.id for memory in [*profile, *relevant]
        ]
        self._trace(
            TraceEvent.MEMORY_SEARCH_COMPLETED,
            {
                "turn_id": self.state.current_turn_id,
                "profile_memory_ids": [memory.id for memory in profile],
                "relevant_memory_ids": [memory.id for memory in relevant],
                "loaded_memory_ids": self.state.loaded_memory_ids,
                "memory_namespace": namespace,
            },
        )

    def _select_skills(self, user_message: str) -> None:
        """Select and lazy-load procedural skills that should shape this turn."""
        self._selected_skills = []
        self.state.loaded_skill_names = []

        if self.skill_registry is None:
            return

        self._trace(
            TraceEvent.SKILL_SELECTION_STARTED,
            {"turn_id": self.state.current_turn_id, "query": user_message},
        )
        self._selected_skills = self.skill_registry.load_selected_skills(
            user_message,
            pinned_names=self.pinned_skill_names,
            limit=self.max_skills_per_turn,
        )
        self.state.loaded_skill_names = [
            selection.skill.name for selection in self._selected_skills
        ]
        routing_result = self.skill_registry.last_routing_result
        self._trace(
            TraceEvent.SKILL_SELECTION_COMPLETED,
            {
                "turn_id": self.state.current_turn_id,
                "explicit_skill_names": list(routing_result.explicit_skill_names),
                "loaded_skill_names": self.state.loaded_skill_names,
                "skills": [
                    {
                        "name": selection.skill.name,
                        "path": str(selection.skill.path),
                        "score": selection.score,
                        "matched_keywords": selection.matched_keywords,
                        "reason": selection.reason,
                        "stage": selection.stage,
                        "version": (
                            selection.skill.manifest.version
                            if selection.skill.manifest is not None
                            else None
                        ),
                        "source": (
                            selection.skill.manifest.source
                            if selection.skill.manifest is not None
                            else None
                        ),
                        "trust": (
                            selection.skill.manifest.trust
                            if selection.skill.manifest is not None
                            else None
                        ),
                        "digest": selection.skill.digest,
                        "loaded_resources": list(selection.skill.loaded_resources),
                    }
                    for selection in self._selected_skills
                ],
                "decisions": [
                    decision.to_dict() for decision in routing_result.decisions
                ],
            },
        )
        self._record_selected_skill_usage()

    async def _select_skills_async(self, user_message: str) -> None:
        self._selected_skills = []
        self.state.loaded_skill_names = []
        registry = self.async_skill_registry
        if registry is None:
            await asyncio.to_thread(self._select_skills, user_message)
            return
        self._trace(
            TraceEvent.SKILL_SELECTION_STARTED,
            {"turn_id": self.state.current_turn_id, "query": user_message},
        )
        selections = await call_async_service(
            registry,
            "load_selected_skills",
            user_message,
            pinned_names=self.pinned_skill_names,
            limit=self.max_skills_per_turn,
        )
        self._selected_skills = list(selections)
        self.state.loaded_skill_names = [
            selection.skill.name for selection in self._selected_skills
        ]
        routing_result = getattr(registry, "last_routing_result")
        self._trace(
            TraceEvent.SKILL_SELECTION_COMPLETED,
            {
                "turn_id": self.state.current_turn_id,
                "explicit_skill_names": list(
                    routing_result.explicit_skill_names
                ),
                "loaded_skill_names": self.state.loaded_skill_names,
                "skills": [
                    {
                        "name": selection.skill.name,
                        "path": str(selection.skill.path),
                        "score": selection.score,
                        "matched_keywords": selection.matched_keywords,
                        "reason": selection.reason,
                        "stage": selection.stage,
                        "version": (
                            selection.skill.manifest.version
                            if selection.skill.manifest is not None
                            else None
                        ),
                        "source": (
                            selection.skill.manifest.source
                            if selection.skill.manifest is not None
                            else None
                        ),
                        "trust": (
                            selection.skill.manifest.trust
                            if selection.skill.manifest is not None
                            else None
                        ),
                        "digest": selection.skill.digest,
                        "loaded_resources": list(
                            selection.skill.loaded_resources
                        ),
                    }
                    for selection in self._selected_skills
                ],
                "decisions": [
                    decision.to_dict()
                    for decision in routing_result.decisions
                ],
            },
        )
        await self._record_selected_skill_usage_async()

    def _record_selected_skill_usage(self) -> None:
        if self.skill_lifecycle_store is None:
            return
        turn = self.state.turns[-1] if self.state.turns else None
        if turn is None:
            return
        versions: list[dict[str, str]] = []
        for selection in self._selected_skills:
            manifest = selection.skill.manifest
            digest = selection.skill.digest
            root = selection.skill.root
            if (
                manifest is None
                or digest is None
                or root is None
                or self.skill_lifecycle is None
            ):
                continue
            resolved_root = root.resolve()
            if resolved_root.parent == self.skill_lifecycle.project_skills_dir:
                scope = "project"
            elif (
                resolved_root.parent
                == self.skill_lifecycle.profile_skills_dir
            ):
                scope = "profile"
            else:
                continue
            try:
                matched = self.skill_lifecycle_store.get_skill(
                    selection.skill.name,
                    scope=scope,
                )
                if matched.digest != digest:
                    continue
                self.skill_lifecycle_store.record_usage(
                    name=matched.name,
                    version=matched.version,
                    digest=matched.digest,
                    kind=SkillUsageKind.USE,
                    source_event_id=turn.turn_id,
                    scope=matched.scope,
                )
            except (KeyError, OSError, ValueError) as exc:
                self._trace(
                    "skill_usage_record_failed",
                    {
                        "turn_id": turn.turn_id,
                        "skill_name": selection.skill.name,
                        "scope": scope,
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            versions.append(
                {
                    "name": matched.name,
                    "scope": matched.scope,
                    "version": matched.version,
                    "digest": matched.digest,
                }
            )
        if versions:
            turn.extension_metadata["loaded_skill_versions"] = versions

    async def _record_selected_skill_usage_async(self) -> None:
        if self.skill_lifecycle_store is None:
            return
        turn = self.state.turns[-1] if self.state.turns else None
        if turn is None:
            return
        versions: list[dict[str, str]] = []
        for selection in self._selected_skills:
            manifest = selection.skill.manifest
            digest = selection.skill.digest
            root = selection.skill.root
            if (
                manifest is None
                or digest is None
                or root is None
                or self.skill_lifecycle is None
            ):
                continue
            resolved_root = root.resolve()
            if resolved_root.parent == self.skill_lifecycle.project_skills_dir:
                scope = "project"
            elif (
                resolved_root.parent
                == self.skill_lifecycle.profile_skills_dir
            ):
                scope = "profile"
            else:
                continue
            try:
                matched = await call_async_service(
                    self.skill_lifecycle_store,
                    "get_skill",
                    selection.skill.name,
                    scope=scope,
                )
                if matched.digest != digest:
                    continue
                await call_async_service(
                    self.skill_lifecycle_store,
                    "record_usage",
                    name=matched.name,
                    version=matched.version,
                    digest=matched.digest,
                    kind=SkillUsageKind.USE,
                    source_event_id=turn.turn_id,
                    scope=matched.scope,
                )
            except (KeyError, OSError, ValueError) as exc:
                self._trace(
                    "skill_usage_record_failed",
                    {
                        "turn_id": turn.turn_id,
                        "skill_name": selection.skill.name,
                        "scope": scope,
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            versions.append(
                {
                    "name": matched.name,
                    "scope": matched.scope,
                    "version": matched.version,
                    "digest": matched.digest,
                }
            )
        if versions:
            turn.extension_metadata["loaded_skill_versions"] = versions

    def confirm_skill_success(
        self,
        *,
        turn_id: str | None = None,
    ) -> tuple[SkillLifecycleRecord, ...]:
        """Record host-confirmed success for exact skill versions on one run."""
        if self.skill_lifecycle_store is None:
            raise RuntimeError("skill lifecycle is not configured")
        selected_turn = next(
            (
                turn
                for turn in reversed(self.state.turns)
                if turn_id is None or turn.turn_id == turn_id
            ),
            None,
        )
        if selected_turn is None:
            raise KeyError(f"turn {turn_id!r} does not exist")
        if selected_turn.status != "completed":
            raise ValueError("skill success requires a completed host run")
        raw_versions = selected_turn.extension_metadata.get(
            "loaded_skill_versions",
            [],
        )
        if not isinstance(raw_versions, list):
            return ()
        records: list[SkillLifecycleRecord] = []
        for item in raw_versions:
            if not isinstance(item, dict):
                continue
            required = ("name", "scope", "version", "digest")
            if not all(isinstance(item.get(key), str) for key in required):
                continue
            records.append(
                self.skill_lifecycle_store.record_usage(
                    name=item["name"],
                    scope=item["scope"],
                    version=item["version"],
                    digest=item["digest"],
                    kind=SkillUsageKind.SUCCESS,
                    source_event_id=f"{selected_turn.turn_id}:host-success",
                    host_confirmed=True,
                )
            )
        return tuple(records)

    async def confirm_skill_success_async(
        self,
        *,
        turn_id: str | None = None,
    ) -> tuple[SkillLifecycleRecord, ...]:
        """Await host-confirmed success recording for exact skill versions."""

        if self.skill_lifecycle_store is None:
            raise RuntimeError("skill lifecycle is not configured")
        selected_turn = next(
            (
                turn
                for turn in reversed(self.state.turns)
                if turn_id is None or turn.turn_id == turn_id
            ),
            None,
        )
        if selected_turn is None:
            raise KeyError(f"turn {turn_id!r} does not exist")
        if selected_turn.status != "completed":
            raise ValueError("skill success requires a completed host run")
        raw_versions = selected_turn.extension_metadata.get(
            "loaded_skill_versions",
            [],
        )
        if not isinstance(raw_versions, list):
            return ()
        records: list[SkillLifecycleRecord] = []
        for item in raw_versions:
            if not isinstance(item, dict):
                continue
            required = ("name", "scope", "version", "digest")
            if not all(isinstance(item.get(key), str) for key in required):
                continue
            records.append(
                await call_async_service(
                    self.skill_lifecycle_store,
                    "record_usage",
                    name=item["name"],
                    scope=item["scope"],
                    version=item["version"],
                    digest=item["digest"],
                    kind=SkillUsageKind.SUCCESS,
                    source_event_id=(
                        f"{selected_turn.turn_id}:host-success"
                    ),
                    host_confirmed=True,
                )
            )
        return tuple(records)

    def review_learning(
        self,
        *,
        trigger: LearningReviewTrigger | str = LearningReviewTrigger.MANUAL,
        turn_id: str | None = None,
        host_confirmed_success: bool = False,
    ) -> LearningReviewOutcome:
        """Review one completed turn without granting the reviewer tool authority."""
        if self.learning_reviewer is None:
            raise RuntimeError("learning reviewer is not configured")
        selected_turn = next(
            (
                turn
                for turn in reversed(self.state.turns)
                if turn_id is None or turn.turn_id == turn_id
            ),
            None,
        )
        if selected_turn is None:
            raise KeyError(f"turn {turn_id!r} does not exist")
        if selected_turn.final_answer is None:
            raise ValueError("learning review requires a finished turn")
        manifests = tuple(
            skill.manifest.to_dict()
            for skill in (
                self.skill_registry.list_visible_skills()
                if self.skill_registry is not None
                else ()
            )
            if skill.manifest is not None
        )
        return self.learning_reviewer.review(
            LearningReviewContext(
                trigger=LearningReviewTrigger(trigger),
                user_message=selected_turn.user_message,
                assistant_response=selected_turn.final_answer,
                turn_id=selected_turn.turn_id,
                source_trace=(
                    str(self.trace_logger.path)
                    if self.trace_logger is not None
                    else None
                ),
                tool_call_count=selected_turn.tool_call_count,
                host_confirmed_success=host_confirmed_success,
                current_skill_manifests=manifests,
            )
        )

    async def review_learning_async(
        self,
        *,
        trigger: LearningReviewTrigger | str = LearningReviewTrigger.MANUAL,
        turn_id: str | None = None,
        host_confirmed_success: bool = False,
    ) -> LearningReviewOutcome:
        """Await a restricted learning review through hosted services."""

        if self.learning_reviewer is None:
            raise RuntimeError("learning reviewer is not configured")
        selected_turn = next(
            (
                turn
                for turn in reversed(self.state.turns)
                if turn_id is None or turn.turn_id == turn_id
            ),
            None,
        )
        if selected_turn is None:
            raise KeyError(f"turn {turn_id!r} does not exist")
        if selected_turn.final_answer is None:
            raise ValueError("learning review requires a finished turn")
        visible_skills = (
            await call_async_service(
                self.async_skill_registry,
                "list_visible_skills",
            )
            if self.async_skill_registry is not None
            else ()
        )
        manifests = tuple(
            skill.manifest.to_dict()
            for skill in visible_skills
            if skill.manifest is not None
        )
        return cast(
            LearningReviewOutcome,
            await call_async_service(
                self.learning_reviewer,
                "review",
                LearningReviewContext(
                    trigger=LearningReviewTrigger(trigger),
                    user_message=selected_turn.user_message,
                    assistant_response=selected_turn.final_answer,
                    turn_id=selected_turn.turn_id,
                    source_trace=(
                        str(self.trace_logger.path)
                        if self.trace_logger is not None
                        else None
                    ),
                    tool_call_count=selected_turn.tool_call_count,
                    host_confirmed_success=host_confirmed_success,
                    current_skill_manifests=manifests,
                ),
            ),
        )

    def _turn_started_after(self, previous_turn_count: int) -> TurnState | None:
        if len(self.state.turns) <= previous_turn_count:
            return None
        return self.state.turns[-1]

    def _terminalize_exception(self, turn: TurnState, exc: BaseException) -> None:
        """Finish an active turn without masking the exception that escaped it."""
        if turn.status not in {"in_progress", "waiting_for_approval"}:
            return
        cancelled = isinstance(exc, asyncio.CancelledError)
        if cancelled:
            message = "Turn cancelled."
            turn.cancel(message)
        else:
            detail = redact_text(str(exc)).strip()
            message = f"Turn failed with {type(exc).__name__}"
            if detail:
                message = f"{message}: {detail}"
            message, _redaction_metadata = self._redact_text(
                TraceEvent.TURN_FAILED,
                message,
                {
                    "path": "exception.message",
                    "exception_type": type(exc).__name__,
                    "turn_id": turn.turn_id,
                },
            )
            message = message.strip() or "Turn failed."
            turn.fail(message)
        self.state.errors.append(message)
        self.state.final_answer = message
        self.memory.add_assistant_message(message)
        self.state.messages = self.memory.recent()
        self._plan_execution.clear(turn)
        failure_payload = {
            "turn_id": turn.turn_id,
            "message": message,
            "status": turn.status,
            "exception_type": type(exc).__name__,
            "turn": turn.to_dict(),
        }
        for event_type, payload in (
            (TraceEvent.TURN_FAILED, failure_payload),
            (TraceEvent.TURN_FINISHED, self._turn_effects.state_snapshot(turn)),
        ):
            try:
                self._trace(event_type, payload)
            except BaseException:
                # The original failure remains authoritative. A healthy sink can
                # still receive the other terminal event on the next iteration.
                continue

    def _record_model_accounting(
        self,
        turn: TurnState,
        *,
        request_index: int,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        fallback_attempts: object = None,
        purpose: str = "agent_action",
    ) -> tuple[dict | None, dict | None]:
        usage_payload = usage.to_dict() if usage is not None else None
        cost_payload = cost.to_dict() if cost is not None else None
        attempt_payloads = _fallback_attempt_payloads(fallback_attempts)
        if self.usage_accounting is not None:
            entries = self.usage_accounting.commit_model_request(
                turn_id=turn.turn_id,
                request_index=request_index,
                purpose=purpose,
                usage=usage,
                cost=cost,
                fallback_attempts=fallback_attempts,
            )
            self._trace(
                TraceEvent.BUDGET_COMMITTED,
                {
                    "turn_id": turn.turn_id,
                    "request_index": request_index,
                    "resource_kind": "model",
                    "entry_ids": [entry.id for entry in entries],
                    "source_event_ids": [
                        entry.source_event_id for entry in entries
                    ],
                },
            )
        if usage_payload is None and cost_payload is None and not attempt_payloads:
            return None, None

        report = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "purpose": purpose,
            "usage": usage_payload,
            "cost": cost_payload,
        }
        if attempt_payloads:
            report["fallback_attempts"] = attempt_payloads
        turn.model_usage_reports.append(report)
        turn.model_usage_totals = _aggregate_model_usage_reports(
            turn.model_usage_reports
        )
        self.state.last_usage_report = turn.model_usage_totals
        return usage_payload, cost_payload

    async def _record_model_accounting_async(
        self,
        turn: TurnState,
        *,
        request_index: int,
        usage: LLMUsage | None,
        cost: LLMCost | None,
        fallback_attempts: object = None,
        purpose: str = "agent_action",
    ) -> tuple[dict | None, dict | None]:
        service = self.async_usage_accounting
        if service is None:
            return await asyncio.to_thread(
                self._record_model_accounting,
                turn,
                request_index=request_index,
                usage=usage,
                cost=cost,
                fallback_attempts=fallback_attempts,
                purpose=purpose,
            )
        usage_payload = usage.to_dict() if usage is not None else None
        cost_payload = cost.to_dict() if cost is not None else None
        attempt_payloads = _fallback_attempt_payloads(fallback_attempts)
        entries = await call_async_service(
            service,
            "commit_model_request",
            turn_id=turn.turn_id,
            request_index=request_index,
            purpose=purpose,
            usage=usage,
            cost=cost,
            fallback_attempts=fallback_attempts,
        )
        self._trace(
            TraceEvent.BUDGET_COMMITTED,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "resource_kind": "model",
                "entry_ids": [entry.id for entry in entries],
                "source_event_ids": [
                    entry.source_event_id for entry in entries
                ],
            },
        )
        if usage_payload is None and cost_payload is None and not attempt_payloads:
            return None, None
        report = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "purpose": purpose,
            "usage": usage_payload,
            "cost": cost_payload,
        }
        if attempt_payloads:
            report["fallback_attempts"] = attempt_payloads
        turn.model_usage_reports.append(report)
        turn.model_usage_totals = _aggregate_model_usage_reports(
            turn.model_usage_reports
        )
        self.state.last_usage_report = turn.model_usage_totals
        return usage_payload, cost_payload

    def _reserve_model_accounting(
        self,
        turn: TurnState,
        *,
        request_index: int,
        messages: list[dict[str, str]],
        purpose: str,
        repair_attempts: int = 0,
    ) -> dict | None:
        if self.usage_accounting is None:
            return None
        try:
            reservation = self.usage_accounting.reserve_model_request(
                turn_id=turn.turn_id,
                request_index=request_index,
                messages=messages,
                purpose=purpose,
                repair_attempts=repair_attempts,
            )
        except BudgetExceededError as exc:
            payload = {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "resource_kind": "model",
                "scope": exc.scope.value,
                "dimension": exc.dimension,
                "limit": exc.limit,
                "committed": exc.committed,
                "reserved": exc.reserved,
                "requested": exc.requested,
                "message": str(exc),
            }
            turn.extension_metadata["budget_exhausted"] = payload
            self._trace(TraceEvent.BUDGET_EXHAUSTED, payload)
            raise
        payload = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "resource_kind": "model",
            "reservation_id": reservation.id,
            "scope": reservation.budget.scope.value,
            "reserved_model_calls": reservation.reserved_model_calls,
            "reserved_tokens": reservation.reserved_tokens,
            "reserved_cost": reservation.reserved_cost.to_dict(),
            "expires_at": (
                reservation.expires_at.isoformat()
                if reservation.expires_at is not None
                else None
            ),
        }
        self._trace(TraceEvent.BUDGET_RESERVED, payload)
        return payload

    async def _reserve_model_accounting_async(
        self,
        turn: TurnState,
        *,
        request_index: int,
        messages: list[dict[str, str]],
        purpose: str,
        repair_attempts: int = 0,
    ) -> dict | None:
        service = self.async_usage_accounting
        if service is None:
            return await asyncio.to_thread(
                self._reserve_model_accounting,
                turn,
                request_index=request_index,
                messages=messages,
                purpose=purpose,
                repair_attempts=repair_attempts,
            )
        try:
            reservation = await call_async_service(
                service,
                "reserve_model_request",
                turn_id=turn.turn_id,
                request_index=request_index,
                messages=messages,
                purpose=purpose,
                repair_attempts=repair_attempts,
            )
        except BudgetExceededError as exc:
            payload = {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "resource_kind": "model",
                "scope": exc.scope.value,
                "dimension": exc.dimension,
                "limit": exc.limit,
                "committed": exc.committed,
                "reserved": exc.reserved,
                "requested": exc.requested,
                "message": str(exc),
            }
            turn.extension_metadata["budget_exhausted"] = payload
            self._trace(TraceEvent.BUDGET_EXHAUSTED, payload)
            raise
        payload = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "resource_kind": "model",
            "reservation_id": reservation.id,
            "scope": reservation.budget.scope.value,
            "reserved_model_calls": reservation.reserved_model_calls,
            "reserved_tokens": reservation.reserved_tokens,
            "reserved_cost": reservation.reserved_cost.to_dict(),
            "expires_at": (
                reservation.expires_at.isoformat()
                if reservation.expires_at is not None
                else None
            ),
        }
        self._trace(TraceEvent.BUDGET_RESERVED, payload)
        return payload

    def _release_model_accounting(
        self,
        turn: TurnState,
        *,
        request_index: int,
        reason: str,
    ) -> dict | None:
        if self.usage_accounting is None:
            return None
        reservation = self.usage_accounting.release_model_request(
            turn_id=turn.turn_id,
            request_index=request_index,
        )
        if reservation is None:
            return None
        payload = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "resource_kind": "model",
            "reservation_id": reservation.id,
            "reason": reason,
        }
        self._trace(TraceEvent.BUDGET_RELEASED, payload)
        return payload

    async def _release_model_accounting_async(
        self,
        turn: TurnState,
        *,
        request_index: int,
        reason: str,
    ) -> dict | None:
        service = self.async_usage_accounting
        if service is None:
            return await asyncio.to_thread(
                self._release_model_accounting,
                turn,
                request_index=request_index,
                reason=reason,
            )
        reservation = await call_async_service(
            service,
            "release_model_request",
            turn_id=turn.turn_id,
            request_index=request_index,
        )
        if reservation is None:
            return None
        payload = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "resource_kind": "model",
            "reservation_id": reservation.id,
            "reason": reason,
        }
        self._trace(TraceEvent.BUDGET_RELEASED, payload)
        return payload

    def _trace(self, event_type: str, payload: dict | None = None) -> None:
        payload = dict(payload or {})
        payload = self._redact_event_payload(event_type, payload)
        if self.trace_logger is not None:
            self.trace_logger.log(event_type, payload)
        if self.event_callback is not None:
            self.event_callback(event_type, payload)
        if self.audit_callback is not None:
            self.audit_callback(event_type, payload)
        if self.event_sink is not None:
            self.event_sink(AgentEvent(event_type, payload))

    def _redact_text(
        self, event_type: str, text: str, metadata: dict
    ) -> tuple[str, dict]:
        if self.redaction_callback is None:
            return text, {"redacted": False}
        try:
            redacted = self.redaction_callback(event_type, text, metadata)
        except Exception as exc:
            if self.redaction_fail_closed:
                return "[redaction failed]", {
                    "redacted": True,
                    "redaction_error": str(exc),
                    "fail_closed": True,
                }
            return text, {
                "redacted": False,
                "redaction_error": str(exc),
                "fail_closed": False,
            }
        if not isinstance(redacted, str):
            redacted = str(redacted)
        return redacted, {"redacted": redacted != text}

    def _redact_event_payload(self, event_type: str, payload: dict) -> dict:
        if self.redaction_callback is None:
            return payload

        redacted_any = False
        error: str | None = None

        def redact_value(value: object, path: str) -> object:
            nonlocal redacted_any, error
            if isinstance(value, str):
                redacted, metadata = self._redact_text(
                    event_type, value, {"path": path}
                )
                redacted_any = redacted_any or bool(metadata.get("redacted"))
                if metadata.get("redaction_error"):
                    error = str(metadata["redaction_error"])
                    redacted_any = redacted_any or bool(metadata.get("fail_closed"))
                return redacted
            if isinstance(value, dict):
                return {
                    key: redact_value(item, f"{path}.{key}")
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [
                    redact_value(item, f"{path}[{index}]")
                    for index, item in enumerate(value)
                ]
            return value

        redacted_payload = redact_value(payload, "payload")
        if not isinstance(redacted_payload, dict):
            return payload
        if redacted_any:
            redacted_payload["_redacted"] = True
        if error is not None:
            redacted_payload["_redaction_error"] = error
        return redacted_payload

    def _tool_context_for_turn(self, turn: TurnState) -> ToolExecutionContext | None:
        if turn.turn_id in self._tool_contexts:
            return self._tool_contexts[turn.turn_id]
        default_context = self.default_tool_context
        if (
            not turn.tool_context_metadata
            and default_context is None
            and self.tool_context_lifecycle is None
        ):
            return None
        context = ToolExecutionContext(
            metadata={
                **(default_context.metadata if default_context is not None else {}),
                **turn.tool_context_metadata,
                "conversation_id": self.state.conversation_id,
                "turn_id": turn.turn_id,
            },
            deps=default_context.deps if default_context is not None else None,
            execution_session=(
                default_context.execution_session
                if default_context is not None
                else None
            ),
        )
        if (
            self.tool_context_lifecycle is not None
            and context.execution_session is None
        ):
            context = self.tool_context_lifecycle.open(context)
        self._tool_contexts[turn.turn_id] = context
        return context

    async def _tool_context_for_turn_async(
        self,
        turn: TurnState,
    ) -> ToolExecutionContext | None:
        if turn.turn_id in self._tool_contexts:
            return self._tool_contexts[turn.turn_id]
        default_context = self.default_tool_context
        if (
            not turn.tool_context_metadata
            and default_context is None
            and self.tool_context_lifecycle is None
        ):
            return None
        context = ToolExecutionContext(
            metadata={
                **(
                    default_context.metadata
                    if default_context is not None
                    else {}
                ),
                **turn.tool_context_metadata,
                "conversation_id": self.state.conversation_id,
                "turn_id": turn.turn_id,
            },
            deps=(
                default_context.deps if default_context is not None else None
            ),
            execution_session=(
                default_context.execution_session
                if default_context is not None
                else None
            ),
        )
        if (
            self.tool_context_lifecycle is not None
            and context.execution_session is None
        ):
            context = await self.tool_context_lifecycle.open_async(context)
        self._tool_contexts[turn.turn_id] = context
        return context

    def _release_tool_context(self, turn: TurnState) -> None:
        """Release request-scoped host dependencies after terminal work."""
        context = self._tool_contexts.pop(turn.turn_id, None)
        if context is not None and self.tool_context_lifecycle is not None:
            self.tool_context_lifecycle.close(context)

    async def _release_tool_context_async(self, turn: TurnState) -> None:
        """Release request-scoped host dependencies after terminal async work."""
        context = self._tool_contexts.pop(turn.turn_id, None)
        if context is not None and self.tool_context_lifecycle is not None:
            await self.tool_context_lifecycle.aclose(context)

    def _write_tool_output_artifact(self, name: str, content: str) -> dict | None:
        if self.trace_logger is None:
            return None
        return self.trace_logger.write_artifact(name, content)

    async def _write_tool_output_artifact_async(
        self,
        name: str,
        content: str,
    ) -> dict | None:
        store = self.async_artifact_store
        if store is None:
            return await asyncio.to_thread(
                self._write_tool_output_artifact,
                name,
                content,
            )
        record = await call_async_service(store, "write", name, content)
        if record is None:
            return None
        if isinstance(record, dict):
            return record
        reference = getattr(record, "reference", None)
        if callable(reference):
            return cast(dict, reference())
        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            return cast(dict, to_dict())
        raise TypeError("async artifact store returned an unsupported record")

    async def _flush_async_services(self) -> None:
        for service in self.async_flushables:
            await call_async_service(service, "flush")

    async def _flush_async_services_after_error(
        self,
        original: BaseException,
    ) -> None:
        await _await_cleanup_after_error(
            self._flush_async_services(),
            original,
        )


async def _await_cleanup_after_error(
    cleanup: Awaitable[object],
    original: BaseException,
) -> None:
    """Complete required cleanup without replacing the triggering exception."""

    if not isinstance(original, asyncio.CancelledError):
        try:
            await cleanup
        except BaseException as cleanup_error:
            original.add_note(
                "async cleanup also failed with "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        return

    task = asyncio.ensure_future(cleanup)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except BaseException as cleanup_error:
            original.add_note(
                "async cleanup also failed with "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
    except BaseException as cleanup_error:
        original.add_note(
            "async cleanup also failed with "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )


def _fallback_attempt_payloads(fallback_attempts: object) -> list[dict]:
    if not fallback_attempts:
        return []
    payloads: list[dict] = []
    if not isinstance(fallback_attempts, list):
        return payloads
    for attempt in fallback_attempts:
        if hasattr(attempt, "to_dict"):
            payload = attempt.to_dict()
        elif isinstance(attempt, dict):
            payload = dict(attempt)
        else:
            payload = {"attempt": str(attempt)}
        payloads.append(payload)
    return payloads


def _aggregate_model_usage_reports(reports: list[dict]) -> dict:
    request_usages: list[LLMUsage | None] = []
    request_costs: list[LLMCost | None] = []
    failed_attempt_usages: list[LLMUsage | None] = []
    failed_attempt_costs: list[LLMCost | None] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        request_usages.append(usage_from_dict(report.get("usage")))
        request_costs.append(cost_from_dict(report.get("cost")))
        attempts = report.get("fallback_attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("success") is not False:
                continue
            failed_attempt_usages.append(usage_from_dict(attempt.get("usage")))
            failed_attempt_costs.append(cost_from_dict(attempt.get("cost")))

    usage = aggregate_usage(
        [*request_usages, *failed_attempt_usages], source="turn_total"
    )
    cost = aggregate_cost([*request_costs, *failed_attempt_costs])
    return {
        "request_count": len(
            [report for report in reports if isinstance(report, dict)]
        ),
        "usage": usage.to_dict() if usage is not None else None,
        "cost": cost.to_dict() if cost is not None else None,
    }


def _coerce_turn_context_sections(
    values: list[TurnContextSection | dict[str, Any] | str] | None,
) -> list[TurnContextSection]:
    if not values:
        return []
    sections: list[TurnContextSection] = []
    for index, value in enumerate(values, start=1):
        if isinstance(value, TurnContextSection):
            sections.append(value)
            continue
        if isinstance(value, str):
            sections.append(TurnContextSection(id=f"context-{index}", content=value))
            continue
        if isinstance(value, dict):
            content = value.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            section_id = value.get("id") or value.get("source_id") or f"context-{index}"
            raw_metadata = value.get("metadata")
            metadata = (
                cast(dict[str, Any], raw_metadata)
                if isinstance(raw_metadata, dict)
                else {}
            )
            sections.append(
                TurnContextSection(
                    id=str(section_id),
                    title=value.get("title")
                    if isinstance(value.get("title"), str)
                    else None,
                    source=value.get("source")
                    if isinstance(value.get("source"), str)
                    else None,
                    content=content,
                    metadata=metadata,
                )
            )
    return sections


def _coerce_tool_execution_context(
    value: ToolExecutionContext | dict | None,
) -> ToolExecutionContext | None:
    if value is None:
        return None
    if isinstance(value, ToolExecutionContext):
        return value
    return ToolExecutionContext(metadata=value)
