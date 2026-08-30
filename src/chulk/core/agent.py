"""Public agent composition, lifecycle, memory selection, and resource ownership."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import threading
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from chulk.core.action_loop import run_action_loop, run_action_loop_async
from chulk.core.action_runtime import (
    ActionLoopRuntime,
    AgentRuntimeComponents,
    AgentTurnCancelled,
)
from chulk.core.async_cleanup import await_cleanup_after_error
from chulk.core.context import ContextBudget, TurnContextSection
from chulk.core.events import RuntimeEventDispatcher, TraceEvent
from chulk.core.model_accounting import ModelAccounting
from chulk.core.model_transport import ModelTransport
from chulk.core.plan_execution import (
    PlanExecution,
)
from chulk.core.planning import read_only_planning_tool_names
from chulk.core.prompts import BASE_SYSTEM_PROMPT
from chulk.core.state import AgentState, TurnState
from chulk.core.signals import DurableApprovalPaused
from chulk.core.tool_execution import ToolExecutor
from chulk.core.turn_effects import TurnEffects
from chulk.llm.lifecycle import aclose_resources, close_resources
from chulk.hosting.async_utils import call_async_service
from chulk.hosting.transcripts import (
    TranscriptRuntime,
)
from chulk.hosting.tool_catalog import (
    ToolCatalogRuntime,
)
from chulk.media import (
    MediaInputPart,
    MediaProcessorRegistry,
    TextInputPart,
    UserInput,
)
from chulk.memory import (
    ConversationMemory,
    MemoryPolicy,
)
from chulk.memory.context import MemoryContextService
from chulk.skills.registry import SkillContextService
from chulk.skills.reviewer import LearningRuntime
from chulk.tools import ToolRegistry
from chulk.tools.permissions import (
    ToolPermissionPolicy,
)
from chulk.tools.registry import ToolExecutionContext
from chulk.core.tool_context import ToolContextRuntime
from chulk.core.resource_lifecycle import RuntimeResourceLifecycle
from chulk.redaction import redact_text
from chulk.resources import HostResource, deduplicate_resources
from chulk.streaming import (
    FinalAnswerStreamingMode,
    OutputPolicyFailureMode,
)

if TYPE_CHECKING:
    pass


class Agent:
    """Coordinates model calls, memory retrieval, skill loading, and tools."""

    def __init__(self, components: AgentRuntimeComponents) -> None:
        self._components = components
        llm_client = components.llm_client
        state = components.state
        memory = components.memory
        memory_store = components.memory_store
        memory_policy = components.memory_policy
        skill_registry = components.skill_registry
        tool_registry = components.tool_registry
        trace_logger = components.trace_logger
        system_prompt = components.system_prompt or BASE_SYSTEM_PROMPT
        max_tool_calls_per_turn = components.max_tool_calls_per_turn
        max_json_repair_attempts = components.max_json_repair_attempts
        max_skills_per_turn = components.max_skills_per_turn
        max_skill_content_chars = components.max_skill_content_chars
        trace_max_prompt_chars = components.trace_max_prompt_chars
        max_observation_chars = components.max_observation_chars
        max_tool_stdout_chars = components.max_tool_stdout_chars
        max_tool_stderr_chars = components.max_tool_stderr_chars
        max_reflection_attempts = components.max_reflection_attempts
        permission_policy = components.permission_policy
        permission_callback = components.permission_callback
        plan_step_verifier = components.plan_step_verifier
        async_plan_step_verifier = components.async_plan_step_verifier
        context_budget = components.context_budget
        max_model_output_tokens = components.max_model_output_tokens
        stream_idle_timeout_seconds = components.stream_idle_timeout_seconds
        event_callback = components.event_callback
        event_sink = components.event_sink
        audit_callback = components.audit_callback
        redaction_callback = components.redaction_callback
        redaction_fail_closed = components.redaction_fail_closed
        final_answer_streaming = components.final_answer_streaming
        output_policy = components.output_policy
        async_output_policy = components.async_output_policy
        output_policy_failure_mode = components.output_policy_failure_mode
        pinned_skill_names = components.pinned_skill_names
        mcp_servers = components.mcp_servers
        mcp_bridge_tool_names = components.mcp_bridge_tool_names
        owned_resources = components.owned_resources
        default_tool_context = components.default_tool_context
        runtime_metadata = components.runtime_metadata
        tool_context_lifecycle = components.tool_context_lifecycle
        profile_id = components.profile_id
        usage_accounting = components.usage_accounting
        skill_lifecycle_store = components.skill_lifecycle_store
        skill_lifecycle = components.skill_lifecycle
        learning_proposals = components.learning_proposals
        learning_reviewer = components.learning_reviewer
        plugin_registry = components.plugin_registry
        plugin_audit_report = components.plugin_audit_report
        goal_execution = components.goal_execution
        content_store = components.content_store
        media_processors = components.media_processors
        execution_scope = components.execution_scope
        tool_policy_hooks = components.tool_policy_hooks
        transcript_resolver = components.transcript_resolver
        async_transcript_resolver = components.async_transcript_resolver
        transcript_timeout_seconds = components.transcript_timeout_seconds
        tool_catalog_resolver = components.tool_catalog_resolver
        async_tool_catalog_resolver = components.async_tool_catalog_resolver
        tool_catalog_timeout_seconds = components.tool_catalog_timeout_seconds
        close_trace_logger = components.close_trace_logger
        restore_plan_context = components.restore_plan_context
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
        if stream_idle_timeout_seconds is not None and stream_idle_timeout_seconds <= 0:
            raise ValueError("stream_idle_timeout_seconds must be greater than zero")
        self.stream_idle_timeout_seconds = stream_idle_timeout_seconds
        self.profile_id = profile_id
        self.llm_client = llm_client
        self.state = state or AgentState()
        self.memory = memory or ConversationMemory()
        resolved_memory_policy = memory_policy or (
            MemoryPolicy(memory_store, "automatic")
            if memory_store is not None
            else None
        )
        resolved_tool_registry = tool_registry or ToolRegistry()
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
        resolved_permission_policy = permission_policy or ToolPermissionPolicy()
        self.events = components.event_dispatcher or RuntimeEventDispatcher(
            trace_logger=trace_logger,
            event_callback=event_callback,
            event_sink=event_sink,
            audit_callback=audit_callback,
            public_event_sink=components.public_event_sink,
            redaction_callback=redaction_callback,
            redaction_fail_closed=redaction_fail_closed,
        )
        self._trace = self.events.emit
        self.final_answer_streaming = FinalAnswerStreamingMode(final_answer_streaming)
        self.output_policy_failure_mode = OutputPolicyFailureMode(
            output_policy_failure_mode
        )
        self.pinned_skill_names = pinned_skill_names or []
        self.mcp_servers = tuple(mcp_servers or ())
        self.mcp_bridge_tool_names = list(mcp_bridge_tool_names or [])
        self._owned_resources = list(owned_resources or [])
        self._closed = False
        self._cancel_requested = threading.Event()
        self.default_tool_context = default_tool_context
        self.runtime_metadata = deepcopy(runtime_metadata or {})
        self.execution_scope = execution_scope
        self.transcripts = TranscriptRuntime(
            state=self.state,
            memory=self.memory,
            execution_scope=execution_scope,
            resolver=transcript_resolver,
            async_resolver=async_transcript_resolver,
            timeout_seconds=transcript_timeout_seconds,
        )
        self.catalog = ToolCatalogRuntime(
            registry=resolved_tool_registry,
            state=self.state,
            execution_scope=execution_scope,
            resolver=tool_catalog_resolver,
            async_resolver=async_tool_catalog_resolver,
            timeout_seconds=tool_catalog_timeout_seconds,
        )
        self._close_trace_logger = close_trace_logger
        self.resolved_services = components.resolved_services
        self.plugin_registry = plugin_registry
        self.plugin_audit_report = plugin_audit_report
        self.goal_execution = goal_execution
        self.content_store = content_store
        self.media_processors = media_processors or MediaProcessorRegistry()
        self.tool_contexts = ToolContextRuntime(
            conversation_id=lambda: self.state.conversation_id,
            default_context=default_tool_context,
            lifecycle=tool_context_lifecycle,
        )
        self.resources = components.resource_lifecycle or RuntimeResourceLifecycle(
            trace_logger=trace_logger,
            async_artifact_store=components.async_artifact_store,
            async_flushables=components.async_flushables,
        )
        self.async_event_buffer = components.async_event_buffer
        self._model_accounting = ModelAccounting(
            state=self.state,
            trace=self.events.emit,
            usage_accounting=usage_accounting,
            async_usage_accounting=components.async_usage_accounting,
        )
        self.memory_context = MemoryContextService(
            state=self.state,
            store=memory_store,
            policy=resolved_memory_policy,
            async_store=components.async_memory_store,
            async_policy=components.async_memory_policy,
            trace=self.events.emit,
        )
        self.skill_context = SkillContextService(
            state=self.state,
            registry=skill_registry,
            lifecycle_store=skill_lifecycle_store,
            lifecycle=skill_lifecycle,
            async_registry=components.async_skill_registry,
            pinned_names=self.pinned_skill_names,
            limit=self.max_skills_per_turn,
            trace=self.events.emit,
        )
        self.learning = LearningRuntime(
            state=self.state,
            reviewer=learning_reviewer,
            registry=skill_registry,
            async_registry=components.async_skill_registry,
            trace_logger=self.trace_logger,
            proposals=learning_proposals,
        )
        if restore_plan_context:
            self._restore_plan_turn_context()
        self.state.conversation_summary = self.memory.conversation_summary
        self._tool_executor = ToolExecutor(
            registry=resolved_tool_registry,
            permission_policy=resolved_permission_policy,
            permission_callback=permission_callback,
            trace=self.events.emit,
            get_context=self.tool_contexts.get,
            usage_accounting=usage_accounting,
            goal_execution=self.goal_execution,
            execution_scope=self.execution_scope,
            policy_hooks=tool_policy_hooks,
            get_context_async=self.tool_contexts.get_async,
            async_usage_accounting=components.async_usage_accounting,
            flush_async=self.resources.flush,
        )
        self._plan_execution = PlanExecution(
            state=self.state,
            memory=self.memory,
            trace=self.events.emit,
            verifier=plan_step_verifier,
            async_verifier=async_plan_step_verifier,
        )
        self._turn_effects = TurnEffects(
            state=self.state,
            memory=self.memory,
            llm_client=self.llm_client,
            plan=self._plan_execution,
            trace=self.events.emit,
            redact_text=self.events.redact_text,
            artifact_writer=self.resources.write_artifact,
            planning_tool_names=lambda: read_only_planning_tool_names(
                resolved_tool_registry.list_tools()
            ),
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            max_reflection_attempts=self.max_reflection_attempts,
            max_observation_chars=self.max_observation_chars,
            max_tool_stdout_chars=self.max_tool_stdout_chars,
            max_tool_stderr_chars=self.max_tool_stderr_chars,
            async_artifact_writer=self.resources.write_artifact_async,
        )
        self._model_transport = ModelTransport(
            llm_client=self.llm_client,
            state=self.state,
            memory=self.memory,
            tool_registry=resolved_tool_registry,
            system_prompt=self.system_prompt,
            context_budget=self.context_budget,
            skill_registry=skill_registry,
            get_profile_memories=lambda: self.memory_context.profile_memories,
            get_relevant_memories=lambda: self.memory_context.relevant_memories,
            get_selected_skills=lambda: self.skill_context.selections,
            trace=self.events.emit,
            record_accounting=self._model_accounting.record,
            reserve_accounting=self._model_accounting.reserve,
            release_accounting=self._model_accounting.release,
            resolve_mcp_approval=self._tool_executor.resolve_hosted_mcp_approval,
            mcp_servers=self.mcp_servers,
            mcp_bridge_tool_names=tuple(self.mcp_bridge_tool_names),
            max_skill_content_chars=self.max_skill_content_chars,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            max_json_repair_attempts=self.max_json_repair_attempts,
            max_reflection_attempts=self.max_reflection_attempts,
            trace_max_prompt_chars=self.trace_max_prompt_chars,
            max_output_tokens=self.max_model_output_tokens,
            stream_idle_timeout_seconds=self.stream_idle_timeout_seconds,
            record_accounting_async=self._model_accounting.record_async,
            reserve_accounting_async=self._model_accounting.reserve_async,
            release_accounting_async=self._model_accounting.release_async,
            flush_async=self.resources.flush,
            redact_text=self.events.redact_text,
            output_policy=output_policy,
            async_output_policy=async_output_policy,
            output_policy_failure_mode=self.output_policy_failure_mode,
        )
        self._turn_effects.final_answer_streaming = self.final_answer_streaming
        self._turn_effects.stream_final_answer = (
            self._model_transport.stream_final_answer
        )
        self._turn_effects.stream_final_answer_async = (
            self._model_transport.stream_final_answer_async
        )
        self._action_runtime = ActionLoopRuntime(
            model=self._model_transport,
            tools=self._tool_executor,
            effects=self._turn_effects,
            async_flush=self.resources.flush,
            cancelled=self._cancel_requested.is_set,
        )
        self.catalog.add_registry_consumer(self._tool_executor.set_registry)
        self.catalog.add_registry_consumer(self._model_transport.set_tool_registry)

    @property
    def closed(self) -> bool:
        """Return whether the runtime has been finalized."""
        return self._closed

    def close(self) -> None:
        """Finalize owned closeable resources exactly once."""
        if self._closed:
            return
        self._cancel_requested.set()
        self._closed = True
        failures = self.tool_contexts.close()
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
        self.events.clear_callbacks()
        if failures:
            raise RuntimeError(
                f"Failed to close {len(failures)} owned agent resource(s)"
            ) from failures[0]

    async def aclose(self) -> None:
        """Finalize owned closeable resources exactly once from an async host."""
        if self._closed:
            return
        self._cancel_requested.set()
        failures = await self.tool_contexts.aclose()
        while self._owned_resources:
            resource = self._owned_resources[-1]
            try:
                await aclose_resources((resource,))
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
            self._owned_resources.pop()
        if self.trace_logger is not None and self._close_trace_logger:
            try:
                self.trace_logger.close()
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        self.events.clear_callbacks()
        self._closed = True
        if failures:
            raise RuntimeError(
                f"Failed to close {len(failures)} owned agent resource(s)"
            ) from failures[0]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Agent is closed")

    def cancel_active_turn(self) -> bool:
        """Request cooperative cancellation and interrupt the active model transport."""
        active = any(turn.status == "in_progress" for turn in self.state.turns)
        if not active:
            return False
        self._cancel_requested.set()
        close_resources((self._model_transport.llm_client,))
        return True

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
        projection, prepared, media_sections, media_events = self._prepare_user_input(
            user_input, turn_id=turn_id
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
        (
            projection,
            prepared,
            media_sections,
            media_events,
        ) = await self._prepare_user_input_async(
            user_input,
            turn_id=turn_id,
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
        self._cancel_requested.clear()
        effective_turn_id = turn_id or str(uuid4())
        if self.transcripts.enabled:
            snapshot = self.transcripts.resolve(effective_turn_id)
            extension_metadata = self.transcripts.apply(
                snapshot,
                extension_metadata,
            )
        catalog = self.catalog.resolve(
            clean_message,
            turn_id=effective_turn_id,
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata,
        )
        extension_metadata = self.catalog.activate(
            catalog,
            extension_metadata,
        )
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
                turn_id=effective_turn_id,
            )
            if isinstance(turn_or_response, str):
                return turn_or_response
            turn = turn_or_response
            result = self._run_action_loop(turn, require_plan=require_plan)
        except DurableApprovalPaused:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                turn.status = "waiting_for_approval"
                self.tool_contexts.release(turn)
            raise
        except BaseException as exc:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                self._terminalize_exception(turn, exc)
                self.tool_contexts.release(turn)
            raise
        if turn.status != "waiting_for_approval":
            self.tool_contexts.release(turn)
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
        self._cancel_requested.clear()
        effective_turn_id = turn_id or str(uuid4())
        if self.transcripts.enabled:
            snapshot = await self.transcripts.resolve_async(effective_turn_id)
            extension_metadata = self.transcripts.apply(
                snapshot,
                extension_metadata,
            )
        catalog = await self.catalog.resolve_async(
            clean_message,
            turn_id=effective_turn_id,
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata,
        )
        extension_metadata = self.catalog.activate(
            catalog,
            extension_metadata,
        )
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
                turn_id=effective_turn_id,
            )
            if isinstance(turn_or_response, str):
                return turn_or_response
            turn = turn_or_response
            result = await self._run_action_loop_async(turn, require_plan=require_plan)
        except DurableApprovalPaused as exc:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                turn.status = "waiting_for_approval"
                await self.resources.flush_after_error(exc)
                await await_cleanup_after_error(
                    self.tool_contexts.release_async(turn),
                    exc,
                )
            raise
        except BaseException as exc:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                self._terminalize_exception(turn, exc)
                await self.resources.flush_after_error(exc)
                await await_cleanup_after_error(
                    self.tool_contexts.release_async(turn),
                    exc,
                )
            raise
        if turn.status != "waiting_for_approval":
            await self.tool_contexts.release_async(turn)
        await self.resources.flush()
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
            _coerce_tool_execution_context(tool_context) or self.default_tool_context
        )
        turn = TurnState(
            user_message=clean_message,
            turn_id=turn_id or str(uuid4()),
            available_tool_names=[
                tool.name for tool in self.catalog.active_registry.list_tools()
            ],
            context_sections=turn_context_sections,
            resources=list(_context_resources(turn_context_sections)),
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
        await self.tool_contexts.prepare_async(turn, execution_context)
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
        for resource in turn.resources:
            self._trace(
                TraceEvent.HOST_RESOURCE_AVAILABLE,
                {
                    "turn_id": turn.turn_id,
                    "origin": "context",
                    "resource": resource.to_dict(),
                },
            )

        await self.memory_context.extract_async(clean_message)
        await self.memory_context.select_async(clean_message)
        await self.skill_context.select_async(clean_message)
        turn.extracted_memory_ids = list(self.state.extracted_memory_ids)
        turn.loaded_memory_ids = list(self.state.loaded_memory_ids)
        turn.loaded_skill_names = list(self.state.loaded_skill_names)
        self.memory.add_user_message(clean_message)
        self._trace(
            TraceEvent.USER_MESSAGE,
            {"turn_id": turn.turn_id, "content": clean_message},
        )
        await self.resources.flush()
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
                tool.name for tool in self.catalog.active_registry.list_tools()
            ],
            context_sections=turn_context_sections,
            resources=list(_context_resources(turn_context_sections)),
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
        self.tool_contexts.prepare(turn, execution_context)
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
        for resource in turn.resources:
            self._trace(
                TraceEvent.HOST_RESOURCE_AVAILABLE,
                {
                    "turn_id": turn.turn_id,
                    "origin": "context",
                    "resource": resource.to_dict(),
                },
            )

        self.memory_context.extract(clean_message)
        self.memory_context.select(clean_message)
        self.skill_context.select(clean_message)
        turn.extracted_memory_ids = list(self.state.extracted_memory_ids)
        turn.loaded_memory_ids = list(self.state.loaded_memory_ids)
        turn.loaded_skill_names = list(self.state.loaded_skill_names)
        self.memory.add_user_message(clean_message)
        self._trace(
            TraceEvent.USER_MESSAGE, {"turn_id": turn.turn_id, "content": clean_message}
        )

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
        projection = user_input.textual_projection(max_chars=self.max_observation_chars)
        prepared_parts: list[TextInputPart] = []
        sections: list[TurnContextSection] = []
        events: list[dict] = []
        usage_accounting = self._model_accounting.usage_accounting
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
            if usage_accounting is not None:
                usage_reservation = usage_accounting.reserve_media_transform(
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
                if usage_accounting is not None and usage_reservation is not None:
                    usage_accounting.release_media_transform(usage_reservation)
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
            if usage_accounting is not None and usage_reservation is not None:
                usage_accounting.commit_media_transform(
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
        content_store = self._components.async_content_store
        processors = self._components.async_media_processors
        if content_store is None or processors is None:
            return await asyncio.to_thread(
                self._prepare_user_input,
                user_input,
                turn_id=turn_id,
            )
        projection = user_input.textual_projection(max_chars=self.max_observation_chars)
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
                raise ValueError("typed media metadata does not match stored content")
            processor, transform = await call_async_service(
                processors,
                "choose",
                item,
            )
            capability = processor.capability
            usage_reservation = None
            if self._model_accounting.async_usage_accounting is not None:
                usage_reservation = await call_async_service(
                    self._model_accounting.async_usage_accounting,
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
                        "Native media transforms require a model client media adapter."
                    )
                bounded_text = result.text[: self.max_observation_chars]
                sections.append(
                    TurnContextSection(
                        id=(
                            "media-" + item.content_ref.id.removeprefix("content:")[:16]
                        ),
                        title=(f"Media transform: {item.file_name or item.kind.value}"),
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
                    self._model_accounting.async_usage_accounting is not None
                    and usage_reservation is not None
                ):
                    await call_async_service(
                        self._model_accounting.async_usage_accounting,
                        "commit_media_transform",
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
                    usage_reservation = None
            except BaseException as exc:
                if (
                    self._model_accounting.async_usage_accounting is not None
                    and usage_reservation is not None
                ):
                    await await_cleanup_after_error(
                        call_async_service(
                            self._model_accounting.async_usage_accounting,
                            "release_media_transform",
                            usage_reservation,
                        ),
                        exc,
                    )
                raise
        if not prepared_parts:
            prepared_parts.append(TextInputPart(projection, external_content=True))
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
        self._cancel_requested.clear()
        turn = self._pending_plan_turn() or self._resumable_plan_turn()
        if turn is not None:
            self.transcripts.revalidate(turn)
            self.catalog.revalidate(turn)
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
                self.tool_contexts.release(turn)

    async def approve_plan_async(self) -> str:
        """Approve the pending plan and continue it with async tool execution."""
        self._ensure_open()
        self._cancel_requested.clear()
        turn = self._pending_plan_turn() or self._resumable_plan_turn()
        if turn is not None:
            await self.transcripts.revalidate_async(turn)
            await self.catalog.revalidate_async(turn)
        try:
            turn_or_response = self._prepare_plan_execution()
            if isinstance(turn_or_response, str):
                result = turn_or_response
            else:
                turn = turn_or_response
                await self.resources.flush()
                result = await self._run_action_loop_async(
                    turn,
                    require_plan=False,
                )
        except BaseException as exc:
            if turn is not None:
                self._terminalize_exception(turn, exc)
                await self.resources.flush_after_error(exc)
                await await_cleanup_after_error(
                    self.tool_contexts.release_async(turn),
                    exc,
                )
            raise
        if turn is not None:
            await self.tool_contexts.release_async(turn)
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
            self.tool_contexts.release(turn)

    async def reject_plan_async(self) -> str:
        """Reject or cancel a plan without entering a synchronous host path."""

        self._ensure_open()
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
            await self.resources.flush()
        except BaseException as exc:
            await await_cleanup_after_error(
                self.tool_contexts.release_async(turn),
                exc,
            )
            raise
        await self.tool_contexts.release_async(turn)
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
        return run_action_loop(self._action_runtime, turn, require_plan=require_plan)

    async def _run_action_loop_async(
        self, turn: TurnState, *, require_plan: bool
    ) -> str:
        """Run model/tool iterations, awaiting async tool calls."""
        return await run_action_loop_async(
            self._action_runtime,
            turn,
            require_plan=require_plan,
        )

    def _restore_plan_turn_context(self) -> None:
        """Restore context that shaped a pending or resumable approved plan."""
        turn = self._pending_plan_turn() or self._resumable_plan_turn()
        if turn is None:
            return
        self.skill_context.restore(turn)
        self.memory_context.restore(turn)

    async def restore_plan_turn_context_async(self) -> None:
        """Restore persisted plan context through native async services."""

        turn = self._pending_plan_turn() or self._resumable_plan_turn()
        if turn is None:
            return
        await self.skill_context.restore_async(turn)
        await self.memory_context.restore_async(turn)

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

    def _turn_started_after(self, previous_turn_count: int) -> TurnState | None:
        if len(self.state.turns) <= previous_turn_count:
            return None
        return self.state.turns[-1]

    def _terminalize_exception(self, turn: TurnState, exc: BaseException) -> None:
        """Finish an active turn without masking the exception that escaped it."""
        if turn.status not in {"in_progress", "waiting_for_approval"}:
            return
        cancelled = (
            isinstance(exc, (asyncio.CancelledError, AgentTurnCancelled))
            or self._cancel_requested.is_set()
        )
        if cancelled:
            message = "Turn cancelled."
            turn.cancel(message)
        else:
            detail = redact_text(str(exc)).strip()
            message = f"Turn failed with {type(exc).__name__}"
            if detail:
                message = f"{message}: {detail}"
            message, _redaction_metadata = self.events.redact_text(
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
                    persist_content=bool(value.get("persist_content", True)),
                    resource=(
                        HostResource.from_dict(value["resource"])
                        if isinstance(value.get("resource"), dict)
                        else None
                    ),
                )
            )
    return sections


def _context_resources(
    sections: list[TurnContextSection],
) -> tuple[HostResource, ...]:
    return deduplicate_resources(
        section.resource for section in sections if section.resource is not None
    )


def _coerce_tool_execution_context(
    value: ToolExecutionContext | dict | None,
) -> ToolExecutionContext | None:
    if value is None:
        return None
    if isinstance(value, ToolExecutionContext):
        return value
    return ToolExecutionContext(metadata=value)
