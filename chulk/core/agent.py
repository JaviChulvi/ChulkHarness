"""Public agent composition, lifecycle, memory selection, and resource ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

from chulk.core.action_loop import run_action_loop, run_action_loop_async
from chulk.core.action_runtime import ActionLoopRuntime
from chulk.core.context import ContextBudget, TurnContextSection
from chulk.core.events import AgentEvent, TraceEvent
from chulk.core.model_transport import ModelTransport
from chulk.core.plan_execution import PlanExecution
from chulk.core.planning import read_only_planning_tool_names
from chulk.core.prompts import BASE_SYSTEM_PROMPT
from chulk.core.state import AgentState, TurnState
from chulk.core.tool_execution import ToolExecutor
from chulk.core.turn_effects import TurnEffects
from chulk.llm import LLMCost, LLMClient, LLMUsage
from chulk.llm.capabilities import client_requires_mcp_bridge
from chulk.llm.usage import aggregate_cost, aggregate_usage, cost_from_dict, usage_from_dict
from chulk.mcp import MCPServerConfig
from chulk.memory.constants import PROFILE_MEMORY_TAGS
from chulk.memory import (
    ConversationMemory,
    MemoryPolicy,
    MemoryRecord,
    SQLiteMemoryStore,
    route_memory_candidates,
    select_memories_for_prompt,
)
from chulk.skills import SkillRegistry, SkillSelection
from chulk.tools import ToolRegistry
from chulk.tools.permissions import (
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    ToolPermissionPolicy,
)
from chulk.tools.registry import ToolExecutionContext
from chulk.tracing import JSONLTraceLogger
from chulk.redaction import redact_text


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
        permission_callback: Callable[[PermissionRequest, PermissionDecisionRecord], PermissionDecision | bool] | None = None,
        context_budget: ContextBudget | None = None,
        event_callback: Callable[[str, dict], None] | None = None,
        event_sink: Callable[[AgentEvent], None] | None = None,
        redaction_callback: Callable[[str, str, dict], str] | None = None,
        redaction_fail_closed: bool = False,
        pinned_skill_names: list[str] | None = None,
        mcp_servers: list[MCPServerConfig] | tuple[MCPServerConfig, ...] | None = None,
        mcp_bridge_tool_names: list[str] | None = None,
        owned_resources: list[object] | tuple[object, ...] | None = None,
        default_tool_context: ToolExecutionContext | None = None,
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
        self.llm_client = llm_client
        self.state = state or AgentState()
        self.memory = memory or ConversationMemory()
        self.memory_store = memory_store
        self.memory_policy = memory_policy or (
            MemoryPolicy(memory_store, "automatic") if memory_store is not None else None
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
        self.redaction_callback = redaction_callback
        self.redaction_fail_closed = redaction_fail_closed
        self.pinned_skill_names = pinned_skill_names or []
        self.mcp_servers = tuple(mcp_servers or ())
        self.mcp_bridge_tool_names = list(mcp_bridge_tool_names or [])
        self._owned_resources = list(owned_resources or [])
        self._closed = False
        self._tool_contexts: dict[str, ToolExecutionContext | None] = {}
        self.default_tool_context = default_tool_context
        self._profile_memories: list[MemoryRecord] = []
        self._relevant_memories: list[MemoryRecord] = []
        self._selected_skills: list[SkillSelection] = []
        self._restore_pending_turn_context()
        self.state.conversation_summary = self.memory.conversation_summary
        self._tool_executor = ToolExecutor(
            registry=self.tool_registry,
            permission_policy=self.permission_policy,
            permission_callback=self.permission_callback,
            trace=self._trace,
            get_context=self._tool_context_for_turn,
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
            resolve_mcp_approval=self._tool_executor.resolve_hosted_mcp_approval,
            mcp_servers=self.mcp_servers,
            max_skill_content_chars=self.max_skill_content_chars,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            max_json_repair_attempts=self.max_json_repair_attempts,
            max_reflection_attempts=self.max_reflection_attempts,
            trace_max_prompt_chars=self.trace_max_prompt_chars,
        )
        self._action_runtime = ActionLoopRuntime(
            model=self._model_transport,
            tools=self._tool_executor,
            effects=self._turn_effects,
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
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        if self.trace_logger is not None:
            try:
                self.trace_logger.close()
            except Exception as exc:  # pragma: no cover - defensive aggregation
                failures.append(exc)
        self.event_callback = None
        self.event_sink = None
        self._tool_contexts.clear()
        if failures:
            raise RuntimeError(f"Failed to close {len(failures)} owned agent resource(s)") from failures[0]

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
            )
            if isinstance(turn_or_response, str):
                return turn_or_response
            turn = turn_or_response
            result = self._run_action_loop(turn, require_plan=require_plan)
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
    ) -> str:
        """Start a user turn and run it with async tool execution."""
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
            )
            if isinstance(turn_or_response, str):
                return turn_or_response
            turn = turn_or_response
            result = await self._run_action_loop_async(turn, require_plan=require_plan)
        except BaseException as exc:
            turn = turn or self._turn_started_after(previous_turn_count)
            if turn is not None:
                self._terminalize_exception(turn, exc)
                self._release_tool_context(turn)
            raise
        if turn.status != "waiting_for_approval":
            self._release_tool_context(turn)
        return result

    def _start_user_turn(
        self,
        clean_message: str,
        *,
        context_sections: list[TurnContextSection | dict | str] | None,
        prompt_profile: str | None,
        locale: str | None,
        extension_metadata: dict | None,
        tool_context: ToolExecutionContext | dict | None,
    ) -> TurnState | str:
        """Create and trace a user turn before model/tool execution."""
        if self.has_pending_plan():
            return "A plan is waiting for approval. Use /approve to execute it or /reject to cancel it."

        turn_context_sections = _coerce_turn_context_sections(context_sections)
        execution_context = _coerce_tool_execution_context(tool_context) or self.default_tool_context
        turn = TurnState(
            user_message=clean_message,
            available_tool_names=[tool.name for tool in self.tool_registry.list_tools()],
            context_sections=turn_context_sections,
            prompt_profile=prompt_profile,
            locale=locale,
            extension_metadata=extension_metadata or {},
            tool_context_metadata=execution_context.metadata if execution_context else {},
        )
        if execution_context is None:
            execution_context = ToolExecutionContext()
        execution_context = ToolExecutionContext(
            metadata={
                **execution_context.metadata,
                "conversation_id": self.state.conversation_id,
                "turn_id": turn.turn_id,
            },
            deps=execution_context.deps,
        )
        self.state.current_turn_id = turn.turn_id
        self.state.available_tool_names = turn.available_tool_names
        self.state.turns.append(turn)
        self._trace(TraceEvent.TURN_STARTED, {"turn": turn.to_dict()})
        if turn_context_sections or prompt_profile or locale:
            self._trace(
                TraceEvent.TURN_CONTEXT_SELECTED,
                {
                    "turn_id": turn.turn_id,
                    "context_section_ids": [section.id for section in turn_context_sections],
                    "context_sections": [section.to_dict() for section in turn_context_sections],
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
        self._trace(TraceEvent.USER_MESSAGE, {"turn_id": turn.turn_id, "content": clean_message})

        self._tool_contexts[turn.turn_id] = execution_context
        return turn

    def has_pending_plan(self) -> bool:
        """Return True when a turn is paused on a plan awaiting approval."""
        turn = self._pending_plan_turn()
        return bool(turn and turn.active_plan and not turn.plan_approved)

    def approve_plan(self) -> str:
        """Approve the pending plan and continue the paused turn."""
        self._ensure_open()
        self._refresh_action_runtime()
        turn = self._pending_plan_turn()
        try:
            turn_or_response = self._approve_pending_plan()
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
        turn = self._pending_plan_turn()
        try:
            turn_or_response = self._approve_pending_plan()
            if isinstance(turn_or_response, str):
                return turn_or_response
            turn = turn_or_response
            return await self._run_action_loop_async(turn, require_plan=False)
        except BaseException as exc:
            if turn is not None:
                self._terminalize_exception(turn, exc)
            raise
        finally:
            if turn is not None:
                self._release_tool_context(turn)

    def _approve_pending_plan(self) -> TurnState | str:
        """Mark the pending plan approved and return its paused turn."""
        turn = self._pending_plan_turn()
        if turn is None or turn.active_plan is None:
            return "No plan is waiting for approval."

        try:
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
        except BaseException:
            self._release_tool_context(turn)
            raise
        return turn

    def reject_plan(self) -> str:
        """Reject the pending plan without executing tools."""
        self._ensure_open()
        self._refresh_action_runtime()
        turn = self._pending_plan_turn()
        if turn is None or turn.active_plan is None:
            return "No plan is waiting for approval."

        try:
            message = "Plan rejected. No tools were run."
            turn.reject_plan(message)
            self.state.pending_plan_turn_id = None
            self.state.active_plan = None
            self.state.final_answer = message
            self.memory.add_assistant_message(message)
            self.state.messages = self.memory.recent()
            self._trace(
                TraceEvent.PLAN_REJECTED,
                {"turn_id": turn.turn_id, "plan": turn.active_plan.to_dict()},
            )
            self._trace(TraceEvent.TURN_FINISHED, self._turn_effects.state_snapshot(turn))
            return message
        finally:
            self._release_tool_context(turn)

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

    async def _run_action_loop_async(self, turn: TurnState, *, require_plan: bool) -> str:
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

        tools = self._tool_executor
        tools.registry = self.tool_registry
        tools.permission_policy = self.permission_policy
        tools.permission_callback = self.permission_callback

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

    def _restore_pending_turn_context(self) -> None:
        """Restore the exact skills and memories that shaped a pending plan."""
        pending_id = self.state.pending_plan_turn_id
        if pending_id is None:
            return
        turn = next(
            (item for item in self.state.turns if item.turn_id == pending_id),
            None,
        )
        if turn is None:
            return

        if self.skill_registry is not None:
            for name in turn.loaded_skill_names:
                skill = self.skill_registry.get_skill(name)
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

        if self.memory_store is not None:
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
                },
            )

    def _select_long_term_memories(self, user_message: str) -> None:
        """Select durable memories that should shape this turn."""
        self._profile_memories = []
        self._relevant_memories = []
        self.state.loaded_memory_ids = []

        if self.memory_store is None or self.memory_policy is None or not self.memory_policy.retrieval_enabled:
            return

        self._trace(
            TraceEvent.MEMORY_SEARCH_STARTED,
            {"turn_id": self.state.current_turn_id, "query": user_message},
        )
        profile, relevant = select_memories_for_prompt(self.memory_store, user_message)
        self._profile_memories = profile
        self._relevant_memories = relevant
        self.state.loaded_memory_ids = [memory.id for memory in [*profile, *relevant]]
        self._trace(
            TraceEvent.MEMORY_SEARCH_COMPLETED,
            {
                "turn_id": self.state.current_turn_id,
                "profile_memory_ids": [memory.id for memory in profile],
                "relevant_memory_ids": [memory.id for memory in relevant],
                "loaded_memory_ids": self.state.loaded_memory_ids,
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
        pinned_selections: list[SkillSelection] = []
        pinned_names: set[str] = set()
        for name in self.pinned_skill_names:
            skill = self.skill_registry.get_skill(name)
            if skill is None:
                continue
            self.skill_registry.load_content(skill.name)
            pinned_names.add(skill.name)
            pinned_selections.append(
                SkillSelection(
                    skill=skill,
                    score=10_000,
                    matched_keywords=["pinned"],
                )
            )

        auto_selections = self.skill_registry.load_selected_skills(
            user_message,
            limit=self.max_skills_per_turn,
        )
        self._selected_skills = [
            *pinned_selections,
            *(selection for selection in auto_selections if selection.skill.name not in pinned_names),
        ][: self.max_skills_per_turn]
        self.state.loaded_skill_names = [selection.skill.name for selection in self._selected_skills]
        self._trace(
            TraceEvent.SKILL_SELECTION_COMPLETED,
            {
                "turn_id": self.state.current_turn_id,
                "loaded_skill_names": self.state.loaded_skill_names,
                "skills": [
                    {
                        "name": selection.skill.name,
                        "path": str(selection.skill.path),
                        "score": selection.score,
                        "matched_keywords": selection.matched_keywords,
                    }
                    for selection in self._selected_skills
                ],
            },
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
        turn.model_usage_totals = _aggregate_model_usage_reports(turn.model_usage_reports)
        self.state.last_usage_report = turn.model_usage_totals
        return usage_payload, cost_payload

    def _trace(self, event_type: str, payload: dict | None = None) -> None:
        payload = self._redact_event_payload(event_type, payload or {})
        if self.trace_logger is not None:
            self.trace_logger.log(event_type, payload)
        if self.event_callback is not None:
            self.event_callback(event_type, payload)
        if self.event_sink is not None:
            self.event_sink(AgentEvent(event_type, payload))

    def _redact_text(self, event_type: str, text: str, metadata: dict) -> tuple[str, dict]:
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
                redacted, metadata = self._redact_text(event_type, value, {"path": path})
                redacted_any = redacted_any or bool(metadata.get("redacted"))
                if metadata.get("redaction_error"):
                    error = str(metadata["redaction_error"])
                    redacted_any = redacted_any or bool(metadata.get("fail_closed"))
                return redacted
            if isinstance(value, dict):
                return {key: redact_value(item, f"{path}.{key}") for key, item in value.items()}
            if isinstance(value, list):
                return [redact_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
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
        if not turn.tool_context_metadata and default_context is None:
            return None
        return ToolExecutionContext(
            metadata={
                **(default_context.metadata if default_context is not None else {}),
                **turn.tool_context_metadata,
                "conversation_id": self.state.conversation_id,
                "turn_id": turn.turn_id,
            },
            deps=default_context.deps if default_context is not None else None,
        )

    def _release_tool_context(self, turn: TurnState) -> None:
        """Release request-scoped host dependencies after terminal work."""
        self._tool_contexts.pop(turn.turn_id, None)

    def _write_tool_output_artifact(self, name: str, content: str) -> dict | None:
        if self.trace_logger is None:
            return None
        return self.trace_logger.write_artifact(name, content)


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

    usage = aggregate_usage([*request_usages, *failed_attempt_usages], source="turn_total")
    cost = aggregate_cost([*request_costs, *failed_attempt_costs])
    return {
        "request_count": len([report for report in reports if isinstance(report, dict)]),
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
            metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
            sections.append(
                TurnContextSection(
                    id=str(section_id),
                    title=value.get("title") if isinstance(value.get("title"), str) else None,
                    source=value.get("source") if isinstance(value.get("source"), str) else None,
                    content=content,
                    metadata=metadata,
                )
            )
    return sections


def _coerce_tool_execution_context(value: ToolExecutionContext | dict | None) -> ToolExecutionContext | None:
    if value is None:
        return None
    if isinstance(value, ToolExecutionContext):
        return value
    return ToolExecutionContext(metadata=value)
