"""Prompt construction and model transports used by the action loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import re

from chulk.core.actions import AgentAction, action_json_schema_for
from chulk.core.context import AgentPrompt, ContextBudget
from chulk.core.events import TraceEvent
from chulk.core.prompt_builder import build_agent_prompt
from chulk.core.planning import read_only_planning_tool_names
from chulk.core.reflection import (
    ReflectionParseError,
    ReflectionResult,
    build_reflection_messages,
    parse_reflection_response,
)
from chulk.core.state import AgentState, TurnState
from chulk.core.trace_format import format_action_trace, format_model_request_trace
from chulk.llm import LLMActionError, LLMActionResult, LLMClient, LLMError
from chulk.llm.base import call_async_with_supported_kwargs, call_with_supported_kwargs
from chulk.llm.capabilities import (
    client_supports_hosted_mcp_tools,
    client_supports_native_tool_calling,
)
from chulk.llm.tools import PlanningToolAvailability, provider_action_tools
from chulk.mcp import MCPServerConfig
from chulk.memory import ConversationMemory, MemoryRecord
from chulk.skills import SkillRegistry, SkillSelection
from chulk.tools import Tool, ToolRegistry


MAX_SUMMARY_SOURCE_CHARS = 12000
MAX_SUMMARY_CHARS = 4000
SUMMARY_COMPACTION_PASSES = 3
MAX_UNPARSED_MODEL_OUTPUT_CHARS = 2000


TraceCallback = Callable[[str, dict | None], None]
AccountingCallback = Callable[..., tuple[dict | None, dict | None]]


@dataclass(frozen=True)
class ProtocolFailure:
    """A terminal structured-action protocol failure already recorded in state."""

    message: str


@dataclass
class ModelTransport:
    """Build prompts and perform sync/async model requests without an Agent dependency."""

    llm_client: LLMClient
    state: AgentState
    memory: ConversationMemory
    tool_registry: ToolRegistry
    system_prompt: str
    context_budget: ContextBudget
    skill_registry: SkillRegistry | None
    get_profile_memories: Callable[[], list[MemoryRecord]]
    get_relevant_memories: Callable[[], list[MemoryRecord]]
    get_selected_skills: Callable[[], list[SkillSelection]]
    trace: TraceCallback
    record_accounting: AccountingCallback
    resolve_mcp_approval: Callable[[dict, TurnState], bool]
    mcp_servers: tuple[MCPServerConfig, ...]
    max_skill_content_chars: int
    max_tool_calls_per_turn: int
    max_json_repair_attempts: int
    max_reflection_attempts: int
    trace_max_prompt_chars: int

    def build_prompt(self, turn: TurnState, *, require_plan: bool) -> AgentPrompt:
        """Build the model input and context report."""
        native_action_protocol = self._native_tool_calling_enabled()
        action_tools = self._action_tools(require_plan=require_plan)
        planning_tools = self._planning_tool_availability(
            turn, require_plan=require_plan
        )
        native_tool_declarations = provider_action_tools(
            action_tools,
            planning_tools=planning_tools,
        )
        if native_action_protocol and not require_plan and self._hosted_mcp_enabled():
            native_tool_declarations.extend(
                _safe_hosted_mcp_declaration(server) for server in self.mcp_servers
            )
        return build_agent_prompt(
            system_prompt=self.system_prompt,
            memory=self.memory,
            profile_memories=self.get_profile_memories(),
            relevant_memories=self.get_relevant_memories(),
            selected_skills=self.get_selected_skills(),
            tool_registry=self.tool_registry,
            max_skill_content_chars=self.max_skill_content_chars,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            context_sections=turn.context_sections,
            prompt_profile=turn.prompt_profile,
            locale=turn.locale,
            planning_enabled=require_plan or turn.active_plan is not None,
            active_plan=turn.active_plan,
            plan_approved=turn.plan_approved,
            require_plan=require_plan,
            native_action_protocol=native_action_protocol,
            native_tool_declarations=(
                native_tool_declarations if native_action_protocol else []
            ),
            context_budget=self.context_budget,
        )

    def compact_prompt(
        self,
        prompt: AgentPrompt,
        turn: TurnState,
        *,
        require_plan: bool,
    ) -> AgentPrompt:
        """Summarize old raw messages that would otherwise be dropped."""
        current_prompt = prompt
        for _ in range(SUMMARY_COMPACTION_PASSES):
            pending_messages = self.memory.consume_pending_summary_messages()
            omitted_messages = current_prompt.omitted_messages
            messages = _dedupe_messages([*pending_messages, *omitted_messages])
            if not messages:
                return current_prompt
            summary, fallback, error = self._summarize(messages, turn)
            current_prompt = self._apply_summary(
                turn,
                require_plan=require_plan,
                pending_messages=pending_messages,
                omitted_messages=omitted_messages,
                summary=summary,
                fallback=fallback,
                error=error,
            )
        return current_prompt

    async def compact_prompt_async(
        self,
        prompt: AgentPrompt,
        turn: TurnState,
        *,
        require_plan: bool,
    ) -> AgentPrompt:
        """Summarize old raw messages without blocking the event loop."""
        current_prompt = prompt
        for _ in range(SUMMARY_COMPACTION_PASSES):
            pending_messages = self.memory.consume_pending_summary_messages()
            omitted_messages = current_prompt.omitted_messages
            messages = _dedupe_messages([*pending_messages, *omitted_messages])
            if not messages:
                return current_prompt
            summary, fallback, error = await self._summarize_async(messages, turn)
            current_prompt = self._apply_summary(
                turn,
                require_plan=require_plan,
                pending_messages=pending_messages,
                omitted_messages=omitted_messages,
                summary=summary,
                fallback=fallback,
                error=error,
            )
        return current_prompt

    def request_action(
        self,
        turn: TurnState,
        prompt: AgentPrompt,
        *,
        require_plan: bool,
    ) -> AgentAction | ProtocolFailure:
        """Request and record one validated action over the sync transport."""
        native_action_protocol = prompt.action_transport == "provider_native"
        hosted_mcp_enabled = (
            native_action_protocol and not require_plan and self._hosted_mcp_enabled()
        )
        messages = self._record_model_request(
            turn,
            prompt,
            hosted_mcp_enabled=hosted_mcp_enabled,
        )
        try:
            result = call_with_supported_kwargs(
                self.llm_client.complete_action,
                messages,
                max_repair_attempts=self.max_json_repair_attempts,
                action_schema=self._action_schema(
                    turn,
                    require_plan=require_plan,
                ),
                tools=(
                    self._action_tools(require_plan=require_plan)
                    if native_action_protocol
                    else None
                ),
                planning_tools=(
                    self._planning_tool_availability(turn, require_plan=require_plan)
                    if native_action_protocol
                    else None
                ),
                hosted_mcp_servers=(self.mcp_servers if hosted_mcp_enabled else None),
                mcp_approval_callback=(
                    (lambda request: self.resolve_mcp_approval(request, turn))
                    if hosted_mcp_enabled
                    else None
                ),
            )
        except LLMActionError as exc:
            return self._record_protocol_failure(turn, exc)
        return self._record_action_result(turn, result)

    async def request_action_async(
        self,
        turn: TurnState,
        prompt: AgentPrompt,
        *,
        require_plan: bool,
    ) -> AgentAction | ProtocolFailure:
        """Request and record one validated action over the async transport."""
        native_action_protocol = prompt.action_transport == "provider_native"
        hosted_mcp_enabled = (
            native_action_protocol and not require_plan and self._hosted_mcp_enabled()
        )
        messages = self._record_model_request(
            turn,
            prompt,
            hosted_mcp_enabled=hosted_mcp_enabled,
        )
        try:
            result = await call_async_with_supported_kwargs(
                self.llm_client.acomplete_action,
                messages,
                max_repair_attempts=self.max_json_repair_attempts,
                action_schema=self._action_schema(
                    turn,
                    require_plan=require_plan,
                ),
                tools=(
                    self._action_tools(require_plan=require_plan)
                    if native_action_protocol
                    else None
                ),
                planning_tools=(
                    self._planning_tool_availability(turn, require_plan=require_plan)
                    if native_action_protocol
                    else None
                ),
                hosted_mcp_servers=(self.mcp_servers if hosted_mcp_enabled else None),
                mcp_approval_callback=(
                    (lambda request: self.resolve_mcp_approval(request, turn))
                    if hosted_mcp_enabled
                    else None
                ),
            )
        except LLMActionError as exc:
            return self._record_protocol_failure(turn, exc)
        return self._record_action_result(turn, result)

    def reflect(self, proposed_answer: str, turn: TurnState) -> ReflectionResult:
        """Review a proposed answer through the sync text transport."""
        attempt, messages, request_index = self._start_reflection(proposed_answer, turn)
        try:
            response = self.llm_client.complete_response(messages)
            raw_response = response.content
        except LLMError as exc:
            return self._fail_open_reflection(
                turn,
                proposed_answer,
                attempt=attempt,
                error=str(exc),
                raw_response=None,
                request_index=request_index,
            )
        self._record_reflection_response(
            turn,
            request_index=request_index,
            attempt=attempt,
            raw_response=raw_response,
            response=response,
        )
        return self._parse_reflection(
            turn,
            proposed_answer,
            attempt=attempt,
            raw_response=raw_response,
            request_index=request_index,
        )

    async def reflect_async(
        self, proposed_answer: str, turn: TurnState
    ) -> ReflectionResult:
        """Review a proposed answer through the async text transport."""
        attempt, messages, request_index = self._start_reflection(proposed_answer, turn)
        try:
            response = await self.llm_client.acomplete_response(messages)
            raw_response = response.content
        except LLMError as exc:
            return self._fail_open_reflection(
                turn,
                proposed_answer,
                attempt=attempt,
                error=str(exc),
                raw_response=None,
                request_index=request_index,
            )
        self._record_reflection_response(
            turn,
            request_index=request_index,
            attempt=attempt,
            raw_response=raw_response,
            response=response,
        )
        return self._parse_reflection(
            turn,
            proposed_answer,
            attempt=attempt,
            raw_response=raw_response,
            request_index=request_index,
        )

    def _apply_summary(
        self,
        turn: TurnState,
        *,
        require_plan: bool,
        pending_messages: list[dict[str, str]],
        omitted_messages: list[dict[str, str]],
        summary: str,
        fallback: bool,
        error: str | None,
    ) -> AgentPrompt:
        removed_count = self.memory.remove_messages(omitted_messages)
        summarized_count = len(pending_messages) + removed_count
        self.memory.update_conversation_summary(
            summary,
            summarized_message_count=summarized_count,
        )
        self.state.conversation_summary = self.memory.conversation_summary
        self.trace(
            TraceEvent.CONTEXT_SUMMARY_CREATED,
            {
                "turn_id": turn.turn_id,
                "summary": self.memory.conversation_summary,
                "source_message_count": self.memory.summary_message_count,
                "summarized_message_count": summarized_count,
                "fallback": fallback,
                "error": error,
            },
        )
        return self.build_prompt(turn, require_plan=require_plan)

    def _summarize(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
    ) -> tuple[str, bool, str | None]:
        summary_messages, request_index = self._start_summary(messages, turn)
        try:
            response = self.llm_client.complete_response(summary_messages)
        except LLMError as exc:
            return self._summary_failure(messages, turn, request_index, exc)
        return self._finish_summary(messages, turn, request_index, response)

    async def _summarize_async(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
    ) -> tuple[str, bool, str | None]:
        summary_messages, request_index = self._start_summary(messages, turn)
        try:
            response = await self.llm_client.acomplete_response(summary_messages)
        except LLMError as exc:
            return self._summary_failure(messages, turn, request_index, exc)
        return self._finish_summary(messages, turn, request_index, response)

    def _start_summary(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
    ) -> tuple[list[dict[str, str]], int]:
        summary_messages = _context_summary_messages(
            previous_summary=self.memory.conversation_summary,
            messages=messages,
        )
        turn.model_request_count += 1
        request_index = turn.model_request_count
        payload = format_model_request_trace(
            summary_messages,
            max_prompt_chars=self.trace_max_prompt_chars,
            request_index=request_index,
            turn_id=turn.turn_id,
            loaded_memory_ids=self.state.loaded_memory_ids,
            loaded_skill_names=self.state.loaded_skill_names,
            available_tool_names=turn.available_tool_names,
            context_report={
                "purpose": "context_summary",
                "source_message_count": len(messages),
                "existing_summary": self.memory.conversation_summary is not None,
            },
        )
        payload["purpose"] = "context_summary"
        payload["summary_source_message_count"] = len(messages)
        self.trace(TraceEvent.MODEL_REQUEST_STARTED, payload)
        return summary_messages, request_index

    def _summary_failure(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
        request_index: int,
        exc: LLMError,
    ) -> tuple[str, bool, str]:
        self.trace(
            TraceEvent.MODEL_RESPONSE,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "content": "",
                "purpose": "context_summary",
                "error": str(exc),
            },
        )
        return (
            _fallback_context_summary(self.memory.conversation_summary, messages),
            True,
            str(exc),
        )

    def _finish_summary(self, messages, turn, request_index, response):
        raw_summary = response.content
        fallback_attempts = getattr(self.llm_client, "last_attempts", None)
        self._record_model_selection_outcome(turn, fallback_attempts)
        usage, cost = self.record_accounting(
            turn,
            request_index=request_index,
            usage=response.usage,
            cost=response.cost,
            fallback_attempts=fallback_attempts,
            purpose="context_summary",
        )
        self.trace(
            TraceEvent.MODEL_RESPONSE,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "content": raw_summary,
                "purpose": "context_summary",
                "usage": usage,
                "cost": cost,
            },
        )
        clean_summary = _clean_summary(raw_summary)
        if not clean_summary:
            return (
                _fallback_context_summary(self.memory.conversation_summary, messages),
                True,
                "empty_summary",
            )
        return clean_summary, False, None

    def _record_model_request(
        self,
        turn: TurnState,
        prompt: AgentPrompt,
        *,
        hosted_mcp_enabled: bool,
    ) -> list[dict[str, str]]:
        messages = prompt.messages
        context_report = prompt.context_report.to_dict()
        turn.context_reports.append(context_report)
        self.state.last_context_report = context_report
        turn.model_request_count += 1
        payload = format_model_request_trace(
            messages,
            max_prompt_chars=self.trace_max_prompt_chars,
            request_index=turn.model_request_count,
            turn_id=turn.turn_id,
            loaded_memory_ids=self.state.loaded_memory_ids,
            loaded_skill_names=self.state.loaded_skill_names,
            available_tool_names=turn.available_tool_names,
            context_report=context_report,
        )
        payload["action_transport"] = prompt.action_transport
        payload["hosted_mcp_enabled"] = hosted_mcp_enabled
        payload["hosted_mcp_server_labels"] = (
            [server.label for server in self.mcp_servers] if hosted_mcp_enabled else []
        )
        payload["native_tool_names"] = [
            str(declaration.get("name", ""))
            for declaration in prompt.native_tool_declarations
        ]
        payload["native_tool_declarations"] = _bounded_native_tool_declarations(
            prompt.native_tool_declarations,
            max_chars=self.trace_max_prompt_chars,
        )
        self.trace(TraceEvent.MODEL_REQUEST_STARTED, payload)
        return messages

    def _record_protocol_failure(
        self,
        turn: TurnState,
        exc: LLMActionError,
    ) -> ProtocolFailure:
        self.state.json_repair_attempts += exc.repair_attempts
        self.state.errors.extend(
            f"JSON repair attempt: {error}" for error in exc.errors
        )
        turn.errors.extend(f"JSON repair attempt: {error}" for error in exc.errors)
        usage, cost = self.record_accounting(
            turn,
            request_index=turn.model_request_count,
            usage=exc.usage,
            cost=exc.cost,
        )
        if exc.raw_response:
            self.trace(
                TraceEvent.MODEL_RESPONSE,
                {
                    "turn_id": turn.turn_id,
                    "request_index": turn.model_request_count,
                    "content": exc.raw_response,
                    "repair_attempts": exc.repair_attempts,
                    "repair_errors": exc.errors,
                    "parse_failed": True,
                    "usage": usage,
                    "cost": cost,
                },
            )
        return ProtocolFailure(
            message=_format_action_protocol_failure(str(exc), exc.raw_response)
        )

    def _record_action_result(
        self,
        turn: TurnState,
        result: LLMActionResult,
    ) -> AgentAction:
        action = result.action
        self.state.json_repair_attempts += result.repair_attempts
        self.state.errors.extend(
            f"JSON repair attempt: {error}" for error in result.errors
        )
        turn.errors.extend(f"JSON repair attempt: {error}" for error in result.errors)
        fallback_attempts = getattr(self.llm_client, "last_attempts", None)
        self._record_model_selection_outcome(turn, fallback_attempts)
        if fallback_attempts:
            self.trace(
                TraceEvent.LLM_FALLBACK_ATTEMPTS,
                {
                    "turn_id": turn.turn_id,
                    "request_index": turn.model_request_count,
                    "attempts": [
                        attempt.to_dict()
                        if hasattr(attempt, "to_dict")
                        else {"attempt": str(attempt)}
                        for attempt in fallback_attempts
                    ],
                },
            )
        usage, cost = self.record_accounting(
            turn,
            request_index=turn.model_request_count,
            usage=result.usage,
            cost=result.cost,
            fallback_attempts=fallback_attempts,
        )
        self.trace(
            TraceEvent.MODEL_RESPONSE,
            {
                "turn_id": turn.turn_id,
                "request_index": turn.model_request_count,
                "content": result.raw_response,
                "repair_attempts": result.repair_attempts,
                "repair_errors": result.errors,
                "usage": usage,
                "cost": cost,
                "metadata": result.metadata,
            },
        )
        payload = format_action_trace(action)
        payload["request_index"] = turn.model_request_count
        self.trace(TraceEvent.PARSED_ACTION, payload)
        self.trace(TraceEvent.MODEL_RESPONSE_PARSED, payload)
        return action

    def _start_reflection(
        self,
        proposed_answer: str,
        turn: TurnState,
    ) -> tuple[int, list[dict[str, str]], int]:
        turn.reflection_count += 1
        attempt = turn.reflection_count
        messages = build_reflection_messages(turn, proposed_answer)
        turn.model_request_count += 1
        request_index = turn.model_request_count
        context_report = {
            "purpose": "reflection",
            "reflection_attempt": attempt,
            "proposed_answer_chars": len(proposed_answer),
        }
        self.trace(
            TraceEvent.REFLECTION_STARTED,
            {
                "turn_id": turn.turn_id,
                "reflection_attempt": attempt,
                "proposed_answer": proposed_answer,
            },
        )
        request_payload = format_model_request_trace(
            messages,
            max_prompt_chars=self.trace_max_prompt_chars,
            request_index=request_index,
            turn_id=turn.turn_id,
            loaded_memory_ids=self.state.loaded_memory_ids,
            loaded_skill_names=self.state.loaded_skill_names,
            available_tool_names=turn.available_tool_names,
            context_report=context_report,
        )
        request_payload["purpose"] = "reflection"
        request_payload["reflection_attempt"] = attempt
        self.trace(TraceEvent.MODEL_REQUEST_STARTED, request_payload)
        return attempt, messages, request_index

    def _record_reflection_response(
        self,
        turn,
        *,
        request_index,
        attempt,
        raw_response,
        response,
    ) -> None:
        fallback_attempts = getattr(self.llm_client, "last_attempts", None)
        self._record_model_selection_outcome(turn, fallback_attempts)
        usage, cost = self.record_accounting(
            turn,
            request_index=request_index,
            usage=response.usage,
            cost=response.cost,
            fallback_attempts=fallback_attempts,
            purpose="reflection",
        )
        self.trace(
            TraceEvent.MODEL_RESPONSE,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "content": raw_response,
                "purpose": "reflection",
                "reflection_attempt": attempt,
                "usage": usage,
                "cost": cost,
            },
        )

    def _record_model_selection_outcome(
        self,
        turn: TurnState,
        attempts: object,
    ) -> None:
        if not isinstance(attempts, (list, tuple)) or not attempts:
            return
        serialized = [
            attempt.to_dict()
            if hasattr(attempt, "to_dict")
            else {"attempt": str(attempt)}
            for attempt in attempts
        ]
        turn.extension_metadata["model_attempts"] = serialized
        selection = turn.extension_metadata.get("model_selection")
        if not isinstance(selection, dict):
            return
        successful = next(
            (
                attempt
                for attempt in attempts
                if getattr(attempt, "success", False)
                and isinstance(
                    getattr(attempt, "model_profile_id", None),
                    str,
                )
            ),
            None,
        )
        if successful is None:
            return
        selected_id = successful.model_profile_id
        previous_id = selection.get("selected_profile_id")
        selection["selected_profile_id"] = selected_id
        if selected_id == previous_id:
            return
        failed_count = sum(
            1 for attempt in attempts if not getattr(attempt, "success", False)
        )
        reason = (
            f"runtime fallback selected after {failed_count} failed or "
            "unavailable profile(s)"
        )
        selection["reason"] = reason
        self.trace(
            TraceEvent.MODEL_PROFILE_SELECTED,
            {
                "turn_id": turn.turn_id,
                **selection,
                "phase": "runtime_fallback",
            },
        )

    def _parse_reflection(
        self,
        turn: TurnState,
        proposed_answer: str,
        *,
        attempt: int,
        raw_response: str,
        request_index: int,
    ) -> ReflectionResult:
        try:
            reflection = parse_reflection_response(raw_response)
        except ReflectionParseError as exc:
            return self._fail_open_reflection(
                turn,
                proposed_answer,
                attempt=attempt,
                error=str(exc),
                raw_response=raw_response,
                request_index=request_index,
            )
        record = {
            **reflection.to_dict(),
            "attempt": attempt,
            "proposed_answer": proposed_answer,
        }
        turn.reflections.append(record)
        self.trace(TraceEvent.REFLECTION_COMPLETED, {"turn_id": turn.turn_id, **record})
        return reflection

    def _fail_open_reflection(
        self,
        turn: TurnState,
        proposed_answer: str,
        *,
        attempt: int,
        error: str,
        raw_response: str | None,
        request_index: int,
    ) -> ReflectionResult:
        reason = f"Reflection failed open: {error}"
        reflection = ReflectionResult(approved=True, reason=reason)
        record = {
            **reflection.to_dict(),
            "attempt": attempt,
            "proposed_answer": proposed_answer,
            "error": error,
            "raw_response": raw_response,
        }
        turn.reflections.append(record)
        turn.errors.append(reason)
        self.trace(
            TraceEvent.REFLECTION_FAILED,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                **record,
            },
        )
        return reflection

    def _native_tool_calling_enabled(self) -> bool:
        return client_supports_native_tool_calling(self.llm_client)

    def _hosted_mcp_enabled(self) -> bool:
        return client_supports_hosted_mcp_tools(self.llm_client)

    def _action_tools(self, *, require_plan: bool) -> list[Tool]:
        """Return only tools legal in the current action phase."""
        tools = list(self.tool_registry.list_tools())
        if not require_plan:
            return tools
        read_only_names = read_only_planning_tool_names(tools)
        return [tool for tool in tools if tool.name in read_only_names]

    def _action_schema(
        self,
        turn: TurnState,
        *,
        require_plan: bool,
    ) -> dict:
        action_types: list[str] = []
        if self._action_tools(require_plan=require_plan):
            action_types.append("tool_call")
        planning = self._planning_tool_availability(
            turn,
            require_plan=require_plan,
        )
        if planning.propose_plan:
            action_types.append("plan")
        elif planning.update_plan_step:
            action_types.append("plan_step_update")
        else:
            action_types.insert(0, "final_answer")
        return action_json_schema_for(action_types)

    @staticmethod
    def _planning_tool_availability(
        turn: TurnState,
        *,
        require_plan: bool,
    ) -> PlanningToolAvailability:
        active_plan = turn.active_plan
        return PlanningToolAvailability(
            propose_plan=require_plan and active_plan is None,
            update_plan_step=bool(
                turn.plan_approved
                and active_plan is not None
                and active_plan.active_step() is not None
            ),
        )


def _safe_hosted_mcp_declaration(server: MCPServerConfig) -> dict:
    declaration = {
        "type": "mcp",
        "name": f"mcp:{server.label}",
        "server_label": server.label,
        "server_url": server.server_url,
        "require_approval": server.approval,
        "authorization_configured": bool(server.authorization),
    }
    if server.server_description:
        declaration["server_description"] = server.server_description
    if server.allowed_tools:
        declaration["allowed_tools"] = list(server.allowed_tools)
    if server.defer_loading:
        declaration["defer_loading"] = True
    return declaration


def _bounded_native_tool_declarations(
    declarations: list[dict],
    *,
    max_chars: int,
) -> dict:
    serialized = json.dumps(
        declarations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    limit = max(0, max_chars)
    truncated = len(serialized) > limit
    return {
        "items": None if truncated else declarations,
        "json_preview": serialized[:limit] if truncated else None,
        "char_count": len(serialized),
        "returned_char_count": min(len(serialized), limit),
        "truncated": truncated,
    }


def _dedupe_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen_ids: set[int] = set()
    for message in messages:
        marker = id(message)
        if marker in seen_ids:
            continue
        seen_ids.add(marker)
        deduped.append(message)
    return deduped


def _format_action_protocol_failure(error: str, raw_response: str | None) -> str:
    lines = [
        "Model response was not valid action JSON after repair.",
        "I did not execute any tools from the invalid response.",
        "",
        f"Error: {error}",
    ]
    if raw_response:
        lines.extend(
            [
                "",
                "Unparsed model output:",
                "",
                _format_indented_preview(
                    raw_response,
                    MAX_UNPARSED_MODEL_OUTPUT_CHARS,
                ),
            ]
        )
    return "\n".join(lines)


def _format_indented_preview(text: str, max_chars: int) -> str:
    preview = text.strip()
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "\n... [truncated]"
    return "\n".join(f"    {line}" if line else "" for line in preview.splitlines())


def _context_summary_messages(
    *,
    previous_summary: str | None,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You update a compact, task-local conversation summary for an agent harness. "
                "Preserve decisions, constraints, files or tools used, important results, plan status, "
                "failed attempts, and next actions. Do not store secrets or API keys. "
                "Keep the summary concise and useful for continuing the current task."
            ),
        },
        {
            "role": "user",
            "content": _format_context_summary_request(
                previous_summary=previous_summary,
                messages=messages,
            ),
        },
    ]


def _format_context_summary_request(
    *,
    previous_summary: str | None,
    messages: list[dict[str, str]],
) -> str:
    sections: list[str] = []
    if previous_summary:
        sections.extend(["Previous compact summary:", previous_summary.strip(), ""])
    sections.extend(
        [
            "New older messages to fold into the compact summary:",
            _format_messages_for_summary(messages),
            "",
            "Return only the updated compact summary.",
        ]
    )
    return "\n".join(sections)


def _format_messages_for_summary(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    remaining_chars = MAX_SUMMARY_SOURCE_CHARS
    for message in messages:
        if remaining_chars <= 0:
            lines.append("[older-message input truncated]")
            break
        role = str(message.get("role") or "message")
        content = _clean_summary_source(str(message.get("content") or ""))
        line = f"{role}: {content}"
        if len(line) > remaining_chars:
            line = line[:remaining_chars].rstrip() + "..."
        lines.append(line)
        remaining_chars -= len(line)
    return "\n".join(lines)


def _fallback_context_summary(
    previous_summary: str | None,
    messages: list[dict[str, str]],
) -> str:
    parts = []
    if previous_summary:
        parts.append(previous_summary.strip())
    parts.append("Recent compacted context:")
    for message in messages[:8]:
        role = str(message.get("role") or "message")
        content = _compact_summary_line(
            _clean_summary_source(str(message.get("content") or "")),
            limit=300,
        )
        parts.append(f"- {role}: {content}")
    return _clean_summary("\n".join(parts))


def _clean_summary(value: str) -> str:
    clean = _redact_summary_text(" ".join(value.strip().split()))
    if len(clean) <= MAX_SUMMARY_CHARS:
        return clean
    return clean[:MAX_SUMMARY_CHARS].rstrip() + "..."


def _clean_summary_source(value: str) -> str:
    return _redact_summary_text(" ".join(value.split()))


def _compact_summary_line(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _redact_summary_text(value: str) -> str:
    patterns = [
        r"(?i)\b(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\bsk-[A-Za-z0-9_-]{16,}\b",
    ]
    redacted = value
    for pattern in patterns:
        redacted = re.sub(pattern, "[redacted secret]", redacted)
    return redacted
