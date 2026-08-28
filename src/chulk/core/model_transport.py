"""Prompt construction and model transports used by the action loop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import re

from chulk.core.actions import AgentAction, action_json_schema_for
from chulk.core.async_cleanup import await_cleanup_after_error
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
from chulk.errors import ConfigurationError, ErrorDetails
from chulk.llm import (
    LLMActionError,
    LLMActionResult,
    LLMClient,
    LLMError,
    LLMResponse,
)
from chulk.llm.base import call_async_with_supported_kwargs, call_with_supported_kwargs
from chulk.llm.capabilities import (
    client_supports_hosted_mcp_tools,
    client_supports_native_tool_calling,
)
from chulk.llm.tools import PlanningToolAvailability, provider_action_tools
from chulk.mcp import MCPServerConfig
from chulk.memory import ConversationMemory, MemoryRecord
from chulk.media import ModelRequest, UserInput
from chulk.skills import SkillRegistry, SkillSelection
from chulk.tools import Tool, ToolRegistry
from chulk.streaming import (
    AsyncIncrementalOutputPolicy,
    AsyncPassThroughOutputPolicy,
    FinalAnswerChunk,
    FinalAnswerDeliveryStatus,
    FinalAnswerPolicyDecision,
    IncrementalOutputPolicy,
    OutputPolicyFailureMode,
    PassThroughOutputPolicy,
)


MAX_SUMMARY_SOURCE_CHARS = 12000
MAX_SUMMARY_CHARS = 4000
SUMMARY_COMPACTION_PASSES = 3
MAX_UNPARSED_MODEL_OUTPUT_CHARS = 2000


TraceCallback = Callable[[str, dict | None], None]
AccountingCallback = Callable[..., tuple[dict | None, dict | None]]
ReservationCallback = Callable[..., dict | None]
ReleaseCallback = Callable[..., dict | None]
AsyncAccountingCallback = Callable[
    ...,
    Awaitable[tuple[dict | None, dict | None]],
]
AsyncReservationCallback = Callable[..., Awaitable[dict | None]]
AsyncReleaseCallback = Callable[..., Awaitable[dict | None]]


@dataclass(frozen=True)
class ProtocolFailure:
    """A terminal structured-action protocol failure already recorded in state."""

    message: str


@dataclass(frozen=True)
class FinalAnswerStreamResult:
    """Permitted public content and terminal delivery evidence."""

    content: str
    status: FinalAnswerDeliveryStatus
    public_delta_count: int
    provider_completed: bool
    error: str | None = None


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
    reserve_accounting: ReservationCallback
    release_accounting: ReleaseCallback
    resolve_mcp_approval: Callable[[dict, TurnState], bool]
    mcp_servers: tuple[MCPServerConfig, ...]
    max_skill_content_chars: int
    max_tool_calls_per_turn: int
    max_json_repair_attempts: int
    max_reflection_attempts: int
    trace_max_prompt_chars: int
    max_output_tokens: int | None
    record_accounting_async: AsyncAccountingCallback | None = None
    reserve_accounting_async: AsyncReservationCallback | None = None
    release_accounting_async: AsyncReleaseCallback | None = None
    flush_async: Callable[[], Awaitable[None]] | None = None
    redact_text: Callable[[str, str, dict], tuple[str, dict]] | None = None
    output_policy: IncrementalOutputPolicy | None = None
    async_output_policy: AsyncIncrementalOutputPolicy | None = None
    output_policy_failure_mode: OutputPolicyFailureMode = OutputPolicyFailureMode.CLOSED

    def stream_final_answer(
        self, draft: str, turn: TurnState
    ) -> FinalAnswerStreamResult:
        """Emit a policy-filtered plain-text answer as provider chunks arrive."""
        messages, request_index = self._start_final_answer_stream(turn, draft)
        policy = self.output_policy or PassThroughOutputPolicy()
        parts: list[str] = []
        public_sequence = 0
        provider_sequence = 0
        usage = None
        cost = None
        completed = False
        self.trace(
            TraceEvent.MODEL_STREAM_STARTED,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "source": "incremental_final_answer",
            },
        )
        try:
            stream = call_with_supported_kwargs(
                self.llm_client.stream_final_answer,
                messages,
                max_output_tokens=self.max_output_tokens,
                public_output_committed=lambda: public_sequence > 0,
                before_fallback=lambda: self._reset_output_policy(policy, turn),
            )
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.cost is not None:
                    cost = chunk.cost
                if chunk.type == "completed":
                    completed = True
                    break
                if chunk.type != "text_delta" or not chunk.text:
                    continue
                decision = self._apply_output_policy(
                    policy, chunk.text, provider_sequence, turn
                )
                provider_sequence += 1
                if decision.blocked:
                    return self._finish_final_answer_stream(
                        turn,
                        request_index=request_index,
                        content="".join(parts),
                        status=FinalAnswerDeliveryStatus.BLOCKED,
                        public_delta_count=public_sequence,
                        provider_completed=False,
                        usage=usage,
                        cost=cost,
                        error=decision.reason or "output policy blocked the answer",
                    )
                public_sequence = self._publish_policy_text(
                    decision.text, turn, request_index, public_sequence, parts
                )
                if decision.stop:
                    return self._finish_final_answer_stream(
                        turn,
                        request_index=request_index,
                        content="".join(parts),
                        status=FinalAnswerDeliveryStatus.TRUNCATED,
                        public_delta_count=public_sequence,
                        provider_completed=False,
                        usage=usage,
                        cost=cost,
                        error=decision.reason,
                    )
            decision = self._complete_output_policy(policy, provider_sequence, turn)
            if decision.blocked:
                return self._finish_final_answer_stream(
                    turn,
                    request_index=request_index,
                    content="".join(parts),
                    status=FinalAnswerDeliveryStatus.BLOCKED,
                    public_delta_count=public_sequence,
                    provider_completed=completed,
                    usage=usage,
                    cost=cost,
                    error=decision.reason or "output policy blocked the answer",
                )
            public_sequence = self._publish_policy_text(
                decision.text, turn, request_index, public_sequence, parts
            )
            status = (
                FinalAnswerDeliveryStatus.TRUNCATED
                if decision.stop
                else FinalAnswerDeliveryStatus.COMPLETE
            )
            return self._finish_final_answer_stream(
                turn,
                request_index=request_index,
                content="".join(parts),
                status=status,
                public_delta_count=public_sequence,
                provider_completed=completed,
                usage=usage,
                cost=cost,
                error=decision.reason,
            )
        except BaseException as exc:
            if parts:
                return self._finish_final_answer_stream(
                    turn,
                    request_index=request_index,
                    content="".join(parts),
                    status=FinalAnswerDeliveryStatus.FAILED,
                    public_delta_count=public_sequence,
                    provider_completed=False,
                    usage=usage,
                    cost=cost,
                    error=str(exc),
                )
            self.release_accounting(
                turn, request_index=request_index, reason="final_answer_stream_failed"
            )
            self.trace(
                TraceEvent.MODEL_STREAM_FAILED,
                {
                    "turn_id": turn.turn_id,
                    "request_index": request_index,
                    "source": "incremental_final_answer",
                    "error": str(exc),
                    "partial": bool(parts),
                },
            )
            raise

    async def stream_final_answer_async(
        self, draft: str, turn: TurnState
    ) -> FinalAnswerStreamResult:
        """Native async final-answer stream with no sync iterator adaptation."""
        messages, request_index = await self._start_final_answer_stream_async(turn, draft)
        policy = self.async_output_policy or AsyncPassThroughOutputPolicy()
        parts: list[str] = []
        public_sequence = 0
        provider_sequence = 0
        usage = None
        cost = None
        completed = False
        self.trace(
            TraceEvent.MODEL_STREAM_STARTED,
            {"turn_id": turn.turn_id, "request_index": request_index, "source": "incremental_final_answer"},
        )
        await self._flush_async()
        try:
            stream = call_with_supported_kwargs(
                self.llm_client.astream_final_answer,
                messages,
                max_output_tokens=self.max_output_tokens,
                public_output_committed=lambda: public_sequence > 0,
                before_fallback=lambda: self._reset_output_policy_async(policy, turn),
            )
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.cost is not None:
                    cost = chunk.cost
                if chunk.type == "completed":
                    completed = True
                    break
                if chunk.type != "text_delta" or not chunk.text:
                    continue
                decision = await self._apply_output_policy_async(
                    policy, chunk.text, provider_sequence, turn
                )
                provider_sequence += 1
                if decision.blocked:
                    return await self._finish_final_answer_stream_async(
                        turn, request_index, "".join(parts), FinalAnswerDeliveryStatus.BLOCKED,
                        public_sequence, False, usage, cost,
                        decision.reason or "output policy blocked the answer",
                    )
                public_sequence = self._publish_policy_text(
                    decision.text, turn, request_index, public_sequence, parts
                )
                await self._flush_async()
                if decision.stop:
                    return await self._finish_final_answer_stream_async(
                        turn, request_index, "".join(parts), FinalAnswerDeliveryStatus.TRUNCATED,
                        public_sequence, False, usage, cost, decision.reason,
                    )
            decision = await self._complete_output_policy_async(
                policy, provider_sequence, turn
            )
            if decision.blocked:
                return await self._finish_final_answer_stream_async(
                    turn, request_index, "".join(parts), FinalAnswerDeliveryStatus.BLOCKED,
                    public_sequence, completed, usage, cost,
                    decision.reason or "output policy blocked the answer",
                )
            public_sequence = self._publish_policy_text(
                decision.text, turn, request_index, public_sequence, parts
            )
            await self._flush_async()
            status = FinalAnswerDeliveryStatus.TRUNCATED if decision.stop else FinalAnswerDeliveryStatus.COMPLETE
            return await self._finish_final_answer_stream_async(
                turn, request_index, "".join(parts), status, public_sequence,
                completed, usage, cost, decision.reason,
            )
        except BaseException as exc:
            if parts and not isinstance(exc, asyncio.CancelledError):
                return await self._finish_final_answer_stream_async(
                    turn, request_index, "".join(parts),
                    FinalAnswerDeliveryStatus.FAILED, public_sequence,
                    False, usage, cost, str(exc),
                )
            await await_cleanup_after_error(
                self._release_accounting_async(
                    turn, request_index=request_index, reason="final_answer_stream_failed"
                ),
                exc,
            )
            self.trace(
                TraceEvent.MODEL_STREAM_FAILED,
                {"turn_id": turn.turn_id, "request_index": request_index, "source": "incremental_final_answer", "error": str(exc), "partial": bool(parts)},
            )
            await self._flush_async()
            if parts:
                turn.extension_metadata["final_answer_delivery"] = {
                    "status": FinalAnswerDeliveryStatus.FAILED.value,
                    "public_delta_count": public_sequence,
                    "provider_completed": False,
                    "error": "cancelled" if isinstance(exc, asyncio.CancelledError) else str(exc),
                    "partial_content": "".join(parts),
                }
            raise

    def _final_answer_messages(self, turn: TurnState, draft: str) -> list[dict[str, str]]:
        prompt = self.build_prompt(turn, require_plan=False)
        messages = [dict(message) for message in prompt.messages]
        for message in messages:
            if message.get("role") == "system":
                message["content"] = re.sub(
                    r"<response_protocol>.*?</response_protocol>",
                    "",
                    message.get("content", ""),
                    flags=re.DOTALL,
                ).strip()
        instruction = (
            "Return only the final user-facing answer as plain text. Do not emit JSON, "
            "tool calls, action payloads, repair content, or internal reasoning. The "
            f"validated answer intent was: {draft}"
        )
        messages.append({"role": "user", "content": instruction})
        return messages

    def _start_final_answer_stream(
        self, turn: TurnState, draft: str
    ) -> tuple[list[dict[str, str]], int]:
        messages = self._final_answer_messages(turn, draft)
        turn.model_request_count += 1
        request_index = turn.model_request_count
        self.reserve_accounting(
            turn,
            request_index=request_index,
            messages=messages,
            purpose="incremental_final_answer",
        )
        payload = format_model_request_trace(
            messages,
            max_prompt_chars=self.trace_max_prompt_chars,
            request_index=request_index,
            turn_id=turn.turn_id,
            loaded_memory_ids=self.state.loaded_memory_ids,
            loaded_skill_names=self.state.loaded_skill_names,
            available_tool_names=turn.available_tool_names,
            context_report={"purpose": "incremental_final_answer"},
        )
        payload["purpose"] = "incremental_final_answer"
        payload["action_transport"] = "plain_text_stream"
        self.trace(TraceEvent.MODEL_REQUEST_STARTED, payload)
        return messages, request_index

    async def _start_final_answer_stream_async(
        self, turn: TurnState, draft: str
    ) -> tuple[list[dict[str, str]], int]:
        messages = self._final_answer_messages(turn, draft)
        turn.model_request_count += 1
        request_index = turn.model_request_count
        await self._reserve_accounting_async(
            turn,
            request_index=request_index,
            messages=messages,
            purpose="incremental_final_answer",
        )
        payload = format_model_request_trace(
            messages,
            max_prompt_chars=self.trace_max_prompt_chars,
            request_index=request_index,
            turn_id=turn.turn_id,
            loaded_memory_ids=self.state.loaded_memory_ids,
            loaded_skill_names=self.state.loaded_skill_names,
            available_tool_names=turn.available_tool_names,
            context_report={"purpose": "incremental_final_answer"},
        )
        payload["purpose"] = "incremental_final_answer"
        payload["action_transport"] = "plain_text_stream"
        self.trace(TraceEvent.MODEL_REQUEST_STARTED, payload)
        await self._flush_async()
        return messages, request_index

    def _redact_stream_text(self, text: str, sequence: int, turn: TurnState) -> str:
        if self.redact_text is None:
            return text
        permitted, metadata = self.redact_text(
            TraceEvent.MODEL_STREAM_DELTA,
            text,
            {"turn_id": turn.turn_id, "field": "text", "sequence": sequence},
        )
        if metadata.get("redacted") or metadata.get("redaction_error"):
            turn.extension_metadata.setdefault("final_answer_stream_redactions", []).append(
                {"sequence": sequence, **metadata}
            )
        return permitted

    def _apply_output_policy(
        self,
        policy: IncrementalOutputPolicy,
        text: str,
        sequence: int,
        turn: TurnState,
    ) -> FinalAnswerPolicyDecision:
        chunk = FinalAnswerChunk(
            text=self._redact_stream_text(text, sequence, turn),
            sequence=sequence,
            turn_id=turn.turn_id,
        )
        try:
            return policy.process(chunk)
        except Exception as exc:
            if self.output_policy_failure_mode == OutputPolicyFailureMode.OPEN:
                return FinalAnswerPolicyDecision(text=chunk.text, reason=str(exc))
            return FinalAnswerPolicyDecision(blocked=True, reason=str(exc))

    async def _apply_output_policy_async(
        self,
        policy: AsyncIncrementalOutputPolicy,
        text: str,
        sequence: int,
        turn: TurnState,
    ) -> FinalAnswerPolicyDecision:
        chunk = FinalAnswerChunk(
            text=self._redact_stream_text(text, sequence, turn),
            sequence=sequence,
            turn_id=turn.turn_id,
        )
        try:
            return await policy.process(chunk)
        except Exception as exc:
            if self.output_policy_failure_mode == OutputPolicyFailureMode.OPEN:
                return FinalAnswerPolicyDecision(text=chunk.text, reason=str(exc))
            return FinalAnswerPolicyDecision(blocked=True, reason=str(exc))

    def _complete_output_policy(
        self, policy: IncrementalOutputPolicy, sequence: int, turn: TurnState
    ) -> FinalAnswerPolicyDecision:
        try:
            return policy.complete(turn_id=turn.turn_id, next_sequence=sequence)
        except Exception as exc:
            if self.output_policy_failure_mode == OutputPolicyFailureMode.OPEN:
                return FinalAnswerPolicyDecision(reason=str(exc))
            return FinalAnswerPolicyDecision(blocked=True, reason=str(exc))

    @staticmethod
    def _reset_output_policy(
        policy: IncrementalOutputPolicy, turn: TurnState
    ) -> None:
        reset = getattr(policy, "reset", None)
        if reset is None:
            raise RuntimeError(
                "a buffering output policy must implement reset() for safe fallback"
            )
        reset(turn_id=turn.turn_id)

    async def _complete_output_policy_async(
        self, policy: AsyncIncrementalOutputPolicy, sequence: int, turn: TurnState
    ) -> FinalAnswerPolicyDecision:
        try:
            return await policy.complete(turn_id=turn.turn_id, next_sequence=sequence)
        except Exception as exc:
            if self.output_policy_failure_mode == OutputPolicyFailureMode.OPEN:
                return FinalAnswerPolicyDecision(reason=str(exc))
            return FinalAnswerPolicyDecision(blocked=True, reason=str(exc))

    @staticmethod
    async def _reset_output_policy_async(
        policy: AsyncIncrementalOutputPolicy, turn: TurnState
    ) -> None:
        reset = getattr(policy, "reset", None)
        if reset is None:
            raise RuntimeError(
                "a buffering async output policy must implement reset() for safe fallback"
            )
        await reset(turn_id=turn.turn_id)

    def _publish_policy_text(
        self,
        text: str,
        turn: TurnState,
        request_index: int,
        sequence: int,
        parts: list[str],
    ) -> int:
        if not text:
            return sequence
        parts.append(text)
        self.trace(
            TraceEvent.MODEL_STREAM_DELTA,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "source": "incremental_final_answer",
                "sequence": sequence,
                "text": text,
            },
        )
        return sequence + 1

    def _finish_final_answer_stream(
        self,
        turn: TurnState,
        *,
        request_index: int,
        content: str,
        status: FinalAnswerDeliveryStatus,
        public_delta_count: int,
        provider_completed: bool,
        usage: object,
        cost: object,
        error: str | None,
    ) -> FinalAnswerStreamResult:
        usage_snapshot, cost_snapshot = self.record_accounting(
            turn,
            request_index=request_index,
            usage=usage,
            cost=cost,
            fallback_attempts=getattr(self.llm_client, "last_attempts", None),
            purpose="incremental_final_answer",
        )
        payload = {
            "turn_id": turn.turn_id,
            "request_index": request_index,
            "source": "incremental_final_answer",
            "status": status.value,
            "public_delta_count": public_delta_count,
            "provider_completed": provider_completed,
            "usage": usage_snapshot,
            "cost": cost_snapshot,
            "error": error,
        }
        self.trace(
            TraceEvent.MODEL_STREAM_COMPLETED
            if status in {FinalAnswerDeliveryStatus.COMPLETE, FinalAnswerDeliveryStatus.TRUNCATED}
            else TraceEvent.MODEL_STREAM_FAILED,
            payload,
        )
        self.trace(
            TraceEvent.MODEL_RESPONSE,
            {**payload, "content": content},
        )
        return FinalAnswerStreamResult(
            content=content,
            status=status,
            public_delta_count=public_delta_count,
            provider_completed=provider_completed,
            error=error,
        )

    async def _finish_final_answer_stream_async(
        self,
        turn: TurnState,
        request_index: int,
        content: str,
        status: FinalAnswerDeliveryStatus,
        public_delta_count: int,
        provider_completed: bool,
        usage: object,
        cost: object,
        error: str | None,
    ) -> FinalAnswerStreamResult:
        usage_snapshot, cost_snapshot = await self._record_accounting_async(
            turn,
            request_index=request_index,
            usage=usage,
            cost=cost,
            fallback_attempts=getattr(self.llm_client, "last_attempts", None),
            purpose="incremental_final_answer",
        )
        payload = {
            "turn_id": turn.turn_id, "request_index": request_index,
            "source": "incremental_final_answer", "status": status.value,
            "public_delta_count": public_delta_count,
            "provider_completed": provider_completed, "usage": usage_snapshot,
            "cost": cost_snapshot, "error": error,
        }
        self.trace(
            TraceEvent.MODEL_STREAM_COMPLETED if status in {FinalAnswerDeliveryStatus.COMPLETE, FinalAnswerDeliveryStatus.TRUNCATED} else TraceEvent.MODEL_STREAM_FAILED,
            payload,
        )
        self.trace(TraceEvent.MODEL_RESPONSE, {**payload, "content": content})
        await self._flush_async()
        return FinalAnswerStreamResult(
            content=content, status=status, public_delta_count=public_delta_count,
            provider_completed=provider_completed, error=error
        )

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
        self._ensure_prompt_within_budget(turn, prompt)
        native_action_protocol = prompt.action_transport == "provider_native"
        hosted_mcp_enabled = (
            native_action_protocol and not require_plan and self._hosted_mcp_enabled()
        )
        messages = self._record_model_request(
            turn,
            prompt,
            hosted_mcp_enabled=hosted_mcp_enabled,
        )
        request_kwargs: dict[str, object] = {
            "max_repair_attempts": self.max_json_repair_attempts,
            "action_schema": self._action_schema(
                turn,
                require_plan=require_plan,
            ),
            "tools": (
                self._action_tools(require_plan=require_plan)
                if native_action_protocol
                else None
            ),
            "planning_tools": (
                self._planning_tool_availability(turn, require_plan=require_plan)
                if native_action_protocol
                else None
            ),
            "hosted_mcp_servers": (
                self.mcp_servers if hosted_mcp_enabled else None
            ),
            "mcp_approval_callback": (
                (lambda request: self.resolve_mcp_approval(request, turn))
                if hosted_mcp_enabled
                else None
            ),
        }
        if self.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = self.max_output_tokens
        try:
            result = call_with_supported_kwargs(
                self.llm_client.complete_action_request,
                ModelRequest(
                    messages=tuple(messages),
                    user_input=(
                        turn.model_input
                        if isinstance(turn.model_input, UserInput)
                        else None
                    ),
                    purpose="agent_action",
                ),
                **request_kwargs,
            )
        except LLMActionError as exc:
            return self._record_protocol_failure(turn, exc)
        except BaseException:
            self.release_accounting(
                turn,
                request_index=turn.model_request_count,
                reason="model_transport_failed",
            )
            raise
        return self._record_action_result(turn, result)

    async def request_action_async(
        self,
        turn: TurnState,
        prompt: AgentPrompt,
        *,
        require_plan: bool,
    ) -> AgentAction | ProtocolFailure:
        """Request and record one validated action over the async transport."""
        self._ensure_prompt_within_budget(turn, prompt)
        native_action_protocol = prompt.action_transport == "provider_native"
        hosted_mcp_enabled = (
            native_action_protocol and not require_plan and self._hosted_mcp_enabled()
        )
        messages = await self._record_model_request_async(
            turn,
            prompt,
            hosted_mcp_enabled=hosted_mcp_enabled,
        )
        request_kwargs: dict[str, object] = {
            "max_repair_attempts": self.max_json_repair_attempts,
            "action_schema": self._action_schema(
                turn,
                require_plan=require_plan,
            ),
            "tools": (
                self._action_tools(require_plan=require_plan)
                if native_action_protocol
                else None
            ),
            "planning_tools": (
                self._planning_tool_availability(turn, require_plan=require_plan)
                if native_action_protocol
                else None
            ),
            "hosted_mcp_servers": (
                self.mcp_servers if hosted_mcp_enabled else None
            ),
            "mcp_approval_callback": (
                (lambda request: self.resolve_mcp_approval(request, turn))
                if hosted_mcp_enabled
                else None
            ),
        }
        if self.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = self.max_output_tokens
        try:
            result = await call_async_with_supported_kwargs(
                self.llm_client.acomplete_action_request,
                ModelRequest(
                    messages=tuple(messages),
                    user_input=(
                        turn.model_input
                        if isinstance(turn.model_input, UserInput)
                        else None
                    ),
                    purpose="agent_action",
                ),
                **request_kwargs,
            )
        except LLMActionError as exc:
            return await self._record_protocol_failure_async(turn, exc)
        except BaseException as exc:
            await await_cleanup_after_error(
                self._release_accounting_async(
                    turn,
                    request_index=turn.model_request_count,
                    reason="model_transport_failed",
                ),
                exc,
            )
            raise
        return await self._record_action_result_async(turn, result)

    def reflect(self, proposed_answer: str, turn: TurnState) -> ReflectionResult:
        """Review a proposed answer through the sync text transport."""
        attempt, messages, request_index = self._start_reflection(
            proposed_answer, turn
        )
        try:
            response = self._complete_response(messages)
            raw_response = response.content
        except LLMError as exc:
            self.release_accounting(
                turn,
                request_index=request_index,
                reason="reflection_transport_failed",
            )
            return self._fail_open_reflection(
                turn,
                proposed_answer,
                attempt=attempt,
                error=str(exc),
                raw_response=None,
                request_index=request_index,
            )
        except BaseException:
            self.release_accounting(
                turn,
                request_index=request_index,
                reason="reflection_transport_failed",
            )
            raise
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
        attempt, messages, request_index = await self._start_reflection_async(
            proposed_answer,
            turn,
        )
        try:
            response = await self._complete_response_async(messages)
            raw_response = response.content
        except LLMError as exc:
            await self._release_accounting_async(
                turn,
                request_index=request_index,
                reason="reflection_transport_failed",
            )
            return self._fail_open_reflection(
                turn,
                proposed_answer,
                attempt=attempt,
                error=str(exc),
                raw_response=None,
                request_index=request_index,
            )
        except BaseException as exc:
            await await_cleanup_after_error(
                self._release_accounting_async(
                    turn,
                    request_index=request_index,
                    reason="reflection_transport_failed",
                ),
                exc,
            )
            raise
        await self._record_reflection_response_async(
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
            response = self._complete_response(summary_messages)
        except LLMError as exc:
            return self._summary_failure(messages, turn, request_index, exc)
        except BaseException:
            self.release_accounting(
                turn,
                request_index=request_index,
                reason="context_summary_transport_failed",
            )
            raise
        return self._finish_summary(messages, turn, request_index, response)

    async def _summarize_async(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
    ) -> tuple[str, bool, str | None]:
        summary_messages, request_index = await self._start_summary_async(
            messages,
            turn,
        )
        try:
            response = await self._complete_response_async(summary_messages)
        except LLMError as exc:
            return await self._summary_failure_async(
                messages,
                turn,
                request_index,
                exc,
            )
        except BaseException as exc:
            await await_cleanup_after_error(
                self._release_accounting_async(
                    turn,
                    request_index=request_index,
                    reason="context_summary_transport_failed",
                ),
                exc,
            )
            raise
        return await self._finish_summary_async(
            messages,
            turn,
            request_index,
            response,
        )

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
        self.reserve_accounting(
            turn,
            request_index=request_index,
            messages=summary_messages,
            purpose="context_summary",
        )
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

    async def _start_summary_async(
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
        try:
            await self._reserve_accounting_async(
                turn,
                request_index=request_index,
                messages=summary_messages,
                purpose="context_summary",
            )
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
                    "existing_summary": (
                        self.memory.conversation_summary is not None
                    ),
                },
            )
            payload["purpose"] = "context_summary"
            payload["summary_source_message_count"] = len(messages)
            self.trace(TraceEvent.MODEL_REQUEST_STARTED, payload)
            await self._flush_async()
        except BaseException as exc:
            await await_cleanup_after_error(
                self._release_accounting_async(
                    turn,
                    request_index=request_index,
                    reason="context_summary_setup_failed",
                ),
                exc,
            )
            raise
        return summary_messages, request_index

    def _summary_failure(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
        request_index: int,
        exc: LLMError,
    ) -> tuple[str, bool, str]:
        self.release_accounting(
            turn,
            request_index=request_index,
            reason="context_summary_transport_failed",
        )
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

    async def _summary_failure_async(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
        request_index: int,
        exc: LLMError,
    ) -> tuple[str, bool, str]:
        await self._release_accounting_async(
            turn,
            request_index=request_index,
            reason="context_summary_transport_failed",
        )
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

    async def _finish_summary_async(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
        request_index: int,
        response: LLMResponse,
    ) -> tuple[str, bool, str | None]:
        raw_summary = response.content
        fallback_attempts = getattr(self.llm_client, "last_attempts", None)
        self._record_model_selection_outcome(turn, fallback_attempts)
        usage, cost = await self._record_accounting_async(
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
                _fallback_context_summary(
                    self.memory.conversation_summary,
                    messages,
                ),
                True,
                "empty_summary",
            )
        return clean_summary, False, None

    def _ensure_prompt_within_budget(
        self,
        turn: TurnState,
        prompt: AgentPrompt,
    ) -> None:
        """Reject an irreducibly oversized prompt before invoking a provider."""
        context_report = prompt.context_report.to_dict()
        if context_report["over_budget_tokens"] <= 0:
            return
        turn.context_reports.append(context_report)
        self.state.last_context_report = context_report
        self.trace(
            TraceEvent.CONTEXT_BUDGET_REJECTED,
            {
                "turn_id": turn.turn_id,
                "context_report": context_report,
            },
        )
        raise ConfigurationError(
            "Prompt exceeds the configured input token budget before a provider request.",
            details=ErrorDetails(
                failure_kind="context_budget_exceeded",
                conversation_id=self.state.conversation_id,
                turn_id=turn.turn_id,
                extensions={
                    "input_token_budget": context_report["budget"]["input_token_budget"],
                    "estimated_tokens": context_report["budget_estimated_tokens"],
                    "over_budget_tokens": context_report["over_budget_tokens"],
                },
            ),
        )

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
        self.reserve_accounting(
            turn,
            request_index=turn.model_request_count,
            messages=messages,
            purpose="agent_action",
            repair_attempts=self.max_json_repair_attempts,
        )
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

    async def _record_model_request_async(
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
        try:
            await self._reserve_accounting_async(
                turn,
                request_index=turn.model_request_count,
                messages=messages,
                purpose="agent_action",
                repair_attempts=self.max_json_repair_attempts,
            )
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
                [server.label for server in self.mcp_servers]
                if hosted_mcp_enabled
                else []
            )
            payload["native_tool_names"] = [
                str(declaration.get("name", ""))
                for declaration in prompt.native_tool_declarations
            ]
            payload["native_tool_declarations"] = (
                _bounded_native_tool_declarations(
                    prompt.native_tool_declarations,
                    max_chars=self.trace_max_prompt_chars,
                )
            )
            self.trace(TraceEvent.MODEL_REQUEST_STARTED, payload)
            await self._flush_async()
        except BaseException as exc:
            await await_cleanup_after_error(
                self._release_accounting_async(
                    turn,
                    request_index=turn.model_request_count,
                    reason="model_request_setup_failed",
                ),
                exc,
            )
            raise
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

    async def _record_protocol_failure_async(
        self,
        turn: TurnState,
        exc: LLMActionError,
    ) -> ProtocolFailure:
        self.state.json_repair_attempts += exc.repair_attempts
        self.state.errors.extend(
            f"JSON repair attempt: {error}" for error in exc.errors
        )
        turn.errors.extend(
            f"JSON repair attempt: {error}" for error in exc.errors
        )
        usage, cost = await self._record_accounting_async(
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

    async def _record_action_result_async(
        self,
        turn: TurnState,
        result: LLMActionResult,
    ) -> AgentAction:
        action = result.action
        self.state.json_repair_attempts += result.repair_attempts
        self.state.errors.extend(
            f"JSON repair attempt: {error}" for error in result.errors
        )
        turn.errors.extend(
            f"JSON repair attempt: {error}" for error in result.errors
        )
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
        usage, cost = await self._record_accounting_async(
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
        self.reserve_accounting(
            turn,
            request_index=request_index,
            messages=messages,
            purpose="reflection",
        )
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

    async def _start_reflection_async(
        self,
        proposed_answer: str,
        turn: TurnState,
    ) -> tuple[int, list[dict[str, str]], int]:
        turn.reflection_count += 1
        attempt = turn.reflection_count
        messages = build_reflection_messages(turn, proposed_answer)
        turn.model_request_count += 1
        request_index = turn.model_request_count
        try:
            await self._reserve_accounting_async(
                turn,
                request_index=request_index,
                messages=messages,
                purpose="reflection",
            )
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
            await self._flush_async()
        except BaseException as exc:
            await await_cleanup_after_error(
                self._release_accounting_async(
                    turn,
                    request_index=request_index,
                    reason="reflection_setup_failed",
                ),
                exc,
            )
            raise
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

    async def _record_reflection_response_async(
        self,
        turn: TurnState,
        *,
        request_index: int,
        attempt: int,
        raw_response: str,
        response: LLMResponse,
    ) -> None:
        fallback_attempts = getattr(self.llm_client, "last_attempts", None)
        self._record_model_selection_outcome(turn, fallback_attempts)
        usage, cost = await self._record_accounting_async(
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

    async def _reserve_accounting_async(
        self,
        turn: TurnState,
        **kwargs: object,
    ) -> dict | None:
        callback = self.reserve_accounting_async
        if callback is not None:
            return await callback(turn, **kwargs)
        return await asyncio.to_thread(self.reserve_accounting, turn, **kwargs)

    async def _record_accounting_async(
        self,
        turn: TurnState,
        **kwargs: object,
    ) -> tuple[dict | None, dict | None]:
        callback = self.record_accounting_async
        if callback is not None:
            return await callback(turn, **kwargs)
        return await asyncio.to_thread(self.record_accounting, turn, **kwargs)

    async def _release_accounting_async(
        self,
        turn: TurnState,
        **kwargs: object,
    ) -> dict | None:
        callback = self.release_accounting_async
        if callback is not None:
            return await callback(turn, **kwargs)
        return await asyncio.to_thread(self.release_accounting, turn, **kwargs)

    async def _flush_async(self) -> None:
        if self.flush_async is not None:
            await self.flush_async()

    def _record_model_selection_outcome(
        self,
        turn: TurnState,
        attempts: object,
    ) -> None:
        if not isinstance(attempts, (list, tuple)) or not attempts:
            return
        profile_attempts = [
            attempt
            for attempt in attempts
            if isinstance(getattr(attempt, "model_profile_id", None), str)
        ]
        if not profile_attempts:
            return
        serialized = [
            attempt.to_dict()
            if hasattr(attempt, "to_dict")
            else {"attempt": str(attempt)}
            for attempt in profile_attempts
        ]
        turn.extension_metadata["model_attempts"] = serialized
        selection = turn.extension_metadata.get("model_selection")
        if not isinstance(selection, dict):
            return
        successful = next(
            (
                attempt
                for attempt in profile_attempts
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
            1 for attempt in profile_attempts if not getattr(attempt, "success", False)
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

    def _complete_response(
        self,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        kwargs = (
            {"max_output_tokens": self.max_output_tokens}
            if self.max_output_tokens is not None
            else {}
        )
        return call_with_supported_kwargs(
            self.llm_client.complete_response,
            messages,
            **kwargs,
        )

    async def _complete_response_async(
        self,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        kwargs = (
            {"max_output_tokens": self.max_output_tokens}
            if self.max_output_tokens is not None
            else {}
        )
        return await call_async_with_supported_kwargs(
            self.llm_client.acomplete_response,
            messages,
            **kwargs,
        )

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
