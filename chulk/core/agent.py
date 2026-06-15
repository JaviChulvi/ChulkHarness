"""Core agent orchestration.

This module will own the main agent loop:
user message -> prompt -> model action -> optional tool call -> observation -> final answer.
"""

from __future__ import annotations

from collections.abc import Callable
import re

from chulk.core.actions import FinalAnswerAction, PlanAction, ToolCallAction
from chulk.core.context import AgentPrompt, ContextBudget
from chulk.core.events import TraceEvent
from chulk.core.observations import format_tool_observation
from chulk.core.planning import READ_ONLY_PLANNING_TOOL_NAMES, format_read_only_planning_tools, plan_looks_like_reconnaissance
from chulk.core.prompt_builder import build_agent_prompt
from chulk.core.prompts import BASE_SYSTEM_PROMPT
from chulk.core.reflection import (
    ReflectionParseError,
    ReflectionResult,
    build_reflection_messages,
    parse_reflection_response,
)
from chulk.core.state import AgentState, ObservationRecord, PlanStep, ToolCallRecord, TurnState
from chulk.core.trace_format import format_action_trace, format_model_request_trace
from chulk.llm import LLMActionError, LLMClient, LLMError
from chulk.memory import ConversationMemory, MemoryRecord, SQLiteMemoryStore, select_memories_for_prompt
from chulk.skills import SkillRegistry, SkillSelection
from chulk.tools import ToolRegistry
from chulk.tools.registry import ToolResult
from chulk.tracing import JSONLTraceLogger


MAX_SUMMARY_SOURCE_CHARS = 12000
MAX_SUMMARY_CHARS = 4000
SUMMARY_COMPACTION_PASSES = 3
MAX_UNPARSED_MODEL_OUTPUT_CHARS = 2000


class Agent:
    """Coordinates model calls, memory retrieval, skill loading, and tools."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        state: AgentState | None = None,
        memory: ConversationMemory | None = None,
        memory_store: SQLiteMemoryStore | None = None,
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
        context_budget: ContextBudget | None = None,
        event_callback: Callable[[str, dict], None] | None = None,
        pinned_skill_names: list[str] | None = None,
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
        self.event_callback = event_callback
        self.pinned_skill_names = pinned_skill_names or []
        self._profile_memories: list[MemoryRecord] = []
        self._relevant_memories: list[MemoryRecord] = []
        self._selected_skills: list[SkillSelection] = []
        self.state.conversation_summary = self.memory.conversation_summary

    def run_turn(self, user_message: str) -> str:
        """Run one user turn and return the assistant response."""
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")
        return self._run_user_turn(clean_message, require_plan=False)

    def run_planned_turn(self, user_message: str) -> str:
        """Run one user turn that must propose a plan before tool execution."""
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")
        return self._run_user_turn(clean_message, require_plan=True)

    def _run_user_turn(self, clean_message: str, *, require_plan: bool) -> str:
        """Start a user turn and run it until it completes or waits for approval."""
        if self.has_pending_plan():
            return "A plan is waiting for approval. Use /approve to execute it or /reject to cancel it."

        turn = TurnState(
            user_message=clean_message,
            available_tool_names=[tool.name for tool in self.tool_registry.list_tools()],
        )
        self.state.current_turn_id = turn.turn_id
        self.state.available_tool_names = turn.available_tool_names
        self.state.turns.append(turn)
        self._trace(TraceEvent.TURN_STARTED, {"turn": turn.to_dict()})

        self._extract_long_term_memories(clean_message)
        self._select_long_term_memories(clean_message)
        self._select_skills(clean_message)
        turn.extracted_memory_ids = list(self.state.extracted_memory_ids)
        turn.loaded_memory_ids = list(self.state.loaded_memory_ids)
        turn.loaded_skill_names = list(self.state.loaded_skill_names)
        self.memory.add_user_message(clean_message)
        self._trace(TraceEvent.USER_MESSAGE, {"turn_id": turn.turn_id, "content": clean_message})

        return self._run_action_loop(turn, require_plan=require_plan)

    def has_pending_plan(self) -> bool:
        """Return True when a turn is paused on a plan awaiting approval."""
        turn = self._pending_plan_turn()
        return bool(turn and turn.active_plan and not turn.plan_approved)

    def approve_plan(self) -> str:
        """Approve the pending plan and continue the paused turn."""
        turn = self._pending_plan_turn()
        if turn is None or turn.active_plan is None:
            return "No plan is waiting for approval."

        turn.approve_plan()
        self.state.pending_plan_turn_id = None
        self.state.active_plan = turn.active_plan
        self.memory.add_user_message("User approved the plan. Continue executing the approved plan.")
        self.state.messages = self.memory.recent()
        self._trace(
            TraceEvent.PLAN_APPROVED,
            {"turn_id": turn.turn_id, "plan": turn.active_plan.to_dict()},
        )
        return self._run_action_loop(turn, require_plan=False)

    def reject_plan(self) -> str:
        """Reject the pending plan without executing tools."""
        turn = self._pending_plan_turn()
        if turn is None or turn.active_plan is None:
            return "No plan is waiting for approval."

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
        self._trace(TraceEvent.TURN_FINISHED, self._state_snapshot(turn))
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
        while True:
            prompt = self._build_prompt(turn, require_plan=require_plan)
            prompt = self._compact_context_if_needed(prompt, turn, require_plan=require_plan)
            messages = prompt.messages
            context_report = prompt.context_report.to_dict()
            turn.context_reports.append(context_report)
            self.state.last_context_report = context_report
            max_output_tokens = self._max_output_tokens_for_prompt(context_report)
            turn.model_request_count += 1
            self._trace(
                TraceEvent.MODEL_REQUEST_STARTED,
                format_model_request_trace(
                    messages,
                    max_prompt_chars=self.trace_max_prompt_chars,
                    request_index=turn.model_request_count,
                    turn_id=turn.turn_id,
                    loaded_memory_ids=self.state.loaded_memory_ids,
                    loaded_skill_names=self.state.loaded_skill_names,
                    available_tool_names=turn.available_tool_names,
                    context_report=context_report,
                    max_output_tokens=max_output_tokens,
                ),
            )
            try:
                action_result = self.llm_client.complete_action(
                    messages,
                    max_repair_attempts=self.max_json_repair_attempts,
                    max_output_tokens=max_output_tokens,
                )
            except LLMActionError as exc:
                self.state.json_repair_attempts += exc.repair_attempts
                self.state.errors.extend(f"JSON repair attempt: {error}" for error in exc.errors)
                turn.errors.extend(f"JSON repair attempt: {error}" for error in exc.errors)
                if exc.raw_response:
                    self._trace(
                        TraceEvent.MODEL_RESPONSE,
                        {
                            "turn_id": turn.turn_id,
                            "request_index": turn.model_request_count,
                            "content": exc.raw_response,
                            "repair_attempts": exc.repair_attempts,
                            "repair_errors": exc.errors,
                            "parse_failed": True,
                        },
                    )
                return self._fail_action_protocol_turn(exc, turn)
            action = action_result.action
            self.state.json_repair_attempts += action_result.repair_attempts
            self.state.errors.extend(f"JSON repair attempt: {error}" for error in action_result.errors)
            turn.errors.extend(f"JSON repair attempt: {error}" for error in action_result.errors)
            fallback_attempts = getattr(self.llm_client, "last_attempts", None)
            if fallback_attempts:
                self._trace(
                    TraceEvent.LLM_FALLBACK_ATTEMPTS,
                    {
                        "turn_id": turn.turn_id,
                        "request_index": turn.model_request_count,
                        "attempts": [
                            attempt.to_dict() if hasattr(attempt, "to_dict") else {"attempt": str(attempt)}
                            for attempt in fallback_attempts
                        ],
                    },
                )
            self._trace(
                TraceEvent.MODEL_RESPONSE,
                {
                    "turn_id": turn.turn_id,
                    "request_index": turn.model_request_count,
                    "content": action_result.raw_response,
                    "repair_attempts": action_result.repair_attempts,
                    "repair_errors": action_result.errors,
                },
            )
            self._trace(TraceEvent.PARSED_ACTION, format_action_trace(action))
            self._trace(TraceEvent.MODEL_RESPONSE_PARSED, format_action_trace(action))

            if isinstance(action, PlanAction):
                return self._handle_plan_action(action, turn, require_plan=require_plan)

            if isinstance(action, FinalAnswerAction):
                if require_plan:
                    if turn.planning_feedback_count >= 2:
                        return self._fail_turn("Planning failed because the model answered directly instead of returning a plan.", turn)
                    self._request_plan_revision(
                        turn,
                        feedback=(
                            "Planning feedback: the user explicitly requested /plan, so do not answer directly. "
                            "Use read-only reconnaissance tools if codebase context is needed, then return a plan action "
                            "with concrete implementation steps that can be approved or rejected."
                        ),
                    )
                    return self._run_action_loop(turn, require_plan=True)

                if self._final_answer_needs_revision(action.content, turn):
                    return self._run_action_loop(turn, require_plan=False)

                return self._complete_final_answer(action.content, turn)

            if isinstance(action, ToolCallAction):
                if require_plan:
                    if action.tool_name not in READ_ONLY_PLANNING_TOOL_NAMES:
                        allowed_tools = format_read_only_planning_tools()
                        return self._fail_turn(
                            "Planning can only use read-only reconnaissance tools before approval. "
                            f"Allowed planning tools: {allowed_tools}. "
                            "Return a plan action or retry with one of the allowed tools.",
                            turn,
                        )
                    phase = "planning"
                else:
                    phase = "execution"

                if self._tool_call_count_for_phase(turn, phase) >= self.max_tool_calls_per_turn:
                    if require_plan and phase == "planning" and not turn.planning_tool_limit_feedback_sent:
                        turn.planning_tool_limit_feedback_sent = True
                        self._request_plan_revision(
                            turn,
                            feedback=(
                                "Planning feedback: the read-only reconnaissance tool budget is exhausted. "
                                "Do not call more tools. Return a plan action now using the context already gathered. "
                                "The plan must name concrete files/modules to change, behaviors to add, and tests to update."
                            ),
                        )
                        return self._run_action_loop(turn, require_plan=True)
                    return self._fail_turn(
                        f"Tool call limit reached ({self.max_tool_calls_per_turn}) "
                        f"during {phase} before a final answer.",
                        turn,
                    )
                turn.tool_call_count += 1
                plan_step = None if require_plan else self._start_plan_step(turn)
                tool_call_record = ToolCallRecord(
                    tool_name=action.tool_name,
                    arguments=action.arguments,
                    iteration=turn.tool_call_count,
                    phase=phase,
                )
                turn.tool_calls.append(tool_call_record)
                tool_call_payload = {
                    **tool_call_record.to_dict(),
                    "turn_id": turn.turn_id,
                    "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
                }
                self._trace(TraceEvent.TOOL_CALL_STARTED, tool_call_payload)
                result = self.tool_registry.run(action.tool_name, action.arguments)
                tool_call_record.finish(result)
                self.state.tool_calls.append(
                    {
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
                        "phase": phase,
                        "success": result.success,
                    }
                )
                self._trace(
                    TraceEvent.TOOL_CALL,
                    {
                        "turn_id": turn.turn_id,
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
                        "phase": phase,
                        "success": result.success,
                        "error": result.error,
                    },
                )
                completion_payload = {
                    **tool_call_record.to_dict(),
                    "turn_id": turn.turn_id,
                    "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
                }
                self._trace(
                    TraceEvent.TOOL_CALL_COMPLETED if result.success else TraceEvent.TOOL_CALL_FAILED,
                    completion_payload,
                )
                self._finish_plan_step(turn, plan_step, result)
                observation, output_metadata = self._format_tool_observation(action.tool_name, result)
                self.state.observations.append(
                    {
                        "tool_name": action.tool_name,
                        "observation": observation,
                        "output_metadata": output_metadata,
                    }
                )
                observation_record = ObservationRecord(
                    tool_name=action.tool_name,
                    content=observation,
                    output_metadata=output_metadata,
                )
                turn.observations.append(observation_record)
                self.memory.add_observation(observation)
                self._trace(
                    TraceEvent.TOOL_OBSERVATION,
                    {
                        "turn_id": turn.turn_id,
                        "tool_name": action.tool_name,
                        "observation": observation,
                        "output_metadata": output_metadata,
                    },
                )

    def _build_messages(self, turn: TurnState, *, require_plan: bool) -> list[dict[str, str]]:
        """Build the model input from prompt, tools, and short-term history."""
        return self._build_prompt(turn, require_plan=require_plan).messages

    def _build_prompt(self, turn: TurnState, *, require_plan: bool) -> AgentPrompt:
        """Build the model input and context report."""
        return build_agent_prompt(
            system_prompt=self.system_prompt,
            memory=self.memory,
            profile_memories=self._profile_memories,
            relevant_memories=self._relevant_memories,
            selected_skills=self._selected_skills,
            tool_registry=self.tool_registry,
            max_skill_content_chars=self.max_skill_content_chars,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
            planning_enabled=require_plan or turn.active_plan is not None,
            active_plan=turn.active_plan,
            plan_approved=turn.plan_approved,
            require_plan=require_plan,
            context_budget=self.context_budget,
        )

    def _compact_context_if_needed(self, prompt: AgentPrompt, turn: TurnState, *, require_plan: bool) -> AgentPrompt:
        """Summarize old raw messages that would otherwise be dropped from context."""
        current_prompt = prompt
        for _ in range(SUMMARY_COMPACTION_PASSES):
            pending_messages = self.memory.consume_pending_summary_messages()
            omitted_messages = current_prompt.omitted_messages
            messages_to_summarize = _dedupe_messages([*pending_messages, *omitted_messages])
            if not messages_to_summarize:
                return current_prompt

            summary, fallback, error = self._summarize_context_messages(messages_to_summarize, turn)
            removed_count = self.memory.remove_messages(omitted_messages)
            summarized_message_count = len(pending_messages) + removed_count
            self.memory.update_conversation_summary(summary, summarized_message_count=summarized_message_count)
            self.state.conversation_summary = self.memory.conversation_summary
            self._trace(
                TraceEvent.CONTEXT_SUMMARY_CREATED,
                {
                    "turn_id": turn.turn_id,
                    "summary": self.memory.conversation_summary,
                    "source_message_count": self.memory.summary_message_count,
                    "summarized_message_count": summarized_message_count,
                    "fallback": fallback,
                    "error": error,
                },
            )
            current_prompt = self._build_prompt(turn, require_plan=require_plan)
        return current_prompt

    def _summarize_context_messages(
        self,
        messages: list[dict[str, str]],
        turn: TurnState,
    ) -> tuple[str, bool, str | None]:
        """Return an updated compact conversation summary for older messages."""
        summary_messages = [
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
                    previous_summary=self.memory.conversation_summary,
                    messages=messages,
                ),
            },
        ]
        turn.model_request_count += 1
        request_index = turn.model_request_count
        request_payload = format_model_request_trace(
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
        request_payload["purpose"] = "context_summary"
        request_payload["summary_source_message_count"] = len(messages)
        self._trace(TraceEvent.MODEL_REQUEST_STARTED, request_payload)
        try:
            raw_summary = self.llm_client.complete(summary_messages)
        except LLMError as exc:
            self._trace(
                TraceEvent.MODEL_RESPONSE,
                {
                    "turn_id": turn.turn_id,
                    "request_index": request_index,
                    "content": "",
                    "purpose": "context_summary",
                    "error": str(exc),
                },
            )
            return _fallback_context_summary(self.memory.conversation_summary, messages), True, str(exc)

        self._trace(
            TraceEvent.MODEL_RESPONSE,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "content": raw_summary,
                "purpose": "context_summary",
            },
        )
        clean_summary = _clean_summary(raw_summary)
        if not clean_summary:
            return _fallback_context_summary(self.memory.conversation_summary, messages), True, "empty_summary"
        return clean_summary, False, None

    def _max_output_tokens_for_prompt(self, context_report: dict) -> int | None:
        """Return available output tokens for this request based on prompt size."""
        if not self.context_budget.enabled:
            return None
        prompt_tokens = int(context_report.get("estimated_tokens") or 0)
        remaining_tokens = self.context_budget.max_prompt_tokens - prompt_tokens
        return max(1, remaining_tokens)

    def _handle_plan_action(self, action: PlanAction, turn: TurnState, *, require_plan: bool) -> str:
        if not require_plan:
            return self._fail_turn("Model proposed a new plan after execution had already been approved.", turn)

        plan = action.plan
        if self._plan_needs_revision(plan, turn):
            if turn.planning_feedback_count >= 2:
                return self._fail_turn("Planning failed because the model kept proposing reconnaissance as the plan.", turn)
            self._request_plan_revision(turn, plan=plan)
            return self._run_action_loop(turn, require_plan=True)

        turn.wait_for_plan_approval(plan)
        self.state.active_plan = plan
        self.state.pending_plan_turn_id = turn.turn_id
        response = plan.to_user_text() + "\n\nUse /approve to execute this plan or /reject to cancel it."
        self.memory.add_assistant_message(response)
        self.state.messages = self.memory.recent()
        self._trace(TraceEvent.PLAN_CREATED, {"turn_id": turn.turn_id, "plan": plan.to_dict(), "turn": turn.to_dict()})
        return response

    def _complete_final_answer(self, content: str, turn: TurnState) -> str:
        self.memory.add_assistant_message(content)
        self.state.final_answer = content
        self.state.messages = self.memory.recent()
        turn.complete(content)
        self._clear_active_plan_for_turn(turn)
        self._trace(TraceEvent.FINAL_ANSWER, {"turn_id": turn.turn_id, "content": content})
        self._trace(TraceEvent.TURN_FINISHED, self._state_snapshot(turn))
        return content

    def _final_answer_needs_revision(self, proposed_answer: str, turn: TurnState) -> bool:
        if self.max_reflection_attempts == 0 or turn.reflection_count >= self.max_reflection_attempts:
            return False

        reflection = self._reflect_before_final_answer(proposed_answer, turn)
        if reflection.approved:
            return False

        self._request_reflection_revision(turn, proposed_answer, reflection)
        return True

    def _reflect_before_final_answer(self, proposed_answer: str, turn: TurnState) -> ReflectionResult:
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
        self._trace(
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
        self._trace(TraceEvent.MODEL_REQUEST_STARTED, request_payload)

        try:
            raw_response = self.llm_client.complete(messages)
        except LLMError as exc:
            return self._fail_open_reflection(
                turn,
                proposed_answer,
                attempt=attempt,
                error=str(exc),
                raw_response=None,
                request_index=request_index,
            )

        self._trace(
            TraceEvent.MODEL_RESPONSE,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                "content": raw_response,
                "purpose": "reflection",
                "reflection_attempt": attempt,
            },
        )
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
        self._trace(TraceEvent.REFLECTION_COMPLETED, {"turn_id": turn.turn_id, **record})
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
        self._trace(
            TraceEvent.REFLECTION_FAILED,
            {
                "turn_id": turn.turn_id,
                "request_index": request_index,
                **record,
            },
        )
        return reflection

    def _request_reflection_revision(
        self,
        turn: TurnState,
        proposed_answer: str,
        reflection: ReflectionResult,
    ) -> None:
        feedback = (
            "Reflection feedback: revise the proposed final answer before showing it to the user.\n"
            f"Reason: {reflection.reason}\n"
            f"Instruction: {reflection.feedback}"
        )
        metadata = {
            "reflection_count": turn.reflection_count,
            "approved": reflection.approved,
            "reason": reflection.reason,
            "feedback": reflection.feedback,
            "proposed_answer": proposed_answer,
        }
        self._add_synthetic_observation(
            turn,
            tool_name="reflection_feedback",
            content=feedback,
            output_metadata=metadata,
        )
        self._trace(
            TraceEvent.REFLECTION_REVISION_REQUESTED,
            {
                "turn_id": turn.turn_id,
                **metadata,
            },
        )

    def _plan_needs_revision(self, plan, turn: TurnState) -> bool:
        return plan_looks_like_reconnaissance([(step.title, step.description) for step in plan.steps])

    def _request_plan_revision(self, turn: TurnState, *, plan=None, feedback: str | None = None) -> None:
        turn.planning_feedback_count += 1
        if feedback is None:
            feedback = (
                "Planning feedback: the proposed plan is still mostly reconnaissance. "
                "Do not present read/list/search/explore/inspect steps as the approval plan. "
                "If more context is needed, call read_file or search_files now. "
                "Otherwise return a concrete implementation plan naming the modules/files to change, "
                "the behavior to add, and the tests to update."
            )
        metadata = {"revision_count": turn.planning_feedback_count}
        if plan is not None:
            metadata["rejected_plan"] = plan.to_dict()
        self._add_synthetic_observation(
            turn,
            tool_name="planning_feedback",
            content=feedback,
            output_metadata=metadata,
        )
        self._trace(
            TraceEvent.PLAN_REVISION_REQUESTED,
            {
                "turn_id": turn.turn_id,
                "revision_count": turn.planning_feedback_count,
                "plan": plan.to_dict() if plan is not None else None,
                "feedback": feedback,
            },
        )

    def _add_synthetic_observation(
        self,
        turn: TurnState,
        *,
        tool_name: str,
        content: str,
        output_metadata: dict,
    ) -> None:
        metadata = {**output_metadata, "synthetic": True}
        observation_record = ObservationRecord(
            tool_name=tool_name,
            content=content,
            output_metadata=metadata,
        )
        turn.observations.append(observation_record)
        self.state.observations.append(
            {
                "tool_name": tool_name,
                "observation": content,
                "output_metadata": metadata,
            }
        )
        self.memory.add_observation(content)
        self._trace(
            TraceEvent.TOOL_OBSERVATION,
            {
                "turn_id": turn.turn_id,
                "tool_name": tool_name,
                "observation": content,
                "output_metadata": metadata,
            },
        )

    def _pending_plan_turn(self) -> TurnState | None:
        pending_turn_id = self.state.pending_plan_turn_id
        if pending_turn_id is None:
            return None
        for turn in self.state.turns:
            if turn.turn_id == pending_turn_id:
                return turn
        return None

    def _clear_active_plan_for_turn(self, turn: TurnState) -> None:
        if self.state.pending_plan_turn_id == turn.turn_id:
            self.state.pending_plan_turn_id = None
        if turn.active_plan is not None and self.state.active_plan is turn.active_plan:
            self.state.active_plan = None

    def _tool_call_count_for_phase(self, turn: TurnState, phase: str) -> int:
        return sum(1 for tool_call in turn.tool_calls if tool_call.phase == phase)

    def _start_plan_step(self, turn: TurnState) -> PlanStep | None:
        plan = turn.active_plan
        if plan is None or not turn.plan_approved:
            return None

        step = plan.active_step()
        if step is not None:
            return step

        step = plan.next_pending_step()
        if step is None:
            return None

        step.mark("in_progress")
        self._trace(
            TraceEvent.PLAN_STEP_STARTED,
            {"turn_id": turn.turn_id, "step": step.to_dict(), "plan": plan.to_dict()},
        )
        return step

    def _finish_plan_step(self, turn: TurnState, step: PlanStep | None, result: ToolResult) -> None:
        plan = turn.active_plan
        if plan is None or step is None:
            return

        if result.success:
            step.mark("completed")
            event_type = TraceEvent.PLAN_STEP_COMPLETED
        else:
            step.mark("blocked")
            event_type = TraceEvent.PLAN_STEP_BLOCKED

        self._trace(
            event_type,
            {
                "turn_id": turn.turn_id,
                "step": step.to_dict(),
                "plan": plan.to_dict(),
                "tool_name": result.tool_name,
                "error": result.error,
            },
        )

    def _extract_long_term_memories(self, user_message: str) -> None:
        """Save explicit user-requested memories before retrieval."""
        self.state.extracted_memory_ids = []
        if self.memory_store is None:
            return
        memory_ids = self.memory_store.extract_and_save_memories(user_message)
        self.state.extracted_memory_ids = memory_ids
        if memory_ids:
            self._trace(TraceEvent.MEMORY_EXTRACTION_COMPLETED, {"memory_ids": memory_ids})

    def _select_long_term_memories(self, user_message: str) -> None:
        """Select durable memories that should shape this turn."""
        self._profile_memories = []
        self._relevant_memories = []
        self.state.loaded_memory_ids = []

        if self.memory_store is None:
            return

        self._trace(TraceEvent.MEMORY_SEARCH_STARTED, {"query": user_message})
        profile, relevant = select_memories_for_prompt(self.memory_store, user_message)
        self._profile_memories = profile
        self._relevant_memories = relevant
        self.state.loaded_memory_ids = [memory.id for memory in [*profile, *relevant]]
        self._trace(
            TraceEvent.MEMORY_SEARCH_COMPLETED,
            {
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

        self._trace(TraceEvent.SKILL_SELECTION_STARTED, {"query": user_message})
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

    def _fail_turn(self, message: str, turn: TurnState | None = None) -> str:
        self.state.errors.append(message)
        self.state.final_answer = message
        self.memory.add_assistant_message(message)
        self.state.messages = self.memory.recent()
        if turn is not None:
            turn.fail(message)
            self._clear_active_plan_for_turn(turn)
        self._trace(TraceEvent.TURN_FAILED, {"turn_id": turn.turn_id if turn else None, "message": message})
        if turn is not None:
            self._trace(TraceEvent.TURN_FINISHED, self._state_snapshot(turn))
        return message

    def _fail_action_protocol_turn(self, exc: LLMActionError, turn: TurnState) -> str:
        message = _format_action_protocol_failure(str(exc), exc.raw_response)
        return self._fail_turn(message, turn)

    def _trace(self, event_type: str, payload: dict | None = None) -> None:
        payload = payload or {}
        if self.trace_logger is not None:
            self.trace_logger.log(event_type, payload)
        if self.event_callback is not None:
            self.event_callback(event_type, payload)

    def _state_snapshot(self, turn: TurnState) -> dict:
        return {
            "turn": turn.to_dict(),
            "agent_state": {
                "conversation_id": self.state.conversation_id,
                "current_turn_id": self.state.current_turn_id,
                "message_count": len(self.memory.messages),
                "turn_count": len(self.state.turns),
                "loaded_memory_ids": self.state.loaded_memory_ids,
                "loaded_skill_names": self.state.loaded_skill_names,
                "available_tool_names": self.state.available_tool_names,
                "error_count": len(self.state.errors),
                "final_answer": self.state.final_answer,
                "active_plan": self.state.active_plan.to_dict() if self.state.active_plan else None,
                "pending_plan_turn_id": self.state.pending_plan_turn_id,
                "last_context_report": self.state.last_context_report,
                "conversation_summary": self.state.conversation_summary,
            },
        }

    def _format_tool_observation(self, requested_tool_name: str, result: ToolResult) -> tuple[str, dict]:
        observation, metadata = format_tool_observation(
            requested_tool_name=requested_tool_name,
            result=result,
            max_observation_chars=self.max_observation_chars,
            max_stdout_chars=self.max_tool_stdout_chars,
            max_stderr_chars=self.max_tool_stderr_chars,
            artifact_writer=self._write_tool_output_artifact,
        )
        return observation, metadata

    def _write_tool_output_artifact(self, name: str, content: str) -> dict | None:
        if self.trace_logger is None:
            return None
        return self.trace_logger.write_artifact(name, content)


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
                _format_indented_preview(raw_response, MAX_UNPARSED_MODEL_OUTPUT_CHARS),
            ]
        )
    return "\n".join(lines)


def _format_indented_preview(text: str, max_chars: int) -> str:
    preview = text.strip()
    truncated = len(preview) > max_chars
    if truncated:
        preview = preview[:max_chars].rstrip() + "\n... [truncated]"
    return "\n".join(f"    {line}" if line else "" for line in preview.splitlines())


def _format_context_summary_request(*, previous_summary: str | None, messages: list[dict[str, str]]) -> str:
    sections = []
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


def _fallback_context_summary(previous_summary: str | None, messages: list[dict[str, str]]) -> str:
    parts = []
    if previous_summary:
        parts.append(previous_summary.strip())
    parts.append("Recent compacted context:")
    for message in messages[:8]:
        role = str(message.get("role") or "message")
        content = _compact_summary_line(_clean_summary_source(str(message.get("content") or "")), limit=300)
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
