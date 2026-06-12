"""Core agent orchestration.

This module will own the main agent loop:
user message -> prompt -> model action -> optional tool call -> observation -> final answer.
"""

from __future__ import annotations

from collections.abc import Callable

from src.core.actions import FinalAnswerAction, ToolCallAction
from src.core.events import TraceEvent
from src.core.observations import format_tool_observation
from src.core.prompt_builder import build_agent_messages
from src.core.prompts import BASE_SYSTEM_PROMPT
from src.core.state import AgentState, ObservationRecord, ToolCallRecord, TurnState
from src.core.trace_format import format_action_trace, format_model_request_trace
from src.llm import LLMActionError, LLMClient
from src.memory import ConversationMemory, MemoryRecord, SQLiteMemoryStore, select_memories_for_prompt
from src.skills import SkillRegistry, SkillSelection
from src.tools import ToolRegistry
from src.tools.registry import ToolResult
from src.tracing import JSONLTraceLogger


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
        event_callback: Callable[[str, dict], None] | None = None,
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
        self.event_callback = event_callback
        self._profile_memories: list[MemoryRecord] = []
        self._relevant_memories: list[MemoryRecord] = []
        self._selected_skills: list[SkillSelection] = []

    def run_turn(self, user_message: str) -> str:
        """Run one user turn and return the assistant response."""
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")

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

        while True:
            messages = self._build_messages()
            turn.model_request_count += 1
            self._trace(
                TraceEvent.MODEL_REQUEST_STARTED,
                format_model_request_trace(
                    messages,
                    max_prompt_chars=self.trace_max_prompt_chars,
                    request_index=turn.model_request_count,
                    loaded_memory_ids=self.state.loaded_memory_ids,
                    loaded_skill_names=self.state.loaded_skill_names,
                    available_tool_names=turn.available_tool_names,
                ),
            )
            try:
                action_result = self.llm_client.complete_action(
                    messages,
                    max_repair_attempts=self.max_json_repair_attempts,
                )
            except LLMActionError as exc:
                self.state.json_repair_attempts += exc.repair_attempts
                self.state.errors.extend(f"JSON repair attempt: {error}" for error in exc.errors)
                turn.errors.extend(f"JSON repair attempt: {error}" for error in exc.errors)
                return self._fail_turn(str(exc), turn)
            action = action_result.action
            self.state.json_repair_attempts += action_result.repair_attempts
            self.state.errors.extend(f"JSON repair attempt: {error}" for error in action_result.errors)
            turn.errors.extend(f"JSON repair attempt: {error}" for error in action_result.errors)
            self._trace(
                TraceEvent.MODEL_RESPONSE,
                {
                    "turn_id": turn.turn_id,
                    "content": action_result.raw_response,
                    "repair_attempts": action_result.repair_attempts,
                    "repair_errors": action_result.errors,
                },
            )
            self._trace(TraceEvent.PARSED_ACTION, format_action_trace(action))
            self._trace(TraceEvent.MODEL_RESPONSE_PARSED, format_action_trace(action))

            if isinstance(action, FinalAnswerAction):
                self.memory.add_assistant_message(action.content)
                self.state.final_answer = action.content
                self.state.messages = self.memory.recent()
                turn.complete(action.content)
                self._trace(TraceEvent.FINAL_ANSWER, {"turn_id": turn.turn_id, "content": action.content})
                self._trace(TraceEvent.TURN_FINISHED, self._state_snapshot(turn))
                return action.content

            if isinstance(action, ToolCallAction):
                if turn.tool_call_count >= self.max_tool_calls_per_turn:
                    return self._fail_turn(
                        f"Tool call limit reached ({self.max_tool_calls_per_turn}) before a final answer.",
                        turn,
                    )
                turn.tool_call_count += 1
                tool_call_record = ToolCallRecord(
                    tool_name=action.tool_name,
                    arguments=action.arguments,
                    iteration=turn.tool_call_count,
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
                        "success": result.success,
                    }
                )
                self._trace(
                    TraceEvent.TOOL_CALL,
                    {
                        "turn_id": turn.turn_id,
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
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

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the model input from prompt, tools, and short-term history."""
        return build_agent_messages(
            system_prompt=self.system_prompt,
            memory=self.memory,
            profile_memories=self._profile_memories,
            relevant_memories=self._relevant_memories,
            selected_skills=self._selected_skills,
            tool_registry=self.tool_registry,
            max_skill_content_chars=self.max_skill_content_chars,
            max_tool_calls_per_turn=self.max_tool_calls_per_turn,
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
        self._selected_skills = self.skill_registry.load_selected_skills(
            user_message,
            limit=self.max_skills_per_turn,
        )
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
        self._trace(TraceEvent.TURN_FAILED, {"turn_id": turn.turn_id if turn else None, "message": message})
        if turn is not None:
            self._trace(TraceEvent.TURN_FINISHED, self._state_snapshot(turn))
        return message

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
