"""Core agent orchestration.

This module will own the main agent loop:
user message -> prompt -> model action -> optional tool call -> observation -> final answer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections.abc import Callable
from uuid import uuid4

from src.core.actions import FinalAnswerAction, ToolCallAction
from src.core.prompts import BASE_SYSTEM_PROMPT
from src.core.prompts import (
    JSON_ACTION_PROMPT,
    format_memories_for_prompt,
    format_skills_for_prompt,
    format_tool_call_rules,
    format_tools_for_prompt,
)
from src.llm import LLMActionError, LLMClient
from src.memory import ConversationMemory, MemoryRecord, SQLiteMemoryStore, select_memories_for_prompt
from src.skills import SkillRegistry, SkillSelection
from src.tools import ToolRegistry
from src.tools.output import TextPreview, preview_text
from src.tools.registry import ToolResult
from src.tracing import JSONLTraceLogger


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolCallRecord:
    """Inspectable record for one requested tool call inside a turn."""

    tool_name: str
    arguments: dict
    iteration: int
    started_at: str = field(default_factory=_utc_now)
    ended_at: str | None = None
    resolved_tool_name: str | None = None
    success: bool | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def finish(self, result: ToolResult) -> None:
        self.ended_at = _utc_now()
        self.resolved_tool_name = result.tool_name
        self.success = result.success
        self.error = result.error
        self.metadata = result.metadata

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "iteration": self.iteration,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "resolved_tool_name": self.resolved_tool_name,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ObservationRecord:
    """Inspectable record for one tool observation added back to context."""

    tool_name: str
    content: str
    output_metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "content": self.content,
            "output_metadata": self.output_metadata,
            "created_at": self.created_at,
        }


@dataclass
class TurnState:
    """Inspectable state for one user turn."""

    user_message: str
    turn_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=_utc_now)
    ended_at: str | None = None
    status: str = "in_progress"
    model_request_count: int = 0
    tool_call_count: int = 0
    available_tool_names: list[str] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    extracted_memory_ids: list[str] = field(default_factory=list)
    loaded_skill_names: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    observations: list[ObservationRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_answer: str | None = None

    def complete(self, final_answer: str) -> None:
        self.status = "completed"
        self.final_answer = final_answer
        self.ended_at = _utc_now()

    def fail(self, message: str) -> None:
        self.status = "failed"
        self.final_answer = message
        self.errors.append(message)
        self.ended_at = _utc_now()

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "model_request_count": self.model_request_count,
            "tool_call_count": self.tool_call_count,
            "available_tool_names": self.available_tool_names,
            "loaded_memory_ids": self.loaded_memory_ids,
            "extracted_memory_ids": self.extracted_memory_ids,
            "loaded_skill_names": self.loaded_skill_names,
            "tool_calls": [record.to_dict() for record in self.tool_calls],
            "observations": [record.to_dict() for record in self.observations],
            "errors": self.errors,
            "final_answer": self.final_answer,
        }


@dataclass
class AgentState:
    """Inspectable state for a single agent session."""

    conversation_id: str = field(default_factory=lambda: str(uuid4()))
    current_turn_id: str | None = None
    messages: list[dict] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    extracted_memory_ids: list[str] = field(default_factory=list)
    loaded_skill_names: list[str] = field(default_factory=list)
    available_tool_names: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    turns: list[TurnState] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_answer: str | None = None
    json_repair_attempts: int = 0

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "current_turn_id": self.current_turn_id,
            "messages": self.messages,
            "loaded_memory_ids": self.loaded_memory_ids,
            "extracted_memory_ids": self.extracted_memory_ids,
            "loaded_skill_names": self.loaded_skill_names,
            "available_tool_names": self.available_tool_names,
            "tool_calls": self.tool_calls,
            "observations": self.observations,
            "turns": [turn.to_dict() for turn in self.turns],
            "errors": self.errors,
            "final_answer": self.final_answer,
            "json_repair_attempts": self.json_repair_attempts,
        }


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
        self._trace("turn_started", {"turn": turn.to_dict()})

        self._extract_long_term_memories(clean_message)
        self._select_long_term_memories(clean_message)
        self._select_skills(clean_message)
        turn.extracted_memory_ids = list(self.state.extracted_memory_ids)
        turn.loaded_memory_ids = list(self.state.loaded_memory_ids)
        turn.loaded_skill_names = list(self.state.loaded_skill_names)
        self.memory.add_user_message(clean_message)
        self._trace("user_message", {"turn_id": turn.turn_id, "content": clean_message})

        while True:
            messages = self._build_messages()
            turn.model_request_count += 1
            self._trace(
                "model_request_started",
                _format_model_request_trace(
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
                "model_response",
                {
                    "turn_id": turn.turn_id,
                    "content": action_result.raw_response,
                    "repair_attempts": action_result.repair_attempts,
                    "repair_errors": action_result.errors,
                },
            )
            self._trace("parsed_action", _format_action_trace(action))
            self._trace("model_response_parsed", _format_action_trace(action))

            if isinstance(action, FinalAnswerAction):
                self.memory.add_assistant_message(action.content)
                self.state.final_answer = action.content
                self.state.messages = self.memory.recent()
                turn.complete(action.content)
                self._trace("final_answer", {"turn_id": turn.turn_id, "content": action.content})
                self._trace("turn_finished", self._state_snapshot(turn))
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
                self._trace("tool_call_started", tool_call_payload)
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
                    "tool_call",
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
                    "tool_call_completed" if result.success else "tool_call_failed",
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
                    "tool_observation",
                    {
                        "turn_id": turn.turn_id,
                        "tool_name": action.tool_name,
                        "observation": observation,
                        "output_metadata": output_metadata,
                    },
                )

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the model input from prompt, tools, and short-term history."""
        system_prompt = "\n\n".join(
            [
                self.system_prompt,
                format_memories_for_prompt(
                    profile_memories=self._profile_memories,
                    relevant_memories=self._relevant_memories,
                ),
                format_skills_for_prompt(
                    self._selected_skills,
                    max_chars_per_skill=self.max_skill_content_chars,
                ),
                JSON_ACTION_PROMPT,
                format_tool_call_rules(self.max_tool_calls_per_turn),
                format_tools_for_prompt(self.tool_registry.tool_descriptions_for_prompt()),
            ]
        )
        return [{"role": "system", "content": system_prompt}, *self.memory.recent()]

    def _extract_long_term_memories(self, user_message: str) -> None:
        """Save explicit user-requested memories before retrieval."""
        self.state.extracted_memory_ids = []
        if self.memory_store is None:
            return
        memory_ids = self.memory_store.extract_and_save_memories(user_message)
        self.state.extracted_memory_ids = memory_ids
        if memory_ids:
            self._trace("memory_extraction_completed", {"memory_ids": memory_ids})

    def _select_long_term_memories(self, user_message: str) -> None:
        """Select durable memories that should shape this turn."""
        self._profile_memories = []
        self._relevant_memories = []
        self.state.loaded_memory_ids = []

        if self.memory_store is None:
            return

        self._trace("memory_search_started", {"query": user_message})
        profile, relevant = select_memories_for_prompt(self.memory_store, user_message)
        self._profile_memories = profile
        self._relevant_memories = relevant
        self.state.loaded_memory_ids = [memory.id for memory in [*profile, *relevant]]
        self._trace(
            "memory_search_completed",
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

        self._trace("skill_selection_started", {"query": user_message})
        self._selected_skills = self.skill_registry.load_selected_skills(
            user_message,
            limit=self.max_skills_per_turn,
        )
        self.state.loaded_skill_names = [selection.skill.name for selection in self._selected_skills]
        self._trace(
            "skill_selection_completed",
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
        self._trace("turn_failed", {"turn_id": turn.turn_id if turn else None, "message": message})
        if turn is not None:
            self._trace("turn_finished", self._state_snapshot(turn))
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
        observation, metadata = _format_tool_observation(
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


def _format_action_trace(action: FinalAnswerAction | ToolCallAction) -> dict:
    if isinstance(action, FinalAnswerAction):
        return {"type": action.type}
    return {"type": action.type, "tool_name": action.tool_name, "arguments": action.arguments}


def _format_model_request_trace(
    messages: list[dict[str, str]],
    *,
    max_prompt_chars: int,
    request_index: int,
    loaded_memory_ids: list[str],
    loaded_skill_names: list[str],
    available_tool_names: list[str],
) -> dict:
    prompt_char_count = sum(len(str(message.get("content", ""))) for message in messages)
    remaining_chars = max_prompt_chars
    traced_messages = []
    returned_char_count = 0

    for message in messages:
        content = str(message.get("content", ""))
        content_char_count = len(content)
        returned_content = content[:remaining_chars] if remaining_chars > 0 else ""
        remaining_chars = max(0, remaining_chars - len(returned_content))
        returned_char_count += len(returned_content)
        traced_messages.append(
            {
                "role": str(message.get("role", "")),
                "content": returned_content,
                "content_char_count": content_char_count,
                "returned_content_char_count": len(returned_content),
                "truncated": len(returned_content) < content_char_count,
            }
        )

    return {
        "request_index": request_index,
        "messages": traced_messages,
        "message_count": len(messages),
        "prompt_char_count": prompt_char_count,
        "returned_prompt_char_count": returned_char_count,
        "max_prompt_chars": max_prompt_chars,
        "truncated": returned_char_count < prompt_char_count,
        "loaded_memory_ids": loaded_memory_ids,
        "loaded_skill_names": loaded_skill_names,
        "available_tool_names": available_tool_names,
    }


def _format_tool_observation(
    *,
    requested_tool_name: str,
    result: ToolResult,
    max_observation_chars: int,
    max_stdout_chars: int,
    max_stderr_chars: int,
    artifact_writer,
) -> tuple[str, dict]:
    status = "success" if result.success else "error"
    parts = [f"Tool {result.tool_name} finished with {status}.", result.observation]
    metadata = {
        "requested_tool_name": requested_tool_name,
        "tool_name": result.tool_name,
        "success": result.success,
        "stdout": None,
        "stderr": None,
        "observation": None,
        "artifacts": [],
    }

    if result.stdout:
        stdout_preview = preview_text(result.stdout, max_stdout_chars)
        metadata["stdout"] = stdout_preview.to_metadata()
        parts.append("stdout:\n" + stdout_preview.text)
        _append_artifact_note(parts, metadata, artifact_writer, result.tool_name, "stdout", result.stdout, stdout_preview)

    if result.stderr:
        stderr_preview = preview_text(result.stderr, max_stderr_chars)
        metadata["stderr"] = stderr_preview.to_metadata()
        parts.append("stderr:\n" + stderr_preview.text)
        _append_artifact_note(parts, metadata, artifact_writer, result.tool_name, "stderr", result.stderr, stderr_preview)

    if result.exit_code is not None:
        parts.append(f"exit_code: {result.exit_code}")
    if result.error:
        parts.append(f"error: {result.error}")

    full_observation = "\n".join(parts)
    observation_preview = preview_text(full_observation, max_observation_chars)
    metadata["observation"] = observation_preview.to_metadata()
    if observation_preview.truncated:
        artifact = artifact_writer(f"{result.tool_name}-observation", full_observation)
        if artifact is not None:
            metadata["artifacts"].append({"field": "observation", **artifact})
            artifact_note = _artifact_note("observation", artifact)
            final_observation = _with_required_suffix(
                full_observation,
                suffix=artifact_note,
                max_chars=max_observation_chars,
            )
        else:
            final_observation = observation_preview.text
    else:
        final_observation = observation_preview.text

    return final_observation, metadata


def _append_artifact_note(
    parts: list[str],
    metadata: dict,
    artifact_writer,
    tool_name: str,
    field: str,
    content: str,
    preview: TextPreview,
) -> None:
    if not preview.truncated:
        return
    artifact = artifact_writer(f"{tool_name}-{field}", content)
    if artifact is None:
        parts.append(f"[full {field} omitted from model context; no artifact writer configured]")
        return
    metadata["artifacts"].append({"field": field, **artifact})
    parts.append(_artifact_note(field, artifact))


def _artifact_note(field: str, artifact: dict) -> str:
    return (
        f"[full {field} saved to {artifact['path']}; "
        f"chars={artifact['char_count']}; sha256={artifact['sha256']}]"
    )


def _with_required_suffix(text: str, *, suffix: str, max_chars: int) -> str:
    separator = "\n"
    suffix_block = separator + suffix
    if len(suffix_block) >= max_chars:
        return suffix_block[-max_chars:]
    preview = preview_text(text, max_chars - len(suffix_block))
    return preview.text + suffix_block
