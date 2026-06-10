"""Core agent orchestration.

This module will own the main agent loop:
user message -> prompt -> model action -> optional tool call -> observation -> final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from src.core.actions import FinalAnswerAction, ToolCallAction
from src.core.prompts import BASE_SYSTEM_PROMPT
from src.core.prompts import JSON_ACTION_PROMPT, format_memories_for_prompt, format_skills_for_prompt, format_tools_for_prompt
from src.llm import LLMActionError, LLMClient
from src.memory import ConversationMemory, MemoryRecord, SQLiteMemoryStore, select_memories_for_prompt
from src.skills import SkillRegistry, SkillSelection
from src.tools import ToolRegistry
from src.tracing import JSONLTraceLogger


@dataclass
class AgentState:
    """Inspectable state for a single agent session."""

    conversation_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[dict] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    extracted_memory_ids: list[str] = field(default_factory=list)
    loaded_skill_names: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_answer: str | None = None
    json_repair_attempts: int = 0


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
    ) -> None:
        if max_json_repair_attempts < 0:
            raise ValueError("max_json_repair_attempts cannot be negative")
        if max_skills_per_turn < 1:
            raise ValueError("max_skills_per_turn must be greater than zero")
        if max_skill_content_chars < 1:
            raise ValueError("max_skill_content_chars must be greater than zero")
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
        self._profile_memories: list[MemoryRecord] = []
        self._relevant_memories: list[MemoryRecord] = []
        self._selected_skills: list[SkillSelection] = []

    def run_turn(self, user_message: str) -> str:
        """Run one user turn and return the assistant response."""
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")

        self._extract_long_term_memories(clean_message)
        self._select_long_term_memories(clean_message)
        self._select_skills(clean_message)
        self.memory.add_user_message(clean_message)
        self._trace("user_message", {"content": clean_message})
        tool_calls_used = 0

        while True:
            try:
                action_result = self.llm_client.complete_action(
                    self._build_messages(),
                    max_repair_attempts=self.max_json_repair_attempts,
                )
            except LLMActionError as exc:
                self.state.json_repair_attempts += exc.repair_attempts
                self.state.errors.extend(f"JSON repair attempt: {error}" for error in exc.errors)
                return self._fail_turn(str(exc))
            action = action_result.action
            self.state.json_repair_attempts += action_result.repair_attempts
            self.state.errors.extend(f"JSON repair attempt: {error}" for error in action_result.errors)
            self._trace(
                "model_response",
                {
                    "content": action_result.raw_response,
                    "repair_attempts": action_result.repair_attempts,
                    "repair_errors": action_result.errors,
                },
            )
            self._trace("parsed_action", _format_action_trace(action))

            if isinstance(action, FinalAnswerAction):
                self.memory.add_assistant_message(action.content)
                self.state.final_answer = action.content
                self.state.messages = self.memory.recent()
                self._trace("final_answer", {"content": action.content})
                return action.content

            if isinstance(action, ToolCallAction):
                if tool_calls_used >= self.max_tool_calls_per_turn:
                    return self._fail_turn(
                        f"Tool call limit reached ({self.max_tool_calls_per_turn}) before a final answer."
                    )
                tool_calls_used += 1
                result = self.tool_registry.run(action.tool_name, action.arguments)
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
                        "tool_name": action.tool_name,
                        "arguments": action.arguments,
                        "success": result.success,
                        "error": result.error,
                    },
                )
                observation = result.to_observation()
                self.state.observations.append({"tool_name": action.tool_name, "observation": observation})
                self.memory.add_observation(observation)
                self._trace("tool_observation", {"tool_name": action.tool_name, "observation": observation})

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

    def _fail_turn(self, message: str) -> str:
        self.state.errors.append(message)
        self.state.final_answer = message
        self.memory.add_assistant_message(message)
        self.state.messages = self.memory.recent()
        self._trace("turn_failed", {"message": message})
        return message

    def _trace(self, event_type: str, payload: dict | None = None) -> None:
        if self.trace_logger is not None:
            self.trace_logger.log(event_type, payload or {})


def _format_action_trace(action: FinalAnswerAction | ToolCallAction) -> dict:
    if isinstance(action, FinalAnswerAction):
        return {"type": action.type}
    return {"type": action.type, "tool_name": action.tool_name, "arguments": action.arguments}
