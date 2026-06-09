"""Core agent orchestration.

This module will own the main agent loop:
user message -> prompt -> model action -> optional tool call -> observation -> final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from src.core.actions import ActionParseError, FinalAnswerAction, ToolCallAction, parse_model_response
from src.core.prompts import BASE_SYSTEM_PROMPT
from src.core.prompts import JSON_ACTION_PROMPT, format_tools_for_prompt
from src.llm import LLMClient
from src.memory import ConversationMemory
from src.tools import ToolRegistry


@dataclass
class AgentState:
    """Inspectable state for a single agent session."""

    conversation_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[dict] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    loaded_skill_names: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    final_answer: str | None = None


class Agent:
    """Coordinates model calls, memory retrieval, skill loading, and tools."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        state: AgentState | None = None,
        memory: ConversationMemory | None = None,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str = BASE_SYSTEM_PROMPT,
        max_tool_calls_per_turn: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.state = state or AgentState()
        self.memory = memory or ConversationMemory()
        self.tool_registry = tool_registry or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_tool_calls_per_turn = max_tool_calls_per_turn

    def run_turn(self, user_message: str) -> str:
        """Run one user turn and return the assistant response."""
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")

        self.memory.add_user_message(clean_message)
        tool_calls_used = 0

        while True:
            raw_response = self.llm_client.complete(self._build_messages()).strip()
            try:
                action = parse_model_response(raw_response)
            except ActionParseError as exc:
                return self._fail_turn(f"Model response was not valid action JSON: {exc}")

            if isinstance(action, FinalAnswerAction):
                self.memory.add_assistant_message(action.content)
                self.state.final_answer = action.content
                self.state.messages = self.memory.recent()
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
                observation = result.to_observation()
                self.state.observations.append({"tool_name": action.tool_name, "observation": observation})
                self.memory.add_observation(observation)

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the model input from prompt, tools, and short-term history."""
        system_prompt = "\n\n".join(
            [
                self.system_prompt,
                JSON_ACTION_PROMPT,
                format_tools_for_prompt(self.tool_registry.tool_descriptions_for_prompt()),
            ]
        )
        return [{"role": "system", "content": system_prompt}, *self.memory.recent()]

    def _fail_turn(self, message: str) -> str:
        self.state.errors.append(message)
        self.state.final_answer = message
        self.memory.add_assistant_message(message)
        self.state.messages = self.memory.recent()
        return message
