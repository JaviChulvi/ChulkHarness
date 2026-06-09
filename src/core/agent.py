"""Core agent orchestration.

This module will own the main agent loop:
user message -> prompt -> model action -> optional tool call -> observation -> final answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from src.core.prompts import BASE_SYSTEM_PROMPT
from src.llm import LLMClient
from src.memory import ConversationMemory


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


class Agent:
    """Coordinates model calls, memory retrieval, skill loading, and tools."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        state: AgentState | None = None,
        memory: ConversationMemory | None = None,
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ) -> None:
        self.llm_client = llm_client
        self.state = state or AgentState()
        self.memory = memory or ConversationMemory()
        self.system_prompt = system_prompt

    def run_turn(self, user_message: str) -> str:
        """Run one user turn and return the assistant response."""
        clean_message = user_message.strip()
        if not clean_message:
            raise ValueError("user_message cannot be empty")

        self.memory.add_user_message(clean_message)
        messages = self._build_messages()
        assistant_response = self.llm_client.complete(messages).strip()
        self.memory.add_assistant_message(assistant_response)
        self.state.messages = self.memory.recent()
        return assistant_response

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the Phase 1 model input from prompt and short-term history."""
        return [{"role": "system", "content": self.system_prompt}, *self.memory.recent()]
