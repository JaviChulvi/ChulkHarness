"""Core agent orchestration.

This module will own the main agent loop:
user message -> prompt -> model action -> optional tool call -> observation -> final answer.
"""

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Inspectable state for a single agent session."""

    conversation_id: str
    messages: list[dict] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    loaded_skill_names: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Agent:
    """Coordinates model calls, memory retrieval, skill loading, and tools."""

    def __init__(self, state: AgentState) -> None:
        self.state = state

    def run_turn(self, user_message: str) -> str:
        """Run one user turn.

        Phase 1 will replace this placeholder with the first chat loop.
        """
        self.state.messages.append({"role": "user", "content": user_message})
        return "Agent loop is not implemented yet. See TODO.md."
