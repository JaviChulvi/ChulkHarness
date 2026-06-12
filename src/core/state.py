"""Inspectable agent session and turn state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from src.tools.registry import ToolResult


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolCallRecord:
    """Inspectable record for one requested tool call inside a turn."""

    tool_name: str
    arguments: dict
    iteration: int
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    resolved_tool_name: str | None = None
    success: bool | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def finish(self, result: ToolResult) -> None:
        self.ended_at = utc_now()
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
    created_at: str = field(default_factory=utc_now)

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
    started_at: str = field(default_factory=utc_now)
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
        self.ended_at = utc_now()

    def fail(self, message: str) -> None:
        self.status = "failed"
        self.final_answer = message
        self.errors.append(message)
        self.ended_at = utc_now()

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
