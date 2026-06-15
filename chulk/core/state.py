"""Inspectable agent session and turn state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from chulk.tools.registry import ToolResult


PLAN_STEP_STATUSES = {"pending", "in_progress", "completed", "blocked"}


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PlanStep:
    """One inspectable step in an agent-authored execution plan."""

    id: str
    title: str
    description: str
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.status not in PLAN_STEP_STATUSES:
            allowed = ", ".join(sorted(PLAN_STEP_STATUSES))
            raise ValueError(f"plan step status must be one of: {allowed}")

    def mark(self, status: str) -> None:
        if status not in PLAN_STEP_STATUSES:
            allowed = ", ".join(sorted(PLAN_STEP_STATUSES))
            raise ValueError(f"plan step status must be one of: {allowed}")
        self.status = status

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
        }


@dataclass
class Plan:
    """A user-approved plan for one agent turn."""

    summary: str
    steps: list[PlanStep]
    created_at: str = field(default_factory=utc_now)
    approved_at: str | None = None
    rejected_at: str | None = None

    def approve(self) -> None:
        self.approved_at = utc_now()
        self.rejected_at = None

    def reject(self) -> None:
        self.rejected_at = utc_now()

    def next_pending_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status == "pending":
                return step
        return None

    def active_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status == "in_progress":
                return step
        return None

    def status(self) -> str:
        if self.rejected_at is not None:
            return "rejected"
        if self.approved_at is None:
            return "pending_approval"
        if any(step.status == "blocked" for step in self.steps):
            return "blocked"
        if self.steps and all(step.status == "completed" for step in self.steps):
            return "completed"
        return "approved"

    def to_prompt(self) -> str:
        lines = [f"Plan summary: {self.summary}", "Plan steps:"]
        for step in self.steps:
            lines.append(f"- [{step.status}] {step.id}: {step.title} - {step.description}")
        return "\n".join(lines)

    def to_user_text(self) -> str:
        lines = ["Plan", f"  summary  {self.summary}", "  steps"]
        lines.extend(f"  - [{step.status}] {step.title}: {step.description}" for step in self.steps)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
            "approved_at": self.approved_at,
            "rejected_at": self.rejected_at,
            "status": self.status(),
        }


@dataclass
class ToolCallRecord:
    """Inspectable record for one requested tool call inside a turn."""

    tool_name: str
    arguments: dict
    iteration: int
    phase: str = "execution"
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
            "phase": self.phase,
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
    active_plan: Plan | None = None
    plan_approved: bool = False
    planning_feedback_count: int = 0
    planning_tool_limit_feedback_sent: bool = False
    reflection_count: int = 0
    reflections: list[dict] = field(default_factory=list)
    context_reports: list[dict] = field(default_factory=list)

    def complete(self, final_answer: str) -> None:
        self.status = "completed"
        self.final_answer = final_answer
        self.ended_at = utc_now()

    def fail(self, message: str) -> None:
        self.status = "failed"
        self.final_answer = message
        self.errors.append(message)
        self.ended_at = utc_now()

    def wait_for_plan_approval(self, plan: Plan) -> None:
        self.active_plan = plan
        self.plan_approved = False
        self.status = "waiting_for_approval"

    def approve_plan(self) -> None:
        if self.active_plan is not None:
            self.active_plan.approve()
        self.plan_approved = True
        self.status = "in_progress"

    def reject_plan(self, message: str) -> None:
        if self.active_plan is not None:
            self.active_plan.reject()
        self.plan_approved = False
        self.status = "plan_rejected"
        self.final_answer = message
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
            "active_plan": self.active_plan.to_dict() if self.active_plan else None,
            "plan_approved": self.plan_approved,
            "planning_feedback_count": self.planning_feedback_count,
            "planning_tool_limit_feedback_sent": self.planning_tool_limit_feedback_sent,
            "reflection_count": self.reflection_count,
            "reflections": self.reflections,
            "context_reports": self.context_reports,
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
    active_plan: Plan | None = None
    pending_plan_turn_id: str | None = None
    last_context_report: dict | None = None
    conversation_summary: str | None = None

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
            "active_plan": self.active_plan.to_dict() if self.active_plan else None,
            "pending_plan_turn_id": self.pending_plan_turn_id,
            "last_context_report": self.last_context_report,
            "conversation_summary": self.conversation_summary,
        }
