"""Inspectable agent session and turn state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from chulk.tools.registry import ToolResult
from chulk.core.context import TurnContextSection


PLAN_STEP_STATUSES = {"pending", "in_progress", "completed", "blocked"}


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PlanStepEvidence:
    """Evidence that one plan step made progress or satisfied criteria."""

    content: str
    tool_name: str | None = None
    tool_call_iteration: int | None = None
    created_at: str = field(default_factory=utc_now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_call_iteration": self.tool_call_iteration,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class PlanStep:
    """One inspectable step in an agent-authored execution plan."""

    id: str
    title: str
    description: str
    status: str = "pending"
    depends_on: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    retry_limit: int = 0
    evidence: list[PlanStepEvidence] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    blocked_at: str | None = None
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in PLAN_STEP_STATUSES:
            allowed = ", ".join(sorted(PLAN_STEP_STATUSES))
            raise ValueError(f"plan step status must be one of: {allowed}")
        if self.retry_limit < 0:
            raise ValueError("plan step retry_limit cannot be negative")
        if self.retry_limit != 0:
            self.retry_limit = 0
        self.depends_on = _dedupe_strings(self.depends_on)
        self.acceptance_criteria = _dedupe_strings(self.acceptance_criteria) or [self.description]

    def mark(self, status: str) -> None:
        if status not in PLAN_STEP_STATUSES:
            allowed = ", ".join(sorted(PLAN_STEP_STATUSES))
            raise ValueError(f"plan step status must be one of: {allowed}")
        self.status = status
        now = utc_now()
        if status == "in_progress" and self.started_at is None:
            self.started_at = now
        if status == "completed":
            self.completed_at = now
            self.blocked_at = None
            self.blocked_reason = None
        if status == "blocked":
            self.blocked_at = now

    def add_evidence(
        self,
        content: str,
        *,
        tool_name: str | None = None,
        tool_call_iteration: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        clean_content = content.strip()
        if not clean_content:
            return
        self.evidence.append(
            PlanStepEvidence(
                content=clean_content,
                tool_name=tool_name,
                tool_call_iteration=tool_call_iteration,
                metadata=metadata or {},
            )
        )

    def block(self, reason: str) -> None:
        self.blocked_reason = reason.strip() or "Step blocked."
        self.mark("blocked")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "depends_on": self.depends_on,
            "acceptance_criteria": self.acceptance_criteria,
            "retry_limit": self.retry_limit,
            "evidence": [record.to_dict() for record in self.evidence],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "blocked_at": self.blocked_at,
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class Plan:
    """A user-approved plan for one agent turn."""

    summary: str
    steps: list[PlanStep]
    created_at: str = field(default_factory=utc_now)
    approved_at: str | None = None
    rejected_at: str | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"duplicate plan step id: {step.id}")
            missing = [dependency for dependency in step.depends_on if dependency not in seen]
            if missing:
                missing_text = ", ".join(missing)
                raise ValueError(f"plan step {step.id} depends on unknown or later step(s): {missing_text}")
            seen.add(step.id)

    def approve(self) -> None:
        self.approved_at = utc_now()
        self.rejected_at = None

    def reject(self) -> None:
        self.rejected_at = utc_now()

    def next_pending_step(self) -> PlanStep | None:
        return self.next_ready_step()

    def next_ready_step(self) -> PlanStep | None:
        completed_ids = {step.id for step in self.steps if step.status == "completed"}
        for step in self.steps:
            if step.status == "pending" and all(dependency in completed_ids for dependency in step.depends_on):
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
            if step.depends_on:
                lines.append(f"  Depends on: {', '.join(step.depends_on)}")
            lines.append(f"  Acceptance criteria: {'; '.join(step.acceptance_criteria)}")
            if step.evidence:
                latest = step.evidence[-1]
                lines.append(f"  Evidence: {latest.content}")
            if step.blocked_reason:
                lines.append(f"  Blocked reason: {step.blocked_reason}")
        active = self.active_step()
        if active is not None:
            lines.append(f"Current executable step: {active.id} - {active.title}")
        return "\n".join(lines)

    def to_user_text(self) -> str:
        lines = ["Plan", f"  summary  {self.summary}", "  steps"]
        for step in self.steps:
            lines.append(f"  - [{step.status}] {step.title}: {step.description}")
            if step.evidence:
                lines.append(f"    evidence  {step.evidence[-1].content}")
            if step.blocked_reason:
                lines.append(f"    blocked   {step.blocked_reason}")
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
    plan_step_id: str | None = None
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
            "plan_step_id": self.plan_step_id,
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
    context_sections: list[TurnContextSection] = field(default_factory=list)
    prompt_profile: str | None = None
    locale: str | None = None
    extension_metadata: dict = field(default_factory=dict)
    tool_context_metadata: dict = field(default_factory=dict)
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
    model_usage_reports: list[dict] = field(default_factory=list)
    model_usage_totals: dict = field(default_factory=dict)
    plan_execution_feedback_count: int = 0

    def complete(self, final_answer: str) -> None:
        self.status = "completed"
        self.final_answer = final_answer
        self.ended_at = utc_now()

    def fail(self, message: str) -> None:
        self.status = "failed"
        self.final_answer = message
        self.errors.append(message)
        self.ended_at = utc_now()

    def block(self, message: str) -> None:
        self.status = "blocked"
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
            "context_sections": [section.to_dict() for section in self.context_sections],
            "context_section_ids": [section.id for section in self.context_sections],
            "prompt_profile": self.prompt_profile,
            "locale": self.locale,
            "extension_metadata": self.extension_metadata,
            "tool_context_metadata": self.tool_context_metadata,
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
            "model_usage_reports": self.model_usage_reports,
            "model_usage_totals": self.model_usage_totals,
            "plan_execution_feedback_count": self.plan_execution_feedback_count,
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
    last_usage_report: dict | None = None
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
            "last_usage_report": self.last_usage_report,
            "conversation_summary": self.conversation_summary,
        }


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean_value = value.strip() if isinstance(value, str) else ""
        if not clean_value or clean_value in seen:
            continue
        seen.add(clean_value)
        result.append(clean_value)
    return result
