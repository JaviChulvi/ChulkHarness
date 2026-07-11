"""Structured public SDK result snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlanSnapshot:
    """Public snapshot of a plan at the end of an SDK call."""

    summary: str
    status: str
    steps: list[dict]
    created_at: str | None = None
    approved_at: str | None = None
    rejected_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "status": self.status,
            "steps": list(self.steps),
            "created_at": self.created_at,
            "approved_at": self.approved_at,
            "rejected_at": self.rejected_at,
        }


@dataclass(frozen=True)
class RunResult:
    """Structured result for one SDK agent turn."""

    content: str
    status: str
    turn_id: str | None
    conversation_id: str
    trace_path: Path | None
    usage: dict | None = None
    cost: dict | None = None
    context_report: dict | None = None
    tool_calls: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    loaded_skill_names: list[str] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    plan: PlanSnapshot | None = None
    extension_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status,
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "usage": self.usage,
            "cost": self.cost,
            "context_report": self.context_report,
            "tool_calls": list(self.tool_calls),
            "observations": list(self.observations),
            "loaded_skill_names": list(self.loaded_skill_names),
            "loaded_memory_ids": list(self.loaded_memory_ids),
            "errors": list(self.errors),
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "extension_metadata": self.extension_metadata,
        }


@dataclass(frozen=True)
class PlanResult:
    """Structured result for a planning turn awaiting approval or rejection."""

    content: str
    status: str
    plan: PlanSnapshot | None
    turn_id: str | None
    conversation_id: str
    trace_path: Path | None
    context_report: dict | None = None
    loaded_skill_names: list[str] = field(default_factory=list)
    loaded_memory_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "trace_path": str(self.trace_path) if self.trace_path is not None else None,
            "context_report": self.context_report,
            "loaded_skill_names": list(self.loaded_skill_names),
            "loaded_memory_ids": list(self.loaded_memory_ids),
            "errors": list(self.errors),
        }


def plan_snapshot(plan: Any | None) -> PlanSnapshot | None:
    if plan is None:
        return None
    payload = plan.to_dict()
    return PlanSnapshot(
        summary=payload.get("summary") or "",
        status=payload.get("status") or "unknown",
        steps=list(payload.get("steps") or []),
        created_at=payload.get("created_at"),
        approved_at=payload.get("approved_at"),
        rejected_at=payload.get("rejected_at"),
    )


__all__ = ["PlanResult", "PlanSnapshot", "RunResult"]


def run_result_from_runtime(runtime: Any, content: str | None = None) -> RunResult:
    """Build the public run snapshot from a completed internal runtime turn."""
    state = runtime.state
    turn = state.turns[-1] if state.turns else None
    usage_totals = turn.model_usage_totals if turn is not None else state.last_usage_report or {}
    terminal_content = content
    if terminal_content is None and turn is not None:
        terminal_content = turn.final_answer or (turn.errors[-1] if turn.errors else None)
    if terminal_content is None:
        terminal_content = state.final_answer or (state.errors[-1] if state.errors else "")
    trace_logger = getattr(runtime, "trace_logger", None)
    return RunResult(
        content=terminal_content,
        status=turn.status if turn is not None else "unknown",
        turn_id=turn.turn_id if turn is not None else state.current_turn_id,
        conversation_id=state.conversation_id,
        trace_path=getattr(trace_logger, "path", None),
        usage=usage_totals.get("usage") if isinstance(usage_totals, dict) else None,
        cost=usage_totals.get("cost") if isinstance(usage_totals, dict) else None,
        context_report=(turn.context_reports[-1] if turn is not None and turn.context_reports else state.last_context_report),
        tool_calls=[record.to_dict() for record in turn.tool_calls] if turn is not None else [],
        observations=[record.to_dict() for record in turn.observations] if turn is not None else [],
        loaded_skill_names=list(turn.loaded_skill_names) if turn is not None else list(state.loaded_skill_names),
        loaded_memory_ids=list(turn.loaded_memory_ids) if turn is not None else list(state.loaded_memory_ids),
        errors=list(turn.errors) if turn is not None else list(state.errors),
        plan=plan_snapshot(turn.active_plan if turn is not None else state.active_plan),
        extension_metadata=turn.extension_metadata if turn is not None else {},
    )


__all__.append("run_result_from_runtime")
