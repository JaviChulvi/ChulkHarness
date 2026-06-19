"""Core agent orchestration primitives."""

from typing import Any

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentState",
    "ObservationRecord",
    "Plan",
    "PlanStep",
    "PlanStepEvidence",
    "ToolCallRecord",
    "TraceEvent",
    "TurnContextSection",
    "TurnState",
]


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from chulk.core.agent import Agent

        return Agent
    if name in {"AgentState", "ObservationRecord", "Plan", "PlanStep", "PlanStepEvidence", "ToolCallRecord", "TurnState"}:
        from chulk.core.state import AgentState, ObservationRecord, Plan, PlanStep, PlanStepEvidence, ToolCallRecord, TurnState

        return {
            "AgentState": AgentState,
            "ObservationRecord": ObservationRecord,
            "Plan": Plan,
            "PlanStep": PlanStep,
            "PlanStepEvidence": PlanStepEvidence,
            "ToolCallRecord": ToolCallRecord,
            "TurnState": TurnState,
        }[name]
    if name in {"AgentEvent", "TraceEvent"}:
        from chulk.core.events import AgentEvent, TraceEvent

        return {"AgentEvent": AgentEvent, "TraceEvent": TraceEvent}[name]
    if name == "TurnContextSection":
        from chulk.core.context import TurnContextSection

        return TurnContextSection
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
