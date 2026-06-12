"""Core agent orchestration primitives."""

from typing import Any

__all__ = [
    "Agent",
    "AgentState",
    "ObservationRecord",
    "Plan",
    "PlanStep",
    "ToolCallRecord",
    "TraceEvent",
    "TurnState",
]


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from src.core.agent import Agent

        return Agent
    if name in {"AgentState", "ObservationRecord", "Plan", "PlanStep", "ToolCallRecord", "TurnState"}:
        from src.core.state import AgentState, ObservationRecord, Plan, PlanStep, ToolCallRecord, TurnState

        return {
            "AgentState": AgentState,
            "ObservationRecord": ObservationRecord,
            "Plan": Plan,
            "PlanStep": PlanStep,
            "ToolCallRecord": ToolCallRecord,
            "TurnState": TurnState,
        }[name]
    if name == "TraceEvent":
        from src.core.events import TraceEvent

        return TraceEvent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
