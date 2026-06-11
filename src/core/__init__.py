"""Core agent orchestration primitives."""

from typing import Any

__all__ = ["Agent", "AgentState", "ObservationRecord", "ToolCallRecord", "TurnState"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.core.agent import Agent, AgentState, ObservationRecord, ToolCallRecord, TurnState

        return {
            "Agent": Agent,
            "AgentState": AgentState,
            "ObservationRecord": ObservationRecord,
            "ToolCallRecord": ToolCallRecord,
            "TurnState": TurnState,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
