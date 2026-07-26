"""Small immutable view models for the operator terminal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


Record = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """One locally observed conversation event."""

    kind: str
    text: str
    status: str = "info"


@dataclass(frozen=True, slots=True)
class OperatorSnapshot:
    """Bounded API state rendered by one TUI refresh."""

    profile_id: str
    selected_conversation_id: str | None = None
    profiles: tuple[Record, ...] = ()
    conversations: tuple[Record, ...] = ()
    permissions: tuple[Record, ...] = ()
    proposals: tuple[Record, ...] = ()
    goals: tuple[Record, ...] = ()
    tasks: tuple[Record, ...] = ()
    jobs: tuple[Record, ...] = ()
    usage: tuple[Record, ...] = ()
    traces: tuple[Record, ...] = ()
    artifacts: tuple[Record, ...] = ()
    timeline: tuple[TimelineEntry, ...] = ()
    errors: Mapping[str, str] = field(default_factory=dict)

    @property
    def attention_count(self) -> int:
        return len(self.permissions) + len(self.proposals)

    @property
    def active_work_count(self) -> int:
        terminal = {
            "completed",
            "cancelled",
            "failed",
            "rejected",
            "expired",
            "budget_exhausted",
        }
        return sum(
            str(item.get("status", "")).lower() not in terminal
            for group in (self.goals, self.tasks, self.jobs)
            for item in group
        )


__all__ = ["OperatorSnapshot", "Record", "TimelineEntry"]
