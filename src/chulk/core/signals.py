"""Internal control-flow signals shared by orchestration boundaries."""

from __future__ import annotations

from typing import Any


class DurableApprovalPaused(RuntimeError):
    """A hosted run paused durably and released its current worker."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        super().__init__(str(outcome.reason))


__all__ = ["DurableApprovalPaused"]
