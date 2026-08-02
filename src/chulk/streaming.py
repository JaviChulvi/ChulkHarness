"""Public policy and terminal contracts for incremental final answers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class FinalAnswerStreamingMode(StrEnum):
    """How public final-answer deltas are produced."""

    VALIDATED = "validated_final_answer"
    INCREMENTAL = "incremental"


class OutputPolicyFailureMode(StrEnum):
    """Behavior when a host output policy raises unexpectedly."""

    OPEN = "fail_open"
    CLOSED = "fail_closed"


class FinalAnswerDeliveryStatus(StrEnum):
    """Terminal state of one user-visible final-answer stream."""

    COMPLETE = "complete"
    TRUNCATED = "safely_truncated"
    BLOCKED = "blocked"
    FAILED = "failed_after_partial"


@dataclass(frozen=True)
class FinalAnswerChunk:
    """One provider chunk before it becomes publicly visible."""

    text: str
    sequence: int
    turn_id: str


@dataclass(frozen=True)
class FinalAnswerPolicyDecision:
    """A host policy decision for a chunk or completion flush."""

    text: str = ""
    stop: bool = False
    blocked: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.blocked and self.text:
            raise ValueError("a blocked output-policy decision cannot include text")


class IncrementalOutputPolicy(Protocol):
    """Sync host hook applied before any incremental output is observable."""

    def process(self, chunk: FinalAnswerChunk) -> FinalAnswerPolicyDecision:
        """Transform, buffer, reject, or stop one provider chunk."""

    def complete(self, *, turn_id: str, next_sequence: int) -> FinalAnswerPolicyDecision:
        """Flush buffered permitted text when the provider completes."""

    def reset(self, *, turn_id: str) -> None:
        """Discard buffered text before a pre-commit provider fallback."""


class AsyncIncrementalOutputPolicy(Protocol):
    """Native async host hook for incremental output."""

    async def process(self, chunk: FinalAnswerChunk) -> FinalAnswerPolicyDecision:
        """Transform, buffer, reject, or stop one provider chunk."""

    async def complete(
        self, *, turn_id: str, next_sequence: int
    ) -> FinalAnswerPolicyDecision:
        """Flush buffered permitted text when the provider completes."""

    async def reset(self, *, turn_id: str) -> None:
        """Discard buffered text before a pre-commit provider fallback."""


class PassThroughOutputPolicy:
    """Default policy that exposes each already-redacted chunk unchanged."""

    def process(self, chunk: FinalAnswerChunk) -> FinalAnswerPolicyDecision:
        return FinalAnswerPolicyDecision(text=chunk.text)

    def complete(self, *, turn_id: str, next_sequence: int) -> FinalAnswerPolicyDecision:
        return FinalAnswerPolicyDecision()

    def reset(self, *, turn_id: str) -> None:
        return None


class AsyncPassThroughOutputPolicy:
    """Native async equivalent of :class:`PassThroughOutputPolicy`."""

    async def process(self, chunk: FinalAnswerChunk) -> FinalAnswerPolicyDecision:
        return FinalAnswerPolicyDecision(text=chunk.text)

    async def complete(
        self, *, turn_id: str, next_sequence: int
    ) -> FinalAnswerPolicyDecision:
        return FinalAnswerPolicyDecision()

    async def reset(self, *, turn_id: str) -> None:
        return None


__all__ = [
    "AsyncIncrementalOutputPolicy",
    "AsyncPassThroughOutputPolicy",
    "FinalAnswerChunk",
    "FinalAnswerDeliveryStatus",
    "FinalAnswerPolicyDecision",
    "FinalAnswerStreamingMode",
    "IncrementalOutputPolicy",
    "OutputPolicyFailureMode",
    "PassThroughOutputPolicy",
]
