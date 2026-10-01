"""Durable goal models, transitions, persistence, and operator service."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chulk.goals.runner import GoalRunner as GoalRunner, AsyncGoalRunner as AsyncGoalRunner

from chulk.goals.models import (
    Goal,
    GoalActionCheckpoint,
    GoalActionState,
    GoalApproval,
    GoalClaim,
    GoalModelRequest,
    GoalSliceLimits,
    GoalExecutionResult,
    GoalStopReason,
    GoalCriterion,
    GoalEvent,
    GoalEvidence,
    GoalRisk,
    GoalRetentionPolicy,
    GoalStatus,
    GoalSteering,
    GoalStep,
    GoalStepStatus,
    goal_from_dict,
)
from chulk.goals.service import (
    GoalCancellationPropagator,
    GoalEventCallback,
    GoalService,
    PlanLike,
)
from chulk.goals.runtime import GoalExecutionContext
from chulk.goals.store import (
    DEFAULT_GOAL_LEASE_SECONDS,
    GoalActionConflictError,
    GoalLeaseConflictError,
    GoalNotFoundError,
    GoalRevisionConflictError,
    GoalStore,
    SQLiteGoalStore,
)
from chulk.goals.transitions import InvalidGoalTransitionError

__all__ = [
    "DEFAULT_GOAL_LEASE_SECONDS",
    "Goal",
    "GoalActionCheckpoint",
    "GoalActionConflictError",
    "GoalActionState",
    "GoalApproval",
    "GoalCancellationPropagator",
    "GoalClaim",
    "GoalModelRequest",
    "GoalSliceLimits",
    "GoalExecutionResult",
    "GoalStopReason",
    "GoalRunner",
    "AsyncGoalRunner",
    "GoalExecutionStore",
    "AsyncGoalExecutionStore",
    "GoalCriterion",
    "GoalEvent",
    "GoalEventCallback",
    "GoalEvidence",
    "GoalExecutionContext",
    "GoalLeaseConflictError",
    "GoalNotFoundError",
    "GoalRevisionConflictError",
    "GoalRisk",
    "GoalRetentionPolicy",
    "GoalService",
    "GoalStatus",
    "GoalSteering",
    "GoalStep",
    "GoalStepStatus",
    "GoalStore",
    "InvalidGoalTransitionError",
    "PlanLike",
    "SQLiteGoalStore",
    "goal_from_dict",
]


from chulk.goals.protocols import GoalExecutionStore, AsyncGoalExecutionStore


def __getattr__(name: str):
    if name in {"GoalRunner", "AsyncGoalRunner"}:
        from chulk.goals import runner
        return getattr(runner, name)
    raise AttributeError(name)
