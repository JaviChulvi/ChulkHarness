"""Explicit persistence bindings for local and hosted goal execution."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, Protocol
from chulk.goals.models import Goal, GoalSliceAdmission, GoalModelRequest, GoalActionCheckpoint, GoalClaim, GoalStopReason, GoalActionState

GoalMutation = Callable[[Goal], Goal]


class GoalExecutionStore(Protocol):
    """Profile-scoped transactional goal/claim/continuation persistence."""

    profile_id: str

    def get(self, goal_id: str) -> Goal:
        ...

    def mutate(self, goal_id: str, *, expected_revision: int, kind: str, actor: str, mutation: GoalMutation, payload: Mapping[str, Any] | None=None, claim: GoalClaim | None=None) -> Goal:
        ...

    def heartbeat(self, claim: GoalClaim, *, lease_seconds: int=120, now: datetime | None=None) -> GoalClaim:
        ...

    def release_claim(self, claim: GoalClaim) -> bool:
        ...

    def assert_action_boundary(self, claim: GoalClaim, *, step_id: str, now: datetime | None=None) -> Goal:
        ...

    def begin_model_request(self, claim: GoalClaim, *, step_id: str, conversation_id: str, turn_id: str, request_index: int, purpose: str, goal_revision: int, steering_ids: tuple[str, ...]) -> GoalModelRequest:
        ...

    def acknowledge_model_response(self, claim: GoalClaim, request_id: str, *, response_ref: str) -> GoalModelRequest:
        ...

    def incorporated_steering_ids(self, goal_id: str) -> frozenset[str]:
        ...

    def begin_action(self, claim: GoalClaim, *, step_id: str, idempotency_key: str, action_kind: str, action_ref: str | None=None, now: datetime | None=None, recover_existing: bool=False) -> GoalActionCheckpoint:
        ...

    def link_action_effect(self, claim: GoalClaim, checkpoint_id: str, effect_id: str) -> None: ...

    def finish_action(self, claim: GoalClaim, checkpoint_id: str, *, result: Mapping[str, Any] | None=None, error: str | None=None, now: datetime | None=None) -> GoalActionCheckpoint:
        ...

    def verification(self, claim: GoalClaim, operation_id: str) -> dict[str, Any] | None: ...

    def record_verification(self, claim: GoalClaim, *, step_id: str, operation_id: str, expected_revision: int, context_digest: str, evidence_digest: str, passed: bool, feedback: str) -> int:
        ...

    def execution_state(self, goal_id: str) -> dict[str, Any] | None:
        ...

    def admit_slice(self, goal_id: str, *, runner_id: str, expected_revision: int, turn_id: str, lease_seconds: int=120) -> GoalSliceAdmission:
        ...

    def mark_slice_started(self, claim: GoalClaim, turn_id: str) -> None: ...

    def finish_slice(self, claim: GoalClaim, *, turn_id: str, reason: GoalStopReason, usage: Mapping[str, Any], exhausted_budget: Mapping[str, Any] | None=None) -> None:
        ...

    def apply_verified_step(self, claim: GoalClaim, *, operation_id: str, expected_revision: int, usage: Mapping[str, Any] | None = None) -> Goal:
        ...

    def action_checkpoints(self, goal_id: str, *, state: GoalActionState | str | None=None) -> tuple[GoalActionCheckpoint, ...]:
        ...


class AsyncGoalExecutionStore(Protocol):
    """Profile-scoped transactional goal/claim/continuation persistence."""

    profile_id: str

    async def get(self, goal_id: str) -> Goal:
        ...

    async def mutate(self, goal_id: str, *, expected_revision: int, kind: str, actor: str, mutation: GoalMutation, payload: Mapping[str, Any] | None=None, claim: GoalClaim | None=None) -> Goal:
        ...

    async def heartbeat(self, claim: GoalClaim, *, lease_seconds: int=120, now: datetime | None=None) -> GoalClaim:
        ...

    async def release_claim(self, claim: GoalClaim) -> bool:
        ...

    async def assert_action_boundary(self, claim: GoalClaim, *, step_id: str, now: datetime | None=None) -> Goal:
        ...

    async def begin_model_request(self, claim: GoalClaim, *, step_id: str, conversation_id: str, turn_id: str, request_index: int, purpose: str, goal_revision: int, steering_ids: tuple[str, ...]) -> GoalModelRequest:
        ...

    async def acknowledge_model_response(self, claim: GoalClaim, request_id: str, *, response_ref: str) -> GoalModelRequest:
        ...

    async def incorporated_steering_ids(self, goal_id: str) -> frozenset[str]:
        ...

    async def begin_action(self, claim: GoalClaim, *, step_id: str, idempotency_key: str, action_kind: str, action_ref: str | None=None, now: datetime | None=None, recover_existing: bool=False) -> GoalActionCheckpoint:
        ...

    async def link_action_effect(self, claim: GoalClaim, checkpoint_id: str, effect_id: str) -> None: ...

    async def finish_action(self, claim: GoalClaim, checkpoint_id: str, *, result: Mapping[str, Any] | None=None, error: str | None=None, now: datetime | None=None) -> GoalActionCheckpoint:
        ...

    async def verification(self, claim: GoalClaim, operation_id: str) -> dict[str, Any] | None: ...

    async def record_verification(self, claim: GoalClaim, *, step_id: str, operation_id: str, expected_revision: int, context_digest: str, evidence_digest: str, passed: bool, feedback: str) -> int:
        ...

    async def execution_state(self, goal_id: str) -> dict[str, Any] | None:
        ...

    async def admit_slice(self, goal_id: str, *, runner_id: str, expected_revision: int, turn_id: str, lease_seconds: int=120) -> GoalSliceAdmission:
        ...

    async def mark_slice_started(self, claim: GoalClaim, turn_id: str) -> None: ...

    async def finish_slice(self, claim: GoalClaim, *, turn_id: str, reason: GoalStopReason, usage: Mapping[str, Any], exhausted_budget: Mapping[str, Any] | None=None) -> None:
        ...

    async def apply_verified_step(self, claim: GoalClaim, *, operation_id: str, expected_revision: int, usage: Mapping[str, Any] | None = None) -> Goal:
        ...

    async def action_checkpoints(self, goal_id: str, *, state: GoalActionState | str | None=None) -> tuple[GoalActionCheckpoint, ...]:
        ...
