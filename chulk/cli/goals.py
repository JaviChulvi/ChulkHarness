"""Revision-safe CLI operations for durable goals."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal, InvalidOperation

from chulk.cli.entrypoints import EXIT_OK, EXIT_RUNTIME_ERROR, json_text
from chulk.core.state import TurnState
from chulk.goals import GoalService, GoalStatus
from chulk.sessions import SQLiteSessionStore
from chulk.usage import BudgetScope, ExactCost, RunBudget


def run_goal_command(
    command: str,
    *,
    service: GoalService,
    session_store: SQLiteSessionStore,
    goal_id: str | None = None,
    conversation_id: str | None = None,
    turn_id: str | None = None,
    step_id: str | None = None,
    expected_revision: int | None = None,
    actor: str = "cli",
    instruction: str | None = None,
    reason: str | None = None,
    status: str | None = None,
    limit: int = 100,
    max_model_calls: int | None = None,
    max_tool_calls: int | None = None,
    max_tokens: int | None = None,
    max_cost: str | None = None,
    deadline: str | None = None,
    json_output: bool = False,
    output_func: Callable[[str], None] = print,
    error_func: Callable[[str], None] = print,
) -> int:
    """Run one goal operation with explicit optimistic concurrency."""
    try:
        if command == "list":
            goals = service.store.list(
                status=GoalStatus(status) if status is not None else None,
                limit=limit,
            )
            payload = {
                "ok": True,
                "profile_id": service.store.profile_id,
                "goals": [item.to_dict() for item in goals],
            }
            return _emit(payload, _format_goal_list(goals), json_output, output_func)
        if command == "inspect":
            goal = service.store.get(_required(goal_id, "goal id"))
            payload = {"ok": True, "goal": goal.to_dict()}
            return _emit(payload, _format_goal(goal), json_output, output_func)
        if command == "promote":
            clean_conversation = _required(conversation_id, "conversation id")
            turn = _select_plan_turn(session_store, clean_conversation, turn_id)
            assert turn.active_plan is not None
            goal = service.promote_plan(
                turn.active_plan,
                profile_id=service.store.profile_id,
                conversation_id=clean_conversation,
                turn_id=turn.turn_id,
                budget=_budget(
                    max_model_calls=max_model_calls,
                    max_tool_calls=max_tool_calls,
                    max_tokens=max_tokens,
                    max_cost=max_cost,
                    deadline=deadline,
                ),
                actor=actor,
            )
            payload = {"ok": True, "action": "promoted", "goal": goal.to_dict()}
            return _emit(payload, _format_goal(goal), json_output, output_func)

        clean_goal_id = _required(goal_id, "goal id")
        revision = _revision(expected_revision)
        if command == "approve":
            goal = service.approve(
                clean_goal_id,
                expected_revision=revision,
                approved_by=actor,
                reason=reason,
            )
        elif command == "run":
            goal = service.run(
                clean_goal_id,
                expected_revision=revision,
                actor=actor,
            )
        elif command == "pause":
            goal = service.pause(
                clean_goal_id,
                expected_revision=revision,
                actor=actor,
            )
        elif command == "resume":
            goal = service.resume(
                clean_goal_id,
                expected_revision=revision,
                actor=actor,
            )
        elif command == "steer":
            goal = service.steer(
                clean_goal_id,
                expected_revision=revision,
                instruction=_required(instruction, "steering instruction"),
                created_by=actor,
            )
        elif command == "approve-step":
            goal = service.approve_step(
                clean_goal_id,
                _required(step_id, "step id"),
                expected_revision=revision,
                approved_by=actor,
                reason=reason,
            )
        elif command == "skip-step":
            goal = service.skip_step(
                clean_goal_id,
                _required(step_id, "step id"),
                expected_revision=revision,
                reason=_required(reason, "skip reason"),
                approved_by=actor,
            )
        elif command == "retry-step":
            goal = service.retry_step(
                clean_goal_id,
                _required(step_id, "step id"),
                expected_revision=revision,
                actor=actor,
            )
        elif command == "cancel":
            goal = service.request_cancel(
                clean_goal_id,
                expected_revision=revision,
                actor=actor,
            )
        else:
            raise ValueError(f"Unknown goal command: {command}")
        payload = {"ok": True, "action": command, "goal": goal.to_dict()}
        return _emit(payload, _format_goal(goal), json_output, output_func)
    except (LookupError, RuntimeError, ValueError) as exc:
        payload = {
            "ok": False,
            "status": "goal_error",
            "error": str(exc),
        }
        if json_output:
            output_func(json_text(payload))
        else:
            error_func(f"goal error: {exc}")
        return EXIT_RUNTIME_ERROR


def _select_plan_turn(
    store: SQLiteSessionStore,
    conversation_id: str,
    turn_id: str | None,
) -> TurnState:
    store.get_conversation(conversation_id)
    turns = store.load_turns(conversation_id)
    candidates = [
        item
        for item in turns
        if item.active_plan is not None
        and (turn_id is None or item.turn_id == turn_id)
    ]
    if not candidates:
        suffix = f" and turn {turn_id!r}" if turn_id else ""
        raise ValueError(
            f"no persisted plan found for conversation {conversation_id!r}{suffix}"
        )
    return candidates[-1]


def _budget(
    *,
    max_model_calls: int | None,
    max_tool_calls: int | None,
    max_tokens: int | None,
    max_cost: str | None,
    deadline: str | None,
) -> RunBudget:
    cost = None
    if max_cost is not None:
        try:
            amount = Decimal(max_cost)
        except InvalidOperation as exc:
            raise ValueError("max cost must be an exact decimal amount") from exc
        cost = ExactCost(amount, pricing_known=True)
    parsed_deadline = None
    if deadline is not None:
        try:
            parsed_deadline = datetime.fromisoformat(deadline)
        except ValueError as exc:
            raise ValueError("deadline must be an ISO-8601 timestamp") from exc
        if parsed_deadline.tzinfo is None:
            raise ValueError("deadline must include a timezone")
    return RunBudget(
        scope=BudgetScope.GOAL,
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        max_tokens=max_tokens,
        max_cost=cost,
        deadline=parsed_deadline,
    )


def _revision(value: int | None) -> int:
    if value is None:
        raise ValueError("mutating goal commands require --revision")
    if value < 0:
        raise ValueError("goal revision cannot be negative")
    return value


def _required(value: str | None, label: str) -> str:
    clean = value.strip() if value is not None else ""
    if not clean:
        raise ValueError(f"{label} is required")
    return clean


def _emit(
    payload: dict,
    text: str,
    json_output: bool,
    output_func: Callable[[str], None],
) -> int:
    output_func(json_text(payload) if json_output else text)
    return EXIT_OK


def _format_goal_list(goals: tuple) -> str:
    if not goals:
        return "Goals:\n  no goals"
    lines = ["Goals:"]
    for goal in goals:
        lines.append(
            f"  {goal.id[:12]}  {goal.status.value:<10}  "
            f"r{goal.revision}  {goal.title}"
        )
    return "\n".join(lines)


def _format_goal(goal) -> str:
    lines = [
        f"Goal {goal.id}",
        f"  title      {goal.title}",
        f"  status     {goal.status.value}",
        f"  revision   {goal.revision}",
        f"  profile    {goal.profile_id}",
        f"  cancel     {'requested' if goal.cancellation_requested else 'no'}",
        "  steps",
    ]
    for step in goal.steps:
        lines.append(
            f"    [{step.status.value}] {step.id}: {step.title} "
            f"(attempt {step.attempt}/{step.max_attempts})"
        )
    if goal.missing_criterion_ids:
        lines.append(
            f"  evidence   missing {', '.join(goal.missing_criterion_ids)}"
        )
    else:
        lines.append("  evidence   complete")
    return "\n".join(lines)


__all__ = ["run_goal_command"]
