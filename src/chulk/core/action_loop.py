"""Thin sync and async transport drivers for one reduced agent turn."""

from __future__ import annotations

from uuid import uuid4

from chulk.core.action_runtime import ActionLoopPort, AgentTurnCancelled
from chulk.core.actions import AgentAction, PlanStepUpdateAction, parse_model_response
from chulk.core.trace_format import format_action_trace
from chulk.core.events import TraceEvent
from chulk.goals.runtime import GoalSliceExhausted
from chulk.core.model_transport import ProtocolFailure
from chulk.core.state import TurnState
from chulk.core.transitions import (
    ExecuteToolEffect,
    ModelActionSignal,
    PlanStepResultSignal,
    PrepareIterationSignal,
    ProtocolFailureSignal,
    ReflectionResultSignal,
    ToolResultSignal,
    TransitionOutcome,
    TransitionSignal,
    reduce_transition,
)
from chulk.core.turn_effects import (
    PendingReflection,
    PendingToolExecution,
    TransitionApplication,
)


def run_action_loop(runtime: ActionLoopPort, turn: TurnState, *, require_plan: bool) -> str:
    try:
        return _drive_action_loop(runtime, turn, require_plan=require_plan)
    except GoalSliceExhausted as exc:
        return runtime.effects.yield_turn(exc.dimension, turn)


async def run_action_loop_async(runtime: ActionLoopPort, turn: TurnState, *, require_plan: bool) -> str:
    try:
        return await _drive_action_loop_async(runtime, turn, require_plan=require_plan)
    except GoalSliceExhausted as exc:
        result = runtime.effects.yield_turn(exc.dimension, turn)
        await _flush(runtime)
        return result


def _pending_signal(turn: TurnState) -> TransitionSignal | None:
    pending = turn.extension_metadata.get("goal_pending")
    if not pending:
        return None
    if pending["phase"] == "operation_complete":
        turn.extension_metadata.pop("goal_pending", None)
        return None
    if pending["phase"] == "protocol_failure":
        return ProtocolFailureSignal(message=pending["message"])
    if pending["phase"] == "reflection_result":
        return ReflectionResultSignal(**pending["result"])
    payload = dict(pending["action"])
    if payload["type"] == "plan_step_update":
        payload = {"type": payload.pop("type"), "step_update": payload}
    return _model_signal(parse_model_response(payload))


def _checkpoint(runtime: ActionLoopPort, turn: TurnState) -> None:
    if runtime.effects.goal_continuation:
        runtime.effects.trace(TraceEvent.TURN_CHECKPOINTED, runtime.effects.state_snapshot(turn))


def _remember_action(runtime: ActionLoopPort, turn: TurnState, result: AgentAction | ProtocolFailure) -> None:
    if runtime.effects.goal_continuation and not isinstance(result, ProtocolFailure):
        if "goal_pending" not in turn.extension_metadata:
            turn.extension_metadata["goal_pending"] = {"phase": "action", "action": format_action_trace(result), "operation_id": uuid4().hex}
        _checkpoint(runtime, turn)


def _drive_action_loop(
    runtime: ActionLoopPort,
    turn: TurnState,
    *,
    require_plan: bool,
) -> str:
    """Drive a turn through blocking model, tool, and reflection transports."""
    while True:
        _raise_if_cancelled(runtime)
        preparation = _apply_signal(
            runtime,
            turn,
            require_plan=require_plan,
            signal=PrepareIterationSignal(),
        )
        if preparation.outcome == TransitionOutcome.STOP:
            return _response(preparation)
        if preparation.outcome != TransitionOutcome.PROCEED:
            raise RuntimeError("Plan preparation must proceed or stop")

        signal = _pending_signal(turn)
        if signal is None:
            prompt = runtime.model.build_prompt(turn, require_plan=require_plan)
            prompt = runtime.model.compact_prompt(
                prompt,
                turn,
                require_plan=require_plan,
            )
            model_result = runtime.model.request_action(
                turn,
                prompt,
                require_plan=require_plan,
            )
            _remember_action(runtime, turn, model_result)
            signal = _model_signal(model_result)
        _raise_if_cancelled(runtime)
        application = _apply_signal(
            runtime,
            turn,
            require_plan=require_plan,
            signal=signal,
        )
        if application.outcome == TransitionOutcome.AWAIT_RESULT:
            pending = application.pending
            if isinstance(pending, PendingToolExecution):
                result = runtime.tools.execute(
                    pending.effect.action.tool_name,
                    pending.effect.action.arguments,
                    turn,
                )
                _raise_if_cancelled(runtime)
                application = _apply_signal(
                    runtime,
                    turn,
                    require_plan=require_plan,
                    signal=_tool_signal(pending, result),
                    pending=pending,
                    tool_result=result,
                )
            elif isinstance(pending, PendingReflection):
                reflection = runtime.model.reflect(pending.proposed_answer, turn)
                _raise_if_cancelled(runtime)
                result_signal = ReflectionResultSignal(
                    proposed_answer=pending.proposed_answer,
                    approved=reflection.approved,
                    reason=reflection.reason,
                    feedback=reflection.feedback,
                )
                if runtime.effects.goal_continuation:
                    turn.extension_metadata["goal_pending"] = {
                        "phase": "reflection_result",
                        "result": {"proposed_answer": pending.proposed_answer,
                                   "approved": reflection.approved, "reason": reflection.reason,
                                   "feedback": reflection.feedback},
                    }
                    _checkpoint(runtime, turn)
                application = _apply_signal(
                    runtime, turn, require_plan=require_plan, signal=result_signal,
                )
            else:  # pragma: no cover - validated by TurnEffects
                raise RuntimeError("Unknown pending action-loop operation")

        turn.extension_metadata.pop("goal_pending", None)
        _checkpoint(runtime, turn)
        if application.outcome == TransitionOutcome.STOP:
            return _response(application)
        if application.outcome != TransitionOutcome.CONTINUE:
            raise RuntimeError("Action transition must continue, await a result, or stop")


async def _drive_action_loop_async(
    runtime: ActionLoopPort,
    turn: TurnState,
    *,
    require_plan: bool,
) -> str:
    """Drive a turn through async model, tool, and reflection transports."""
    while True:
        _raise_if_cancelled(runtime)
        preparation = await _apply_signal_async(
            runtime,
            turn,
            require_plan=require_plan,
            signal=PrepareIterationSignal(),
        )
        if preparation.outcome == TransitionOutcome.STOP:
            await _flush(runtime)
            return _response(preparation)
        if preparation.outcome != TransitionOutcome.PROCEED:
            raise RuntimeError("Plan preparation must proceed or stop")

        signal = _pending_signal(turn)
        if signal is None:
            prompt = await runtime.model.build_prompt_async(turn, require_plan=require_plan)
            prompt = await runtime.model.compact_prompt_async(
                prompt,
                turn,
                require_plan=require_plan,
            )
            model_result = await runtime.model.request_action_async(
                turn,
                prompt,
                require_plan=require_plan,
            )
            _remember_action(runtime, turn, model_result)
            await _flush(runtime)
            signal = _model_signal(model_result)
        _raise_if_cancelled(runtime)
        application = await _apply_signal_async(
            runtime,
            turn,
            require_plan=require_plan,
            signal=signal,
        )
        if application.outcome == TransitionOutcome.AWAIT_RESULT:
            pending = application.pending
            if isinstance(pending, PendingToolExecution):
                await _flush(runtime)
                result = await runtime.tools.execute_async(
                    pending.effect.action.tool_name,
                    pending.effect.action.arguments,
                    turn,
                )
                _raise_if_cancelled(runtime)
                application = await _apply_signal_async(
                    runtime,
                    turn,
                    require_plan=require_plan,
                    signal=_tool_signal(pending, result),
                    pending=pending,
                    tool_result=result,
                )
            elif isinstance(pending, PendingReflection):
                await _flush(runtime)
                reflection = await runtime.model.reflect_async(
                    pending.proposed_answer,
                    turn,
                )
                _raise_if_cancelled(runtime)
                result_signal = ReflectionResultSignal(
                    proposed_answer=pending.proposed_answer,
                    approved=reflection.approved,
                    reason=reflection.reason,
                    feedback=reflection.feedback,
                )
                if runtime.effects.goal_continuation:
                    turn.extension_metadata["goal_pending"] = {
                        "phase": "reflection_result",
                        "result": {"proposed_answer": pending.proposed_answer,
                                   "approved": reflection.approved, "reason": reflection.reason,
                                   "feedback": reflection.feedback},
                    }
                    _checkpoint(runtime, turn)
                    await _flush(runtime)
                application = await _apply_signal_async(
                    runtime, turn, require_plan=require_plan, signal=result_signal,
                )
            else:  # pragma: no cover - validated by TurnEffects
                raise RuntimeError("Unknown pending action-loop operation")

        turn.extension_metadata.pop("goal_pending", None)
        _checkpoint(runtime, turn)
        if application.outcome == TransitionOutcome.STOP:
            await _flush(runtime)
            return _response(application)
        if application.outcome != TransitionOutcome.CONTINUE:
            raise RuntimeError("Action transition must continue, await a result, or stop")
        await _flush(runtime)


def _apply_signal(
    runtime: ActionLoopPort,
    turn: TurnState,
    *,
    require_plan: bool,
    signal: TransitionSignal,
    pending: PendingToolExecution | None = None,
    tool_result=None,
) -> TransitionApplication:
    snapshot = runtime.effects.snapshot(turn, require_plan=require_plan)
    transition = reduce_transition(snapshot, signal)
    if runtime.effects.goal_continuation and isinstance(transition.effect, ExecuteToolEffect):
        runtime.tools.admit(transition.effect.action.tool_name, turn)
    return runtime.effects.apply(
        turn,
        transition,
        pending=pending,
        tool_result=tool_result,
    )


async def _apply_signal_async(
    runtime: ActionLoopPort,
    turn: TurnState,
    *,
    require_plan: bool,
    signal: TransitionSignal,
    pending: PendingToolExecution | None = None,
    tool_result=None,
) -> TransitionApplication:
    snapshot = runtime.effects.snapshot(turn, require_plan=require_plan)
    transition = reduce_transition(snapshot, signal)
    if runtime.effects.goal_continuation and isinstance(transition.effect, ExecuteToolEffect):
        await runtime.tools.admit_async(transition.effect.action.tool_name, turn)
    return await runtime.effects.apply_async(
        turn,
        transition,
        pending=pending,
        tool_result=tool_result,
    )


async def _flush(runtime: ActionLoopPort) -> None:
    if runtime.async_flush is not None:
        await runtime.async_flush()


def _raise_if_cancelled(runtime: ActionLoopPort) -> None:
    if runtime.is_cancelled():
        raise AgentTurnCancelled("Turn cancelled by the host")


def _model_signal(result: AgentAction | ProtocolFailure) -> TransitionSignal:
    if isinstance(result, ProtocolFailure):
        return ProtocolFailureSignal(message=result.message)
    if isinstance(result, PlanStepUpdateAction):
        return PlanStepResultSignal(action=result)
    return ModelActionSignal(action=result)


def _tool_signal(pending, result) -> ToolResultSignal:
    return ToolResultSignal(
        tool_name=result.tool_name,
        phase=pending.effect.phase,
        success=result.success,
        has_plan_step=pending.plan_step is not None,
        error=result.error,
        failure_kind=result.failure_kind,
        exit_code=result.exit_code,
    )


def _response(application: TransitionApplication) -> str:
    if application.outcome != TransitionOutcome.STOP or application.response is None:
        raise RuntimeError("Only a stopped transition has a response")
    return application.response
