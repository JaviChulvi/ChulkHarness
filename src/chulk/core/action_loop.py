"""Thin sync and async transport drivers for one reduced agent turn."""

from __future__ import annotations

from chulk.core.action_runtime import ActionLoopPort
from chulk.core.actions import AgentAction, PlanStepUpdateAction
from chulk.core.model_transport import ProtocolFailure
from chulk.core.state import TurnState
from chulk.core.transitions import (
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


def run_action_loop(
    runtime: ActionLoopPort,
    turn: TurnState,
    *,
    require_plan: bool,
) -> str:
    """Drive a turn through blocking model, tool, and reflection transports."""
    while True:
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
        application = _apply_signal(
            runtime,
            turn,
            require_plan=require_plan,
            signal=_model_signal(model_result),
        )
        if application.outcome == TransitionOutcome.AWAIT_RESULT:
            pending = application.pending
            if isinstance(pending, PendingToolExecution):
                result = runtime.tools.execute(
                    pending.effect.action.tool_name,
                    pending.effect.action.arguments,
                    turn,
                )
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
                application = _apply_signal(
                    runtime,
                    turn,
                    require_plan=require_plan,
                    signal=ReflectionResultSignal(
                        proposed_answer=pending.proposed_answer,
                        approved=reflection.approved,
                        reason=reflection.reason,
                        feedback=reflection.feedback,
                    ),
                )
            else:  # pragma: no cover - validated by TurnEffects
                raise RuntimeError("Unknown pending action-loop operation")

        if application.outcome == TransitionOutcome.STOP:
            return _response(application)
        if application.outcome != TransitionOutcome.CONTINUE:
            raise RuntimeError("Action transition must continue, await a result, or stop")


async def run_action_loop_async(
    runtime: ActionLoopPort,
    turn: TurnState,
    *,
    require_plan: bool,
) -> str:
    """Drive a turn through async model, tool, and reflection transports."""
    while True:
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

        prompt = runtime.model.build_prompt(turn, require_plan=require_plan)
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
        application = await _apply_signal_async(
            runtime,
            turn,
            require_plan=require_plan,
            signal=_model_signal(model_result),
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
                application = await _apply_signal_async(
                    runtime,
                    turn,
                    require_plan=require_plan,
                    signal=ReflectionResultSignal(
                        proposed_answer=pending.proposed_answer,
                        approved=reflection.approved,
                        reason=reflection.reason,
                        feedback=reflection.feedback,
                    ),
                )
            else:  # pragma: no cover - validated by TurnEffects
                raise RuntimeError("Unknown pending action-loop operation")

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
    return await runtime.effects.apply_async(
        turn,
        transition,
        pending=pending,
        tool_result=tool_result,
    )


async def _flush(runtime: ActionLoopPort) -> None:
    if runtime.async_flush is not None:
        await runtime.async_flush()


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
