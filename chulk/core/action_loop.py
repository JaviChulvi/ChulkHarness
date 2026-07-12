"""Explicit model-action-tool execution loops for one agent turn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chulk.core.actions import AgentAction, FinalAnswerAction, PlanAction, PlanStepUpdateAction, ToolCallAction
from chulk.core.context import AgentPrompt
from chulk.core.events import TraceEvent
from chulk.core.planning import format_read_only_planning_tools
from chulk.core.state import ObservationRecord, PlanStep, ToolCallRecord, TurnState
from chulk.core.trace_format import format_action_trace, format_model_request_trace
from chulk.llm import LLMActionError, LLMActionResult
from chulk.tools.registry import ToolResult

if TYPE_CHECKING:
    from chulk.core.agent import Agent


def _record_model_request(agent: Agent, turn: TurnState, prompt: AgentPrompt) -> list[dict[str, str]]:
    """Record the exact prompt and accounting metadata before one model request."""
    messages = prompt.messages
    context_report = prompt.context_report.to_dict()
    turn.context_reports.append(context_report)
    agent.state.last_context_report = context_report
    turn.model_request_count += 1
    agent._trace(
        TraceEvent.MODEL_REQUEST_STARTED,
        format_model_request_trace(
            messages,
            max_prompt_chars=agent.trace_max_prompt_chars,
            request_index=turn.model_request_count,
            turn_id=turn.turn_id,
            loaded_memory_ids=agent.state.loaded_memory_ids,
            loaded_skill_names=agent.state.loaded_skill_names,
            available_tool_names=turn.available_tool_names,
            context_report=context_report,
        ),
    )
    return messages


def _record_action_protocol_error(agent: Agent, turn: TurnState, exc: LLMActionError) -> str:
    """Record one invalid model action and finish the turn consistently."""
    agent.state.json_repair_attempts += exc.repair_attempts
    agent.state.errors.extend(f"JSON repair attempt: {error}" for error in exc.errors)
    turn.errors.extend(f"JSON repair attempt: {error}" for error in exc.errors)
    usage_payload, cost_payload = agent._record_model_accounting(
        turn,
        request_index=turn.model_request_count,
        usage=exc.usage,
        cost=exc.cost,
    )
    if exc.raw_response:
        agent._trace(
            TraceEvent.MODEL_RESPONSE,
            {
                "turn_id": turn.turn_id,
                "request_index": turn.model_request_count,
                "content": exc.raw_response,
                "repair_attempts": exc.repair_attempts,
                "repair_errors": exc.errors,
                "parse_failed": True,
                "usage": usage_payload,
                "cost": cost_payload,
            },
        )
    return agent._fail_action_protocol_turn(exc, turn)


def _record_action_result(agent: Agent, turn: TurnState, action_result: LLMActionResult) -> AgentAction:
    """Normalize action accounting and traces shared by sync and async loops."""
    action = action_result.action
    agent.state.json_repair_attempts += action_result.repair_attempts
    agent.state.errors.extend(f"JSON repair attempt: {error}" for error in action_result.errors)
    turn.errors.extend(f"JSON repair attempt: {error}" for error in action_result.errors)
    fallback_attempts = getattr(agent.llm_client, "last_attempts", None)
    if fallback_attempts:
        agent._trace(
            TraceEvent.LLM_FALLBACK_ATTEMPTS,
            {
                "turn_id": turn.turn_id,
                "request_index": turn.model_request_count,
                "attempts": [
                    attempt.to_dict() if hasattr(attempt, "to_dict") else {"attempt": str(attempt)}
                    for attempt in fallback_attempts
                ],
            },
        )
    usage_payload, cost_payload = agent._record_model_accounting(
        turn,
        request_index=turn.model_request_count,
        usage=action_result.usage,
        cost=action_result.cost,
        fallback_attempts=fallback_attempts,
    )
    agent._trace(
        TraceEvent.MODEL_RESPONSE,
        {
            "turn_id": turn.turn_id,
            "request_index": turn.model_request_count,
            "content": action_result.raw_response,
            "repair_attempts": action_result.repair_attempts,
            "repair_errors": action_result.errors,
            "usage": usage_payload,
            "cost": cost_payload,
            "metadata": action_result.metadata,
        },
    )
    agent._trace(TraceEvent.PARSED_ACTION, format_action_trace(action))
    agent._trace(TraceEvent.MODEL_RESPONSE_PARSED, format_action_trace(action))
    return action


def _start_tool_call(
    agent: Agent,
    turn: TurnState,
    action: ToolCallAction,
    *,
    phase: str,
    require_plan: bool,
) -> tuple[ToolCallRecord, PlanStep | None]:
    """Create and trace the durable record before a tool side effect starts."""
    turn.tool_call_count += 1
    plan_step = None if require_plan else agent._active_plan_step_for_tool(turn)
    tool_call_record = ToolCallRecord(
        tool_name=action.tool_name,
        arguments=action.arguments,
        iteration=turn.tool_call_count,
        phase=phase,
        plan_step_id=plan_step.id if plan_step else None,
    )
    turn.tool_calls.append(tool_call_record)
    agent._trace(
        TraceEvent.TOOL_CALL_STARTED,
        {
            **tool_call_record.to_dict(),
            "turn_id": turn.turn_id,
            "max_tool_calls_per_turn": agent.max_tool_calls_per_turn,
        },
    )
    return tool_call_record, plan_step


def _finish_tool_call(
    agent: Agent,
    turn: TurnState,
    action: ToolCallAction,
    *,
    phase: str,
    plan_step: PlanStep | None,
    tool_call_record: ToolCallRecord,
    result: ToolResult,
) -> str | None:
    """Record a completed tool call, its observation, and plan evidence."""
    tool_call_record.finish(result)
    state_tool_call: dict[str, object] = {
        "tool_name": action.tool_name,
        "arguments": action.arguments,
        "phase": phase,
        "success": result.success,
    }
    if tool_call_record.plan_step_id is not None:
        state_tool_call["plan_step_id"] = tool_call_record.plan_step_id
    agent.state.tool_calls.append(state_tool_call)
    agent._trace(
        TraceEvent.TOOL_CALL,
        {
            "turn_id": turn.turn_id,
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            "phase": phase,
            "plan_step_id": tool_call_record.plan_step_id,
            "success": result.success,
            "error": result.error,
        },
    )
    agent._trace(
        TraceEvent.TOOL_CALL_COMPLETED if result.success else TraceEvent.TOOL_CALL_FAILED,
        {
            **tool_call_record.to_dict(),
            "turn_id": turn.turn_id,
            "max_tool_calls_per_turn": agent.max_tool_calls_per_turn,
        },
    )
    observation, output_metadata = agent._format_tool_observation(action.tool_name, result)
    agent.state.observations.append(
        {
            "tool_name": action.tool_name,
            "observation": observation,
            "output_metadata": output_metadata,
        }
    )
    turn.observations.append(
        ObservationRecord(
            tool_name=action.tool_name,
            content=observation,
            output_metadata=output_metadata,
        )
    )
    agent.memory.add_observation(observation)
    agent._trace(
        TraceEvent.TOOL_OBSERVATION,
        {
            "turn_id": turn.turn_id,
            "tool_name": action.tool_name,
            "observation": observation,
            "output_metadata": output_metadata,
        },
    )
    if plan_step is None:
        return None
    if result.success:
        agent._record_plan_tool_evidence(plan_step, tool_call_record, observation, output_metadata)
        return None
    return agent._handle_plan_step_tool_failure(
        turn,
        plan_step,
        tool_call_record,
        result,
        observation,
        output_metadata,
    )


def run_action_loop(agent: Agent, turn: TurnState, *, require_plan: bool) -> str:
    """Run model/tool iterations until the turn pauses or completes."""
    while True:
        blocked_response = agent._prepare_plan_execution_step(turn, require_plan=require_plan)
        if blocked_response is not None:
            return blocked_response

        prompt = agent._build_prompt(turn, require_plan=require_plan)
        prompt = agent._compact_context_if_needed(prompt, turn, require_plan=require_plan)
        messages = _record_model_request(agent, turn, prompt)
        available_tools: list[object] = list(agent.tool_registry.list_tools())
        try:
            action_result = agent.llm_client.complete_action(
                messages,
                max_repair_attempts=agent.max_json_repair_attempts,
                tools=available_tools,
                hosted_mcp_servers=agent.mcp_servers,
                mcp_approval_callback=lambda approval_request: agent._resolve_hosted_mcp_approval(
                    approval_request,
                    turn,
                ),
            )
        except LLMActionError as exc:
            return _record_action_protocol_error(agent, turn, exc)
        action = _record_action_result(agent, turn, action_result)

        if isinstance(action, PlanAction):
            return agent._handle_plan_action(action, turn, require_plan=require_plan)

        if isinstance(action, PlanStepUpdateAction):
            step_update_response = agent._handle_plan_step_update(action, turn, require_plan=require_plan)
            if step_update_response is not None:
                return step_update_response
            continue

        if isinstance(action, FinalAnswerAction):
            if require_plan:
                if turn.planning_feedback_count >= 2:
                    return agent._fail_turn("Planning failed because the model answered directly instead of returning a plan.", turn)
                agent._request_plan_revision(
                    turn,
                    feedback=(
                        "Planning feedback: the user explicitly requested /plan, so do not answer directly. "
                        "Use read-only reconnaissance tools if codebase context is needed, then return a plan action "
                        "with concrete implementation steps that can be approved or rejected."
                    ),
                )
                return agent._run_action_loop(turn, require_plan=True)

            if agent._approved_plan_incomplete(turn):
                if turn.plan_execution_feedback_count >= 1:
                    return agent._fail_turn(
                        "Plan execution failed because the model returned a final answer before completing the approved plan.",
                        turn,
                    )
                agent._request_plan_execution_feedback(
                    turn,
                    feedback=(
                        "Plan execution feedback: the approved plan is not complete. "
                        "Continue the current executable step with a tool call, or return a plan_step_update "
                        "if the step's acceptance criteria are already satisfied. Do not return final_answer yet."
                    ),
                )
                return agent._run_action_loop(turn, require_plan=False)

            if agent._final_answer_needs_revision(action.content, turn):
                return agent._run_action_loop(turn, require_plan=False)

            return agent._complete_final_answer(action.content, turn)

        if isinstance(action, ToolCallAction):
            if require_plan:
                planning_tool_names = agent._read_only_planning_tool_names()
                if action.tool_name not in planning_tool_names:
                    allowed_tools = format_read_only_planning_tools(planning_tool_names)
                    return agent._fail_turn(
                        "Planning can only use read-only reconnaissance tools before approval. "
                        f"Allowed planning tools: {allowed_tools}. "
                        "Return a plan action or retry with one of the allowed tools.",
                        turn,
                    )
                phase = "planning"
            else:
                phase = "execution"

            if agent._tool_call_count_for_phase(turn, phase) >= agent.max_tool_calls_per_turn:
                if require_plan and phase == "planning" and not turn.planning_tool_limit_feedback_sent:
                    turn.planning_tool_limit_feedback_sent = True
                    agent._request_plan_revision(
                        turn,
                        feedback=(
                            "Planning feedback: the read-only reconnaissance tool budget is exhausted. "
                            "Do not call more tools. Return a plan action now using the context already gathered. "
                            "The plan must name concrete files/modules to change, behaviors to add, and tests to update."
                        ),
                    )
                    return agent._run_action_loop(turn, require_plan=True)
                return agent._fail_turn(
                    f"Tool call limit reached ({agent.max_tool_calls_per_turn}) "
                    f"during {phase} before a final answer.",
                    turn,
                )
            tool_call_record, plan_step = _start_tool_call(
                agent,
                turn,
                action,
                phase=phase,
                require_plan=require_plan,
            )
            result = agent._execute_tool_with_retries(action.tool_name, action.arguments, turn)
            blocked_response = _finish_tool_call(
                agent,
                turn,
                action,
                phase=phase,
                plan_step=plan_step,
                tool_call_record=tool_call_record,
                result=result,
            )
            if blocked_response is not None:
                return blocked_response


async def run_action_loop_async(agent: Agent, turn: TurnState, *, require_plan: bool) -> str:
    """Run model/tool iterations, awaiting async tool calls."""
    while True:
        blocked_response = agent._prepare_plan_execution_step(turn, require_plan=require_plan)
        if blocked_response is not None:
            return blocked_response

        prompt = agent._build_prompt(turn, require_plan=require_plan)
        prompt = await agent._compact_context_if_needed_async(prompt, turn, require_plan=require_plan)
        messages = _record_model_request(agent, turn, prompt)
        available_tools: list[object] = list(agent.tool_registry.list_tools())
        try:
            action_result = await agent.llm_client.acomplete_action(
                messages,
                max_repair_attempts=agent.max_json_repair_attempts,
                tools=available_tools,
                hosted_mcp_servers=agent.mcp_servers,
                mcp_approval_callback=lambda approval_request: agent._resolve_hosted_mcp_approval(
                    approval_request,
                    turn,
                ),
            )
        except LLMActionError as exc:
            return _record_action_protocol_error(agent, turn, exc)

        action = _record_action_result(agent, turn, action_result)

        if isinstance(action, PlanAction):
            return await agent._handle_plan_action_async(action, turn, require_plan=require_plan)

        if isinstance(action, PlanStepUpdateAction):
            step_update_response = agent._handle_plan_step_update(action, turn, require_plan=require_plan)
            if step_update_response is not None:
                return step_update_response
            continue

        if isinstance(action, FinalAnswerAction):
            if require_plan:
                if turn.planning_feedback_count >= 2:
                    return agent._fail_turn("Planning failed because the model answered directly instead of returning a plan.", turn)
                agent._request_plan_revision(
                    turn,
                    feedback=(
                        "Planning feedback: the user explicitly requested /plan, so do not answer directly. "
                        "Use read-only reconnaissance tools if codebase context is needed, then return a plan action "
                        "with concrete implementation steps that can be approved or rejected."
                    ),
                )
                return await agent._run_action_loop_async(turn, require_plan=True)

            if agent._approved_plan_incomplete(turn):
                if turn.plan_execution_feedback_count >= 1:
                    return agent._fail_turn(
                        "Plan execution failed because the model returned a final answer before completing the approved plan.",
                        turn,
                    )
                agent._request_plan_execution_feedback(
                    turn,
                    feedback=(
                        "Plan execution feedback: the approved plan is not complete. "
                        "Continue the current executable step with a tool call, or return a plan_step_update "
                        "if the step's acceptance criteria are already satisfied. Do not return final_answer yet."
                    ),
                )
                return await agent._run_action_loop_async(turn, require_plan=False)

            if await agent._final_answer_needs_revision_async(action.content, turn):
                return await agent._run_action_loop_async(turn, require_plan=False)

            return agent._complete_final_answer(action.content, turn)

        if isinstance(action, ToolCallAction):
            if require_plan:
                planning_tool_names = agent._read_only_planning_tool_names()
                if action.tool_name not in planning_tool_names:
                    allowed_tools = format_read_only_planning_tools(planning_tool_names)
                    return agent._fail_turn(
                        "Planning can only use read-only reconnaissance tools before approval. "
                        f"Allowed planning tools: {allowed_tools}. "
                        "Return a plan action or retry with one of the allowed tools.",
                        turn,
                    )
                phase = "planning"
            else:
                phase = "execution"

            if agent._tool_call_count_for_phase(turn, phase) >= agent.max_tool_calls_per_turn:
                if require_plan and phase == "planning" and not turn.planning_tool_limit_feedback_sent:
                    turn.planning_tool_limit_feedback_sent = True
                    agent._request_plan_revision(
                        turn,
                        feedback=(
                            "Planning feedback: the read-only reconnaissance tool budget is exhausted. "
                            "Do not call more tools. Return a plan action now using the context already gathered. "
                            "The plan must name concrete files/modules to change, behaviors to add, and tests to update."
                        ),
                    )
                    return await agent._run_action_loop_async(turn, require_plan=True)
                return agent._fail_turn(
                    f"Tool call limit reached ({agent.max_tool_calls_per_turn}) "
                    f"during {phase} before a final answer.",
                    turn,
                )
            tool_call_record, plan_step = _start_tool_call(
                agent,
                turn,
                action,
                phase=phase,
                require_plan=require_plan,
            )
            result = await agent._execute_tool_with_retries_async(action.tool_name, action.arguments, turn)
            blocked_response = _finish_tool_call(
                agent,
                turn,
                action,
                phase=phase,
                plan_step=plan_step,
                tool_call_record=tool_call_record,
                result=result,
            )
            if blocked_response is not None:
                return blocked_response
