"""Session restoration and recovery for runtime assembly."""

from __future__ import annotations

from collections.abc import Callable

from chulk.core import AgentState, TurnState
from chulk.hosting import ExecutionScope
from chulk.hosting.async_utils import call_async_service
from chulk.redaction import redact_text
from chulk.sessions import ConversationSummaryRecord, SQLiteSessionStore


def session_result_redactor(
    callback: Callable[[str, str, dict], str] | None,
    *,
    fail_closed: bool,
) -> Callable[[str], str]:
    """Compose baseline secret redaction with an optional host policy."""

    def redact(value: str) -> str:
        safe_value = redact_text(value)
        if callback is None:
            return safe_value
        try:
            custom_value = callback(
                "session_search_result",
                safe_value,
                {"source": "session_search"},
            )
        except Exception:
            return "[redaction failed]" if fail_closed else safe_value
        return redact_text(str(custom_value))

    return redact


def summary_source_ordinal(summary: ConversationSummaryRecord | None) -> int:
    """Return the durable ordinal covered by a logical conversation summary."""
    if summary is None:
        return 0
    source_ordinal = summary.metadata.get("source_message_ordinal")
    if (
        isinstance(source_ordinal, int)
        and not isinstance(source_ordinal, bool)
        and source_ordinal >= 0
    ):
        return source_ordinal
    return summary.source_message_count


def create_agent_state(
    session_store: SQLiteSessionStore,
    conversation_id: str | None,
    *,
    execution_scope: ExecutionScope | None = None,
    unresolved_tool_handler: Callable[
        [
            SQLiteSessionStore,
            str,
            TurnState,
            list[dict[str, object]],
        ],
        None,
    ]
    | None = None,
) -> AgentState:
    """Create fresh state or rebuild state for an existing conversation."""
    if conversation_id is None:
        return AgentState()

    conversation = session_store.get_conversation(conversation_id)
    if execution_scope is not None:
        raw_scope = conversation.metadata.get("execution_scope")
        if not isinstance(raw_scope, dict):
            raise ValueError("hosted conversation has no persisted execution scope")
        execution_scope.assert_resumable(ExecutionScope.from_dict(raw_scope))
    state = AgentState(conversation_id=conversation.id)
    state.turns = session_store.load_turns(conversation.id)
    if not state.turns:
        return state

    latest_turn = state.turns[-1]
    _reconcile_terminal_turn_message(
        session_store,
        conversation.id,
        conversation.status,
        latest_turn,
    )
    _reconcile_blocked_plan_turn(
        session_store,
        conversation.id,
        latest_turn,
    )
    if latest_turn.status == "in_progress":
        hosted_requests = session_store.load_uncheckpointed_hosted_mcp_requests(
            conversation.id,
            latest_turn.turn_id,
            checkpointed_request_count=latest_turn.model_request_count,
        )
        if hosted_requests:
            _block_uncertain_hosted_mcp_request(
                session_store,
                conversation.id,
                latest_turn,
                hosted_requests,
            )
        else:
            unreconciled_calls = [
                record.to_dict()
                for record in latest_turn.tool_calls
                if record.success is None or record.ended_at is None
            ]
            if not unreconciled_calls:
                unreconciled_calls = session_store.load_tool_calls_without_observations(
                    conversation.id,
                    latest_turn.turn_id,
                )
            if unreconciled_calls:
                handler = unresolved_tool_handler or block_unresolved_tool_intent
                handler(
                    session_store,
                    conversation.id,
                    latest_turn,
                    unreconciled_calls,
                )
    state.current_turn_id = latest_turn.turn_id
    state.loaded_memory_ids = list(latest_turn.loaded_memory_ids)
    state.extracted_memory_ids = list(latest_turn.extracted_memory_ids)
    state.loaded_skill_names = list(latest_turn.loaded_skill_names)
    state.available_tool_names = list(latest_turn.available_tool_names)
    state.errors = [error for turn in state.turns for error in turn.errors]
    state.final_answer = latest_turn.final_answer
    if latest_turn.context_reports:
        state.last_context_report = latest_turn.context_reports[-1]
    if latest_turn.model_usage_totals:
        state.last_usage_report = latest_turn.model_usage_totals
    if (
        latest_turn.status == "waiting_for_approval"
        and latest_turn.active_plan is not None
        and not latest_turn.plan_approved
    ):
        state.active_plan = latest_turn.active_plan
        state.pending_plan_turn_id = latest_turn.turn_id
    elif latest_turn.can_continue_approved_plan():
        state.active_plan = latest_turn.active_plan
    return state


async def create_agent_state_async(
    session_store: object,
    conversation_id: str | None,
    *,
    execution_scope: ExecutionScope,
) -> AgentState:
    """Restore hosted state exclusively through awaited session operations."""
    if conversation_id is None:
        return AgentState()
    conversation = await call_async_service(
        session_store,
        "get_conversation",
        conversation_id,
    )
    raw_scope = conversation.metadata.get("execution_scope")
    if not isinstance(raw_scope, dict):
        raise ValueError("hosted conversation has no persisted execution scope")
    execution_scope.assert_resumable(ExecutionScope.from_dict(raw_scope))
    state = AgentState(conversation_id=conversation.id)
    state.turns = list(
        await call_async_service(
            session_store,
            "load_turns",
            conversation.id,
        )
    )
    if not state.turns:
        return state

    latest_turn = state.turns[-1]
    if latest_turn.status in {"in_progress", "waiting_for_approval"}:
        terminal_message = await call_async_service(
            session_store,
            "load_terminal_turn_message",
            conversation.id,
            latest_turn.turn_id,
        )
        if terminal_message is not None:
            content = terminal_message["content"]
            kind = terminal_message["kind"]
            if kind == "final":
                latest_turn.complete(content)
            elif kind == "plan_rejected":
                latest_turn.reject_plan(content)
            elif kind == "failed":
                plan_status = (
                    latest_turn.active_plan.status()
                    if latest_turn.active_plan is not None
                    else None
                )
                if conversation.status == "cancelled":
                    latest_turn.cancel(content)
                elif (
                    conversation.status == "blocked"
                    or plan_status == "blocked"
                ):
                    latest_turn.block(content)
                else:
                    latest_turn.fail(content)
            await call_async_service(
                session_store,
                "save_turn_snapshot",
                conversation.id,
                latest_turn.to_dict(),
            )

    plan = latest_turn.active_plan
    if (
        latest_turn.status == "in_progress"
        and plan is not None
        and plan.status() == "blocked"
    ):
        blocked_step = next(
            (step for step in plan.steps if step.status == "blocked"),
            None,
        )
        if blocked_step is not None:
            reason = blocked_step.blocked_reason or "Step blocked."
            message = f"Plan step blocked: {blocked_step.title}. {reason}"
            latest_turn.block(message)
            await _save_recovery_terminal_async(
                session_store,
                conversation.id,
                latest_turn,
                message,
                message_key_suffix="blocked_plan_checkpoint",
                metadata={"recovery": "blocked_plan_checkpoint"},
            )

    if latest_turn.status == "in_progress":
        hosted_requests = await call_async_service(
            session_store,
            "load_uncheckpointed_hosted_mcp_requests",
            conversation.id,
            latest_turn.turn_id,
            checkpointed_request_count=latest_turn.model_request_count,
        )
        if hosted_requests:
            latest = hosted_requests[-1]
            raw_request_index = latest.get("request_index")
            request_index = (
                raw_request_index
                if isinstance(raw_request_index, int)
                and not isinstance(raw_request_index, bool)
                else 0
            )
            reason = (
                "Turn execution stopped after restart because hosted MCP "
                f"request {request_index} may have executed a remote operation "
                "without a durable checkpoint. Chulk will not replay it "
                "automatically; inspect remote state before retrying."
            )
            active_step = plan.active_step() if plan is not None else None
            if active_step is not None:
                active_step.block(reason)
            latest_turn.block(reason)
            await _save_recovery_terminal_async(
                session_store,
                conversation.id,
                latest_turn,
                reason,
                message_key_suffix="uncertain_hosted_mcp",
                metadata={
                    "recovery": "uncertain_hosted_mcp",
                    "request_index": request_index,
                },
            )
        else:
            unresolved_calls = [
                record.to_dict()
                for record in latest_turn.tool_calls
                if record.success is None or record.ended_at is None
            ]
            if not unresolved_calls:
                unresolved_calls = await call_async_service(
                    session_store,
                    "load_tool_calls_without_observations",
                    conversation.id,
                    latest_turn.turn_id,
                )
            if unresolved_calls:
                latest = unresolved_calls[-1]
                tool_name = str(latest.get("tool_name") or "tool")
                raw_iteration = latest.get("iteration")
                iteration = (
                    raw_iteration
                    if isinstance(raw_iteration, int)
                    and not isinstance(raw_iteration, bool)
                    else 0
                )
                reason = (
                    "Turn execution stopped after restart because tool call "
                    f"{tool_name} (iteration {iteration}) has no matching "
                    "persisted observation. Chulk will not replay it "
                    "automatically; inspect external state before retrying."
                )
                active_step = plan.active_step() if plan is not None else None
                if active_step is not None:
                    active_step.block(reason)
                latest_turn.block(reason)
                await _save_recovery_terminal_async(
                    session_store,
                    conversation.id,
                    latest_turn,
                    reason,
                    message_key_suffix="unresolved_tool_intent",
                    metadata={"recovery": "unresolved_tool_intent"},
                )

    state.current_turn_id = latest_turn.turn_id
    state.loaded_memory_ids = list(latest_turn.loaded_memory_ids)
    state.extracted_memory_ids = list(latest_turn.extracted_memory_ids)
    state.loaded_skill_names = list(latest_turn.loaded_skill_names)
    state.available_tool_names = list(latest_turn.available_tool_names)
    state.errors = [error for turn in state.turns for error in turn.errors]
    state.final_answer = latest_turn.final_answer
    if latest_turn.context_reports:
        state.last_context_report = latest_turn.context_reports[-1]
    if latest_turn.model_usage_totals:
        state.last_usage_report = latest_turn.model_usage_totals
    if (
        latest_turn.status == "waiting_for_approval"
        and latest_turn.active_plan is not None
        and not latest_turn.plan_approved
    ):
        state.active_plan = latest_turn.active_plan
        state.pending_plan_turn_id = latest_turn.turn_id
    elif latest_turn.can_continue_approved_plan():
        state.active_plan = latest_turn.active_plan
    return state


async def _save_recovery_terminal_async(
    session_store: object,
    conversation_id: str,
    turn: TurnState,
    message: str,
    *,
    message_key_suffix: str,
    metadata: dict[str, object],
) -> None:
    saved = await call_async_service(
        session_store,
        "save_terminal_turn_bundle",
        conversation_id,
        turn_id=turn.turn_id,
        content=message,
        message_key=f"{turn.turn_id}:assistant:{message_key_suffix}",
        turn=turn.to_dict(),
        metadata=metadata,
    )
    if not saved:
        raise RuntimeError(
            "hosted session store did not persist a terminal recovery bundle"
        )


def _reconcile_terminal_turn_message(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    conversation_status: str,
    turn: TurnState,
) -> None:
    """Terminalize a legacy turn whose terminal message preceded its snapshot."""
    if turn.status not in {"in_progress", "waiting_for_approval"}:
        return
    terminal_message = session_store.load_terminal_turn_message(
        conversation_id,
        turn.turn_id,
    )
    if terminal_message is None:
        return
    content = terminal_message["content"]
    kind = terminal_message["kind"]
    if kind == "final":
        turn.complete(content)
    elif kind == "plan_rejected":
        turn.reject_plan(content)
    elif kind == "failed":
        plan_status = turn.active_plan.status() if turn.active_plan is not None else None
        if conversation_status == "cancelled":
            turn.cancel(content)
        elif conversation_status == "blocked" or plan_status == "blocked":
            turn.block(content)
        else:
            turn.fail(content)
    else:  # pragma: no cover - constrained by SQLiteSessionStore
        return
    session_store.save_turn_snapshot(conversation_id, turn.to_dict())


def _reconcile_blocked_plan_turn(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
) -> None:
    """Terminalize a checkpoint that already contains a blocked plan step."""
    plan = turn.active_plan
    if turn.status != "in_progress" or plan is None or plan.status() != "blocked":
        return
    blocked_step = next(
        (step for step in plan.steps if step.status == "blocked"),
        None,
    )
    if blocked_step is None:  # pragma: no cover - Plan.status enforces this
        return
    reason = blocked_step.blocked_reason or "Step blocked."
    message = f"Plan step blocked: {blocked_step.title}. {reason}"
    turn.block(message)
    _save_recovery_terminal(
        session_store,
        conversation_id,
        turn,
        message,
        message_key_suffix="blocked_plan_checkpoint",
        metadata={"recovery": "blocked_plan_checkpoint"},
    )


def _block_uncertain_hosted_mcp_request(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
    hosted_requests: list[dict[str, object]],
) -> None:
    """Fail closed when a hosted provider request lacks a durable checkpoint."""
    latest = hosted_requests[-1]
    raw_request_index = latest.get("request_index")
    request_index = (
        raw_request_index
        if isinstance(raw_request_index, int)
        and not isinstance(raw_request_index, bool)
        else 0
    )
    reason = (
        "Turn execution stopped after restart because hosted MCP request "
        f"{request_index} may have executed a remote operation without a durable "
        "checkpoint. Chulk will not replay it automatically; inspect remote state "
        "before retrying."
    )
    plan = turn.active_plan
    active_step = plan.active_step() if plan is not None else None
    if active_step is not None:
        active_step.block(reason)
    turn.block(reason)
    _save_recovery_terminal(
        session_store,
        conversation_id,
        turn,
        reason,
        message_key_suffix="uncertain_hosted_mcp",
        metadata={
            "recovery": "uncertain_hosted_mcp",
            "request_index": request_index,
        },
    )


def block_unresolved_tool_intent(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
    unresolved_calls: list[dict[str, object]],
) -> None:
    """Fail closed when execution stopped after intent but before a result."""
    latest = unresolved_calls[-1]
    tool_name = str(latest.get("tool_name") or "tool")
    raw_iteration = latest.get("iteration")
    iteration = (
        raw_iteration
        if isinstance(raw_iteration, int) and not isinstance(raw_iteration, bool)
        else 0
    )
    reason = (
        "Turn execution stopped after restart because "
        f"tool call {tool_name} (iteration {iteration}) has no matching persisted "
        "observation. Chulk will not replay it automatically; inspect external "
        "state before retrying."
    )
    plan = turn.active_plan
    active_step = plan.active_step() if plan is not None else None
    if active_step is not None:
        active_step.block(reason)
    turn.block(reason)
    _save_recovery_terminal(
        session_store,
        conversation_id,
        turn,
        reason,
        message_key_suffix="unresolved_tool_intent",
        metadata={"recovery": "unresolved_tool_intent"},
    )


def _save_recovery_terminal(
    session_store: SQLiteSessionStore,
    conversation_id: str,
    turn: TurnState,
    content: str,
    *,
    message_key_suffix: str,
    metadata: dict[str, object],
) -> None:
    saved = session_store.save_terminal_turn_bundle(
        conversation_id,
        turn_id=turn.turn_id,
        content=content,
        message_key=f"{turn.turn_id}:assistant:{message_key_suffix}",
        turn=turn.to_dict(),
        metadata=metadata,
    )
    if not saved:  # pragma: no cover - TurnState guarantees a valid payload
        raise RuntimeError("Failed to persist terminal recovery state")
