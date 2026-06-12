"""Event-driven session recorder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.events import TraceEvent
from src.sessions.sqlite_store import SQLiteSessionStore


class SessionRecorder:
    """Persist agent trace events into SQLite session tables."""

    def __init__(
        self,
        store: SQLiteSessionStore,
        conversation_id: str,
        *,
        provider: str,
        model: str,
        trace_path: Path | str | None = None,
    ) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.current_turn_id: str | None = None
        self._observation_counts: dict[str, int] = {}
        self.store.create_conversation(
            conversation_id,
            provider=provider,
            model=model,
            trace_path=str(trace_path) if trace_path is not None else None,
        )

    def callback(self, event_type: str, payload: dict[str, Any]) -> None:
        """Persist the subset of trace events needed to resume and inspect sessions."""
        if event_type == TraceEvent.TURN_STARTED:
            turn = payload.get("turn")
            if isinstance(turn, dict):
                self.current_turn_id = turn.get("turn_id")
                self.store.save_turn_snapshot(self.conversation_id, turn)
            return

        if event_type == TraceEvent.USER_MESSAGE:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is not None:
                self.current_turn_id = turn_id
            self.store.save_message(
                self.conversation_id,
                turn_id=turn_id,
                role="user",
                content=str(payload.get("content") or ""),
                message_key=f"{turn_id}:user" if turn_id else None,
            )
            return

        if event_type == TraceEvent.MODEL_REQUEST_STARTED:
            self.store.save_model_request(self.conversation_id, {**payload, "turn_id": _payload_turn_id(payload, self.current_turn_id)})
            return

        if event_type == TraceEvent.MODEL_RESPONSE:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            self.store.save_model_response(self.conversation_id, {**payload, "turn_id": turn_id})
            return

        if event_type in {TraceEvent.TOOL_CALL_STARTED, TraceEvent.TOOL_CALL_COMPLETED, TraceEvent.TOOL_CALL_FAILED}:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            self.store.save_tool_call(self.conversation_id, {**payload, "turn_id": turn_id})
            return

        if event_type == TraceEvent.TOOL_OBSERVATION:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is None:
                return
            count = self._observation_counts.get(turn_id, 0) + 1
            self._observation_counts[turn_id] = count
            observation = str(payload.get("observation") or "")
            tool_name = str(payload.get("tool_name") or "tool")
            key = f"{turn_id}:observation:{count}"
            self.store.save_observation(
                self.conversation_id,
                turn_id=turn_id,
                tool_name=tool_name,
                content=observation,
                output_metadata=_safe_dict(payload.get("output_metadata")),
                observation_key=key,
            )
            self.store.save_message(
                self.conversation_id,
                turn_id=turn_id,
                role="observation",
                content=observation,
                message_key=key,
                metadata={"tool_name": tool_name},
            )
            return

        if event_type == TraceEvent.PLAN_CREATED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            plan = _safe_dict(payload.get("plan"))
            turn = payload.get("turn")
            if turn_id is not None:
                self.current_turn_id = turn_id
            if isinstance(turn, dict):
                self.store.save_turn_snapshot(self.conversation_id, turn)
            self.store.save_message(
                self.conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=_plan_response_text(plan),
                message_key=f"{turn_id}:assistant:plan" if turn_id else None,
                metadata={"event": event_type},
            )
            self.store.update_conversation_status(self.conversation_id, "waiting_for_approval")
            return

        if event_type == TraceEvent.PLAN_APPROVED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is not None:
                self.current_turn_id = turn_id
            self.store.save_message(
                self.conversation_id,
                turn_id=turn_id,
                role="user",
                content="User approved the plan. Continue executing the approved plan.",
                message_key=f"{turn_id}:user:plan_approved" if turn_id else None,
                metadata={"event": event_type, "internal": True},
            )
            self.store.update_conversation_status(self.conversation_id, "active")
            return

        if event_type == TraceEvent.PLAN_REJECTED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            self.store.save_message(
                self.conversation_id,
                turn_id=turn_id,
                role="assistant",
                content="Plan rejected. No tools were run.",
                message_key=f"{turn_id}:assistant:plan_rejected" if turn_id else None,
                metadata={"event": event_type},
            )
            self.store.update_conversation_status(self.conversation_id, "plan_rejected")
            return

        if event_type == TraceEvent.FINAL_ANSWER:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            self.store.save_message(
                self.conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=str(payload.get("content") or ""),
                message_key=f"{turn_id}:assistant:final" if turn_id else None,
            )
            return

        if event_type == TraceEvent.TURN_FAILED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            self.store.save_message(
                self.conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=str(payload.get("message") or ""),
                message_key=f"{turn_id}:assistant:failed" if turn_id else None,
                metadata={"event": event_type},
            )
            self.store.update_conversation_status(self.conversation_id, "failed")
            return

        if event_type == TraceEvent.TURN_FINISHED:
            turn = payload.get("turn")
            if isinstance(turn, dict):
                self.store.save_turn_snapshot(self.conversation_id, turn)


def _payload_turn_id(payload: dict[str, Any], fallback: str | None) -> str | None:
    turn_id = payload.get("turn_id")
    return turn_id if isinstance(turn_id, str) and turn_id else fallback


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _plan_response_text(plan: dict[str, Any]) -> str:
    summary = str(plan.get("summary") or "")
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    lines = ["Plan", f"  summary  {summary}", "  steps"]
    for step in steps:
        if not isinstance(step, dict):
            continue
        status = step.get("status") or "pending"
        title = step.get("title") or "Untitled step"
        description = step.get("description") or ""
        lines.append(f"  - [{status}] {title}: {description}")
    return "\n".join(lines) + "\n\nUse /approve to execute this plan or /reject to cancel it."
