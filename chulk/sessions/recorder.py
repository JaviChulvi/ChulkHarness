"""Event-driven session recorder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chulk.core.events import TraceEvent
from chulk.sessions.sqlite_store import SQLiteSessionStore


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
        lazy: bool = False,
    ) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.current_turn_id: str | None = None
        self._observation_counts: dict[str, int] = {}
        self.provider = provider
        self.model = model
        self.trace_path = str(trace_path) if trace_path is not None else None
        self.persisted = False
        if not lazy:
            self._ensure_conversation()

    def callback(self, event_type: str, payload: dict[str, Any]) -> None:
        """Persist the subset of trace events needed to resume and inspect sessions."""
        self._ensure_conversation()
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

        if event_type == TraceEvent.CONTEXT_SUMMARY_CREATED:
            self.store.save_conversation_summary(
                self.conversation_id,
                content=str(payload.get("summary") or ""),
                source_message_count=int(payload.get("source_message_count") or 0),
                metadata={
                    "event": event_type,
                    "turn_id": _payload_turn_id(payload, self.current_turn_id),
                    "summarized_message_count": int(payload.get("summarized_message_count") or 0),
                    "fallback": bool(payload.get("fallback")),
                },
            )
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
            payload_index = payload.get("observation_index")
            if (
                isinstance(payload_index, int)
                and not isinstance(payload_index, bool)
                and payload_index > 0
            ):
                count = payload_index
            else:
                previous_count = self._observation_counts.get(turn_id)
                if previous_count is None:
                    previous_count = self.store.max_observation_index(
                        self.conversation_id,
                        turn_id,
                    )
                count = previous_count + 1
            observation = str(payload.get("observation") or "")
            tool_name = str(payload.get("tool_name") or "tool")
            action_context = payload.get("tool_action_context")
            turn = payload.get("turn")
            self.store.save_tool_observation_bundle(
                self.conversation_id,
                turn_id=turn_id,
                observation_index=count,
                tool_name=tool_name,
                content=observation,
                output_metadata=_safe_dict(payload.get("output_metadata")),
                action_context=action_context if isinstance(action_context, str) else None,
                turn=turn if isinstance(turn, dict) else None,
            )
            self._observation_counts[turn_id] = max(
                count,
                self._observation_counts.get(turn_id, 0),
            )
            return

        if event_type == TraceEvent.PLAN_CREATED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            turn = payload.get("turn")
            if turn_id is not None:
                self.current_turn_id = turn_id
            if isinstance(turn, dict):
                self.store.save_turn_snapshot(self.conversation_id, turn)
            self.store.update_conversation_status(self.conversation_id, "waiting_for_approval")
            return

        if event_type == TraceEvent.PLAN_APPROVED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is not None:
                self.current_turn_id = turn_id
            turn = payload.get("turn")
            if isinstance(turn, dict):
                self.store.save_turn_snapshot(self.conversation_id, turn)
            self.store.update_conversation_status(self.conversation_id, "active")
            return

        if event_type in {
            TraceEvent.PLAN_STEP_STARTED,
            TraceEvent.PLAN_STEP_COMPLETED,
            TraceEvent.PLAN_STEP_BLOCKED,
        }:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is not None:
                self.current_turn_id = turn_id
            turn = payload.get("turn")
            if isinstance(turn, dict):
                self.store.save_turn_snapshot(self.conversation_id, turn)
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
            status = "cancelled" if payload.get("status") == "cancelled" else "failed"
            self.store.save_message(
                self.conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=str(payload.get("message") or ""),
                message_key=f"{turn_id}:assistant:failed" if turn_id else None,
                metadata={"event": event_type},
            )
            self.store.update_conversation_status(self.conversation_id, status)
            return

        if event_type == TraceEvent.TURN_FINISHED:
            turn = payload.get("turn")
            if isinstance(turn, dict):
                self.store.save_turn_snapshot(self.conversation_id, turn)

    def _ensure_conversation(self) -> None:
        if self.persisted:
            return
        self.store.create_conversation(
            self.conversation_id,
            provider=self.provider,
            model=self.model,
            trace_path=self.trace_path,
        )
        self.persisted = True


def _payload_turn_id(payload: dict[str, Any], fallback: str | None) -> str | None:
    turn_id = payload.get("turn_id")
    return turn_id if isinstance(turn_id, str) and turn_id else fallback


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
