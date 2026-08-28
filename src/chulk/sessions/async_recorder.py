"""Event-driven session recorder for native async hosted stores."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from chulk.core.events import TraceEvent
from chulk.hosting.async_utils import call_async_service


class AsyncSessionRecorder:
    """Journal trace events and persist them through awaited store methods."""

    def __init__(
        self,
        store: object,
        conversation_id: str,
        *,
        provider: str,
        model: str,
        trace_path: Path | str | None = None,
        lazy: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.current_turn_id: str | None = None
        self._observation_counts: dict[str, int] = {}
        self.provider = provider
        self.model = model
        self.trace_path = str(trace_path) if trace_path is not None else None
        self.metadata = dict(metadata or {})
        self.persisted = False
        self.lazy = lazy
        self._pending: list[tuple[str, dict[str, Any]]] = []

    def callback(self, event_type: str, payload: dict[str, Any]) -> None:
        """Queue one immutable event for the next explicit await boundary."""

        self._pending.append((event_type, deepcopy(payload)))

    async def initialize(self) -> None:
        """Create eager conversations before the runtime is exposed."""

        if not self.lazy:
            await self._ensure_conversation()

    async def flush(self) -> None:
        """Persist queued events in order, retaining failed operations for retry."""

        while self._pending:
            event_type, payload = self._pending[0]
            await self._persist(event_type, payload)
            self._pending.pop(0)

    async def _persist(self, event_type: str, payload: dict[str, Any]) -> None:
        await self._ensure_conversation()
        if event_type == TraceEvent.TURN_STARTED:
            turn = payload.get("turn")
            if isinstance(turn, dict):
                self.current_turn_id = _payload_turn_id(turn, None)
                await self._call("save_turn_snapshot", self.conversation_id, turn)
            return

        if event_type == TraceEvent.USER_MESSAGE:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is not None:
                self.current_turn_id = turn_id
            await self._call(
                "save_message",
                self.conversation_id,
                turn_id=turn_id,
                role="user",
                content=str(payload.get("content") or ""),
                message_key=f"{turn_id}:user" if turn_id else None,
            )
            return

        if event_type == TraceEvent.MODEL_REQUEST_STARTED:
            await self._call(
                "save_model_request",
                self.conversation_id,
                {
                    **payload,
                    "turn_id": _payload_turn_id(payload, self.current_turn_id),
                },
            )
            return

        if event_type == TraceEvent.CONTEXT_SUMMARY_CREATED:
            await self._call(
                "save_conversation_summary",
                self.conversation_id,
                content=str(payload.get("summary") or ""),
                source_message_count=int(payload.get("source_message_count") or 0),
                metadata={
                    "event": event_type,
                    "turn_id": _payload_turn_id(payload, self.current_turn_id),
                    "summarized_message_count": int(
                        payload.get("summarized_message_count") or 0
                    ),
                    "fallback": bool(payload.get("fallback")),
                    "checkpoint_v1": _safe_dict(payload.get("checkpoint")),
                },
            )
            return

        if event_type == TraceEvent.MODEL_RESPONSE:
            await self._call(
                "save_model_response",
                self.conversation_id,
                {
                    **payload,
                    "turn_id": _payload_turn_id(payload, self.current_turn_id),
                },
            )
            return

        if event_type in {
            TraceEvent.TOOL_CALL_STARTED,
            TraceEvent.TOOL_CALL_COMPLETED,
            TraceEvent.TOOL_CALL_FAILED,
        }:
            await self._call(
                "save_tool_call",
                self.conversation_id,
                {
                    **payload,
                    "turn_id": _payload_turn_id(payload, self.current_turn_id),
                },
            )
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
                    previous_count = await self._call(
                        "max_observation_index",
                        self.conversation_id,
                        turn_id,
                    )
                count = previous_count + 1
            await self._call(
                "save_tool_observation_bundle",
                self.conversation_id,
                turn_id=turn_id,
                observation_index=count,
                tool_name=str(payload.get("tool_name") or "tool"),
                content=str(payload.get("observation") or ""),
                output_metadata=_safe_dict(payload.get("output_metadata")),
                action_context=(
                    payload.get("tool_action_context")
                    if isinstance(payload.get("tool_action_context"), str)
                    else None
                ),
                turn=(
                    payload.get("turn")
                    if isinstance(payload.get("turn"), dict)
                    else None
                ),
            )
            self._observation_counts[turn_id] = max(
                count,
                self._observation_counts.get(turn_id, 0),
            )
            return

        if event_type == TraceEvent.PLAN_CREATED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            turn = payload.get("turn")
            display_message = payload.get("display_message")
            if turn_id is not None:
                self.current_turn_id = turn_id
            if isinstance(turn, dict):
                await self._call(
                    "save_turn_snapshot",
                    self.conversation_id,
                    turn,
                )
            if isinstance(display_message, str) and display_message.strip():
                await self._call(
                    "save_message",
                    self.conversation_id,
                    turn_id=turn_id,
                    role="assistant",
                    content=display_message,
                    message_key=(
                        f"{turn_id}:assistant:plan_display" if turn_id else None
                    ),
                    metadata={"event": event_type, "prompt_excluded": True},
                )
            await self._call(
                "update_conversation_status",
                self.conversation_id,
                "waiting_for_approval",
            )
            return

        if event_type == TraceEvent.PLAN_APPROVED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is not None:
                self.current_turn_id = turn_id
            turn = payload.get("turn")
            if isinstance(turn, dict):
                await self._call(
                    "save_turn_snapshot",
                    self.conversation_id,
                    turn,
                )
            await self._call(
                "update_conversation_status",
                self.conversation_id,
                "active",
            )
            return

        if event_type in {
            TraceEvent.PLAN_STEP_STARTED,
            TraceEvent.PLAN_STEP_COMPLETED,
        }:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is not None:
                self.current_turn_id = turn_id
            turn = payload.get("turn")
            if isinstance(turn, dict):
                await self._call(
                    "save_turn_snapshot",
                    self.conversation_id,
                    turn,
                )
            return

        if event_type == TraceEvent.PLAN_STEP_BLOCKED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            if turn_id is not None:
                self.current_turn_id = turn_id
            turn = payload.get("turn")
            if isinstance(turn, dict):
                content = str(turn.get("final_answer") or "")
                if turn.get("status") == "blocked" and await self._save_terminal_turn(
                    turn_id=turn_id,
                    content=content,
                    message_key_suffix="failed",
                    turn=turn,
                    metadata={"event": event_type},
                ):
                    return
                await self._call(
                    "save_turn_snapshot",
                    self.conversation_id,
                    turn,
                )
            return

        if event_type == TraceEvent.PLAN_REJECTED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            message = "Plan rejected. No tools were run."
            if await self._save_terminal_turn(
                turn_id=turn_id,
                content=message,
                message_key_suffix="plan_rejected",
                turn=payload.get("turn"),
                metadata={"event": event_type},
            ):
                return
            await self._call(
                "save_message",
                self.conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=message,
                message_key=(
                    f"{turn_id}:assistant:plan_rejected" if turn_id else None
                ),
                metadata={"event": event_type},
            )
            await self._call(
                "update_conversation_status",
                self.conversation_id,
                "plan_rejected",
            )
            return

        if event_type == TraceEvent.FINAL_ANSWER:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            content = str(payload.get("content") or "")
            if await self._save_terminal_turn(
                turn_id=turn_id,
                content=content,
                message_key_suffix="final",
                turn=payload.get("turn"),
                metadata={"event": event_type},
            ):
                return
            await self._call(
                "save_message",
                self.conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=content,
                message_key=f"{turn_id}:assistant:final" if turn_id else None,
            )
            return

        if event_type == TraceEvent.TURN_FAILED:
            turn_id = _payload_turn_id(payload, self.current_turn_id)
            content = str(payload.get("message") or "")
            if await self._save_terminal_turn(
                turn_id=turn_id,
                content=content,
                message_key_suffix="failed",
                turn=payload.get("turn"),
                metadata={"event": event_type},
            ):
                return
            raw_status = payload.get("status")
            status = (
                raw_status
                if raw_status in {"blocked", "cancelled", "failed"}
                else "failed"
            )
            await self._call(
                "save_message",
                self.conversation_id,
                turn_id=turn_id,
                role="assistant",
                content=content,
                message_key=f"{turn_id}:assistant:failed" if turn_id else None,
                metadata={"event": event_type},
            )
            await self._call(
                "update_conversation_status",
                self.conversation_id,
                status,
            )
            return

        if event_type == TraceEvent.TURN_FINISHED:
            turn = payload.get("turn")
            if isinstance(turn, dict):
                await self._call(
                    "save_turn_snapshot",
                    self.conversation_id,
                    turn,
                )

    async def _save_terminal_turn(
        self,
        *,
        turn_id: str | None,
        content: str,
        message_key_suffix: str,
        turn: object,
        metadata: dict[str, Any],
    ) -> bool:
        if turn_id is None or not isinstance(turn, dict):
            return False
        return bool(
            await self._call(
                "save_terminal_turn_bundle",
                self.conversation_id,
                turn_id=turn_id,
                content=content,
                message_key=f"{turn_id}:assistant:{message_key_suffix}",
                turn=turn,
                metadata=metadata,
            )
        )

    async def _ensure_conversation(self) -> None:
        if self.persisted:
            return
        await self._call(
            "create_conversation",
            self.conversation_id,
            provider=self.provider,
            model=self.model,
            trace_path=self.trace_path,
            metadata=self.metadata,
        )
        self.persisted = True

    async def _call(self, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
        return await call_async_service(
            self.store,
            method_name,
            *args,
            **kwargs,
        )


def _payload_turn_id(
    payload: dict[str, Any],
    fallback: str | None,
) -> str | None:
    turn_id = payload.get("turn_id")
    return turn_id if isinstance(turn_id, str) and turn_id else fallback


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["AsyncSessionRecorder"]
