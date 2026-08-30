"""Adapter from Chulk's stable events to AI SDK UI message stream chunks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from chulk.events import AgentEvent, EventName


def ai_sdk_ui_chunks(events: Iterable[AgentEvent]) -> Iterator[dict[str, Any]]:
    """Project safe public events into AI SDK's extensible UI chunk format.

    Chulk remains the source of truth; tool and approval details use custom
    ``data-chulk-*`` chunks so no durable information is discarded.
    """
    started = False
    text_id: str | None = None
    for event in events:
        if event.name == EventName.RUN_STARTED.value:
            started = True
            text_id = f"chulk-{event.turn_id or event.event_id}"
            yield {"type": "start", "messageId": event.turn_id or event.event_id}
            yield {"type": "start-step"}
        elif event.name == EventName.MODEL_DELTA.value:
            if text_id is None:
                text_id = f"chulk-{event.turn_id or event.event_id}"
                yield {"type": "start", "messageId": event.turn_id or event.event_id}
                yield {"type": "start-step"}
            yield {"type": "text-delta", "id": text_id, "delta": event.to_dict()["payload"].get("text", "")}
        elif event.name in {EventName.TOOL_CALL_STARTED.value, EventName.TOOL_CALL_COMPLETED.value, EventName.TOOL_CALL_FAILED.value}:
            yield _data_chunk("tool", event)
        elif event.name in {EventName.APPROVAL_REQUESTED.value, EventName.PERMISSION_REQUESTED.value}:
            yield _data_chunk("approval", event)
        elif event.name == EventName.RUN_COMPLETED.value:
            if text_id is not None:
                yield {"type": "text-end", "id": text_id}
            if started:
                yield {"type": "finish-step"}
                yield {"type": "finish", "finishReason": "stop"}
        elif event.name == EventName.RUN_FAILED.value:
            yield {"type": "error", "errorText": "Chulk run failed."}


def _data_chunk(kind: str, event: AgentEvent) -> dict[str, Any]:
    return {
        "type": f"data-chulk-{kind}",
        "id": event.event_id,
        "data": event.to_dict(),
    }


__all__ = ["ai_sdk_ui_chunks"]
