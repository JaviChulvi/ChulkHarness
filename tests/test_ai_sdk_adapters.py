from __future__ import annotations

from dataclasses import replace

import pytest

from chulk.events import (
    AgentEvent,
    EventName,
    ModelDeltaPayload,
    RunCompletedPayload,
    RunStartedPayload,
)
from chulk.llm import LLMActionResult, MiddlewareLLMClient, wrap_model_client
from chulk.core.actions import FinalAnswerAction
from chulk.media import ModelRequest
from chulk.results import RunResult, RunStatus
from chulk.server import ai_sdk_ui_chunks


class RecordingMiddleware:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def prepare(self, request: ModelRequest) -> ModelRequest:
        self.calls.append("prepare")
        return request

    def observe(self, request: ModelRequest, result: LLMActionResult) -> LLMActionResult:
        self.calls.append("observe")
        return replace(result, metadata={"observed": True})


class ActionClient:
    def complete_action_request(self, request: ModelRequest, **kwargs) -> LLMActionResult:
        return LLMActionResult(
            action=FinalAnswerAction(type="final_answer", content="done"),
            raw_response='{"type":"final_answer","content":"done"}',
        )


class LegacyActionClient:
    def complete_action_request(self, request: ModelRequest) -> LLMActionResult:
        return _action_result()

    async def acomplete_action_request(self, request: ModelRequest) -> LLMActionResult:
        return _action_result()


def _action_result() -> LLMActionResult:
    return LLMActionResult(
        action=FinalAnswerAction(type="final_answer", content="done"),
        raw_response='{"type":"final_answer","content":"done"}',
    )


def test_model_middleware_wraps_the_validated_action_request():
    middleware = RecordingMiddleware()
    client = wrap_model_client(ActionClient(), middleware)

    result = client.complete_action_request(ModelRequest(messages=[]))

    assert isinstance(client, MiddlewareLLMClient)
    assert middleware.calls == ["prepare", "observe"]
    assert result.metadata == {"observed": True}


def test_model_middleware_preserves_legacy_action_client_signatures():
    result = wrap_model_client(LegacyActionClient()).complete_action_request(
        ModelRequest(messages=[]), action_schema={"type": "object"}
    )

    assert result.action.content == "done"


@pytest.mark.asyncio
async def test_model_middleware_preserves_legacy_async_action_client_signatures():
    result = await wrap_model_client(LegacyActionClient()).acomplete_action_request(
        ModelRequest(messages=[]), action_schema={"type": "object"}
    )

    assert result.action.content == "done"


def test_ai_sdk_adapter_keeps_chulk_events_as_custom_data():
    events = (
        AgentEvent(
            name=EventName.RUN_STARTED.value,
            conversation_id="conversation",
            turn_id="turn",
            payload=RunStartedPayload(message="hello"),
        ),
        AgentEvent(
            name=EventName.MODEL_DELTA.value,
            conversation_id="conversation",
            turn_id="turn",
            payload=ModelDeltaPayload(text="Hello"),
        ),
        AgentEvent(
            name=EventName.RUN_COMPLETED.value,
            conversation_id="conversation",
            turn_id="turn",
            payload=RunCompletedPayload(
                RunResult(
                    content="Hello",
                    status=RunStatus.COMPLETED,
                    turn_id="turn",
                    conversation_id="conversation",
                    trace_path=None,
                )
            ),
        ),
    )

    chunks = list(ai_sdk_ui_chunks(events))

    assert chunks == [
        {"type": "start", "messageId": "turn"},
        {"type": "start-step"},
        {"type": "text-start", "id": "chulk-turn"},
        {"type": "text-delta", "id": "chulk-turn", "delta": "Hello"},
        {"type": "text-end", "id": "chulk-turn"},
        {"type": "finish-step"},
        {"type": "finish", "finishReason": "stop"},
    ]
