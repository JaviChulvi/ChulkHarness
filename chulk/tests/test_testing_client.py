from __future__ import annotations

import json

import pytest

from chulk.core.actions import FinalAnswerAction, ToolCallAction
from chulk.llm import LLMError
from chulk.testing import ScriptedLLMClient


MESSAGES = [{"role": "user", "content": "hello"}]


def test_scripted_client_completes_and_records_immutable_call_snapshots() -> None:
    client = ScriptedLLMClient(["first", "second"])

    response = client.complete_response(MESSAGES, max_output_tokens=20)

    assert response.content == "first"
    assert response.provider == "scripted"
    assert response.model == "scripted"
    assert response.usage is not None and response.usage.estimated
    assert response.cost is not None and response.cost.provider == "scripted"
    assert client.remaining == 1
    assert client.call_log[0]["messages"] == ({"role": "user", "content": "hello"},)
    assert client.call_log[0]["max_output_tokens"] == 20


@pytest.mark.parametrize(
    ("script", "expected_type"),
    [
        ({"type": "final_answer", "content": "done"}, FinalAnswerAction),
        (ToolCallAction(type="tool_call", tool_name="lookup", arguments={"id": "A-1"}), ToolCallAction),
    ],
)
def test_scripted_client_uses_normal_validated_action_contract(script: object, expected_type: type) -> None:
    client = ScriptedLLMClient([script])  # type: ignore[list-item]

    result = client.complete_action(MESSAGES)

    assert isinstance(result.action, expected_type)
    assert json.loads(result.raw_response)["type"] in {"final_answer", "tool_call"}


def test_scripted_client_streams_in_deterministic_chunks() -> None:
    client = ScriptedLLMClient(["abcdefgh"], chunk_size=3)

    chunks = list(client.stream_complete(MESSAGES))

    assert [chunk.text for chunk in chunks[:-1]] == ["abc", "def", "gh"]
    assert chunks[-1].type == "completed"
    assert chunks[-1].usage is not None


def test_scripted_client_reports_exhaustion_as_non_retryable_provider_error() -> None:
    client = ScriptedLLMClient([])

    with pytest.raises(LLMError, match="script is exhausted") as raised:
        client.complete(MESSAGES)

    assert raised.value.provider == "scripted"
    assert raised.value.retryable is False


def test_testing_module_does_not_widen_top_level_exports() -> None:
    import chulk
    import chulk.testing as testing

    assert testing.__all__ == ["ScriptedLLMClient"]
    assert "ScriptedLLMClient" not in chulk.__all__
