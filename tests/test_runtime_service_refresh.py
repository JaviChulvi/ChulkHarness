"""Owner API tests for intentionally replaceable runtime services."""

from __future__ import annotations

import asyncio

import pytest

from chulk.core.actions import FinalAnswerAction
from chulk.core.events import TraceEvent
from chulk.core.state import TurnState
from chulk.llm.capabilities import LLMCapabilities
from chulk.testing import ScriptedLLMClient
from chulk.tools import Tool, ToolRegistry
from chulk.tools.permissions import ToolPermissionPolicy
from tests.core_agent import build_core_agent as Agent


@pytest.mark.parametrize(
    "failure",
    [None, RuntimeError("failed"), asyncio.CancelledError()],
)
def test_model_client_override_restores_after_every_exit(failure: BaseException | None) -> None:
    original = ScriptedLLMClient([])
    replacement = ScriptedLLMClient([])
    agent = Agent(original)

    if failure is None:
        with agent._model_transport.override_client(replacement):
            assert agent._model_transport.llm_client is replacement
    else:
        with pytest.raises(type(failure)):
            with agent._model_transport.override_client(replacement):
                raise failure

    assert agent._model_transport.llm_client is original


def test_catalog_updates_model_and_execution_registry_together() -> None:
    agent = Agent(ScriptedLLMClient([]))
    replacement = ToolRegistry()
    replacement.register(
        Tool(
            name="inspect_repo",
            description="Inspect the repository.",
            args_schema={"type": "object"},
            callable=lambda _arguments: None,
        )
    )

    agent.catalog.set_registry(replacement)

    assert agent.catalog.active_registry is replacement
    assert agent._model_transport.tool_registry is replacement
    assert agent._tool_executor.registry is replacement
    snapshot = agent._turn_effects.snapshot(
        TurnState(user_message="inspect"),
        require_plan=True,
    )
    assert snapshot.planning_tool_names == frozenset({"inspect_repo"})


def test_model_client_override_updates_client_dependent_effects() -> None:
    class StreamingScriptedLLMClient(ScriptedLLMClient):
        capabilities = LLMCapabilities(supports_streaming=True)

    events: list[str] = []
    replacement = StreamingScriptedLLMClient(
        [FinalAnswerAction(type="final_answer", content="streamed")]
    )
    agent = Agent(
        ScriptedLLMClient([]),
        event_callback=lambda event, _payload: events.append(event),
    )

    with agent._model_transport.override_client(replacement):
        assert agent.run_turn("hello") == "streamed"

    assert TraceEvent.MODEL_STREAM_STARTED in events


def test_event_capture_and_permission_replacement_are_owned() -> None:
    base_events: list[str] = []
    captured_events: list[str] = []
    agent = Agent(
        ScriptedLLMClient([]),
        event_callback=lambda event, _payload: base_events.append(event),
    )
    replacement_policy = ToolPermissionPolicy(name="evaluation")

    agent._tool_executor.set_permission_policy(replacement_policy)
    with agent.events.capture(
        lambda event, _payload: captured_events.append(event)
    ):
        agent.events.emit("captured")
    agent.events.emit("base-only")

    assert agent._tool_executor.permission_policy is replacement_policy
    assert captured_events == ["captured"]
    assert base_events == ["captured", "base-only"]
