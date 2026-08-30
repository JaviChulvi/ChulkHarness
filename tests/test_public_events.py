"""Tests for normalized public events and generator-style runs."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from chulk import (
    EVENT_SCHEMA_VERSION,
    Agent,
    AgentConfig,
    AgentEvent,
    AsyncAgent,
    EventName,
    LearningProposalChangedPayload,
    RunCompletedPayload,
    RunFailedPayload,
    RunStartedPayload,
    Tools,
    Skills,
)
from chulk._sdk.event_channel import RunEventChannel
from chulk._sdk.events import project_event
from chulk.llm import LLMCapabilities, LLMClient


class StreamingLLM(LLMClient):
    capabilities = LLMCapabilities(supports_streaming=True)
    provider = "test"
    model = "events"

    def __init__(self, responses: list[str] | None = None, *, delay: float = 0) -> None:
        self.responses = responses or [json.dumps({"type": "final_answer", "content": "done"})]
        self.delay = delay
        self._lock = threading.Lock()

    def complete(self, messages, *, max_output_tokens=None) -> str:
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            if len(self.responses) == 1:
                return self.responses[0]
            return self.responses.pop(0)


def _agent(tmp_path, llm: LLMClient | None = None, **kwargs) -> Agent:
    return Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=llm or StreamingLLM(),
        tools=[],
        skills=[],
        **kwargs,
    )


def test_event_envelope_and_catalog_are_stable(tmp_path):
    facade = _agent(tmp_path)

    events = list(facade.run_events("hello"))

    assert [event.name for event in events] == [
        "run.started",
        "memory.loaded",
        "skill.loaded",
        "budget.reserved",
        "model.request.started",
        "budget.committed",
        "model.response.completed",
        "model.delta",
        "run.completed",
    ]
    assert all(event.schema_version == EVENT_SCHEMA_VERSION == 3 for event in events)
    assert all(event.profile_id == "default" for event in events)
    assert all(event.conversation_id == facade.conversation_id for event in events)
    assert all(event.turn_id == events[0].turn_id for event in events)
    assert all(event.timestamp.endswith("+00:00") for event in events)
    assert all(event.name in {name.value for name in EventName} for event in events)
    assert len({event.event_id for event in events}) == len(events)
    assert all(event.run_id == facade.execution_scope.run_id for event in events)
    assert events[0].causation_id is None
    assert all(
        event.causation_id == previous.event_id
        for previous, event in zip(events, events[1:])
    )
    assert all(set(event.to_dict()) == {
        "event_id",
        "name",
        "timestamp",
        "schema_version",
        "conversation_id",
        "turn_id",
        "profile_id",
        "execution_scope",
        "run_id",
        "step_id",
        "correlation_id",
        "causation_id",
        "source_event_id",
        "idempotency_key",
        "payload",
        "extensions",
    } for event in events)
    assert isinstance(events[-1].payload, RunCompletedPayload)
    assert events[-1].payload.result.content == "done"


def test_event_reader_maps_schema_v1_to_the_implicit_default_profile():
    event = AgentEvent.from_dict(
        {
            "name": "run.started",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "schema_version": 1,
            "conversation_id": "conversation",
            "turn_id": "turn",
            "payload": {"message": "hello"},
            "extensions": {},
        }
    )

    assert event.schema_version == 1
    assert event.profile_id == "default"
    assert event.to_dict()["payload"] == {"message": "hello"}


def test_event_reader_preserves_schema_v2_profile_ownership():
    event = AgentEvent.from_dict(
        {
            "name": "run.started",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "schema_version": 2,
            "profile_id": "work",
            "conversation_id": "conversation",
            "turn_id": "turn",
            "payload": {"message": "hello"},
            "extensions": {},
        }
    )

    assert event.profile_id == "work"
    assert event.to_dict()["profile_id"] == "work"
    assert event.event_id.startswith("legacy_")


def test_event_reader_preserves_schema_v3_identity_and_causation():
    event = AgentEvent.from_dict(
        {
            "event_id": "event-2",
            "name": "step.started",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "schema_version": 3,
            "profile_id": "work",
            "conversation_id": "conversation",
            "turn_id": "turn",
            "execution_scope": {
                "tenant_id": "tenant",
                "workspace_id": "workspace",
                "actor_id": "operator",
                "agent_id": "agent",
                "agent_version": "1.0.0",
                "run_id": "run-1",
                "conversation_id": "conversation",
                "trigger_id": "trigger-1",
                "channel_id": None,
                "parent_run_id": None,
                "grants": ["tools:read"],
            },
            "run_id": "run-1",
            "step_id": "step-1",
            "correlation_id": "turn",
            "causation_id": "event-1",
            "source_event_id": "trigger-1",
            "idempotency_key": "transition-1",
            "payload": {"status": "running"},
            "extensions": {},
        }
    )

    assert event.event_id == "event-2"
    assert event.execution_scope is not None
    assert event.execution_scope.tenant_id == "tenant"
    assert event.run_id == "run-1"
    assert event.step_id == "step-1"
    assert event.causation_id == "event-1"


def test_projection_excludes_unknown_internal_events(tmp_path):
    facade = _agent(tmp_path)

    assert project_event(facade.runtime, "future_internal_diagnostic", {"secret": "value"}) is None


def test_learning_proposal_changes_emit_typed_content_free_events(tmp_path):
    events: list[AgentEvent] = []
    facade = _agent(tmp_path, on_event=events.append)
    service = facade.runtime.learning.proposals
    assert service is not None

    proposal = service.create(
        Skills.LearningProposalDraft(
            kind=Skills.LearningProposalKind.MEMORY_CREATE,
            rationale="Explicit durable preference.",
            content="User prefers direct summaries.",
        )
    )
    facade.approve_learning_proposal(proposal.id)

    learning_events = [
        event
        for event in events
        if event.name == EventName.LEARNING_PROPOSAL_CHANGED.value
    ]
    assert [event.payload.action for event in learning_events] == [
        "created",
        "approved",
    ]
    assert all(
        isinstance(event.payload, LearningProposalChangedPayload)
        for event in learning_events
    )
    assert all("content" not in event.to_dict()["payload"] for event in learning_events)


def test_tool_permission_and_plan_lifecycle_use_curated_names(tmp_path):
    tool_llm = StreamingLLM(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "calculator",
                    "arguments_json": json.dumps({"expression": "2 + 2"}),
                }
            ),
            json.dumps({"type": "final_answer", "content": "Four."}),
        ]
    )
    tool_agent = Agent(
        config=AgentConfig(project_root=tmp_path / "tool"),
        llm=tool_llm,
        tools=[Tools.calculator],
        skills=[],
    )

    tool_names = [event.name for event in tool_agent.run_events("Calculate 2 + 2")]

    assert "permission.requested" in tool_names
    assert "permission.resolved" in tool_names
    assert "tool.call.started" in tool_names
    assert "tool.call.completed" in tool_names

    plan_payload = {
        "summary": "Update the documentation.",
        "steps": [
            {
                "id": "1",
                "title": "Edit documentation",
                "description": "Apply the requested documentation update.",
                "status": "pending",
            }
        ],
    }
    plan_agent = _agent(
        tmp_path / "plan",
        llm=StreamingLLM(
            [
                json.dumps(
                    {
                        "type": "plan",
                        "content": None,
                        "tool_name": None,
                        "arguments_json": "{}",
                        "plan_json": json.dumps(plan_payload),
                    }
                )
            ]
        ),
    )
    plan_events: list[AgentEvent] = []

    plan_agent.plan_result("Update the documentation safely", on_event=plan_events.append)

    assert EventName.PLAN_CREATED.value in [event.name for event in plan_events]


def test_callback_and_generator_receive_public_events_without_terminal_duplication(tmp_path):
    constructor_events: list[AgentEvent] = []
    run_callbacks: list[AgentEvent] = []
    deltas: list[str] = []
    facade = _agent(tmp_path, on_event=constructor_events.append)

    generated = list(
        facade.run_events(
            "hello",
            on_event=run_callbacks.append,
            on_delta=deltas.append,
        )
    )

    assert [event.name for event in constructor_events] == [event.name for event in run_callbacks]
    assert generated[-1].name == EventName.RUN_COMPLETED.value
    assert sum(event.name == EventName.RUN_COMPLETED.value for event in generated) == 1
    assert sum(event.name == EventName.RUN_COMPLETED.value for event in run_callbacks) == 1
    assert "".join(deltas) == "done"


def test_generator_failure_is_reported_in_band(tmp_path):
    facade = _agent(tmp_path)
    facade.close()

    events = list(facade.run_events("cannot run"))

    assert len(events) == 1
    assert events[0].name == EventName.RUN_FAILED.value
    assert isinstance(events[0].payload, RunFailedPayload)
    assert events[0].payload.error["category"] == "configuration"


def test_sync_generator_does_not_reuse_a_previous_failed_turn(tmp_path):
    facade = _agent(tmp_path)

    def fail_first_run(event: AgentEvent) -> None:
        if event.name == EventName.RUN_STARTED.value:
            raise RuntimeError("first run failed")

    first_events = list(facade.run_events("first", on_event=fail_first_run))
    previous_turn_id = first_events[-1].turn_id

    events = list(facade.run_events(""))

    assert len(events) == 1
    assert events[0].name == EventName.RUN_FAILED.value
    assert events[0].turn_id is None
    assert isinstance(events[0].payload, RunFailedPayload)
    assert events[0].payload.error["category"] == "configuration"
    assert "user_message cannot be empty" in events[0].payload.error["message"]
    assert "result" not in events[0].payload.error
    assert previous_turn_id is not None


@pytest.mark.asyncio
async def test_async_generator_does_not_reuse_a_previous_failed_turn_after_close(tmp_path):
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=StreamingLLM(),
        tools=[],
        skills=[],
    )

    def fail_first_run(event: AgentEvent) -> None:
        if event.name == EventName.RUN_STARTED.value:
            raise RuntimeError("first run failed")

    first_events = [event async for event in facade.run_events_async("first", on_event=fail_first_run)]
    previous_turn_id = first_events[-1].turn_id
    await facade.close()

    events = [event async for event in facade.run_events_async("second")]

    assert len(events) == 1
    assert events[0].name == EventName.RUN_FAILED.value
    assert events[0].turn_id is None
    assert isinstance(events[0].payload, RunFailedPayload)
    assert events[0].payload.error["category"] == "configuration"
    assert "result" not in events[0].payload.error
    assert previous_turn_id is not None


def test_sync_generator_terminates_when_run_callback_raises(tmp_path):
    facade = _agent(tmp_path)
    callback_events: list[str] = []

    def fail_callback(event: AgentEvent) -> None:
        callback_events.append(event.name)
        raise RuntimeError("callback failed")

    events = list(facade.run_events("hello", on_event=fail_callback))

    assert [event.name for event in events] == [EventName.RUN_STARTED.value, EventName.RUN_FAILED.value]
    assert callback_events == [EventName.RUN_STARTED.value, EventName.RUN_FAILED.value]


def test_event_channel_finish_preserves_queued_events_under_backpressure():
    channel = RunEventChannel(max_events=2)

    def event(name: str) -> AgentEvent:
        return AgentEvent(
            name=name,
            conversation_id="conversation",
            payload=RunStartedPayload(message=name),
        )

    first = event("test.first")
    second = event("test.second")
    terminal = event(EventName.RUN_COMPLETED.value)
    channel.publish(first)
    channel.publish(second)
    producer = threading.Thread(target=channel.finish, args=(terminal,))
    producer.start()

    assert channel.get() is first
    assert channel.get() is second
    assert channel.get() is terminal
    producer.join(timeout=1)
    assert not producer.is_alive()
    assert not isinstance(channel.get(), AgentEvent)


def test_sync_generator_early_close_cleans_callbacks_and_gate(tmp_path):
    llm = StreamingLLM(
        [
            json.dumps({"type": "final_answer", "content": "first"}),
            json.dumps({"type": "final_answer", "content": "second"}),
        ],
        delay=0.03,
    )
    facade = _agent(tmp_path, llm=llm)
    stream = facade.run_events("first")

    assert next(stream).name == EventName.RUN_STARTED.value
    stream.close()

    assert facade._handle._active_on_event is None
    assert facade.run("second") == "second"


@pytest.mark.asyncio
async def test_async_generator_matches_sync_sequence_and_terminal_result(tmp_path):
    sync_facade = _agent(tmp_path / "sync")
    async_facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path / "async"),
        llm=StreamingLLM(),
        tools=[],
        skills=[],
    )
    sync_names = [event.name for event in sync_facade.run_events("hello")]
    async_events = [event async for event in async_facade.run_events_async("hello")]

    assert [event.name for event in async_events] == sync_names
    assert isinstance(async_events[-1].payload, RunCompletedPayload)
    assert async_events[-1].payload.result.content == "done"


@pytest.mark.asyncio
async def test_async_generator_terminates_when_run_callback_raises(tmp_path):
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=StreamingLLM(),
        tools=[],
        skills=[],
    )
    callback_events: list[str] = []

    def fail_callback(event: AgentEvent) -> None:
        callback_events.append(event.name)
        raise RuntimeError("callback failed")

    async def collect() -> list[AgentEvent]:
        return [event async for event in facade.run_events_async("hello", on_event=fail_callback)]

    events = await asyncio.wait_for(collect(), timeout=1)

    assert [event.name for event in events] == [EventName.RUN_STARTED.value, EventName.RUN_FAILED.value]
    assert callback_events == [EventName.RUN_STARTED.value, EventName.RUN_FAILED.value]


@pytest.mark.asyncio
async def test_async_generator_close_cancels_delivery_and_releases_gate(tmp_path):
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=StreamingLLM(
            [
                json.dumps({"type": "final_answer", "content": "first"}),
                json.dumps({"type": "final_answer", "content": "second"}),
            ],
            delay=0.03,
        ),
        tools=[],
        skills=[],
    )
    stream = facade.run_events_async("first")

    first = await anext(stream)
    assert first.name == EventName.RUN_STARTED.value
    await stream.aclose()

    assert facade._handle.handle._active_on_event is None
    assert await facade.run("second") == "second"


@pytest.mark.asyncio
async def test_async_consumer_cancellation_keeps_standard_semantics(tmp_path):
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=StreamingLLM(delay=0.05),
        tools=[],
        skills=[],
    )

    async def consume() -> None:
        async for _event in facade.run_events_async("hello"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert facade._handle.handle._active_on_event is None
