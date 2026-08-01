"""Concurrency policy tests for public Agent facades."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from chulk import Agent, AgentConfig, AgentEvent, AsyncAgent
from chulk.llm import LLMClient


class SerializedProbeLLM(LLMClient):
    provider = "test"
    model = "serialized"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.block_first = False

    def complete(self, messages, *, max_output_tokens=None) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            call_number = len(getattr(self, "calls", []))
            self.calls = [*getattr(self, "calls", []), call_number]
        try:
            if self.block_first and call_number == 0:
                self.first_started.set()
                assert self.release_first.wait(timeout=2)
            return json.dumps({"type": "final_answer", "content": "done"})
        finally:
            with self._lock:
                self.active -= 1


def _sync_agent(tmp_path, llm: LLMClient) -> Agent:
    return Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
    )


def test_overlapping_sync_calls_wait_in_submission_order(tmp_path):
    llm = SerializedProbeLLM()
    llm.block_first = True
    facade = _sync_agent(tmp_path, llm)
    completed: list[str] = []

    first = threading.Thread(target=lambda: completed.append(facade.run("first")))
    second = threading.Thread(target=lambda: completed.append(facade.run("second")))
    first.start()
    assert llm.first_started.wait(timeout=2)
    second.start()
    assert len(llm.calls) == 1
    llm.release_first.set()
    first.join()
    second.join()

    assert completed == ["done", "done"]
    assert [turn.user_message for turn in facade.state.turns] == ["first", "second"]
    assert llm.max_active == 1


def test_one_hundred_sync_runs_have_no_event_cross_talk(tmp_path):
    facade = _sync_agent(tmp_path, SerializedProbeLLM())
    event_groups: list[list[AgentEvent]] = [[] for _ in range(100)]
    threads = [
        threading.Thread(
            target=facade.run,
            args=(f"sync-{index}",),
            kwargs={"on_event": event_groups[index].append},
        )
        for index in range(100)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(facade.state.turns) == 100
    for events in event_groups:
        turn_ids = {event.turn_id for event in events}
        assert len(turn_ids) == 1
        assert events[0].name == "run.started"
        assert events[-1].name == "run.completed"


@pytest.mark.asyncio
async def test_one_hundred_async_runs_serialize_and_keep_event_ownership(tmp_path):
    llm = SerializedProbeLLM()
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
    )
    event_groups: list[list[AgentEvent]] = [[] for _ in range(100)]

    await asyncio.gather(
        *(
            facade.run(f"async-{index}", on_event=event_groups[index].append)
            for index in range(100)
        )
    )

    assert [turn.user_message for turn in facade.state.turns] == [f"async-{index}" for index in range(100)]
    assert llm.max_active == 1
    for events in event_groups:
        assert len({event.turn_id for event in events}) == 1
        assert events[-1].name == "run.completed"


@pytest.mark.asyncio
async def test_separate_agents_can_run_independently(tmp_path):
    first_llm = SerializedProbeLLM()
    second_llm = SerializedProbeLLM()
    first = AsyncAgent(
        config=AgentConfig(project_root=tmp_path / "first"),
        llm=first_llm,
        tools=[],
        skills=[],
    )
    second = AsyncAgent(
        config=AgentConfig(project_root=tmp_path / "second"),
        llm=second_llm,
        tools=[],
        skills=[],
    )

    first_events, second_events = [], []
    await asyncio.gather(
        first.run("first", on_event=first_events.append),
        second.run("second", on_event=second_events.append),
    )

    assert {event.conversation_id for event in first_events} == {first.conversation_id}
    assert {event.conversation_id for event in second_events} == {second.conversation_id}
    assert first.conversation_id != second.conversation_id
