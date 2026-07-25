"""Outside-in tests for shared durable conversation dispatch."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading

import pytest

from chulk.config import load_config
from chulk.llm import LLMClient
from chulk.profiles import ProfileRuntimeFactory, SQLiteProfileStore
from chulk.runtime import create_agent
from chulk.server import ConversationDispatcher


class OrderedLLM(LLMClient):
    provider = "test"
    model = "ordered"

    def __init__(self) -> None:
        self.messages: list[str] = []

    def complete(self, messages, *, max_output_tokens=None) -> str:
        message = messages[-1]["content"]
        self.messages.append(message)
        return json.dumps({"type": "final_answer", "content": f"answer {message}"})


class BlockingLLM(LLMClient):
    provider = "test"
    model = "blocking"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def complete(self, messages, *, max_output_tokens=None) -> str:
        self.started.set()
        self.release.wait(timeout=5)
        return json.dumps({"type": "final_answer", "content": "late answer"})


def _dispatcher(tmp_path: Path, llm: LLMClient) -> ConversationDispatcher:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = SQLiteProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    )
    factory = ProfileRuntimeFactory(config, profile_store=store)

    def build(profile_id, conversation_id, metadata):
        resolved = factory.resolve(profile_id)
        return create_agent(
            resolved.config,
            llm_client=llm,
            tool_specs=[],
            skill_specs=[],
            conversation_id=conversation_id,
            conversation_metadata=dict(metadata or {}),
            profile_id=profile_id,
        )

    return ConversationDispatcher(factory, agent_builder=build)


@pytest.mark.asyncio
async def test_dispatcher_serializes_api_and_gateway_work_in_submission_order(
    tmp_path,
) -> None:
    llm = OrderedLLM()
    dispatcher = _dispatcher(tmp_path, llm)
    conversation = await dispatcher.create_conversation("default")

    first = await dispatcher.submit(
        "default",
        conversation["id"],
        "first",
        source="api",
        idempotency_key="one",
    )
    second_task = dispatcher.submit_and_wait(
        "default",
        conversation["id"],
        "second",
        source="gateway",
        idempotency_key="two",
    )
    first_done, second_done = await asyncio.gather(
        dispatcher.submit_and_wait(
            "default",
            conversation["id"],
            "first",
            source="api",
            idempotency_key="one",
        ),
        second_task,
    )

    assert first.id == first_done.id
    assert first_done.status == second_done.status == "completed"
    assert llm.messages == ["first", "second"]
    assert [
        record.event.name
        for record in dispatcher.journal("default", conversation["id"]).list(
            conversation["id"]
        )
    ][0] == "run.started"
    await dispatcher.close()


@pytest.mark.asyncio
async def test_dispatcher_returns_same_command_for_idempotent_replay(tmp_path) -> None:
    dispatcher = _dispatcher(tmp_path, OrderedLLM())
    conversation = await dispatcher.create_conversation("default")

    first = await dispatcher.submit(
        "default",
        conversation["id"],
        "hello",
        idempotency_key="same",
    )
    replay = await dispatcher.submit(
        "default",
        conversation["id"],
        "hello",
        idempotency_key="same",
    )

    assert replay.id == first.id
    with pytest.raises(ValueError, match="different command"):
        await dispatcher.submit(
            "default",
            conversation["id"],
            "different",
            idempotency_key="same",
        )
    await dispatcher.close()


@pytest.mark.asyncio
async def test_dispatcher_cancels_active_and_queued_commands(tmp_path) -> None:
    llm = BlockingLLM()
    dispatcher = _dispatcher(tmp_path, llm)
    conversation = await dispatcher.create_conversation("default")
    first = await dispatcher.submit(
        "default",
        conversation["id"],
        "first",
        idempotency_key="first",
    )
    second = await dispatcher.submit(
        "default",
        conversation["id"],
        "second",
        idempotency_key="second",
    )
    await asyncio.to_thread(llm.started.wait, 2)

    assert await dispatcher.cancel("default", conversation["id"])
    llm.release.set()
    await asyncio.sleep(0.05)

    assert dispatcher.get_command(
        "default", conversation["id"], first.id
    ).status == "cancelled"
    assert dispatcher.get_command(
        "default", conversation["id"], second.id
    ).status == "cancelled"
    await dispatcher.close()
