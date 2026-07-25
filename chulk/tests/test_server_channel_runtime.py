"""Channel command and conversation execution tests."""

from __future__ import annotations

import json

import pytest

from chulk.config import load_config
from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    InboundEnvelope,
    TextPart,
    TrustLevel,
)
from chulk.llm import LLMClient
from chulk.profiles import ProfileRuntimeFactory
from chulk.runtime import create_agent
from chulk.server.channel_runtime import ChannelConversationExecutor
from chulk.server.dispatcher import ConversationDispatcher


class ChannelLLM(LLMClient):
    provider = "test"
    model = "channel"

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return json.dumps(
            {"type": "final_answer", "content": f"answer {messages[-1]['content']}"}
        )


def _envelope(
    text: str,
    *,
    event_id: str,
    conversation_id: str | None = None,
) -> InboundEnvelope:
    return InboundEnvelope(
        event_id=event_id,
        idempotency_key=f"discord:primary:{event_id}",
        identity=ChannelIdentity("discord", "primary", "user-7"),
        destination_id="channel-9",
        parts=(TextPart(text),),
        scope=ChannelScope.DIRECT,
        authentication=AuthenticationState.AUTHENTICATED,
        trust=TrustLevel.TRUSTED,
        extensions={"conversation_id": conversation_id},
    )


def _executor(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    factory = ProfileRuntimeFactory(config)

    def build(profile_id, conversation_id, metadata):
        resolved = factory.resolve(profile_id)
        return create_agent(
            resolved.config,
            llm_client=ChannelLLM(),
            tool_specs=[],
            skill_specs=[],
            profile_id=profile_id,
            conversation_id=conversation_id,
            conversation_metadata=dict(metadata or {}),
        )

    dispatcher = ConversationDispatcher(factory, agent_builder=build)
    return ChannelConversationExecutor(dispatcher), dispatcher


@pytest.mark.asyncio
async def test_shared_new_status_and_normal_message_reuse_channel_conversation(
    tmp_path,
) -> None:
    execute, dispatcher = _executor(tmp_path)

    created = (await execute("default", _envelope("/new", event_id="1")))[0]
    status = (await execute("default", _envelope("/status", event_id="2")))[0]
    answered = (await execute("default", _envelope("hello", event_id="3")))[0]

    assert created.extensions["channel_command"] == "new"
    assert created.conversation_id == status.conversation_id
    assert answered.conversation_id == created.conversation_id
    assert answered.text == "answer hello"
    assert "Provider: openai" in status.text
    await dispatcher.close()


@pytest.mark.asyncio
async def test_shared_commands_do_not_enter_the_model_and_unknowns_fail_closed(
    tmp_path,
) -> None:
    execute, dispatcher = _executor(tmp_path)

    skills = (await execute("default", _envelope("/skills", event_id="1")))[0]
    memory = (await execute("default", _envelope("/memory", event_id="2")))[0]
    unknown = (await execute("default", _envelope("/explode", event_id="3")))[0]

    assert "files" in skills.text
    assert "Memory namespace:" in memory.text
    assert unknown.extensions["status"] == "failed"
    assert unknown.extensions["error"] == "unknown_channel_command"
    await dispatcher.close()


@pytest.mark.asyncio
async def test_requested_missing_conversation_returns_typed_failure(tmp_path) -> None:
    execute, dispatcher = _executor(tmp_path)

    result = (
        await execute(
            "default",
            _envelope("hello", event_id="1", conversation_id="missing"),
        )
    )[0]

    assert result.extensions["error"] == "conversation_not_found"
    assert result.conversation_id == "missing"
    await dispatcher.close()
