"""Externally owned transcript contracts for hosted runs."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest

from chulk import (
    AgentConfig,
    AsyncExternalTranscriptSessionRuntimeServices,
    AsyncHostedRuntime,
    AsyncServiceBinding,
    ChulkError,
    ExecutionScope,
    ExternalTranscriptSessionRuntimeServices,
    ExternalTranscriptSnapshot,
    HostedRuntime,
    ServiceBinding,
    TranscriptMessage,
    TranscriptConflictError,
    TranscriptResolutionError,
)
from chulk.hosting.reference import InMemoryServiceHub
from chulk.core.state import ToolCallRecord, TurnState
from chulk.hosting.transcript_reference import (
    AsyncInMemoryExecutionJournal,
    AsyncInMemoryTranscriptProjectionSink,
    InMemoryExecutionJournal,
    InMemoryTranscriptProjectionSink,
)
from chulk.llm import LLMClient
from chulk.llm.messages import (
    chat_messages,
    local_chat_messages,
    split_instructions,
)
from chulk.llm.providers.anthropic import _normalize_conversation


class RecordingLLM(LLMClient):
    def __init__(self, content: str = "hosted answer") -> None:
        self.content = content
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.requests.append(messages)
        return json.dumps({"type": "final_answer", "content": self.content})


class QueueLLM(RecordingLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = responses

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.requests.append(messages)
        return self.responses.pop(0)


def _scope(
    conversation_id: str = "conversation-1",
    *,
    tenant_id: str = "tenant-a",
) -> ExecutionScope:
    return ExecutionScope(
        tenant_id=tenant_id,
        workspace_id="workspace",
        actor_id="default",
        agent_id="support",
        agent_version="1",
        run_id="run-1",
        conversation_id=conversation_id,
    )


def _snapshot(
    content: str = "authoritative prior message",
) -> ExternalTranscriptSnapshot:
    return ExternalTranscriptSnapshot(
        conversation_id="conversation-1",
        revision="revision-1",
        messages=(
            TranscriptMessage(
                id="message-1",
                role="user",
                content=content,
                ordinal=1,
                created_at="2026-08-02T10:00:00Z",
            ),
            TranscriptMessage(
                id="message-2",
                role="assistant",
                content="authoritative prior answer",
                ordinal=2,
                created_at="2026-08-02T10:00:01Z",
            ),
        ),
    )


def _sync_services(
    journal: InMemoryExecutionJournal,
    sink: InMemoryTranscriptProjectionSink,
):
    base = InMemoryServiceHub().services()
    return replace(
        base,
        sessions=ServiceBinding.host(
            ExternalTranscriptSessionRuntimeServices(journal, sink)
        ),
    )


def _async_services(
    journal: AsyncInMemoryExecutionJournal,
    sink: AsyncInMemoryTranscriptProjectionSink,
):
    base = InMemoryServiceHub().async_services()
    return replace(
        base,
        sessions=AsyncServiceBinding.host(
            AsyncExternalTranscriptSessionRuntimeServices(journal, sink)
        ),
    )


def test_external_transcript_is_authoritative_and_not_duplicated(
    tmp_path: Path,
) -> None:
    journal = InMemoryExecutionJournal()
    sink = InMemoryTranscriptProjectionSink()
    llm = RecordingLLM()
    snapshot = _snapshot()
    requests = []
    runtime = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        services=_sync_services(journal, sink),
        execution_scope=_scope(),
        conversation_id="conversation-1",
        transcript_resolver=lambda request: requests.append(request) or snapshot,
    )

    assert runtime.run("current host input") == "hosted answer"

    assert len(requests) == 1
    assert requests[0].scope == _scope()
    assert any(
        message["content"] == "authoritative prior message"
        for message in llm.requests[0]
    )
    turn = journal.load_turns("conversation-1")[-1]
    assert turn.user_message == "[externally owned transcript]"
    assert turn.final_answer is None
    assert len(sink.projections) == 1
    projection = sink.projections[0]
    assert projection.content == "hosted answer"
    assert projection.input_transcript_digest == snapshot.digest
    assert projection.conversation_id == "conversation-1"
    assert list(tmp_path.iterdir()) == []
    runtime.close()


def test_external_transcript_snapshot_is_validated_and_immutable() -> None:
    metadata = {"source": ["crm"]}
    message = TranscriptMessage(
        id="message-1",
        role="user",
        content="hello",
        ordinal=1,
        metadata=metadata,
    )
    metadata["source"].append("mutated")
    snapshot = ExternalTranscriptSnapshot(
        conversation_id="conversation-1",
        messages=(message,),
        summary="older context",
        summary_message_count=3,
    )

    assert message.metadata == {"source": ["crm"]}
    assert snapshot.digest == ExternalTranscriptSnapshot(
        conversation_id="conversation-1",
        messages=(message,),
        summary="older context",
        summary_message_count=3,
    ).digest
    with pytest.raises(ValueError, match="duplicate message ids"):
        ExternalTranscriptSnapshot(
            conversation_id="conversation-1",
            messages=(message, replace(message, ordinal=2)),
        )
    with pytest.raises(ValueError, match="requires a transcript summary"):
        ExternalTranscriptSnapshot(
            conversation_id="conversation-1",
            summary_message_count=1,
        )


def test_tool_roles_project_to_portable_semantic_context() -> None:
    snapshot = ExternalTranscriptSnapshot(
        conversation_id="conversation-1",
        messages=(
            TranscriptMessage("message-1", "system", "instructions", 1),
            TranscriptMessage("message-2", "assistant", "I checked", 2),
            TranscriptMessage("message-3", "tool", "tool result", 3),
            TranscriptMessage(
                "message-4",
                "observation",
                "host observation",
                4,
            ),
        ),
    )

    projected = snapshot.prompt_messages()
    assert projected == [
        {"role": "system", "content": "instructions"},
        {"role": "assistant", "content": "I checked"},
        {
            "role": "user",
            "content": "[External tool context]\ntool result",
        },
        {
            "role": "user",
            "content": "[External observation context]\nhost observation",
        },
    ]

    _, responses_messages = split_instructions(projected)
    assert {message["role"] for message in responses_messages} <= {
        "user",
        "assistant",
    }
    assert {message["role"] for message in chat_messages(projected)} <= {
        "system",
        "user",
        "assistant",
    }
    assert {
        message["role"] for message in _normalize_conversation(projected)
    } <= {"user", "assistant"}
    assert {
        message["role"] for message in local_chat_messages(projected)
    } <= {"user", "assistant"}


def test_invalid_tool_context_fails_before_provider_work(tmp_path: Path) -> None:
    llm = RecordingLLM()
    runtime = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        services=_sync_services(
            InMemoryExecutionJournal(),
            InMemoryTranscriptProjectionSink(),
        ),
        execution_scope=_scope(),
        transcript_resolver=lambda _request: ExternalTranscriptSnapshot(
            conversation_id="conversation-1",
            messages=(
                TranscriptMessage("message-tool", "tool", " ", 1),
            ),
        ),
    )

    with pytest.raises(
        TranscriptResolutionError,
        match="tool context cannot be empty",
    ):
        runtime.run("must not reach provider")
    assert llm.requests == []
    runtime.close()


def test_transcript_resolution_fails_before_provider_work(tmp_path: Path) -> None:
    journal = InMemoryExecutionJournal()
    llm = RecordingLLM()
    runtime = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        services=_sync_services(journal, InMemoryTranscriptProjectionSink()),
        execution_scope=_scope(),
        transcript_resolver=lambda _request: ExternalTranscriptSnapshot(
            conversation_id="another-conversation"
        ),
    )

    with pytest.raises(TranscriptResolutionError, match="conversation_id"):
        runtime.run("must not reach provider")
    assert llm.requests == []
    assert journal.load_turns("conversation-1") == []
    runtime.close()


def test_execution_journal_enforces_scope_isolation(tmp_path: Path) -> None:
    journal = InMemoryExecutionJournal()
    sink = InMemoryTranscriptProjectionSink()
    first = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=RecordingLLM(),
        tools=[],
        skills=[],
        services=_sync_services(journal, sink),
        execution_scope=_scope(),
        transcript_resolver=lambda _request: _snapshot(),
    )
    assert first.run("first") == "hosted answer"
    first.close()

    with pytest.raises(Exception, match="tenant"):
        HostedRuntime(
            config=AgentConfig(project_root=tmp_path),
            llm=RecordingLLM(),
            tools=[],
            skills=[],
            services=_sync_services(journal, sink),
            execution_scope=_scope(tenant_id="tenant-b"),
            transcript_resolver=lambda _request: _snapshot(),
        )


def test_projection_sink_is_idempotent(tmp_path: Path) -> None:
    journal = InMemoryExecutionJournal()
    sink = InMemoryTranscriptProjectionSink()
    runtime = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=RecordingLLM(),
        tools=[],
        skills=[],
        services=_sync_services(journal, sink),
        execution_scope=_scope(),
        transcript_resolver=lambda _request: _snapshot(),
    )
    assert runtime.run("first") == "hosted answer"
    projection = sink.projections[0]
    sink.emit(projection)
    assert sink.projections == [projection]
    runtime.close()


def test_plan_resume_rejects_a_changed_external_transcript(
    tmp_path: Path,
) -> None:
    plan = {
        "summary": "Change the record",
        "steps": [
            {
                "id": "1",
                "title": "Implement",
                "description": "Apply the requested change.",
                "status": "pending",
                "acceptance_criteria": ["The change is complete."],
            }
        ],
    }
    llm = QueueLLM(
        [
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": json.dumps(plan),
                    "step_update_json": "{}",
                }
            )
        ]
    )
    current = [_snapshot()]
    runtime = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        services=_sync_services(
            InMemoryExecutionJournal(),
            InMemoryTranscriptProjectionSink(),
        ),
        execution_scope=_scope(),
        transcript_resolver=lambda _request: current[0],
    )

    assert runtime.plan_result("prepare a change").status == "waiting_for_approval"
    current[0] = _snapshot("host changed the transcript")
    with pytest.raises(TranscriptConflictError, match="changed"):
        runtime.approve_result()
    assert len(llm.requests) == 1
    runtime.close()


@pytest.mark.asyncio
async def test_native_async_external_transcript_and_timeout(tmp_path: Path) -> None:
    journal = AsyncInMemoryExecutionJournal()
    sink = AsyncInMemoryTranscriptProjectionSink()
    llm = RecordingLLM("async answer")

    async def resolve(_request):
        return _snapshot()

    runtime = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        services=_async_services(journal, sink),
        execution_scope=_scope(),
        async_transcript_resolver=resolve,
    )
    assert await runtime.run("async input") == "async answer"
    assert sink.projections[0].content == "async answer"
    await runtime.close()

    slow_llm = RecordingLLM()

    async def slow(_request):
        await asyncio.sleep(1)
        return _snapshot()

    timed = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=slow_llm,
        tools=[],
        skills=[],
        services=_async_services(
            AsyncInMemoryExecutionJournal(),
            AsyncInMemoryTranscriptProjectionSink(),
        ),
        execution_scope=_scope(),
        async_transcript_resolver=slow,
        transcript_timeout_seconds=0.01,
    )
    with pytest.raises(TranscriptResolutionError, match="timed out"):
        await timed.run("timeout")
    assert slow_llm.requests == []
    await timed.close()


@pytest.mark.asyncio
async def test_native_async_recovery_persists_blocked_turn_once(
    tmp_path: Path,
) -> None:
    class CountingJournal(AsyncInMemoryExecutionJournal):
        def __init__(self) -> None:
            super().__init__()
            self.save_count = 0

        async def save_turn_snapshot(self, conversation_id, turn) -> None:
            self.save_count += 1
            await super().save_turn_snapshot(conversation_id, turn)

    journal = CountingJournal()
    scope = _scope()
    turn = TurnState(
        user_message="[externally owned transcript]",
        turn_id="turn-uncertain-effect",
    )
    turn.tool_call_count = 1
    turn.tool_calls.append(
        ToolCallRecord(
            tool_name="external_mutation",
            arguments={"value": "once"},
            iteration=1,
        )
    )
    await journal.bind_scope("conversation-1", scope)
    await journal.save_turn_snapshot("conversation-1", turn.to_dict())
    journal.save_count = 0

    sync_journal = InMemoryExecutionJournal()
    sync_journal.bind_scope("conversation-1", scope)
    sync_journal.save_turn_snapshot("conversation-1", turn.to_dict())
    sync_runtime = HostedRuntime(
        config=AgentConfig(project_root=tmp_path / "sync"),
        llm=RecordingLLM(),
        tools=[],
        skills=[],
        services=_sync_services(
            sync_journal,
            InMemoryTranscriptProjectionSink(),
        ),
        execution_scope=scope,
        transcript_resolver=lambda _request: _snapshot(),
    )
    sync_persisted = sync_journal.load_turns("conversation-1")[-1]
    assert sync_persisted.status == "blocked"
    sync_runtime.close()

    async def resolve(_request):
        return _snapshot()

    first = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path / "first"),
        llm=RecordingLLM(),
        tools=[],
        skills=[],
        services=_async_services(
            journal,
            AsyncInMemoryTranscriptProjectionSink(),
        ),
        execution_scope=scope,
        async_transcript_resolver=resolve,
    )
    assert first.state.turns[-1].status == "blocked"
    persisted = await journal.load_turns("conversation-1")
    assert persisted[-1].status == "blocked"
    assert persisted[-1].final_answer == sync_persisted.final_answer
    assert persisted[-1].user_message == "[externally owned transcript]"
    assert _snapshot().messages[0].content not in json.dumps(
        persisted[-1].to_dict()
    )
    assert journal.save_count == 1
    await first.close()

    second = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path / "second"),
        llm=RecordingLLM(),
        tools=[],
        skills=[],
        services=_async_services(
            journal,
            AsyncInMemoryTranscriptProjectionSink(),
        ),
        execution_scope=scope,
        async_transcript_resolver=resolve,
    )
    assert second.state.turns[-1].status == "blocked"
    assert journal.save_count == 1
    await second.close()


@pytest.mark.asyncio
async def test_native_async_recovery_surfaces_checkpoint_failure(
    tmp_path: Path,
) -> None:
    class FailingJournal(AsyncInMemoryExecutionJournal):
        fail_checkpoint = False

        async def save_turn_snapshot(self, conversation_id, turn) -> None:
            if self.fail_checkpoint:
                raise RuntimeError("recovery checkpoint unavailable")
            await super().save_turn_snapshot(conversation_id, turn)

    journal = FailingJournal()
    scope = _scope()
    turn = TurnState(
        user_message="[externally owned transcript]",
        turn_id="turn-failed-checkpoint",
    )
    turn.tool_calls.append(
        ToolCallRecord(
            tool_name="external_mutation",
            arguments={},
            iteration=1,
        )
    )
    await journal.bind_scope("conversation-1", scope)
    await journal.save_turn_snapshot("conversation-1", turn.to_dict())
    journal.fail_checkpoint = True

    async def resolve(_request):
        return _snapshot()

    with pytest.raises(ChulkError, match="recovery checkpoint unavailable") as exc:
        await AsyncHostedRuntime.create(
            config=AgentConfig(project_root=tmp_path),
            llm=RecordingLLM(),
            tools=[],
            skills=[],
            services=_async_services(
                journal,
                AsyncInMemoryTranscriptProjectionSink(),
            ),
            execution_scope=scope,
            async_transcript_resolver=resolve,
        )
    assert isinstance(exc.value.__cause__, RuntimeError)
