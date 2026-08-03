from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import asyncio
import json

import pytest

from chulk import (
    Agent,
    AgentConfig,
    AsyncAgent,
    AsyncHostedRuntime,
    EventName,
    FinalAnswerDeliveryStatus,
    FinalAnswerPolicyDecision,
    FinalAnswerStreamingMode,
    ExecutionScope,
    OutputPolicyFailureMode,
)
from chulk.hosting.reference import InMemoryServiceHub
from chulk.llm import FallbackChain, LLMClient, LLMError, LLMStreamChunk


class IncrementalLLM(LLMClient):
    provider = "test"
    model = "incremental"

    def __init__(self, chunks: tuple[str, ...], *, fail_after: int | None = None) -> None:
        self.chunks = chunks
        self.fail_after = fail_after
        self.completed = False
        self.sync_stream_called = False
        self.async_stream_called = False

    def complete(self, messages, *, max_output_tokens=None) -> str:
        return json.dumps({"type": "final_answer", "content": "validated intent"})

    def stream_final_answer(
        self, messages, *, max_output_tokens=None
    ) -> Iterator[LLMStreamChunk]:
        self.sync_stream_called = True
        if self.fail_after == 0 and not self.chunks:
            raise LLMError(
                "stream failed", code="server_error", retryable=True, fallback_eligible=True
            )
        for index, text in enumerate(self.chunks):
            if self.fail_after == index:
                raise LLMError(
                    "stream failed",
                    code="server_error",
                    retryable=True,
                    fallback_eligible=True,
                )
            yield LLMStreamChunk(type="text_delta", text=text)
        self.completed = True
        yield LLMStreamChunk(type="completed")

    async def astream_final_answer(
        self, messages, *, max_output_tokens=None
    ) -> AsyncIterator[LLMStreamChunk]:
        self.async_stream_called = True
        if self.fail_after == 0 and not self.chunks:
            raise LLMError(
                "stream failed", code="server_error", retryable=True, fallback_eligible=True
            )
        for index, text in enumerate(self.chunks):
            if self.fail_after == index:
                raise LLMError(
                    "stream failed",
                    code="server_error",
                    retryable=True,
                    fallback_eligible=True,
                )
            yield LLMStreamChunk(type="text_delta", text=text)
            await asyncio.sleep(0)
        self.completed = True
        yield LLMStreamChunk(type="completed")


class UppercasePolicy:
    def process(self, chunk):
        return FinalAnswerPolicyDecision(text=chunk.text.upper())

    def complete(self, *, turn_id: str, next_sequence: int):
        return FinalAnswerPolicyDecision()

    def reset(self, *, turn_id: str):
        return None


class BlockingPolicy:
    def process(self, chunk):
        return FinalAnswerPolicyDecision(blocked=True, reason="unsafe output")

    def complete(self, *, turn_id: str, next_sequence: int):
        return FinalAnswerPolicyDecision()

    def reset(self, *, turn_id: str):
        return None


class AsyncUppercasePolicy:
    async def process(self, chunk):
        return FinalAnswerPolicyDecision(text=chunk.text.upper())

    async def complete(self, *, turn_id: str, next_sequence: int):
        return FinalAnswerPolicyDecision()

    async def reset(self, *, turn_id: str):
        return None


class BufferingPolicy:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.reset_count = 0

    def process(self, chunk):
        self.parts.append(chunk.text)
        return FinalAnswerPolicyDecision()

    def complete(self, *, turn_id: str, next_sequence: int):
        return FinalAnswerPolicyDecision(text="".join(self.parts))

    def reset(self, *, turn_id: str):
        self.parts.clear()
        self.reset_count += 1


def _agent(tmp_path, llm: LLMClient, **kwargs) -> Agent:
    return Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL,
        **kwargs,
    )


def test_incremental_deltas_precede_provider_completion_and_reconstruct_result(tmp_path) -> None:
    llm = IncrementalLLM(("hello ", "world"))
    observed_before_completion: list[bool] = []

    def on_event(event) -> None:
        if event.name == EventName.MODEL_DELTA.value:
            observed_before_completion.append(not llm.completed)

    facade = _agent(tmp_path, llm, on_event=on_event)
    events = list(facade.run_events("hello"))
    deltas = [event.payload.text for event in events if event.name == EventName.MODEL_DELTA.value]
    result = events[-1].payload.result

    assert deltas == ["hello ", "world"]
    assert "".join(deltas) == result.content == "hello world"
    assert observed_before_completion == [True, True]
    assert result.final_answer_delivery is not None
    assert result.final_answer_delivery.status is FinalAnswerDeliveryStatus.COMPLETE
    assert result.final_answer_delivery.public_delta_count == 2


def test_policy_runs_before_publication_and_blocked_text_never_leaks(tmp_path) -> None:
    transformed = _agent(tmp_path / "transformed", IncrementalLLM(("secret",)), output_policy=UppercasePolicy())
    transformed_events = list(transformed.run_events("hello"))
    assert [
        event.payload.text
        for event in transformed_events
        if event.name == EventName.MODEL_DELTA.value
    ] == ["SECRET"]
    assert transformed_events[-1].payload.result.content == "SECRET"

    blocked = _agent(tmp_path / "blocked", IncrementalLLM(("must-not-leak",)), output_policy=BlockingPolicy())
    blocked_events = list(blocked.run_events("hello"))
    assert not [event for event in blocked_events if event.name == EventName.MODEL_DELTA.value]
    result = blocked_events[-1].payload.error["result"]
    assert result["content"] == ""
    assert result["final_answer_delivery"]["status"] == FinalAnswerDeliveryStatus.BLOCKED.value


def test_fail_closed_policy_exception_never_leaks_rejected_chunk(tmp_path) -> None:
    class BrokenPolicy:
        def process(self, chunk):
            raise RuntimeError("policy unavailable")

        def complete(self, *, turn_id: str, next_sequence: int):
            return FinalAnswerPolicyDecision()

        def reset(self, *, turn_id: str):
            return None

    facade = _agent(
        tmp_path,
        IncrementalLLM(("rejected",)),
        output_policy=BrokenPolicy(),
        output_policy_failure_mode=OutputPolicyFailureMode.CLOSED,
    )
    events = list(facade.run_events("hello"))
    assert not [event for event in events if event.name == EventName.MODEL_DELTA.value]
    assert events[-1].payload.error["result"]["final_answer_delivery"]["status"] == FinalAnswerDeliveryStatus.BLOCKED.value


def test_fallback_only_switches_before_stream_output(tmp_path) -> None:
    first = IncrementalLLM((), fail_after=0)
    second = IncrementalLLM(("fallback",))
    facade = _agent(tmp_path / "before", FallbackChain([first, second]))
    assert facade.run_result("hello").content == "fallback"
    assert second.sync_stream_called

    buffered_policy = BufferingPolicy()
    buffered_failure = IncrementalLLM(("discarded", "never"), fail_after=1)
    buffered_fallback = IncrementalLLM(("permitted",))
    facade = _agent(
        tmp_path / "buffered",
        FallbackChain([buffered_failure, buffered_fallback]),
        output_policy=buffered_policy,
    )
    assert facade.run_result("hello").content == "permitted"
    assert buffered_policy.reset_count == 1

    partial = IncrementalLLM(("partial", "never"), fail_after=1)
    forbidden = IncrementalLLM(("forbidden",))
    facade = _agent(tmp_path / "after", FallbackChain([partial, forbidden]))
    result = facade.run_result("hello")
    assert result.content == "partial"
    assert result.final_answer_delivery.status is FinalAnswerDeliveryStatus.FAILED
    assert not forbidden.sync_stream_called


@pytest.mark.asyncio
async def test_async_stream_uses_native_iterator_and_async_policy(tmp_path) -> None:
    llm = IncrementalLLM(("native ", "async"))
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL,
        async_output_policy=AsyncUppercasePolicy(),
    )
    events = [event async for event in facade.run_events_async("hello")]
    deltas = [event.payload.text for event in events if event.name == EventName.MODEL_DELTA.value]
    assert llm.async_stream_called
    assert not llm.sync_stream_called
    assert deltas == ["NATIVE ", "ASYNC"]
    assert events[-1].payload.result.content == "NATIVE ASYNC"


@pytest.mark.asyncio
async def test_async_hosted_create_awaits_policy_and_honors_failure_mode(
    tmp_path,
) -> None:
    class RecordingPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        async def process(self, chunk):
            self.calls.append(("process", chunk.text))
            return FinalAnswerPolicyDecision(text=chunk.text.upper())

        async def complete(self, *, turn_id: str, next_sequence: int):
            self.calls.append(("complete", next_sequence))
            return FinalAnswerPolicyDecision(text="!")

        async def reset(self, *, turn_id: str):
            self.calls.append(("reset", turn_id))

    scope = ExecutionScope.local()
    policy = RecordingPolicy()
    runtime = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path / "policy"),
        llm=IncrementalLLM(("native ", "hosted")),
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=scope,
        final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL,
        async_output_policy=policy,
    )
    events = [event async for event in runtime.run_events_async("hello")]
    assert [
        event.payload.text
        for event in events
        if event.name == EventName.MODEL_DELTA.value
    ] == ["NATIVE ", "HOSTED", "!"]
    assert events[-1].payload.result.content == "NATIVE HOSTED!"
    assert policy.calls == [
        ("process", "native "),
        ("process", "hosted"),
        ("complete", 2),
    ]
    await runtime.close()

    class BrokenPolicy(RecordingPolicy):
        async def process(self, chunk):
            raise RuntimeError("policy unavailable")

    fail_open = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path / "fail-open"),
        llm=IncrementalLLM(("permitted",)),
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=scope,
        final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL,
        async_output_policy=BrokenPolicy(),
        output_policy_failure_mode=OutputPolicyFailureMode.OPEN,
    )
    assert (await fail_open.run_result("hello")).content == "permitted!"
    await fail_open.close()


@pytest.mark.asyncio
async def test_closing_async_consumer_cancels_provider_and_terminalizes_turn(tmp_path) -> None:
    class WaitingLLM(IncrementalLLM):
        def __init__(self) -> None:
            super().__init__(("partial",))
            self.closed = False

        async def astream_final_answer(self, messages, *, max_output_tokens=None):
            self.async_stream_called = True
            try:
                yield LLMStreamChunk(type="text_delta", text="partial")
                await asyncio.Event().wait()
            finally:
                self.closed = True

    llm = WaitingLLM()
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL,
    )
    stream = facade.run_events_async("hello")
    while True:
        event = await anext(stream)
        if event.name == EventName.MODEL_DELTA.value:
            assert event.payload.text == "partial"
            break
    await asyncio.wait_for(stream.aclose(), timeout=1)

    turn = facade.runtime.state.turns[-1]
    assert llm.closed
    assert turn.status == "cancelled"
    assert turn.extension_metadata["final_answer_delivery"]["public_delta_count"] == 1
    assert turn.extension_metadata["final_answer_delivery"]["error"] == "cancelled"
