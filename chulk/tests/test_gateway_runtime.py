"""Outside-in tests for shared gateway execution and delivery."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    DeliveryTarget,
    GatewayBackpressureError,
    GatewayLimits,
    GatewayRuntime,
    InboundEnvelope,
    OutboundEnvelope,
    SQLiteGatewayLedger,
    SQLiteGatewayRouter,
    TextPart,
    TrustLevel,
)


def _envelope(event_id: str, *, destination: str = "chat-9") -> InboundEnvelope:
    return InboundEnvelope(
        event_id=event_id,
        idempotency_key=f"fake:primary:{event_id}",
        identity=ChannelIdentity("fake", "primary", "user-7"),
        destination_id=destination,
        parts=(TextPart(f"message {event_id}"),),
        scope=ChannelScope.DIRECT,
        authentication=AuthenticationState.AUTHENTICATED,
        trust=TrustLevel.TRUSTED,
    )


class FakeAdapter:
    name = "fake"
    account_id = "primary"

    def __init__(self) -> None:
        self.acknowledged: list[str] = []
        self.delivered: list[str] = []
        self.closed = False

    async def receive(self) -> AsyncIterator[InboundEnvelope]:
        if False:
            yield _envelope("unused")

    async def deliver(self, envelope: OutboundEnvelope) -> DeliveryReceipt:
        self.delivered.append(envelope.text or "")
        return DeliveryReceipt(
            envelope_id=envelope.envelope_id,
            state=DeliveryState.DELIVERED,
            attempt=1,
        )

    async def acknowledge(self, envelope: InboundEnvelope) -> None:
        self.acknowledged.append(envelope.event_id)

    async def close(self) -> None:
        self.closed = True


def _runtime(tmp_path, executor, *, max_pending: int = 10):
    ledger = SQLiteGatewayLedger(tmp_path / "control.sqlite")
    router = SQLiteGatewayRouter(tmp_path / "control.sqlite")
    router.add_route(
        adapter="fake",
        account_id="primary",
        principal_id="user-7",
        profile_id="work",
    )
    adapter = FakeAdapter()
    runtime = GatewayRuntime(
        ledger=ledger,
        router=router,
        adapters=(adapter,),
        executor=executor,
        limits=GatewayLimits(
            global_concurrency=2,
            profile_concurrency=1,
            max_pending=max_pending,
        ),
    )
    return runtime, ledger, adapter


@pytest.mark.asyncio
async def test_runtime_acknowledges_only_after_durable_ingest_then_delivers(tmp_path) -> None:
    calls: list[tuple[str, str]] = []

    async def execute(
        profile_id: str,
        envelope: InboundEnvelope,
    ) -> tuple[OutboundEnvelope, ...]:
        calls.append((profile_id, envelope.event_id))
        return (
            OutboundEnvelope(
                profile_id=profile_id,
                conversation_id="conversation-1",
                target=DeliveryTarget("fake", "primary", envelope.destination_id),
                text=f"answer {envelope.event_id}",
            ),
        )

    runtime, ledger, adapter = _runtime(tmp_path, execute)
    assert await runtime.accept(adapter, _envelope("1"))
    assert adapter.acknowledged == ["1"]

    assert await runtime.run_once() == (1, 1)
    assert calls == [("work", "1")]
    assert adapter.delivered == ["answer 1"]
    assert ledger.pending_count() == 0


@pytest.mark.asyncio
async def test_unrouted_identity_is_durably_ignored_without_execution(tmp_path) -> None:
    async def execute(
        _profile_id: str,
        _envelope: InboundEnvelope,
    ) -> tuple[OutboundEnvelope, ...]:
        raise AssertionError("unrouted work must not execute")

    runtime, ledger, adapter = _runtime(tmp_path, execute)
    unknown = InboundEnvelope(
        event_id="2",
        idempotency_key="fake:primary:2",
        identity=ChannelIdentity("fake", "primary", "unknown"),
        destination_id="chat-9",
        parts=(TextPart("hello"),),
        scope=ChannelScope.DIRECT,
        authentication=AuthenticationState.AUTHENTICATED,
        trust=TrustLevel.UNTRUSTED,
    )

    assert not await runtime.accept(adapter, unknown)
    assert adapter.acknowledged == ["2"]
    assert ledger.pending_count() == 0


@pytest.mark.asyncio
async def test_backpressure_leaves_transport_event_unacknowledged(tmp_path) -> None:
    async def execute(
        _profile_id: str,
        _envelope: InboundEnvelope,
    ) -> tuple[OutboundEnvelope, ...]:
        return ()

    runtime, _ledger, adapter = _runtime(tmp_path, execute, max_pending=1)
    assert await runtime.accept(adapter, _envelope("1"))

    with pytest.raises(GatewayBackpressureError):
        await runtime.accept(adapter, _envelope("2", destination="chat-10"))
    assert adapter.acknowledged == ["1"]


@pytest.mark.asyncio
async def test_active_cancellation_reaches_executor_and_terminal_state(tmp_path) -> None:
    started = asyncio.Event()

    async def execute(
        _profile_id: str,
        _envelope: InboundEnvelope,
    ) -> tuple[OutboundEnvelope, ...]:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    runtime, ledger, adapter = _runtime(tmp_path, execute)
    await runtime.accept(adapter, _envelope("1"))
    processing = asyncio.create_task(runtime.process_available())
    await started.wait()
    inbox_id = next(iter(runtime._active_executions))

    assert await runtime.cancel(inbox_id)
    await processing

    record = ledger.get_inbox(inbox_id)
    assert record is not None
    assert record.state == "cancelled"
