"""Shared channel ingestion, execution, and delivery orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
import logging

from chulk.gateway.ledger import ExecutionClaim, SQLiteGatewayLedger
from chulk.gateway.models import (
    DeliveryReceipt,
    DeliveryState,
    InboundEnvelope,
    OutboundEnvelope,
    TextPart,
)
from chulk.gateway.protocol import ChannelAdapter
from chulk.gateway.routing import SQLiteGatewayRouter


LOGGER = logging.getLogger(__name__)
UNROUTED_PROFILE_ID = "_gateway"
EnvelopeExecutor = Callable[
    [str, InboundEnvelope],
    Awaitable[tuple[OutboundEnvelope, ...]],
]


@dataclass(frozen=True, slots=True)
class GatewayLimits:
    """Bounded shared gateway queue and worker settings."""

    global_concurrency: int = 4
    profile_concurrency: int = 1
    max_pending: int = 1_000
    execution_lease_seconds: int = 300
    delivery_lease_seconds: int = 120
    idle_delay_seconds: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "global_concurrency",
            "profile_concurrency",
            "max_pending",
            "execution_lease_seconds",
            "delivery_lease_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.idle_delay_seconds <= 0:
            raise ValueError("idle_delay_seconds must be greater than zero")


class GatewayRuntime:
    """Run all adapters through one durable, profile-aware control plane."""

    def __init__(
        self,
        *,
        ledger: SQLiteGatewayLedger,
        router: SQLiteGatewayRouter,
        adapters: Iterable[ChannelAdapter],
        executor: EnvelopeExecutor,
        limits: GatewayLimits | None = None,
        propagate_delivery_errors: bool = False,
        delivery_retry_delay_seconds: float | None = None,
    ) -> None:
        self.ledger = ledger
        self.router = router
        self.executor = executor
        self.limits = limits or GatewayLimits()
        if (
            delivery_retry_delay_seconds is not None
            and delivery_retry_delay_seconds < 0
        ):
            raise ValueError("delivery_retry_delay_seconds cannot be negative")
        self.propagate_delivery_errors = propagate_delivery_errors
        self.delivery_retry_delay_seconds = delivery_retry_delay_seconds
        self._adapters = {
            (adapter.name, adapter.account_id): adapter for adapter in adapters
        }
        if not self._adapters:
            raise ValueError("at least one channel adapter is required")
        self._active_executions: dict[str, asyncio.Task[None]] = {}
        self._stopping = asyncio.Event()
        self._closed = False

    async def accept(
        self,
        adapter: ChannelAdapter,
        envelope: InboundEnvelope,
    ) -> bool:
        """Persist an event before acknowledging it to the transport."""
        expected = (adapter.name, adapter.account_id)
        actual = (envelope.identity.adapter, envelope.identity.account_id)
        if actual != expected:
            raise ValueError("adapter identity does not match the inbound envelope")
        route = self.router.resolve(envelope)
        if route is None:
            pairing_code = _pairing_code(envelope)
            if pairing_code is not None:
                route = self.router.consume_pairing(pairing_code, envelope)
                if route is not None:
                    self.ledger.ignore(
                        envelope,
                        profile_id=route.profile_id,
                        reason="pairing challenge consumed",
                    )
                    await adapter.acknowledge(envelope)
                    return False
        if route is None:
            self.ledger.ignore(
                envelope,
                profile_id=UNROUTED_PROFILE_ID,
                reason="identity is not paired or allowed",
            )
            await adapter.acknowledge(envelope)
            return False
        self.ledger.ingest(
            envelope,
            profile_id=route.profile_id,
            max_pending=self.limits.max_pending,
        )
        await adapter.acknowledge(envelope)
        return True

    async def process_available(self) -> int:
        """Run one bounded wave of eligible executions."""
        tasks: list[asyncio.Task[None]] = []
        while len(tasks) < self.limits.global_concurrency:
            claim = await asyncio.to_thread(
                self.ledger.claim_execution,
                global_limit=self.limits.global_concurrency,
                profile_limit=self.limits.profile_concurrency,
                lease_seconds=self.limits.execution_lease_seconds,
            )
            if claim is None:
                break
            task = asyncio.create_task(self._execute_claim(claim))
            self._active_executions[claim.record.id] = task
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def deliver_available(self) -> int:
        """Deliver all currently due outbox records in deterministic order."""
        delivered = 0
        while True:
            record = await asyncio.to_thread(
                self.ledger.claim_delivery,
                lease_seconds=self.limits.delivery_lease_seconds,
            )
            if record is None:
                return delivered
            adapter = self._adapters.get(
                (record.envelope.target.adapter, record.envelope.target.account_id)
            )
            if adapter is None:
                receipt = DeliveryReceipt(
                    envelope_id=record.id,
                    state=DeliveryState.FAILED,
                    attempt=record.attempt_count,
                    error_code="adapter_unavailable",
                    error_message="configured adapter is unavailable",
                )
            else:
                delivery_error: Exception | None = None
                try:
                    returned = await adapter.deliver(record.envelope)
                except Exception as exc:
                    delivery_error = exc
                    LOGGER.warning(
                        "Gateway delivery failed (%s)",
                        type(exc).__name__,
                    )
                    receipt = DeliveryReceipt(
                        envelope_id=record.id,
                        state=DeliveryState.RETRYABLE,
                        attempt=record.attempt_count,
                        checkpoint=record.checkpoint,
                        retry_after_seconds=(
                            self.delivery_retry_delay_seconds
                            if self.delivery_retry_delay_seconds is not None
                            else min(60.0, 2.0**record.attempt_count)
                        ),
                        error_code=type(exc).__name__,
                        error_message="channel delivery failed",
                    )
                else:
                    receipt = DeliveryReceipt(
                        envelope_id=record.id,
                        state=returned.state,
                        attempt=record.attempt_count,
                        adapter_message_id=returned.adapter_message_id,
                        checkpoint=returned.checkpoint,
                        retry_after_seconds=(
                            returned.retry_after_seconds
                            if returned.retry_after_seconds is not None
                            else (
                                min(60.0, 2.0**record.attempt_count)
                                if returned.state is DeliveryState.RETRYABLE
                                else None
                            )
                        ),
                        error_code=returned.error_code,
                        error_message=returned.error_message,
                        recorded_at=returned.recorded_at,
                        extensions=returned.extensions,
                    )
            assert record.delivery_token is not None
            await asyncio.to_thread(
                self.ledger.record_delivery,
                record.id,
                record.delivery_token,
                receipt,
            )
            delivered += 1
            if (
                adapter is not None
                and delivery_error is not None
                and self.propagate_delivery_errors
            ):
                raise delivery_error

    async def run_once(self) -> tuple[int, int]:
        """Recover leases, execute one worker wave, then drain due deliveries."""
        await asyncio.to_thread(self.ledger.recover_expired_executions)
        executed = await self.process_available()
        delivered = await self.deliver_available()
        return executed, delivered

    async def run_forever(self) -> None:
        """Receive from every adapter while shared workers drain durable state."""
        receivers = [
            asyncio.create_task(self._receive_adapter(adapter))
            for adapter in self._adapters.values()
        ]
        worker = asyncio.create_task(self._worker_loop())
        tasks = (*receivers, worker)
        try:
            done, _pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error
        finally:
            self._stopping.set()
            for task in receivers:
                task.cancel()
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
            await self.close()

    async def cancel(self, inbox_id: str) -> bool:
        """Propagate a durable cancellation request to an active task."""
        requested = await asyncio.to_thread(
            self.ledger.request_cancellation,
            inbox_id,
        )
        task = self._active_executions.get(inbox_id)
        if task is not None:
            task.cancel()
        return requested

    async def close(self) -> None:
        """Stop active work and close each configured adapter once."""
        if self._closed:
            return
        self._closed = True
        self._stopping.set()
        tasks = tuple(self._active_executions.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for adapter in self._adapters.values():
            await adapter.close()

    async def _receive_adapter(self, adapter: ChannelAdapter) -> None:
        async for envelope in adapter.receive():
            if self._stopping.is_set():
                return
            await self.accept(adapter, envelope)

    async def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            executed, delivered = await self.run_once()
            if executed == 0 and delivered == 0:
                await asyncio.sleep(self.limits.idle_delay_seconds)

    async def _execute_claim(self, claim: ExecutionClaim) -> None:
        renewal = asyncio.create_task(self._renew_execution(claim))
        try:
            responses = await self.executor(
                claim.record.profile_id,
                claim.record.envelope,
            )
            completed = await asyncio.to_thread(
                self.ledger.complete_execution,
                claim.record.id,
                claim.execution_token,
                responses,
            )
            if not completed:
                LOGGER.warning("Gateway execution lost its durable claim")
        except asyncio.CancelledError:
            current = await asyncio.to_thread(
                self.ledger.get_inbox,
                claim.record.id,
            )
            if current is not None and current.cancellation_requested:
                await asyncio.to_thread(
                    self.ledger.mark_execution_cancelled,
                    claim.record.id,
                    claim.execution_token,
                )
            else:
                await asyncio.to_thread(
                    self.ledger.quarantine_execution,
                    claim.record.id,
                    claim.execution_token,
                    error="gateway execution cancelled at an uncertain checkpoint",
                )
            raise
        except Exception as exc:
            LOGGER.error("Gateway execution failed (%s)", type(exc).__name__)
            await asyncio.to_thread(
                self.ledger.quarantine_execution,
                claim.record.id,
                claim.execution_token,
                error=f"execution failed at an uncertain checkpoint: {type(exc).__name__}",
            )
        finally:
            renewal.cancel()
            with suppress(asyncio.CancelledError):
                await renewal
            self._active_executions.pop(claim.record.id, None)

    async def _renew_execution(self, claim: ExecutionClaim) -> None:
        interval = max(0.1, self.limits.execution_lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            renewed = await asyncio.to_thread(
                self.ledger.renew_execution,
                claim.record.id,
                claim.execution_token,
                lease_seconds=self.limits.execution_lease_seconds,
            )
            if not renewed:
                return


def _pairing_code(envelope: InboundEnvelope) -> str | None:
    if len(envelope.parts) != 1 or not isinstance(envelope.parts[0], TextPart):
        return None
    code = envelope.parts[0].text.strip()
    return code if code and len(code) <= 128 else None


__all__ = [
    "EnvelopeExecutor",
    "GatewayLimits",
    "GatewayRuntime",
    "UNROUTED_PROFILE_ID",
]
