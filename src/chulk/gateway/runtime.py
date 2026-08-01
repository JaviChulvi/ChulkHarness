"""Shared channel ingestion, execution, and delivery orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import logging
from typing import Any

from chulk.events import AgentEvent, DeliveryPayload, EventName
from chulk.gateway.commands import parse_channel_command
from chulk.gateway.ledger import ExecutionClaim
from chulk.gateway.models import (
    DeliveryReceipt,
    DeliveryState,
    InboundEnvelope,
    OutboundEnvelope,
    TextPart,
)
from chulk.gateway.protocol import ChannelAdapter, DeliveryReconciler
from chulk.gateway.stores import (
    AsyncGatewayRouter,
    AsyncGatewayRunSubmitter,
    AsyncGatewayScopeResolver,
    AsyncGatewayStore,
    GatewayRouter,
    GatewayRunSubmitter,
    GatewayRunTarget,
    GatewayScopeResolver,
    GatewayStore,
)


LOGGER = logging.getLogger(__name__)
UNROUTED_PROFILE_ID = "_gateway"
EnvelopeExecutor = Callable[
    [str, InboundEnvelope],
    Awaitable[tuple[OutboundEnvelope, ...]],
]


class GatewayPoisonEventError(ValueError):
    """Raised before side effects when an input cannot become a valid run."""


@dataclass(frozen=True, slots=True)
class GatewayLimits:
    """Bounded shared gateway queue and worker settings."""

    global_concurrency: int = 4
    profile_concurrency: int = 1
    max_pending: int = 1_000
    execution_lease_seconds: int = 300
    delivery_lease_seconds: int = 120
    max_delivery_attempts: int = 5
    idle_delay_seconds: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "global_concurrency",
            "profile_concurrency",
            "max_pending",
            "execution_lease_seconds",
            "delivery_lease_seconds",
            "max_delivery_attempts",
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
        ledger: GatewayStore | AsyncGatewayStore,
        router: GatewayRouter | AsyncGatewayRouter,
        adapters: Iterable[ChannelAdapter],
        executor: EnvelopeExecutor,
        limits: GatewayLimits | None = None,
        propagate_delivery_errors: bool = False,
        delivery_retry_delay_seconds: float | None = None,
        scope_resolver: GatewayScopeResolver | AsyncGatewayScopeResolver | None = None,
        run_submitter: GatewayRunSubmitter | AsyncGatewayRunSubmitter | None = None,
        event_sink: object | None = None,
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
        if (scope_resolver is None) != (run_submitter is None):
            raise ValueError(
                "hosted gateway requires both scope_resolver and run_submitter"
            )
        self.scope_resolver = scope_resolver
        self.run_submitter = run_submitter
        self.event_sink = event_sink
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
        route = await _service_call(self.router, "resolve", envelope)
        if route is None:
            pairing_code = _pairing_code(envelope)
            if pairing_code is not None:
                route = await _service_call(
                    self.router,
                    "consume_pairing",
                    pairing_code,
                    envelope,
                )
                if route is not None:
                    await _service_call(
                        self.ledger,
                        "ignore",
                        envelope,
                        profile_id=route.profile_id,
                        reason="pairing challenge consumed",
                    )
                    await adapter.acknowledge(envelope)
                    return False
        if route is None:
            await _service_call(
                self.ledger,
                "ignore",
                envelope,
                profile_id=UNROUTED_PROFILE_ID,
                reason="identity is not paired or allowed",
            )
            await adapter.acknowledge(envelope)
            return False
        target: GatewayRunTarget | None = None
        if self.scope_resolver is not None:
            try:
                target = await _service_call(
                    self.scope_resolver,
                    "resolve",
                    route,
                    envelope,
                )
                if not isinstance(target, GatewayRunTarget):
                    raise GatewayPoisonEventError(
                        "gateway scope resolver returned an invalid run target"
                    )
            except (GatewayPoisonEventError, ValueError) as exc:
                poisoned = await _service_call(
                    self.ledger,
                    "ingest",
                    envelope,
                    profile_id=route.profile_id,
                    max_pending=self.limits.max_pending,
                )
                await _service_call(
                    self.ledger,
                    "dead_letter_inbox",
                    poisoned.record.id,
                    error=str(exc),
                )
                await adapter.acknowledge(envelope)
                return False
        ingest_options: dict[str, Any] = {
            "profile_id": route.profile_id,
            "max_pending": self.limits.max_pending,
        }
        if target is not None:
            ingest_options["run_target"] = target
        ingested = await _service_call(
            self.ledger,
            "ingest",
            envelope,
            **ingest_options,
        )
        if target is not None:
            assert self.run_submitter is not None
            try:
                run_id = await _service_call(
                    self.run_submitter,
                    "submit",
                    target,
                    envelope,
                    idempotency_key=envelope.idempotency_key,
                )
                if run_id != target.scope.run_id:
                    raise GatewayPoisonEventError(
                        "gateway submitter returned a different durable run id"
                    )
            except (GatewayPoisonEventError, ValueError) as exc:
                await _service_call(
                    self.ledger,
                    "dead_letter_inbox",
                    ingested.record.id,
                    error=str(exc),
                )
                await adapter.acknowledge(envelope)
                return False
        if ingested.created and _is_stop_command(envelope):
            cancelled_ids = await _service_call(
                self.ledger,
                "request_conversation_cancellation",
                profile_id=route.profile_id,
                conversation_key=ingested.record.conversation_key,
                exclude_inbox_id=ingested.record.id,
            )
            for inbox_id in cancelled_ids:
                task = self._active_executions.get(inbox_id)
                if task is not None:
                    task.cancel()
        await adapter.acknowledge(envelope)
        return True

    async def process_available(self) -> int:
        """Run one bounded wave of eligible executions."""
        queued_before = datetime.now(timezone.utc)
        tasks: list[asyncio.Task[None]] = []
        while len(tasks) < self.limits.global_concurrency:
            claim = await _service_call(
                self.ledger,
                "claim_execution",
                global_limit=self.limits.global_concurrency,
                profile_limit=self.limits.profile_concurrency,
                adapter_keys=tuple(self._adapters),
                queued_before=queued_before,
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
            record = await _service_call(
                self.ledger,
                "claim_delivery",
                adapter_keys=tuple(self._adapters),
                lease_seconds=self.limits.delivery_lease_seconds,
            )
            if record is None:
                return delivered
            await self._publish_delivery(
                record,
                EventName.DELIVERY_STARTED,
                status="delivering",
                action="claim",
            )
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
            if (
                receipt.state is DeliveryState.RETRYABLE
                and record.attempt_count >= self.limits.max_delivery_attempts
            ):
                receipt = DeliveryReceipt(
                    envelope_id=record.id,
                    state=DeliveryState.DEAD_LETTER,
                    attempt=record.attempt_count,
                    checkpoint=receipt.checkpoint,
                    error_code=receipt.error_code or "delivery_attempts_exhausted",
                    error_message=(
                        receipt.error_message
                        or "delivery attempts exhausted"
                    ),
                )
            assert record.delivery_token is not None
            await _service_call(
                self.ledger,
                "record_delivery",
                record.id,
                record.delivery_token,
                receipt,
            )
            if receipt.state is DeliveryState.DELIVERED:
                event_name = EventName.DELIVERY_COMPLETED
                status = "delivered"
            elif receipt.state in {
                DeliveryState.ACCEPTED,
                DeliveryState.UNKNOWN,
            }:
                event_name = EventName.DELIVERY_UNKNOWN
                status = "unknown"
            elif receipt.state is DeliveryState.DEAD_LETTER:
                event_name = EventName.DELIVERY_DEAD_LETTERED
                status = "dead_letter"
            else:
                event_name = EventName.DELIVERY_FAILED
                status = receipt.state.value
            await self._publish_delivery(
                record,
                event_name,
                status=status,
                action="record",
                reason=receipt.error_message or receipt.error_code,
            )
            delivered += 1
            if (
                adapter is not None
                and delivery_error is not None
                and self.propagate_delivery_errors
            ):
                raise delivery_error

    async def reconcile_available(self, *, limit: int = 100) -> int:
        """Resolve ambiguous provider outcomes without resending them."""
        records = await _service_call(
            self.ledger,
            "list_reconciliation_required",
            limit=limit,
        )
        reconciled = 0
        for record in records:
            adapter = self._adapters.get(
                (record.envelope.target.adapter, record.envelope.target.account_id)
            )
            if adapter is None or not isinstance(adapter, DeliveryReconciler):
                continue
            receipt = await adapter.reconcile(
                record.envelope,
                checkpoint=record.checkpoint,
                attempt=record.attempt_count,
            )
            if receipt.state in {
                DeliveryState.ACCEPTED,
                DeliveryState.UNKNOWN,
            }:
                continue
            changed = await _service_call(
                self.ledger,
                "reconcile_delivery",
                record.id,
                receipt,
            )
            if changed:
                await self._publish_delivery(
                    record,
                    EventName.DELIVERY_RECONCILED,
                    status=receipt.state.value,
                    action="reconcile",
                    reason=receipt.error_message or receipt.error_code,
                )
            reconciled += int(bool(changed))
        return reconciled

    async def run_once(self) -> tuple[int, int]:
        """Recover leases, execute one worker wave, then drain due deliveries."""
        await _service_call(self.ledger, "recover_expired_executions")
        await self.reconcile_available()
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
        requested = await _service_call(
            self.ledger,
            "request_cancellation",
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
            completed = await _service_call(
                self.ledger,
                "complete_execution",
                claim.record.id,
                claim.execution_token,
                responses,
            )
            if not completed:
                LOGGER.warning("Gateway execution lost its durable claim")
        except asyncio.CancelledError:
            current = await _service_call(
                self.ledger,
                "get_inbox",
                claim.record.id,
            )
            if current is not None and current.cancellation_requested:
                await _service_call(
                    self.ledger,
                    "mark_execution_cancelled",
                    claim.record.id,
                    claim.execution_token,
                )
            else:
                await _service_call(
                    self.ledger,
                    "quarantine_execution",
                    claim.record.id,
                    claim.execution_token,
                    error="gateway execution cancelled at an uncertain checkpoint",
                )
            raise
        except GatewayPoisonEventError as exc:
            await _service_call(
                self.ledger,
                "dead_letter_execution",
                claim.record.id,
                claim.execution_token,
                error=str(exc),
            )
        except Exception as exc:
            LOGGER.error("Gateway execution failed (%s)", type(exc).__name__)
            await _service_call(
                self.ledger,
                "quarantine_execution",
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
            renewed = await _service_call(
                self.ledger,
                "renew_execution",
                claim.record.id,
                claim.execution_token,
                lease_seconds=self.limits.execution_lease_seconds,
            )
            if not renewed:
                return

    async def _publish_delivery(
        self,
        record: Any,
        name: EventName,
        *,
        status: str,
        action: str,
        reason: str | None = None,
    ) -> None:
        if self.event_sink is None:
            return
        inbox = await _service_call(
            self.ledger,
            "get_inbox",
            record.inbox_id,
        )
        target = inbox.run_target if inbox is not None else None
        if target is None:
            return
        scope = target.scope
        event = AgentEvent(
            name=name.value,
            conversation_id=scope.conversation_id or scope.run_id,
            profile_id=record.profile_id,
            execution_scope=scope,
            run_id=scope.run_id,
            correlation_id=scope.run_id,
            source_event_id=inbox.envelope.event_id,
            idempotency_key=(
                f"delivery:{record.id}:{record.attempt_count}:{status}"
            ),
            payload=DeliveryPayload(
                delivery_id=record.id,
                status=status,
                action=action,
                target=(
                    f"{record.envelope.target.adapter}:"
                    f"{record.envelope.target.account_id}:"
                    f"{record.envelope.target.destination_id}"
                ),
                reason=reason,
                extensions={
                    "attempt": record.attempt_count,
                    "checkpoint": record.checkpoint,
                },
            ),
        )
        await _service_call(self.event_sink, "emit", event)


async def _service_call(
    service: object,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Await native async stores and isolate sync reference adapters in a thread."""
    method = getattr(service, method_name)
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    result = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _pairing_code(envelope: InboundEnvelope) -> str | None:
    if len(envelope.parts) != 1 or not isinstance(envelope.parts[0], TextPart):
        return None
    code = envelope.parts[0].text.strip()
    return code if code and len(code) <= 128 else None


def _is_stop_command(envelope: InboundEnvelope) -> bool:
    if len(envelope.parts) != 1 or not isinstance(envelope.parts[0], TextPart):
        return False
    parsed = parse_channel_command(envelope.parts[0].text)
    return parsed is not None and parsed.name == "stop"


__all__ = [
    "EnvelopeExecutor",
    "GatewayLimits",
    "GatewayPoisonEventError",
    "GatewayRuntime",
    "UNROUTED_PROFILE_ID",
]
