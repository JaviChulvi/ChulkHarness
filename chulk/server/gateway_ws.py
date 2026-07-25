"""Typed WebSocket adapter hosted on the shared durable gateway."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DeliveryReceipt,
    DeliveryState,
    DeliveryTarget,
    GatewayLimits,
    GatewayRuntime,
    InboundEnvelope,
    OutboundEnvelope,
    SQLiteGatewayLedger,
    SQLiteGatewayRouter,
    TextPart,
    TrustLevel,
)
from chulk.profiles import ProfileNotFoundError
from chulk.server.dispatcher import ConversationDispatcher


GATEWAY_PROTOCOL_VERSION = 1
GATEWAY_ADAPTER_NAME = "websocket"


@dataclass(frozen=True, slots=True)
class GatewayHello:
    profile_id: str
    client_id: str

    @classmethod
    def from_dict(cls, value: object) -> GatewayHello:
        body = _message(value, "hello")
        return cls(
            profile_id=_required(body.get("profile_id"), "profile_id", 64),
            client_id=_required(body.get("client_id"), "client_id", 128),
        )


@dataclass(frozen=True, slots=True)
class GatewayMessage:
    event_id: str
    idempotency_key: str
    message: str
    conversation_id: str | None = None
    mode: str = "run"

    @classmethod
    def from_dict(cls, value: object) -> GatewayMessage:
        body = _message(value, "message")
        mode = body.get("mode", "run")
        if mode not in {"run", "plan"}:
            raise ValueError("mode must be 'run' or 'plan'")
        return cls(
            event_id=_required(body.get("event_id"), "event_id", 128),
            idempotency_key=_required(
                body.get("idempotency_key"),
                "idempotency_key",
                256,
            ),
            message=_required(body.get("message"), "message", 100_000),
            conversation_id=_optional(
                body.get("conversation_id"),
                "conversation_id",
                128,
            ),
            mode=mode,
        )


class WebSocketChannelAdapter:
    """One authenticated owner WebSocket as a normal channel adapter."""

    name = GATEWAY_ADAPTER_NAME

    def __init__(self, websocket, *, account_id: str, ledger: SQLiteGatewayLedger) -> None:
        self.websocket = websocket
        self.account_id = account_id
        self.ledger = ledger
        self._send_lock = asyncio.Lock()
        self.closed = False

    async def receive(self):
        if False:
            yield

    async def deliver(self, envelope: OutboundEnvelope) -> DeliveryReceipt:
        await self.send(
            {
                "type": "message.completed",
                "schema_version": GATEWAY_PROTOCOL_VERSION,
                "envelope_id": envelope.envelope_id,
                "conversation_id": envelope.conversation_id,
                "text": envelope.text,
                "reply_to_event_id": envelope.reply_to_event_id,
                "sequence": envelope.sequence,
                "final": envelope.final,
            }
        )
        return DeliveryReceipt(
            envelope_id=envelope.envelope_id,
            state=DeliveryState.DELIVERED,
            attempt=1,
        )

    async def acknowledge(self, envelope: InboundEnvelope) -> None:
        record = self.ledger.find_inbox(
            adapter=self.name,
            account_id=self.account_id,
            idempotency_key=envelope.idempotency_key,
        )
        await self.send(
            {
                "type": "message.accepted",
                "schema_version": GATEWAY_PROTOCOL_VERSION,
                "event_id": envelope.event_id,
                "inbox_id": record.id if record is not None else None,
                "created": bool(record and record.state == "queued"),
            }
        )

    async def send(self, value: Mapping[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_json(dict(value))

    async def close(self) -> None:
        self.closed = True


async def serve_gateway_websocket(
    websocket,
    *,
    dispatcher: ConversationDispatcher,
) -> None:
    """Serve one socket through the shared router, ledger, and dispatcher."""
    try:
        await websocket.accept()
        raw_hello = await websocket.receive_json()
        hello = GatewayHello.from_dict(raw_hello)
        dispatcher.runtime_factory.resolve(hello.profile_id)
    except ProfileNotFoundError:
        await _close_with_error(websocket, "profile_not_found", 4404)
        return
    except (ValueError, TypeError):
        await _close_with_error(websocket, "invalid_hello", 4400)
        return

    account_id = uuid4().hex
    control_path = dispatcher.runtime_factory.base_config.runtime_dir / "control.sqlite"
    ledger = SQLiteGatewayLedger(control_path)
    router = SQLiteGatewayRouter(control_path)
    route = router.add_route(
        adapter=GATEWAY_ADAPTER_NAME,
        account_id=account_id,
        principal_id="local-owner",
        destination_id=hello.client_id,
        profile_id=hello.profile_id,
    )
    adapter = WebSocketChannelAdapter(
        websocket,
        account_id=account_id,
        ledger=ledger,
    )

    async def execute(
        profile_id: str,
        envelope: InboundEnvelope,
    ) -> tuple[OutboundEnvelope, ...]:
        text = "\n".join(
            part.text for part in envelope.parts if isinstance(part, TextPart)
        )
        requested = envelope.extensions.get("conversation_id")
        conversation_id = (
            str(requested)
            if isinstance(requested, str) and requested
            else (
                await dispatcher.create_conversation(
                    profile_id,
                    metadata={"channel": GATEWAY_ADAPTER_NAME},
                )
            )["id"]
        )
        mode = envelope.extensions.get("mode", "run")
        command = await dispatcher.submit_and_wait(
            profile_id,
            conversation_id,
            text,
            mode="plan" if mode == "plan" else "run",
            source="gateway",
            idempotency_key=envelope.idempotency_key,
        )
        result = command.result or {}
        answer = result.get("content")
        if not isinstance(answer, str):
            answer = command.error or f"Command ended with status {command.status}."
        return (
            OutboundEnvelope(
                profile_id=profile_id,
                conversation_id=conversation_id,
                target=DeliveryTarget(
                    GATEWAY_ADAPTER_NAME,
                    account_id,
                    envelope.destination_id,
                ),
                text=answer,
                reply_to_event_id=envelope.event_id,
                extensions={"command_id": command.id, "status": command.status},
            ),
        )

    runtime = GatewayRuntime(
        ledger=ledger,
        router=router,
        adapters=(adapter,),
        executor=execute,
        limits=GatewayLimits(global_concurrency=4, profile_concurrency=1),
    )
    processing: set[asyncio.Task[tuple[int, int]]] = set()
    await adapter.send(
        {
            "type": "ready",
            "schema_version": GATEWAY_PROTOCOL_VERSION,
            "profile_id": hello.profile_id,
            "client_id": hello.client_id,
        }
    )
    try:
        while True:
            value = await websocket.receive_json()
            if not isinstance(value, Mapping):
                raise ValueError("gateway frame must be an object")
            message_type = value.get("type")
            if message_type == "message":
                incoming = GatewayMessage.from_dict(value)
                envelope = InboundEnvelope(
                    event_id=incoming.event_id,
                    idempotency_key=incoming.idempotency_key,
                    identity=ChannelIdentity(
                        GATEWAY_ADAPTER_NAME,
                        account_id,
                        "local-owner",
                    ),
                    destination_id=hello.client_id,
                    parts=(TextPart(incoming.message),),
                    scope=ChannelScope.LOCAL_OPERATOR,
                    authentication=AuthenticationState.AUTHENTICATED,
                    trust=TrustLevel.OWNER,
                    thread_id=incoming.conversation_id,
                    extensions={
                        "conversation_id": incoming.conversation_id,
                        "mode": incoming.mode,
                    },
                )
                await runtime.accept(adapter, envelope)
                task = asyncio.create_task(runtime.run_once())
                processing.add(task)
                task.add_done_callback(processing.discard)
            elif message_type == "cancel":
                inbox_id = _optional(value.get("inbox_id"), "inbox_id", 128)
                conversation_id = _optional(
                    value.get("conversation_id"),
                    "conversation_id",
                    128,
                )
                cancelled = False
                if inbox_id is not None:
                    cancelled = await runtime.cancel(inbox_id)
                if conversation_id is not None:
                    cancelled = (
                        await dispatcher.cancel(hello.profile_id, conversation_id)
                        or cancelled
                    )
                await adapter.send(
                    {
                        "type": "cancel.accepted",
                        "schema_version": GATEWAY_PROTOCOL_VERSION,
                        "cancelled": cancelled,
                    }
                )
            elif message_type == "ping":
                await adapter.send(
                    {
                        "type": "pong",
                        "schema_version": GATEWAY_PROTOCOL_VERSION,
                    }
                )
            else:
                raise ValueError("unknown gateway frame type")
    except Exception as exc:
        if type(exc).__name__ != "WebSocketDisconnect":
            await _send_error(adapter, "invalid_frame")
    finally:
        router.remove_route(route.id)
        await runtime.close()
        if processing:
            await asyncio.gather(*processing, return_exceptions=True)


async def _send_error(adapter: WebSocketChannelAdapter, code: str) -> None:
    try:
        await adapter.send(
            {
                "type": "error",
                "schema_version": GATEWAY_PROTOCOL_VERSION,
                "error": {"code": code, "message": code.replace("_", " ")},
            }
        )
    except Exception:
        return


async def _close_with_error(websocket, code: str, close_code: int) -> None:
    try:
        await websocket.send_json(
            {
                "type": "error",
                "schema_version": GATEWAY_PROTOCOL_VERSION,
                "error": {"code": code, "message": code.replace("_", " ")},
            }
        )
    finally:
        await websocket.close(code=close_code)


def _message(value: object, expected_type: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("gateway frame must be an object")
    if value.get("type") != expected_type:
        raise ValueError(f"gateway frame type must be {expected_type!r}")
    version = value.get("schema_version", GATEWAY_PROTOCOL_VERSION)
    if version != GATEWAY_PROTOCOL_VERSION:
        raise ValueError("unsupported gateway schema_version")
    return value


def _required(value: object, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized or "\x00" in normalized:
        raise ValueError(f"{field} is required")
    if len(normalized) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    return normalized


def _optional(value: object, field: str, max_chars: int) -> str | None:
    return None if value is None else _required(value, field, max_chars)


__all__ = [
    "GATEWAY_ADAPTER_NAME",
    "GATEWAY_PROTOCOL_VERSION",
    "GatewayHello",
    "GatewayMessage",
    "WebSocketChannelAdapter",
    "serve_gateway_websocket",
]
