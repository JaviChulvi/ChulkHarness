"""Owner-local authentication, browser checks, and bounded request policy."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import secrets
from typing import Any


DEFAULT_ALLOWED_ORIGINS = (
    "http://127.0.0.1",
    "http://localhost",
)
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class RequestBodyTooLargeError(ValueError):
    """Raised while streaming a body beyond the configured maximum."""


class ControlTokenStore:
    """Generate and rotate one owner-readable local control credential."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()

    def load_or_create(self) -> str:
        if self.path.exists():
            token = self.path.read_text(encoding="utf-8").strip()
            if not token:
                return self.rotate()
            self._secure_permissions()
            return token
        return self.rotate()

    def rotate(self) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        token = secrets.token_urlsafe(48)
        temporary = self.path.with_name(f".{self.path.name}.{secrets.token_hex(8)}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"{token}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._secure_permissions()
        finally:
            if temporary.exists():
                temporary.unlink()
        return token

    def matches(self, candidate: str) -> bool:
        return hmac.compare_digest(self.load_or_create(), candidate)

    def csrf_token(self) -> str:
        return sha256(f"chulk-csrf:{self.load_or_create()}".encode()).hexdigest()

    def _secure_permissions(self) -> None:
        try:
            self.path.chmod(0o600)
        except OSError:
            pass


class SlidingWindowRateLimiter:
    """Small in-process limiter for an owner-local API."""

    def __init__(self, *, requests: int = 120, window_seconds: float = 60.0) -> None:
        if requests < 1 or window_seconds <= 0:
            raise ValueError("rate limit values must be positive")
        self.requests = requests
        self.window_seconds = window_seconds
        self._entries: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        loop = asyncio.get_running_loop()
        now = loop.time()
        threshold = now - self.window_seconds
        async with self._lock:
            entries = self._entries[key]
            while entries and entries[0] <= threshold:
                entries.popleft()
            if len(entries) >= self.requests:
                return False
            entries.append(now)
            return True


class ControlAuditLog:
    """Write structured request metadata without prompts or credentials."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        method: str,
        path: str,
        status: int,
        client: str | None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "path": path,
            "status": status,
            "client": client,
        }
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


class ControlSecurityMiddleware:
    """Authenticate HTTP/WS traffic before application routing."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        token_store: ControlTokenStore,
        allowed_origins: Iterable[str] = DEFAULT_ALLOWED_ORIGINS,
        max_body_bytes: int = 1_000_000,
        max_concurrent_requests: int = 32,
        request_timeout_seconds: float = 30.0,
        rate_limiter: SlidingWindowRateLimiter | None = None,
        audit_log: ControlAuditLog | None = None,
    ) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be greater than zero")
        if max_concurrent_requests < 1 or request_timeout_seconds <= 0:
            raise ValueError("request execution limits must be positive")
        self.app = app
        self.token_store = token_store
        self.allowed_origins = tuple(origin.rstrip("/") for origin in allowed_origins)
        self.max_body_bytes = max_body_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self._request_slots = asyncio.Semaphore(max_concurrent_requests)
        self.rate_limiter = rate_limiter or SlidingWindowRateLimiter()
        self.audit_log = audit_log

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        origin = headers.get("origin")
        if origin is not None and not self._allowed_origin(origin):
            await self._reject(scope, receive, send, 403, "origin_not_allowed")
            return
        token = _control_token(scope, headers)
        if token is None or not self.token_store.matches(token):
            await self._reject(scope, receive, send, 401, "authentication_required")
            return
        if (
            scope["type"] == "http"
            and scope.get("method") in UNSAFE_METHODS
            and origin is not None
            and not hmac.compare_digest(
                headers.get("x-chulk-csrf", ""),
                self.token_store.csrf_token(),
            )
        ):
            await self._reject(scope, receive, send, 403, "csrf_check_failed")
            return
        client = scope.get("client")
        client_name = str(client[0]) if client else "local"
        if not await self.rate_limiter.allow(client_name):
            await self._reject(scope, receive, send, 429, "rate_limit_exceeded")
            return
        if scope["type"] == "http":
            content_length = headers.get("content-length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError:
                    await self._reject(scope, receive, send, 400, "invalid_content_length")
                    return
                if declared > self.max_body_bytes:
                    await self._reject(scope, receive, send, 413, "request_too_large")
                    return
            receive = self._bounded_receive(receive)
        status = 101 if scope["type"] == "websocket" else 500

        async def audit_send(message: dict[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            elif message["type"] == "websocket.accept":
                status = 101
            elif message["type"] == "websocket.close" and status != 101:
                status = int(message.get("code", 1000))
            await send(message)

        try:
            if (
                scope["type"] == "http"
                and not str(scope.get("path", "")).endswith("/events")
            ):
                try:
                    await asyncio.wait_for(
                        self._call_with_slot(scope, receive, audit_send),
                        timeout=self.request_timeout_seconds,
                    )
                except TimeoutError:
                    if status == 500:
                        await self._reject(
                            scope,
                            receive,
                            audit_send,
                            504,
                            "request_timeout",
                        )
            else:
                await self.app(scope, receive, audit_send)
        finally:
            if self.audit_log is not None:
                await asyncio.to_thread(
                    self.audit_log.write,
                    method=scope.get("method", "WEBSOCKET"),
                    path=scope.get("path", ""),
                    status=status,
                    client=client_name,
                )

    async def _call_with_slot(self, scope, receive, send) -> None:
        async with self._request_slots:
            await self.app(scope, receive, send)

    def _bounded_receive(self, receive):
        received = 0

        async def bounded():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise RequestBodyTooLargeError(
                        "request body exceeds configured limit"
                    )
            return message

        return bounded

    def _allowed_origin(self, origin: str) -> bool:
        normalized = origin.rstrip("/")
        return any(
            normalized == allowed
            or normalized.startswith(f"{allowed}:")
            for allowed in self.allowed_origins
        )

    async def _reject(self, scope, receive, send, status: int, code: str) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4400 + status % 100})
            return
        body = json.dumps(
            {
                "schema_version": 1,
                "error": {"code": code, "message": code.replace("_", " ")},
            },
            separators=(",", ":"),
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _control_token(scope: Mapping[str, Any], headers: Mapping[str, str]) -> str | None:
    authorization = headers.get("authorization", "")
    scheme, separator, value = authorization.partition(" ")
    if separator and scheme.lower() == "bearer" and value:
        return value
    if scope["type"] == "websocket":
        for protocol in headers.get("sec-websocket-protocol", "").split(","):
            candidate = protocol.strip()
            if candidate.startswith("chulk.control."):
                return candidate.removeprefix("chulk.control.")
    return None


__all__ = [
    "ControlAuditLog",
    "ControlSecurityMiddleware",
    "ControlTokenStore",
    "DEFAULT_ALLOWED_ORIGINS",
    "RequestBodyTooLargeError",
    "SlidingWindowRateLimiter",
]
