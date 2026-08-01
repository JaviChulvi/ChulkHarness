"""Hosted synchronous and asynchronous SDK runtimes."""

from __future__ import annotations

from collections.abc import Awaitable
import inspect
from typing import Any, Callable, TypeVar

from chulk._sdk.agent import Agent
from chulk._sdk.async_agent import AsyncAgent
from chulk.hosting import (
    AsyncRuntimeServices,
    AsyncServiceBinding,
    ExecutionScope,
    RuntimeServices,
)
from chulk.hosting.services import ResolvedRuntimeServices


T = TypeVar("T")


class HostedRuntime(Agent):
    """Synchronous SDK facade that requires a complete hosted boundary."""

    def __init__(
        self,
        *,
        services: RuntimeServices,
        execution_scope: ExecutionScope,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            services=services,
            execution_scope=execution_scope,
            **kwargs,
        )


class AsyncHostedRuntime(AsyncAgent):
    """Asynchronous SDK facade that requires async-hosted service contracts."""

    def __init__(
        self,
        *,
        services: AsyncRuntimeServices,
        execution_scope: ExecutionScope,
        **kwargs: Any,
    ) -> None:
        if any(
            isinstance(getattr(services, name), AsyncServiceBinding)
            for name in services.__dataclass_fields__
        ):
            raise ValueError(
                "native async service bindings require "
                "await AsyncHostedRuntime.create(...)"
            )
        sync_boundary = services.as_sync_services()
        super().__init__(
            services=sync_boundary,
            execution_scope=execution_scope,
            **kwargs,
        )
        from chulk.hosting.sinks import BufferedAsyncEventSink

        sink = self.runtime.public_event_sink
        event_buffer = BufferedAsyncEventSink(sink)
        self.runtime.public_event_sink = event_buffer
        self.runtime.async_event_buffer = event_buffer
        self._async_owned_services: ResolvedRuntimeServices | None = None
        self._async_host_flushables: tuple[object, ...] = ()

    @classmethod
    async def create(
        cls,
        *,
        services: AsyncRuntimeServices,
        execution_scope: ExecutionScope,
        **kwargs: Any,
    ) -> "AsyncHostedRuntime":
        """Resolve async factories without blocking, then construct the runtime."""
        resolved = await services.resolve_async(execution_scope)
        bindings = resolved.host_bindings()
        flushables: list[object] = []
        fields = {
            name: getattr(bindings, name)
            for name in bindings.__dataclass_fields__
        }
        from chulk.hosting.sinks import (
            BufferedAsyncAuditSink,
            BufferedAsyncTraceSink,
        )

        if _has_async_method(resolved.traces, "log"):
            trace_sink = BufferedAsyncTraceSink(resolved.traces)
            fields["traces"] = type(bindings.traces).host(trace_sink)
            flushables.append(trace_sink)
        if _has_async_method(resolved.audit, "record"):
            audit_sink = BufferedAsyncAuditSink(resolved.audit)
            fields["audit"] = type(bindings.audit).host(audit_sink)
            flushables.append(audit_sink)
        compatibility = AsyncRuntimeServices(
            **fields
        )
        try:
            runtime = cls(
                services=compatibility,
                execution_scope=execution_scope,
                **kwargs,
            )
        except BaseException:
            await resolved.aclose_owned()
            raise
        runtime._async_owned_services = resolved
        runtime._async_host_flushables = tuple(flushables)
        return runtime

    async def _invoke_async(
        self,
        operation: str,
        call: Callable[[], Awaitable[T]],
        *,
        serialized: bool = False,
    ) -> T:
        try:
            return await super()._invoke_async(
                operation,
                call,
                serialized=serialized,
            )
        finally:
            for flushable in self._async_host_flushables:
                flush = getattr(flushable, "flush")
                await flush()

    async def close(self) -> None:
        owned = self._async_owned_services
        try:
            await super().close()
        finally:
            if owned is not None:
                self._async_owned_services = None
                await owned.aclose_owned()


def _has_async_method(value: object, name: str) -> bool:
    return inspect.iscoroutinefunction(getattr(value, name, None))
