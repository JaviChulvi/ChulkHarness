"""Native async hosted-service invocation helpers."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any


async def call_async_service(
    service: object,
    method_name: str,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Await a native async method or isolate an explicit sync binding.

    Native coroutine methods always execute on the caller's event loop. Sync
    services remain supported as an explicit compatibility boundary and are
    never invoked on that loop.
    """

    method = getattr(service, method_name)
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    invocation = asyncio.create_task(
        asyncio.to_thread(method, *args, **kwargs)
    )
    try:
        result = await asyncio.shield(invocation)
    except asyncio.CancelledError as cancellation:
        await _finish_cancelled_sync_call(invocation, cancellation)
        raise
    if inspect.isawaitable(result):
        return await result
    return result


async def _finish_cancelled_sync_call(
    invocation: asyncio.Task[Any],
    cancellation: asyncio.CancelledError,
) -> None:
    """Wait for an in-flight sync call without replacing cancellation."""

    try:
        await asyncio.shield(invocation)
    except asyncio.CancelledError:
        try:
            await invocation
        except BaseException as error:
            cancellation.add_note(
                "cancelled sync service call also failed with "
                f"{type(error).__name__}: {error}"
            )
    except BaseException as error:
        cancellation.add_note(
            "cancelled sync service call also failed with "
            f"{type(error).__name__}: {error}"
        )


async def close_async_resource(resource: object) -> None:
    """Close one resource through its native async lifecycle when available."""

    aclose = getattr(resource, "aclose", None)
    if callable(aclose):
        result = aclose()
        if inspect.isawaitable(result):
            await result
        return
    close = getattr(resource, "close", None)
    if callable(close):
        await call_async_service(resource, "close")


__all__ = ["call_async_service", "close_async_resource"]
