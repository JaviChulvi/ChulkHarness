"""Close sync and async provider transports without leaking awaitables."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
import inspect
from threading import Thread
from typing import Any


def close_resources(resources: Iterable[object]) -> None:
    """Close distinct resources from synchronous hosts."""
    failures: list[Exception] = []
    for resource in _distinct(resources):
        try:
            close = getattr(resource, "close", None)
            if callable(close) and not _is_default_llm_close(close):
                result = close()
            else:
                aclose = getattr(resource, "aclose", None)
                if not callable(aclose):
                    continue
                result = aclose()
            if inspect.isawaitable(result):
                _run_awaitable(result)
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise ExceptionGroup("Failed to close provider resources", failures)


async def aclose_resources(resources: Iterable[object]) -> None:
    """Close distinct resources from asynchronous hosts."""
    failures: list[Exception] = []
    for resource in _distinct(resources):
        try:
            aclose = getattr(resource, "aclose", None)
            if callable(aclose):
                result = aclose()
            else:
                close = getattr(resource, "close", None)
                if not callable(close):
                    continue
                result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise ExceptionGroup("Failed to close provider resources", failures)


def _distinct(resources: Iterable[object]) -> tuple[object, ...]:
    result: list[object] = []
    identities: set[int] = set()
    for resource in resources:
        if resource is None or id(resource) in identities:
            continue
        identities.add(id(resource))
        result.append(resource)
    return tuple(result)


def _run_awaitable(awaitable: Any) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(awaitable)
        return

    failure: list[BaseException] = []

    def run() -> None:
        try:
            asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - defensive cross-thread propagation
            failure.append(exc)

    thread = Thread(target=run, name="chulk-provider-close")
    thread.start()
    thread.join()
    if failure:
        raise failure[0]


def _is_default_llm_close(close: Any) -> bool:
    from chulk.llm.base import LLMClient

    return getattr(close, "__func__", None) is LLMClient.close


__all__ = ["aclose_resources", "close_resources"]
