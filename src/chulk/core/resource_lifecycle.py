"""Artifact and async journal lifecycle ownership."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from chulk.core.async_cleanup import await_cleanup_after_error
from chulk.hosting.async_utils import call_async_service


class RuntimeResourceLifecycle:
    """Write trace artifacts and flush resolved async services."""

    def __init__(
        self,
        *,
        trace_logger: Any = None,
        async_artifact_store: Any = None,
        async_flushables: tuple[object, ...] = (),
    ) -> None:
        self.trace_logger = trace_logger
        self.async_artifact_store = async_artifact_store
        self.async_flushables = async_flushables

    def write_artifact(self, name: str, content: str) -> dict | None:
        if self.trace_logger is None:
            return None
        return cast(dict | None, self.trace_logger.write_artifact(name, content))

    async def write_artifact_async(self, name: str, content: str) -> dict | None:
        if self.async_artifact_store is None:
            return await asyncio.to_thread(self.write_artifact, name, content)
        record = await call_async_service(
            self.async_artifact_store,
            "write",
            name,
            content,
        )
        if record is None or isinstance(record, dict):
            return record
        for method_name in ("reference", "to_dict"):
            method = getattr(record, method_name, None)
            if callable(method):
                return cast(dict, method())
        raise TypeError("async artifact store returned an unsupported record")

    async def flush(self) -> None:
        failure: BaseException | None = None
        for service in self.async_flushables:
            try:
                await call_async_service(service, "flush")
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        "async journal flush also failed with "
                        f"{type(exc).__name__}: {exc}"
                    )
        if failure is not None:
            raise failure

    async def flush_after_error(self, original: BaseException) -> None:
        await await_cleanup_after_error(self.flush(), original)
