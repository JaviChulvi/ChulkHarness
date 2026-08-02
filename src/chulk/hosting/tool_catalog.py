"""Request-scoped immutable tool catalogs for hosted turns."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import inspect
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

from chulk.hosting.scope import ExecutionScope
from chulk.errors import ChulkError

if TYPE_CHECKING:
    from chulk.tools.registry import Tool, ToolRegistry


class ToolCatalogResolutionError(ChulkError):
    """A hosted catalog could not be resolved safely before model work."""


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(deepcopy(dict(value)))


@dataclass(frozen=True, slots=True)
class ToolCatalogRequest:
    """Bounded immutable metadata supplied to a hosted catalog resolver."""

    scope: ExecutionScope
    conversation_id: str
    turn_id: str
    user_message_digest: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope.conversation_id != self.conversation_id:
            raise ValueError("tool catalog request does not match the execution scope")
        if not self.turn_id.strip():
            raise ValueError("tool catalog turn_id cannot be empty")
        if len(self.user_message_digest) != 64:
            raise ValueError("tool catalog user_message_digest must be SHA-256")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @classmethod
    def for_turn(
        cls,
        scope: ExecutionScope,
        *,
        conversation_id: str,
        turn_id: str,
        user_message: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolCatalogRequest":
        return cls(
            scope=scope,
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_message_digest=sha256(user_message.encode("utf-8")).hexdigest(),
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class ResolvedToolCatalog:
    """Validated tool definitions with a deterministic immutable identity."""

    _tools: tuple[Tool, ...]
    digest: str
    entries: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_tools(cls, tools: Iterable[Tool]) -> "ResolvedToolCatalog":
        from chulk.tools.registry import Tool, ToolRegistry

        registry = ToolRegistry()
        for tool in deepcopy(tuple(tools)):
            if not isinstance(tool, Tool):
                raise TypeError("tool catalog entries must be Tool instances")
            registry.register(tool)
        normalized = tuple(sorted(registry.list_tools(), key=lambda item: item.name))
        entries = tuple(_catalog_entry(tool) for tool in normalized)
        payload = json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = sha256(payload.encode("utf-8")).hexdigest()
        return cls(
            _tools=deepcopy(normalized),
            digest=digest,
            entries=tuple(_freeze_mapping(item) for item in entries),
        )

    @property
    def tools(self) -> tuple[Tool, ...]:
        """Return detached definitions so callers cannot mutate the snapshot."""
        return deepcopy(self._tools)

    def create_registry(self) -> ToolRegistry:
        """Create the private execution registry for one turn."""
        from chulk.tools.registry import ToolRegistry

        registry = ToolRegistry()
        for tool in self.tools:
            registry.register(tool)
        return registry

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "tools": [deepcopy(dict(entry)) for entry in self.entries],
        }


if TYPE_CHECKING:
    ToolCatalogValue: TypeAlias = ResolvedToolCatalog | Iterable[Tool]
else:
    ToolCatalogValue = Any
ToolCatalogResolver: TypeAlias = Callable[[ToolCatalogRequest], ToolCatalogValue]
AsyncToolCatalogResolver: TypeAlias = Callable[
    [ToolCatalogRequest], Awaitable[ToolCatalogValue]
]


def resolve_tool_catalog(
    resolver: ToolCatalogResolver,
    request: ToolCatalogRequest,
    *,
    timeout_seconds: float | None = None,
) -> ResolvedToolCatalog:
    """Resolve and validate a sync catalog without accepting async adaptation."""
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("tool catalog timeout_seconds must be greater than zero")
    executor: ThreadPoolExecutor | None = None
    try:
        if timeout_seconds is None:
            value = resolver(request)
        else:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(resolver, request)
            try:
                value = future.result(timeout=timeout_seconds)
            except FutureTimeoutError as exc:
                future.cancel()
                raise ToolCatalogResolutionError(
                    "tool catalog resolver timed out"
                ) from exc
    except BaseException as exc:
        if isinstance(exc, ToolCatalogResolutionError):
            raise
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ToolCatalogResolutionError("tool catalog resolver failed") from exc
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    if inspect.isawaitable(value):
        if inspect.iscoroutine(value):
            value.close()
        raise ToolCatalogResolutionError(
            "sync tool catalog resolver returned an awaitable"
        )
    return _coerce_catalog(value)


async def resolve_tool_catalog_async(
    resolver: AsyncToolCatalogResolver,
    request: ToolCatalogRequest,
    *,
    timeout_seconds: float | None = None,
) -> ResolvedToolCatalog:
    """Await and validate a native async catalog with fail-closed timeout."""
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("tool catalog timeout_seconds must be greater than zero")
    try:
        pending = resolver(request)
        if not inspect.isawaitable(pending):
            raise TypeError("async tool catalog resolver must return an awaitable")
        if timeout_seconds is None:
            value = await pending
        else:
            async with asyncio.timeout(timeout_seconds):
                value = await pending
    except asyncio.CancelledError:
        raise
    except TimeoutError as exc:
        raise ToolCatalogResolutionError("tool catalog resolver timed out") from exc
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ToolCatalogResolutionError("tool catalog resolver failed") from exc
    return _coerce_catalog(value)


def _coerce_catalog(value: ToolCatalogValue) -> ResolvedToolCatalog:
    if isinstance(value, ResolvedToolCatalog):
        return ResolvedToolCatalog.from_tools(value.tools)
    try:
        return ResolvedToolCatalog.from_tools(value)
    except (TypeError, ValueError) as exc:
        raise ToolCatalogResolutionError("tool catalog is invalid") from exc


def _catalog_entry(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "arguments": deepcopy(tool.args_schema),
        "output": deepcopy(tool.output_schema),
        "identity": tool.resolved_identity().to_dict(),
        "policy": tool.resolved_policy().to_dict(),
        "application_events": [
            schema.to_dict() for schema in tool.application_event_schemas
        ],
    }


__all__ = [
    "AsyncToolCatalogResolver",
    "ResolvedToolCatalog",
    "ToolCatalogRequest",
    "ToolCatalogResolutionError",
    "ToolCatalogResolver",
    "resolve_tool_catalog",
    "resolve_tool_catalog_async",
]
