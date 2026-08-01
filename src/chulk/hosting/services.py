"""Public service boundaries used by filesystem-free hosted runtimes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Generic, Protocol, TypeVar, runtime_checkable

from chulk.events import AgentEvent
from chulk.hosting.scope import ExecutionScope


T = TypeVar("T")


class ResourceOwnership(StrEnum):
    """Which side must close a resolved service resource."""

    HOST = "host"
    RUNTIME = "runtime"


@dataclass(frozen=True, slots=True)
class ServiceBinding(Generic[T]):
    """One service value or scope-bound factory plus explicit ownership."""

    value: T | None = None
    factory: Callable[[ExecutionScope], T] | None = None
    ownership: ResourceOwnership = ResourceOwnership.HOST

    def __post_init__(self) -> None:
        if (self.value is None) == (self.factory is None):
            raise ValueError("service binding requires exactly one value or factory")
        object.__setattr__(self, "ownership", ResourceOwnership(self.ownership))

    @classmethod
    def host(cls, value: T) -> "ServiceBinding[T]":
        return cls(value=value, ownership=ResourceOwnership.HOST)

    @classmethod
    def runtime(cls, value: T) -> "ServiceBinding[T]":
        return cls(value=value, ownership=ResourceOwnership.RUNTIME)

    @classmethod
    def scoped(
        cls,
        factory: Callable[[ExecutionScope], T],
        *,
        ownership: ResourceOwnership = ResourceOwnership.RUNTIME,
    ) -> "ServiceBinding[T]":
        return cls(factory=factory, ownership=ownership)

    def resolve(self, scope: ExecutionScope) -> T:
        resource = self.value if self.factory is None else self.factory(scope)
        if resource is None:
            raise ValueError("hosted service factory returned no resource")
        return resource


@dataclass(frozen=True, slots=True)
class AsyncServiceBinding(Generic[T]):
    """Native async service value or scope-bound factory."""

    value: T | None = None
    factory: Callable[[ExecutionScope], Awaitable[T]] | None = None
    ownership: ResourceOwnership = ResourceOwnership.HOST

    def __post_init__(self) -> None:
        if (self.value is None) == (self.factory is None):
            raise ValueError(
                "async service binding requires exactly one value or factory"
            )
        object.__setattr__(self, "ownership", ResourceOwnership(self.ownership))

    @classmethod
    def host(cls, value: T) -> "AsyncServiceBinding[T]":
        return cls(value=value, ownership=ResourceOwnership.HOST)

    @classmethod
    def runtime(cls, value: T) -> "AsyncServiceBinding[T]":
        return cls(value=value, ownership=ResourceOwnership.RUNTIME)

    @classmethod
    def scoped(
        cls,
        factory: Callable[[ExecutionScope], Awaitable[T]],
        *,
        ownership: ResourceOwnership = ResourceOwnership.RUNTIME,
    ) -> "AsyncServiceBinding[T]":
        return cls(factory=factory, ownership=ownership)

    async def resolve(self, scope: ExecutionScope) -> T:
        resource = self.value
        if self.factory is not None:
            resource = await self.factory(scope)
        if resource is None:
            raise ValueError("async hosted service factory returned no resource")
        return resource


@runtime_checkable
class MemoryService(Protocol):
    namespace: str

    def profile_memories(self, limit: int = 50) -> list[Any]: ...

    def search_memory(self, query: str, limit: int = 5) -> list[Any]: ...

    def get_memory(self, memory_id: str, *, include_archived: bool = False) -> Any: ...


@runtime_checkable
class SessionService(Protocol):
    def create_conversation(
        self,
        conversation_id: str,
        *,
        provider: str,
        model: str,
        trace_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...

    def get_conversation(self, conversation_id_or_prefix: str) -> Any: ...

    def load_turns(self, conversation_id: str) -> list[Any]: ...

    def load_latest_summary(self, conversation_id: str) -> Any: ...

    def load_recent_messages(
        self,
        conversation_id: str,
        limit: int,
        *,
        after_ordinal: int = 0,
    ) -> list[dict[str, str]]: ...


@runtime_checkable
class SkillService(Protocol):
    last_routing_result: Any

    def load_metadata(self) -> None: ...

    def configure_environment(
        self,
        *,
        available_tools: set[str],
        capabilities: set[str],
    ) -> None: ...

    def list_visible_skills(self) -> list[Any]: ...


@runtime_checkable
class TraceSink(Protocol):
    path: Any
    artifact_store: Any

    def log(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> None: ...

    def activate(self) -> None: ...

    def write_artifact(self, name: str, content: str) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def write(self, label: str, content: str) -> Any: ...

    def read(self, artifact_id: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class UsageService(Protocol):
    def reserve_model_request(self, **kwargs: Any) -> Any: ...

    def commit_model_request(self, **kwargs: Any) -> list[Any]: ...

    def release_model_request(self, reservation: Any) -> Any: ...

    def reserve_tool_call(self, **kwargs: Any) -> Any: ...

    def commit_tool_call(self, **kwargs: Any) -> list[Any]: ...

    def release_tool_call(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class AuditSink(Protocol):
    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        scope: ExecutionScope,
    ) -> None: ...


@runtime_checkable
class EventSink(Protocol):
    """Host-owned destination for redacted public event envelopes."""

    def emit(self, event: AgentEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionRuntimeServices:
    """Session persistence plus the host's scoped read/search facade."""

    store: Any
    search: Any


@dataclass(frozen=True, slots=True)
class SkillRuntimeServices:
    """Skill registry plus optional hosted lifecycle services."""

    registry: Any
    lifecycle_store: Any = None
    lifecycle: Any = None
    learning_proposals: Any = None
    learning_reviewer: Any = None


@runtime_checkable
class AsyncMemoryService(Protocol):
    async def profile_memories(self, limit: int = 50) -> list[Any]: ...

    async def search_memory(self, query: str, limit: int = 5) -> list[Any]: ...


@runtime_checkable
class AsyncSessionService(Protocol):
    async def create_conversation(self, conversation_id: str, **kwargs: Any) -> Any: ...

    async def load_turns(self, conversation_id: str) -> list[Any]: ...


@runtime_checkable
class AsyncTraceSink(Protocol):
    async def log(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> None: ...


@runtime_checkable
class AsyncArtifactStore(Protocol):
    async def write(self, label: str, content: str) -> Any: ...

    async def read(self, artifact_id: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class AsyncAuditSink(Protocol):
    async def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        scope: ExecutionScope,
    ) -> None: ...


@runtime_checkable
class AsyncEventSink(Protocol):
    async def emit(self, event: AgentEvent) -> None: ...


@runtime_checkable
class AsyncUsageService(Protocol):
    async def reserve_model_request(self, **kwargs: Any) -> Any: ...

    async def commit_model_request(self, **kwargs: Any) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    """Complete sync service bundle for a hosted runtime.

    Every field is mandatory so hosted construction cannot silently fall back to
    a local database, directory, trace, artifact, plugin, or execution backend.
    """

    memory: ServiceBinding[Any]
    sessions: ServiceBinding[Any]
    skills: ServiceBinding[Any]
    traces: ServiceBinding[Any]
    artifacts: ServiceBinding[Any]
    usage: ServiceBinding[Any]
    audit: ServiceBinding[Any]
    execution: ServiceBinding[Any]
    plugins: ServiceBinding[Any]
    content: ServiceBinding[Any]
    media: ServiceBinding[Any]
    tool_policy: ServiceBinding[Any]
    runs: ServiceBinding[Any]
    approvals: ServiceBinding[Any]
    events: ServiceBinding[Any]

    def resolve(self, scope: ExecutionScope) -> "ResolvedRuntimeServices":
        values: dict[str, Any] = {}
        owned: list[object] = []
        owned_ids: set[int] = set()
        try:
            for name in self.__dataclass_fields__:
                binding = getattr(self, name)
                if not isinstance(binding, ServiceBinding):
                    raise TypeError(
                        f"hosted service {name} must be a ServiceBinding"
                    )
                try:
                    resource = binding.resolve(scope)
                except Exception as exc:
                    raise ValueError(
                        f"hosted service {name} could not be resolved "
                        f"({type(exc).__name__})"
                    ) from exc
                values[name] = resource
                if (
                    binding.ownership is ResourceOwnership.RUNTIME
                    and id(resource) not in owned_ids
                ):
                    owned.append(resource)
                    owned_ids.add(id(resource))
        except Exception:
            for resource in reversed(owned):
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
            raise
        return ResolvedRuntimeServices(**values, owned_resources=tuple(owned))


@dataclass(frozen=True, slots=True)
class AsyncRuntimeServices:
    """Complete async hosted service bundle.

    The async bundle is intentionally distinct: async hosts must supply native
    async implementations rather than relying on event-loop-blocking adapters.
    """

    memory: ServiceBinding[Any] | AsyncServiceBinding[Any]
    sessions: ServiceBinding[Any] | AsyncServiceBinding[Any]
    skills: ServiceBinding[Any] | AsyncServiceBinding[Any]
    traces: ServiceBinding[Any] | AsyncServiceBinding[Any]
    artifacts: ServiceBinding[Any] | AsyncServiceBinding[Any]
    usage: ServiceBinding[Any] | AsyncServiceBinding[Any]
    audit: ServiceBinding[Any] | AsyncServiceBinding[Any]
    execution: ServiceBinding[Any] | AsyncServiceBinding[Any]
    plugins: ServiceBinding[Any] | AsyncServiceBinding[Any]
    content: ServiceBinding[Any] | AsyncServiceBinding[Any]
    media: ServiceBinding[Any] | AsyncServiceBinding[Any]
    tool_policy: ServiceBinding[Any] | AsyncServiceBinding[Any]
    runs: ServiceBinding[Any] | AsyncServiceBinding[Any]
    approvals: ServiceBinding[Any] | AsyncServiceBinding[Any]
    events: ServiceBinding[Any] | AsyncServiceBinding[Any]

    def as_sync_services(self) -> RuntimeServices:
        """Return the compatibility bundle when every binding is synchronous."""
        bindings: dict[str, ServiceBinding[Any]] = {}
        for name in self.__dataclass_fields__:
            binding = getattr(self, name)
            if not isinstance(binding, ServiceBinding):
                raise ValueError(
                    "native async service bindings require "
                    "await AsyncHostedRuntime.create(...)"
                )
            bindings[name] = binding
        return RuntimeServices(**bindings)

    def resolve(self, scope: ExecutionScope) -> "ResolvedRuntimeServices":
        return self.as_sync_services().resolve(scope)

    async def resolve_async(
        self,
        scope: ExecutionScope,
    ) -> "ResolvedRuntimeServices":
        values: dict[str, Any] = {}
        owned: list[object] = []
        owned_ids: set[int] = set()
        try:
            for name in self.__dataclass_fields__:
                binding = getattr(self, name)
                try:
                    if isinstance(binding, AsyncServiceBinding):
                        resource = await binding.resolve(scope)
                    elif isinstance(binding, ServiceBinding):
                        resource = await asyncio.to_thread(
                            binding.resolve,
                            scope,
                        )
                    else:
                        raise TypeError(
                            f"hosted service {name} must be a service binding"
                        )
                except Exception as exc:
                    raise ValueError(
                        f"hosted service {name} could not be resolved "
                        f"({type(exc).__name__})"
                    ) from exc
                values[name] = resource
                if (
                    binding.ownership is ResourceOwnership.RUNTIME
                    and id(resource) not in owned_ids
                ):
                    owned.append(resource)
                    owned_ids.add(id(resource))
        except BaseException:
            await _aclose_resources(reversed(owned))
            raise
        return ResolvedRuntimeServices(
            **values,
            owned_resources=tuple(owned),
        )


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeServices:
    memory: Any
    sessions: Any
    skills: Any
    traces: Any
    artifacts: Any
    usage: Any
    audit: Any
    execution: Any
    plugins: Any
    content: Any
    media: Any
    tool_policy: Any
    runs: Any
    approvals: Any
    events: Any
    owned_resources: tuple[object, ...]

    def host_bindings(self) -> RuntimeServices:
        """Create a non-owning bundle for the synchronous compatibility core."""
        return RuntimeServices(
            **{
                name: ServiceBinding.host(getattr(self, name))
                for name in RuntimeServices.__dataclass_fields__
            }
        )

    async def aclose_owned(self) -> None:
        """Close only runtime-owned resources through native async methods."""
        await _aclose_resources(reversed(self.owned_resources))


async def _aclose_resources(resources: Any) -> None:
    for resource in resources:
        aclose = getattr(resource, "aclose", None)
        if callable(aclose):
            await aclose()
            continue
        close = getattr(resource, "close", None)
        if callable(close):
            await asyncio.to_thread(close)


# Compatibility names retained for applications using the phase-A API.
TraceService = TraceSink
ArtifactService = ArtifactStore
AuditService = AuditSink
AsyncTraceService = AsyncTraceSink
