"""Public service boundaries used by filesystem-free hosted runtimes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
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
    namespace: str

    async def profile_memories(self, limit: int = 50) -> list[Any]: ...

    async def search_memory(
        self,
        query: str,
        limit: int = 5,
        *,
        include_archived: bool = False,
    ) -> list[Any]: ...

    async def get_memory(
        self,
        memory_id: str,
        *,
        include_archived: bool = False,
    ) -> Any: ...

    async def save_memory(self, content: str, **kwargs: Any) -> str: ...

    async def create_memory_proposal(
        self,
        content: str,
        **kwargs: Any,
    ) -> str: ...

    async def list_memory_proposals(self, **kwargs: Any) -> list[Any]: ...

    async def approve_memory_proposal(self, proposal_id: str) -> Any: ...

    async def reject_memory_proposal(self, proposal_id: str) -> Any: ...

    async def list_memories(self, **kwargs: Any) -> list[Any]: ...

    async def delete_memory(self, memory_id: str) -> bool: ...

    async def update_memory(self, memory_id: str, **kwargs: Any) -> bool: ...

    async def summarize_memories(self, **kwargs: Any) -> str: ...

    async def archive_memory(self, memory_id: str) -> bool: ...

    async def restore_memory(self, memory_id: str) -> bool: ...

    async def compact_memories(self) -> int: ...

    async def import_markdown(self, path: Any) -> list[str]: ...

    async def export_markdown(self, path: Any, **kwargs: Any) -> int: ...


@runtime_checkable
class AsyncSessionService(Protocol):
    async def create_conversation(self, conversation_id: str, **kwargs: Any) -> Any: ...

    async def get_conversation(self, conversation_id_or_prefix: str) -> Any: ...

    async def load_turns(self, conversation_id: str) -> list[Any]: ...

    async def load_latest_summary(self, conversation_id: str) -> Any: ...

    async def load_recent_messages(
        self,
        conversation_id: str,
        limit: int,
        *,
        after_ordinal: int = 0,
    ) -> list[dict[str, str]]: ...

    async def save_turn_snapshot(
        self,
        conversation_id: str,
        turn: dict[str, Any],
    ) -> None: ...

    async def save_conversation_summary(
        self,
        conversation_id: str,
        **kwargs: Any,
    ) -> Any: ...

    async def save_message(self, conversation_id: str, **kwargs: Any) -> None: ...

    async def save_model_request(
        self,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> None: ...

    async def save_model_response(
        self,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> None: ...

    async def save_tool_call(
        self,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> None: ...

    async def save_tool_observation_bundle(
        self,
        conversation_id: str,
        **kwargs: Any,
    ) -> None: ...

    async def max_observation_index(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> int: ...

    async def save_terminal_turn_bundle(
        self,
        conversation_id: str,
        **kwargs: Any,
    ) -> bool: ...

    async def update_conversation_status(
        self,
        conversation_id: str,
        status: str,
    ) -> None: ...

    async def load_terminal_turn_message(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> Any: ...

    async def load_uncheckpointed_hosted_mcp_requests(
        self,
        conversation_id: str,
        turn_id: str,
        *,
        checkpointed_request_count: int,
    ) -> list[dict[str, Any]]: ...

    async def load_tool_calls_without_observations(
        self,
        conversation_id: str,
        turn_id: str,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class AsyncSkillService(Protocol):
    last_routing_result: Any

    async def load_metadata(self) -> None: ...

    async def clear(self) -> None: ...

    async def register(self, skill: Any, *, replace: bool = False) -> None: ...

    async def register_path(self, path: Any) -> Any: ...

    async def register_directory(self, path: Any) -> list[Any]: ...

    async def configure_environment(
        self,
        *,
        available_tools: set[str],
        capabilities: set[str],
    ) -> None: ...

    async def restrict_to(self, names: list[str]) -> None: ...

    async def list_visible_skills(self) -> list[Any]: ...

    async def load_selected_skills(
        self,
        user_request: str,
        *,
        pinned_names: tuple[str, ...] | list[str] = (),
        limit: int | None = None,
    ) -> list[Any]: ...

    async def get_skill(
        self,
        name: str,
        *,
        visible_only: bool = False,
    ) -> Any: ...

    async def load_content(self, name: str) -> str: ...

@runtime_checkable
class AsyncSkillLifecycleStore(Protocol):
    async def get_skill(self, name: str, **kwargs: Any) -> Any: ...

    async def list_skills(self, **kwargs: Any) -> list[Any]: ...

    async def list_revisions(self, name: str, **kwargs: Any) -> list[Any]: ...

    async def record_usage(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class AsyncSkillLifecycleService(Protocol):
    async def rollback(self, revision_id: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class AsyncLearningProposalService(Protocol):
    async def list(self, **kwargs: Any) -> list[Any]: ...

    async def get(self, proposal_id: str) -> Any: ...

    async def approve(self, proposal_id: str, **kwargs: Any) -> Any: ...

    async def reject(self, proposal_id: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class AsyncLearningReviewer(Protocol):
    async def review(self, context: Any) -> Any: ...


@runtime_checkable
class AsyncTraceSink(Protocol):
    path: Any

    async def log(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> None: ...

    async def activate(self) -> None: ...


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

    async def release_model_request(self, **kwargs: Any) -> Any: ...

    async def reserve_tool_call(self, **kwargs: Any) -> Any: ...

    async def commit_tool_call(self, **kwargs: Any) -> list[Any]: ...

    async def release_tool_call(self, **kwargs: Any) -> Any: ...

    async def reserve_media_transform(self, **kwargs: Any) -> Any: ...

    async def commit_media_transform(
        self,
        reservation: Any,
        **kwargs: Any,
    ) -> list[Any]: ...

    async def release_media_transform(self, reservation: Any) -> Any: ...

    async def query(self, **kwargs: Any) -> Any: ...

    async def group(self, group_by: Any, **kwargs: Any) -> list[Any]: ...


@runtime_checkable
class AsyncPluginService(Protocol):
    profile_id: str

    async def verify_startup(self) -> Any: ...

    async def inspect(self, path: Any) -> Any: ...

    async def register_local(self, path: Any, **kwargs: Any) -> Any: ...

    async def install(self, path: Any, **kwargs: Any) -> Any: ...

    async def plan_update(self, path: Any) -> Any: ...

    async def update(self, path: Any, **kwargs: Any) -> Any: ...

    async def uninstall(self, plugin_name: str, **kwargs: Any) -> Any: ...

    async def rollback(self, plugin_name: str, **kwargs: Any) -> Any: ...

    async def revoke(self, plugin_name: str, **kwargs: Any) -> Any: ...

    async def list(self) -> list[Any]: ...

    async def audit(self) -> Any: ...

    async def load_entry_point(
        self,
        plugin_name: str,
        category: Any,
        entry_name: str,
        **kwargs: Any,
    ) -> Any: ...


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
                        resource = await _resolve_sync_binding_async(
                            binding,
                            scope,
                            owned_ids=owned_ids,
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
        except BaseException as exc:
            try:
                await _aclose_resources(reversed(owned))
            except BaseException as cleanup_error:
                exc.add_note(
                    "async service resolution cleanup also failed with "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
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
    _pending_owned_resources: list[object] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_pending_owned_resources",
            list(reversed(self.owned_resources)),
        )

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
        failure: BaseException | None = None
        for resource in tuple(self._pending_owned_resources):
            completed = False
            try:
                await _aclose_resources((resource,))
            except asyncio.CancelledError as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        "another async resource close was cancelled"
                    )
            except BaseException as exc:
                completed = True
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        "another async resource close failed with "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                completed = True
            if completed:
                for index, pending in enumerate(
                    self._pending_owned_resources
                ):
                    if pending is resource:
                        del self._pending_owned_resources[index]
                        break
        if failure is not None:
            raise failure


async def _aclose_resources(resources: Any) -> None:
    failure: BaseException | None = None
    for resource in resources:
        try:
            aclose = getattr(resource, "aclose", None)
            if callable(aclose):
                await aclose()
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(
                    "another async resource close failed with "
                    f"{type(exc).__name__}: {exc}"
                )
    if failure is not None:
        raise failure


async def _resolve_sync_binding_async(
    binding: ServiceBinding[Any],
    scope: ExecutionScope,
    *,
    owned_ids: set[int],
) -> Any:
    """Resolve a sync compatibility binding without leaking on cancellation."""

    resolution = asyncio.create_task(
        asyncio.to_thread(binding.resolve, scope)
    )
    try:
        return await asyncio.shield(resolution)
    except asyncio.CancelledError as cancellation:
        cleanup = asyncio.create_task(
            _reclaim_cancelled_sync_resolution(
                resolution,
                binding.ownership,
                owned_ids=owned_ids,
            )
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            try:
                await cleanup
            except BaseException as cleanup_error:
                cancellation.add_note(
                    "cancelled sync service resolution cleanup also failed "
                    f"with {type(cleanup_error).__name__}: {cleanup_error}"
                )
        except BaseException as cleanup_error:
            cancellation.add_note(
                "cancelled sync service resolution cleanup also failed "
                f"with {type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


async def _reclaim_cancelled_sync_resolution(
    resolution: asyncio.Task[Any],
    ownership: ResourceOwnership,
    *,
    owned_ids: set[int],
) -> None:
    resource = await resolution
    if (
        ownership is ResourceOwnership.RUNTIME
        and id(resource) not in owned_ids
    ):
        await _aclose_resources((resource,))


# Compatibility names retained for applications using the phase-A API.
TraceService = TraceSink
ArtifactService = ArtifactStore
AuditService = AuditSink
AsyncTraceService = AsyncTraceSink
