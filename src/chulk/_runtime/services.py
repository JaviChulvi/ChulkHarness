"""Hosted service resolution for runtime assembly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chulk.core import AgentState
from chulk.execution import ExecutionBackend
from chulk.hosting import ExecutionScope, RuntimeServices
from chulk.hosting.services import ResolvedRuntimeServices
from chulk.media import ContentStore, MediaProcessorRegistry
from chulk.plugins import LocalPluginRegistry


@dataclass(frozen=True)
class RuntimeServiceResolution:
    """Normalized local or hosted inputs used by assembly."""

    conversation_id: str | None
    execution_scope: ExecutionScope | None
    hosted_state: AgentState | None
    services: ResolvedRuntimeServices | None


def resolve_runtime_services(
    services: RuntimeServices | None,
    execution_scope: ExecutionScope | None,
    *,
    conversation_id: str | None,
    tool_specs: Iterable[object] | None,
    skill_specs: object | Iterable[object] | None,
    execution_backend: ExecutionBackend | None,
    plugin_registry: LocalPluginRegistry | None,
    content_store: ContentStore | None,
    media_processors: MediaProcessorRegistry | None,
    memory_namespace: str | None,
) -> RuntimeServiceResolution:
    """Validate hosted inputs and resolve their scoped service bundle."""
    if services is None:
        if execution_scope is not None:
            raise ValueError("execution_scope is only accepted with hosted services")
        return RuntimeServiceResolution(
            conversation_id=conversation_id,
            execution_scope=None,
            hosted_state=None,
            services=None,
        )

    conflicts = [
        name
        for name, value in (
            ("execution_backend", execution_backend),
            ("plugin_registry", plugin_registry),
            ("content_store", content_store),
            ("media_processors", media_processors),
            ("memory_namespace", memory_namespace),
        )
        if value is not None
    ]
    if conflicts:
        raise ValueError(
            "hosted services cannot be combined with individual runtime "
            "injections: " + ", ".join(conflicts)
        )
    if tool_specs is None:
        raise ValueError("hosted runtime requires an explicit tools collection")
    if skill_specs is None:
        raise ValueError("hosted runtime requires an explicit skills collection")
    if execution_scope is None:
        raise ValueError("hosted runtime requires an ExecutionScope")

    requested_conversation_id = conversation_id or execution_scope.conversation_id
    if (
        conversation_id is not None
        and execution_scope.conversation_id not in {None, conversation_id}
    ):
        raise ValueError(
            "execution scope conversation_id does not match conversation_id"
        )
    hosted_state: AgentState | None = None
    if requested_conversation_id is None:
        hosted_state = AgentState()
        requested_conversation_id = hosted_state.conversation_id
    resolved_scope = execution_scope.with_conversation(requested_conversation_id)
    return RuntimeServiceResolution(
        conversation_id=(requested_conversation_id if hosted_state is None else None),
        execution_scope=resolved_scope,
        hosted_state=hosted_state,
        services=services.resolve(resolved_scope),
    )
