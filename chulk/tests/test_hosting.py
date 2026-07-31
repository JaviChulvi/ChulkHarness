"""Hosted runtime, scope, service, and tool-policy contract tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
import inspect
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from chulk import (
    AgentConfig,
    AsyncHostedRuntime,
    AsyncRuntimeServices,
    AsyncServiceBinding,
    ConfigurationError,
    ExecutionScope,
    ExecutionScopeError,
    HostedRuntime,
    ToolEffect,
    ToolIdentity,
    ToolPolicy,
    ToolPolicyHooks,
    ResourceKind,
    ToolRisk,
    ToolConcurrency,
    UsageGroupBy,
    DataClassification,
)
from chulk.core.state import TurnState
from chulk._sdk.config import coerce_config
from chulk.hosting.reference import InMemoryServiceHub
from chulk.hosting.services import (
    ServiceBinding,
    SessionRuntimeServices,
    SkillRuntimeServices,
)
from chulk.llm import LLMCost, LLMClient, LLMUsage
from chulk.memory.security import MemorySecretError
from chulk.plugins import PluginAuditReport
from chulk.runtime import create_async_hosted_agent
from chulk.skills import LearningReviewOutcome, Skill, SkillManifest
from chulk.tools import ToolExecutionContext, ToolRegistry, ToolResult
from chulk.tools import (
    archive_memory as archive_memory_ref,
    compact_memories as compact_memories_ref,
    delete_memory as delete_memory_ref,
    export_memories as export_memories_ref,
    import_memories as import_memories_ref,
    list_memories as list_memories_ref,
    read_trace_artifact as read_trace_artifact_ref,
    restore_memory as restore_memory_ref,
    run_cmd as run_cmd_ref,
    save_memory as save_memory_ref,
    search_memory as search_memory_ref,
    session_read as session_read_ref,
    session_search as session_search_ref,
    summarize_memories as summarize_memories_ref,
    update_memory as update_memory_ref,
)
from chulk.tools.registry import Tool
from chulk.tools.shell import ShellExecutionDecision, ShellExecutionRequest


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.requests.append(messages)
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class _LoopBoundService:
    """Expose only awaitable methods and record their event-loop affinity."""

    def __init__(
        self,
        name: str,
        service: object,
        calls: list[tuple[str, str, int]],
    ) -> None:
        self._name = name
        self._service = service
        self._calls = calls

    def __getattr__(self, name: str):
        value = getattr(self._service, name)
        if not callable(value):
            return value

        async def invoke(*args, **kwargs):
            loop = asyncio.get_running_loop()
            self._calls.append(
                (self._name, name, id(loop))
            )
            result = value(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return invoke


async def _resolve_async_binding(binding, scope: ExecutionScope):
    if isinstance(binding, AsyncServiceBinding):
        return await binding.resolve(scope)
    return await asyncio.to_thread(binding.resolve, scope)


def _loop_bound_async_services(
    hub: InMemoryServiceHub,
    calls: list[tuple[str, str, int]],
    *,
    policy_hooks: ToolPolicyHooks | None = None,
) -> AsyncRuntimeServices:
    base = hub.async_services(policy_hooks=policy_hooks)
    bindings = {}

    for service_name in base.__dataclass_fields__:
        binding = getattr(base, service_name)
        if service_name == "tool_policy":
            bindings[service_name] = AsyncServiceBinding.host(
                policy_hooks or ToolPolicyHooks()
            )
            continue

        async def resolve(
            scope: ExecutionScope,
            *,
            name: str = service_name,
            source=binding,
        ):
            service = await _resolve_async_binding(source, scope)
            if isinstance(service, SessionRuntimeServices):
                return SessionRuntimeServices(
                    store=_LoopBoundService(
                        "sessions.store",
                        service.store,
                        calls,
                    ),
                    search=_LoopBoundService(
                        "sessions.search",
                        service.search,
                        calls,
                    ),
                )
            if isinstance(service, SkillRuntimeServices):
                return SkillRuntimeServices(
                    registry=_LoopBoundService(
                        "skills.registry",
                        service.registry,
                        calls,
                    ),
                    lifecycle_store=service.lifecycle_store,
                    lifecycle=service.lifecycle,
                    learning_proposals=service.learning_proposals,
                    learning_reviewer=service.learning_reviewer,
                )
            return _LoopBoundService(name, service, calls)

        bindings[service_name] = AsyncServiceBinding.scoped(
            resolve,
            ownership="host",
        )

    return AsyncRuntimeServices(**bindings)


def _final(content: str = "hosted ok") -> str:
    return json.dumps({"type": "final_answer", "content": content})


def _tool_call(name: str) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": name,
            "arguments_json": "{}",
        }
    )


def _scope(
    *,
    tenant_id: str = "tenant-a",
    conversation_id: str | None = None,
    grants: frozenset[str] = frozenset(),
) -> ExecutionScope:
    return ExecutionScope(
        tenant_id=tenant_id,
        workspace_id="workspace",
        actor_id="default",
        agent_id="support-agent",
        agent_version="1.2.0",
        run_id="run-1",
        conversation_id=conversation_id,
        grants=grants,
    )


def test_execution_scope_is_canonical_and_children_cannot_escalate() -> None:
    first = _scope(grants=frozenset({"catalog:read", "orders:read"}))
    reordered = ExecutionScope.from_dict(
        {
            **first.to_dict(),
            "grants": ["orders:read", "catalog:read"],
        }
    )

    assert reordered.key == first.key
    assert _scope(tenant_id="tenant-b").key != first.key
    child = first.child(run_id="run-2", grants={"catalog:read"})
    assert child.parent_run_id == first.run_id
    assert child.grants == frozenset({"catalog:read"})

    with pytest.raises(ExecutionScopeError, match="broaden grants"):
        first.child(run_id="run-3", grants={"catalog:read", "orders:write"})


def test_hosted_runtime_is_filesystem_free_and_events_are_scoped(
    tmp_path: Path,
) -> None:
    hub = InMemoryServiceHub()
    public_events = []
    agent = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=hub.services(),
        execution_scope=_scope(),
        on_event=public_events.append,
    )

    assert agent.run("hello") == "hosted ok"
    bound_scope = agent.runtime.execution_scope
    assert bound_scope is not None
    assert bound_scope.conversation_id == agent.conversation_id
    assert list(tmp_path.iterdir()) == []
    assert public_events
    assert all(
        event.extensions["execution_scope_key"] == bound_scope.key
        for event in public_events
    )
    json.dumps([event.to_dict() for event in public_events])

    trace = agent.runtime.trace_logger
    agent.close()
    assert trace.closed is False


def test_hosted_construction_fails_before_resolving_services_without_scope(
    tmp_path: Path,
) -> None:
    calls = 0

    def counted_factory(scope: ExecutionScope) -> object:
        nonlocal calls
        calls += 1
        return object()

    binding = ServiceBinding.scoped(counted_factory)
    services = InMemoryServiceHub().services()
    services = type(services)(
        **{
            **{
                name: getattr(services, name)
                for name in services.__dataclass_fields__
            },
            "memory": binding,
        }
    )

    with pytest.raises(Exception, match="ExecutionScope"):
        HostedRuntime(
            config=AgentConfig(project_root=tmp_path),
            llm=FakeLLM([_final()]),
            tools=[],
            skills=[],
            services=services,
            execution_scope=None,  # type: ignore[arg-type]
        )

    assert calls == 0
    assert list(tmp_path.iterdir()) == []


def test_service_resolution_closes_runtime_owned_resources_on_failure() -> None:
    closed = 0

    class Resource:
        def close(self) -> None:
            nonlocal closed
            closed += 1

    shared = Resource()
    base = InMemoryServiceHub().services()
    fields = {
        name: getattr(base, name)
        for name in base.__dataclass_fields__
    }
    fields["memory"] = ServiceBinding.runtime(shared)
    fields["sessions"] = ServiceBinding.runtime(shared)
    fields["skills"] = ServiceBinding.scoped(
        lambda scope: (_ for _ in ()).throw(RuntimeError("secret detail"))
    )
    services = type(base)(**fields)

    with pytest.raises(ValueError, match=r"skills.*RuntimeError"):
        services.resolve(_scope())

    assert closed == 1


@pytest.mark.asyncio
async def test_async_service_resolution_closes_every_owned_resource_on_failure() -> None:
    closed: list[str] = []

    class Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        async def aclose(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    async def fail_factory(_scope: ExecutionScope):
        raise LookupError("service unavailable")

    base = InMemoryServiceHub().async_services()
    fields = {
        name: getattr(base, name)
        for name in base.__dataclass_fields__
    }
    fields["memory"] = AsyncServiceBinding.runtime(Resource("memory"))
    fields["sessions"] = AsyncServiceBinding.runtime(
        Resource("sessions", fail=True)
    )
    fields["skills"] = AsyncServiceBinding.scoped(fail_factory)
    services = AsyncRuntimeServices(**fields)

    with pytest.raises(ValueError, match=r"skills.*LookupError") as error:
        await services.resolve_async(_scope())

    assert closed == ["sessions", "memory"]
    assert any(
        "sessions close failed" in note
        for note in getattr(error.value, "__notes__", ())
    )


@pytest.mark.asyncio
async def test_cancelled_sync_service_resolution_reclaims_late_resource() -> None:
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    class Resource:
        def close(self) -> None:
            closed.set()

    def resolve(_scope: ExecutionScope) -> Resource:
        started.set()
        if not release.wait(timeout=1):
            raise TimeoutError("test did not release service factory")
        return Resource()

    base = InMemoryServiceHub().async_services()
    fields = {
        name: getattr(base, name)
        for name in base.__dataclass_fields__
    }
    fields["memory"] = ServiceBinding.scoped(resolve)
    services = AsyncRuntimeServices(**fields)
    task = asyncio.create_task(services.resolve_async(_scope()))
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed.is_set()


@pytest.mark.asyncio
async def test_async_hosted_construction_closes_owned_resources_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    class Client:
        def bind_config(self, _config):
            raise RuntimeError("client bind failed")

        async def aclose(self) -> None:
            closed.append("client")

    class Resource:
        async def aclose(self) -> None:
            closed.append("resource")
            raise RuntimeError("resource close failed")

    client = Client()
    monkeypatch.setattr(
        "chulk.runtime._default_llm_client_factory",
        lambda _config: client,
    )
    base = InMemoryServiceHub().async_services()
    fields = {
        name: getattr(base, name)
        for name in base.__dataclass_fields__
    }
    fields["content"] = AsyncServiceBinding.runtime(Resource())

    with pytest.raises(RuntimeError, match="client bind failed") as error:
        await create_async_hosted_agent(
            coerce_config(AgentConfig(project_root=tmp_path)),
            services=AsyncRuntimeServices(**fields),
            execution_scope=_scope(),
            tool_specs=[],
            skill_specs=[],
        )

    assert closed == ["client", "resource"]
    assert any(
        "resource close failed" in note
        for note in getattr(error.value, "__notes__", ())
    )


def test_hosted_service_failure_maps_to_redacted_configuration_error(
    tmp_path: Path,
) -> None:
    base = InMemoryServiceHub().services()
    fields = {
        name: getattr(base, name)
        for name in base.__dataclass_fields__
    }
    fields["memory"] = ServiceBinding.scoped(
        lambda scope: (_ for _ in ()).throw(
            RuntimeError("secret-service-detail")
        )
    )

    with pytest.raises(ConfigurationError) as raised:
        HostedRuntime(
            config=AgentConfig(project_root=tmp_path),
            llm=FakeLLM([_final()]),
            tools=[],
            skills=[],
            services=type(base)(**fields),
            execution_scope=_scope(),
        )

    assert "memory" in str(raised.value)
    assert "secret-service-detail" not in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_in_memory_services_isolate_identical_resource_ids_by_tenant() -> None:
    hub = InMemoryServiceHub()
    first = _scope(tenant_id="tenant-a", conversation_id="conversation")
    second = _scope(tenant_id="tenant-b", conversation_id="conversation")
    first_services = hub.services().resolve(first)
    second_services = hub.services().resolve(second)

    first_store = first_services.sessions.store
    second_store = second_services.sessions.store
    first_store.create_conversation(
        "conversation",
        provider="fake",
        model="fake",
    )

    assert first_store.get_conversation("conversation").id == "conversation"
    with pytest.raises(ValueError, match="No hosted session"):
        second_store.get_conversation("conversation")
    assert first_services.memory is not second_services.memory
    assert first_services.skills.registry is not second_services.skills.registry
    assert first_services.usage is not second_services.usage
    first_artifact = first_services.artifacts.write("private", "tenant-a")
    with pytest.raises(KeyError):
        second_services.artifacts.read(first_artifact.artifact_id)


@pytest.mark.asyncio
async def test_in_memory_reference_services_cover_management_edges() -> None:
    hub = InMemoryServiceHub()
    scope = _scope(conversation_id="conversation")

    assert hub.audit_events(scope) == ()
    assert hub.active_usage_reservations(scope) == ()
    services = hub.services().resolve(scope)

    sessions = services.sessions.store
    sessions.create_conversation(
        "conversation",
        provider="fake",
        model="fake",
    )
    first_summary = sessions.save_conversation_summary(
        "conversation",
        content="first",
        source_message_count=1,
    )
    updated_summary = sessions.save_conversation_summary(
        "conversation",
        content="updated",
        source_message_count=2,
    )
    assert updated_summary.id == first_summary.id
    sessions.save_message(
        "conversation",
        role="user",
        content=" ",
    )
    sessions.save_message(
        "conversation",
        role="user",
        content="hello",
        message_key="message-1",
    )
    sessions.save_message(
        "conversation",
        role="user",
        content="duplicate",
        message_key="message-1",
    )
    assert sessions.max_observation_index("conversation", "turn-1") == 0
    terminal_turn = TurnState(user_message="hello")
    terminal_turn.complete("done")
    assert sessions.save_terminal_turn_bundle(
        "conversation",
        turn_id=terminal_turn.turn_id,
        content="done",
        message_key="assistant-1",
        turn=terminal_turn.to_dict(),
    )
    assert sessions.load_terminal_turn_message(
        "conversation",
        terminal_turn.turn_id,
    ) == {"content": "done", "kind": "final"}
    with pytest.raises(PermissionError, match="cross-scope"):
        sessions.get_conversation("other")

    memory = services.memory
    assert memory.summarize_memories() == "No memories found."
    approved_id = memory.create_memory_proposal(
        "approved memory",
        tags=["profile"],
        evidence="host review",
        conversation_id="conversation",
        turn_id="turn-1",
    )
    rejected_id = memory.create_memory_proposal("rejected memory")
    assert len(memory.list_memory_proposals(status=None)) == 2
    approved = memory.approve_memory_proposal(approved_id)
    assert memory.approve_memory_proposal(approved_id) is approved
    rejected = memory.reject_memory_proposal(rejected_id)
    assert memory.reject_memory_proposal(rejected_id) is rejected
    assert approved.accepted_memory_id is not None
    assert memory.archive_memory(approved.accepted_memory_id)
    assert memory.get_memory(approved.accepted_memory_id) is None
    assert (
        memory.get_memory(
            approved.accepted_memory_id,
            include_archived=True,
        )
        is not None
    )
    assert memory.search_memory(
        "approved",
        include_archived=True,
    )
    original = memory.get_memory(
        approved.accepted_memory_id,
        include_archived=True,
    )
    with pytest.raises(MemorySecretError):
        memory.update_memory(
            approved.accepted_memory_id,
            content=(
                "OPENAI_API_KEY="
                "sk-proj-FakeReferenceCredential123456789"
            ),
        )
    assert (
        memory.get_memory(
            approved.accepted_memory_id,
            include_archived=True,
        )
        == original
    )

    usage = services.usage
    model = usage.reserve_model_request(turn_id="turn-1", request_index=0)
    usage.commit_model_request(
        turn_id="turn-1",
        request_index=0,
        purpose="model",
        usage=LLMUsage(
            input_tokens=4,
            output_tokens=3,
            total_tokens=7,
            cached_input_tokens=2,
        ),
        cost=LLMCost(
            Decimal("0.025"),
            pricing_known=True,
            provider="reference",
            model="test-model",
        ),
    )
    tool = usage.reserve_tool_call(turn_id="turn-1", tool_call_index=1)
    usage.commit_tool_call(
        turn_id="turn-1",
        tool_call_index=1,
        tool_name="lookup",
        attempt=2,
        success=False,
        failure_kind="environment_failure",
    )
    media = usage.reserve_media_transform(
        turn_id="turn-1",
        operation_index=2,
        pricing_per_unit=Decimal("0.01"),
        units=2,
        network_access=True,
    )
    usage.commit_media_transform(
        media,
        processor="resize",
        provider="reference-media",
        content_ref="content:1",
        byte_length=128,
        units=2,
        unit_name="images",
        network_access=True,
        retains_data=False,
    )
    released_media = usage.reserve_media_transform(
        turn_id="turn-1",
        operation_index=3,
    )
    assert usage.release_media_transform(released_media) is released_media
    assert usage.release_model_request(
        turn_id="turn-1",
        request_index=0,
    ) is None
    assert usage.release_tool_call(
        turn_id="turn-1",
        tool_call_index=1,
    ) is None
    assert model.resource_kind is ResourceKind.MODEL
    assert tool.resource_kind is ResourceKind.TOOL
    usage_entries = usage.query(limit=10).entries
    model_entry = next(
        entry
        for entry in usage_entries
        if entry.resource_kind is ResourceKind.MODEL
    )
    tool_entry = next(
        entry
        for entry in usage_entries
        if entry.resource_kind is ResourceKind.TOOL
    )
    media_entry = next(
        entry
        for entry in usage_entries
        if entry.resource_kind is ResourceKind.MEDIA
    )
    assert model_entry.units["total_tokens"] == Decimal(7)
    assert model_entry.cost.amount == Decimal("0.025")
    assert model_entry.provider == "reference"
    assert model_entry.model == "test-model"
    assert tool_entry.units["tool_calls"] == Decimal(1)
    assert tool_entry.tool_or_service == "lookup"
    assert tool_entry.metadata["attempt"] == 2
    assert media_entry.units["bytes"] == Decimal(128)
    assert media_entry.units["images"] == Decimal(2)
    assert media_entry.cost.amount == Decimal("0.02")
    assert media_entry.provider == "reference-media"
    aggregates = usage.group(UsageGroupBy.RESOURCE_KIND)
    model_aggregate = next(item for item in aggregates if item.key == "model")
    assert model_aggregate.total_tokens == 7
    for group_by in UsageGroupBy:
        assert usage.group(group_by)
    usage.reserve_model_request(turn_id="turn-1", request_index=4)
    fallback_entries = usage.commit_model_request(
        turn_id="turn-1",
        request_index=4,
        purpose="fallback",
        fallback_attempts=[
            SimpleNamespace(
                provider="first",
                model="model-a",
                model_profile_id="profile-a",
                success=False,
                error_code="rate_limited",
                usage=LLMUsage(input_tokens=2, output_tokens=1),
                cost=LLMCost(Decimal("0.01"), pricing_known=True),
            ),
            {
                "provider": "second",
                "model": "model-b",
                "model_profile_id": "profile-b",
                "success": True,
                "error_code": None,
                "usage": LLMUsage(
                    input_tokens=3,
                    output_tokens=2,
                ).to_dict(),
                "cost": LLMCost(
                    Decimal("0.02"),
                    pricing_known=True,
                ).to_dict(),
            },
        ],
    )
    assert len(fallback_entries) == 2
    assert fallback_entries[0].units["total_tokens"] == Decimal(3)
    assert fallback_entries[0].cost.amount == Decimal("0.01")
    assert fallback_entries[0].metadata["success"] is False
    assert fallback_entries[1].units["total_tokens"] == Decimal(5)
    assert fallback_entries[1].cost.amount == Decimal("0.02")
    assert fallback_entries[1].provider == "second"

    skills = services.skills.registry
    first_skill = Skill(
        name="first",
        description="first",
        path=Path("first"),
        loaded_content="first content",
        digest="sha256:first",
    )
    second_skill = Skill(
        name="second",
        description="second",
        path=Path("second"),
        digest="sha256:second",
    )
    skills.register(first_skill)
    with pytest.raises(ValueError, match="already registered"):
        skills.register(first_skill)
    skills.register(second_skill)
    selections = skills.load_selected_skills(
        "request",
        pinned_names=("missing", "first", "second"),
        limit=1,
    )
    assert [selection.skill for selection in selections] == [first_skill]
    assert {
        decision.reason for decision in skills.last_routing_result.decisions
    } == {"not_found", "pinned", "skill_count_limit"}
    assert skills.load_content("first") == "first content"
    skills.restrict_to(["first"])
    assert skills.list_visible_skills() == [first_skill]

    services.audit.record("checked", {}, scope=scope)
    assert hub.audit_events(scope)
    trace = services.traces
    trace.write_artifact("evidence", "body")
    trace.close()
    with pytest.raises(RuntimeError, match="trace service is closed"):
        trace.log("late")

    backend = services.execution
    session = backend.open_session(SimpleNamespace())
    with pytest.raises(PermissionError, match="does not expose execution tools"):
        session.missing()
    with pytest.raises(PermissionError, match="does not expose execution tools"):
        await session.missing_async()
    await session.aclose()
    await backend.aclose()
    assert session.closed
    assert backend.closed
    assert services.plugins.audit().profile_id == "default"
    assert services.content.sweep_expired() == 0


def test_in_memory_reference_usage_is_paginated_and_fully_aggregated() -> None:
    usage = InMemoryServiceHub().services().resolve(_scope()).usage
    for index in range(105):
        usage.reserve_tool_call(
            turn_id="turn-1",
            tool_call_index=index,
        )
        usage.commit_tool_call(
            turn_id="turn-1",
            tool_call_index=index,
            tool_name="lookup",
            attempt=1,
            success=True,
            failure_kind=None,
        )

    first = usage.query(limit=50)
    second = usage.query(limit=50, cursor=first.next_cursor)
    third = usage.query(limit=50, cursor=second.next_cursor)

    assert len(first.entries) == 50
    assert first.next_cursor is not None
    assert len(second.entries) == 50
    assert second.next_cursor is not None
    assert len(third.entries) == 5
    assert third.next_cursor is None
    assert len(
        {
            entry.id
            for page in (first, second, third)
            for entry in page.entries
        }
    ) == 105
    aggregate = usage.group(UsageGroupBy.RESOURCE_KIND)
    assert len(aggregate) == 1
    assert aggregate[0].entry_count == 105
    assert aggregate[0].tool_calls == 105
    with pytest.raises(ValueError, match="bounded query limit"):
        usage.group(UsageGroupBy.RESOURCE_KIND, limit=100)
    with pytest.raises(ValueError, match="invalid usage cursor"):
        usage.query(cursor="not-a-cursor")


def test_hosted_resume_rejects_persisted_scope_mismatch(
    tmp_path: Path,
) -> None:
    hub = InMemoryServiceHub()
    requested_scope = _scope(tenant_id="tenant-a")
    original = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=hub.services(),
        execution_scope=requested_scope,
    )
    assert original.run("start") == "hosted ok"
    original_scope = original.execution_scope
    original.close()

    base = hub.services()
    original_services = base.resolve(original_scope)
    fields = {
        name: getattr(base, name)
        for name in base.__dataclass_fields__
    }
    fields["sessions"] = ServiceBinding.host(
        SessionRuntimeServices(
            store=original_services.sessions.store,
            search=original_services.sessions.search,
        )
    )
    mismatched_services = type(base)(**fields)

    with pytest.raises(Exception, match="scope authority mismatch"):
        HostedRuntime(
            config=AgentConfig(project_root=tmp_path),
            llm=FakeLLM([_final()]),
            tools=[],
            skills=[],
            services=mismatched_services,
            execution_scope=_scope(
                tenant_id="tenant-b",
                conversation_id=original_scope.conversation_id,
            ),
            conversation_id=original_scope.conversation_id,
        )


@pytest.mark.asyncio
async def test_async_host_policy_authorizes_before_credentials_and_hides_secrets(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    secret = "secret-host-token"
    scope = _scope(grants=frozenset({"catalog:read"}))
    schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    identity = ToolIdentity.from_schemas(
        "catalog_lookup",
        version="2.1.0",
        input_schema=schema,
    )
    policy = ToolPolicy(
        version="3.0.0",
        required_grants=frozenset({"catalog:read"}),
        risk=ToolRisk.LOW,
        effect=ToolEffect.READ,
    )

    async def authorize(*args) -> bool:
        calls.append("authorize")
        return True

    async def credentials(*args) -> dict[str, str]:
        calls.append("credentials")
        return {"token": secret}

    async def effect_key(*args) -> str:
        calls.append("effect_key")
        return "catalog:SKU-42"

    def lookup(
        arguments: dict,
        context: ToolExecutionContext | None,
    ) -> ToolResult:
        calls.append("tool")
        assert context is not None
        assert context.scope == agent.runtime.execution_scope
        assert context.credentials["token"] == secret
        return ToolResult(
            tool_name="catalog_lookup",
            success=True,
            observation="catalog item found",
            value={"found": True},
        )

    tool = Tool(
        name="catalog_lookup",
        description="Read one catalog item.",
        args_schema=schema,
        callable=lookup,
        accepts_context=True,
        identity=identity,
        policy=policy,
    )
    hub = InMemoryServiceHub()
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_tool_call("catalog_lookup"), _final("done")]),
        tools=[tool],
        skills=[],
        services=hub.async_services(
            policy_hooks=ToolPolicyHooks(
                authorize=authorize,
                resolve_credentials=credentials,
                derive_effect_key=effect_key,
            )
        ),
        execution_scope=scope,
    )

    result = await agent.run_result(
        "look it up",
        tool_context=ToolExecutionContext(
            scope=_scope(tenant_id="attempted-escalation")
        ),
    )

    assert result.content == "done"
    assert calls == ["authorize", "effect_key", "credentials", "tool"]
    call = result.tool_calls[0]
    assert call.metadata["tool_identity"]["version"] == "2.1.0"
    assert call.metadata["tool_policy"]["version"] == "3.0.0"
    assert call.metadata["effect_key"] == "catalog:SKU-42"
    serialized = json.dumps(
        {
            "events": hub.trace_events(agent.runtime.execution_scope),
            "result": result.to_dict(),
            "prompts": agent.runtime.llm_client.requests,
            "context": ToolExecutionContext(
                credentials={"token": secret}
            ).to_dict(),
        },
        default=str,
    )
    assert secret not in serialized
    await agent.close()


@pytest.mark.asyncio
async def test_async_host_events_are_awaited_on_the_running_loop(
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()

    class AsyncSink:
        def __init__(self) -> None:
            self.events = []
            self.loop_ids: list[int] = []

        async def emit(self, event) -> None:
            await asyncio.sleep(0)
            self.loop_ids.append(id(asyncio.get_running_loop()))
            self.events.append(event)

    sink = AsyncSink()
    services = InMemoryServiceHub().async_services()
    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["events"] = ServiceBinding.host(sink)
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final("async events")]),
        tools=[],
        skills=[],
        services=type(services)(**fields),
        execution_scope=_scope(),
    )

    assert await agent.run("hello") == "async events"
    assert sink.events
    assert set(sink.loop_ids) == {id(loop)}
    assert all(event.execution_scope is not None for event in sink.events)
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_shell_binding_preserves_host_safety_options(
    tmp_path: Path,
) -> None:
    requests: list[ShellExecutionRequest] = []

    class UncontainedPolicy:
        def prepare(
            self,
            request: ShellExecutionRequest,
        ) -> ShellExecutionDecision:
            requests.append(request)
            return ShellExecutionDecision.allow(
                request.command,
                policy_name="test-uncontained",
                shell=True,
                containment_applied=False,
            )

    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final("unused")]),
        tools=[run_cmd_ref],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
        shell_execution_policy=UncontainedPolicy(),
        require_shell_containment=True,
    )

    result = await agent.runtime.tool_registry.run_async(
        "run_cmd",
        {"command": "printf must-not-run"},
    )

    assert result.error == "containment_required"
    assert len(requests) == 1
    assert requests[0].command == "printf must-not-run"
    assert result.metadata["child_process_started"] is False
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_factory_and_owned_cleanup_are_native(
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    calls: list[tuple[str, int]] = []

    class OwnedSink:
        def __init__(self) -> None:
            self.events = []
            self.closed = False

        async def emit(self, event) -> None:
            calls.append(("emit", id(asyncio.get_running_loop())))
            self.events.append(event)

        async def aclose(self) -> None:
            calls.append(("close", id(asyncio.get_running_loop())))
            self.closed = True

    sink = OwnedSink()

    async def create_sink(scope):
        calls.append(("factory", id(asyncio.get_running_loop())))
        return sink

    services = InMemoryServiceHub().async_services()
    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["events"] = AsyncServiceBinding.scoped(create_sink)
    native_services = type(services)(**fields)

    with pytest.raises(ValueError, match="AsyncHostedRuntime.create"):
        AsyncHostedRuntime(
            config=AgentConfig(project_root=tmp_path),
            llm=FakeLLM([_final()]),
            tools=[],
            skills=[],
            services=native_services,
            execution_scope=_scope(),
        )

    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final("native")]),
        tools=[],
        skills=[],
        services=native_services,
        execution_scope=_scope(),
    )
    assert await agent.run("hello") == "native"
    await agent.close()

    assert sink.events
    assert sink.closed
    assert calls[0] == ("factory", id(loop))
    assert {loop_id for _name, loop_id in calls} == {id(loop)}


@pytest.mark.asyncio
async def test_async_hosted_runtime_awaits_native_trace_and_audit_sinks(
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()

    class TraceSink:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.loop_ids: list[int] = []

        async def log(self, event_type, payload=None, *, turn_id=None) -> None:
            self.events.append(event_type)
            self.loop_ids.append(id(asyncio.get_running_loop()))

    class AuditSink:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.loop_ids: list[int] = []

        async def record(self, event_type, payload, *, scope) -> None:
            self.events.append(event_type)
            self.loop_ids.append(id(asyncio.get_running_loop()))

    class EventSink:
        async def emit(self, event) -> None:
            return None

    trace = TraceSink()
    audit = AuditSink()
    services = InMemoryServiceHub().async_services()
    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["traces"] = AsyncServiceBinding.host(trace)
    fields["audit"] = AsyncServiceBinding.host(audit)
    fields["events"] = AsyncServiceBinding.host(EventSink())
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final("native sinks")]),
        tools=[],
        skills=[],
        services=type(services)(**fields),
        execution_scope=_scope(),
    )

    assert await agent.run("hello") == "native sinks"
    assert "turn_started" in trace.events
    assert "turn_started" in audit.events
    assert set(trace.loop_ids) == {id(loop)}
    assert set(audit.loop_ids) == {id(loop)}
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_runtime_awaits_all_turn_services_natively(
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    calls: list[tuple[str, str, int]] = []

    async def authorize(*_args) -> bool:
        calls.append(("tool_policy", "authorize", id(asyncio.get_running_loop())))
        return True

    async def effect_key(*_args) -> str:
        calls.append(
            ("tool_policy", "derive_effect_key", id(asyncio.get_running_loop()))
        )
        return "native:verbose"

    async def redact(*_args) -> str:
        calls.append(("tool_policy", "redact", id(asyncio.get_running_loop())))
        return "native redacted result"

    tool = Tool(
        name="verbose",
        description="Return output large enough to require an artifact.",
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        callable=lambda _arguments: ToolResult(
            tool_name="verbose",
            success=True,
            observation="native result",
            stdout="HEAD-" + ("native-output-" * 20) + "TAIL",
        ),
        policy=ToolPolicy(
            effect=ToolEffect.READ,
            required_grants=frozenset({"catalog:read"}),
        ),
    )
    hub = InMemoryServiceHub()
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(
            project_root=tmp_path,
            max_tool_stdout_chars=32,
        ),
        llm=FakeLLM([_tool_call("verbose"), _final("native complete")]),
        tools=[tool],
        skills=[],
        services=_loop_bound_async_services(
            hub,
            calls,
            policy_hooks=ToolPolicyHooks(
                authorize=authorize,
                derive_effect_key=effect_key,
                redact=redact,
            ),
        ),
        execution_scope=_scope(grants=frozenset({"catalog:read"})),
    )

    result = await agent.run_result("run the native service contract")
    output = agent.state.observations[0]["output_metadata"]
    artifact = next(
        item for item in output["artifacts"] if item["field"] == "stdout"
    )
    artifact_read = await agent.read_artifact(artifact["artifact_id"])
    search_page = await agent.search_sessions("native")
    proposals = await agent.list_memory_proposals()
    usage = await agent.query_usage()
    grouped_usage = await agent.group_usage(UsageGroupBy.RESOURCE_KIND)
    plugins = await agent.list_plugins()

    assert result.content == "native complete"
    assert "native redacted result" in result.observations[0].content
    assert "TAIL" in artifact_read["content"]
    assert search_page.query == "native"
    assert proposals == ()
    assert usage.entries
    assert grouped_usage
    assert plugins == ()
    assert {loop_id for _service, _method, loop_id in calls} == {id(loop)}

    invoked = {(service, method) for service, method, _loop_id in calls}
    expected = {
        ("plugins", "verify_startup"),
        ("skills.registry", "load_metadata"),
        ("skills.registry", "configure_environment"),
        ("skills.registry", "load_selected_skills"),
        ("memory", "profile_memories"),
        ("memory", "search_memory"),
        ("sessions.store", "create_conversation"),
        ("sessions.store", "save_turn_snapshot"),
        ("sessions.store", "save_message"),
        ("sessions.store", "save_model_request"),
        ("sessions.store", "save_model_response"),
        ("sessions.store", "save_tool_call"),
        ("sessions.store", "save_tool_observation_bundle"),
        ("sessions.store", "save_terminal_turn_bundle"),
        ("sessions.search", "search"),
        ("execution", "open_session_async"),
        ("usage", "reserve_model_request"),
        ("usage", "commit_model_request"),
        ("usage", "reserve_tool_call"),
        ("usage", "commit_tool_call"),
        ("usage", "query"),
        ("usage", "group"),
        ("plugins", "list"),
        ("artifacts", "write"),
        ("artifacts", "read"),
        ("traces", "log"),
        ("audit", "record"),
        ("events", "emit"),
        ("tool_policy", "authorize"),
        ("tool_policy", "derive_effect_key"),
        ("tool_policy", "redact"),
    }
    assert expected <= invoked
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_tool_refs_await_native_services(
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    calls: list[tuple[str, str, int]] = []
    hub = InMemoryServiceHub()
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[
            archive_memory_ref,
            compact_memories_ref,
            delete_memory_ref,
            export_memories_ref,
            import_memories_ref,
            list_memories_ref,
            read_trace_artifact_ref,
            restore_memory_ref,
            save_memory_ref,
            search_memory_ref,
            session_read_ref,
            session_search_ref,
            summarize_memories_ref,
            update_memory_ref,
        ],
        skills=[],
        services=_loop_bound_async_services(hub, calls),
        execution_scope=_scope(),
    )
    resolved = agent._resolved_async_services()
    artifact = await resolved.artifacts.write("native", "artifact body")
    await resolved.sessions.store.save_message(
        agent.conversation_id,
        role="user",
        content="native session evidence",
    )

    save_result = await agent.tool_registry.run_async(
        "save_memory",
        {"content": "native memory"},
    )
    memory_id = save_result.metadata["memory_id"]
    artifact_result = await agent.tool_registry.run_async(
        "read_trace_artifact",
        {"artifact_id": artifact.artifact_id},
    )
    memory_result = await agent.tool_registry.run_async(
        "search_memory",
        {"query": "native"},
    )
    session_result = await agent.tool_registry.run_async(
        "session_search",
        {"query": "native"},
    )
    session_read_result = await agent.tool_registry.run_async(
        "session_read",
        {"conversation_id": agent.conversation_id, "ordinal": 1},
    )
    list_result = await agent.tool_registry.run_async(
        "list_memories",
        {},
    )
    update_result = await agent.tool_registry.run_async(
        "update_memory",
        {"memory_id": memory_id, "content": "updated native memory"},
    )
    missing_update = await agent.tool_registry.run_async(
        "update_memory",
        {"memory_id": "missing", "content": "missing"},
    )
    summary_result = await agent.tool_registry.run_async(
        "summarize_memories",
        {"query": "updated"},
    )
    archive_result = await agent.tool_registry.run_async(
        "archive_memory",
        {"memory_id": memory_id},
    )
    duplicate_archive = await agent.tool_registry.run_async(
        "archive_memory",
        {"memory_id": memory_id},
    )
    restore_result = await agent.tool_registry.run_async(
        "restore_memory",
        {"memory_id": memory_id},
    )
    duplicate_restore = await agent.tool_registry.run_async(
        "restore_memory",
        {"memory_id": memory_id},
    )
    export_result = await agent.tool_registry.run_async(
        "export_memories",
        {"path": "memories.md"},
    )
    import_result = await agent.tool_registry.run_async(
        "import_memories",
        {"path": "memories.md"},
    )
    compact_result = await agent.tool_registry.run_async(
        "compact_memories",
        {},
    )
    delete_result = await agent.tool_registry.run_async(
        "delete_memory",
        {"memory_id": memory_id},
    )
    missing_delete = await agent.tool_registry.run_async(
        "delete_memory",
        {"memory_id": memory_id},
    )

    assert save_result.success
    assert artifact_result.success
    assert "artifact body" in artifact_result.observation
    assert memory_result.success
    assert "native memory" in memory_result.observation
    assert session_result.success
    assert "native session evidence" in session_result.observation
    assert session_read_result.success
    assert "native session evidence" in session_read_result.observation
    assert list_result.success
    assert update_result.success
    assert not missing_update.success
    assert summary_result.success
    assert "updated native memory" in summary_result.observation
    assert archive_result.success
    assert not duplicate_archive.success
    assert restore_result.success
    assert not duplicate_restore.success
    assert export_result.success
    assert import_result.success
    assert compact_result.success
    assert delete_result.success
    assert not missing_delete.success
    relevant = {
        (service, method, loop_id)
        for service, method, loop_id in calls
        if (service, method)
        in {
            ("artifacts", "read"),
            ("memory", "search_memory"),
            ("sessions.search", "search"),
            ("sessions.search", "read_window"),
        }
    }
    assert relevant == {
        ("artifacts", "read", id(loop)),
        ("memory", "search_memory", id(loop)),
        ("sessions.search", "search", id(loop)),
        ("sessions.search", "read_window", id(loop)),
    }
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_artifact_tool_accepts_mapping_results(
    tmp_path: Path,
) -> None:
    class MappingArtifacts:
        async def read(self, artifact_id: str, **_kwargs):
            return {
                "artifact_id": artifact_id,
                "content": "mapping artifact evidence",
                "byte_count": 25,
                "total_byte_count": 25,
                "truncated": False,
            }

    hub = InMemoryServiceHub()
    services = hub.async_services()
    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["artifacts"] = AsyncServiceBinding.host(MappingArtifacts())
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[read_trace_artifact_ref],
        skills=[],
        services=AsyncRuntimeServices(**fields),
        execution_scope=_scope(),
    )
    artifact_id = f"art_{'a' * 32}"

    result = await agent.tool_registry.run_async(
        "read_trace_artifact",
        {"artifact_id": artifact_id},
    )

    assert result.success
    assert result.value["content"] == "mapping artifact evidence"
    assert result.metadata == {
        "artifact_id": artifact_id,
        "byte_count": 25,
        "total_byte_count": 25,
        "truncated": False,
    }
    await agent.close()


@pytest.mark.asyncio
async def test_reference_profile_memories_exclude_archived_and_general_records(
    tmp_path: Path,
) -> None:
    llm = FakeLLM([_final("done")])
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
    )
    memory = agent._resolved_async_services().memory
    await memory.save_memory(
        "active preference marker",
        tags=["preference"],
    )
    archived_id = await memory.save_memory(
        "archived workflow marker",
        tags=["workflow"],
    )
    await memory.archive_memory(archived_id)
    await memory.save_memory(
        "project-only marker",
        tags=["project"],
    )

    assert await agent.run("load profile") == "done"

    requests = json.dumps(llm.requests)
    assert "active preference marker" in requests
    assert "archived workflow marker" not in requests
    assert "project-only marker" not in requests
    await agent.close()


@pytest.mark.asyncio
async def test_reference_session_resume_excludes_prompt_only_messages(
    tmp_path: Path,
) -> None:
    hub = InMemoryServiceHub()
    scope = _scope()
    first = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=hub.async_services(),
        execution_scope=scope,
    )
    conversation_id = first.conversation_id
    assert await first.run("persist conversation") == "hosted ok"
    store = first._resolved_async_services().sessions.store
    await store.save_message(
        conversation_id,
        role="user",
        content="visible historical question",
    )
    await store.save_message(
        conversation_id,
        role="assistant",
        content="hidden plan display",
        metadata={"prompt_excluded": True},
    )
    await first.close()

    llm = FakeLLM([_final("resumed")])
    resumed = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        services=hub.async_services(),
        execution_scope=scope,
        conversation_id=conversation_id,
    )

    assert await resumed.run("continue") == "resumed"

    requests = json.dumps(llm.requests)
    assert "visible historical question" in requests
    assert "hidden plan display" not in requests
    await resumed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"sensitive": True},
        {"contains_secrets": True},
        {"content_class": "sensitive"},
    ],
)
async def test_reference_session_windows_fail_closed_for_sensitive_messages(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    hub = InMemoryServiceHub()
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[session_read_ref, session_search_ref],
        skills=[],
        services=hub.async_services(),
        execution_scope=_scope(),
    )
    store = agent._resolved_async_services().sessions.store
    await store.save_message(
        agent.conversation_id,
        role="user",
        content="safe session evidence",
    )
    await store.save_message(
        agent.conversation_id,
        role="assistant",
        content="sensitive customer token",
        metadata=metadata,
    )

    default_window = await agent.read_session_window(
        agent.conversation_id,
        ordinal=2,
        before=1,
        after=0,
    )
    explicit_window = await agent.read_session_window(
        agent.conversation_id,
        ordinal=2,
        before=1,
        after=0,
        include_sensitive=True,
    )
    tool_window = await agent.tool_registry.run_async(
        "session_read",
        {
            "conversation_id": agent.conversation_id,
            "ordinal": 2,
            "before": 1,
            "after": 0,
        },
    )
    search = await agent.search_sessions("sensitive customer token")

    assert [message.content for message in default_window.messages] == [
        "safe session evidence"
    ]
    assert [message.content for message in explicit_window.messages] == [
        "safe session evidence",
        "sensitive customer token",
    ]
    assert explicit_window.messages[-1].sensitive is True
    assert "sensitive customer token" not in tool_window.observation
    assert search.hits == ()
    await agent.close()


@pytest.mark.asyncio
async def test_reference_session_results_redact_and_paginate(
    tmp_path: Path,
) -> None:
    raw_secret = "unclassified-secret-value"
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[session_read_ref, session_search_ref],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
    )
    store = agent._resolved_async_services().sessions.store
    await store.save_message(
        agent.conversation_id,
        role="user",
        content=f"pagination api_key={raw_secret}",
    )
    await store.save_message(
        agent.conversation_id,
        role="assistant",
        content="pagination follow-up",
    )

    first_search = await agent.search_sessions("pagination", limit=1)
    assert first_search.next_cursor is not None
    second_search = await agent.search_sessions(
        "pagination",
        limit=1,
        cursor=first_search.next_cursor,
    )
    assert second_search.next_cursor is None
    search_hits = (*first_search.hits, *second_search.hits)
    with pytest.raises(ConfigurationError, match="does not match"):
        await agent.search_sessions(
            "different",
            limit=1,
            cursor=first_search.next_cursor,
        )

    first_window = await agent.read_session_window(
        agent.conversation_id,
        ordinal=1,
        before=0,
        after=1,
        limit=1,
    )
    assert first_window.next_cursor is not None
    second_window = await agent.read_session_window(
        agent.conversation_id,
        ordinal=1,
        before=0,
        after=1,
        limit=1,
        cursor=first_window.next_cursor,
    )
    assert second_window.next_cursor is None
    window_messages = (
        *first_window.messages,
        *second_window.messages,
    )

    tool_search = await agent.tool_registry.run_async(
        "session_search",
        {"query": "pagination"},
    )
    tool_window = await agent.tool_registry.run_async(
        "session_read",
        {
            "conversation_id": agent.conversation_id,
            "ordinal": 1,
            "before": 0,
            "after": 0,
        },
    )

    assert len({hit.message_id for hit in search_hits}) == 2
    assert len({message.message_id for message in window_messages}) == 2
    serialized = json.dumps(
        {
            "search": [hit.to_dict() for hit in search_hits],
            "window": [message.to_dict() for message in window_messages],
            "tool_search": tool_search.observation,
            "tool_window": tool_window.observation,
        }
    )
    assert raw_secret not in serialized
    assert "[redacted]" in serialized
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_management_fails_closed_without_services(
    tmp_path: Path,
) -> None:
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
    )

    assert await agent.list_learning_proposals() == ()
    assert await agent.list_governed_skills() == ()
    assert await agent.list_skill_revisions("missing") == ()
    for operation in (
        lambda: agent.get_learning_proposal("missing"),
        lambda: agent.approve_learning_proposal("missing"),
        lambda: agent.reject_learning_proposal("missing"),
        lambda: agent.rollback_skill("missing"),
        agent.confirm_skill_success,
        agent.review_learning,
    ):
        with pytest.raises(Exception, match="not configured"):
            await operation()

    core = agent.runtime
    memory_policy = core.async_memory_policy
    core.async_memory_policy = None
    assert await agent.list_memory_proposals() == ()
    with pytest.raises(Exception, match="Memory is not configured"):
        await agent.approve_memory_proposal("missing")
    with pytest.raises(Exception, match="Memory is not configured"):
        await agent.reject_memory_proposal("missing")
    core.async_memory_policy = memory_policy

    class LifecycleStore:
        async def record_usage(self, **_kwargs):
            raise AssertionError("invalid versions must not be recorded")

    core.skill_lifecycle_store = LifecycleStore()
    with pytest.raises(KeyError, match="does not exist"):
        await core.confirm_skill_success_async(turn_id="missing")
    incomplete = TurnState(user_message="incomplete")
    core.state.turns.append(incomplete)
    with pytest.raises(ValueError, match="completed host run"):
        await core.confirm_skill_success_async(turn_id=incomplete.turn_id)
    incomplete.complete("done")
    incomplete.extension_metadata["loaded_skill_versions"] = "invalid"
    assert (
        await core.confirm_skill_success_async(turn_id=incomplete.turn_id)
        == ()
    )
    incomplete.extension_metadata["loaded_skill_versions"] = [
        None,
        {"name": 1},
    ]
    assert (
        await core.confirm_skill_success_async(turn_id=incomplete.turn_id)
        == ()
    )

    class Reviewer:
        async def review(self, _context):
            return LearningReviewOutcome(
                skipped=True,
                rationale="nothing to learn",
            )

    core.learning_reviewer = Reviewer()
    core.async_skill_registry = None
    with pytest.raises(KeyError, match="does not exist"):
        await core.review_learning_async(turn_id="missing")
    unfinished = TurnState(user_message="unfinished")
    core.state.turns.append(unfinished)
    with pytest.raises(ValueError, match="finished turn"):
        await core.review_learning_async(turn_id=unfinished.turn_id)
    unfinished.complete("done")
    outcome = await core.review_learning_async(turn_id=unfinished.turn_id)
    assert outcome.skipped
    with pytest.raises(Exception, match="proposals are not configured"):
        await agent.review_learning(turn_id=unfinished.turn_id)

    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_close_preserves_first_failure_and_finishes_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
    )
    closed: list[str] = []

    async def fail_flush() -> None:
        closed.append("flush")
        raise RuntimeError("flush failed")

    class Owned:
        async def aclose_owned(self) -> None:
            closed.append("owned")
            raise RuntimeError("owned close failed")

    monkeypatch.setattr(agent.runtime, "_flush_async_services", fail_flush)
    agent._async_owned_services = Owned()

    with pytest.raises(Exception, match="flush failed") as error:
        await agent.close()

    assert closed == ["flush", "owned"]
    failure = error.value.__cause__ or error.value
    assert any(
        "owned close failed" in note
        for note in getattr(failure, "__notes__", ())
    )
    assert agent._async_owned_services is None


@pytest.mark.asyncio
async def test_async_hosted_close_can_retry_after_cancellation(
    tmp_path: Path,
) -> None:
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
    )
    entered = asyncio.Event()
    context = object()

    class Lifecycle:
        def __init__(self) -> None:
            self.calls = 0
            self.closed: list[object] = []

        async def aclose(self, current: object) -> None:
            self.calls += 1
            if self.calls == 1:
                entered.set()
                await asyncio.Future()
            self.closed.append(current)

    lifecycle = Lifecycle()
    agent.runtime.tool_context_lifecycle = lifecycle
    agent.runtime._tool_contexts["cancelled-close"] = context
    close_task = asyncio.create_task(agent.close())
    await asyncio.wait_for(entered.wait(), timeout=1)

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert not agent.closed
    assert not agent.runtime.closed
    assert agent._async_owned_services is not None

    await agent.close()

    assert lifecycle.calls == 2
    assert lifecycle.closed == [context]
    assert agent.runtime._tool_contexts == {}
    assert agent._async_owned_services is None
    assert agent.closed


@pytest.mark.asyncio
async def test_async_hosted_close_retries_cancelled_core_resource(
    tmp_path: Path,
) -> None:
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
    )
    entered = asyncio.Event()

    class Resource:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        async def aclose(self) -> None:
            self.calls += 1
            if self.calls == 1:
                entered.set()
                await asyncio.Future()
            self.closed = True

    resource = Resource()
    agent.runtime._owned_resources.append(resource)
    close_task = asyncio.create_task(agent.close())
    await asyncio.wait_for(entered.wait(), timeout=1)

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert not agent.closed
    assert agent.runtime._owned_resources == [resource]

    await agent.close()

    assert resource.calls == 2
    assert resource.closed
    assert agent.runtime._owned_resources == []
    assert agent.closed


@pytest.mark.asyncio
async def test_async_hosted_close_waits_for_active_serialized_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    flushes: list[str] = []

    async def active_operation() -> None:
        entered.set()
        await release.wait()

    async def tracked_flush() -> None:
        flushes.append("flush")

    monkeypatch.setattr(
        agent.runtime,
        "_flush_async_services",
        tracked_flush,
    )
    active_task = asyncio.create_task(
        agent._invoke_async(
            "active_operation",
            active_operation,
            serialized=True,
        )
    )
    await entered.wait()
    close_task = asyncio.create_task(agent.close())
    await asyncio.sleep(0)

    assert not close_task.done()
    assert flushes == []
    assert not agent.closed

    release.set()
    await active_task
    await close_task

    assert flushes
    assert agent.closed


@pytest.mark.asyncio
async def test_async_hosted_close_waits_for_active_management_operation(
    tmp_path: Path,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class ArtifactStore:
        def __init__(self) -> None:
            self.closed = False

        async def read(self, _artifact_id: str, **_kwargs):
            entered.set()
            await release.wait()
            return {"artifact_id": "artifact-1"}

        async def aclose(self) -> None:
            self.closed = True

    artifacts = ArtifactStore()
    services = InMemoryServiceHub().async_services()
    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["artifacts"] = AsyncServiceBinding.runtime(artifacts)
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=AsyncRuntimeServices(**fields),
        execution_scope=_scope(),
    )

    read_task = asyncio.create_task(agent.read_artifact("artifact-1"))
    await entered.wait()
    close_task = asyncio.create_task(agent.close())
    await asyncio.sleep(0)

    assert not close_task.done()
    assert not artifacts.closed
    assert not agent.closed

    release.set()
    assert await read_task == {"artifact_id": "artifact-1"}
    await close_task

    assert artifacts.closed
    assert agent.closed


@pytest.mark.asyncio
async def test_async_hosted_flush_attempts_every_journal(
    tmp_path: Path,
) -> None:
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=InMemoryServiceHub().async_services(),
        execution_scope=_scope(),
    )
    calls: list[str] = []

    class Journal:
        def __init__(self, name: str, *, error: str | None = None) -> None:
            self.name = name
            self.error = error

        async def flush(self) -> None:
            calls.append(self.name)
            if self.error is not None:
                raise RuntimeError(self.error)

    agent.runtime.async_flushables = (
        Journal("traces", error="trace flush failed"),
        Journal("sessions"),
        Journal("audit", error="audit flush failed"),
        Journal("events"),
    )

    with pytest.raises(RuntimeError, match="trace flush failed") as error:
        await agent.runtime._flush_async_services()

    assert calls == ["traces", "sessions", "audit", "events"]
    assert any(
        "audit flush failed" in note
        for note in getattr(error.value, "__notes__", ())
    )

    agent.runtime.async_flushables = ()
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_predispatch_flush_failure_releases_reservation(
    tmp_path: Path,
) -> None:
    hub = InMemoryServiceHub()
    llm = FakeLLM([_final()])
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[],
        skills=[],
        services=hub.async_services(),
        execution_scope=_scope(),
    )

    class FailingJournal:
        async def flush(self) -> None:
            raise RuntimeError("journal unavailable")

    agent.runtime.async_flushables = (FailingJournal(),)

    with pytest.raises(Exception, match="journal unavailable"):
        await agent.run("fail before model dispatch")

    assert llm.requests == []
    assert hub.active_usage_reservations(agent.execution_scope) == ()
    assert agent.state.turns[-1].status == "failed"

    agent.runtime.async_flushables = ()
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_learning_facade_awaits_proposal_service(
    tmp_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    calls: list[tuple[str, int]] = []
    proposal = {
        "id": "proposal-1",
        "profile_id": "default",
        "kind": "memory",
        "status": "pending",
    }
    skill = {
        "profile_id": "default",
        "scope": "project",
        "name": "review",
        "version": "1.0.0",
        "digest": "sha256:skill",
        "source": "host",
        "trust": "reviewed",
        "status": "active",
        "active_revision_id": "revision-1",
    }

    class ProposalService:
        async def list(self, **_kwargs):
            calls.append(("proposals.list", id(asyncio.get_running_loop())))
            return [proposal]

        async def get(self, _proposal_id):
            calls.append(("proposals.get", id(asyncio.get_running_loop())))
            return proposal

        async def approve(self, _proposal_id, **_kwargs):
            calls.append(("proposals.approve", id(asyncio.get_running_loop())))
            return {**proposal, "status": "approved"}

        async def reject(self, _proposal_id, **_kwargs):
            calls.append(("proposals.reject", id(asyncio.get_running_loop())))
            return {**proposal, "status": "rejected"}

    class LifecycleStore:
        async def list_skills(self, **_kwargs):
            calls.append(("skills.list", id(asyncio.get_running_loop())))
            return [
                SimpleNamespace(
                    name=skill["name"],
                    scope=skill["scope"],
                    version=skill["version"],
                    digest=skill["digest"],
                )
            ]

        async def list_revisions(self, _name, **_kwargs):
            calls.append(("skills.revisions", id(asyncio.get_running_loop())))
            return [{**skill, "id": "revision-1"}]

        async def record_usage(self, **_kwargs):
            calls.append(("skills.usage", id(asyncio.get_running_loop())))
            return skill

    class Lifecycle:
        async def rollback(self, _revision_id, **_kwargs):
            calls.append(("skills.rollback", id(asyncio.get_running_loop())))
            return skill

    class Reviewer:
        async def review(self, _context):
            calls.append(("learning.review", id(asyncio.get_running_loop())))
            return LearningReviewOutcome(
                skipped=False,
                rationale="reviewed",
                proposal_ids=("proposal-1",),
                review_run_id="review-run-1",
            )

    class Plugins:
        profile_id = "default"

        async def verify_startup(self):
            return PluginAuditReport(profile_id=self.profile_id)

        async def inspect(self, _path):
            calls.append(("plugins.inspect", id(asyncio.get_running_loop())))
            return "inspection"

        async def register_local(self, _path, **_kwargs):
            calls.append(("plugins.register", id(asyncio.get_running_loop())))
            return "registered"

        async def install(self, _path, **_kwargs):
            calls.append(("plugins.install", id(asyncio.get_running_loop())))
            return "installed"

        async def plan_update(self, _path):
            calls.append(("plugins.plan", id(asyncio.get_running_loop())))
            return "plan"

        async def update(self, _path, **_kwargs):
            calls.append(("plugins.update", id(asyncio.get_running_loop())))
            return "updated"

        async def uninstall(self, _name, **_kwargs):
            calls.append(("plugins.uninstall", id(asyncio.get_running_loop())))
            return "uninstalled"

        async def rollback(self, _name, **_kwargs):
            calls.append(("plugins.rollback", id(asyncio.get_running_loop())))
            return "rolled-back"

        async def revoke(self, _name, **_kwargs):
            calls.append(("plugins.revoke", id(asyncio.get_running_loop())))
            return "revoked"

        async def list(self):
            calls.append(("plugins.list", id(asyncio.get_running_loop())))
            return ["listed"]

        async def audit(self):
            calls.append(("plugins.audit", id(asyncio.get_running_loop())))
            return PluginAuditReport(profile_id=self.profile_id)

        async def load_entry_point(self, *_args, **_kwargs):
            calls.append(("plugins.load", id(asyncio.get_running_loop())))
            return "entry-point"

    class Usage:
        async def query(self, **_kwargs):
            calls.append(("usage.query", id(asyncio.get_running_loop())))
            from chulk.usage import UsagePage

            return UsagePage(entries=())

        async def group(self, _group_by, **_kwargs):
            calls.append(("usage.group", id(asyncio.get_running_loop())))
            return []

    hub = InMemoryServiceHub()
    services = hub.async_services()
    proposal_service = ProposalService()
    skill_binding = services.skills

    async def skills(scope: ExecutionScope) -> SkillRuntimeServices:
        assert isinstance(skill_binding, AsyncServiceBinding)
        resolved = await skill_binding.resolve(scope)
        return replace(
            resolved,
            lifecycle_store=LifecycleStore(),
            lifecycle=Lifecycle(),
            learning_proposals=proposal_service,
            learning_reviewer=Reviewer(),
        )

    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["skills"] = AsyncServiceBinding.scoped(
        skills,
        ownership="host",
    )
    fields["plugins"] = AsyncServiceBinding.host(Plugins())
    fields["usage"] = AsyncServiceBinding.host(Usage())
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=AsyncRuntimeServices(**fields),
        execution_scope=_scope(),
    )

    turn = TurnState(user_message="completed")
    turn.extension_metadata["loaded_skill_versions"] = [
        {
            "name": skill["name"],
            "scope": skill["scope"],
            "version": skill["version"],
            "digest": skill["digest"],
        }
    ]
    turn.complete("done")
    agent.state.turns.append(turn)

    assert isinstance(agent.usage_ledger, Usage)
    assert agent.session_search is not None
    assert (await agent.list_learning_proposals())[0].id == "proposal-1"
    assert (await agent.get_learning_proposal("proposal-1")).id == "proposal-1"
    assert (
        await agent.approve_learning_proposal("proposal-1")
    ).status == "approved"
    assert (
        await agent.reject_learning_proposal("proposal-1")
    ).status == "rejected"
    assert (await agent.review_learning()).review_run_id == "review-run-1"
    assert (await agent.list_governed_skills())[0].name == "review"
    assert (await agent.rollback_skill("revision-1")).name == "review"
    assert (
        await agent.list_skill_revisions("review")
    )[0].id == "revision-1"
    assert (await agent.confirm_skill_success())[0].name == "review"
    assert await agent.inspect_plugin(tmp_path) == "inspection"
    assert await agent.register_local_plugin(
        tmp_path,
        approved_by="host",
        acknowledge_host_authority=True,
    ) == "registered"
    assert await agent.install_plugin(
        tmp_path,
        approved_by="host",
        acknowledge_host_authority=True,
    ) == "installed"
    assert await agent.plan_plugin_update(tmp_path) == "plan"
    assert await agent.update_plugin(
        tmp_path,
        approved_by="host",
        acknowledge_host_authority=True,
    ) == "updated"
    assert await agent.uninstall_plugin(
        "plugin",
        approved_by="host",
    ) == "uninstalled"
    assert await agent.rollback_plugin(
        "plugin",
        approved_by="host",
    ) == "rolled-back"
    assert await agent.revoke_plugin(
        "plugin",
        reason="test",
        revoked_by="host",
    ) == "revoked"
    assert await agent.list_plugins() == ("listed",)
    assert (await agent.audit_plugins()).profile_id == "default"
    assert await agent.load_plugin_entry_point(
        "plugin",
        "tools",
        "entry",
    ) == "entry-point"
    assert (await agent.query_usage()).entries == ()
    assert await agent.group_usage("resource_kind") == ()
    assert {loop_id for _name, loop_id in calls} == {id(loop)}
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_skill_usage_uses_hosted_scope_metadata(
    tmp_path: Path,
) -> None:
    skill = Skill(
        name="review",
        description="Review the current change.",
        path=Path("hosted/review/SKILL.md"),
        metadata={"scope": "project"},
        loaded_content="Review the change.",
        manifest=SkillManifest(
            name="review",
            version="1.0.0",
            description="Review the current change.",
            source="host",
            trust="reviewed",
        ),
        digest="sha256:review",
        root=Path("hosted/review"),
    )
    recorded: list[dict[str, object]] = []

    class LifecycleStore:
        async def get_skill(self, name: str, *, scope: str):
            assert (name, scope) == ("review", "project")
            return SimpleNamespace(
                name=name,
                scope=scope,
                version="1.0.0",
                digest="sha256:review",
            )

        async def record_usage(self, **kwargs):
            recorded.append(kwargs)

    class Lifecycle:
        async def rollback(self, _revision_id: str, **_kwargs):
            raise AssertionError("rollback is not expected")

    hub = InMemoryServiceHub()
    services = hub.async_services()
    skill_binding = services.skills

    async def skills(scope: ExecutionScope) -> SkillRuntimeServices:
        resolved = await _resolve_async_binding(skill_binding, scope)
        await resolved.registry.register(skill)
        return replace(
            resolved,
            lifecycle_store=LifecycleStore(),
            lifecycle=Lifecycle(),
        )

    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["skills"] = AsyncServiceBinding.scoped(
        skills,
        ownership="host",
    )
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=["review"],
        services=AsyncRuntimeServices(**fields),
        execution_scope=_scope(),
    )

    assert await agent.run("review this") == "hosted ok"
    assert recorded == [
        {
            "name": "review",
            "version": "1.0.0",
            "digest": "sha256:review",
            "kind": "use",
            "source_event_id": agent.state.turns[-1].turn_id,
            "scope": "project",
        }
    ]
    assert agent.state.turns[-1].extension_metadata[
        "loaded_skill_versions"
    ] == [
        {
            "name": "review",
            "scope": "project",
            "version": "1.0.0",
            "digest": "sha256:review",
        }
    ]
    await agent.close()


@pytest.mark.asyncio
async def test_async_hosted_preparation_failure_closes_open_execution_context(
    tmp_path: Path,
) -> None:
    class Session:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class Execution:
        def __init__(self) -> None:
            self.session = Session()

        async def open_session_async(self, _request):
            return self.session

    class FailingMemory:
        namespace = "failing"

        async def profile_memories(self, *, limit: int = 50):
            del limit
            raise RuntimeError("memory preparation failed")

    hub = InMemoryServiceHub()
    services = hub.async_services()
    execution = Execution()
    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["execution"] = AsyncServiceBinding.host(execution)
    fields["memory"] = AsyncServiceBinding.host(FailingMemory())
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
        services=AsyncRuntimeServices(**fields),
        execution_scope=_scope(),
    )

    with pytest.raises(Exception, match="memory preparation failed"):
        await agent.run("fail after opening the execution context")

    assert execution.session.closed
    assert agent.runtime._tool_contexts == {}
    await agent.close()


@pytest.mark.asyncio
async def test_sync_and_async_hosted_turns_have_observable_parity(
    tmp_path: Path,
) -> None:
    tool = Tool(
        name="lookup",
        description="Return one deterministic lookup.",
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        callable=lambda _arguments: ToolResult(
            tool_name="lookup",
            success=True,
            observation="lookup complete",
            value={"found": True},
        ),
    )
    sync_hub = InMemoryServiceHub()
    sync_agent = HostedRuntime(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_tool_call("lookup"), _final("parity")]),
        tools=[tool],
        skills=[],
        services=sync_hub.services(),
        execution_scope=_scope(),
    )
    sync_result = sync_agent.run_result("compare hosted paths")

    async_hub = InMemoryServiceHub()
    async_agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_tool_call("lookup"), _final("parity")]),
        tools=[tool],
        skills=[],
        services=async_hub.async_services(),
        execution_scope=_scope(),
    )
    async_result = await async_agent.run_result("compare hosted paths")

    def result_signature(result):
        return {
            "content": result.content,
            "status": result.status,
            "errors": result.errors,
            "tools": tuple(
                (
                    call.tool_name,
                    dict(call.arguments),
                    call.phase,
                    call.success,
                    call.failure_kind,
                )
                for call in result.tool_calls
            ),
            "observations": tuple(
                (observation.tool_name, observation.content)
                for observation in result.observations
            ),
            "skills": result.loaded_skill_names,
        }

    assert result_signature(async_result) == result_signature(sync_result)
    assert [
        event["type"]
        for event in async_hub.trace_events(async_agent.execution_scope)
    ] == [
        event["type"]
        for event in sync_hub.trace_events(sync_agent.execution_scope)
    ]
    await async_agent.close()
    sync_agent.close()


class _HangingAsyncLLM(LLMClient):
    def __init__(self, started: asyncio.Event) -> None:
        self.started = started

    def complete(self, messages, **kwargs) -> str:
        raise AssertionError("the synchronous model path must not run")

    async def acomplete_action(self, messages, **kwargs):
        self.started.set()
        await asyncio.Future()
        raise AssertionError(f"unreachable: {messages!r} {kwargs!r}")


@pytest.mark.asyncio
async def test_async_hosted_cancellation_survives_usage_release_failure(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    hub = InMemoryServiceHub()
    services = hub.async_services()
    usage_binding = services.usage

    class FailingReleaseUsage:
        def __init__(self, delegate: object) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str):
            return getattr(self.delegate, name)

        async def release_model_request(self, **kwargs):
            await self.delegate.release_model_request(**kwargs)
            raise RuntimeError("usage release failed")

    async def usage(scope: ExecutionScope) -> FailingReleaseUsage:
        resolved = await _resolve_async_binding(usage_binding, scope)
        return FailingReleaseUsage(resolved)

    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["usage"] = AsyncServiceBinding.scoped(
        usage,
        ownership="host",
    )
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=_HangingAsyncLLM(started),
        tools=[],
        skills=[],
        services=AsyncRuntimeServices(**fields),
        execution_scope=_scope(),
    )
    task = asyncio.create_task(agent.run("wait for cancellation"))
    await asyncio.wait_for(started.wait(), timeout=1)
    scope = agent.execution_scope

    task.cancel()
    with pytest.raises(asyncio.CancelledError) as error:
        await task

    assert hub.active_usage_reservations(scope) == ()
    assert agent.state.turns[-1].status == "cancelled"
    assert any(
        "usage release failed" in note
        for note in getattr(error.value, "__notes__", ())
    )
    await agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
async def test_async_hosted_interruption_releases_leases_and_flushes_terminal_state(
    tmp_path: Path,
    interruption: str,
) -> None:
    started = asyncio.Event()
    hub = InMemoryServiceHub()
    calls: list[tuple[str, str, int]] = []

    class OwnedAudit:
        def __init__(self) -> None:
            self.records = []
            self.closed = False

        async def record(self, event_type, payload, *, scope) -> None:
            self.records.append((event_type, payload, scope))

        async def aclose(self) -> None:
            self.closed = True

    audit = OwnedAudit()
    services = _loop_bound_async_services(hub, calls)
    fields = {
        name: getattr(services, name)
        for name in services.__dataclass_fields__
    }
    fields["audit"] = AsyncServiceBinding.runtime(audit)
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=_HangingAsyncLLM(started),
        tools=[],
        skills=[],
        services=AsyncRuntimeServices(**fields),
        execution_scope=_scope(),
    )
    task = asyncio.create_task(agent.run("wait for interruption"))
    await asyncio.wait_for(started.wait(), timeout=1)
    scope = agent.execution_scope
    assert ("sessions.store", "save_model_request") in {
        (service, method) for service, method, _loop_id in calls
    }
    assert hub.active_usage_reservations(scope)

    if interruption == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.01)

    assert hub.active_usage_reservations(scope) == ()
    assert agent.state.turns[-1].status == "cancelled"
    assert agent.state.turns[-1].ended_at is not None
    assert [event["type"] for event in hub.trace_events(scope)][-2:] == [
        "turn_failed",
        "turn_finished",
    ]
    assert hub.public_events(scope)[-1].name == "run.failed"
    await agent.close()
    assert audit.records[-2][0] == "turn_failed"
    assert audit.records[-1][0] == "turn_finished"
    assert audit.closed


def test_tool_registry_rejects_schema_identity_mismatch() -> None:
    registry = ToolRegistry()
    tool = Tool(
        name="lookup",
        description="Lookup.",
        args_schema={"type": "object", "properties": {}},
        callable=lambda arguments: "ok",
        identity=ToolIdentity.from_schemas(
            "lookup",
            input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
        ),
    )

    with pytest.raises(ValueError, match="input schema"):
        registry.register(tool)


def test_parallel_safe_policy_is_limited_to_read_effects() -> None:
    read_policy = ToolPolicy(
        effect=ToolEffect.READ,
        concurrency=ToolConcurrency.PARALLEL_SAFE,
    )
    assert read_policy.concurrency is ToolConcurrency.PARALLEL_SAFE

    with pytest.raises(ValueError, match="read-only"):
        ToolPolicy(
            effect=ToolEffect.EXTERNAL_WRITE,
            concurrency=ToolConcurrency.PARALLEL_SAFE,
        )


def test_tool_identity_digest_changes_when_implementation_changes() -> None:
    schema = {"type": "object", "properties": {}}
    identity = ToolIdentity.from_schemas("lookup", input_schema=schema)
    first = Tool(
        name="lookup",
        description="Lookup.",
        args_schema=schema,
        callable=lambda arguments: "first",
        identity=identity,
    )
    second = Tool(
        name="lookup",
        description="Lookup.",
        args_schema=schema,
        callable=lambda arguments: "second",
        identity=identity,
    )
    first_registry = ToolRegistry()
    second_registry = ToolRegistry()

    first_registry.register(first)
    second_registry.register(second)

    assert (
        first_registry.get("lookup").resolved_identity().digest
        != second_registry.get("lookup").resolved_identity().digest
    )


@pytest.mark.asyncio
async def test_denied_host_tool_never_resolves_credentials(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    async def deny(*args) -> bool:
        calls.append("authorize")
        return False

    async def credentials(*args) -> dict[str, str]:
        calls.append("credentials")
        return {"token": "must-not-resolve"}

    tool = Tool(
        name="denied_lookup",
        description="Denied lookup.",
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        callable=lambda arguments: "unexpected",
        policy=ToolPolicy(required_grants=frozenset({"catalog:read"})),
    )
    hub = InMemoryServiceHub()
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_tool_call("denied_lookup"), _final()]),
        tools=[tool],
        skills=[],
        services=hub.async_services(
            policy_hooks=ToolPolicyHooks(
                authorize=deny,
                resolve_credentials=credentials,
            )
        ),
        execution_scope=_scope(grants=frozenset({"catalog:read"})),
    )

    result = await agent.run_result("try")

    assert calls == ["authorize"]
    assert result.tool_calls[0].failure_kind == "user_blocked"
    await agent.close()


@pytest.mark.asyncio
async def test_secret_classified_tool_output_is_withheld_everywhere(
    tmp_path: Path,
) -> None:
    secret = "classified-result-secret"
    tool = Tool(
        name="secret_lookup",
        description="Return classified data.",
        args_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        callable=lambda arguments: ToolResult(
            tool_name="secret_lookup",
            success=True,
            observation=secret,
            value={"secret": secret},
            metadata={"structured_output": {"secret": secret}},
        ),
        policy=ToolPolicy(
            output_classification=DataClassification.SECRET,
        ),
    )
    hub = InMemoryServiceHub()
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_tool_call("secret_lookup"), _final()]),
        tools=[tool],
        skills=[],
        services=hub.async_services(),
        execution_scope=_scope(),
    )

    result = await agent.run_result("try")

    serialized = json.dumps(
        {
            "result": result.to_dict(),
            "events": hub.trace_events(agent.runtime.execution_scope),
            "prompts": agent.runtime.llm_client.requests,
        },
        default=str,
    )
    assert secret not in serialized
    assert "secret tool output withheld" in serialized
    await agent.close()
