"""Hosted runtime, scope, service, and tool-policy contract tests."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

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
    ToolRisk,
    ToolConcurrency,
    DataClassification,
)
from chulk.hosting.reference import InMemoryServiceHub
from chulk.hosting.services import (
    ServiceBinding,
    SessionRuntimeServices,
    SkillRuntimeServices,
)
from chulk.llm import LLMClient
from chulk.tools import ToolExecutionContext, ToolRegistry, ToolResult
from chulk.tools.registry import Tool


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

    assert result.content == "native complete"
    assert "native redacted result" in result.observations[0].content
    assert "TAIL" in artifact_read["content"]
    assert search_page.query == "native"
    assert proposals == ()
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
