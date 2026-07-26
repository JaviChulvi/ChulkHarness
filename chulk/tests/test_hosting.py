"""Hosted runtime, scope, service, and tool-policy contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chulk import (
    AgentConfig,
    AsyncHostedRuntime,
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
from chulk.hosting.services import ServiceBinding, SessionRuntimeServices
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
    agent = AsyncHostedRuntime(
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
    agent = AsyncHostedRuntime(
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
    agent = AsyncHostedRuntime(
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
