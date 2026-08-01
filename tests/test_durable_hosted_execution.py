"""Hosted runtime integration for the shared durable-run effect owner."""

from __future__ import annotations

import asyncio
import json

import pytest

from chulk import (
    AgentConfig,
    ApprovalDecision,
    ApprovalOutcomeKind,
    ApprovalValidation,
    AsyncDurableApprovalService,
    AsyncDurableHostedExecutor,
    AsyncHostedRuntime,
    DurableApprovalService,
    DurableHostedExecutor,
    DurableRunStatus,
    ExecutionScope,
    HostedRuntime,
    EffectLifecyclePayload,
    RunLifecyclePayload,
    RunSubmission,
    StepDefinition,
    ToolEffect,
    ToolIdentity,
    ToolPolicy,
    ToolPolicyHooks,
)
from chulk.hosting.reference import InMemoryServiceHub
from chulk.llm import LLMClient
from chulk.tools.registry import Tool


class _LLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, messages, **kwargs) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _tool_call() -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": "update_ticket",
            "arguments_json": json.dumps({"ticket_id": "42"}),
        }
    )


def _final() -> str:
    return json.dumps({"type": "final_answer", "content": "done"})


def _scope() -> ExecutionScope:
    return ExecutionScope(
        tenant_id="tenant",
        workspace_id="workspace",
        actor_id="default",
        agent_id="support",
        agent_version="1.0.0",
        run_id="run-durable",
        grants=frozenset({"tickets:write"}),
    )


def _submission() -> RunSubmission:
    return RunSubmission(
        idempotency_key="trigger-1",
        input_digest="sha256:input",
        definition_digest="sha256:definition",
        steps=(StepDefinition(id="agent", name="Execute agent turn"),),
    )


def _tool(callable, *, requires_confirmation: bool = False):
    schema = {
        "type": "object",
        "properties": {"ticket_id": {"type": "string"}},
        "required": ["ticket_id"],
        "additionalProperties": False,
    }
    return Tool(
        name="update_ticket",
        description="Update a ticket.",
        args_schema=schema,
        callable=callable,
        requires_confirmation=requires_confirmation,
        permission_level=(
            "external_service" if requires_confirmation else "read"
        ),
        identity=ToolIdentity.from_schemas(
            "update_ticket",
            version="2.0.0",
            input_schema=schema,
            input_schema_version="3.0.0",
        ),
        policy=ToolPolicy(
            version="4.0.0",
            required_grants=frozenset({"tickets:write"}),
            effect=ToolEffect.EXTERNAL_WRITE,
        ),
    )


def _agent(
    tmp_path,
    hub,
    callable,
    *,
    requires_confirmation: bool = False,
    scope: ExecutionScope | None = None,
    conversation_id: str | None = None,
    resolve_credentials=None,
) -> HostedRuntime:
    return HostedRuntime(
        config=AgentConfig(
            project_root=tmp_path,
            permission_profile="workspace-write",
        ),
        llm=_LLM([_tool_call(), _final()]),
        tools=[
            _tool(
                callable,
                requires_confirmation=requires_confirmation,
            )
        ],
        skills=[],
        conversation_id=conversation_id,
        services=hub.services(
            policy_hooks=ToolPolicyHooks(
                authorize=lambda *args: True,
                derive_effect_key=lambda *args: "ticket:42:update",
                resolve_credentials=resolve_credentials,
            )
        ),
        execution_scope=scope or _scope(),
        permission_callback=lambda *args: True,
    )


def test_durable_hosted_execution_records_effect_before_tool_call(
    tmp_path,
) -> None:
    hub = InMemoryServiceHub()
    observed: list[str] = []
    agent = _agent(tmp_path, hub, lambda arguments: observed.append("tool") or "ok")
    runs = agent.runtime.run_store

    outcome = DurableHostedExecutor(agent, runs).execute(
        "update ticket 42",
        _submission(),
        worker_id="worker-a",
        step_id="agent",
    )

    assert outcome.run.status is DurableRunStatus.COMPLETED
    assert observed == ["tool"]
    effects = runs.effects(agent.execution_scope, agent.execution_scope.run_id)
    assert len(effects) == 1
    assert effects[0].status.value == "completed"
    event_names = [
        event.name
        for event in runs.events(
            agent.execution_scope,
            agent.execution_scope.run_id,
        )
    ]
    assert event_names.index("effect.intended") < event_names.index(
        "effect.started"
    )
    assert event_names.index("effect.started") < event_names.index(
        "effect.completed"
    )
    assert event_names[-2:] == ["step.completed", "run.completed"]
    public = hub.public_events(agent.execution_scope)
    durable = [
        event
        for event in public
        if "durable_sequence" in event.extensions
    ]
    assert durable
    assert isinstance(durable[0].payload, RunLifecyclePayload)
    assert any(
        isinstance(event.payload, EffectLifecyclePayload)
        for event in durable
    )
    assert all(
        event.causation_id == previous.event_id
        for previous, event in zip(durable, durable[1:])
    )
    agent.runtime.trace_logger.events.clear()
    assert runs.effects(
        agent.execution_scope,
        agent.execution_scope.run_id,
    )[0].status.value == "completed"
    assert runs.events(
        agent.execution_scope,
        agent.execution_scope.run_id,
    )[-1].name == "run.completed"
    assert list(tmp_path.iterdir()) == []


def test_mutating_tool_failure_becomes_unknown_and_requires_reconciliation(
    tmp_path,
) -> None:
    hub = InMemoryServiceHub()

    def fail(arguments):
        raise ConnectionError("connection closed after dispatch")

    agent = _agent(tmp_path, hub, fail)
    runs = agent.runtime.run_store

    outcome = DurableHostedExecutor(agent, runs).execute(
        "update ticket 42",
        _submission(),
        worker_id="worker-a",
        step_id="agent",
    )

    assert outcome.run.status is DurableRunStatus.UNKNOWN
    assert runs.effects(
        agent.execution_scope,
        agent.execution_scope.run_id,
    )[0].status.value == "unknown"
    assert "run.unknown" in [
        event.name
        for event in runs.events(
            agent.execution_scope,
            agent.execution_scope.run_id,
        )
    ]


def test_policy_ask_pauses_worker_and_consumed_approval_resumes_effect(
    tmp_path,
) -> None:
    hub = InMemoryServiceHub()
    calls: list[str] = []
    credential_resolutions: list[str] = []
    agent = _agent(
        tmp_path,
        hub,
        lambda arguments: calls.append("tool") or "ok",
        requires_confirmation=True,
        resolve_credentials=(
            lambda *args: credential_resolutions.append("resolved")
            or {"token": "runtime-only"}
        ),
    )
    runs = agent.runtime.run_store

    paused = DurableHostedExecutor(agent, runs).execute(
        "update ticket 42",
        _submission(),
        worker_id="worker-a",
        step_id="agent",
    )

    assert paused.run.status is DurableRunStatus.WAITING_FOR_APPROVAL
    assert paused.approval.kind is ApprovalOutcomeKind.PAUSED
    assert paused.approval.approval.effect_id
    assert calls == []
    assert credential_resolutions == []
    assert runs.claim(
        agent.execution_scope,
        worker_id="worker-b",
    ) is None

    approval = paused.approval.approval
    service = DurableApprovalService(
        agent.runtime.approval_store,
        runs,
    )
    service.decide(
        agent.execution_scope,
        approval.id,
        ApprovalDecision.APPROVE,
        decided_by="operator",
        reason="ticket update reviewed",
        idempotency_key="decision-1",
    )
    resumed = service.resume(
        agent.execution_scope,
        approval.id,
        ApprovalValidation(
            scope=agent.execution_scope,
            tool_name=approval.tool_name,
            tool_version=approval.tool_version,
            schema_version=approval.schema_version,
            arguments_digest=approval.arguments_digest,
            policy_version=approval.policy_version,
        ),
        actor="operator",
    )
    assert resumed.resumed

    restarted = _agent(
        tmp_path,
        hub,
        lambda arguments: calls.append("tool") or "ok",
        requires_confirmation=True,
        scope=agent.execution_scope,
        conversation_id=agent.conversation_id,
        resolve_credentials=(
            lambda *args: credential_resolutions.append("resolved")
            or {"token": "runtime-only"}
        ),
    )
    completed = DurableHostedExecutor(restarted, runs).execute(
        "update ticket 42",
        _submission(),
        worker_id="worker-b",
        step_id="agent",
    )

    assert completed.run.status is DurableRunStatus.COMPLETED
    assert calls == ["tool"]
    assert credential_resolutions == ["resolved"]
    assert restarted.runtime.approval_store.get(
        restarted.execution_scope,
        approval.id,
    ).status.value == "consumed"


@pytest.mark.asyncio
async def test_async_hosted_execution_uses_async_run_services(tmp_path) -> None:
    hub = InMemoryServiceHub()
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=_LLM([_tool_call(), _final()]),
        tools=[_tool(lambda arguments: "ok")],
        skills=[],
        services=hub.async_services(
            policy_hooks=ToolPolicyHooks(
                authorize=lambda *args: True,
                derive_effect_key=lambda *args: "ticket:42:update",
            )
        ),
        execution_scope=_scope(),
        permission_callback=lambda *args: True,
    )

    outcome = await AsyncDurableHostedExecutor(
        agent,
        agent.runtime.run_store,
    ).execute(
        "update ticket 42",
        _submission(),
        worker_id="worker-async",
        step_id="agent",
    )

    assert outcome.run.status is DurableRunStatus.COMPLETED
    effects = await agent.runtime.run_store.effects(
        agent.execution_scope,
        agent.execution_scope.run_id,
    )
    assert effects[0].status.value == "completed"
    assert hub.public_events(agent.execution_scope)[-1].name == "run.completed"
    await agent.close()


@pytest.mark.asyncio
async def test_async_durable_cancellation_quarantines_effect_and_blocks_replay(
    tmp_path,
) -> None:
    started = asyncio.Event()
    calls = 0

    async def uncertain_write(_arguments):
        nonlocal calls
        calls += 1
        started.set()
        await asyncio.Future()

    hub = InMemoryServiceHub()
    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(project_root=tmp_path),
        llm=_LLM([_tool_call(), _final()]),
        tools=[_tool(uncertain_write)],
        skills=[],
        services=hub.async_services(
            policy_hooks=ToolPolicyHooks(
                authorize=lambda *args: True,
                derive_effect_key=lambda *args: "ticket:42:update",
            )
        ),
        execution_scope=_scope(),
    )
    executor = AsyncDurableHostedExecutor(
        agent,
        agent.runtime.run_store,
    )
    task = asyncio.create_task(
        executor.execute(
            "update ticket 42",
            _submission(),
            worker_id="worker-cancelled",
            step_id="agent",
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    run = await agent.runtime.run_store.get(
        agent.execution_scope,
        agent.execution_scope.run_id,
    )
    effects = await agent.runtime.run_store.effects(
        agent.execution_scope,
        agent.execution_scope.run_id,
    )
    attempts = await agent.runtime.run_store.attempts(
        agent.execution_scope,
        agent.execution_scope.run_id,
        step_id="agent",
    )
    assert run.status is DurableRunStatus.UNKNOWN
    assert len(effects) == 1
    assert effects[0].status.value == "unknown"
    assert attempts[-1].status.value == "unknown"
    assert hub.active_usage_reservations(agent.execution_scope) == ()

    duplicate = await executor.execute(
        "update ticket 42",
        _submission(),
        worker_id="worker-retry",
        step_id="agent",
    )
    assert duplicate.duplicate
    assert not duplicate.claimed
    assert duplicate.run.status is DurableRunStatus.UNKNOWN
    assert calls == 1
    await agent.close()


@pytest.mark.asyncio
async def test_async_policy_ask_resumes_in_another_runtime(tmp_path) -> None:
    hub = InMemoryServiceHub()
    calls: list[str] = []
    credential_resolutions: list[str] = []

    async def resolve_credentials(*_args):
        credential_resolutions.append("resolved")
        return {"token": "runtime-only"}

    agent = await AsyncHostedRuntime.create(
        config=AgentConfig(
            project_root=tmp_path,
            permission_profile="workspace-write",
        ),
        llm=_LLM([_tool_call(), _final()]),
        tools=[
            _tool(
                lambda arguments: calls.append("tool") or "ok",
                requires_confirmation=True,
            )
        ],
        skills=[],
        services=hub.async_services(
            policy_hooks=ToolPolicyHooks(
                authorize=lambda *args: True,
                derive_effect_key=lambda *args: "ticket:42:update",
                resolve_credentials=resolve_credentials,
            )
        ),
        execution_scope=_scope(),
    )
    runs = agent.runtime.run_store
    paused = await AsyncDurableHostedExecutor(agent, runs).execute(
        "update ticket 42",
        _submission(),
        worker_id="worker-a",
        step_id="agent",
    )
    approval = paused.approval.approval
    assert paused.run.status is DurableRunStatus.WAITING_FOR_APPROVAL
    assert calls == []
    assert credential_resolutions == []

    service = AsyncDurableApprovalService(
        agent.runtime.approval_store,
        runs,
    )
    await service.decide(
        agent.execution_scope,
        approval.id,
        ApprovalDecision.APPROVE,
        decided_by="operator",
        reason="reviewed",
        idempotency_key="async-decision-1",
    )
    resumed = await service.resume(
        agent.execution_scope,
        approval.id,
        ApprovalValidation(
            scope=agent.execution_scope,
            tool_name=approval.tool_name,
            tool_version=approval.tool_version,
            schema_version=approval.schema_version,
            arguments_digest=approval.arguments_digest,
            policy_version=approval.policy_version,
        ),
        actor="operator",
    )
    assert resumed.resumed

    restarted = await AsyncHostedRuntime.create(
        config=AgentConfig(
            project_root=tmp_path,
            permission_profile="workspace-write",
        ),
        llm=_LLM([_tool_call(), _final()]),
        tools=[
            _tool(
                lambda arguments: calls.append("tool") or "ok",
                requires_confirmation=True,
            )
        ],
        skills=[],
        conversation_id=agent.conversation_id,
        services=hub.async_services(
            policy_hooks=ToolPolicyHooks(
                authorize=lambda *args: True,
                derive_effect_key=lambda *args: "ticket:42:update",
                resolve_credentials=resolve_credentials,
            )
        ),
        execution_scope=agent.execution_scope,
    )
    completed = await AsyncDurableHostedExecutor(
        restarted,
        restarted.runtime.run_store,
    ).execute(
        "update ticket 42",
        _submission(),
        worker_id="worker-b",
        step_id="agent",
    )

    assert completed.run.status is DurableRunStatus.COMPLETED
    assert calls == ["tool"]
    assert credential_resolutions == ["resolved"]
    await restarted.close()
    await agent.close()
