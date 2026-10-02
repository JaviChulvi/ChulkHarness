"""Process-loss barriers around the real goal, session and effect owners."""
from datetime import datetime, timedelta, timezone

import pytest

from chulk import Agent, AsyncAgent, AgentConfig, Tool
from chulk.core.plan_execution import PlanStepVerification
from chulk.goals import GoalRunner, AsyncGoalRunner
from chulk.runs.store import SQLiteRunStore
from chulk.testing import ScriptedLLMClient
from chulk.goals import GoalStore, GoalService, GoalStep
from chulk.usage import RunBudget
from chulk.sessions.sqlite_store import SQLiteSessionStore


def _goal(tmp_path):
    store = GoalStore(tmp_path / "state.sqlite")
    service = GoalService(store)
    goal = service.create(title="Recover work", acceptance_criteria=("Verified",),
        steps=(GoalStep(id="step-0", title="Work", description="Inspect and verify", acceptance_criterion_ids=("criterion-1",)),),
        budget=RunBudget(max_model_calls=100))
    return service, service.approve(goal.id, expected_revision=goal.revision, approved_by="owner")


def _completion():
    return [{"type": "plan_step_update", "step_update": {"step_id": "step-0", "status": "completed", "evidence": "Done"}},
            {"type": "final_answer", "content": "Done"}]


class ProcessLost(BaseException):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("barrier", ["result", "response", "dispatch", "observation", "verification", "progress", "lost_result"])
async def test_recover_known_work_without_repeating_dispatch(tmp_path, monkeypatch, asynchronous, barrier):
    service, goal = _goal(tmp_path)
    now = datetime.now(timezone.utc)
    service.store.clock = lambda: now
    monkeypatch.setattr("chulk.runs._store_clock.utc_now", lambda: now)
    calls = []
    @Tool
    def read_value() -> str:
        """Read one value."""
        calls.append(1)
        return "known evidence"
    model = ScriptedLLMClient([{"type": "tool_call", "tool_name": "read_value", "arguments": {}}, *_completion()])
    runs = SQLiteRunStore(service.store.db_path)
    tripped = False
    def crash_once(original):
        def invoke(*args, **kwargs):
            nonlocal tripped
            result = original(*args, **kwargs)
            if not tripped:
                tripped = True
                raise ProcessLost()
            return result
        return invoke
    if barrier == "result":
        monkeypatch.setattr(runs, "record_effect_result", crash_once(runs.record_effect_result))
    if barrier == "verification":
        monkeypatch.setattr(service.store, "record_verification", crash_once(service.store.record_verification))
    if barrier == "progress":
        monkeypatch.setattr(service.store, "apply_verified_step", crash_once(service.store.apply_verified_step))
    if barrier == "lost_result":
        def lose_result(*args, **kwargs):
            nonlocal tripped
            tripped = True
            raise ProcessLost()
        monkeypatch.setattr(runs, "record_effect_result", lose_result)
    if barrier == "dispatch":
        # The dispatch marker is durable, but the result is unknown. Recovery stops.
        monkeypatch.setattr(runs, "mark_effect_started", crash_once(runs.mark_effect_started))
    def factory(context, conversation_id):
        agent = (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path,
            store_path=service.store.db_path, max_reflection_attempts=0), llm=model, tools=[read_value], skills=[],
            goal_execution=context, conversation_id=conversation_id)
        runtime = agent.runtime
        terminalize = runtime._terminalize_exception
        runtime._terminalize_exception = lambda turn, exc: None if isinstance(exc, ProcessLost) else terminalize(turn, exc)
        if barrier in {"response", "observation"}:
            # Wrap persistence itself: the process disappears immediately after commit.
            method = "save_model_response" if barrier == "response" else "save_tool_observation_bundle"
            if not tripped:
                monkeypatch.setattr(SQLiteSessionStore, method, crash_once(getattr(SQLiteSessionStore, method)))
        return agent
    verified = []
    def verify(_):
        verified.append(1)
        return PlanStepVerification(True, "Host checked evidence")
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, runs=runs, agent_factory=factory,
        verifier=verify, clock=lambda: now)
    with pytest.raises(ProcessLost):
        if asynchronous:
            await runner.run(goal.id)
        else:
            runner.run(goal.id)
    now += timedelta(seconds=121)
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    if barrier in {"dispatch", "lost_result"}:
        assert result.stop_reason.value == "recovery_required"
        assert calls == ([1] if barrier == "lost_result" else [])
        from chulk.goals import GoalActionConflictError
        blocked = service.block_step(goal.id, "step-0", expected_revision=result.goal.revision, actor="owner", reason="uncertain")
        with pytest.raises(GoalActionConflictError, match="durable effect"):
            service.retry_step(goal.id, "step-0", expected_revision=blocked.revision, actor="owner")
        assert service.store.get(goal.id).steps[0].attempt == 1
    else:
        assert result.stop_reason.value == "completed", result.detail
        assert calls == [1]
        assert result.usage["tool_calls"] == 1
        assert result.usage["model_calls"] == 3
        assert result.goal.steps[0].attempt == 1
        assert verified == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_hosted_goal_approval_restart_preserves_user_pause(tmp_path, asynchronous):
    from dataclasses import replace
    from chulk import (HostedRuntime, AsyncHostedRuntime, ExecutionScope, ServiceBinding,
                       ApprovalDecision, ApprovalValidation, DurableApprovalService,
                       ToolEffect, ToolPolicy)
    from chulk.tools.registry import Tool as RegisteredTool
    from chulk.hosting.reference import InMemoryServiceHub
    from chulk.hosting.services import SessionRuntimeServices
    from chulk.approvals.store import SQLiteApprovalStore
    from chulk.runs.async_store import AsyncRunStoreAdapter
    from chulk.usage import ModelUsageAccounting, SQLiteUsageStore, UsageDimensions, UsageLedger

    service, goal = _goal(tmp_path)
    hub = InMemoryServiceHub()
    runs = SQLiteRunStore(service.store.db_path)
    approvals = SQLiteApprovalStore(service.store.db_path)
    scope = ExecutionScope(tenant_id="tenant", workspace_id="workspace", actor_id="owner",
        agent_id="worker", agent_version="1", run_id=f"goal:{goal.id}")
    model = ScriptedLLMClient([{"type": "tool_call", "tool_name": "publish", "arguments": {}}, *_completion()])
    calls = []
    tool = RegisteredTool(name="publish", description="Publish approved work", args_schema={"type": "object", "properties": {}},
        callable=lambda _: calls.append(1) or "receipt", permission_level="external_service", requires_confirmation=True,
        policy=ToolPolicy(effect=ToolEffect.EXTERNAL_WRITE))

    class Usage(ModelUsageAccounting):
        def group(self, group_by, **kwargs):
            return UsageLedger(self.store.db_path).group(group_by, **kwargs)

    def usage(bound_scope):
        return Usage(SQLiteUsageStore(service.store.db_path), client=model,
            dimensions=UsageDimensions(conversation_id=bound_scope.conversation_id, goal_id=goal.id),
            budget=service.store.get(goal.id).budget, max_output_tokens=1000)

    services = replace(hub.async_services() if asynchronous else hub.services(),
        sessions=ServiceBinding.host(SessionRuntimeServices(SQLiteSessionStore(service.store.db_path), None)),
        usage=ServiceBinding.scoped(usage), runs=ServiceBinding.host(runs), approvals=ServiceBinding.host(approvals))
    observed_scopes = []
    def options(context, conversation_id):
        observed_scopes.append(context.execution_scope)
        return dict(config=AgentConfig(project_root=tmp_path, max_reflection_attempts=0, permission_profile="workspace-write"),
            llm=model, tools=[tool], skills=[], goal_execution=context, services=services,
            execution_scope=context.execution_scope, conversation_id=conversation_id)
    if asynchronous:
        async def factory(context, conversation_id):
            return await AsyncHostedRuntime.create(**options(context, conversation_id))
    else:
        def factory(context, conversation_id):
            return HostedRuntime(**options(context, conversation_id))
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store,
        runs=AsyncRunStoreAdapter(runs) if asynchronous else runs, execution_scope=scope,
        agent_factory=factory, verifier=lambda _: PlanStepVerification(True, "Approved receipt"))
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == "approval_required", result.detail
    assert calls == []
    bound = observed_scopes[-1]
    assert runs.get(bound, bound.run_id).status.value == "waiting_for_approval"
    approval = approvals.list(bound)[0]
    current = service.store.get(goal.id)
    service.pause(goal.id, expected_revision=current.revision, actor="owner")
    approval_service = DurableApprovalService(approvals, runs)
    approval_service.decide(bound, approval.id, ApprovalDecision.APPROVE, decided_by="owner", reason="Reviewed", idempotency_key="approve")
    approval_service.resume(bound, approval.id, ApprovalValidation(scope=bound, tool_name=approval.tool_name,
        tool_version=approval.tool_version, schema_version=approval.schema_version,
        arguments_digest=approval.arguments_digest, policy_version=approval.policy_version), actor="owner")
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == "paused"
    assert calls == []
    current = service.store.get(goal.id)
    service.resume(goal.id, expected_revision=current.revision, actor="owner")
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == "completed", result.detail
    assert calls == [1]
    assert result.goal.steps[0].attempt == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dispatched", [False, True])
async def test_unsent_model_can_resume_but_unknown_request_keeps_allowance(tmp_path, monkeypatch, asynchronous, dispatched):
    from chulk.usage import ModelUsageAccounting
    from chulk.storage.sqlite import sqlite_connection
    service, goal = _goal(tmp_path)
    now = datetime.now(timezone.utc)
    service.store.clock = lambda: now
    monkeypatch.setattr("chulk.runs._store_clock.utc_now", lambda: now)
    model = ScriptedLLMClient(_completion())
    original = ModelUsageAccounting.mark_model_dispatched
    first = True
    def interrupt(accounting, **kwargs):
        nonlocal first
        if first:
            first = False
            if dispatched:
                original(accounting, **kwargs)
            raise ProcessLost()
        return original(accounting, **kwargs)
    monkeypatch.setattr(ModelUsageAccounting, "mark_model_dispatched", interrupt)
    def factory(context, conversation_id):
        agent = (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path, store_path=service.store.db_path, max_reflection_attempts=0),
            llm=model, tools=[], skills=[], goal_execution=context, conversation_id=conversation_id)
        agent.runtime._terminalize_exception = lambda *_: None
        return agent
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, agent_factory=factory,
        verifier=lambda _: PlanStepVerification(True, "Checked"), clock=lambda: now)
    with pytest.raises(ProcessLost):
        if asynchronous:
            await runner.run(goal.id)
        else:
            runner.run(goal.id)
    now += timedelta(seconds=121)
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == ("recovery_required" if dispatched else "completed"), result.detail
    with sqlite_connection(service.store.db_path) as conn:
        active = conn.execute("SELECT COUNT(*) FROM usage_reservations WHERE state = 'active' AND dispatched = 1").fetchone()[0]
    assert (active > 0) is dispatched
    if dispatched:
        again = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
        assert again.stop_reason.value == "recovery_required"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_deadline_stops_repair_before_another_provider_call(tmp_path, monkeypatch, asynchronous):
    from chulk.llm import LLMClient
    service, goal = _goal(tmp_path)
    now = datetime.now(timezone.utc)
    service.store.clock = lambda: now
    monkeypatch.setattr("chulk.runs._store_clock.utc_now", lambda: now)
    goal = service.update_budget(goal.id, expected_revision=goal.revision,
        budget=RunBudget(max_model_calls=100, deadline=now + timedelta(seconds=1)), actor="owner")
    calls = []
    class Model(LLMClient):
        def complete(self, messages, **kwargs):
            nonlocal now
            calls.append(1)
            now += timedelta(seconds=2)
            return "invalid JSON requiring repair"
    def factory(context, conversation_id):
        return (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path, store_path=service.store.db_path, max_reflection_attempts=0),
            llm=Model(), tools=[], skills=[], goal_execution=context, conversation_id=conversation_id)
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, agent_factory=factory,
        verifier=lambda _: PlanStepVerification(True, "Checked"), clock=lambda: now)
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == "budget_exhausted", result.detail
    assert "deadline" in result.detail
    assert calls == [1]


def test_recovery_migration_upgrades_reopens_and_rolls_back(tmp_path):
    from chulk.storage.sqlite import initialize_sqlite_database, sqlite_connection, SQLiteMigration, SQLiteMigrationError
    from chulk.storage.migrations import SQLITE_MIGRATIONS
    path = tmp_path / 'upgrade.sqlite'
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:24])
    def fail(conn):
        SQLITE_MIGRATIONS[24].apply(conn)
        raise RuntimeError('interrupted migration')
    with pytest.raises(SQLiteMigrationError):
        initialize_sqlite_database(path, migrations=(*SQLITE_MIGRATIONS[:24], SQLiteMigration(25, 'interrupted', fail)))
    with sqlite_connection(path) as conn:
        assert 'result_ref' not in {row['name'] for row in conn.execute('PRAGMA table_info(durable_effects)')}
    initialize_sqlite_database(path)
    initialize_sqlite_database(path)
    with sqlite_connection(path) as conn:
        assert 'result_ref' in {row['name'] for row in conn.execute('PRAGMA table_info(durable_effects)')}
        assert 'dispatched' in {row['name'] for row in conn.execute('PRAGMA table_info(usage_reservations)')}


@pytest.mark.asyncio
@pytest.mark.parametrize('asynchronous', [False, True])
async def test_restart_between_admission_and_first_turn_is_safe(tmp_path, monkeypatch, asynchronous):
    service, goal = _goal(tmp_path)
    now = datetime.now(timezone.utc)
    service.store.clock = lambda: now
    monkeypatch.setattr('chulk.runs._store_clock.utc_now', lambda: now)
    first = True
    model = ScriptedLLMClient(_completion())
    def factory(context, conversation_id):
        nonlocal first
        if first:
            first = False
            raise ProcessLost()
        return (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path,
            store_path=service.store.db_path, max_reflection_attempts=0), llm=model, tools=[], skills=[],
            goal_execution=context, conversation_id=conversation_id)
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, agent_factory=factory,
        verifier=lambda _: PlanStepVerification(True, 'checked'), clock=lambda: now)
    with pytest.raises(ProcessLost):
        if asynchronous:
            await runner.run(goal.id)
        else:
            runner.run(goal.id)
    now += timedelta(seconds=121)
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == 'completed'
    assert result.goal.steps[0].attempt == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('phase', ['reflection', 'final_answer', 'protocol_failure', 'context_summary'])
async def test_known_auxiliary_response_survives_receipt_crash(tmp_path, monkeypatch, asynchronous, phase):
    from chulk import FinalAnswerStreamingMode
    service, goal = _goal(tmp_path)
    now = datetime.now(timezone.utc)
    service.store.clock = lambda: now
    monkeypatch.setattr('chulk.runs._store_clock.utc_now', lambda: now)
    script = ['invalid'] if phase == 'protocol_failure' else [*_completion(), {'approved': True, 'reason': 'checked'}, 'Final']
    if phase == 'context_summary':
        script.insert(0, {'objective': 'Historical discussion'})
    model = ScriptedLLMClient(script)
    response_index = {'reflection': 3, 'final_answer': 4, 'protocol_failure': 1, 'context_summary': 1}[phase]
    original = SQLiteSessionStore.save_model_response
    tripped = False
    def save(*args, **kwargs):
        nonlocal tripped
        result = original(*args, **kwargs)
        # The durable snapshot and raw response commit together.
        payload = kwargs.get('payload', args[-1] if args else {})
        if payload.get('request_index') == response_index and not tripped:
            tripped = True
            raise ProcessLost()
        return result
    monkeypatch.setattr(SQLiteSessionStore, 'save_model_response', save)
    def factory(context, conversation_id):
        agent = (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path,
            store_path=service.store.db_path, max_reflection_attempts=1), llm=model, tools=[], skills=[],
            goal_execution=context, conversation_id=conversation_id, final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL)
        if phase == 'context_summary':
            from chulk.core.context import ContextBudget
            agent.runtime._model_transport.context_budget = ContextBudget(max_prompt_tokens=2600, response_reserve_tokens=0)
            if conversation_id is None:
                agent.runtime.memory.add_user_message('Historical source ' + 'a' * 10_000)
                agent.runtime.memory.add_assistant_message('Historical response ' + 'b' * 10_000)
        agent.runtime._model_transport.max_json_repair_attempts = 0
        terminalize = agent.runtime._terminalize_exception
        agent.runtime._terminalize_exception = lambda turn, exc: None if isinstance(exc, ProcessLost) else terminalize(turn, exc)
        return agent
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, agent_factory=factory,
        verifier=lambda _: PlanStepVerification(True, 'checked'), clock=lambda: now)
    with pytest.raises(ProcessLost):
        if asynchronous:
            await runner.run(goal.id)
        else:
            runner.run(goal.id)
    now += timedelta(seconds=121)
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == ('blocked' if phase == 'protocol_failure' else 'completed'), result.detail
    assert len(model.call_log) == (1 if phase == 'protocol_failure' else 5 if phase == 'context_summary' else 4)


@pytest.mark.asyncio
async def test_sync_tool_does_not_block_async_claim_renewal(tmp_path, monkeypatch):
    import threading
    service, goal = _goal(tmp_path)
    runs = SQLiteRunStore(service.store.db_path)
    renewed = threading.Event()
    entered = threading.Event()
    original = runs.renew
    def renew(*args, **kwargs):
        result = original(*args, **kwargs)
        if entered.is_set():
            renewed.set()
        return result
    monkeypatch.setattr(runs, 'renew', renew)
    @Tool
    def wait_for_renewal() -> str:
        """Wait for the independent heartbeat to renew ownership."""
        entered.set()
        assert renewed.wait(3), 'synchronous tool blocked the async heartbeat'
        return 'renewed'
    model = ScriptedLLMClient([{'type': 'tool_call', 'tool_name': 'wait_for_renewal', 'arguments': {}}, *_completion()])
    runner = AsyncGoalRunner(service.store, runs=runs, renewal_seconds=.02,
        agent_factory=lambda context, conversation_id: AsyncAgent(config=AgentConfig(project_root=tmp_path,
            store_path=service.store.db_path, max_reflection_attempts=0), llm=model, tools=[wait_for_renewal], skills=[],
            goal_execution=context, conversation_id=conversation_id),
        verifier=lambda _: PlanStepVerification(True, 'checked'))
    result = await runner.run(goal.id)
    assert result.stop_reason.value == 'completed', result.detail
    assert renewed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('asynchronous', [False, True])
async def test_durable_claim_lost_before_progress_commit(tmp_path, monkeypatch, asynchronous):
    service, goal = _goal(tmp_path)
    now = datetime.now(timezone.utc)
    durable_now = now
    service.store.clock = lambda: now
    monkeypatch.setattr('chulk.runs._store_clock.utc_now', lambda: durable_now)
    model = ScriptedLLMClient(_completion())
    def factory(context, conversation_id):
        return (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path,
            store_path=service.store.db_path, max_reflection_attempts=0), llm=model, tools=[], skills=[],
            goal_execution=context, conversation_id=conversation_id)
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, agent_factory=factory,
        verifier=lambda _: PlanStepVerification(True, 'checked'), clock=lambda: now)
    if asynchronous:
        original_async = runner._finish_async
        async def finish_async(*args, **kwargs):
            nonlocal durable_now
            durable_now += timedelta(seconds=121)
            return await original_async(*args, **kwargs)
        monkeypatch.setattr(runner, '_finish_async', finish_async)
        result = await runner.run(goal.id)
    else:
        original = runner._finish
        def finish(*args, **kwargs):
            nonlocal durable_now
            durable_now += timedelta(seconds=121)
            return original(*args, **kwargs)
        monkeypatch.setattr(runner, '_finish', finish)
        result = runner.run(goal.id)
    assert result.stop_reason.value == 'lease_lost'
    assert service.store.get(goal.id).steps[0].status.value == 'running'
    assert len(model.call_log) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('asynchronous', [False, True])
async def test_opaque_provider_tools_fail_before_admission_to_model(tmp_path, monkeypatch, asynchronous):
    from chulk.errors import ConfigurationError
    from chulk.runs.models import RunStatus
    service, goal = _goal(tmp_path)
    model = ScriptedLLMClient([])
    runs = SQLiteRunStore(service.store.db_path)
    scopes = []
    def factory(context, conversation_id):
        scopes.append(context.execution_scope)
        agent = (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path,
            store_path=service.store.db_path), llm=model, tools=[], skills=[],
            goal_execution=context, conversation_id=conversation_id)
        agent.runtime._model_transport.mcp_servers = [object()]
        return agent
    monkeypatch.setattr('chulk.goals.runner.client_supports_hosted_mcp_tools', lambda _: True)
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, runs=runs, agent_factory=factory,
        verifier=lambda _: PlanStepVerification(True, 'checked'))
    with pytest.raises(ConfigurationError, match='journaled registry tools'):
        if asynchronous:
            await runner.run(goal.id)
        else:
            runner.run(goal.id)
    assert len(model.call_log) == 0
    assert runs.get(scopes[0], scopes[0].run_id).status is RunStatus.QUEUED


@pytest.mark.asyncio
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('purpose', ['action', 'response', 'stream'])
async def test_refreshed_fallback_rechecks_ownership_before_next_provider(asynchronous, purpose):
    from chulk.llm import LLMClient, LLMError, FallbackChain, LLMModelCapabilities
    from chulk.model_profiles.client import RefreshingLLMClient, RequestClientLease
    dispatched = []
    class Failing(LLMClient):
        provider = 'primary'
        model = 'fake'
        def complete(self, messages, **kwargs):
            dispatched.append(1)
            raise LLMError('unavailable', code='server_error', fallback_eligible=True)
    secondary = ScriptedLLMClient(_completion())
    client = RefreshingLLMClient(provider='fallback', model='fake', model_profile_id='profile',
        credential_ref=None, model_capabilities=LLMModelCapabilities(provider='fallback', model='fake', context_window_tokens=10000, default_response_reserve_tokens=1000),
        client_factory=lambda: RequestClientLease(FallbackChain([Failing(), secondary])))
    def boundary():
        if dispatched:
            raise RuntimeError('ownership lost')
    async def async_boundary():
        boundary()
    messages = [{'role': 'user', 'content': 'Work'}]
    with pytest.raises(RuntimeError, match='ownership lost'):
        if asynchronous:
            if purpose == 'action':
                await client.acomplete_action(messages, before_dispatch=async_boundary)
            elif purpose == 'response':
                await client.acomplete_response(messages, before_dispatch=async_boundary)
            else:
                async for _ in client.astream_final_answer(messages, before_dispatch=async_boundary):
                    pass
        elif purpose == 'action':
            client.complete_action(messages, before_dispatch=boundary)
        elif purpose == 'response':
            client.complete_response(messages, before_dispatch=boundary)
        else:
            list(client.stream_final_answer(messages, before_dispatch=boundary))
    assert dispatched == [1]
    assert secondary.call_log == ()
