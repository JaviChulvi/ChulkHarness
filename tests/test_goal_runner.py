"""Offline acceptance for the foreground goal coordinator."""
import pytest

from chulk import Agent, AsyncAgent, AgentConfig, Tool
from chulk.core.plan_execution import PlanStepVerification
from chulk.goals import GoalService, GoalStore, GoalStep
from chulk.goals.runner import GoalRunner, AsyncGoalRunner
from chulk.usage import RunBudget
from chulk.testing import ScriptedLLMClient


def _goal(tmp_path, *, count=1, budget=100):
    store = GoalStore(tmp_path / "state.sqlite")
    service = GoalService(store)
    goal = service.create(title="Implement", acceptance_criteria=tuple(f"Verify step {i}" for i in range(count)),
        steps=tuple(GoalStep(id=f"step-{i}", title=f"Step {i}", description=f"Implement and verify step {i}",
                            acceptance_criterion_ids=(f"criterion-{i+1}",), depends_on=(f"step-{i-1}",) if i else ()) for i in range(count)),
        budget=RunBudget(max_model_calls=budget))
    return service, service.approve(goal.id, expected_revision=goal.revision, approved_by="owner")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_runner_finishes_twelve_tools_through_bounded_slices(tmp_path, asynchronous):
    service, goal = _goal(tmp_path)
    calls = []
    @Tool
    def inspect_item(index: int) -> str:
        """Inspect one result."""
        calls.append(index)
        return f"Evidence {index}"
    llm = ScriptedLLMClient([
        *[{"type": "tool_call", "tool_name": "inspect_item", "arguments": {"index": i}} for i in range(12)],
        {"type": "plan_step_update", "step_update": {"step_id": "step-0", "status": "completed", "evidence": "Verified"}},
        {"type": "final_answer", "content": "Done"},
    ])
    config = AgentConfig(project_root=tmp_path, store_path=service.store.db_path, max_reflection_attempts=0)
    def factory(context, conversation_id):
        return (AsyncAgent if asynchronous else Agent)(config=config, llm=llm, tools=[inspect_item], skills=[],
            goal_execution=context, conversation_id=conversation_id)
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, agent_factory=factory,
        verifier=lambda request: PlanStepVerification(len(calls) == 12, "Twelve results verified"))
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == "completed"
    assert result.goal.status.value == "completed"
    assert result.goal.steps[0].attempt == 1
    assert calls == list(range(12))
    assert result.usage["tool_calls"] == 12
    assert len(result.goal.evidence) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_runner_respects_dependencies_and_single_slice_boundary(tmp_path, asynchronous):
    service, goal = _goal(tmp_path, count=2)
    llm = ScriptedLLMClient([
        *[{"type": "plan_step_update", "step_update": {"step_id": f"step-{i}", "status": "completed", "evidence": "Done"}}
          if j == 0 else {"type": "final_answer", "content": f"Done {i}"} for i in range(2) for j in range(2)]
    ])
    def factory(context, conversation_id):
        return (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path, store_path=service.store.db_path, max_reflection_attempts=0), llm=llm, tools=[], skills=[], goal_execution=context, conversation_id=conversation_id)
    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(service.store, agent_factory=factory, verifier=lambda _: PlanStepVerification(True, "Host verified"))
    first = await runner.run_slice(goal.id) if asynchronous else runner.run_slice(goal.id)
    assert first.stop_reason.value == "step_completed"
    assert first.goal.steps[1].attempt == 0
    second = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert second.stop_reason.value == "completed"
    assert second.conversation_id == first.conversation_id
    assert second.turn_id != first.turn_id
    assert [s.attempt for s in second.goal.steps] == [1, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["yielded", "completed", "rejected_then_completed"])
async def test_runner_slice_replays_projected_plan_without_original_tools(tmp_path, monkeypatch, asynchronous, outcome):
    from chulk import GoalSliceLimits
    from chulk.tools.registry import ToolRegistry
    from chulk.tracing import ReplayFixture, Trace
    from chulk.tracing.execution import execute_replay_fixture, execute_replay_fixture_async

    service, goal = _goal(tmp_path)
    calls, traces = [], []
    completed = outcome != "yielded"
    verification_calls = []

    @Tool
    def inspect_item(index: int) -> str:
        """Inspect one result."""
        calls.append(index)
        return f"Evidence {index}"

    actions = [
        {"type": "tool_call", "tool_name": "inspect_item", "arguments": {"index": index}}
        for index in range(2)
    ]
    if completed:
        completion, answer = _completion()
        actions = [completion, actions[0], answer]
        if outcome == "rejected_then_completed":
            actions.insert(0, completion)
    llm = ScriptedLLMClient(actions)

    def verifier(_request):
        verification_calls.append(1)
        passed = outcome != "rejected_then_completed" or len(verification_calls) > 1
        return PlanStepVerification(passed, "Verified" if passed else "Recheck the acceptance criteria")

    def factory(context, conversation_id):
        agent = (AsyncAgent if asynchronous else Agent)(
            config=AgentConfig(project_root=tmp_path, store_path=service.store.db_path, max_reflection_attempts=0),
            llm=llm, tools=[inspect_item], skills=[], goal_execution=context, conversation_id=conversation_id,
        )
        traces.append(agent.trace_path)
        return agent

    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(
        service.store, agent_factory=factory, verifier=verifier,
        slice_limits=GoalSliceLimits(max_tool_calls=1),
    )
    result = await runner.run_slice(goal.id) if asynchronous else runner.run_slice(goal.id)
    expected_status = "completed" if completed else "yielded"
    assert result.stop_reason.value == expected_status
    fixture = ReplayFixture.from_trace(Trace.from_jsonl(traces[0]), acknowledge_sensitive_data=True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Replay must not invoke original tools or access the goal store")

    monkeypatch.setattr(ToolRegistry, "run", forbidden)
    monkeypatch.setattr(ToolRegistry, "run_async", forbidden)
    monkeypatch.setattr(GoalStore, "get", forbidden)
    for report in (execute_replay_fixture(fixture), await execute_replay_fixture_async(fixture)):
        assert report.ok, report.mismatches
        assert report.actual["result"]["status"] == expected_status
        assert report.tool_results_consumed == (0 if completed else 1)
    assert calls == ([] if completed else [0])
    assert len(verification_calls) == (2 if outcome == "rejected_then_completed" else int(completed))


def _runner(tmp_path, service, responses, *, tools=(), verifier=None, **kwargs):
    llm = ScriptedLLMClient(responses)
    def factory(context, conversation_id):
        return Agent(config=AgentConfig(project_root=tmp_path, store_path=service.store.db_path, max_reflection_attempts=0),
                     llm=llm, tools=tools, skills=[], goal_execution=context, conversation_id=conversation_id)
    return GoalRunner(service.store, agent_factory=factory,
                      verifier=verifier or (lambda _: PlanStepVerification(True, "Host verified")), **kwargs), llm


def _completion():
    return [{"type": "plan_step_update", "step_update": {"step_id": "step-0", "status": "completed", "evidence": "Done"}},
            {"type": "final_answer", "content": "Done"}]


def test_missing_verifier_and_unbounded_budget_never_build_agent(tmp_path):
    from chulk import ConfigurationError
    service, goal = _goal(tmp_path)
    def forbidden(*args):
        pytest.fail("Factory must not run without admission prerequisites")
    runner = GoalRunner(service.store, agent_factory=forbidden)
    with pytest.raises(ConfigurationError, match="verifier"):
        runner.run(goal.id)
    assert service.store.get(goal.id) == goal
    changed = service.update_budget(goal.id, expected_revision=goal.revision, budget=RunBudget(), actor="owner")
    runner.verifier = lambda _: PlanStepVerification(True, "ok")
    with pytest.raises(ConfigurationError, match="finite"):
        runner.run(changed.id)


def test_pause_during_tool_preserves_result_and_can_resume_without_retry(tmp_path):
    service, goal = _goal(tmp_path)
    calls = []
    @Tool
    def pause_after_read() -> str:
        """Read a value and allow the operator to pause."""
        calls.append(1)
        current = service.store.get(goal.id)
        service.pause(goal.id, expected_revision=current.revision, actor="owner")
        return "Known result"
    runner, _ = _runner(tmp_path, service, [{"type": "tool_call", "tool_name": "pause_after_read", "arguments": {}}, *_completion()], tools=[pause_after_read])
    result = runner.run(goal.id)
    assert result.stop_reason.value == "paused"
    assert len(service.store.action_checkpoints(goal.id)) == 1
    service.resume(goal.id, expected_revision=result.goal.revision, actor="owner")
    done = runner.run(goal.id)
    assert done.stop_reason.value == "completed"
    assert done.goal.steps[0].attempt == 1
    assert calls == [1]


def test_cancellation_during_tool_preserves_known_result_and_stops_dispatch(tmp_path):
    service, goal = _goal(tmp_path)
    calls = []
    @Tool
    def cancel_after_read() -> str:
        """Read a value while cancellation is requested."""
        calls.append(1)
        current = service.store.get(goal.id)
        service.request_cancel(goal.id, expected_revision=current.revision, actor="owner")
        return "Known result"
    runner, _ = _runner(tmp_path, service, [{"type": "tool_call", "tool_name": "cancel_after_read", "arguments": {}}, *_completion()], tools=[cancel_after_read])
    result = runner.run(goal.id)
    assert result.stop_reason.value == "cancelled"
    assert result.usage["tool_calls"] == 1
    assert service.store.execution_state(goal.id)["stop_reason"] == "cancelled"
    assert service.store.action_checkpoints(goal.id)[0].result is not None
    assert calls == [1]


def test_selected_step_approval_does_not_override_later_pause(tmp_path):
    from dataclasses import replace
    from chulk.goals import GoalRisk
    service, goal = _goal(tmp_path)
    goal = service.store.mutate(goal.id, expected_revision=goal.revision, kind="goal.step_risk", actor="owner",
        mutation=lambda current: replace(current, steps=(replace(current.steps[0], risk=GoalRisk.HIGH),)))
    runner, _ = _runner(tmp_path, service, _completion())
    waiting = runner.run(goal.id)
    assert waiting.stop_reason.value == "approval_required"
    assert waiting.goal.steps[0].attempt == 0
    paused = service.pause(goal.id, expected_revision=waiting.goal.revision, actor="owner")
    approved = service.approve_step(goal.id, "step-0", expected_revision=paused.revision, approved_by="owner")
    assert runner.run(goal.id).stop_reason.value == "paused"
    service.resume(goal.id, expected_revision=approved.revision, actor="owner")
    assert runner.run(goal.id).stop_reason.value == "completed"


def test_goal_budget_exhaustion_requires_explicit_change_and_keeps_usage(tmp_path):
    service, goal = _goal(tmp_path, budget=1)
    runner, _ = _runner(tmp_path, service, _completion())
    stopped = runner.run(goal.id)
    assert stopped.stop_reason.value == "budget_exhausted"
    assert runner.run(goal.id).stop_reason.value == "budget_exhausted"
    updated = service.update_budget(goal.id, expected_revision=stopped.goal.revision,
                                    budget=RunBudget(max_model_calls=10), actor="owner")
    assert updated.status.value == "paused"
    service.resume(goal.id, expected_revision=updated.revision, actor="owner")
    result = runner.run(goal.id)
    assert result.stop_reason.value == "completed"
    assert result.goal.steps[0].attempt == 1
    assert result.usage["model_calls"] == 2


def test_claim_expiry_during_tool_cannot_dispatch_or_commit_progress(tmp_path):
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    service, goal = _goal(tmp_path)
    service.store.clock = lambda: now
    calls = []
    @Tool
    def lose_lease() -> str:
        """Read a value after which this lease expires."""
        nonlocal now
        calls.append(1)
        now += timedelta(seconds=121)
        return "Result arrived after lease expiry"
    runner, _ = _runner(tmp_path, service, [{"type": "tool_call", "tool_name": "lose_lease", "arguments": {}}, *_completion()], tools=[lose_lease], clock=lambda: now)
    result = runner.run(goal.id)
    assert result.stop_reason.value == "lease_lost"
    assert result.goal.steps[0].status.value == "running"
    assert not result.goal.evidence
    assert calls == [1]
    # The durable-run claim is still live even though the goal clock advanced.
    assert runner.run(goal.id).stop_reason.value == "lease_lost"


def test_concurrent_claim_and_stale_progress_are_rejected(tmp_path):
    from chulk.goals import GoalLeaseConflictError, GoalRevisionConflictError
    service, goal = _goal(tmp_path)
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    first = service.store.admit_slice(goal.id, runner_id="first", expected_revision=goal.revision, turn_id="first")
    with pytest.raises(GoalLeaseConflictError):
        service.store.admit_slice(goal.id, runner_id="second", expected_revision=first.goal.revision, turn_id="second")
    service.store.record_verification(first.claim, step_id="step-0", operation_id="verification", expected_revision=first.goal.revision,
                                     context_digest="context", evidence_digest="evidence", passed=True, feedback="Verified")
    steered = service.steer(goal.id, expected_revision=first.goal.revision, instruction="New criterion", created_by="owner")
    with pytest.raises(GoalRevisionConflictError):
        service.store.apply_verified_step(first.claim, operation_id="verification", expected_revision=steered.revision)
    assert not service.store.get(goal.id).evidence


def test_verifier_receives_current_instructions_and_goal(tmp_path):
    service, goal = _goal(tmp_path)
    def verifier(request):
        assert request.goal.id == goal.id
        assert request.acceptance_criteria == ("Verify step 0",)
        assert request.goal.active_steering[0].instruction == "Retain the API"
        return PlanStepVerification(True, "Goal and instructions verified")
    service.steer(goal.id, expected_revision=goal.revision, instruction="Retain the API", created_by="owner")
    runner, _ = _runner(tmp_path, service, _completion(), verifier=verifier)
    assert runner.run(goal.id).stop_reason.value == "completed"


@pytest.mark.asyncio
async def test_native_async_store_admits_without_sync_calls(tmp_path):
    import asyncio
    from dataclasses import replace
    from chulk.goals import GoalRisk
    from chulk.hosting.async_utils import call_async_service
    service, goal = _goal(tmp_path)
    goal = service.store.mutate(goal.id, expected_revision=goal.revision, kind="goal.step_risk", actor="owner",
        mutation=lambda current: replace(current, steps=(replace(current.steps[0], risk=GoalRisk.HIGH),)))
    loop = asyncio.get_running_loop()
    called = []
    class NativeStore:
        profile_id = service.store.profile_id
        def __getattr__(self, name):
            async def method(*args, **kwargs):
                assert asyncio.get_running_loop() is loop
                called.append(name)
                return await call_async_service(service.store, name, *args, **kwargs)
            return method
    def forbidden(*args):
        pytest.fail("No worker is allocated while waiting for selected-step approval")
    runner = AsyncGoalRunner(NativeStore(), agent_factory=forbidden,
                             async_verifier=lambda _: None)
    assert (await runner.run(goal.id)).stop_reason.value == "approval_required"
    assert called == ["get", "execution_state", "mutate", "admit_slice"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_claim_is_renewed_while_verifier_waits(tmp_path, asynchronous):
    import threading
    from datetime import datetime, timezone, timedelta
    service, goal = _goal(tmp_path)
    now = datetime.now(timezone.utc)
    service.store.clock = lambda: now
    renewed = threading.Event()
    original = service.store.heartbeat
    def heartbeat(*args, **kwargs):
        result = original(*args, **kwargs)
        renewed.set()
        return result
    service.store.heartbeat = heartbeat
    def verifier(request):
        nonlocal now
        now += timedelta(seconds=61)
        assert renewed.wait(5)
        return PlanStepVerification(True, "Verified after a wait")
    runner, _ = _runner(tmp_path, service, _completion(), verifier=verifier, renewal_seconds=.01, clock=lambda: now)
    if asynchronous:
        runner = AsyncGoalRunner(service.store, agent_factory=runner.agent_factory, verifier=verifier, renewal_seconds=.01, clock=lambda: now)
        result = await runner.run(goal.id)
    else:
        result = runner.run(goal.id)
    assert result.stop_reason.value == "completed"
    assert renewed.is_set()


def test_runner_migration_upgrade_rollback_reopen_and_future_rejection(tmp_path):
    import sqlite3
    from chulk.storage import initialize_sqlite_database, SQLiteMigration, SQLiteMigrationError, UnsupportedSQLiteSchemaVersionError
    from chulk.storage.migrations import SQLITE_MIGRATIONS
    path = tmp_path / "old.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:23])
    def interrupted(conn):
        SQLITE_MIGRATIONS[23].apply(conn)
        raise RuntimeError("Interrupted migration")
    with pytest.raises(SQLiteMigrationError):
        initialize_sqlite_database(path, migrations=(*SQLITE_MIGRATIONS[:23], SQLiteMigration(24, "interrupted", interrupted)))
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 23
        assert not conn.execute("SELECT name FROM sqlite_schema WHERE name='goal_slices'").fetchone()
    assert initialize_sqlite_database(path).to_version == SQLITE_MIGRATIONS[-1].version
    assert initialize_sqlite_database(path).backup_path is None
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 999")
    with pytest.raises(UnsupportedSQLiteSchemaVersionError):
        initialize_sqlite_database(path)


def test_verification_survives_yield_before_final_answer(tmp_path):
    from chulk import GoalSliceLimits
    service, goal = _goal(tmp_path)
    calls = []
    runner, _ = _runner(tmp_path, service, _completion(), slice_limits=GoalSliceLimits(max_model_calls=3),
                        verifier=lambda request: (calls.append(request) or PlanStepVerification(True, "Verified")))
    first = runner.run_slice(goal.id)
    assert first.stop_reason.value == "yielded"
    assert first.goal.steps[0].status.value == "running"
    paused = service.pause(goal.id, expected_revision=first.goal.revision, actor="owner")
    service.resume(goal.id, expected_revision=paused.revision, actor="owner")
    final = runner.run(goal.id)
    assert final.stop_reason.value == "completed"
    assert len(calls) == 1
    assert final.goal.steps[0].attempt == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
async def test_verified_goal_rejects_later_tools_across_slices(tmp_path, asynchronous, repeated):
    from chulk import GoalSliceLimits
    service, goal = _goal(tmp_path)
    calls = []
    verified = []

    @Tool
    def invalidate_result() -> str:
        """Change the previously verified result."""
        calls.append("invalidated")
        return "Result is now invalid"

    tool_action = {"type": "tool_call", "tool_name": "invalidate_result", "arguments": {}}
    completion, answer = _completion()
    llm = ScriptedLLMClient([completion, tool_action, tool_action if repeated else answer])

    def factory(context, conversation_id):
        return (AsyncAgent if asynchronous else Agent)(
            config=AgentConfig(project_root=tmp_path, store_path=service.store.db_path, max_reflection_attempts=0),
            llm=llm, tools=[invalidate_result], skills=[], goal_execution=context, conversation_id=conversation_id,
        )

    def verifier(request):
        verified.append(request)
        return PlanStepVerification(not calls, "Current result verified")

    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(
        service.store, agent_factory=factory, verifier=verifier, slice_limits=GoalSliceLimits(max_model_calls=3),
    )
    first = await runner.run_slice(goal.id) if asynchronous else runner.run_slice(goal.id)
    assert first.stop_reason.value == "yielded"
    result = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert result.stop_reason.value == ("blocked" if repeated else "completed")
    assert len(verified) == 1
    assert calls == []
    assert result.usage["tool_calls"] == 0
    assert result.goal.steps[0].attempt == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_changed_budget_pauses_and_drains_before_resuming(tmp_path, asynchronous):
    from chulk.goals import GoalLeaseConflictError
    service, goal = _goal(tmp_path)

    @Tool
    def lower_budget() -> str:
        """Record a result while the operator changes the budget."""
        current = service.store.get(goal.id)
        paused = service.update_budget(goal.id, expected_revision=current.revision,
                                       budget=RunBudget(max_model_calls=1), actor="owner")
        assert paused.status.value == "paused"
        for restart in (service.run, service.resume):
            with pytest.raises(GoalLeaseConflictError, match="drain"):
                restart(goal.id, expected_revision=paused.revision, actor="owner")
        return "Known result before budget change"

    llm = ScriptedLLMClient([{"type": "tool_call", "tool_name": "lower_budget", "arguments": {}}, *_completion()])

    def factory(context, conversation_id):
        return (AsyncAgent if asynchronous else Agent)(
            config=AgentConfig(project_root=tmp_path, store_path=service.store.db_path, max_reflection_attempts=0),
            llm=llm, tools=[lower_budget], skills=[], goal_execution=context, conversation_id=conversation_id,
        )

    runner = (AsyncGoalRunner if asynchronous else GoalRunner)(
        service.store, agent_factory=factory, verifier=lambda _: PlanStepVerification(True, "Verified"),
    )
    paused = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert paused.stop_reason.value == "paused"
    assert paused.usage["model_calls"] == 1
    assert len(llm.call_log) == 1
    assert service.store.action_checkpoints(goal.id)[0].result is not None
    service.resume(goal.id, expected_revision=paused.goal.revision, actor="owner")
    exhausted = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert exhausted.stop_reason.value == "budget_exhausted"
    assert len(llm.call_log) == 1
    updated = service.update_budget(goal.id, expected_revision=exhausted.goal.revision,
                                    budget=RunBudget(max_model_calls=10), actor="owner")
    service.resume(goal.id, expected_revision=updated.revision, actor="owner")
    completed = await runner.run(goal.id) if asynchronous else runner.run(goal.id)
    assert completed.stop_reason.value == "completed"
    assert completed.usage["model_calls"] == 3
    assert completed.usage["tool_calls"] == 1
    assert completed.goal.steps[0].attempt == 1


def test_budget_noop_and_paused_update_preserve_operator_state(tmp_path):
    service, goal = _goal(tmp_path)
    running = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    unchanged = service.update_budget(goal.id, expected_revision=running.revision,
                                      budget=running.budget, actor="owner")
    assert unchanged == running
    paused = service.pause(goal.id, expected_revision=unchanged.revision, actor="owner")
    changed = service.update_budget(goal.id, expected_revision=paused.revision,
                                    budget=RunBudget(max_model_calls=10), actor="owner")
    assert changed.status.value == "paused"
    assert changed.approvals == paused.approvals


def test_runner_reports_required_context_overflow_without_provider_calls(tmp_path):
    from dataclasses import replace
    from chulk.core.context import ContextBudget
    service, goal = _goal(tmp_path)
    service.store.mutate(goal.id, expected_revision=goal.revision, kind="goal.constraints", actor="owner",
                         mutation=lambda current: replace(current, constraints=("Required " + "x" * 20000,)))
    runner, llm = _runner(tmp_path, service, [])
    original = runner.agent_factory
    def factory(*args):
        agent = original(*args)
        agent.runtime._model_transport.context_budget = ContextBudget(max_prompt_tokens=1000, response_reserve_tokens=0)
        return agent
    runner.agent_factory = factory
    result = runner.run(goal.id)
    assert result.stop_reason.value == "required_context_overflow"
    assert llm.call_log == ()
    assert service.store.model_requests(goal.id) == ()
