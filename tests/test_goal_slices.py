"""Public, offline continuation and durable no-progress regressions."""

from datetime import datetime, timedelta, timezone

import pytest

from chulk import Agent, AgentConfig, AsyncAgent, GoalSliceLimits, RunStatus, Tool
from chulk.core.state import TurnState
from chulk.goals import GoalStore
from chulk.testing import ScriptedLLMClient
from chulk.usage import SQLiteUsageStore
from tests.test_goal_context import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reopen", [False, True])
async def test_twelve_operations_span_five_five_two_without_new_user_messages(tmp_path, asynchronous, reopen):
    service, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits()
    dispatched = []

    @Tool
    def inspect_item(index: int) -> str:
        """Inspect one item."""
        dispatched.append(index)
        return f"Result {index}"

    llm = ScriptedLLMClient([
        *[{"type": "tool_call", "tool_name": "inspect_item", "arguments": {"index": i}} for i in range(12)],
        {"type": "final_answer", "content": "Finished"},
    ])
    options = dict(config=AgentConfig(project_root=tmp_path / "runtime", store_path=tmp_path / "usage.sqlite", max_reflection_attempts=0),
                   llm=llm, tools=[inspect_item], skills=[], goal_execution=execution)
    cls = AsyncAgent if asynchronous else Agent
    agent = cls(**options)
    events = []
    result = await agent.run_result("Inspect twelve items", on_event=events.append) if asynchronous else agent.run_result("Inspect twelve items", on_event=events.append)
    assert result.status is RunStatus.YIELDED
    assert result.content == ""
    assert dispatched == list(range(5))
    assert events[-1].name == "run.yielded"
    assert not any(event.name == "run.completed" for event in events)
    conversation_id = agent.conversation_id
    for expected in (10, 12):
        if reopen:
            if asynchronous:
                await agent.close()
            else:
                agent.close()
            agent = cls(**options, conversation_id=conversation_id)
        result = await agent.continue_goal_slice() if asynchronous else agent.continue_goal_slice()
        assert dispatched == list(range(expected))
    assert result.status is RunStatus.COMPLETED
    assert result.content == "Finished"
    assert service.store.get(execution.goal_id).step("work").attempt == 1
    entries = SQLiteUsageStore(tmp_path / "usage.sqlite").list_entries()
    assert sum(int(item.units.get("tool_calls", 0)) for item in entries) == 12
    assert len(llm.call_log) == 13
    assert len([m for m in agent.runtime.memory.messages if m["role"] == "user"]) == 1
    if asynchronous:
        await agent.close()
    else:
        agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_yield_before_reflection_does_not_lose_answer_or_consume_reflection(tmp_path, asynchronous):
    _, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits(max_model_calls=1)
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "Known answer"}, {"approved": True, "reason": "Verified"}])
    agent = (AsyncAgent if asynchronous else Agent)(
        config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=1),
        llm=llm, tools=[], skills=[], goal_execution=execution,
    )
    agent.runtime._model_transport.max_json_repair_attempts = 0
    result = await agent.run_result("Work") if asynchronous else agent.run_result("Work")
    assert result.status is RunStatus.YIELDED
    assert agent.runtime.state.turns[-1].reflection_count == 0
    result = await agent.continue_goal_slice() if asynchronous else agent.continue_goal_slice()
    assert result.status is RunStatus.COMPLETED
    assert result.content == "Known answer"
    assert len(llm.call_log) == 2
    if asynchronous:
        await agent.close()
    else:
        agent.close()


def test_rejections_persist_across_store_reopen_and_duplicate_content(tmp_path):
    service, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits()
    for index in range(3):
        execution.store = GoalStore(service.store.db_path, clock=service.store.clock)
        context = execution.verification_context()
        count = execution.record_verification(TurnState("work", turn_id=f"turn-{index}"), context=context, passed=False, feedback="No test evidence")
        assert count == index + 1
    goal = service.store.get(execution.goal_id)
    assert goal.status.value == "blocked"
    assert "verification_stagnation" in goal.last_error


def test_changed_evidence_resets_rejections_but_duplicate_result_does_not(tmp_path):
    from chulk.tools import ToolResult
    _, execution = _execution(tmp_path)
    for index, content in enumerate(("a", "b", "b")):
        checkpoint = execution.begin_tool(turn_id=f"t-{index}", tool_call_index=1, attempt=1, tool_name="inspect")
        execution.finish_tool(checkpoint, ToolResult("inspect", True, content))
        count = execution.record_verification(TurnState("work", turn_id=f"t-{index}"), context=execution.verification_context(), passed=False, feedback="Missing proof")
        assert count == (1 if index < 2 else 2)


def test_impossible_fresh_slice_is_configuration_error_before_provider(tmp_path):
    from chulk import ConfigurationError
    _, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits(max_model_calls=1)
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "unused"}])
    with Agent(config=AgentConfig(project_root=tmp_path / "runtime"), llm=llm, tools=[], skills=[], goal_execution=execution) as agent:
        with pytest.raises(ConfigurationError, match="fresh goal slice"):
            agent.run_result("Work")
        assert llm.call_log == ()
        assert agent.runtime.state.turns[-1].model_request_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_three_verifier_rejections_block_across_slice_restarts(tmp_path, asynchronous):
    from chulk.core.plan_execution import PlanStepVerification
    service, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits(max_model_calls=1)
    plan = {"type": "plan", "plan": {"summary": "Implement the change", "steps": [
        {"id": "work", "title": "Implement", "description": "Implement the change in the owner and run regression tests"}
    ]}}
    rejected = {"type": "plan_step_update", "step_update": {"step_id": "work", "status": "completed", "evidence": "Done"}}
    llm = ScriptedLLMClient([plan, rejected, rejected, rejected])
    cls = AsyncAgent if asynchronous else Agent
    options = dict(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
                   llm=llm, tools=[], skills=[], goal_execution=execution,
                   plan_step_verifier=lambda _: PlanStepVerification(False, "Tests have not run"))
    agent = cls(**options)
    agent.runtime._model_transport.max_json_repair_attempts = 0
    if asynchronous:
        await agent.plan_result("Implement the change")
        result = await agent.approve_result()
    else:
        agent.plan_result("Implement the change")
        result = agent.approve_result()
    assert result.status is RunStatus.YIELDED
    conversation = agent.conversation_id
    for index in range(3):
        if asynchronous:
            await agent.close()
        else:
            agent.close()
        agent = cls(**options, conversation_id=conversation)
        agent.runtime._model_transport.max_json_repair_attempts = 0
        result = await agent.continue_goal_slice() if asynchronous else agent.continue_goal_slice()
        assert result.status is (RunStatus.BLOCKED if index == 2 else RunStatus.YIELDED)
    assert "verification_stagnation" in service.store.get(execution.goal_id).last_error
    assert llm.remaining == 0
    if asynchronous:
        await agent.close()
    else:
        agent.close()


def test_repair_requests_are_committed_as_provider_calls(tmp_path):
    _, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits(max_model_calls=3)
    llm = ScriptedLLMClient(["invalid", "still invalid", {"type": "final_answer", "content": "Repaired"}])
    path = tmp_path / "usage.sqlite"
    with Agent(config=AgentConfig(project_root=tmp_path / "runtime", store_path=path, max_reflection_attempts=0), llm=llm, tools=[], skills=[], goal_execution=execution) as agent:
        assert agent.run_result("Work").status is RunStatus.COMPLETED
    entries = SQLiteUsageStore(path).list_entries()
    assert sum(int(item.units.get("model_calls", 0)) for item in entries) == 3
    assert len(llm.call_log) == 3


def test_pending_summary_history_survives_refused_admission(tmp_path):
    _, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits(max_model_calls=1)
    llm = ScriptedLLMClient([])
    with Agent(config=AgentConfig(project_root=tmp_path / "runtime", history_limit=2), llm=llm, tools=[], skills=[], goal_execution=execution) as agent:
        memory = agent.runtime.memory
        for i in range(4):
            memory.add_user_message(f"Unsummarized history {i}")
        # A summary may consume the only admitted call; source must remain if admission fails.
        execution.slice_clock = lambda: datetime.now(timezone.utc) - timedelta(seconds=120)
        result = agent.run_result("Work")
        assert result.status is RunStatus.YIELDED
        assert memory.pending_summary_messages()
        assert llm.call_log == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_tool_retry_resumes_at_next_attempt(tmp_path, asynchronous):
    from chulk import ToolRetryPolicy
    _, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits(max_tool_calls=1)
    attempts = []

    @Tool(retry_policy=ToolRetryPolicy(max_attempts=3), idempotent=True)
    def flaky() -> str:
        """Return a result on the third retry."""
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise RuntimeError("Transient failure")
        return "Recovered"

    llm = ScriptedLLMClient([{"type": "tool_call", "tool_name": "flaky", "arguments": {}}, {"type": "final_answer", "content": "Complete"}])
    agent = (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0), llm=llm, tools=[flaky], skills=[], goal_execution=execution)
    result = await agent.run_result("Work") if asynchronous else agent.run_result("Work")
    assert result.status is RunStatus.YIELDED
    for index in (2, 3):
        result = await agent.continue_goal_slice() if asynchronous else agent.continue_goal_slice()
        assert attempts == list(range(1, index + 1))
    assert result.status is RunStatus.COMPLETED
    assert len(result.tool_calls[0].attempts) == 3
    assert len(llm.call_log) == 2
    if asynchronous:
        await agent.close()
    else:
        agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_final_stream_resumes_after_known_reflection(tmp_path, asynchronous):
    from chulk import FinalAnswerStreamingMode
    _, execution = _execution(tmp_path)
    execution.slice_limits = GoalSliceLimits(max_model_calls=1)
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "Draft"}, {"approved": True, "reason": "Accepted"}, "Final"])
    agent = (AsyncAgent if asynchronous else Agent)(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=1), llm=llm, tools=[], skills=[], goal_execution=execution, final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL)
    agent.runtime._model_transport.max_json_repair_attempts = 0
    result = await agent.run_result("Work") if asynchronous else agent.run_result("Work")
    assert result.status is RunStatus.YIELDED
    result = await agent.continue_goal_slice() if asynchronous else agent.continue_goal_slice()
    assert result.status is RunStatus.YIELDED
    assert agent.runtime.state.turns[-1].extension_metadata["goal_pending"]["phase"] == "reflection_result"
    result = await agent.continue_goal_slice() if asynchronous else agent.continue_goal_slice()
    assert result.status is RunStatus.COMPLETED
    assert result.content == "Final"
    assert len(llm.call_log) == 3
    if asynchronous:
        await agent.close()
    else:
        agent.close()


def test_verification_migration_rolls_back_and_reopens(tmp_path):
    import sqlite3
    from chulk.storage import initialize_sqlite_database, SQLiteMigration, SQLiteMigrationError
    from chulk.storage.migrations import SQLITE_MIGRATIONS
    path = tmp_path / "upgrade.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:22])
    def interrupted(conn):
        SQLITE_MIGRATIONS[22].apply(conn)
        raise RuntimeError("Power loss")
    with pytest.raises(SQLiteMigrationError):
        initialize_sqlite_database(path, migrations=(*SQLITE_MIGRATIONS[:22], SQLiteMigration(23, "interrupted", interrupted)))
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 22
        assert conn.execute("SELECT name FROM sqlite_schema WHERE name='goal_verifications'").fetchone() is None
    assert initialize_sqlite_database(path).to_version == 23
    assert initialize_sqlite_database(path).from_version == 23


def test_global_call_budget_survives_reopen(tmp_path):
    from chulk import BudgetExceededError
    from chulk.usage import RunBudget
    from dataclasses import replace
    service, execution = _execution(tmp_path)
    goal = service.store.get(execution.goal_id)
    service.store.mutate(goal.id, expected_revision=goal.revision, kind="goal.budget_changed", actor="owner", mutation=lambda current: replace(current, budget=RunBudget(scope="goal", max_model_calls=3)))
    execution.slice_limits = GoalSliceLimits(max_model_calls=2)
    @Tool
    def inspect() -> str:
        """Inspect evidence."""
        return "Evidence"
    llm = ScriptedLLMClient([{"type": "tool_call", "tool_name": "inspect", "arguments": {}}] * 5)
    options = dict(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0), llm=llm, tools=[inspect], skills=[], goal_execution=execution)
    with Agent(**options) as agent:
        agent.runtime._model_transport.max_json_repair_attempts = 0
        assert agent.run_result("Investigate").status is RunStatus.YIELDED
        conversation = agent.conversation_id
    with Agent(**options, conversation_id=conversation) as agent:
        agent.runtime._model_transport.max_json_repair_attempts = 0
        with pytest.raises(BudgetExceededError) as error:
            agent.continue_goal_slice()
        assert error.value.scope.value == "goal"
        assert error.value.dimension == "model_calls"
        assert len(llm.call_log) == 3
