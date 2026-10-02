"""Durable goal context through local and host-owned runtime boundaries."""

from datetime import datetime, timedelta, timezone
from html import unescape
import json
from pathlib import Path
import sqlite3

import pytest

from chulk import (
    Agent, AgentConfig, AsyncAgent, AsyncHostedRuntime, ChulkError,
    ConfigurationError, ExecutionScope, FinalAnswerStreamingMode, HostedRuntime, Tool,
)
from chulk.core.context import ContextBudget
from chulk.goals import (
    GoalActionConflictError, GoalLeaseConflictError, GoalRevisionConflictError,
    GoalService, GoalStep, GoalStore,
    InvalidGoalTransitionError, goal_from_dict,
)
from chulk.hosting.reference import InMemoryServiceHub
from chulk.llm import LLMStreamChunk
from chulk.storage import (
    SQLiteMigration, SQLiteMigrationError, UnsupportedSQLiteSchemaVersionError,
    initialize_sqlite_database, sqlite_connection,
)
from chulk.storage.migrations import SQLITE_MIGRATIONS, SQLITE_SCHEMA_VERSION
from chulk.testing import ScriptedLLMClient
from chulk.usage import RunBudget


NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _execution(tmp_path: Path, *, constraints: tuple[str, ...] = ("Never publish",)):
    service = GoalService(GoalStore(tmp_path / "goals.sqlite", clock=lambda: NOW), clock=lambda: NOW)
    goal = service.create(
        title="Review", description="Review the actual change",
        constraints=constraints, acceptance_criteria=("Tests demonstrate success",),
        steps=(GoalStep(id="work", title="Review", description="Inspect and verify",
                        acceptance_criterion_ids=("criterion-1",)),),
        budget=RunBudget(max_model_calls=100),
    )
    goal = service.approve(goal.id, expected_revision=goal.revision, approved_by="owner")
    goal = service.run(goal.id, expected_revision=goal.revision, actor="owner")
    goal = service.start_step(goal.id, "work", expected_revision=goal.revision, actor="runner")
    return service, service.claim_execution(goal.id, "work", expected_revision=goal.revision, runner_id="runner")


def _steer(service, execution, instruction="STEERING_SENTINEL", **kwargs):
    goal = service.store.get(execution.goal_id)
    return service.steer(goal.id, expected_revision=goal.revision, instruction=instruction,
                         created_by="owner", **kwargs)


def _context(call):
    text = call["messages"][0]["content"]
    assert text.count("<goal_context>") == 1
    payload = text.split("<goal_context>", 1)[1].split("</goal_context>", 1)[0]
    return json.loads(unescape(payload[payload.index("{"):]))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async", "hosted", "async_hosted"])
async def test_live_context_and_durable_receipt_have_sync_async_hosted_parity(tmp_path, mode):
    service, execution = _execution(tmp_path)
    goal = _steer(service, execution)
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "Reviewed"}] * 2)
    options = dict(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
                   llm=llm, tools=[], skills=[], goal_execution=execution)
    hub = InMemoryServiceHub()
    scope = ExecutionScope(tenant_id="tenant", workspace_id="workspace", actor_id="default",
                           agent_id="review", agent_version="1", run_id="run")
    if mode == "async_hosted":
        agent = await AsyncHostedRuntime.create(**options, services=hub.async_services(), execution_scope=scope)
    elif mode == "hosted":
        agent = HostedRuntime(**options, services=hub.services(), execution_scope=scope)
    else:
        agent = (AsyncAgent if mode == "async" else Agent)(**options)
    for index in range(2):
        if mode in {"async", "async_hosted"}:
            result = await agent.run_result("Continue")
        else:
            result = agent.run_result("Continue")
        assert result.status.value == "completed"
        context = _context(llm.call_log[index])
        assert context["description"] == goal.description
        assert context["constraints"] == ["Never publish"]
        assert context["acceptance_criteria"][0]["description"] == "Tests demonstrate success"
        assert context["active_step"]["id"] == "work"
        assert context["instructions"] == [{"id": goal.steering[0].id,
                                            "instruction": "STEERING_SENTINEL",
                                            "incorporated": bool(index), "fulfilled": False,
                                            "fulfillment_evidence_ids": []}]
    reopened = GoalStore(service.store.db_path, clock=lambda: NOW)
    receipts = reopened.model_requests(goal.id)
    assert len(receipts) == 2
    assert {item.goal_revision for item in receipts} == {goal.revision}
    assert all(item.response_ref and item.incorporated_at for item in receipts)
    assert reopened.incorporated_steering_ids(goal.id) == {goal.steering[0].id}
    assert reopened.get(goal.id).evidence == ()
    assert reopened.get(goal.id).status.value == "running"
    if mode in {"async", "async_hosted"}:
        await agent.close()
    else:
        agent.close()
    execution.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_steering_during_tool_is_read_before_the_next_request(tmp_path, asynchronous):
    service, execution = _execution(tmp_path)

    @Tool
    def investigate() -> str:
        """Inspect a source and receive an operator correction."""
        _steer(service, execution, "Use the corrected scope")
        return "New observation"

    llm = ScriptedLLMClient([{"type": "tool_call", "tool_name": "investigate", "arguments": {}},
                            {"type": "final_answer", "content": "Reviewed"}])
    agent = (AsyncAgent if asynchronous else Agent)(
        config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
        llm=llm, tools=[investigate], skills=[], goal_execution=execution,
    )
    if asynchronous:
        await agent.run_result("Investigate")
    else:
        agent.run_result("Investigate")
    assert _context(llm.call_log[0])["instructions"] == []
    assert _context(llm.call_log[1])["instructions"][0]["instruction"] == "Use the corrected scope"
    if asynchronous:
        await agent.close()
    else:
        agent.close()


def test_explicit_supersession_preserves_history_and_legacy_defaults(tmp_path):
    service, execution = _execution(tmp_path)
    first = _steer(service, execution, "Use path A")
    replacement = _steer(service, execution, "Use path B", supersedes=(first.steering[0].id,))
    assert len(replacement.steering) == 2
    assert replacement.active_steering == (replacement.steering[-1],)
    restored = goal_from_dict(replacement.to_dict())
    assert restored == replacement
    legacy = replacement.to_dict()
    del legacy["description"], legacy["constraints"]
    for item in legacy["steering"]:
        item.pop("supersedes")
    assert goal_from_dict(legacy).description == replacement.title
    assert len(goal_from_dict(legacy).active_steering) == 2
    with pytest.raises(InvalidGoalTransitionError, match="active"):
        _steer(service, execution, supersedes=(first.steering[0].id,))
    context = execution.context()
    assert [item["instruction"] for item in context["instructions"]] == ["Use path B"]


def test_fulfillment_requires_host_recorded_evidence_and_does_not_retire_instruction(tmp_path):
    service, execution = _execution(tmp_path)
    goal = _steer(service, execution)
    instruction_id = goal.steering[-1].id
    with pytest.raises(InvalidGoalTransitionError, match="recorded evidence"):
        service.fulfill_steering(goal.id, instruction_id, expected_revision=goal.revision,
                                evidence_ids=(), verified_by="host")
    goal = service.add_evidence(goal.id, expected_revision=goal.revision,
                                summary="Host checked the corrected scope", criterion_ids=("criterion-1",),
                                step_id="work", reference="artifact:verified-scope", recorded_by="host")
    goal = service.fulfill_steering(goal.id, instruction_id, expected_revision=goal.revision,
                                  evidence_ids=(goal.evidence[-1].id,), verified_by="host")
    context = execution.context()["instructions"][0]
    assert context["fulfilled"] is True
    assert context["incorporated"] is False
    assert context["instruction"] == "STEERING_SENTINEL"
    assert context["fulfillment_evidence_ids"] == [goal.evidence[-1].id]
    assert goal_from_dict(goal.to_dict()) == goal


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_failed_response_persistence_leaves_steering_pending_on_reopen(tmp_path, monkeypatch, asynchronous):
    service, execution = _execution(tmp_path)
    goal = _steer(service, execution)
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "Observed"}])
    agent = (AsyncAgent if asynchronous else Agent)(
        config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
        llm=llm, tools=[], skills=[], goal_execution=execution,
    )
    def crash(*_args, **_kwargs):
        raise RuntimeError("response persistence interrupted")
    monkeypatch.setattr(agent.runtime._components.session_store, "save_model_response", crash)
    with pytest.raises(ChulkError, match="response persistence interrupted"):
        if asynchronous:
            await agent.run_result("Continue")
        else:
            agent.run_result("Continue")
    reopened = GoalStore(service.store.db_path, clock=lambda: NOW)
    assert reopened.incorporated_steering_ids(goal.id) == frozenset()
    assert len(reopened.model_requests(goal.id, pending_only=True)) == 1
    assert execution.context()["instructions"][0]["incorporated"] is False


def test_durable_response_can_be_reconciled_idempotently_after_cursor_interruption(tmp_path, monkeypatch):
    service, execution = _execution(tmp_path)
    goal = _steer(service, execution)
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "Observed"}])
    agent = Agent(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
                  llm=llm, tools=[], skills=[], goal_execution=execution)
    original = service.store.acknowledge_model_response
    def crash(*_args, **_kwargs):
        raise RuntimeError("incorporation write interrupted")
    monkeypatch.setattr(service.store, "acknowledge_model_response", crash)
    with pytest.raises(ChulkError, match="incorporation write interrupted"):
        agent.run_result("Continue")
    receipt = service.store.model_requests(goal.id, pending_only=True)[0]
    with agent.runtime._components.session_store._connect() as conn:
        response = conn.execute("SELECT raw_response FROM conversation_model_requests WHERE turn_id = ?",
                                (receipt.turn_id,)).fetchone()
    assert "Observed" in response["raw_response"]
    reference = f"model-response:{receipt.conversation_id}:{receipt.turn_id}:{receipt.request_index}"
    monkeypatch.setattr(service.store, "acknowledge_model_response", original)
    execution.close()
    execution = service.claim_execution(goal.id, "work", expected_revision=goal.revision, runner_id="restarted")
    acknowledged = execution.acknowledge_response(receipt.id, response_ref=reference)
    assert execution.acknowledge_response(receipt.id, response_ref=reference) == acknowledged
    assert len(llm.call_log) == 1
    assert execution.context()["instructions"][0]["incorporated"] is True
    assert service.store.get(goal.id).evidence == ()
    with pytest.raises(GoalActionConflictError, match="reference changed"):
        execution.acknowledge_response(receipt.id, response_ref="different")
    agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async_hosted"])
async def test_oversized_mandatory_context_stops_before_any_provider_request(tmp_path, mode):
    service, execution = _execution(tmp_path, constraints=("Required constraint " + "x" * 20_000,))
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "unreachable"}])
    options = dict(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
                   llm=llm, tools=[], skills=[], goal_execution=execution)
    if mode == "async_hosted":
        scope = ExecutionScope(tenant_id="tenant", workspace_id="workspace", actor_id="default",
                               agent_id="review", agent_version="1", run_id="run")
        agent = await AsyncHostedRuntime.create(**options, services=InMemoryServiceHub().async_services(), execution_scope=scope)
    else:
        agent = Agent(**options)
    agent.runtime._model_transport.context_budget = ContextBudget(max_prompt_tokens=1000, response_reserve_tokens=0)
    with pytest.raises(ConfigurationError) as error:
        if mode == "async_hosted":
            await agent.run_result("Continue")
        else:
            agent.run_result("Continue")
    assert error.value.details.failure_kind == "context_budget_exceeded"
    assert llm.call_log == ()
    assert service.store.model_requests(execution.goal_id) == ()
    if mode == "async_hosted":
        await agent.close()
    else:
        agent.close()


def test_v21_upgrade_backfills_only_goal_state_and_reopens(tmp_path):
    path = tmp_path / "legacy.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:21])
    service, execution = _execution(tmp_path / "source")
    snapshot = service.store.get(execution.goal_id).to_dict()
    del snapshot["description"], snapshot["constraints"]
    with sqlite_connection(path) as conn:
        conn.execute("""INSERT INTO goals (id, profile_id, title, status, revision, snapshot_json,
                                          created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                     (snapshot["id"], "default", snapshot["title"], "running", snapshot["revision"],
                      json.dumps(snapshot), NOW.isoformat(), NOW.isoformat()))
    report = initialize_sqlite_database(path)
    assert report.from_version == 21
    assert report.to_version == SQLITE_SCHEMA_VERSION
    assert report.backup_path is not None
    assert GoalStore(path).get(snapshot["id"]).description == snapshot["title"]
    assert GoalStore(path).get(snapshot["id"]).constraints == ()
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SQLITE_SCHEMA_VERSION


def test_successive_compactions_preserve_all_mandatory_state(tmp_path):
    constraints = tuple(f"Constraint {index}: " + "c" * 120 for index in range(12))
    service, execution = _execution(tmp_path, constraints=constraints)
    goal = _steer(service, execution, "Keep the accepted scope across every compaction")
    summary = {"objective": "Historical discussion", "constraints": ["An obsolete historical rule"]}
    llm = ScriptedLLMClient([summary, {"type": "final_answer", "content": "Continued"}] * 3)
    agent = Agent(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
                  llm=llm, tools=[], skills=[], goal_execution=execution)
    agent.runtime._model_transport.context_budget = ContextBudget(max_prompt_tokens=2600, response_reserve_tokens=0)
    for _ in range(3):
        agent.runtime.memory.add_user_message("Historical source " + "a" * 10_000)
        agent.runtime.memory.add_assistant_message("Historical response " + "b" * 10_000)
        agent.run_result("Continue current work")
        context = _context(llm.call_log[-1])
        assert context["constraints"] == list(constraints)
        assert context["description"] == goal.description
        assert context["acceptance_criteria"] == [item.to_dict() for item in goal.acceptance_criteria]
        assert context["instructions"][0]["instruction"] == goal.steering[0].instruction
    assert len(llm.call_log) == 6
    assert len(service.store.model_requests(goal.id)) == 3
    agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_hosted_pause_control_is_independent_of_usage_service(tmp_path, asynchronous):
    service, execution = _execution(tmp_path)
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "unreachable"}])
    scope = ExecutionScope(tenant_id="tenant", workspace_id="workspace", actor_id="default",
                           agent_id="review", agent_version="1", run_id="run")
    options = dict(config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
                   llm=llm, tools=[], skills=[], goal_execution=execution, execution_scope=scope)
    hub = InMemoryServiceHub()
    if asynchronous:
        agent = await AsyncHostedRuntime.create(**options, services=hub.async_services())
    else:
        agent = HostedRuntime(**options, services=hub.services())
    goal = service.store.get(execution.goal_id)
    service.pause(goal.id, expected_revision=goal.revision, actor="owner")
    with pytest.raises(ChulkError, match="not running"):
        if asynchronous:
            await agent.run_result("Continue")
        else:
            agent.run_result("Continue")
    assert llm.call_log == ()
    if asynchronous:
        await agent.close()
    else:
        agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_reflection_receives_live_context_and_acknowledges_durable_response(tmp_path, asynchronous):
    service, execution = _execution(tmp_path)
    _steer(service, execution, "Inspect the safety requirement")
    llm = ScriptedLLMClient([{"type": "final_answer", "content": "Reviewed"},
                            {"approved": True, "reason": "The evidence is consistent"}])
    agent = (AsyncAgent if asynchronous else Agent)(
        config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=1),
        llm=llm, tools=[], skills=[], goal_execution=execution,
    )
    if asynchronous:
        await agent.run_result("Continue")
    else:
        agent.run_result("Continue")
    assert _context(llm.call_log[-1])["instructions"][0]["instruction"] == "Inspect the safety requirement"
    receipts = service.store.model_requests(execution.goal_id)
    assert {item.purpose for item in receipts} == {"agent_action", "reflection"}
    assert all(item.response_ref for item in receipts)
    if asynchronous:
        await agent.close()
    else:
        agent.close()


def test_request_admission_rejects_stale_context_omission_duplicate_and_expired_claim(tmp_path):
    service, execution = _execution(tmp_path)
    context = execution.context()
    goal = _steer(service, execution)
    options = dict(conversation_id="conversation", turn_id="turn", request_index=1, purpose="agent_action")
    with pytest.raises(GoalRevisionConflictError):
        execution.begin_model_request(context=context, **options)
    context = execution.context()
    omitted = {**context, "instructions": []}
    with pytest.raises(GoalActionConflictError, match="omitted active steering"):
        execution.begin_model_request(context=omitted, **options)
    receipt = execution.begin_model_request(context=context, **options)
    assert receipt.steering_ids == (goal.steering[-1].id,)
    with pytest.raises(GoalActionConflictError, match="already exists"):
        execution.begin_model_request(context=context, **options)
    service.store.clock = lambda: NOW + timedelta(seconds=121)
    with pytest.raises(GoalLeaseConflictError, match="expired"):
        execution.acknowledge_response(receipt.id, response_ref="persisted-response")
    assert service.store.incorporated_steering_ids(goal.id) == frozenset()


def test_context_migration_failure_rolls_back_and_future_schema_is_rejected(tmp_path):
    path = tmp_path / "rollback.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:21])
    def fail_after_context_migration(conn):
        SQLITE_MIGRATIONS[21].apply(conn)
        raise RuntimeError("interrupted context migration")
    with pytest.raises(SQLiteMigrationError) as error:
        initialize_sqlite_database(path, migrations=(*SQLITE_MIGRATIONS[:21],
            SQLiteMigration(22, "interrupted-context", fail_after_context_migration)))
    assert error.value.backup_path is not None
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 21
        assert conn.execute("SELECT 1 FROM sqlite_schema WHERE name = 'goal_model_requests'").fetchone() is None
    assert initialize_sqlite_database(path).to_version == SQLITE_SCHEMA_VERSION
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 999")
    with pytest.raises(UnsupportedSQLiteSchemaVersionError):
        GoalStore(path)


def test_historical_goal_script_preserves_writer_transaction_and_rolls_back(tmp_path):
    path = tmp_path / "historical.sqlite"
    initialize_sqlite_database(path, migrations=SQLITE_MIGRATIONS[:14])
    def fail_after_goal_migration(conn):
        SQLITE_MIGRATIONS[14].apply(conn)
        assert conn.in_transaction
        raise RuntimeError("interrupted goal migration")
    with pytest.raises(SQLiteMigrationError):
        initialize_sqlite_database(path, migrations=(*SQLITE_MIGRATIONS[:14],
            SQLiteMigration(15, "interrupted-goals", fail_after_goal_migration)))
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 14
        assert conn.execute("SELECT 1 FROM sqlite_schema WHERE name = 'goals'").fetchone() is None
    assert initialize_sqlite_database(path).to_version == SQLITE_SCHEMA_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_incremental_answer_has_goal_context_and_separate_durable_receipt(tmp_path, asynchronous):
    service, execution = _execution(tmp_path)
    _steer(service, execution, "Include checked evidence")
    class StreamingClient(ScriptedLLMClient):
        def stream_final_answer(self, messages, **kwargs):
            self.stream_messages = messages
            yield LLMStreamChunk(type="text_delta", text="Checked")
            yield LLMStreamChunk(type="completed")

        async def astream_final_answer(self, messages, **kwargs):
            self.stream_messages = messages
            yield LLMStreamChunk(type="text_delta", text="Checked")
            yield LLMStreamChunk(type="completed")
    llm = StreamingClient([{"type": "final_answer", "content": "Draft"}])
    agent = (AsyncAgent if asynchronous else Agent)(
        config=AgentConfig(project_root=tmp_path / "runtime", max_reflection_attempts=0),
        llm=llm, tools=[], skills=[], goal_execution=execution,
        final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL,
    )
    result = await agent.run_result("Continue") if asynchronous else agent.run_result("Continue")
    assert result.content == "Checked"
    assert _context({"messages": llm.stream_messages})["instructions"][0]["instruction"] == "Include checked evidence"
    receipts = service.store.model_requests(execution.goal_id)
    assert {item.purpose for item in receipts} == {"agent_action", "incremental_final_answer"}
    assert all(item.response_ref for item in receipts)
    if asynchronous:
        await agent.close()
    else:
        agent.close()
