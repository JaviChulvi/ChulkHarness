from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from chulk.config import load_config
from chulk import Agent as PublicAgent
from chulk import AgentConfig, BudgetPayload, RunFailedPayload
from chulk.goals import (
    GoalLeaseConflictError,
    GoalService,
    GoalStep,
    GoalStore,
)
from chulk.llm import FallbackChain, LLMActionError, LLMClient, LLMError
from chulk.llm.pricing import estimate_cost
from chulk.llm.usage import LLMUsage
from tests.core_agent import create_runtime_agent as create_agent
from chulk.testing import ScriptedLLMClient
from chulk.tools import Tool, ToolResult
from chulk.usage import (
    BudgetExceededError,
    BudgetScope,
    ExactCost,
    ReservationState,
    RunBudget,
    SQLiteUsageStore,
    UsageDimensions,
    UsageGroupBy,
)


class OpenAIScriptedClient(ScriptedLLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"


class FailingOpenAIClient(LLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"

    def complete(self, messages, *, max_output_tokens=None):
        del messages, max_output_tokens
        raise LLMError(
            "provider unavailable",
            provider=self.provider,
            model=self.model,
            code="server_error",
            retryable=False,
            fallback_eligible=True,
        )


class AccountedFailureClient(LLMClient):
    provider = "openai"
    model = "gpt-4.1-mini"
    model_profile_id = "primary"

    def complete_action(self, messages, **kwargs):
        del messages, kwargs
        usage = LLMUsage(
            input_tokens=100,
            output_tokens=10,
            total_tokens=110,
            cache_miss_input_tokens=100,
        )
        raise LLMActionError(
            "invalid provider action",
            provider=self.provider,
            model=self.model,
            code="action_shape_error",
            fallback_eligible=True,
            usage=usage,
            cost=estimate_cost(self.provider, self.model, usage),
        )

    def complete(self, messages, *, max_output_tokens=None):
        raise AssertionError("complete_action should be used")


def _config(tmp_path: Path):
    return load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "openai",
            "CHULK_MODEL": "gpt-4.1-mini",
        }
    )


def test_runtime_ingests_each_model_request_into_the_durable_ledger(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = OpenAIScriptedClient([{"type": "final_answer", "content": "done"}])
    agent = create_agent(
        config,
        llm_client=client,
        usage_dimensions=UsageDimensions(
            profile_id="default",
            channel="sdk",
        ),
    )

    assert agent.run_turn("hello") == "done"

    store = SQLiteUsageStore(config.store_path)
    entries = store.list_entries()
    reservations = store.list_reservations()
    assert len(entries) == 1
    assert entries[0].provider == "openai"
    assert entries[0].model == "gpt-4.1-mini"
    assert entries[0].dimensions.channel == "sdk"
    assert entries[0].dimensions.conversation_id == agent.state.conversation_id
    assert entries[0].dimensions.turn_id == agent.state.turns[-1].turn_id
    assert entries[0].units["model_calls"] == 1
    assert entries[0].units["total_tokens"] > 0
    assert entries[0].cost.pricing_known
    assert entries[0].cost.amount is not None
    assert len(reservations) == 1
    assert reservations[0].state is ReservationState.COMMITTED


def test_cost_budget_stops_before_the_provider_request(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = OpenAIScriptedClient(
        [{"type": "final_answer", "content": "must not run"}]
    )
    agent = create_agent(
        config,
        llm_client=client,
        run_budget=RunBudget(
            max_cost=ExactCost(
                Decimal("0.000001"),
                pricing_known=True,
            )
        ),
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        agent.run_turn("hello")

    assert exc_info.value.dimension == "cost"
    assert client.remaining == 1
    turn = agent.state.turns[-1]
    assert turn.status == "failed"
    assert turn.model_request_count == 1
    assert turn.extension_metadata["budget_exhausted"]["dimension"] == "cost"
    assert SQLiteUsageStore(config.store_path).list_entries() == ()


def test_shared_goal_reservation_prevents_parallel_children_overspending(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    child_budget = RunBudget(
        scope=BudgetScope.CHILD_TASK,
        max_model_calls=10,
    )
    goal_budget = RunBudget(
        scope=BudgetScope.GOAL,
        max_model_calls=3,
    )
    holder = create_agent(
        config,
        llm_client=OpenAIScriptedClient(
            [{"type": "final_answer", "content": "held"}]
        ),
        run_budget=child_budget,
        additional_run_budgets=(goal_budget,),
        usage_dimensions=UsageDimensions(
            profile_id="default",
            goal_id="goal-shared",
            child_task_id="child-a",
        ),
    )
    assert holder._model_accounting.usage_accounting is not None
    held = holder._model_accounting.usage_accounting.reserve_model_request(
        turn_id="held-turn",
        request_index=1,
        messages=[{"role": "user", "content": "hold allowance"}],
        purpose="child",
    )
    blocked_client = OpenAIScriptedClient(
        [{"type": "final_answer", "content": "must not run"}]
    )
    blocked = create_agent(
        config,
        llm_client=blocked_client,
        run_budget=child_budget,
        additional_run_budgets=(goal_budget,),
        usage_dimensions=UsageDimensions(
            profile_id="default",
            goal_id="goal-shared",
            child_task_id="child-b",
        ),
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        blocked.run_turn("compete for the same goal budget")

    assert exc_info.value.scope is BudgetScope.GOAL
    assert blocked_client.remaining == 1
    assert holder._model_accounting.usage_accounting.release_model_request(
        turn_id="held-turn",
        request_index=1,
    ) is not None
    assert SQLiteUsageStore(config.store_path).get_reservation(
        held.id
    ).state is ReservationState.RELEASED


def test_shared_tool_hold_reconciles_if_secondary_commit_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    agent = create_agent(
        config,
        llm_client=OpenAIScriptedClient(
            [{"type": "final_answer", "content": "unused"}]
        ),
        run_budget=RunBudget(
            scope=BudgetScope.CHILD_TASK,
            max_tool_calls=2,
        ),
        additional_run_budgets=(
            RunBudget(scope=BudgetScope.GOAL, max_tool_calls=2),
        ),
        usage_dimensions=UsageDimensions(
            profile_id="default",
            goal_id="goal-tool-recovery",
            child_task_id="child-tool-recovery",
        ),
    )
    assert agent._model_accounting.usage_accounting is not None
    agent._model_accounting.usage_accounting.reserve_tool_call(
        turn_id="tool-turn",
        tool_call_index=1,
        attempt=1,
        tool_name="calculator",
    )
    original_commit = SQLiteUsageStore.commit
    calls = 0

    def interrupt_constraint(self, reservation_id, entries):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated crash during shared commit")
        return original_commit(self, reservation_id, entries)

    monkeypatch.setattr(SQLiteUsageStore, "commit", interrupt_constraint)
    with pytest.raises(RuntimeError, match="shared commit"):
        agent._model_accounting.usage_accounting.commit_tool_call(
            turn_id="tool-turn",
            tool_call_index=1,
            attempt=1,
            tool_name="calculator",
            success=True,
            failure_kind=None,
        )

    monkeypatch.setattr(SQLiteUsageStore, "commit", original_commit)
    store = SQLiteUsageStore(config.store_path)
    assert [item.state for item in store.list_reservations()] == [
        ReservationState.COMMITTED,
        ReservationState.ACTIVE,
    ]
    assert store.reconcile_committed_constraints() == 1
    assert all(
        item.state is ReservationState.COMMITTED
        for item in store.list_reservations()
    )


def test_tool_budget_stops_before_the_next_tool_attempt_and_records_goal_usage(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = OpenAIScriptedClient(
        [
            {
                "type": "tool_call",
                "tool_name": "calculator",
                "arguments": {"expression": "1 + 1"},
            },
            {
                "type": "tool_call",
                "tool_name": "calculator",
                "arguments": {"expression": "2 + 2"},
            },
            {"type": "final_answer", "content": "must not run"},
        ]
    )
    agent = create_agent(
        config,
        llm_client=client,
        run_budget=RunBudget(
            scope="goal",
            max_tool_calls=1,
        ),
        usage_dimensions=UsageDimensions(
            profile_id="default",
            goal_id="goal-1",
        ),
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        agent.run_turn("calculate twice")

    assert exc_info.value.dimension == "tool_calls"
    assert client.remaining == 1
    entries = SQLiteUsageStore(config.store_path).list_entries()
    tool_entries = [item for item in entries if item.resource_kind.value == "tool"]
    assert len(tool_entries) == 1
    assert tool_entries[0].tool_or_service == "calculator"
    assert tool_entries[0].dimensions.goal_id == "goal-1"
    assert tool_entries[0].units["tool_calls"] == 1
    assert agent.state.turns[-1].extension_metadata["budget_exhausted"][
        "resource_kind"
    ] == "tool"


def test_goal_runtime_checkpoints_active_tool_then_observes_pause_boundary(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    service = GoalService(GoalStore(config.store_path))
    goal = service.create(
        title="Pause safely",
        acceptance_criteria=("The active action is recorded.",),
        steps=(
            GoalStep(
                id="work",
                title="Work",
                description="Run one governed tool.",
                acceptance_criterion_ids=("criterion-1",),
            ),
        ),
        budget=RunBudget(
            scope="goal",
            max_model_calls=3,
            max_tool_calls=2,
        ),
    )
    goal = service.approve(
        goal.id,
        expected_revision=goal.revision,
        approved_by="owner",
    )
    goal = service.run(
        goal.id,
        expected_revision=goal.revision,
        actor="owner",
    )
    goal = service.start_step(
        goal.id,
        "work",
        expected_revision=goal.revision,
        actor="runner",
    )
    execution = service.claim_execution(
        goal.id,
        "work",
        expected_revision=goal.revision,
        runner_id="runner",
    )

    def pause_goal(_arguments: dict) -> ToolResult:
        current = service.store.get(goal.id)
        service.pause(
            goal.id,
            expected_revision=current.revision,
            actor="owner",
        )
        return ToolResult("pause_goal", True, "paused")

    client = OpenAIScriptedClient(
        [
            {
                "type": "tool_call",
                "tool_name": "pause_goal",
                "arguments": {},
            },
            {"type": "final_answer", "content": "must not run"},
        ]
    )
    agent = create_agent(
        config,
        llm_client=client,
        tool_specs=[
            Tool(
                name="pause_goal",
                description="Pause the governed goal.",
                args_schema={"type": "object", "properties": {}},
                callable=pause_goal,
            )
        ],
        goal_execution=execution,
    )

    with pytest.raises(GoalLeaseConflictError, match="not running"):
        agent.run_turn("work until paused")

    checkpoints = service.store.action_checkpoints(goal.id)
    assert len(checkpoints) == 1
    assert checkpoints[0].state.value == "completed"
    assert service.store.get(goal.id).status.value == "paused"
    assert client.remaining == 1
    entries = SQLiteUsageStore(config.store_path).list_entries()
    assert {entry.dimensions.goal_id for entry in entries} == {goal.id}
    assert {entry.resource_kind.value for entry in entries} == {"model", "tool"}


def test_budget_exhaustion_is_a_typed_public_error_and_event(
    tmp_path: Path,
) -> None:
    client = OpenAIScriptedClient(
        [{"type": "final_answer", "content": "must not run"}]
    )
    budget = RunBudget(
        max_cost=ExactCost(Decimal("0.000001"), pricing_known=True)
    )
    facade = PublicAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=client,
        tools=[],
        skills=[],
        run_budget=budget,
    )

    events = list(facade.run_events("hello"))

    exhausted = next(
        event for event in events if event.name == "budget.exhausted"
    )
    assert isinstance(exhausted.payload, BudgetPayload)
    assert exhausted.payload.dimension == "cost"
    assert exhausted.payload.scope == "turn"
    assert events[-1].name == "run.failed"
    assert isinstance(events[-1].payload, RunFailedPayload)
    assert events[-1].payload.error["category"] == "budget_exhausted"
    assert client.remaining == 1


def test_sdk_exposes_profile_owned_usage_queries(tmp_path: Path) -> None:
    facade = PublicAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=OpenAIScriptedClient(
            [{"type": "final_answer", "content": "done"}]
        ),
        tools=[],
        skills=[],
        usage_dimensions=UsageDimensions(
            profile_id="default",
            channel="sdk",
        ),
    )

    assert facade.run("hello") == "done"
    page = facade.query_usage(channel="sdk")
    groups = facade.group_usage(UsageGroupBy.MODEL)

    assert len(page.entries) == 1
    assert page.entries[0].dimensions.profile_id == "default"
    assert page.entries[0].dimensions.channel == "sdk"
    assert groups[0].key == "openai:gpt-4.1-mini"


def test_transport_failure_releases_the_request_allowance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    agent = create_agent(config, llm_client=FailingOpenAIClient())

    with pytest.raises(LLMError, match="provider unavailable"):
        agent.run_turn("hello")

    reservations = SQLiteUsageStore(config.store_path).list_reservations()
    assert len(reservations) == 1
    assert reservations[0].state is ReservationState.RELEASED


def test_checkpoint_reconciles_an_interrupted_ledger_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    original_commit = SQLiteUsageStore.commit

    def interrupt_commit(self, reservation_id, entries):
        del self, reservation_id, entries
        raise RuntimeError("simulated crash before ledger commit")

    monkeypatch.setattr(SQLiteUsageStore, "commit", interrupt_commit)
    first = create_agent(
        config,
        llm_client=OpenAIScriptedClient(
            [{"type": "final_answer", "content": "checkpointed"}]
        ),
        run_budget=RunBudget(
            scope=BudgetScope.CHILD_TASK,
            max_model_calls=10,
        ),
        additional_run_budgets=(
            RunBudget(scope=BudgetScope.GOAL, max_model_calls=10),
        ),
        usage_dimensions=UsageDimensions(
            profile_id="default",
            goal_id="goal-recovery",
            child_task_id="child-recovery",
        ),
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        first.run_turn("hello")

    conversation_id = first.state.conversation_id
    reservations = SQLiteUsageStore(config.store_path).list_reservations()
    assert len(reservations) == 2
    assert all(
        reservation.state is ReservationState.ACTIVE
        for reservation in reservations
    )
    assert SQLiteUsageStore(config.store_path).list_entries() == ()

    monkeypatch.setattr(SQLiteUsageStore, "commit", original_commit)
    resumed = create_agent(
        config,
        llm_client=OpenAIScriptedClient(
            [{"type": "final_answer", "content": "unused"}]
        ),
        conversation_id=conversation_id,
    )

    entries = SQLiteUsageStore(config.store_path).list_entries()
    reservations = SQLiteUsageStore(config.store_path).list_reservations()
    assert len(entries) == 1
    assert entries[0].metadata["recovered"] is True
    assert entries[0].units["model_calls"] == 1
    assert entries[0].units["total_tokens"] > 0
    assert len(reservations) == 2
    assert all(
        reservation.state is ReservationState.COMMITTED
        for reservation in reservations
    )

    reopened = create_agent(
        config,
        llm_client=OpenAIScriptedClient(
            [{"type": "final_answer", "content": "unused"}]
        ),
        conversation_id=conversation_id,
    )
    assert len(SQLiteUsageStore(config.store_path).list_entries()) == 1

    reopened.close()
    resumed.close()
    first.close()


def test_fallback_attempts_are_attributed_without_double_counting(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    success = OpenAIScriptedClient(
        [{"type": "final_answer", "content": "fallback worked"}]
    )
    success.provider = "openai"
    success.model = "gpt-4.1"
    success.model_profile_id = "secondary"
    chain = FallbackChain(
        providers=[AccountedFailureClient(), success],
    )
    agent = create_agent(config, llm_client=chain)

    assert agent.run_turn("hello") == "fallback worked"

    entries = SQLiteUsageStore(config.store_path).list_entries()
    assert len(entries) == 2
    assert [entry.model_profile_id for entry in entries] == [
        "primary",
        "secondary",
    ]
    assert [entry.metadata["success"] for entry in entries] == [False, True]
    assert sum(entry.units["model_calls"] for entry in entries) == 2
    assert sum(entry.units["total_tokens"] for entry in entries) > 0
    assert all(":attempt:" in entry.source_event_id for entry in entries)


def test_checkpoint_recovers_each_fallback_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    success = OpenAIScriptedClient(
        [{"type": "final_answer", "content": "checkpointed fallback"}]
    )
    success.provider = "openai"
    success.model = "gpt-4.1"
    success.model_profile_id = "secondary"
    original_commit = SQLiteUsageStore.commit

    def interrupt_commit(self, reservation_id, entries):
        del self, reservation_id, entries
        raise RuntimeError("simulated fallback accounting crash")

    monkeypatch.setattr(SQLiteUsageStore, "commit", interrupt_commit)
    first = create_agent(
        config,
        llm_client=FallbackChain(
            providers=[AccountedFailureClient(), success],
        ),
    )

    with pytest.raises(RuntimeError, match="fallback accounting crash"):
        first.run_turn("hello")

    monkeypatch.setattr(SQLiteUsageStore, "commit", original_commit)
    resumed = create_agent(
        config,
        llm_client=OpenAIScriptedClient(
            [{"type": "final_answer", "content": "unused"}]
        ),
        conversation_id=first.state.conversation_id,
    )

    entries = SQLiteUsageStore(config.store_path).list_entries()
    assert [entry.model_profile_id for entry in entries] == [
        "primary",
        "secondary",
    ]
    assert all(entry.metadata["recovered"] is True for entry in entries)
    assert sum(entry.units["model_calls"] for entry in entries) == 2
    assert all(":attempt:" in entry.source_event_id for entry in entries)

    resumed.close()
    first.close()
