"""Tests for immutable typed public result snapshots."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
import json

import pytest

from chulk import (
    Agent,
    AgentConfig,
    ContextReport,
    ContextSection,
    Cost,
    Observation,
    Plan,
    PlanStatus,
    PlanStep,
    PlanStepEvidence,
    PlanStepStatus,
    RunResult,
    RunStatus,
    Tool,
    ToolCall,
    Usage,
)
from chulk._sdk.results import cost_snapshot, usage_snapshot
from chulk.llm import LLMClient, LLMCost, LLMUsage


class FakeLLM(LLMClient):
    provider = "test"
    model = "typed-results"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, messages, *, max_output_tokens=None) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def _final(content: str = "done") -> str:
    return json.dumps({"type": "final_answer", "content": content})


def test_run_result_uses_typed_immutable_records(tmp_path):
    @Tool
    def lookup(query: str) -> str:
        """Return a result."""
        return f"found {query}"

    llm = FakeLLM(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "lookup",
                    "arguments_json": json.dumps({"query": "sdk"}),
                }
            ),
            _final(),
        ]
    )
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=llm,
        tools=[lookup],
        skills=[],
    )

    result = facade.run_result("look up sdk", extension_metadata={"adapter": {"version": 1}})

    assert isinstance(result, RunResult)
    assert result.status is RunStatus.COMPLETED
    assert isinstance(result.usage, Usage)
    assert result.cost is not None
    assert isinstance(result.cost, Cost)
    assert isinstance(result.context_report, ContextReport)
    assert isinstance(result.tool_calls[0], ToolCall)
    assert isinstance(result.observations[0], Observation)
    assert result.tool_calls[0].arguments["query"] == "sdk"

    with pytest.raises(TypeError):
        result.extension_metadata["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        result.extension_metadata["adapter"]["version"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        result.tool_calls[0].arguments["query"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.content = "changed"  # type: ignore[misc]


def test_snapshot_is_detached_from_later_runtime_mutation(tmp_path):
    metadata = {"nested": {"value": 1}}
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
    )

    result = facade.run_result("hello", extension_metadata=metadata)
    facade.state.turns[-1].extension_metadata["nested"]["value"] = 99
    facade.state.turns[-1].errors.append("later")
    metadata["nested"]["value"] = 100

    assert result.extension_metadata["nested"]["value"] == 1
    assert "later" not in result.errors


def test_plan_and_context_snapshots_use_finite_statuses_and_tuples(tmp_path):
    plan_payload = {
        "summary": "Update docs",
        "steps": [
            {
                "id": "1",
                "title": "Edit docs",
                "description": "Apply the documentation update.",
                "status": "pending",
                "depends_on": [],
                "acceptance_criteria": ["Docs are updated"],
            }
        ],
    }
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM(
            [
                json.dumps(
                    {
                        "type": "plan",
                        "content": None,
                        "tool_name": None,
                        "arguments_json": "{}",
                        "plan_json": json.dumps(plan_payload),
                    }
                )
            ]
        ),
        tools=[],
        skills=[],
    )

    result = facade.plan_result("Update docs")

    assert result.status is RunStatus.WAITING_FOR_APPROVAL
    assert isinstance(result.plan, Plan)
    assert result.plan.status is PlanStatus.PENDING_APPROVAL
    assert isinstance(result.plan.steps, tuple)
    assert isinstance(result.plan.steps[0], PlanStep)
    assert result.plan.steps[0].status is PlanStepStatus.PENDING
    assert result.plan.steps[0].acceptance_criteria == ("Docs are updated",)
    assert result.context_report is not None
    assert isinstance(result.context_report.sections, tuple)


def test_unknown_statuses_do_not_masquerade_as_current_values():
    step = PlanStep(id="1", title="Future", description="Future", status="future")  # type: ignore[arg-type]
    plan = Plan(summary="Future", status="future", steps=(step,))  # type: ignore[arg-type]
    result = RunResult(
        content="",
        status="future",  # type: ignore[arg-type]
        turn_id=None,
        conversation_id="conversation",
        trace_path=None,
        plan=plan,
    )

    assert step.status is PlanStepStatus.UNKNOWN
    assert plan.status is PlanStatus.UNKNOWN
    assert result.status is RunStatus.UNKNOWN


def test_unknown_cost_remains_distinct_from_zero_cost():
    unknown = Cost(amount=None, pricing_known=False)
    free = Cost(amount=Decimal("0"), pricing_known=True)

    assert unknown.amount is None
    assert free.amount == Decimal("0")
    assert unknown.to_dict()["amount"] is None
    assert free.to_dict()["amount"] == "0"


def test_cache_write_fields_survive_public_snapshots_and_old_payloads():
    usage = usage_snapshot(
        LLMUsage(
            input_tokens=100,
            cache_write_input_tokens=20,
        )
    )
    cost = cost_snapshot(
        LLMCost(
            amount=Decimal("0.5"),
            pricing_known=True,
            cache_write_input_cost=Decimal("0.1"),
        )
    )
    old_usage = usage_snapshot({"input_tokens": 10})
    old_cost = cost_snapshot({"amount": "0.2", "pricing_known": True})

    assert usage is not None
    assert usage.cache_write_input_tokens == 20
    assert cost is not None
    assert cost.cache_write_input_cost == Decimal("0.1")
    assert old_usage is not None
    assert old_usage.cache_write_input_tokens == 0
    assert old_cost is not None
    assert old_cost.cache_write_input_cost is None


def test_cache_write_fields_preserve_public_positional_constructors():
    usage = Usage(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        True,
        True,
        "legacy",
        {"old": True},
    )
    cost = Cost(
        Decimal("1"),
        "EUR",
        True,
        True,
        Decimal("0.2"),
        Decimal("0.1"),
        Decimal("0.7"),
        "test",
        "legacy",
        "https://models.example/pricing",
        "2026-07-13",
    )

    assert usage.cache_miss_input_tokens == 6
    assert usage.reasoning_tokens == 7
    assert usage.cache_write_input_tokens == 0
    assert cost.output_cost == Decimal("0.7")
    assert cost.pricing_last_checked == "2026-07-13"
    assert cost.cache_write_input_cost is None


def test_every_nested_contract_collection_is_immutable():
    usage = Usage(raw={"provider": {"buckets": [1, 2]}})
    call = ToolCall(
        tool_name="lookup",
        arguments={"filters": {"tags": ["sdk"]}},
        iteration=1,
        metadata={"timing": {"attempts": [1]}},
    )
    observation = Observation(
        tool_name="lookup",
        content="done",
        output_metadata={"artifacts": [{"path": "result.txt"}]},
    )
    section = ContextSection(
        name="history",
        label="History",
        char_count=1,
        estimated_tokens=1,
        metadata={"roles": {"user": 1}},
    )
    evidence = PlanStepEvidence(content="verified", metadata={"checks": ["tests"]})
    step = PlanStep(
        id="1",
        title="Verify",
        description="Verify",
        depends_on=("0",),
        acceptance_criteria=("tests pass",),
        evidence=(evidence,),
    )

    assert usage.raw["provider"]["buckets"] == (1, 2)
    assert call.arguments["filters"]["tags"] == ("sdk",)
    assert call.metadata["timing"]["attempts"] == (1,)
    assert observation.output_metadata["artifacts"][0]["path"] == "result.txt"
    assert section.metadata["roles"]["user"] == 1
    assert step.depends_on == ("0",)
    assert step.evidence[0].metadata["checks"] == ("tests",)

    for mapping, key in (
        (usage.raw, "new"),
        (call.arguments, "new"),
        (call.metadata, "new"),
        (observation.output_metadata, "new"),
        (section.metadata, "new"),
        (evidence.metadata, "new"),
    ):
        with pytest.raises(TypeError):
            mapping[key] = True  # type: ignore[index]


def test_serialization_returns_fresh_plain_mutable_data(tmp_path):
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
    )
    result = facade.run_result("hello", extension_metadata={"nested": {"value": 1}})

    first = result.to_dict()
    second = result.to_dict()
    first["extension_metadata"]["nested"]["value"] = 2
    first["errors"].append("adapter-only")

    assert second["extension_metadata"]["nested"]["value"] == 1
    assert result.extension_metadata["nested"]["value"] == 1
    assert "adapter-only" not in result.errors
    assert isinstance(first["loaded_skill_names"], list)
    assert first["trace_path"] == str(result.trace_path)


def test_empty_and_no_pending_results_keep_typed_empty_values(tmp_path):
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
    )

    completed = facade.run_result("hello")
    neutral = facade.approve_result()

    assert completed.tool_calls == ()
    assert completed.observations == ()
    assert neutral.status is RunStatus.NO_PENDING_PLAN
    assert neutral.turn_id is None
    assert neutral.to_dict()["tool_calls"] == []


def test_failure_and_terminal_event_serialization_preserve_typed_shape(tmp_path):
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLM([_final()]),
        tools=[],
        skills=[],
    )

    terminal = list(facade.run_events("hello"))[-1]
    terminal_data = terminal.to_dict()
    failed = RunResult(
        content="failed",
        status=RunStatus.FAILED,
        turn_id="turn",
        conversation_id="conversation",
        trace_path=None,
        errors=("failed",),
    )

    assert terminal_data["payload"]["result"]["status"] == "completed"
    assert terminal_data["payload"]["result"]["usage"]["total_tokens"] >= 0
    assert failed.to_dict()["status"] == "failed"
    assert failed.to_dict()["errors"] == ["failed"]
