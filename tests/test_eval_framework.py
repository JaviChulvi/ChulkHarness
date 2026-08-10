"""Public contract tests for the complete agent evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from chulk import Agent, AgentConfig, AsyncAgent
from chulk.evals import (
    AsyncEvalRunner,
    CallableGrader,
    EvalCase,
    EvalDataset,
    EvalReference,
    EvalReport,
    EvalRunner,
    EvalSafetyPolicy,
    EvalSuite,
    EvalTarget,
    EvalTurn,
    EvalTurnResult,
    ExactAnswerGrader,
    LLMJudgeGrader,
    MetricThreshold,
    SQLiteEvalStore,
    StatusGrader,
    ToolCallGrader,
    TrialResult,
    compare_reports,
    export_report,
)
from chulk.testing import ScriptedLLMClient
from chulk.results import Observation, RunResult, RunStatus, ToolCall
from chulk.tools import Tool, ToolPermissionLevel


def _agent(context):
    return Agent(
        config=AgentConfig(project_root=context.workspace),
        llm=context.llm,
        tools=[],
        skills=[],
    )


def _suite(dataset: EvalDataset, **kwargs) -> EvalSuite:
    return EvalSuite(
        name="support",
        dataset=dataset,
        targets=(EvalTarget("sdk", _agent),),
        graders=(ExactAnswerGrader(), StatusGrader()),
        required_graders=("answer.exact", "run.status"),
        thresholds={"pass_rate": MetricThreshold(min=1.0)},
        **kwargs,
    )


def test_dataset_jsonl_is_strict_versioned_and_digest_stable(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    payload = {
        "schema_version": 1,
        "id": "ready",
        "turns": [{"input": "Ready?", "scripted_responses": [{"type": "final_answer", "content": "Ready."}]}],
        "reference": {"answer": "Ready."},
        "tags": ["smoke"],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    first = EvalDataset.from_jsonl(path)
    second = EvalDataset.from_jsonl(path)

    assert first.digest == second.digest
    assert first.filtered(tags=("smoke",)).cases[0].id == "ready"
    payload["unknown"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown eval case fields"):
        EvalDataset.from_jsonl(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be empty"):
        EvalDataset.from_jsonl(path)


def test_dataset_rejects_duplicates_and_unsafe_replay_paths() -> None:
    case = EvalCase("same", (EvalTurn("hello"),))
    with pytest.raises(ValueError, match="duplicate"):
        EvalDataset((case, case))
    with pytest.raises(ValueError, match="inside the dataset"):
        EvalCase("unsafe", (EvalTurn("hello"),), replay_fixture="../secret.json")


def test_runner_executes_multiturn_public_agent_and_quality_gate() -> None:
    case = EvalCase(
        "conversation",
        (
            EvalTurn("Remember alpha", ({"type": "final_answer", "content": "Remembered."},)),
            EvalTurn("What did I say?", ({"type": "final_answer", "content": "alpha"},)),
        ),
        EvalReference(answer="alpha"),
    )

    report = EvalRunner().run(_suite(EvalDataset((case,))))

    assert isinstance(report, EvalReport)
    assert report.passed is True
    assert report.metrics["pass_rate"] == 1.0
    assert len(report.cases[0].trials[0].turns) == 2
    assert report.cases[0].trials[0].turns[0].events


@pytest.mark.asyncio
async def test_async_runner_matches_sync_result_shape() -> None:
    async def factory(context):
        return AsyncAgent(
            config=AgentConfig(project_root=context.workspace),
            llm=context.llm,
            tools=[],
            skills=[],
        )

    case = EvalCase(
        "ready",
        (EvalTurn("Ready?", ({"type": "final_answer", "content": "Ready."},)),),
        EvalReference(answer="Ready."),
    )
    suite = EvalSuite(
        "async",
        EvalDataset((case,)),
        (EvalTarget("async-sdk", factory),),
        (ExactAnswerGrader(),),
        required_graders=("answer.exact",),
    )

    report = await AsyncEvalRunner().run(suite)

    assert report.passed is True
    assert report.cases[0].trials[0].final_result.content == "Ready."


def test_side_effecting_tools_are_denied_unless_named() -> None:
    dangerous = Tool(
        name="write_record",
        description="Write a record.",
        args_schema={"type": "object", "properties": {}, "additionalProperties": False},
        callable=lambda _arguments: "written",
        permission_level=ToolPermissionLevel.WRITE,
    )

    def factory(context):
        return Agent(
            config=AgentConfig(project_root=context.workspace),
            llm=context.llm,
            tools=[dangerous],
            skills=[],
        )

    case = EvalCase(
        "safe",
        (EvalTurn("Do nothing", ({"type": "final_answer", "content": "done"},)),),
    )
    base = dict(name="safe", dataset=EvalDataset((case,)), targets=(EvalTarget("sdk", factory),))

    denied = EvalRunner().run(EvalSuite(**base))
    allowed = EvalRunner().run(
        EvalSuite(**base, safety=EvalSafetyPolicy(allowed_tool_names=("write_record",)))
    )

    assert denied.operational_errors
    assert "denied side-effecting tools" in denied.operational_errors[0]
    assert not allowed.operational_errors


def test_callable_and_llm_judge_graders_are_normalized() -> None:
    case = EvalCase(
        "quality",
        (EvalTurn("Answer", ({"type": "final_answer", "content": "clear answer"},)),),
    )
    judge = ScriptedLLMClient(['{"score": 0.9, "passed": true, "reason": "Clear"}'])
    suite = EvalSuite(
        "quality",
        EvalDataset((case,)),
        (EvalTarget("sdk", _agent),),
        (
            CallableGrader("custom", lambda _case, _trial: 0.75),
            LLMJudgeGrader(judge, "The answer must be clear."),
        ),
        required_graders=("custom", "quality.judge"),
    )

    report = EvalRunner().run(suite)

    grades = report.cases[0].trials[0].grades
    assert [grade.score for grade in grades] == [0.75, 0.9]
    assert grades[1].details["prompt_version"] == "1"


def test_tool_grader_matches_arguments_results_and_failures() -> None:
    reference = EvalReference(
        tool_sequence=("lookup",),
        tool_arguments={"lookup": {"id": 7}},
        tool_results={"lookup": {"found": True}},
        tool_failures=(),
    )
    case = EvalCase("tools", (EvalTurn("look up", reference=reference),))
    result = RunResult(
        "done",
        RunStatus.COMPLETED,
        None,
        "test",
        None,
        tool_calls=(ToolCall("lookup", {"id": 7}, 1, success=True),),
        observations=(Observation("lookup", '{"found": true}'),),
    )
    trial = TrialResult(
        "tools",
        "sdk",
        1,
        (EvalTurnResult(0, result, (), 0.01),),
        0.01,
    )

    grade = ToolCallGrader().grade(case, trial)

    assert grade.passed is True
    assert grade.details["results"] == {"lookup": {"found": True}}


def test_sqlite_store_baseline_comparison_and_all_exports(tmp_path: Path) -> None:
    case = EvalCase(
        "ready",
        (EvalTurn("Ready?", ({"type": "final_answer", "content": "Ready."},)),),
        EvalReference(answer="Ready."),
    )
    report = EvalRunner().run(_suite(EvalDataset((case,))))
    store = SQLiteEvalStore(tmp_path / "store.sqlite")

    store.save_report(report)
    store.set_baseline("support", report.id)
    stored = store.get_report(report.id)
    comparison = compare_reports(stored, store.get_baseline("support"))

    assert store.list_reports()[0].passed is True
    assert comparison.metric_deltas["pass_rate"] == 0
    assert comparison.matched_graders == (
        "sdk:ready:answer.exact",
        "sdk:ready:run.status",
    )
    for suffix, format_name in (("json", "json"), ("jsonl", "jsonl"), ("xml", "junit"), ("html", "html")):
        output = export_report(stored, tmp_path / f"report.{suffix}", format=format_name)
        assert output.stat().st_size > 0

    other = dict(stored)
    other["suite_name"] = "other"
    with pytest.raises(ValueError, match="same suite"):
        compare_reports(stored, other)


def test_thresholds_are_explicit_and_informational_graders_do_not_fail() -> None:
    case = EvalCase(
        "info",
        (EvalTurn("Answer", ({"type": "final_answer", "content": "actual"},)),),
        EvalReference(answer="expected"),
    )
    suite = EvalSuite(
        "info",
        EvalDataset((case,)),
        (EvalTarget("sdk", _agent),),
        (ExactAnswerGrader(),),
    )

    report = EvalRunner().run(suite)

    assert report.cases[0].trials[0].grades[0].passed is False
    assert report.cases[0].trials[0].passed is True
    assert report.passed is True
    assert report.metrics["pass_rate"] == 1.0


def test_suite_rejects_unknown_required_graders_fixtures_and_duplicate_targets() -> None:
    dataset = EvalDataset((EvalCase("case", (EvalTurn("hello"),), fixture="api"),))
    with pytest.raises(ValueError, match="unknown required graders"):
        EvalSuite(
            "invalid",
            EvalDataset((EvalCase("case", (EvalTurn("hello"),)),)),
            (EvalTarget("sdk", _agent),),
            required_graders=("missing",),
        )
    with pytest.raises(ValueError, match="unknown eval fixtures"):
        EvalSuite("invalid", dataset, (EvalTarget("sdk", _agent),))
    with pytest.raises(ValueError, match="target names must be unique"):
        EvalSuite(
            "invalid",
            EvalDataset((EvalCase("case", (EvalTurn("hello"),)),)),
            (EvalTarget("sdk", _agent), EvalTarget("sdk", _agent)),
        )


def test_metrics_are_grouped_by_target_provider_model_and_tag() -> None:
    case = EvalCase(
        "ready",
        (EvalTurn("Ready?", ({"type": "final_answer", "content": "Ready."},)),),
        EvalReference(answer="Ready."),
        tags=("smoke",),
    )
    suite = EvalSuite(
        "grouped",
        EvalDataset((case,)),
        (EvalTarget("sdk", _agent, provider="fake", model="model-1"),),
        (ExactAnswerGrader(),),
        required_graders=("answer.exact",),
    )

    metrics = EvalRunner().run(suite).metrics

    assert metrics["target.sdk.pass_rate"] == 1.0
    assert metrics["provider.fake.pass_rate"] == 1.0
    assert metrics["model.model-1.pass_rate"] == 1.0
    assert metrics["tag.smoke.pass_rate"] == 1.0


def test_sync_runner_honors_explicit_bounded_concurrency() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    class SlowAgent:
        def run_result(self, _message, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return RunResult("done", RunStatus.COMPLETED, None, "test", None)

        def close(self):
            return None

    dataset = EvalDataset(tuple(EvalCase(f"case-{index}", (EvalTurn("run"),)) for index in range(4)))
    suite = EvalSuite(
        "parallel",
        dataset,
        (EvalTarget("slow", lambda _context: SlowAgent()),),
        concurrency=2,
    )

    report = EvalRunner().run(suite)

    assert not report.operational_errors
    assert peak == 2


def test_required_grader_errors_are_operational_and_report_has_provenance() -> None:
    case = EvalCase(
        "broken-grader",
        (EvalTurn("Answer", ({"type": "final_answer", "content": "answer"},)),),
    )

    def fail(_case, _trial):
        raise RuntimeError("grader unavailable")

    suite = EvalSuite(
        "broken-grader",
        EvalDataset((case,)),
        (EvalTarget("sdk", _agent),),
        (CallableGrader("required", fail),),
        required_graders=("required",),
    )

    report = EvalRunner().run(suite)

    assert report.operational_errors
    assert "grader unavailable" in report.operational_errors[0]
    assert report.metadata["chulk_version"]
    assert len(report.metadata["suite_fingerprint"]) == 64
    assert "p95_latency_seconds" in report.metrics
