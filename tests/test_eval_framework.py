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
    GradeResult,
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


def test_eval_models_deep_freeze_nested_public_data() -> None:
    metadata = {"nested": {"values": [1, 2]}}
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    reference = EvalReference(json_schema=schema)
    case = EvalCase(
        "immutable",
        (EvalTurn("hello"),),
        reference,
        metadata=metadata,
    )
    dataset = EvalDataset((case,))
    digest = dataset.digest

    metadata["nested"]["values"].append(3)
    schema["properties"]["answer"]["type"] = "number"

    assert dataset.digest == digest
    assert case.metadata["nested"]["values"] == (1, 2)
    assert reference.json_schema is not None
    assert reference.json_schema["properties"]["answer"]["type"] == "string"
    with pytest.raises(TypeError):
        case.metadata["nested"]["changed"] = True


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
    assert report.metrics["pass_all_k"] == 1.0
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

    assert not denied.operational_errors
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
        max_total_cost=1.0,
        safety=EvalSafetyPolicy(allow_unknown_cost=True),
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


def test_all_export_formats_redact_live_report_payloads(tmp_path: Path) -> None:
    report = {
        "id": "live-secret-run",
        "suite_name": "live",
        "status": "completed",
        "passed": False,
        "metrics": {"pass_rate": 0.0},
        "threshold_failures": [],
        "operational_errors": ["provider failed with Bearer live-secret-token"],
        "metadata": {"api_key": "metadata-secret"},
        "cases": [
            {
                "target_name": "api_key=target-secret",
                "case_id": "secret-case",
                "passed": False,
                "trials": [
                    {
                        "duration_seconds": 0.1,
                        "exception": "password=trial-secret",
                        "turns": [
                            {
                                "result": {
                                    "content": "sk-liveanswer123456",
                                    "tool_calls": [
                                        {
                                            "tool_name": "lookup",
                                            "arguments": {"authorization": "tool-secret"},
                                        }
                                    ],
                                    "observations": [
                                        {"content": "token=observation-secret"}
                                    ],
                                    "errors": ["cookie=error-secret"],
                                },
                                "events": [
                                    {"payload": {"credential": "event-secret"}}
                                ],
                            }
                        ],
                        "grades": [],
                    }
                ],
            }
        ],
    }
    secrets = (
        "live-secret-token",
        "metadata-secret",
        "target-secret",
        "trial-secret",
        "sk-liveanswer123456",
        "tool-secret",
        "observation-secret",
        "error-secret",
        "event-secret",
    )

    for suffix, format_name in (
        ("json", "json"),
        ("jsonl", "jsonl"),
        ("xml", "junit"),
        ("html", "html"),
    ):
        output = export_report(
            report,
            tmp_path / f"live-report.{suffix}",
            format=format_name,
        )
        rendered = output.read_text(encoding="utf-8")
        assert "[redacted]" in rendered
        assert all(secret not in rendered for secret in secrets)


def test_html_export_prioritizes_verdict_cases_and_graders(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "id": "run-123",
        "suite_name": "support<script>",
        "status": "completed",
        "passed": True,
        "started_at": "2026-08-10T21:20:46+00:00",
        "threshold_failures": [],
        "operational_errors": [],
        "metrics": {
            "pass_rate": 1.0,
            "case_count": 1,
            "p95_latency_seconds": 0.186,
            "total_tokens": 26_285,
            "total_cost": 0,
            "grader.answer.exact.pass_rate": 1.0,
        },
        "cases": [
            {
                "target_name": "sdk<script>",
                "case_id": "ready<script>",
                "passed": True,
                "trials": [
                    {
                        "duration_seconds": 0.186,
                        "grades": [
                            {
                                "grader": "answer.exact<script>",
                                "score": 1.0,
                                "passed": True,
                                "error": None,
                                "details": {"required": True},
                            },
                            {
                                "grader": "style.informational",
                                "score": 0.5,
                                "passed": False,
                                "error": None,
                                "details": {"required": False},
                            },
                        ],
                    }
                ],
            }
        ],
    }

    output = export_report(report, tmp_path / "report.html", format="html")
    html = output.read_text(encoding="utf-8")

    assert '<span class="badge pass">Passed</span>' in html
    assert "All required checks and suite thresholds passed." in html
    assert "<strong>100%</strong>" in html
    assert "Case results" in html
    assert "Grader outcomes" in html
    assert '<span class="status info">info</span>' in html
    assert '<details class="panel">' in html
    assert "All recorded metrics" in html
    assert html.index("Case results") < html.index("All recorded metrics")
    assert "support&lt;script&gt;" in html
    assert "ready&lt;script&gt;" in html
    assert "answer.exact&lt;script&gt;" in html
    assert "<script>" not in html


def test_html_export_surfaces_failed_quality_and_operational_details(tmp_path: Path) -> None:
    report = {
        "id": "failed-run",
        "suite_name": "failed",
        "status": "completed",
        "passed": False,
        "metrics": {"pass_rate": 0.0},
        "cases": [],
        "threshold_failures": ["pass_rate < 1"],
        "operational_errors": ["provider <offline>"],
    }

    output = export_report(report, tmp_path / "failed.html", format="html")
    html = output.read_text(encoding="utf-8")

    assert '<span class="badge fail">Needs attention</span>' in html
    assert "Threshold failures" in html
    assert "pass_rate &lt; 1" in html
    assert "Operational errors" in html
    assert "provider &lt;offline&gt;" in html


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
    assert metrics["target.sdk.case_count"] == 1.0
    assert metrics["target.sdk.trial_count"] == 1.0
    assert metrics["target.sdk.pass_at_k"] == 1.0
    assert metrics["target.sdk.pass_all_k"] == 1.0
    assert metrics["target.sdk.p95_latency_seconds"] >= 0.0
    assert metrics["target.sdk.total_tokens"] == metrics["total_tokens"]
    assert metrics["target.sdk.total_cost"] == metrics["total_cost"]
    assert metrics["target.sdk.exception_count"] == 0.0
    assert metrics["target.sdk.error_rate"] == 0.0
    assert metrics["provider.fake.pass_rate"] == 1.0
    assert metrics["provider.fake.case_count"] == 1.0
    assert metrics["model.model-1.pass_rate"] == 1.0
    assert metrics["model.model-1.pass_at_k"] == 1.0
    assert metrics["model.model-1.pass_all_k"] == 1.0
    assert metrics["tag.smoke.pass_rate"] == 1.0
    assert metrics["tag.smoke.trial_count"] == 1.0


def test_compare_reports_includes_paired_case_outcomes() -> None:
    def report(report_id: str, outcomes: dict[str, bool]) -> dict[str, object]:
        return {
            "id": report_id,
            "suite_name": "paired",
            "metrics": {"pass_rate": sum(outcomes.values()) / len(outcomes)},
            "metadata": {"targets": [{"name": "agent", "fingerprint": "target-v1"}]},
            "cases": [
                {
                    "target_name": "agent",
                    "case_id": case_id,
                    "passed": passed,
                    "trials": [],
                }
                for case_id, passed in outcomes.items()
            ],
        }

    comparison = compare_reports(
        report("current", {"one": True, "two": True, "three": True, "four": False}),
        report("baseline", {"one": False, "two": False, "three": False, "four": False}),
    )

    assert comparison.current_wins == 3
    assert comparison.baseline_wins == 0
    assert comparison.ties == 1
    assert comparison.discordant_count == 3
    assert comparison.mcnemar_p_value == pytest.approx(0.25)
    payload = comparison.to_dict()
    assert payload["current_wins"] == 3
    assert payload["mcnemar_p_value"] == pytest.approx(0.25)


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


@pytest.mark.asyncio
async def test_runners_reject_grades_for_a_different_grader_identity() -> None:
    class MismatchedGrader:
        name = "required"

        def grade(self, _case, _trial):
            return GradeResult("typo", 1.0, True, "incorrect identity")

        async def grade_async(self, _case, _trial):
            return GradeResult("typo", 1.0, True, "incorrect identity")

    case = EvalCase(
        "mismatched-grader",
        (EvalTurn("Answer", ({"type": "final_answer", "content": "answer"},)),),
    )
    suite = EvalSuite(
        "mismatched-grader",
        EvalDataset((case,)),
        (EvalTarget("sdk", _agent),),
        (MismatchedGrader(),),
        required_graders=("required",),
    )

    reports = (EvalRunner().run(suite), await AsyncEvalRunner().run(suite))

    for report in reports:
        grade = report.cases[0].trials[0].grades[0]
        assert report.passed is False
        assert report.metrics["pass_rate"] == 0.0
        assert report.operational_errors
        assert grade.grader == "required"
        assert grade.details["required"] is True
        assert grade.error is not None
        assert "returned GradeResult for 'typo'" in grade.error
