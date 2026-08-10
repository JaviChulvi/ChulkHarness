"""CLI and optional dashboard coverage for evaluation results."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from chulk.cli.evals import initialize_eval_project, load_eval_suite, run_eval_command
from chulk.evals import EvalReport, EvalRunStatus, EvalRunner, SQLiteEvalStore
from chulk.main import main
from chulk.cli.parser import build_parser


def _report(
    report_id: str = "run-1",
    *,
    suite_name: str = "starter",
    status: EvalRunStatus = EvalRunStatus.COMPLETED,
    mode: str = "scripted",
    provider: str = "fake",
    model: str = "v1",
    tags: tuple[str, ...] = (),
    started_at: str | None = None,
) -> EvalReport:
    now = started_at or datetime.now(timezone.utc).isoformat()
    metrics = {"case_count": 0.0, "pass_rate": 1.0, "total_cost": 0.0}
    metrics.update({f"tag.{tag}.pass_rate": 1.0 for tag in tags})
    return EvalReport(
        report_id,
        suite_name,
        "digest",
        now,
        now,
        (),
        metrics,
        metadata={
            "mode": mode,
            "targets": [
                {
                    "name": "agent",
                    "provider": provider,
                    "model": model,
                    "fingerprint": f"{provider}:{model}",
                }
            ],
            "graders": [],
        },
        status=status,
    )


def test_eval_init_is_non_overwriting_and_suite_loads(tmp_path: Path) -> None:
    first = initialize_eval_project(tmp_path)
    original = (tmp_path / "evals" / "suite.py").read_text(encoding="utf-8")
    second = initialize_eval_project(tmp_path)

    assert len(first) == 2
    assert second == ()
    assert (tmp_path / "evals" / "suite.py").read_text(encoding="utf-8") == original
    assert load_eval_suite(f"{tmp_path / 'evals' / 'suite.py'}:suite").name == "starter"


def test_eval_parser_exposes_resume_filters_and_coverage_gate() -> None:
    run = build_parser().parse_args(
        ["eval", "run", "suite.py:suite", "--resume", "run-id"]
    )
    listing = build_parser().parse_args(
        [
            "eval",
            "list",
            "--status",
            "completed",
            "--provider",
            "fake",
            "--tag",
            "smoke",
            "--offset",
            "2",
        ]
    )
    comparison = build_parser().parse_args(
        ["eval", "compare", "run-id", "--min-coverage", "0.9"]
    )

    assert run.resume_from == "run-id"
    assert listing.status == "completed"
    assert listing.provider == "fake"
    assert listing.tag == ["smoke"]
    assert listing.offset == 2
    assert comparison.min_baseline_coverage == 0.9


def test_eval_cli_list_show_baseline_compare_and_export(tmp_path: Path) -> None:
    store = SQLiteEvalStore(tmp_path / "store.sqlite")
    store.save_report(_report())
    output: list[str] = []

    assert run_eval_command("list", store=store, json_output=True, output_func=output.append) == 0
    assert run_eval_command("baseline-set", store=store, suite_name="starter", report_id="run-1", output_func=output.append) == 0
    assert run_eval_command("compare", store=store, report_id="run-1", json_output=True, output_func=output.append) == 0
    assert run_eval_command("show", store=store, report_id="run-1", output_func=output.append) == 0
    assert '"metadata"' in output[-1]
    assert run_eval_command("export", store=store, report_id="run-1", output_path=tmp_path / "report.html", output_func=output.append) == 0
    assert (tmp_path / "report.html").exists()


def test_eval_list_filters_status_mode_target_provider_model_tag_and_page(
    tmp_path: Path,
) -> None:
    store = SQLiteEvalStore(tmp_path / "filters.sqlite")
    store.save_report(
        _report(
            "match",
            provider="openai",
            model="gpt-test",
            tags=("smoke",),
            started_at="2026-08-10T10:00:00+00:00",
        )
    )
    store.save_report(
        _report(
            "other",
            status=EvalRunStatus.INTERRUPTED,
            mode="live",
            provider="other",
            model="other-model",
            started_at="2026-08-09T10:00:00+00:00",
        )
    )
    output: list[str] = []

    code = run_eval_command(
        "list",
        store=store,
        status="completed",
        mode="scripted",
        target_name="agent",
        provider="openai",
        model="gpt-test",
        tags=("smoke",),
        started_after="2026-08-10T00:00:00+00:00",
        started_before="2026-08-11T00:00:00+00:00",
        limit=1,
        offset=0,
        json_output=True,
        output_func=output.append,
    )

    assert code == 0
    assert [item["id"] for item in json.loads(output[-1])["reports"]] == ["match"]
    assert store.list_reports(status="interrupted", mode="live")[0].id == "other"


def test_eval_run_overrides_and_all_exit_codes(tmp_path: Path) -> None:
    initialize_eval_project(tmp_path)
    suite_ref = f"{tmp_path / 'evals' / 'suite.py'}:suite"
    store = SQLiteEvalStore(tmp_path / "runs.sqlite")
    output: list[str] = []
    errors: list[str] = []

    passed = run_eval_command(
        "run",
        store=store,
        suite_ref=suite_ref,
        tags=("smoke",),
        trials=2,
        concurrency=2,
        timeout_seconds=2,
        provider="fake",
        model="fake-model",
        json_output=True,
        output_func=output.append,
        error_func=errors.append,
    )
    payload = json.loads(output[-1])
    assert passed == 0
    assert payload["metrics"]["trial_count"] == 2
    assert payload["metadata"]["targets"][0]["provider"] == "fake"
    assert payload["metadata"]["targets"][0]["model"] == "fake-model"

    quality_path = tmp_path / "quality.py"
    quality_path.write_text(_suite_source(answer="actual", expected="expected"), encoding="utf-8")
    assert run_eval_command(
        "run",
        store=store,
        suite_ref=f"{quality_path}:suite",
        output_func=output.append,
        error_func=errors.append,
    ) == 1

    assert run_eval_command(
        "run",
        store=store,
        suite_ref=suite_ref,
        mode="live",
        output_func=output.append,
        error_func=errors.append,
    ) == 2

    operational_path = tmp_path / "operational.py"
    operational_path.write_text(_operational_suite_source(), encoding="utf-8")
    assert run_eval_command(
        "run",
        store=store,
        suite_ref=f"{operational_path}:suite",
        output_func=output.append,
        error_func=errors.append,
    ) == 3

    cost_path = tmp_path / "cost.py"
    cost_path.write_text(_cost_suite_source(), encoding="utf-8")
    assert run_eval_command(
        "run",
        store=store,
        suite_ref=f"{cost_path}:suite",
        mode="live",
        max_total_cost=1.0,
        output_func=output.append,
        error_func=errors.append,
    ) == 3


def test_eval_resume_export_formats_and_baseline_coverage_exit(tmp_path: Path) -> None:
    suite_path = tmp_path / "resume_suite.py"
    marker = tmp_path / "interrupt"
    calls = tmp_path / "calls.log"
    marker.touch()
    suite_path.write_text(
        _resumable_suite_source(marker=marker, calls=calls),
        encoding="utf-8",
    )
    store = SQLiteEvalStore(tmp_path / "resume.sqlite")
    suite = replace(load_eval_suite(f"{suite_path}:suite"), store=store)
    with pytest.raises(KeyboardInterrupt):
        EvalRunner().run(suite)
    interrupted_id = store.list_reports()[0].id
    marker.unlink()

    output: list[str] = []
    assert run_eval_command(
        "run",
        store=store,
        suite_ref=f"{suite_path}:suite",
        resume_from=interrupted_id,
        json_output=True,
        output_func=output.append,
    ) == 0
    assert json.loads(output[-1])["id"] == interrupted_id
    assert calls.read_text(encoding="utf-8").splitlines() == ["one", "two", "two"]

    store.set_baseline("resume", interrupted_id)
    assert run_eval_command(
        "compare",
        store=store,
        report_id=interrupted_id,
        min_baseline_coverage=1.0,
        output_func=output.append,
    ) == 0

    for suffix, format_name in (
        ("json", "json"),
        ("jsonl", "jsonl"),
        ("xml", "junit"),
        ("html", "html"),
    ):
        destination = tmp_path / f"report.{suffix}"
        assert run_eval_command(
            "export",
            store=store,
            report_id=interrupted_id,
            output_path=destination,
            export_format=format_name,
            output_func=output.append,
        ) == 0
        assert destination.exists()
    json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert (tmp_path / "report.jsonl").read_text(encoding="utf-8").strip()
    ElementTree.parse(tmp_path / "report.xml")
    assert "<!doctype html>" in (tmp_path / "report.html").read_text(encoding="utf-8")

    store.save_report(_report("unfinished", status=EvalRunStatus.INTERRUPTED))
    unfinished_xml = tmp_path / "unfinished.xml"
    assert run_eval_command(
        "export",
        store=store,
        report_id="unfinished",
        output_path=unfinished_xml,
        export_format="junit",
        output_func=output.append,
    ) == 0
    unfinished_suite = ElementTree.parse(unfinished_xml).getroot()
    assert unfinished_suite.attrib["errors"] == "1"
    assert unfinished_suite.find(".//error") is not None


def test_yaml_paths_are_relative_and_real_main_eval_flow_is_credential_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    initialized = initialize_eval_project(tmp_path)
    assert initialized
    yaml_path = tmp_path / "evals" / "suite.yaml"
    yaml_path.write_text("suite: suite.py:suite\n", encoding="utf-8")
    assert load_eval_suite(str(yaml_path)).name == "starter"

    project = tmp_path / "project"
    output: list[str] = []
    errors: list[str] = []
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(project))
    assert main(
        ["eval", "init", str(project), "--json"],
        output_func=output.append,
        error_func=errors.append,
    ) == 0
    exit_code = main(
        [
            "eval",
            "run",
            f"{project / 'evals' / 'suite.py'}:suite",
            "--tag",
            "smoke",
            "--json",
        ],
        output_func=output.append,
        error_func=errors.append,
    )
    report = json.loads(output[-1])
    assert exit_code == 0, report["operational_errors"]
    assert report["passed"] is True
    assert not errors


def _suite_source(*, answer: str, expected: str) -> str:
    return f'''from chulk.evals import EvalCase, EvalDataset, EvalReference, EvalSuite, EvalTarget, EvalTurn, ExactAnswerGrader, MetricThreshold
from chulk.results import RunResult, RunStatus

class Agent:
    def run_result(self, message, **kwargs):
        return RunResult({answer!r}, RunStatus.COMPLETED, None, "conversation", None)
    def close(self):
        pass

suite = EvalSuite(
    "quality",
    EvalDataset((EvalCase("case", (EvalTurn("run"),), EvalReference(answer={expected!r})),)),
    (EvalTarget("agent", lambda context: Agent()),),
    (ExactAnswerGrader(),),
    required_graders=("answer.exact",),
    thresholds={{"pass_rate": MetricThreshold(min=1.0)}},
)
'''


def _operational_suite_source() -> str:
    return '''from chulk.evals import EvalCase, EvalDataset, EvalSuite, EvalTarget, EvalTurn

class Agent:
    def run_result(self, message, **kwargs):
        raise RuntimeError("provider unavailable")
    def close(self):
        pass

suite = EvalSuite(
    "operational",
    EvalDataset((EvalCase("case", (EvalTurn("run"),)),)),
    (EvalTarget("agent", lambda context: Agent()),),
)
'''


def _resumable_suite_source(*, marker: Path, calls: Path) -> str:
    return f'''from pathlib import Path
from chulk.evals import EvalCase, EvalDataset, EvalReference, EvalSuite, EvalTarget, EvalTurn, ExactAnswerGrader
from chulk.results import RunResult, RunStatus

MARKER = Path({str(marker)!r})
CALLS = Path({str(calls)!r})

class Agent:
    def __init__(self, case_id):
        self.case_id = case_id
    def run_result(self, message, **kwargs):
        with CALLS.open("a", encoding="utf-8") as stream:
            stream.write(self.case_id + "\\n")
        if self.case_id == "two" and MARKER.exists():
            raise KeyboardInterrupt
        return RunResult("done", RunStatus.COMPLETED, None, "conversation", None)
    def close(self):
        pass

def factory(context):
    return Agent(context.case_id)

suite = EvalSuite(
    "resume",
    EvalDataset(tuple(
        EvalCase(case_id, (EvalTurn("run"),), EvalReference(answer="done"))
        for case_id in ("one", "two")
    )),
    (EvalTarget("agent", factory),),
    (ExactAnswerGrader(),),
    required_graders=("answer.exact",),
)
'''


def _cost_suite_source() -> str:
    return '''from decimal import Decimal
from chulk.evals import EvalCase, EvalDataset, EvalSuite, EvalTarget, EvalTurn
from chulk.results import Cost, RunResult, RunStatus

class Agent:
    def run_result(self, message, **kwargs):
        return RunResult(
            "done",
            RunStatus.COMPLETED,
            None,
            "conversation",
            None,
            cost=Cost(amount=Decimal("2"), pricing_known=True),
        )
    def close(self):
        pass

suite = EvalSuite(
    "cost",
    EvalDataset((EvalCase("case", (EvalTurn("run"),)),)),
    (EvalTarget("agent", lambda context: Agent()),),
)
'''
