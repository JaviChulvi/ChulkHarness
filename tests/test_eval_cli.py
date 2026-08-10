"""CLI and optional dashboard coverage for evaluation results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from chulk.cli.evals import initialize_eval_project, load_eval_suite, run_eval_command
from chulk.evals import EvalReport, SQLiteEvalStore


def _report(report_id: str = "run-1") -> EvalReport:
    now = datetime.now(timezone.utc).isoformat()
    return EvalReport(
        report_id, "starter", "digest", now, now, (),
        {"case_count": 0.0, "pass_rate": 1.0, "total_cost": 0.0},
    )


def test_eval_init_is_non_overwriting_and_suite_loads(tmp_path: Path) -> None:
    first = initialize_eval_project(tmp_path)
    original = (tmp_path / "evals" / "suite.py").read_text(encoding="utf-8")
    second = initialize_eval_project(tmp_path)

    assert len(first) == 2
    assert second == ()
    assert (tmp_path / "evals" / "suite.py").read_text(encoding="utf-8") == original
    assert load_eval_suite(f"{tmp_path / 'evals' / 'suite.py'}:suite").name == "starter"


def test_eval_cli_list_show_baseline_compare_and_export(tmp_path: Path) -> None:
    store = SQLiteEvalStore(tmp_path / "store.sqlite")
    store.save_report(_report())
    output: list[str] = []

    assert run_eval_command("list", store=store, json_output=True, output_func=output.append) == 0
    assert run_eval_command("baseline-set", store=store, suite_name="starter", report_id="run-1", output_func=output.append) == 0
    assert run_eval_command("compare", store=store, report_id="run-1", json_output=True, output_func=output.append) == 0
    assert run_eval_command("export", store=store, report_id="run-1", output_path=tmp_path / "report.html", output_func=output.append) == 0
    assert (tmp_path / "report.html").exists()
