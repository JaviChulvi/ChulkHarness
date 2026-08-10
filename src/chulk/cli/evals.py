"""Command-line workflows for agent evaluation suites."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml

from chulk.evals import (
    EvalReport,
    EvalRunner,
    EvalSafetyPolicy,
    EvalSuite,
    EvaluationMode,
    SQLiteEvalStore,
    compare_reports,
    export_report,
)
from chulk.evals.reporting import report_matches_filters


EXIT_EVAL_OK = 0
EXIT_EVAL_QUALITY = 1
EXIT_EVAL_CONFIG = 2
EXIT_EVAL_OPERATIONAL = 3


def run_eval_command(
    command: str,
    *,
    store: SQLiteEvalStore,
    suite_ref: str | None = None,
    report_id: str | None = None,
    baseline_id: str | None = None,
    suite_name: str | None = None,
    resume_from: str | None = None,
    tags: tuple[str, ...] = (),
    trials: int | None = None,
    concurrency: int | None = None,
    timeout_seconds: float | None = None,
    mode: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_total_cost: float | None = None,
    allow_unknown_cost: bool = False,
    fail_fast: bool = False,
    status: str | None = None,
    target_name: str | None = None,
    started_after: str | None = None,
    started_before: str | None = None,
    limit: int = 100,
    offset: int = 0,
    min_baseline_coverage: float | None = None,
    output_path: Path | str | None = None,
    export_format: str | None = None,
    init_path: Path | str | None = None,
    json_output: bool = False,
    output_func: Callable[[str], None] = print,
    error_func: Callable[[str], None] = print,
) -> int:
    try:
        if command == "init":
            created = initialize_eval_project(init_path or Path.cwd())
            return _emit({"created": [str(path) for path in created]}, json_output, output_func)
        if command == "run":
            if suite_ref is None:
                raise ValueError("eval run requires a suite reference")
            suite = load_eval_suite(suite_ref)
            targets = tuple(
                replace(
                    target,
                    provider=provider if provider is not None else target.provider,
                    model=model if model is not None else target.model,
                )
                for target in suite.targets
            )
            safety = suite.safety
            if allow_unknown_cost:
                safety = EvalSafetyPolicy(
                    allowed_tool_names=safety.allowed_tool_names,
                    allow_unknown_cost=True,
                )
            suite = replace(
                suite,
                targets=targets,
                trials=trials if trials is not None else suite.trials,
                concurrency=concurrency if concurrency is not None else suite.concurrency,
                timeout_seconds=timeout_seconds if timeout_seconds is not None else suite.timeout_seconds,
                mode=EvaluationMode(mode) if mode is not None else suite.mode,
                max_total_cost=max_total_cost if max_total_cost is not None else suite.max_total_cost,
                safety=safety,
                fail_fast=fail_fast or suite.fail_fast,
            )
            if suite.store is None:
                suite = replace(suite, store=store)
            report = EvalRunner().run(
                suite,
                tags=tags,
                resume_from=resume_from,
            )
            if not isinstance(report, EvalReport):
                raise TypeError("suite execution did not return EvalReport")
            if output_path is not None:
                export_report(report, output_path, format=export_format)  # type: ignore[arg-type]
            _emit(report.to_dict(), json_output, output_func)
            if report.operational_errors:
                return EXIT_EVAL_OPERATIONAL
            return EXIT_EVAL_OK if report.passed else EXIT_EVAL_QUALITY
        if command == "list":
            if not 1 <= limit <= 10_000:
                raise ValueError("eval list limit must be between 1 and 10000")
            if offset < 0:
                raise ValueError("eval list offset must be non-negative")
            normalized_after = _optional_iso_timestamp(started_after, "started-after")
            normalized_before = _optional_iso_timestamp(started_before, "started-before")
            payload_filters = bool(target_name or provider or model or tags)
            summaries = store.list_reports(
                suite_name=suite_name,
                status=status,
                mode=mode,
                started_after=normalized_after,
                started_before=normalized_before,
                limit=10_000 if payload_filters else limit,
                offset=0 if payload_filters else offset,
            )
            if payload_filters:
                summaries = tuple(
                    summary
                    for summary in summaries
                    if report_matches_filters(
                        store.get_report(summary.id),
                        target_name=target_name,
                        provider=provider,
                        model=model,
                        tags=tags,
                    )
                )[offset : offset + limit]
            return _emit({"reports": [item.to_dict() for item in summaries]}, json_output, output_func)
        if command == "show":
            if report_id is None:
                raise ValueError("eval show requires a report id")
            payload = store.get_report(report_id)
            if json_output:
                return _emit(payload, True, output_func)
            output_func(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return EXIT_EVAL_OK
        if command == "compare":
            if report_id is None:
                raise ValueError("eval compare requires a report id")
            current = store.get_report(report_id)
            baseline = store.get_report(baseline_id) if baseline_id else store.get_baseline(str(current["suite_name"]))
            if baseline is None:
                raise ValueError("no baseline is configured for this suite")
            if min_baseline_coverage is not None and not (
                0.0 <= min_baseline_coverage <= 1.0
            ):
                raise ValueError("minimum baseline coverage must be between 0 and 1")
            comparison = compare_reports(current, baseline)
            _emit(comparison.to_dict(), json_output, output_func)
            if (
                min_baseline_coverage is not None
                and comparison.baseline_coverage < min_baseline_coverage
            ):
                return EXIT_EVAL_QUALITY
            return EXIT_EVAL_OK
        if command == "baseline-set":
            if suite_name is None or report_id is None:
                raise ValueError("eval baseline set requires suite name and report id")
            store.set_baseline(suite_name, report_id)
            return _emit({"suite_name": suite_name, "report_id": report_id, "baseline": True}, json_output, output_func)
        if command == "export":
            if report_id is None or output_path is None:
                raise ValueError("eval export requires report id and output path")
            destination = export_report(store.get_report(report_id), output_path, format=export_format)  # type: ignore[arg-type]
            return _emit({"report_id": report_id, "output": str(destination)}, json_output, output_func)
        raise ValueError(f"unknown eval command: {command}")
    except (FileNotFoundError, ImportError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
        error_func(json.dumps({"status": "configuration_error", "error": str(exc)}) if json_output else f"evaluation configuration error: {exc}")
        return EXIT_EVAL_CONFIG
    except Exception as exc:
        error_func(json.dumps({"status": "operational_error", "error": str(exc)}) if json_output else f"evaluation operational error: {exc}")
        return EXIT_EVAL_OPERATIONAL


def load_eval_suite(reference: str) -> EvalSuite:
    if reference.endswith((".yaml", ".yml")):
        yaml_path = Path(reference).expanduser().resolve()
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(payload.get("suite"), str):
            raise ValueError("eval YAML must contain a suite: module:symbol reference")
        reference = payload["suite"]
        source, separator, symbol = reference.rpartition(":")
        if separator and source.endswith(".py") and not Path(source).is_absolute():
            reference = f"{(yaml_path.parent / source).resolve()}:{symbol}"
    if ":" not in reference:
        raise ValueError("suite reference must use module:symbol or path.py:symbol")
    source, symbol = reference.rsplit(":", 1)
    if source.endswith(".py"):
        path = Path(source).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(f"chulk_eval_suite_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load eval suite module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(source)
    value = getattr(module, symbol, None)
    if callable(value) and not isinstance(value, EvalSuite):
        value = value()
    if not isinstance(value, EvalSuite):
        raise TypeError(f"{reference} did not resolve to EvalSuite")
    return value


def initialize_eval_project(root: Path | str) -> tuple[Path, ...]:
    destination = Path(root).expanduser().resolve() / "evals"
    destination.mkdir(parents=True, exist_ok=True)
    suite_path = destination / "suite.py"
    cases_path = destination / "cases.jsonl"
    created: list[Path] = []
    if not cases_path.exists():
        cases_path.write_text(
            json.dumps({
                "schema_version": 1,
                "id": "ready",
                "turns": [{"input": "Are you ready?", "scripted_responses": [{"type": "final_answer", "content": "Ready."}]}],
                "reference": {"answer": "Ready.", "status": "completed"},
                "tags": ["smoke"],
            }, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        created.append(cases_path)
    if not suite_path.exists():
        suite_path.write_text(
            '''from pathlib import Path\n\nfrom chulk import Agent, AgentConfig\nfrom chulk.evals import (\n    EvalDataset, EvalSuite, EvalTarget, ExactAnswerGrader, StatusGrader, MetricThreshold,\n)\n\nROOT = Path(__file__).parent\n\ndef create_agent(context):\n    return Agent(\n        config=AgentConfig(project_root=context.workspace),\n        llm=context.llm,\n        tools=[],\n        skills=[],\n    )\n\nsuite = EvalSuite(\n    name="starter",\n    dataset=EvalDataset.from_jsonl(ROOT / "cases.jsonl"),\n    targets=(EvalTarget("agent", create_agent),),\n    graders=(ExactAnswerGrader(), StatusGrader()),\n    required_graders=("answer.exact", "run.status"),\n    thresholds={"pass_rate": MetricThreshold(min=1.0)},\n)\n''',
            encoding="utf-8",
        )
        created.append(suite_path)
    return tuple(created)


def _emit(payload: Mapping[str, Any], json_output: bool, output_func: Callable[[str], None]) -> int:
    if json_output:
        output_func(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
    elif "reports" in payload:
        reports = payload["reports"]
        output_func("No evaluation runs." if not reports else "\n".join(
            f"{item['id']}  {item['suite_name']}  {item.get('status', 'completed')}  "
            f"{'pass' if item['passed'] else 'fail'}  {item['pass_rate']:.1%}"
            for item in reports
        ))
    elif "suite_name" in payload and "metrics" in payload:
        output_func(
            f"Evaluation {payload['suite_name']} · {'passed' if payload.get('passed') else 'failed'}\n"
            f"run {payload.get('id')} · pass rate {float(payload['metrics'].get('pass_rate', 0)):.1%}"
        )
    else:
        output_func(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return EXIT_EVAL_OK


def _optional_iso_timestamp(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"eval list {name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


__all__ = [
    "EXIT_EVAL_CONFIG", "EXIT_EVAL_OK", "EXIT_EVAL_OPERATIONAL", "EXIT_EVAL_QUALITY",
    "initialize_eval_project", "load_eval_suite", "run_eval_command",
]
