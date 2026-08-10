"""Evaluation comparison and portable report exporters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape
import json
from pathlib import Path
from typing import Any, Literal
from xml.etree.ElementTree import Element, SubElement, tostring

from .models import EvalReport


ReportFormat = Literal["json", "jsonl", "junit", "html"]


@dataclass(frozen=True)
class EvalComparison:
    current_id: str
    baseline_id: str
    metric_deltas: Mapping[str, float]
    matched_cases: tuple[str, ...]
    new_cases: tuple[str, ...]
    removed_cases: tuple[str, ...]
    matched_graders: tuple[str, ...] = ()
    new_graders: tuple[str, ...] = ()
    removed_graders: tuple[str, ...] = ()
    grade_score_deltas: Mapping[str, float] = field(default_factory=dict)
    baseline_coverage: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_id": self.current_id,
            "baseline_id": self.baseline_id,
            "metric_deltas": dict(self.metric_deltas),
            "matched_cases": list(self.matched_cases),
            "new_cases": list(self.new_cases),
            "removed_cases": list(self.removed_cases),
            "matched_graders": list(self.matched_graders),
            "new_graders": list(self.new_graders),
            "removed_graders": list(self.removed_graders),
            "grade_score_deltas": dict(self.grade_score_deltas),
            "baseline_coverage": self.baseline_coverage,
        }


def compare_reports(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> EvalComparison:
    if current.get("suite_name") != baseline.get("suite_name"):
        raise ValueError("evaluation baselines must belong to the same suite")
    current_metrics = _number_mapping(current.get("metrics"))
    baseline_metrics = _number_mapping(baseline.get("metrics"))
    deltas = {
        key: current_metrics[key] - baseline_metrics[key]
        for key in sorted(set(current_metrics) & set(baseline_metrics))
    }
    current_cases = _case_keys(current)
    baseline_cases = _case_keys(baseline)
    current_graders = _grader_scores(current)
    baseline_graders = _grader_scores(baseline)
    matched_case_ids = set(current_cases) & set(baseline_cases)
    new_case_ids = set(current_cases) - set(baseline_cases)
    removed_case_ids = set(baseline_cases) - set(current_cases)
    matched_grader_ids = set(current_graders) & set(baseline_graders)
    new_grader_ids = set(current_graders) - set(baseline_graders)
    removed_grader_ids = set(baseline_graders) - set(current_graders)
    return EvalComparison(
        str(current.get("id", "")), str(baseline.get("id", "")), deltas,
        tuple(sorted(current_cases[item] for item in matched_case_ids)),
        tuple(sorted(_identity_label(current_cases[item], item[0]) for item in new_case_ids)),
        tuple(sorted(_identity_label(baseline_cases[item], item[0]) for item in removed_case_ids)),
        tuple(sorted(current_graders[item][0] for item in matched_grader_ids)),
        tuple(sorted(_identity_label(current_graders[item][0], item[2]) for item in new_grader_ids)),
        tuple(sorted(_identity_label(baseline_graders[item][0], item[2]) for item in removed_grader_ids)),
        {
            current_graders[item][0]: current_graders[item][1] - baseline_graders[item][1]
            for item in sorted(matched_grader_ids)
        },
        len(matched_case_ids) / len(baseline_cases) if baseline_cases else 1.0,
    )


def report_matches_filters(
    report: Mapping[str, Any],
    *,
    target_name: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    tags: tuple[str, ...] = (),
) -> bool:
    """Match report payload dimensions that are not indexed by evaluation stores."""
    metadata = report.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    targets = metadata.get("targets")
    targets = targets if isinstance(targets, list) else []
    matching_targets = [item for item in targets if isinstance(item, Mapping)]
    if target_name is not None:
        matching_targets = [
            item for item in matching_targets if item.get("name") == target_name
        ]
    if provider is not None:
        matching_targets = [
            item for item in matching_targets if item.get("provider") == provider
        ]
    if model is not None:
        matching_targets = [
            item for item in matching_targets if item.get("model") == model
        ]
    if (target_name is not None or provider is not None or model is not None) and not matching_targets:
        return False
    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    return all(f"tag.{tag}.pass_rate" in metrics for tag in tags)


def export_report(report: EvalReport | Mapping[str, Any], path: Path | str, *, format: ReportFormat | None = None) -> Path:
    destination = Path(path).expanduser().resolve()
    payload = report.to_dict() if isinstance(report, EvalReport) else dict(report)
    selected = format or _format_from_suffix(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if selected == "json":
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    elif selected == "jsonl":
        rows = payload.get("cases", [])
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows)
    elif selected == "junit":
        text = _junit(payload)
    elif selected == "html":
        text = _html(payload)
    else:  # pragma: no cover - Literal plus CLI validation
        raise ValueError(f"unsupported eval report format: {selected}")
    destination.write_text(text, encoding="utf-8")
    return destination


def _junit(report: Mapping[str, Any]) -> str:
    cases = report.get("cases", [])
    cases = cases if isinstance(cases, list) else []
    threshold_failures = report.get("threshold_failures")
    threshold_failures = (
        threshold_failures if isinstance(threshold_failures, list) else []
    )
    operational_errors = report.get("operational_errors")
    operational_errors = (
        operational_errors if isinstance(operational_errors, list) else []
    )
    lifecycle_error = report.get("status", "completed") != "completed"
    quality_gate = bool(threshold_failures)
    operational_gate = bool(operational_errors or lifecycle_error)
    suite = Element(
        "testsuite",
        {
            "name": str(report.get("suite_name", "chulk-evals")),
            "tests": str(len(cases) + int(quality_gate) + int(operational_gate)),
            "failures": str(
                sum(
                    not bool(item.get("passed"))
                    for item in cases
                    if isinstance(item, Mapping)
                )
                + int(bool(threshold_failures))
            ),
            "errors": str(int(bool(operational_errors or lifecycle_error))),
        },
    )
    for item in cases:
        if not isinstance(item, Mapping):
            continue
        node = SubElement(
            suite,
            "testcase",
            {
                "classname": str(item.get("target_name", "agent")),
                "name": str(item.get("case_id", "case")),
                "time": f"{_case_duration(item):.6f}",
            },
        )
        if not item.get("passed"):
            failure = SubElement(node, "failure", {"message": "evaluation quality gate failed"})
            failure.text = json.dumps(item, ensure_ascii=False, default=str)
    if quality_gate:
        gate = SubElement(
            suite,
            "testcase",
            {"classname": "chulk.evals", "name": "suite.quality_gate"},
        )
        failure = SubElement(
            gate,
            "failure",
            {"message": "evaluation threshold failed"},
        )
        failure.text = "\n".join(str(item) for item in threshold_failures)
    if operational_gate:
        gate = SubElement(
            suite,
            "testcase",
            {"classname": "chulk.evals", "name": "suite.operational"},
        )
        error = SubElement(
            gate,
            "error",
            {"message": "evaluation operational failure"},
        )
        messages = [str(item) for item in operational_errors]
        if lifecycle_error:
            messages.append(f"run status is {report.get('status')}")
        error.text = "\n".join(messages)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + tostring(suite, encoding="unicode") + "\n"


def _case_duration(case: Mapping[str, Any]) -> float:
    trials = case.get("trials")
    if not isinstance(trials, list):
        return 0.0
    return sum(
        float(trial.get("duration_seconds", 0.0))
        for trial in trials
        if isinstance(trial, Mapping)
        and isinstance(trial.get("duration_seconds", 0.0), int | float)
    )


def _html(report: Mapping[str, Any]) -> str:
    metrics = _number_mapping(report.get("metrics"))
    cases = [
        item
        for item in report.get("cases", [])
        if isinstance(item, Mapping)
    ] if isinstance(report.get("cases"), list) else []
    passed = bool(report.get("passed"))
    run_status = str(report.get("status", "completed"))
    state = "pass" if passed and run_status == "completed" else "fail"
    verdict = "Passed" if state == "pass" else "Needs attention"
    status_copy = (
        "All required checks and suite thresholds passed."
        if state == "pass"
        else "Review failed checks, thresholds, or operational errors below."
    )
    summary_cards = "".join(
        _summary_card(label, value, detail)
        for label, value, detail in (
            (
                "Pass rate",
                _format_percent(metrics.get("pass_rate", 0.0)),
                "required quality gates",
            ),
            (
                "Cases",
                _format_count(metrics.get("case_count", float(len(cases)))),
                _pluralize(len(cases), "evaluated case"),
            ),
            (
                "P95 latency",
                _format_duration(
                    metrics.get(
                        "p95_latency_seconds",
                        metrics.get("mean_latency_seconds", 0.0),
                    )
                ),
                "per trial",
            ),
            (
                "Tokens",
                _format_count(
                    metrics.get("total_tokens", metrics.get("agent_tokens", 0.0))
                ),
                "agent and judge usage",
            ),
            (
                "Cost",
                _format_cost(metrics.get("total_cost", 0.0)),
                "recorded total",
            ),
        )
    )
    case_rows = "".join(_html_case_row(item) for item in cases)
    if not case_rows:
        case_rows = '<tr><td class="empty" colspan="5">No cases were recorded.</td></tr>'
    grades = _html_grade_summary(cases)
    grade_rows = "".join(
        _html_grade_row(name, values)
        for name, values in grades.items()
    )
    if not grade_rows:
        grade_rows = '<tr><td class="empty" colspan="5">No grader results were recorded.</td></tr>'
    metric_rows = "".join(
        '<div class="metric"><dt>'
        f'{escape(name)}</dt><dd>{escape(_format_number(value))}</dd></div>'
        for name, value in sorted(metrics.items())
    )
    alerts = _html_alerts(report)
    title = escape(str(report.get("suite_name", "Chulk evaluation")))
    report_id = escape(str(report.get("id", "")) or "not recorded")
    started_at = escape(str(report.get("started_at", "")) or "not recorded")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · evaluation report</title><style>
:root{{--ink:#0d1b2a;--muted:#617080;--paper:#f3f6f8;--surface:#fff;--line:#d9e1e7;--line-strong:#b9c5cf;--green:#0f7b5d;--green-soft:#e8f5f0;--red:#b42318;--red-soft:#fff0ee;--blue:#315ee7;--shadow:0 18px 50px rgba(13,27,42,.07)}}
*{{box-sizing:border-box}}html{{background:var(--paper)}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 "Avenir Next",Avenir,"Segoe UI",ui-sans-serif,system-ui,sans-serif}}
main{{width:min(1180px,calc(100% - 40px));margin:0 auto;padding:48px 0 64px}}.hero{{position:relative;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:40px;overflow:hidden;background:var(--surface);border:1px solid var(--line);border-radius:22px;padding:34px 38px 32px;box-shadow:var(--shadow)}}
.hero:before{{content:"";position:absolute;inset:0 auto 0 0;width:7px;background:var(--green)}}.hero.fail:before{{background:var(--red)}}.eyebrow{{margin:0 0 10px;color:var(--muted);font:700 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.13em;text-transform:uppercase}}
h1{{max-width:780px;margin:0;font:700 clamp(32px,5vw,55px)/1.02 ui-rounded,"Avenir Next Rounded","Arial Rounded MT Bold","Avenir Next",sans-serif;letter-spacing:-.045em;overflow-wrap:anywhere}}.lede{{max-width:650px;margin:17px 0 0;color:var(--muted);font-size:16px}}
.verdict{{display:flex;min-width:190px;flex-direction:column;align-items:flex-end;justify-content:space-between;text-align:right}}.badge{{display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:8px 12px;background:var(--green-soft);color:var(--green);font:800 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase}}.badge:before{{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}}.badge.fail{{background:var(--red-soft);color:var(--red)}}
.run{{max-width:260px;margin-top:32px;color:var(--muted);font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}}.summary{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:16px 0 28px}}.summary-card{{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:15px;padding:18px 19px}}.summary-card span{{display:block;color:var(--muted);font-size:12px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}}.summary-card strong{{display:block;margin-top:7px;font-size:25px;line-height:1.1;letter-spacing:-.03em;font-variant-numeric:tabular-nums}}.summary-card small{{display:block;margin-top:6px;color:var(--muted);font-size:11px}}
.alerts{{display:grid;gap:10px;margin:0 0 20px}}.alert{{border:1px solid #f1c7c2;border-radius:14px;background:var(--red-soft);padding:15px 18px;color:#7d2018}}.alert strong{{display:block;margin-bottom:4px}}.alert ul{{margin:6px 0 0;padding-left:20px}}
.panel{{margin-top:16px;background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:25px 28px}}.section-heading{{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:18px}}h2{{margin:0;font-size:20px;line-height:1.2;letter-spacing:-.02em}}.count{{flex:none;border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--muted);font:11px/1 ui-monospace,SFMono-Regular,Menlo,monospace}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:14px 12px;border-top:1px solid var(--line);text-align:left;vertical-align:middle}}thead th{{border-top:0;color:var(--muted);font-size:11px;letter-spacing:.07em;text-transform:uppercase}}tbody th{{font-weight:700}}td{{font-variant-numeric:tabular-nums}}.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}}.empty{{padding:30px 12px;color:var(--muted);text-align:center}}
.status{{display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:6px 9px;font:800 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.07em;text-transform:uppercase}}.status:before{{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}}.status.pass{{background:var(--green-soft);color:var(--green)}}.status.fail{{background:var(--red-soft);color:var(--red)}}.status.info{{background:#edf2ff;color:var(--blue)}}
details.panel{{padding:0}}details summary{{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:22px 28px;cursor:pointer;font-weight:750;list-style:none}}details summary::-webkit-details-marker{{display:none}}details summary:after{{content:"+";color:var(--muted);font:20px/1 ui-monospace,SFMono-Regular,Menlo,monospace}}details[open] summary:after{{content:"−"}}details summary:focus-visible{{outline:3px solid rgba(49,94,231,.25);outline-offset:3px;border-radius:14px}}.metric-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 28px;padding:0 28px 24px}}.metric{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:16px;padding:11px 0;border-top:1px solid var(--line)}}.metric dt{{min-width:0;color:var(--muted);font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}}.metric dd{{margin:0;font:700 12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}}
footer{{display:flex;justify-content:space-between;gap:24px;margin-top:20px;padding:0 4px;color:var(--muted);font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}footer span:last-child{{text-align:right;overflow-wrap:anywhere}}
@media(max-width:900px){{.summary{{grid-template-columns:repeat(3,minmax(0,1fr))}}.metric-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:640px){{main{{width:min(100% - 24px,1180px);padding:20px 0 36px}}.hero{{display:block;padding:27px 24px 25px;border-radius:18px}}.verdict{{min-width:0;align-items:flex-start;margin-top:24px;text-align:left}}.run{{margin-top:18px}}.summary{{grid-template-columns:repeat(2,minmax(0,1fr));margin-bottom:20px}}.panel{{padding:21px 18px}}.section-heading{{align-items:flex-start}}th,td{{padding:12px 10px;white-space:nowrap}}details summary{{padding:20px 18px}}.metric-grid{{grid-template-columns:1fr;padding:0 18px 20px}}footer{{display:block}}footer span{{display:block}}footer span:last-child{{margin-top:5px;text-align:left}}}}
@media print{{html,body{{background:#fff}}main{{width:100%;padding:0}}.hero,.summary-card,.panel{{box-shadow:none}}details .metric-grid{{display:grid}}details summary:after{{display:none}}}}
</style></head><body><main><header class="hero {state}"><div><p class="eyebrow">Agent evaluation report</p><h1>{title}</h1><p class="lede">{status_copy}</p></div><div class="verdict"><span class="badge {state}">{verdict}</span><div class="run">run {report_id}<br>{started_at}</div></div></header>
<section class="summary" aria-label="Evaluation summary">{summary_cards}</section>{alerts}
<section class="panel"><div class="section-heading"><div><p class="eyebrow">Execution</p><h2>Case results</h2></div><span class="count">{escape(_pluralize(len(cases), "case"))}</span></div><div class="table-wrap"><table><thead><tr><th scope="col">Case</th><th scope="col">Target</th><th scope="col">Trials</th><th scope="col">Duration</th><th scope="col">Result</th></tr></thead><tbody>{case_rows}</tbody></table></div></section>
<section class="panel"><div class="section-heading"><div><p class="eyebrow">Quality gates</p><h2>Grader outcomes</h2></div><span class="count">{escape(_pluralize(len(grades), "grader"))}</span></div><div class="table-wrap"><table><thead><tr><th scope="col">Grader</th><th scope="col">Required</th><th scope="col">Score</th><th scope="col">Passed</th><th scope="col">Result</th></tr></thead><tbody>{grade_rows}</tbody></table></div></section>
<details class="panel"><summary>All recorded metrics <span class="count">{len(metrics)}</span></summary><dl class="metric-grid">{metric_rows}</dl></details>
<footer><span>Generated by Chulk evals</span><span>{report_id}</span></footer>
</main></body></html>"""


def _summary_card(label: str, value: str, detail: str) -> str:
    return (
        '<div class="summary-card">'
        f"<span>{escape(label)}</span><strong>{escape(value)}</strong>"
        f"<small>{escape(detail)}</small></div>"
    )


def _html_case_row(case: Mapping[str, Any]) -> str:
    trials = case.get("trials")
    trial_count = len(trials) if isinstance(trials, list) else 0
    state = "pass" if case.get("passed") else "fail"
    return (
        f'<tr><th scope="row">{escape(str(case.get("case_id", "")))}</th>'
        f'<td>{escape(str(case.get("target_name", "")))}</td>'
        f'<td class="mono">{trial_count}</td>'
        f'<td class="mono">{escape(_format_duration(_case_duration(case)))}</td>'
        f'<td><span class="status {state}">{state}</span></td></tr>'
    )


def _html_grade_summary(
    cases: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for case in cases:
        trials = case.get("trials")
        if not isinstance(trials, list):
            continue
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            grades = trial.get("grades")
            if not isinstance(grades, list):
                continue
            for grade in grades:
                if not isinstance(grade, Mapping):
                    continue
                name = str(grade.get("grader") or "unnamed grader")
                details = grade.get("details")
                details = details if isinstance(details, Mapping) else {}
                item = summary.setdefault(
                    name,
                    {"required": False, "scores": [], "passed": 0, "total": 0},
                )
                item["required"] = bool(item["required"] or details.get("required"))
                score = grade.get("score")
                if isinstance(score, int | float) and not isinstance(score, bool):
                    item["scores"].append(float(score))
                item["total"] += 1
                if grade.get("passed") and not grade.get("error"):
                    item["passed"] += 1
    return summary


def _html_grade_row(name: str, values: Mapping[str, Any]) -> str:
    scores = values.get("scores")
    scores = scores if isinstance(scores, list) else []
    average = sum(scores) / len(scores) if scores else 0.0
    passed = int(values.get("passed", 0))
    total = int(values.get("total", 0))
    required = bool(values.get("required"))
    state = "pass" if passed == total and total else "fail" if required else "info"
    label = "pass" if state == "pass" else "fail" if state == "fail" else "info"
    return (
        f'<tr><th scope="row">{escape(name)}</th>'
        f'<td>{"Yes" if required else "No"}</td>'
        f'<td class="mono">{escape(_format_percent(average))}</td>'
        f'<td class="mono">{passed}/{total}</td>'
        f'<td><span class="status {state}">{label}</span></td></tr>'
    )


def _html_alerts(report: Mapping[str, Any]) -> str:
    blocks: list[str] = []
    for title, key in (
        ("Threshold failures", "threshold_failures"),
        ("Operational errors", "operational_errors"),
    ):
        values = report.get(key)
        if not isinstance(values, list) or not values:
            continue
        items = "".join(f"<li>{escape(str(item))}</li>" for item in values)
        blocks.append(f'<div class="alert"><strong>{title}</strong><ul>{items}</ul></div>')
    return f'<section class="alerts" aria-label="Evaluation issues">{"".join(blocks)}</section>' if blocks else ""


def _format_percent(value: float) -> str:
    percentage = value * 100
    return f"{percentage:.0f}%" if percentage.is_integer() else f"{percentage:.1f}%"


def _format_duration(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.0f} ms"
    return f"{value:.2f} s"


def _format_count(value: float) -> str:
    return f"{int(value):,}" if value.is_integer() else f"{value:,.1f}"


def _format_cost(value: float) -> str:
    if value == 0:
        return "$0.00"
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"


def _format_number(value: float) -> str:
    return f"{int(value):,}" if value.is_integer() else f"{value:.6g}"


def _pluralize(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _number_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, int | float) and not isinstance(item, bool)
    }


def _case_keys(report: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        return {}
    targets = _target_identities(report)
    output: dict[tuple[str, str], str] = {}
    for item in cases:
        if not isinstance(item, Mapping):
            continue
        target_name = str(item.get("target_name") or "")
        case_id = str(item.get("case_id") or "")
        output[(targets.get(target_name, target_name), case_id)] = (
            f"{target_name}:{case_id}"
        )
    return output


def _grader_scores(
    report: Mapping[str, Any],
) -> dict[tuple[str, str, str], tuple[str, float]]:
    scores: dict[tuple[str, str, str], list[float]] = {}
    labels: dict[tuple[str, str, str], str] = {}
    cases = report.get("cases")
    if not isinstance(cases, list):
        return {}
    targets = _target_identities(report)
    graders = _grader_identities(report)
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        target_name = str(case.get("target_name") or "")
        case_id = str(case.get("case_id") or "")
        prefix = f"{target_name}:{case_id}"
        trials = case.get("trials")
        if not isinstance(trials, list):
            continue
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            grades = trial.get("grades")
            if not isinstance(grades, list):
                continue
            for grade in grades:
                if isinstance(grade, Mapping):
                    grader_name = str(grade.get("grader") or "")
                    details = grade.get("details")
                    details = details if isinstance(details, Mapping) else {}
                    identity = str(
                        details.get("grader_identity")
                        or graders.get(grader_name)
                        or grader_name
                    )
                    key = (
                        targets.get(target_name, target_name),
                        case_id,
                        identity,
                    )
                    labels[key] = f"{prefix}:{grader_name}"
                    score = grade.get("score")
                    if isinstance(score, int | float) and not isinstance(score, bool):
                        scores.setdefault(key, []).append(float(score))
    return {
        key: (labels[key], sum(values) / len(values))
        for key, values in scores.items()
        if values
    }


def _target_identities(report: Mapping[str, Any]) -> dict[str, str]:
    metadata = report.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    targets = metadata.get("targets")
    if not isinstance(targets, list):
        return {}
    output: dict[str, str] = {}
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        name = str(target.get("name") or "")
        identity = str(
            target.get("fingerprint")
            or json.dumps(
                {
                    "name": name,
                    "provider": target.get("provider"),
                    "model": target.get("model"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        output[name] = identity
    return output


def _grader_identities(report: Mapping[str, Any]) -> dict[str, str]:
    metadata = report.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    graders = metadata.get("graders")
    if not isinstance(graders, list):
        return {}
    output: dict[str, str] = {}
    for grader in graders:
        if isinstance(grader, str):
            output[grader] = grader
        elif isinstance(grader, Mapping):
            name = str(grader.get("name") or "")
            output[name] = str(
                grader.get("identity")
                or f"{name}@{grader.get('version', '1')}"
            )
    return output


def _identity_label(label: str, identity: str) -> str:
    return f"{label}@{identity[:12]}"


def _format_from_suffix(path: Path) -> ReportFormat:
    suffix = path.suffix.lower()
    return {".json": "json", ".jsonl": "jsonl", ".xml": "junit", ".html": "html"}.get(suffix, "json")  # type: ignore[return-value]


__all__ = ["EvalComparison", "ReportFormat", "compare_reports", "export_report"]
