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
    rows = "".join(
        f"<tr><th>{escape(name)}</th><td>{value:.6g}</td></tr>"
        for name, value in metrics.items()
    )
    case_rows = ""
    for item in report.get("cases", []) if isinstance(report.get("cases"), list) else []:
        if not isinstance(item, Mapping):
            continue
        state = "pass" if item.get("passed") else "fail"
        case_rows += (
            f'<tr><td>{escape(str(item.get("target_name", "")))}</td>'
            f'<td>{escape(str(item.get("case_id", "")))}</td>'
            f'<td><span class="status {state}">{state}</span></td></tr>'
        )
    title = escape(str(report.get("suite_name", "Chulk evaluation")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · evaluation report</title><style>
:root{{--ink:#17202a;--muted:#65717d;--paper:#f5f7fa;--line:#d7dde4;--blue:#2457d6;--green:#167451;--red:#b42318}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1080px;margin:auto;padding:48px 24px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:3px solid var(--ink);padding-bottom:20px}}
h1{{font-size:clamp(30px,5vw,58px);letter-spacing:-.05em;line-height:.95;margin:0;max-width:720px}}.run{{font:12px ui-monospace,SFMono-Regular,monospace;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:minmax(260px,1fr) 2fr;gap:32px;margin-top:32px}}section{{background:white;border:1px solid var(--line);padding:24px}}h2{{font-size:13px;text-transform:uppercase;letter-spacing:.12em;margin:0 0 16px}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 8px;border-top:1px solid var(--line);text-align:left}}th{{font-weight:600}}.status{{font:11px ui-monospace,monospace;text-transform:uppercase}}.pass{{color:var(--green)}}.fail{{color:var(--red)}}
@media(max-width:760px){{header,.grid{{display:block}}.run{{margin-top:16px}}section{{margin-top:20px;overflow:auto}}}}
</style></head><body><main><header><h1>{title}</h1><div class="run">run {escape(str(report.get('id', '')))}<br>{escape(str(report.get('started_at', '')))}</div></header>
<div class="grid"><section><h2>Measures</h2><table>{rows}</table></section><section><h2>Cases</h2><table><thead><tr><th>Target</th><th>Case</th><th>Result</th></tr></thead><tbody>{case_rows}</tbody></table></section></div>
</main></body></html>"""


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
