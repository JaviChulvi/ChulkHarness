"""Evaluation comparison and portable report exporters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    current_graders = _grader_keys(current)
    baseline_graders = _grader_keys(baseline)
    return EvalComparison(
        str(current.get("id", "")), str(baseline.get("id", "")), deltas,
        tuple(sorted(current_cases & baseline_cases)),
        tuple(sorted(current_cases - baseline_cases)),
        tuple(sorted(baseline_cases - current_cases)),
        tuple(sorted(current_graders & baseline_graders)),
        tuple(sorted(current_graders - baseline_graders)),
        tuple(sorted(baseline_graders - current_graders)),
    )


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
    suite = Element(
        "testsuite",
        {
            "name": str(report.get("suite_name", "chulk-evals")),
            "tests": str(len(cases)),
            "failures": str(sum(not bool(item.get("passed")) for item in cases if isinstance(item, Mapping))),
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
            },
        )
        if not item.get("passed"):
            failure = SubElement(node, "failure", {"message": "evaluation quality gate failed"})
            failure.text = json.dumps(item, ensure_ascii=False, default=str)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + tostring(suite, encoding="unicode") + "\n"


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


def _case_keys(report: Mapping[str, Any]) -> set[str]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        return set()
    return {
        f"{item.get('target_name', '')}:{item.get('case_id', '')}"
        for item in cases if isinstance(item, Mapping)
    }


def _grader_keys(report: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    cases = report.get("cases")
    if not isinstance(cases, list):
        return keys
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        prefix = f"{case.get('target_name', '')}:{case.get('case_id', '')}"
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
                    keys.add(f"{prefix}:{grade.get('grader', '')}")
    return keys


def _format_from_suffix(path: Path) -> ReportFormat:
    suffix = path.suffix.lower()
    return {".json": "json", ".jsonl": "jsonl", ".xml": "junit", ".html": "html"}.get(suffix, "json")  # type: ignore[return-value]


__all__ = ["EvalComparison", "ReportFormat", "compare_reports", "export_report"]
