"""Read-only projections for the optional evaluation dashboard."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from chulk.evals import EvalStore, StoredEvalSummary, compare_reports
from chulk.evals.reporting import report_matches_filters
from chulk.redaction import redact_data
from chulk.tracing import Trace


def list_eval_summaries(
    store: EvalStore,
    *,
    suite_name: str | None,
    status: str | None,
    mode: str | None,
    started_after: str | None,
    started_before: str | None,
    target_name: str | None,
    provider: str | None,
    model: str | None,
    tags: tuple[str, ...],
    limit: int,
    offset: int,
) -> tuple[tuple[StoredEvalSummary, ...], bool]:
    """List a correctly paginated page, including payload-only filters."""
    payload_filters = bool(target_name or provider or model or tags)
    common = {
        "suite_name": suite_name,
        "status": status,
        "mode": mode,
        "started_after": started_after,
        "started_before": started_before,
    }
    if not payload_filters:
        page = store.list_reports(**common, limit=limit + 1, offset=offset)
        return page[:limit], len(page) > limit

    wanted = offset + limit + 1
    matched: list[StoredEvalSummary] = []
    store_offset = 0
    batch_size = min(500, max(100, wanted))
    while len(matched) < wanted:
        page = store.list_reports(
            **common,
            limit=batch_size,
            offset=store_offset,
        )
        if not page:
            break
        for summary in page:
            report = store.get_report(summary.id)
            if report_matches_filters(
                report,
                target_name=target_name,
                provider=provider,
                model=model,
                tags=tags,
            ):
                matched.append(summary)
        store_offset += len(page)
        if len(page) < batch_size:
            break
    window = matched[offset : offset + limit + 1]
    return tuple(window[:limit]), len(window) > limit


def eval_run_detail(
    report: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None,
    traces_dir: Path,
) -> dict[str, Any]:
    """Build the dashboard projection without changing the stored report."""
    baseline_value: dict[str, Any] | None = None
    if baseline is not None:
        baseline_value = {
            "run": {
                "id": str(baseline.get("id") or ""),
                "started_at": str(baseline.get("started_at") or ""),
                "passed": bool(baseline.get("passed")),
                "status": str(baseline.get("status") or "completed"),
            },
            "comparison": compare_reports(report, baseline).to_dict(),
        }
    return {
        "run": report,
        "baseline": baseline_value,
        "dimensions": _dimensions(report),
        "traces": [item[0] for item in _trace_records(report, traces_dir)],
    }


def load_eval_trace(
    report: Mapping[str, Any],
    trace_id: str,
    *,
    traces_dir: Path,
) -> dict[str, Any]:
    """Load a bounded, redacted trace selected only through its stored run record."""
    for descriptor, path in _trace_records(report, traces_dir):
        if descriptor["id"] != trace_id:
            continue
        if not descriptor["available"] or path is None:
            raise FileNotFoundError("evaluation trace is unavailable")
        trace = Trace.from_jsonl(path, max_bytes=5 * 1024 * 1024, max_events=5_000)
        counts = Counter(event.type for event in trace.events)
        return {
            "trace": descriptor,
            "event_count": len(trace.events),
            "event_types": dict(sorted(counts.items())),
            "events": [redact_data(event.to_dict()) for event in trace.events],
        }
    raise KeyError(f"unknown evaluation trace: {trace_id}")


def _dimensions(report: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    metrics = report.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    metadata = report.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw_targets = metadata.get("targets")
    raw_targets = raw_targets if isinstance(raw_targets, list) else []
    targets: list[dict[str, Any]] = []
    providers: set[str] = set()
    models: set[str] = set()
    for raw in raw_targets:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "")
        provider = _optional_text(raw.get("provider"))
        model = _optional_text(raw.get("model"))
        if provider is not None:
            providers.add(provider)
        if model is not None:
            models.add(model)
        targets.append(
            {
                "name": name,
                "provider": provider,
                "model": model,
                "pass_rate": _metric(metrics, f"target.{name}.pass_rate"),
            }
        )
    tags = sorted(
        key.removeprefix("tag.").removesuffix(".pass_rate")
        for key in metrics
        if isinstance(key, str)
        and key.startswith("tag.")
        and key.endswith(".pass_rate")
    )
    return {
        "targets": targets,
        "providers": [
            {"name": item, "pass_rate": _metric(metrics, f"provider.{item}.pass_rate")}
            for item in sorted(providers)
        ],
        "models": [
            {"name": item, "pass_rate": _metric(metrics, f"model.{item}.pass_rate")}
            for item in sorted(models)
        ],
        "tags": [
            {"name": item, "pass_rate": _metric(metrics, f"tag.{item}.pass_rate")}
            for item in tags
        ],
    }


def _trace_records(
    report: Mapping[str, Any],
    traces_dir: Path,
) -> list[tuple[dict[str, Any], Path | None]]:
    output: list[tuple[dict[str, Any], Path | None]] = []
    cases = report.get("cases")
    cases = cases if isinstance(cases, list) else []
    safe_root = traces_dir.expanduser().resolve()
    report_id = str(report.get("id") or "")
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        target_name = str(case.get("target_name") or "")
        case_id = str(case.get("case_id") or "")
        trials = case.get("trials")
        trials = trials if isinstance(trials, list) else []
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            trial_number = int(trial.get("trial") or 0)
            turns = trial.get("turns")
            turns = turns if isinstance(turns, list) else []
            for turn in turns:
                if not isinstance(turn, Mapping):
                    continue
                turn_index = int(turn.get("index") or 0)
                result = turn.get("result")
                result = result if isinstance(result, Mapping) else {}
                raw_path = result.get("trace_path")
                identity = "\0".join(
                    (report_id, target_name, case_id, str(trial_number), str(turn_index))
                )
                trace_id = sha256(identity.encode()).hexdigest()[:20]
                path: Path | None = None
                reason = "not_recorded"
                if isinstance(raw_path, str) and raw_path:
                    candidate = Path(raw_path).expanduser().resolve()
                    try:
                        candidate.relative_to(safe_root)
                    except ValueError:
                        reason = "outside_trace_store"
                    else:
                        if candidate.is_file():
                            path = candidate
                            reason = "available"
                        else:
                            reason = "missing"
                available = path is not None
                descriptor = {
                    "id": trace_id,
                    "target_name": target_name,
                    "case_id": case_id,
                    "trial": trial_number,
                    "turn": turn_index,
                    "conversation_id": str(result.get("conversation_id") or ""),
                    "available": available,
                    "reason": reason,
                    "href": (
                        f"/v1/evals/runs/{report_id}/traces/{trace_id}"
                        if available
                        else None
                    ),
                }
                output.append((descriptor, path))
    return output


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_text(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None
