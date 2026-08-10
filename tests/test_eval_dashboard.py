"""Optional evaluation dashboard coverage."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from starlette.testclient import TestClient

from chulk.config import load_config
from chulk.evals import (
    CaseResult,
    EvalReport,
    EvalRunStatus,
    EvalTurnResult,
    GradeResult,
    SQLiteEvalStore,
    TrialResult,
)
from chulk.results import Cost, RunResult, RunStatus, Usage
from chulk.server import create_control_app
from chulk.server.security import ControlTokenStore
from chulk.tracing import JSONLTraceLogger


def _report(
    report_id: str,
    *,
    started_at: str = "2026-08-10T00:00:00+00:00",
    provider: str = "openai",
    model: str = "gpt-eval",
    tags: tuple[str, ...] = ("smoke",),
    trace_path: Path | None = None,
    passed: bool = True,
    status: EvalRunStatus = EvalRunStatus.COMPLETED,
    reason: str = "matched reference",
) -> EvalReport:
    result = RunResult(
        "safe output",
        RunStatus.COMPLETED,
        "turn-1",
        "conversation-1",
        trace_path,
        usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
        cost=Cost(amount=Decimal("0.001"), pricing_known=True),
    )
    grade = GradeResult(
        "answer.exact",
        1.0 if passed else 0.0,
        passed,
        reason,
        {
            "required": True,
            "grader_version": "2",
            "grader_identity": "answer-exact-v2",
            "evidence": {"expected": "safe output"},
        },
    )
    trial = TrialResult(
        "case-1",
        "sdk",
        1,
        (EvalTurnResult(0, result, (), 0.25),),
        0.25,
        (grade,),
    )
    case = CaseResult("case-1", "sdk", (trial,), passed)
    metrics = {
        "case_count": 1.0,
        "pass_rate": 1.0 if passed else 0.0,
        "p95_latency_seconds": 0.25,
        "total_tokens": 5.0,
        "total_cost": 0.001,
        "target.sdk.pass_rate": 1.0 if passed else 0.0,
        f"provider.{provider}.pass_rate": 1.0 if passed else 0.0,
        f"model.{model}.pass_rate": 1.0 if passed else 0.0,
        **{f"tag.{tag}.pass_rate": 1.0 if passed else 0.0 for tag in tags},
    }
    return EvalReport(
        report_id,
        "starter",
        "digest",
        started_at,
        "2026-08-10T00:00:01+00:00",
        (case,),
        metrics,
        (() if passed else ("pass_rate below 1.0",)),
        (),
        {
            "mode": "scripted",
            "targets": [
                {
                    "name": "sdk",
                    "provider": provider,
                    "model": model,
                    "fingerprint": "sdk-v1",
                }
            ],
            "graders": [
                {
                    "name": "answer.exact",
                    "version": "2",
                    "identity": "answer-exact-v2",
                }
            ],
        },
        status,
    )


def _client(tmp_path: Path, *reports: EvalReport, baseline: str | None = None):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = SQLiteEvalStore(config.store_path)
    for report in reports:
        store.save_report(report)
    if baseline is not None:
        store.set_baseline("starter", baseline)
    tokens = ControlTokenStore(config.runtime_dir / "control.token")
    token = tokens.load_or_create()
    app = create_control_app(
        config,
        token_store=tokens,
        enable_eval_dashboard=True,
    )
    return config, app, {"Authorization": f"Bearer {token}"}


def test_eval_dashboard_is_opt_in_authenticated_and_read_only(tmp_path: Path) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    store = SQLiteEvalStore(config.store_path)
    store.save_report(_report("run-1"))
    tokens = ControlTokenStore(config.runtime_dir / "control.token")
    disabled = create_control_app(config, token_store=tokens)
    enabled = create_control_app(config, token_store=tokens, enable_eval_dashboard=True)
    auth = {"Authorization": f"Bearer {tokens.load_or_create()}"}

    with TestClient(disabled) as client:
        assert client.get("/evals", headers=auth).status_code == 404
        assert client.get("/v1/evals/runs", headers=auth).status_code == 404
    with TestClient(enabled) as client:
        assert client.get("/evals").status_code == 200
        assert client.get("/v1/evals/runs").status_code == 401
        assert client.get("/v1/evals/runs/run-1").status_code == 401
        assert client.post("/v1/evals/runs", headers=auth).status_code == 405
        assert client.post("/v1/evals/runs/run-1", headers=auth).status_code == 405


def test_eval_run_api_filters_after_payload_matching_and_paginates(tmp_path: Path) -> None:
    _, app, auth = _client(
        tmp_path,
        _report("run-old", started_at="2026-08-10T00:00:00+00:00"),
        _report(
            "run-middle",
            started_at="2026-08-10T01:00:00+00:00",
            provider="anthropic",
            model="claude-eval",
            tags=("safety",),
        ),
        _report("run-new", started_at="2026-08-10T02:00:00+00:00"),
    )
    with TestClient(app) as client:
        first = client.get(
            "/v1/evals/runs?provider=openai&tag=smoke&limit=1",
            headers=auth,
        )
        second = client.get(
            "/v1/evals/runs?provider=openai&tag=smoke&limit=1&offset=1",
            headers=auth,
        )
        indexed = client.get(
            "/v1/evals/runs?status=completed&mode=scripted&started_after=2026-08-10T00:30:00%2B00:00",
            headers=auth,
        )

        assert [item["id"] for item in first.json()["runs"]] == ["run-new"]
        assert first.json()["has_more"] is True
        assert first.json()["next_offset"] == 1
        assert [item["id"] for item in second.json()["runs"]] == ["run-old"]
        assert second.json()["has_more"] is False
        assert [item["id"] for item in indexed.json()["runs"]] == [
            "run-new",
            "run-middle",
        ]
        assert client.get("/v1/evals/runs?status=bad", headers=auth).status_code == 400
        assert client.get("/v1/evals/runs?started_after=not-a-date", headers=auth).status_code == 400


def test_eval_detail_includes_baseline_dimensions_evidence_and_missing_trace(
    tmp_path: Path,
) -> None:
    malicious = '<img src=x onerror="alert(1)">'
    baseline = _report("baseline", passed=False)
    missing = load_config(
        {"CHULK_PROJECT_ROOT": str(tmp_path)}
    ).traces_dir / "deleted.jsonl"
    current = _report("current", trace_path=missing, reason=malicious)
    _, app, auth = _client(tmp_path, baseline, current, baseline="baseline")

    with TestClient(app) as client:
        response = client.get("/v1/evals/runs/current", headers=auth)
        value = response.json()

        assert response.status_code == 200
        assert value["baseline"]["run"]["id"] == "baseline"
        assert value["baseline"]["comparison"]["metric_deltas"]["pass_rate"] == 1.0
        assert value["dimensions"]["targets"] == [
            {
                "name": "sdk",
                "provider": "openai",
                "model": "gpt-eval",
                "pass_rate": 1.0,
            }
        ]
        assert value["run"]["cases"][0]["trials"][0]["grades"][0]["reason"] == malicious
        assert value["traces"][0]["available"] is False
        assert value["traces"][0]["reason"] == "missing"
        assert value["traces"][0]["href"] is None


def test_eval_trace_is_bounded_to_trace_store_redacted_and_handles_missing(
    tmp_path: Path,
) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    logger = JSONLTraceLogger(config.traces_dir, "conversation-1")
    logger.log(
        "turn_started",
        {"turn": {"turn_id": "turn-1"}, "api_key": "sk-dashboard-secret-123456"},
    )
    logger.log("turn_finished", {"turn": {"turn_id": "turn-1"}})
    logger.close()
    available = _report("available", trace_path=logger.path)
    outside = _report("outside", trace_path=tmp_path / "outside.jsonl")
    _, app, auth = _client(tmp_path, available, outside)

    with TestClient(app) as client:
        detail = client.get("/v1/evals/runs/available", headers=auth).json()
        descriptor = detail["traces"][0]
        trace = client.get(descriptor["href"], headers=auth)
        unavailable = client.get("/v1/evals/runs/outside", headers=auth).json()["traces"][0]

        assert descriptor["available"] is True
        assert trace.status_code == 200
        assert trace.json()["event_count"] == 4
        assert "sk-dashboard-secret" not in trace.text
        assert "[redacted]" in trace.text
        assert unavailable["available"] is False
        assert unavailable["reason"] == "outside_trace_store"
        missing_path = f"/v1/evals/runs/outside/traces/{unavailable['id']}"
        assert client.get(missing_path, headers=auth).status_code == 404
        assert client.post(descriptor["href"], headers=auth).status_code == 405


def test_eval_dashboard_static_ui_is_safe_and_complete(tmp_path: Path) -> None:
    _, app, _ = _client(tmp_path, _report("run-1"))
    with TestClient(app) as client:
        page = client.get("/evals")
        script = client.get("/evals/assets/evals.js")
        styles = client.get("/evals/assets/evals.css")

    assert page.status_code == script.status_code == styles.status_code == 200
    assert 'id="filter-form"' in page.text
    assert 'id="prev-page"' in page.text
    assert "Baseline seam" in script.text
    assert "Comparison matrix" in script.text
    assert "grade.details" in script.text
    assert "trace.href" in script.text
    assert ".innerHTML" not in script.text
    assert ".textContent" in script.text
    assert ".baseline-seam::before" in styles.text
    assert "prefers-reduced-motion" in styles.text
