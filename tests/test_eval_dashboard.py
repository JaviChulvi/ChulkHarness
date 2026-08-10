"""Optional evaluation dashboard coverage."""

from pathlib import Path

from starlette.testclient import TestClient

from chulk.config import load_config
from chulk.evals import EvalReport, SQLiteEvalStore
from chulk.server import create_control_app
from chulk.server.security import ControlTokenStore


def _report(report_id: str = "run-1") -> EvalReport:
    return EvalReport(
        report_id,
        "starter",
        "digest",
        "2026-08-10T00:00:00+00:00",
        "2026-08-10T00:00:01+00:00",
        (),
        {"case_count": 0.0, "pass_rate": 1.0, "total_cost": 0.0},
    )


def test_eval_dashboard_is_opt_in_authenticated_and_read_only(tmp_path: Path) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    SQLiteEvalStore(config.store_path).save_report(_report())
    tokens = ControlTokenStore(config.runtime_dir / "control.token")
    disabled = create_control_app(config, token_store=tokens)
    enabled = create_control_app(config, token_store=tokens, enable_eval_dashboard=True)
    auth = {"Authorization": f"Bearer {tokens.load_or_create()}"}

    with TestClient(disabled) as client:
        assert client.get("/evals", headers=auth).status_code == 404
    with TestClient(enabled) as client:
        assert client.get("/evals").status_code == 200
        assert client.get("/v1/evals/runs").status_code == 401
        listing = client.get("/v1/evals/runs", headers=auth)
        empty_page = client.get("/v1/evals/runs?limit=1&offset=1", headers=auth)
        detail = client.get("/v1/evals/runs/run-1", headers=auth)
        assert listing.status_code == 200
        assert listing.json()["runs"][0]["id"] == "run-1"
        assert empty_page.json()["runs"] == []
        assert empty_page.json()["limit"] == 1
        assert empty_page.json()["offset"] == 1
        assert detail.json()["run"]["suite_name"] == "starter"
        assert client.post("/v1/evals/runs", headers=auth).status_code == 405
