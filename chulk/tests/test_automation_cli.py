from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from chulk.cli.automations import run_automation_command
from chulk.cli.parser import build_parser
from chulk.scheduling import SQLiteScheduleStore


def test_automation_parser_exposes_full_control_surface() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "automation",
            "update",
            "job-1",
            "--revision",
            "2",
            "--cron",
            "0 9 * * 1-5",
            "--timezone",
            "Europe/Madrid",
        ]
    )
    assert parsed.command == "automation"
    assert parsed.automation_command == "update"
    assert parsed.cron == "0 9 * * 1-5"


def test_automation_cli_lists_inspects_controls_and_history(tmp_path) -> None:
    store = SQLiteScheduleStore(tmp_path / "store.sqlite")
    job = store.create(
        adapter="test",
        destination_id="owner",
        prompt="hello",
        next_run_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    output: list[str] = []
    assert (
        run_automation_command(
            "list",
            store=store,
            json_output=True,
            output_func=output.append,
        )
        == 0
    )
    assert json.loads(output[-1])["jobs"][0]["id"] == job.id

    assert (
        run_automation_command(
            "pause",
            store=store,
            job_id=job.id,
            expected_revision=job.revision,
            idempotency_key="pause-cli",
            json_output=True,
            output_func=output.append,
        )
        == 0
    )
    paused = json.loads(output[-1])["job"]
    assert paused["status"] == "paused"

    assert (
        run_automation_command(
            "history",
            store=store,
            job_id=job.id,
            json_output=True,
            output_func=output.append,
        )
        == 0
    )
    assert json.loads(output[-1])["runs"] == []
