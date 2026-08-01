"""Tests for cooperative control-server lifecycle and CLI behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from chulk.cli.parser import build_parser
from chulk.config import load_config
from chulk.server.lifecycle import ControlServerLedger, run_server_command


def test_server_parser_exposes_lifecycle_and_rotation() -> None:
    parser = build_parser()

    start = parser.parse_args(
        ["server", "start", "--host", "127.0.0.1", "--port", "9000"]
    )
    assert start.server_command == "start"
    assert start.port == 9000
    assert parser.parse_args(["server", "status"]).server_command == "status"
    assert parser.parse_args(["server", "stop"]).server_command == "stop"
    assert (
        parser.parse_args(["server", "rotate-token"]).server_command
        == "rotate-token"
    )


def test_server_ledger_uses_leases_and_cooperative_stop(tmp_path) -> None:
    ledger = ControlServerLedger(tmp_path / "control.sqlite")
    now = datetime.now(timezone.utc)
    token = ledger.acquire(host="127.0.0.1", port=8765, now=now)

    status = ledger.status(now=now)
    assert status.state == "running"
    assert status.port == 8765
    assert ledger.request_stop()
    assert ledger.stop_requested(token)
    assert ledger.release(token)
    assert ledger.status().state == "stopped"


def test_server_ledger_reports_expired_owner_as_stale(tmp_path) -> None:
    ledger = ControlServerLedger(tmp_path / "control.sqlite")
    now = datetime.now(timezone.utc)
    ledger.acquire(
        host="127.0.0.1",
        port=8765,
        lease_seconds=1,
        now=now,
    )

    assert ledger.status(now=now + timedelta(seconds=2)).state == "stale"


def test_server_cli_status_stop_and_token_rotation_do_not_print_token(tmp_path) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    output: list[str] = []
    errors: list[str] = []
    ledger = ControlServerLedger(config.runtime_dir / "control.sqlite")
    token = ledger.acquire(host="127.0.0.1", port=8765)

    assert run_server_command(
        "status",
        config=config,
        host="127.0.0.1",
        port=8765,
        allow_remote=False,
        json_output=True,
        output_func=output.append,
        error_func=errors.append,
    ) == 0
    assert json.loads(output[-1])["server"]["state"] == "running"
    assert run_server_command(
        "stop",
        config=config,
        host="127.0.0.1",
        port=8765,
        allow_remote=False,
        json_output=False,
        output_func=output.append,
        error_func=errors.append,
    ) == 0
    assert ledger.stop_requested(token)
    assert run_server_command(
        "rotate-token",
        config=config,
        host="127.0.0.1",
        port=8765,
        allow_remote=False,
        json_output=False,
        output_func=output.append,
        error_func=errors.append,
    ) == 0
    rendered = "\n".join(output)
    assert "control.token" in rendered
    assert len(rendered) < 1_000
    assert errors == []


def test_server_start_refuses_remote_bind_without_explicit_opt_in(tmp_path) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    errors: list[str] = []

    result = run_server_command(
        "start",
        config=config,
        host="0.0.0.0",
        port=8765,
        allow_remote=False,
        json_output=False,
        output_func=lambda _value: None,
        error_func=errors.append,
    )

    assert result == 2
    assert "remote binding requires" in errors[0]
