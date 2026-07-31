"""CLI tests for gateway lifecycle, routes, and one-time pairing."""

from __future__ import annotations

import json

from chulk.cli.gateway import run_gateway_command
from chulk.cli.parser import build_parser
from chulk.config import load_config
from chulk.gateway import SQLiteGatewayLedger, SQLiteGatewayRouter
from chulk.main import main
from chulk.profiles import SQLiteProfileStore


def _services(tmp_path):
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "gemini",
            "CHULK_MODEL": "gemini-test",
            "CHULK_GEMINI_API_KEY": "fake",
        }
    )
    control_path = config.runtime_dir / "control.sqlite"
    profiles = SQLiteProfileStore(control_path, base_config=config)
    return (
        SQLiteGatewayLedger(control_path),
        SQLiteGatewayRouter(control_path),
        profiles,
    )


def _run(
    command,
    *,
    ledger,
    router,
    profiles,
    outputs,
    errors,
    route_command=None,
    route_id=None,
    profile_id=None,
    principal_id=None,
    json_output=True,
):
    return run_gateway_command(
        command,
        ledger=ledger,
        router=router,
        profile_store=profiles,
        start_func=lambda: 0,
        adapter="telegram",
        account_id="primary",
        route_command=route_command,
        route_id=route_id,
        profile_id=profile_id,
        principal_id=principal_id,
        destination_id=None,
        thread_id=None,
        pairing_ttl_seconds=600,
        include_disabled=False,
        json_output=json_output,
        output_func=outputs.append,
        error_func=errors.append,
    )


def test_gateway_parser_exposes_lifecycle_routes_and_pairing() -> None:
    parser = build_parser()

    assert parser.parse_args(["gateway", "start"]).gateway_command == "start"
    assert parser.parse_args(["gateway", "status"]).gateway_command == "status"
    assert parser.parse_args(["gateway", "stop"]).gateway_command == "stop"
    assert (
        parser.parse_args(["gateway", "routes", "list"]).route_command
        == "list"
    )
    assert (
        parser.parse_args(
            ["gateway", "pair", "--profile", "default"]
        ).gateway_command
        == "pair"
    )


def test_gateway_start_accepts_discord_as_an_optional_adapter(tmp_path) -> None:
    ledger, router, profiles = _services(tmp_path)
    started: list[str] = []

    result = run_gateway_command(
        "start",
        ledger=ledger,
        router=router,
        profile_store=profiles,
        start_func=lambda: started.append("discord") or 0,
        adapter="discord",
        account_id="team-bot",
        route_command=None,
        route_id=None,
        profile_id=None,
        principal_id=None,
        destination_id=None,
        thread_id=None,
        pairing_ttl_seconds=600,
        include_disabled=False,
        json_output=True,
        output_func=lambda _value: None,
        error_func=lambda _value: None,
    )

    assert result == 0
    assert started == ["discord"]


def test_main_dispatches_discord_gateway_without_starting_telegram(
    monkeypatch,
    tmp_path,
) -> None:
    import chulk.discord.main as discord_main

    calls: list[tuple[object, object]] = []

    def run_discord(
        config,
        *,
        control_db_path,
        profile_runtime_factory,
        account_id,
    ):
        calls.append((control_db_path, profile_runtime_factory))
        assert config.project_root == tmp_path.resolve()
        assert account_id == "team-bot"
        return 0

    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(discord_main, "run_discord_gateway", run_discord)

    assert (
        main(
            [
                "gateway",
                "start",
                "--adapter",
                "discord",
                "--account",
                "team-bot",
            ]
        )
        == 0
    )
    assert calls
    assert calls[0][0] == (tmp_path / ".chulk" / "control.sqlite").resolve()


def test_gateway_cli_controls_status_and_cooperative_stop(tmp_path) -> None:
    ledger, router, profiles = _services(tmp_path)
    outputs: list[str] = []
    errors: list[str] = []
    ledger.start_adapter("telegram", "primary")

    assert (
        _run(
            "status",
            ledger=ledger,
            router=router,
            profiles=profiles,
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    assert json.loads(outputs[-1])["adapters"][0]["state"] == "running"
    assert (
        _run(
            "stop",
            ledger=ledger,
            router=router,
            profiles=profiles,
            outputs=outputs,
            errors=errors,
        )
        == 0
    )
    assert ledger.adapter_status("telegram", "primary").stop_requested


def test_gateway_cli_adds_lists_disables_routes_and_creates_pairing(tmp_path) -> None:
    ledger, router, profiles = _services(tmp_path)
    outputs: list[str] = []
    errors: list[str] = []

    assert (
        _run(
            "routes",
            ledger=ledger,
            router=router,
            profiles=profiles,
            outputs=outputs,
            errors=errors,
            route_command="add",
            profile_id="default",
            principal_id="7",
        )
        == 0
    )
    route_id = json.loads(outputs[-1])["route_id"]
    assert (
        _run(
            "routes",
            ledger=ledger,
            router=router,
            profiles=profiles,
            outputs=outputs,
            errors=errors,
            route_command="list",
        )
        == 0
    )
    assert json.loads(outputs[-1])["routes"][0]["profile_id"] == "default"
    assert (
        _run(
            "routes",
            ledger=ledger,
            router=router,
            profiles=profiles,
            outputs=outputs,
            errors=errors,
            route_command="remove",
            route_id=route_id,
        )
        == 0
    )
    assert (
        _run(
            "pair",
            ledger=ledger,
            router=router,
            profiles=profiles,
            outputs=outputs,
            errors=errors,
            profile_id="default",
        )
        == 0
    )
    pairing = json.loads(outputs[-1])
    assert pairing["status"] == "pairing_created"
    assert len(pairing["code"]) >= 24
