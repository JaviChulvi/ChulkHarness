"""CLI contract tests for launching the optional operator TUI."""

from __future__ import annotations

from pathlib import Path

from chulk.cli.parser import build_parser
from chulk.cli.tui import run_tui_command
from chulk.config import load_config


class FakeApp:
    instances: list["FakeApp"] = []

    def __init__(
        self,
        source,
        *,
        refresh_seconds: float,
        no_color: bool,
    ) -> None:
        self.source = source
        self.refresh_seconds = refresh_seconds
        self.no_color = no_color
        self.ran = False
        self.instances.append(self)

    def run(self) -> None:
        self.ran = True


def test_tui_parser_exposes_safe_connection_options() -> None:
    args = build_parser().parse_args(
        [
            "tui",
            "--url",
            "http://localhost:9000",
            "--token-file",
            "owner.token",
            "--refresh-seconds",
            "4",
            "--profile",
            "operations",
            "--no-color",
        ]
    )

    assert args.command == "tui"
    assert args.url == "http://localhost:9000"
    assert args.token_file == Path("owner.token")
    assert args.refresh_seconds == 4
    assert args.profile == "operations"
    assert args.no_color


def test_tui_launcher_reads_token_file_without_exposing_raw_token(tmp_path) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    token_file = tmp_path / "owner.token"
    token_file.write_text("owner-secret\n", encoding="utf-8")
    errors: list[str] = []

    result = run_tui_command(
        config=config,
        profile_id="operations",
        url="http://127.0.0.1:8765",
        token_file=token_file,
        refresh_seconds=3,
        no_color=True,
        error_func=errors.append,
        app_factory=FakeApp,
    )

    app = FakeApp.instances[-1]
    assert result == 0
    assert errors == []
    assert app.ran
    assert app.refresh_seconds == 3
    assert app.no_color
    assert app.source.profile_id == "operations"
    assert app.source.client.token == "owner-secret"


def test_tui_launcher_reports_missing_token_without_creating_it(tmp_path) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    errors: list[str] = []

    result = run_tui_command(
        config=config,
        profile_id="default",
        url="http://127.0.0.1:8765",
        token_file=None,
        refresh_seconds=2,
        no_color=False,
        error_func=errors.append,
        app_factory=FakeApp,
    )

    assert result == 2
    assert "control.token" in errors[0]
    assert not (config.runtime_dir / "control.token").exists()
