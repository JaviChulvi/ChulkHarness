"""CLI launcher for the optional operator terminal."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from chulk.config import Config
from chulk.server import ControlApiClient
from chulk.tui import ControlApiDataSource


def run_tui_command(
    *,
    config: Config,
    profile_id: str,
    url: str,
    token_file: Path | str | None,
    refresh_seconds: float,
    no_color: bool,
    error_func: Callable[[str], None],
    app_factory: Callable[..., Any] | None = None,
) -> int:
    """Load the owner credential and run the Textual operator application."""
    path = (
        Path(token_file).expanduser().resolve()
        if token_file is not None
        else config.runtime_dir / "control.token"
    )
    try:
        token = path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("control token file is empty")
        if refresh_seconds <= 0:
            raise ValueError("refresh interval must be greater than zero")
        if app_factory is None:
            try:
                from chulk.tui.app import OperatorApp
            except ImportError as exc:
                raise RuntimeError(
                    "The operator TUI is unavailable; install chulkharness[tui]"
                ) from exc
            app_factory = OperatorApp
        source = ControlApiDataSource(
            ControlApiClient(url, token),
            profile_id=profile_id,
        )
        app = app_factory(
            source,
            refresh_seconds=refresh_seconds,
            no_color=no_color,
        )
        app.run()
    except (OSError, RuntimeError, ValueError) as exc:
        error_func(f"tui error: {exc}")
        return 2
    return 0


__all__ = ["run_tui_command"]
