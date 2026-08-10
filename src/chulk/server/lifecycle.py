"""Cooperative lifecycle and CLI operations for the local control server."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from chulk.config import Config
from chulk.profiles.store import CONTROL_MIGRATIONS
from chulk.server.app import ServerDependencyError, create_control_app
from chulk.server.security import ControlTokenStore
from chulk.storage import initialize_sqlite_database, sqlite_connection


@dataclass(frozen=True, slots=True)
class ControlServerStatus:
    state: str
    host: str | None = None
    port: int | None = None
    pid: int | None = None
    lease_until: datetime | None = None
    stop_requested: bool = False
    started_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "host": self.host,
            "port": self.port,
            "pid": self.pid,
            "lease_until": (
                self.lease_until.isoformat()
                if self.lease_until is not None
                else None
            ),
            "stop_requested": self.stop_requested,
            "started_at": self.started_at,
        }


class ControlServerLedger:
    """Lease one local server process and expose cooperative stop state."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        initialize_sqlite_database(self.db_path, migrations=CONTROL_MIGRATIONS)

    def acquire(
        self,
        *,
        host: str,
        port: int,
        lease_seconds: int = 30,
        now: datetime | None = None,
    ) -> str:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be greater than zero")
        observed = _observed(now)
        token = uuid4().hex
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM control_server WHERE singleton = 1"
            ).fetchone()
            if (
                row is not None
                and row["state"] == "running"
                and row["lease_until"] is not None
                and _decode(str(row["lease_until"])) > observed
            ):
                raise RuntimeError("control server is already running")
            conn.execute(
                """
                INSERT INTO control_server (
                    singleton, state, instance_token, pid, host, port,
                    lease_until, stop_requested, started_at, updated_at
                ) VALUES (1, 'running', ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    state = 'running',
                    instance_token = excluded.instance_token,
                    pid = excluded.pid,
                    host = excluded.host,
                    port = excluded.port,
                    lease_until = excluded.lease_until,
                    stop_requested = 0,
                    started_at = excluded.started_at,
                    stopped_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    token,
                    os.getpid(),
                    host,
                    port,
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    _encode(observed),
                ),
            )
        return token

    def renew(
        self,
        instance_token: str,
        *,
        lease_seconds: int = 30,
        now: datetime | None = None,
    ) -> bool:
        observed = _observed(now)
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE control_server
                SET lease_until = ?, updated_at = ?
                WHERE singleton = 1 AND state = 'running'
                  AND instance_token = ? AND stop_requested = 0
                """,
                (
                    _encode(observed + timedelta(seconds=lease_seconds)),
                    _encode(observed),
                    instance_token,
                ),
            )
        return cursor.rowcount == 1

    def stop_requested(self, instance_token: str) -> bool:
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT stop_requested FROM control_server
                WHERE singleton = 1 AND state = 'running' AND instance_token = ?
                """,
                (instance_token,),
            ).fetchone()
        return row is None or bool(row["stop_requested"])

    def request_stop(self) -> bool:
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE control_server
                SET stop_requested = 1, updated_at = ?
                WHERE singleton = 1 AND state = 'running'
                  AND lease_until > ?
                """,
                (_encode(_observed(None)), _encode(_observed(None))),
            )
        return cursor.rowcount == 1

    def release(self, instance_token: str) -> bool:
        observed = _encode(_observed(None))
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE control_server
                SET state = 'stopped', instance_token = NULL,
                    lease_until = NULL, stop_requested = 0,
                    stopped_at = ?, updated_at = ?
                WHERE singleton = 1 AND instance_token = ?
                """,
                (observed, observed, instance_token),
            )
        return cursor.rowcount == 1

    def status(self, *, now: datetime | None = None) -> ControlServerStatus:
        observed = _observed(now)
        with sqlite_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM control_server WHERE singleton = 1"
            ).fetchone()
        if row is None:
            return ControlServerStatus("stopped")
        lease = (
            _decode(str(row["lease_until"]))
            if row["lease_until"] is not None
            else None
        )
        state = str(row["state"])
        if state == "running" and (lease is None or lease <= observed):
            state = "stale"
        return ControlServerStatus(
            state=state,
            host=str(row["host"]) if row["host"] is not None else None,
            port=int(row["port"]) if row["port"] is not None else None,
            pid=int(row["pid"]) if row["pid"] is not None else None,
            lease_until=lease,
            stop_requested=bool(row["stop_requested"]),
            started_at=(
                str(row["started_at"]) if row["started_at"] is not None else None
            ),
        )


async def serve_control_server(
    config: Config,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    allow_remote: bool = False,
    enable_eval_dashboard: bool = False,
) -> int:
    """Run Uvicorn until signal or cooperative `server stop`."""
    if not allow_remote and not _is_loopback(host):
        raise ValueError("remote binding requires --allow-remote")
    if port < 1 or port > 65_535:
        raise ValueError("port must be between 1 and 65535")
    try:
        import uvicorn
    except ImportError as exc:
        raise ServerDependencyError(
            "Control server dependencies are unavailable; install chulkharness[server]"
        ) from exc
    ledger = ControlServerLedger(config.runtime_dir / "control.sqlite")
    instance_token = ledger.acquire(host=host, port=port)
    watcher: asyncio.Task[None] | None = None
    try:
        application = create_control_app(
            config,
            enable_eval_dashboard=enable_eval_dashboard,
        )
        server = uvicorn.Server(
            uvicorn.Config(
                application,
                host=host,
                port=port,
                access_log=False,
                server_header=False,
            )
        )

        async def monitor() -> None:
            while not server.should_exit:
                await asyncio.sleep(5)
                if ledger.stop_requested(instance_token):
                    server.should_exit = True
                    return
                if not ledger.renew(instance_token):
                    server.should_exit = True
                    return

        watcher = asyncio.create_task(monitor())
        await server.serve()
    finally:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        ledger.release(instance_token)
    return 0


def run_server_command(
    command: str,
    *,
    config: Config,
    host: str,
    port: int,
    allow_remote: bool,
    enable_eval_dashboard: bool = False,
    json_output: bool,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
    start_func: Callable[[], int] | None = None,
) -> int:
    """Execute one deterministic control-server lifecycle operation."""
    ledger = ControlServerLedger(config.runtime_dir / "control.sqlite")
    tokens = ControlTokenStore(config.runtime_dir / "control.token")
    try:
        if command == "start":
            return (
                start_func()
                if start_func is not None
                else asyncio.run(
                    serve_control_server(
                        config,
                        host=host,
                        port=port,
                        allow_remote=allow_remote,
                        enable_eval_dashboard=enable_eval_dashboard,
                    )
                )
            )
        if command == "status":
            status = ledger.status()
            payload = {"ok": True, "server": status.to_dict()}
            output_func(
                json.dumps(payload, sort_keys=True)
                if json_output
                else _format_status(status)
            )
            return 0
        if command == "stop":
            requested = ledger.request_stop()
            payload = {
                "ok": requested,
                "status": "stop_requested" if requested else "not_running",
            }
            output_func(
                json.dumps(payload, sort_keys=True)
                if json_output
                else (
                    "Control server stop requested."
                    if requested
                    else "Control server is not running."
                )
            )
            return 0 if requested else 2
        if command == "rotate-token":
            tokens.rotate()
            payload = {
                "ok": True,
                "status": "token_rotated",
                "token_path": str(tokens.path),
            }
            output_func(
                json.dumps(payload, sort_keys=True)
                if json_output
                else f"Rotated control token at {tokens.path}."
            )
            return 0
        raise ValueError(f"unknown server command: {command}")
    except (OSError, RuntimeError, ValueError) as exc:
        payload = {"ok": False, "status": "server_error", "error": str(exc)}
        if json_output:
            output_func(json.dumps(payload, sort_keys=True))
        else:
            error_func(f"server error: {exc}")
        return 2


def _format_status(status: ControlServerStatus) -> str:
    if status.state == "stopped":
        return "Control server is stopped."
    return (
        f"Control server {status.state} at {status.host}:{status.port} "
        f"pid={status.pid} stop_requested={status.stop_requested}"
    )


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _observed(value: datetime | None) -> datetime:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _encode(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decode(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


__all__ = [
    "ControlServerLedger",
    "ControlServerStatus",
    "run_server_command",
    "serve_control_server",
]
