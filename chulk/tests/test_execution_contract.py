"""Backend-neutral execution contract tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import time

import pytest

from chulk import (
    CommandExecutionRequest,
    ExecutionSessionRequest,
    FileListRequest,
    FileReadRequest,
    FileSearchRequest,
    FileWriteRequest,
    GitWorktreeBackend,
    HostExecutionBackend,
    ProcessLogsRequest,
    ProcessPollRequest,
    ProcessStartRequest,
    ProcessState,
    TemporaryWorkspaceBackend,
)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )


def _python_command(code: str) -> str:
    arguments = [sys.executable, "-c", code]
    if os.name == "nt":
        return subprocess.list2cmdline(arguments)
    return shlex.join(arguments)


def _backend(kind: str, root: Path):
    if kind == "host":
        return HostExecutionBackend(root)
    if kind == "temporary":
        return TemporaryWorkspaceBackend(
            root,
            require_shell_containment=False,
        )
    if kind == "git":
        _git(root, "init")
        _git(root, "config", "user.email", "tests@example.com")
        _git(root, "config", "user.name", "Chulk Tests")
        _git(root, "add", "seed.txt")
        _git(root, "commit", "-m", "initial")
        return GitWorktreeBackend(
            root,
            require_shell_containment=False,
        )
    raise AssertionError(f"Unknown backend kind: {kind}")


@pytest.mark.parametrize(
    "kind",
    [
        "host",
        "temporary",
        pytest.param(
            "git",
            marks=pytest.mark.skipif(
                shutil.which("git") is None,
                reason="Git is required",
            ),
        ),
    ],
)
def test_execution_backend_contract(kind, tmp_path):
    root = tmp_path / kind
    root.mkdir()
    (root / "seed.txt").write_text("seed data\n", encoding="utf-8")
    backend = _backend(kind, root)
    session = backend.open_session(ExecutionSessionRequest(turn_id=f"{kind}-sync"))

    read = session.read_file(FileReadRequest("seed.txt"))
    write = session.write_file(FileWriteRequest("created.txt", "created data"))
    listed = session.list_files(FileListRequest(recursive=True))
    searched = session.search_files(FileSearchRequest("created data"))
    command = session.run_command(
        CommandExecutionRequest(
            "python -c \"from pathlib import Path;"
            "print(Path('created.txt').read_text())\""
        )
    )

    assert read.success is True
    assert read.observation == "seed data\n"
    assert write.success is True
    assert "created.txt" in listed.observation
    assert "created.txt" in searched.observation
    assert command.success is True
    assert command.stdout == "created data\n"
    assert read.metadata["execution_backend"] == backend.name
    assert write.metadata["execution_workspace_id"] == read.metadata[
        "execution_workspace_id"
    ]
    session.close()
    assert session.closed is True
    with pytest.raises(RuntimeError, match="session is closed"):
        session.read_file(FileReadRequest("seed.txt"))
    backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "host",
        "temporary",
        pytest.param(
            "git",
            marks=pytest.mark.skipif(
                shutil.which("git") is None,
                reason="Git is required",
            ),
        ),
    ],
)
async def test_async_execution_backend_contract(kind, tmp_path):
    root = tmp_path / f"{kind}-async"
    root.mkdir()
    (root / "seed.txt").write_text("seed data\n", encoding="utf-8")
    backend = _backend(kind, root)
    session = await backend.open_session_async(
        ExecutionSessionRequest(turn_id=f"{kind}-async")
    )

    write = await session.write_file_async(
        FileWriteRequest("created.txt", "created data")
    )
    read = await session.read_file_async(FileReadRequest("created.txt"))
    searched = await session.search_files_async(
        FileSearchRequest("created data")
    )

    assert write.success is True
    assert read.observation == "created data"
    assert "created.txt" in searched.observation
    await session.aclose()
    assert session.closed is True
    await backend.aclose()


@pytest.mark.parametrize(
    "kind",
    [
        "host",
        "temporary",
        pytest.param(
            "git",
            marks=pytest.mark.skipif(
                shutil.which("git") is None,
                reason="Git is required",
            ),
        ),
    ],
)
def test_managed_process_contract(kind: str, tmp_path: Path) -> None:
    root = tmp_path / f"{kind}-process"
    root.mkdir()
    (root / "seed.txt").write_text("seed data\n", encoding="utf-8")
    backend = _backend(kind, root)
    session = backend.open_session(
        ExecutionSessionRequest(
            conversation_id=f"{kind}-owner",
            turn_id=f"{kind}-process",
        )
    )
    started = session.start_process(
        ProcessStartRequest(_python_command("print('contract-process')"))
    )
    deadline = time.monotonic() + 5
    while True:
        polled = session.poll_process(ProcessPollRequest(started.value))
        if polled.value.state is not ProcessState.RUNNING:
            break
        if time.monotonic() >= deadline:
            raise AssertionError("managed process contract timed out")
        time.sleep(0.01)
    logs = session.read_process_logs(ProcessLogsRequest(started.value))

    assert started.success is True
    assert polled.value.state is ProcessState.EXITED
    assert polled.exit_code == 0
    assert logs.stdout == "contract-process\n"
    session.close()
    backend.close()


@pytest.mark.asyncio
async def test_async_managed_process_contract(tmp_path: Path) -> None:
    backend = HostExecutionBackend(tmp_path)
    session = await backend.open_session_async(
        ExecutionSessionRequest(conversation_id="async-owner", turn_id="async-process")
    )
    started = await session.start_process_async(
        ProcessStartRequest(_python_command("print('async-process')"))
    )
    deadline = time.monotonic() + 5
    while True:
        polled = await session.poll_process_async(ProcessPollRequest(started.value))
        if polled.value.state is not ProcessState.RUNNING:
            break
        if time.monotonic() >= deadline:
            raise AssertionError("async managed process contract timed out")
        await asyncio.sleep(0.01)
    logs = await session.read_process_logs_async(ProcessLogsRequest(started.value))

    assert polled.exit_code == 0
    assert logs.stdout == "async-process\n"
    await session.aclose()
    await backend.aclose()
