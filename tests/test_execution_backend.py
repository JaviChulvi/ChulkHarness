"""Shared execution backend contracts, routing, and host parity."""

from __future__ import annotations

import json

import pytest

from chulk import (
    Agent,
    AgentConfig,
    AsyncAgent,
    Capabilities,
    CommandExecutionRequest,
    ExecutionSessionRequest,
    FileReadRequest,
    FileSearchRequest,
    FileWriteRequest,
    HostExecutionBackend,
)
from chulk.llm import LLMClient
from chulk.tools.files import read_file, write_file
from chulk.tools.shell import ShellExecutionDecision, run_shell_command


class FakeLLM(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, messages, *, max_output_tokens=None) -> str:
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class RecordingHostBackend(HostExecutionBackend):
    def __init__(self, project_root, **kwargs) -> None:
        super().__init__(project_root, **kwargs)
        self.sessions = []

    def open_session(self, request):
        session = super().open_session(request)
        self.sessions.append(session)
        return session


class ContainedPolicy:
    def prepare(self, request):
        return ShellExecutionDecision.allow(
            request.command,
            policy_name="test-contained",
            shell=True,
            containment_applied=True,
        )


def _tool_call(name: str, **arguments) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": name,
            "arguments_json": json.dumps(arguments),
        }
    )


def _final(content: str = "done") -> str:
    return json.dumps({"type": "final_answer", "content": content})


def test_host_session_preserves_file_and_shell_results_with_execution_evidence(tmp_path):
    backend = HostExecutionBackend(tmp_path)
    session = backend.open_session(
        ExecutionSessionRequest(conversation_id="conversation", turn_id="turn")
    )

    direct_write = write_file(
        {"path": "direct.txt", "content": "shared data", "overwrite": False},
        tmp_path,
    )
    session_write = session.write_file(
        FileWriteRequest(path="session.txt", content="shared data")
    )
    direct_read = read_file({"path": "direct.txt"}, tmp_path)
    session_read = session.read_file(FileReadRequest(path="session.txt"))
    search = session.search_files(FileSearchRequest(query="shared data"))
    command = session.run_command(
        CommandExecutionRequest("python -c \"import sys;sys.stdout.write('host-ok')\"")
    )

    assert (session_write.success, session_write.observation) == (
        direct_write.success,
        direct_write.observation.replace("direct.txt", "session.txt"),
    )
    assert (session_read.success, session_read.observation) == (
        direct_read.success,
        direct_read.observation,
    )
    assert "session.txt" in search.observation
    assert command.stdout == "host-ok"
    assert session_write.change_set is not None
    assert session_write.change_set.changes[0].path == "session.txt"
    for result in (session_write, session_read, search, command):
        assert result.metadata["execution_backend"] == "host"
        assert result.metadata["execution_workspace_id"] == "host-turn"
        assert result.metadata["workspace_mode"] == "host"
        assert result.metadata["network_policy"] == "host_inherited"
        assert result.metadata["change_disposition"] == "apply_directly"


def test_host_session_adapts_existing_shell_policy_and_containment(tmp_path):
    backend = HostExecutionBackend(
        tmp_path,
        shell_execution_policy=ContainedPolicy(),
        require_shell_containment=True,
    )
    session = backend.open_session(ExecutionSessionRequest(turn_id="contained"))

    result = session.run_command(
        CommandExecutionRequest("python -c \"import sys;sys.stdout.write('safe')\"")
    )

    assert result.success is True
    assert result.metadata["effective_policy"] == "test-contained"
    assert result.metadata["containment"] == "contained"
    assert result.metadata["execution_policy"]["containment_applied"] is True


def test_closed_host_session_rejects_operations(tmp_path):
    session = HostExecutionBackend(tmp_path).open_session(
        ExecutionSessionRequest(turn_id="closed")
    )
    session.close()
    session.close()

    with pytest.raises(RuntimeError, match="session is closed"):
        session.read_file(FileReadRequest("missing.txt"))


@pytest.mark.asyncio
async def test_host_backend_async_surface_uses_same_workspace(tmp_path):
    backend = HostExecutionBackend(tmp_path)
    session = await backend.open_session_async(ExecutionSessionRequest(turn_id="async"))

    await session.write_file_async(FileWriteRequest("async.txt", "async data"))
    result = await session.read_file_async(FileReadRequest("async.txt"))
    await session.aclose()

    assert result.success is True
    assert result.observation == "async data"
    assert session.closed is True


def test_runtime_routes_default_tools_through_one_turn_scoped_session(tmp_path):
    backend = RecordingHostBackend(tmp_path)
    (tmp_path / "input.txt").write_text("hello", encoding="utf-8")
    facade = Agent(
        config=AgentConfig(project_root=tmp_path),
        capabilities=Capabilities.read_only(),
        execution_backend=backend,
        llm=FakeLLM(
            [
                _tool_call("read_file", path="input.txt"),
                _final(),
                _tool_call("read_file", path="input.txt"),
                _final(),
            ]
        ),
        skills=[],
    )

    result = facade.run_result("read the input")
    second_result = facade.run_result("read it again")

    assert result.tool_calls[0].success is True
    assert result.tool_calls[0].metadata["execution_backend"] == "host"
    assert second_result.tool_calls[0].success is True
    assert len(backend.sessions) == 2
    assert backend.sessions[0].workspace.workspace_id != backend.sessions[1].workspace.workspace_id
    assert all(session.closed for session in backend.sessions)
    assert facade.runtime.tool_contexts._contexts == {}
    trace_events = [
        json.loads(line)
        for line in facade.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    completed = [
        event["payload"]
        for event in trace_events
        if event["type"] == "tool_call_completed"
    ]
    assert completed[0]["metadata"]["execution_backend"] == "host"
    assert completed[0]["metadata"]["workspace_mode"] == "host"


@pytest.mark.asyncio
async def test_async_runtime_closes_turn_session_through_async_lifecycle(tmp_path):
    backend = RecordingHostBackend(tmp_path)
    (tmp_path / "input.txt").write_text("hello", encoding="utf-8")
    facade = AsyncAgent(
        config=AgentConfig(project_root=tmp_path),
        capabilities=Capabilities.read_only(),
        execution_backend=backend,
        llm=FakeLLM([_tool_call("read_file", path="input.txt"), _final()]),
        skills=[],
    )

    result = await facade.run_result("read the input")

    assert result.tool_calls[0].success is True
    assert len(backend.sessions) == 1
    assert backend.sessions[0].closed is True
    assert facade.runtime.tool_contexts._contexts == {}


def test_direct_shell_behavior_matches_host_session(tmp_path):
    command = "python -c \"import sys;sys.stdout.write('parity')\""
    direct = run_shell_command({"command": command}, tmp_path)
    session = HostExecutionBackend(tmp_path).open_session(
        ExecutionSessionRequest(turn_id="parity")
    )
    routed = session.run_command(CommandExecutionRequest(command))

    assert (
        routed.success,
        routed.observation,
        routed.stdout,
        routed.stderr,
        routed.exit_code,
        routed.error,
        routed.failure_kind,
    ) == (
        direct.success,
        direct.observation,
        direct.stdout,
        direct.stderr,
        direct.exit_code,
        direct.error,
        direct.failure_kind,
    )
