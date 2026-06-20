"""Tests for the public Chulk API."""

import asyncio
import gc
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import warnings
from typing import Annotated, Literal, get_type_hints

import pytest

import chulk.runtime as runtime_module
from chulk import (
    Agent,
    AgentConfig,
    AgentEvent,
    AsyncAgent,
    AsyncChatAgent,
    AgentPreset,
    ChatAgent,
    MCP,
    PermissionDecision,
    PermissionDecisionRecord,
    PermissionRequest,
    PlanResult,
    RunResult,
    Skills,
    Tool,
    ToolPermissionLevel,
    Tools,
    agent,
    skills,
    tool,
    tools,
)
from chulk.config import DEFAULT_DEEPSEEK_MODEL, DEFAULT_LOCAL_MODEL, DEFAULT_MODEL, load_config
from chulk.core.actions import FinalAnswerAction
from chulk.llm import FallbackChain, LLMActionResult, LLMCapabilities, LLMClient, LLMError
from chulk.presets import SoftwareEngineer, software_engineer
from chulk.presets.software_engineer import DEFAULT_AGENT_PLAYBOOK, SOFTWARE_ENGINEER_SYSTEM_PROMPT
from chulk.tools import (
    PermissionDecision as ToolsPermissionDecision,
    PermissionDecisionRecord as ToolsPermissionDecisionRecord,
    PermissionRequest as ToolsPermissionRequest,
    ToolFailureKind,
    ToolRegistry,
    ToolPermissionLevel as ToolsToolPermissionLevel,
)


class FakeLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class StreamingFakeLLMClient(FakeLLMClient):
    capabilities = LLMCapabilities(supports_streaming=True)


class FailingLLMClient(LLMClient):
    provider = "failing"
    model = "broken"

    def complete(self, messages: list[dict[str, str]]) -> str:
        raise LLMError("provider unavailable")


class HostedMCPRecordingLLM(LLMClient):
    def __init__(self) -> None:
        self.hosted_mcp_servers = None

    def complete_action(self, messages: list[dict[str, str]], **kwargs) -> LLMActionResult:
        self.hosted_mcp_servers = kwargs.get("hosted_mcp_servers")
        return LLMActionResult(
            action=FinalAnswerAction(type="final_answer", content="hosted mcp captured"),
            raw_response=json.dumps({"type": "final_answer", "content": "hosted mcp captured"}),
        )


def write_skill(root, name: str, content: str) -> None:
    skill_dir = root / ".chulk" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def test_public_api_exports_capitalized_aliases():
    assert Agent is agent
    assert Tool is tool
    assert Tools is tools
    assert Skills is skills
    assert SoftwareEngineer is software_engineer
    assert PermissionDecision is ToolsPermissionDecision
    assert PermissionDecisionRecord is ToolsPermissionDecisionRecord
    assert PermissionRequest is ToolsPermissionRequest
    assert ToolPermissionLevel is ToolsToolPermissionLevel
    assert callable(Skills.only)
    assert callable(Skills.pin)


def test_runtime_create_agent_type_hints_resolve():
    hints = get_type_hints(runtime_module.create_agent)

    assert "event_sink" in hints


def test_public_chat_agent_disables_default_tools_and_skills(tmp_path):
    config = AgentConfig(project_root=tmp_path)
    handle = ChatAgent(
        config=config,
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "chat only"})]),
    )

    result = handle.run_result("hello")

    assert result.content == "chat only"
    assert handle.runtime.tool_registry.list_tools() == []
    assert handle.runtime.skill_registry.list_skills() == []


def test_public_chat_agent_rejects_tool_configuration(tmp_path):
    with pytest.raises(ValueError, match="tools"):
        ChatAgent(config=AgentConfig(project_root=tmp_path), tools=[])


@pytest.mark.asyncio
async def test_public_async_chat_agent_disables_default_tools_and_skills(tmp_path):
    handle = AsyncChatAgent(
        config=AgentConfig(project_root=tmp_path),
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "async chat"})]),
    )

    result = await handle.run_result("hello")

    assert result.content == "async chat"
    assert handle.runtime.tool_registry.list_tools() == []


def test_public_agent_preset_chat_disables_default_tools_and_skills(tmp_path):
    config = AgentConfig(project_root=tmp_path)
    handle = Agent(
        config=config,
        preset=AgentPreset.chat(),
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "preset chat"})]),
    )

    result = handle.run_result("hello")

    assert result.content == "preset chat"


def test_software_engineer_preset_loads_default_agent_playbook():
    preset = SoftwareEngineer()

    assert "# Default Agent Playbook" in DEFAULT_AGENT_PLAYBOOK
    assert "Read the relevant code before making claims" in SOFTWARE_ENGINEER_SYSTEM_PROMPT
    assert "Use `search_files` to find symbols" in SOFTWARE_ENGINEER_SYSTEM_PROMPT
    assert "If a tool returns `invalid_arguments`" in SOFTWARE_ENGINEER_SYSTEM_PROMPT
    assert preset.system_prompt == SOFTWARE_ENGINEER_SYSTEM_PROMPT


def test_public_agent_with_preset_injects_default_agent_playbook(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    class PromptAwareLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            system_prompt = messages[0]["content"]
            assert "# Default Agent Playbook" in system_prompt
            assert "Treat generated tool arguments as untrusted input" in system_prompt
            assert "Use `apply_patch` for edits to existing text files" in system_prompt
            assert "memory tools only for durable user, project, preference, or prior-work facts" in system_prompt
            return json.dumps({"type": "final_answer", "content": "preset prompt loaded"})

    handle = Agent(config=config, preset=SoftwareEngineer(), llm=PromptAwareLLM(), tools=[], skills=[])

    assert handle.run("hello") == "preset prompt loaded"


def test_public_agent_run_accepts_stream_delta_callback(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    handle = Agent(
        config=config,
        llm=StreamingFakeLLMClient([json.dumps({"type": "final_answer", "content": "streamed callback"})]),
        tools=[],
        skills=[],
    )
    deltas: list[str] = []

    response = handle.run("hello", on_delta=deltas.append)

    assert response == "streamed callback"
    assert "".join(deltas) == "streamed callback"
    assert handle.state.final_answer == "streamed callback"


def test_public_agent_dispatches_events_from_constructor_and_run(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    constructor_events: list[AgentEvent] = []
    run_events: list[AgentEvent] = []
    deltas: list[str] = []
    handle = Agent(
        config=config,
        llm=StreamingFakeLLMClient([json.dumps({"type": "final_answer", "content": "evented answer"})]),
        tools=[],
        skills=[],
        on_event=constructor_events.append,
    )

    response = handle.run("hello", on_delta=deltas.append, on_event=run_events.append)

    assert response == "evented answer"
    assert "".join(deltas) == "evented answer"
    assert constructor_events[0].type == "turn_started"
    assert [event.type for event in run_events if event.type.startswith("model_stream_")] == [
        "model_stream_started",
        "model_stream_delta",
        "model_stream_completed",
    ]


def test_public_agent_run_result_returns_structured_turn_metadata(tmp_path):
    skill_dir = tmp_path / ".chulk" / "skills" / "files"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Files Skill\n\nUse this skill for file work.\n", encoding="utf-8")
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    @Tool
    def echo_label(label: str) -> str:
        """Echo a label."""
        return f"echo: {label}"

    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "echo_label",
                    "arguments_json": json.dumps({"label": "sdk"}),
                    "plan_json": "{}",
                    "step_update_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "structured answer"}),
        ]
    )
    handle = Agent(config=config, llm=llm, tools=[echo_label], skills=[Skills.files])

    result = handle.run_result("edit a file and echo the sdk label")

    assert isinstance(result, RunResult)
    assert result.content == "structured answer"
    assert result.status == "completed"
    assert result.turn_id == handle.state.turns[-1].turn_id
    assert result.conversation_id == handle.conversation_id
    assert result.trace_path == handle.trace_path
    assert result.usage is not None
    assert result.context_report is not None
    assert result.tool_calls[0]["tool_name"] == "echo_label"
    assert result.observations[0]["tool_name"] == "echo_label"
    assert result.loaded_skill_names == ["files"]
    result_dict = result.to_dict()
    assert result_dict["content"] == "structured answer"
    assert result_dict["trace_path"] == str(handle.trace_path)
    assert result_dict["tool_calls"][0]["tool_name"] == "echo_label"
    assert result_dict["plan"] is None


def test_public_agent_run_accepts_adapter_context(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    class ContextAwareLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            system_prompt = messages[0]["content"]
            assert "Employee handbook" in system_prompt
            assert "locale: es-ES" in system_prompt
            return json.dumps({"type": "final_answer", "content": "context accepted"})

    handle = Agent(config=config, llm=ContextAwareLLM(), tools=[], skills=[])

    response = handle.run(
        "answer from docs",
        context_sections=[{"id": "doc-1", "title": "Employee handbook", "content": "Use PTO before year end."}],
        prompt_profile="polp-search",
        locale="es-ES",
        extension_metadata={"confidence": 0.9},
    )

    assert response == "context accepted"
    assert handle.state.turns[0].context_sections[0].id == "doc-1"
    assert handle.state.turns[0].extension_metadata == {"confidence": 0.9}


def test_public_agent_run_result_exposes_extension_metadata(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    handle = Agent(
        config=config,
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "structured"})]),
        tools=[],
        skills=[],
    )

    result = handle.run_result("hello", extension_metadata={"confidence": 0.7})

    assert result.content == "structured"
    assert result.status == "completed"
    assert result.extension_metadata == {"confidence": 0.7}
    assert result.tool_calls == []
    assert result.observations == []


def test_public_agent_redacts_streamed_and_final_output(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    events: list[AgentEvent] = []

    def redact(_event_type: str, text: str, _metadata: dict) -> str:
        return text.replace("SECRET", "[redacted]")

    handle = Agent(
        config=config,
        llm=StreamingFakeLLMClient([json.dumps({"type": "final_answer", "content": "SECRET answer"})]),
        tools=[],
        skills=[],
        on_event=events.append,
        redaction_callback=redact,
    )
    deltas = []

    response = handle.run("hello", on_delta=deltas.append)

    assert response == "[redacted] answer"
    assert "".join(deltas) == "[redacted] answer"
    assert any(event.type == "final_answer" and event.payload["content"] == "[redacted] answer" for event in events)


@pytest.mark.asyncio
async def test_public_async_agent_awaits_decorated_tool(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    @Tool
    async def async_label(label: str) -> str:
        """Return a label asynchronously."""
        return f"async: {label}"

    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "async_label",
                    "arguments_json": json.dumps({"label": "public"}),
                    "plan_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "async done"}),
        ]
    )
    handle = AsyncAgent(config=config, llm=llm, tools=[async_label], skills=[])

    result = await handle.run_result("use async")

    assert result.content == "async done"
    assert result.tool_calls[0]["success"] is True


@pytest.mark.asyncio
async def test_public_async_agent_does_not_block_event_loop_for_llm(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    events = []

    class BlockingLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            events.append("llm-start")
            time.sleep(0.1)
            events.append("llm-end")
            return json.dumps({"type": "final_answer", "content": "async response"})

    async def tick() -> None:
        await asyncio.sleep(0.01)
        events.append("tick")

    handle = AsyncAgent(config=config, llm=BlockingLLM(), tools=[], skills=[])

    result, _ = await asyncio.gather(handle.run_result("hello"), tick())

    assert result.content == "async response"
    assert events.index("tick") < events.index("llm-end")


@pytest.mark.asyncio
async def test_public_async_agent_approve_awaits_decorated_tool(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    calls = []

    @Tool
    async def async_label(label: str) -> str:
        """Return a label asynchronously."""
        calls.append(label)
        return f"async: {label}"

    llm = FakeLLMClient(
        [
            _plan_response(_plan_payload("Run async approval tool")),
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "async_label",
                    "arguments_json": json.dumps({"label": "approved"}),
                    "plan_json": "{}",
                    "step_update_json": "{}",
                }
            ),
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {"step_id": "1", "status": "completed", "evidence": "Async tool completed."}
                    ),
                }
            ),
            json.dumps({"type": "final_answer", "content": "async approval done"}),
        ]
    )
    handle = AsyncAgent(config=config, llm=llm, tools=[async_label], skills=[])

    plan_result = await handle.plan_result("plan async approval")
    result = await handle.approve_result()

    assert plan_result.status == "waiting_for_approval"
    assert result.content == "async approval done"
    assert result.status == "completed"
    assert calls == ["approved"]
    assert result.tool_calls[0]["success"] is True
    assert result.tool_calls[0]["failure_kind"] is None


def test_public_decorated_async_tool_sync_rejection_closes_coroutine():
    @Tool
    async def async_label(label: str) -> str:
        """Return a label asynchronously."""
        return f"async: {label}"

    registry = ToolRegistry()
    registry.register(async_label)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = registry.run("async_label", {"label": "sync"})
        gc.collect()

    assert not result.success
    assert result.failure_kind == ToolFailureKind.ASYNC_REQUIRED
    assert not any("was never awaited" in str(warning.message) for warning in caught)


def test_public_decorated_sync_tool_returning_awaitable_sync_rejection_closes_coroutine():
    async def inner_label() -> str:
        return "async result"

    @Tool
    def deferred_label() -> str:
        """Return an awaitable from a sync tool."""
        return inner_label()

    registry = ToolRegistry()
    registry.register(deferred_label)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = registry.run("deferred_label", {})
        gc.collect()

    assert not result.success
    assert result.failure_kind == ToolFailureKind.ASYNC_REQUIRED
    assert not any("was never awaited" in str(warning.message) for warning in caught)
    async_result = asyncio.run(registry.run_async("deferred_label", {}))
    assert async_result.success
    assert async_result.observation == "async result"


def test_public_agent_runs_decorated_tool(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    @Tool
    def echo_label(label: str) -> str:
        """Echo a label."""
        return f"echo: {label}"

    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "echo_label",
                    "arguments_json": json.dumps({"label": "public"}),
                    "plan_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "echoed"}),
        ]
    )

    handle = Agent(config=config, llm=llm, tools=[echo_label], skills=[])

    response = handle.run("echo the public label")

    assert response == "echoed"
    assert handle("echo again") == "echoed"
    assert "echo_label" in llm.requests[0][0]["content"]
    assert handle.state.tool_calls[0]["tool_name"] == "echo_label"


def test_public_permission_callback_allows_confirming_tool(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    calls = []
    approvals = []

    @Tool(requires_confirmation=True, permission_level=ToolPermissionLevel.EXTERNAL_SERVICE)
    def risky_lookup(value: str) -> str:
        """Run a confirming lookup."""
        calls.append(value)
        return f"looked up {value}"

    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "risky_lookup",
                    "arguments_json": json.dumps({"value": "ok"}),
                    "plan_json": "{}",
                    "step_update_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "approved"}),
        ]
    )

    def approve(request, record):
        approvals.append((request.tool_name, record.decision))
        return True

    handle = Agent(config=config, llm=llm, tools=[risky_lookup], skills=[], permission_callback=approve)

    assert handle.run("run risky lookup") == "approved"
    assert calls == ["ok"]
    assert approvals == [("risky_lookup", PermissionDecision.ASK)]


def test_public_permission_callback_denies_confirming_tool(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    calls = []

    @Tool(requires_confirmation=True, permission_level=ToolPermissionLevel.EXTERNAL_SERVICE)
    def risky_lookup(value: str) -> str:
        """Run a confirming lookup."""
        calls.append(value)
        return f"looked up {value}"

    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "risky_lookup",
                    "arguments_json": json.dumps({"value": "nope"}),
                    "plan_json": "{}",
                    "step_update_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "denied"}),
        ]
    )
    handle = Agent(config=config, llm=llm, tools=[risky_lookup], skills=[], permission_callback=lambda _request, _record: False)

    result = handle.run_result("run risky lookup")

    assert result.content == "denied"
    assert calls == []
    assert result.tool_calls[0]["error"] == "permission_denied"


def test_public_confirming_tool_is_denied_without_callback(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    calls = []

    @Tool(requires_confirmation=True, permission_level=ToolPermissionLevel.SHELL)
    def risky_shell(value: str) -> str:
        """Run a confirming shell-like action."""
        calls.append(value)
        return "ran"

    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "risky_shell",
                    "arguments_json": json.dumps({"value": "run"}),
                    "plan_json": "{}",
                    "step_update_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "blocked"}),
        ]
    )

    result = Agent(config=config, llm=llm, tools=[risky_shell], skills=[]).run_result("run it")

    assert result.content == "blocked"
    assert calls == []
    assert result.tool_calls[0]["error"] == "permission_denied"


def test_public_agent_can_approve_workspace_shell_tool(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "run_cmd",
                    "arguments_json": json.dumps({"command": "printf sdk"}),
                    "plan_json": "{}",
                    "step_update_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "shell approved"}),
        ]
    )

    result = Agent(
        config=config,
        llm=llm,
        tools=[Tools.run_cmd],
        skills=[],
        permission_callback=lambda _request, _record: PermissionDecision.ALLOW,
    ).run_result("run shell")

    assert result.content == "shell approved"
    assert result.tool_calls[0]["success"] is True
    assert "stdout:\nsdk" in result.observations[0]["content"]


def test_public_agent_can_pin_skill(tmp_path):
    skill_dir = tmp_path / ".chulk" / "skills" / "files"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Files Skill\n\nUse this skill for file work.\n", encoding="utf-8")
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    class SkillAwareLLM(LLMClient):
        def complete(self, messages: list[dict[str, str]]) -> str:
            system_prompt = messages[0]["content"]
            assert "Skill: files" in system_prompt
            assert "# Files Skill" in system_prompt
            return json.dumps({"type": "final_answer", "content": "files pinned"})

    handle = Agent(config=config, llm=SkillAwareLLM(), tools=[Tools.calculator], skills=[Skills.files])

    assert handle.run("hello") == "files pinned"
    assert handle.state.loaded_skill_names == ["files"]


def test_public_agent_can_allowlist_skills_by_catalog_name(tmp_path):
    write_skill(tmp_path, "review", "# Review Skill\n\nUse this skill when reviewing code.\n")
    write_skill(tmp_path, "sql", "# SQL Skill\n\nUse this skill when analyzing database queries.\n")
    write_skill(tmp_path, "other", "# Other Skill\n\nUse this skill when other work is needed.\n")
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    llm = FakeLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "other ignored"}),
            json.dumps({"type": "final_answer", "content": "review loaded"}),
        ]
    )

    handle = Agent(config=config, llm=llm, tools=[], skills=Skills.only("review", "sql"))

    assert [skill.name for skill in handle.skill_registry.list_skills()] == ["review", "sql"]
    assert handle.run("other work") == "other ignored"
    assert handle.state.loaded_skill_names == []
    assert "Skill: other" not in llm.requests[0][0]["content"]

    assert handle.run("review this code") == "review loaded"
    assert handle.state.loaded_skill_names == ["review"]
    assert "Skill: review" in llm.requests[1][0]["content"]
    assert "Skill: other" not in llm.requests[1][0]["content"]


def test_public_agent_pin_is_additive_to_allowlist(tmp_path):
    write_skill(tmp_path, "review", "# Review Skill\n\nUse this skill when review work is needed.\n")
    write_skill(tmp_path, "sql", "# SQL Skill\n\nUse this skill when SQL work is needed.\n")
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "review pinned"})])

    handle = Agent(config=config, llm=llm, tools=[], skills=[Skills.only("sql"), Skills.pin("review")])

    assert [skill.name for skill in handle.skill_registry.list_skills()] == ["review", "sql"]
    assert handle.run("hello") == "review pinned"
    assert handle.state.loaded_skill_names == ["review"]
    assert "Skill: review" in llm.requests[0][0]["content"]


def test_public_agent_keeps_raw_string_skill_pins(tmp_path):
    write_skill(tmp_path, "review", "# Review Skill\n\nUse this skill when review work is needed.\n")
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "raw pin"})])

    handle = Agent(config=config, llm=llm, tools=[], skills=["review"])

    assert handle.run("hello") == "raw pin"
    assert handle.state.loaded_skill_names == ["review"]


def test_public_agent_warns_and_skips_missing_skill_names(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    with pytest.warns(UserWarning) as warning_records:
        handle = Agent(
            config=config,
            llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "missing skipped"})]),
            tools=[],
            skills=[Skills.only("missing-only"), Skills.pin("missing-pin"), "missing-raw"],
        )

    messages = [str(record.message) for record in warning_records]
    assert "Skill 'missing-only' requested by allowlist configuration is not registered; skipping." in messages
    assert "Skill 'missing-pin' requested by pin configuration is not registered; skipping." in messages
    assert "Skill 'missing-raw' requested by pin configuration is not registered; skipping." in messages
    assert handle.skill_registry.list_skills() == []
    trace_text = handle.trace_path.read_text(encoding="utf-8")
    assert "skill_config_warning" in trace_text
    assert "missing-only" in trace_text
    assert "missing-pin" in trace_text
    assert "missing-raw" in trace_text


def test_public_agent_empty_skills_disables_catalog_skills(tmp_path):
    write_skill(tmp_path, "review", "# Review Skill\n\nUse this skill when review work is needed.\n")
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    llm = FakeLLMClient([json.dumps({"type": "final_answer", "content": "no skills"})])

    handle = Agent(config=config, llm=llm, tools=[], skills=[])

    assert handle.skill_registry.list_skills() == []
    assert handle.run("review this code") == "no skills"
    assert handle.state.loaded_skill_names == []
    assert "Skill: review" not in llm.requests[0][0]["content"]


def test_public_agent_keeps_path_and_directory_skill_refs(tmp_path):
    write_skill(tmp_path, "review", "# Review Skill\n\nUse this skill when reviewing code.\n")
    path_skill_dir = tmp_path / "path-skills" / "path-review"
    path_skill_dir.mkdir(parents=True)
    (path_skill_dir / "SKILL.md").write_text(
        "# Path Review Skill\n\nUse this skill when path review work is needed.\n",
        encoding="utf-8",
    )
    extra_root = tmp_path / "extra-skills"
    extra_skill_dir = extra_root / "custom"
    extra_skill_dir.mkdir(parents=True)
    (extra_skill_dir / "SKILL.md").write_text(
        "# Custom Skill\n\nUse this skill when custom work is needed.\n",
        encoding="utf-8",
    )
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    llm = FakeLLMClient(
        [
            json.dumps({"type": "final_answer", "content": "path pinned"}),
            json.dumps({"type": "final_answer", "content": "directory selected"}),
        ]
    )

    pinned_handle = Agent(config=config, llm=llm, tools=[], skills=[Skills.path(path_skill_dir)])

    assert pinned_handle.run("hello") == "path pinned"
    assert pinned_handle.state.loaded_skill_names == ["path-review"]

    directory_handle = Agent(config=config, llm=llm, tools=[], skills=[Skills.from_dir(extra_root)])

    assert [skill.name for skill in directory_handle.skill_registry.list_skills()] == [
        "custom",
        "files",
        "memory",
        "review",
        "shell",
    ]
    assert directory_handle.run("custom work") == "directory selected"
    assert directory_handle.state.loaded_skill_names == ["custom"]


def test_fallback_chain_tries_next_provider_and_traces_attempts(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    success = FakeLLMClient([json.dumps({"type": "final_answer", "content": "fallback worked"})])
    chain = FallbackChain([FailingLLMClient(), success])

    handle = agent(config=config, llm=chain, tools=[], skills=[])

    response = handle.run("use fallback")

    runtime_chain = handle.runtime.llm_client
    trace_text = handle.trace_path.read_text(encoding="utf-8")
    assert response == "fallback worked"
    assert [attempt.success for attempt in runtime_chain.last_attempts] == [False, True]
    assert "llm_fallback_attempts" in trace_text


def test_public_agent_config_supports_programmatic_values_and_env_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "local")
    monkeypatch.setenv("CHULK_MODEL", "env-local-model")
    config = AgentConfig(
        project_root=tmp_path,
        store_path=tmp_path / "custom.sqlite",
        traces_dir=tmp_path / "custom-traces",
        skills_dir=tmp_path / "custom-skills",
        permission_profile="read-only",
        local_api_key="local",
    )
    handle = Agent(
        config=config,
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "configured"})]),
        tools=[],
        skills=[],
    )

    response = handle.run("hello")

    assert response == "configured"
    assert handle.runtime.memory_store.db_path == tmp_path / "custom.sqlite"
    assert handle.trace_path.parent == tmp_path / "custom-traces"
    assert handle.runtime.skill_registry.skills_dir == tmp_path / "custom-skills"
    assert handle.runtime.permission_policy.name == "read-only"
    assert handle.runtime.context_budget.max_prompt_tokens == 131_072


def test_public_agent_default_config_uses_cwd_runtime_and_read_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHULK_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)

    handle = Agent(
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "default sdk"})]),
        tools=[],
        skills=[],
    )
    response = handle.run("hello")

    assert response == "default sdk"
    assert handle.runtime.state.conversation_id
    assert handle.runtime.memory_store.db_path == tmp_path / ".chulk" / "store.sqlite"
    assert handle.trace_path.parent == tmp_path / ".chulk" / "traces"
    assert handle.runtime.permission_policy.name == "read-only"


def test_public_agent_default_config_uses_project_root_env_override(monkeypatch, tmp_path):
    launch_cwd = tmp_path / "launch-cwd"
    project_root = tmp_path / "project"
    launch_cwd.mkdir()
    project_root.mkdir()
    monkeypatch.chdir(launch_cwd)
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(project_root))
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)

    handle = Agent(
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "env project"})]),
        tools=[],
        skills=[],
    )
    response = handle.run("hello")

    assert response == "env project"
    assert handle.runtime.memory_store.db_path == project_root / ".chulk" / "store.sqlite"
    assert handle.trace_path.parent == project_root / ".chulk" / "traces"
    assert handle.runtime.skill_registry.skills_dir == project_root / ".chulk" / "skills"


def test_public_agent_config_uses_project_root_from_cwd_dotenv(monkeypatch, tmp_path):
    launch_cwd = tmp_path / "launch-cwd"
    project_root = tmp_path / "project"
    launch_cwd.mkdir()
    project_root.mkdir()
    (launch_cwd / ".env").write_text(f"CHULK_PROJECT_ROOT={project_root}\n", encoding="utf-8")
    monkeypatch.chdir(launch_cwd)
    monkeypatch.delenv("CHULK_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)

    config = AgentConfig().to_config()

    assert config.project_root == project_root
    assert config.runtime_dir == project_root / ".chulk"
    assert config.store_path == project_root / ".chulk" / "store.sqlite"
    assert config.skills_dir == project_root / ".chulk" / "skills"


def test_public_agent_config_explicit_project_root_overrides_env(monkeypatch, tmp_path):
    env_project = tmp_path / "env-project"
    sdk_project = tmp_path / "sdk-project"
    env_project.mkdir()
    sdk_project.mkdir()
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(env_project))

    config = AgentConfig(project_root=sdk_project).to_config()

    assert config.project_root == sdk_project
    assert config.runtime_dir == sdk_project / ".chulk"
    assert config.store_path == sdk_project / ".chulk" / "store.sqlite"


def test_public_agent_default_config_uses_runtime_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHULK_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)
    monkeypatch.setenv("CHULK_RUNTIME_DIR", "env-runtime")

    handle = Agent(
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "env runtime"})]),
        tools=[],
        skills=[],
    )
    response = handle.run("hello")

    assert response == "env runtime"
    assert handle.runtime.memory_store.db_path == tmp_path / "env-runtime" / "store.sqlite"
    assert handle.trace_path.parent == tmp_path / "env-runtime" / "traces"


def test_public_agent_config_from_env_and_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "local")
    monkeypatch.setenv("CHULK_MODEL", "env-model")
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)
    runtime_dir = tmp_path / "runtime"

    config = AgentConfig.from_env(
        project_root=tmp_path,
        runtime_dir=runtime_dir,
        local_api_key="local",
    ).to_config()

    assert config.project_root == tmp_path
    assert config.runtime_dir == runtime_dir
    assert config.llm_provider == "local"
    assert config.model == "env-model"
    assert config.store_path == runtime_dir / "store.sqlite"
    assert config.traces_dir == runtime_dir / "traces"
    assert config.skills_dir == runtime_dir / "skills"
    assert config.skills_dirs[-1] == runtime_dir / "skills"
    assert config.permission_profile == "read-only"


def test_public_agent_config_from_env_captures_cwd_before_later_cwd_change(monkeypatch, tmp_path):
    launch_cwd = tmp_path / "launch-cwd"
    later_cwd = tmp_path / "later-cwd"
    launch_cwd.mkdir()
    later_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)
    monkeypatch.delenv("CHULK_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)

    agent_config = AgentConfig.from_env()
    monkeypatch.chdir(later_cwd)
    config = agent_config.to_config()

    assert config.project_root == launch_cwd
    assert config.runtime_dir == launch_cwd / ".chulk"
    assert config.store_path == launch_cwd / ".chulk" / "store.sqlite"
    assert config.traces_dir == launch_cwd / ".chulk" / "traces"
    assert config.skills_dir == launch_cwd / ".chulk" / "skills"


def test_public_agent_config_from_env_captures_project_root_env_before_later_cwd_change(monkeypatch, tmp_path):
    launch_cwd = tmp_path / "launch-cwd"
    later_cwd = tmp_path / "later-cwd"
    project_root = tmp_path / "project"
    launch_cwd.mkdir()
    later_cwd.mkdir()
    project_root.mkdir()
    monkeypatch.chdir(launch_cwd)
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(project_root))
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)

    agent_config = AgentConfig.from_env()
    monkeypatch.chdir(later_cwd)
    config = agent_config.to_config()

    assert config.project_root == project_root
    assert config.runtime_dir == project_root / ".chulk"
    assert config.store_path == project_root / ".chulk" / "store.sqlite"
    assert config.skills_dir == project_root / ".chulk" / "skills"


def test_public_agent_config_from_env_captures_project_root_dotenv_before_later_cwd_change(monkeypatch, tmp_path):
    launch_cwd = tmp_path / "launch-cwd"
    later_cwd = tmp_path / "later-cwd"
    project_root = tmp_path / "project"
    launch_cwd.mkdir()
    later_cwd.mkdir()
    project_root.mkdir()
    (launch_cwd / ".env").write_text(f"CHULK_PROJECT_ROOT={project_root}\n", encoding="utf-8")
    monkeypatch.chdir(launch_cwd)
    monkeypatch.delenv("CHULK_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)

    agent_config = AgentConfig.from_env()
    monkeypatch.chdir(later_cwd)
    config = agent_config.to_config()

    assert config.project_root == project_root
    assert config.runtime_dir == project_root / ".chulk"
    assert config.store_path == project_root / ".chulk" / "store.sqlite"
    assert config.skills_dir == project_root / ".chulk" / "skills"


def test_public_agent_provider_constructor_captures_project_root_before_later_cwd_change(monkeypatch, tmp_path):
    launch_cwd = tmp_path / "launch-cwd"
    later_cwd = tmp_path / "later-cwd"
    launch_cwd.mkdir()
    later_cwd.mkdir()
    monkeypatch.chdir(launch_cwd)
    monkeypatch.delenv("CHULK_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)

    agent_config = AgentConfig.local(api_key="local")
    monkeypatch.chdir(later_cwd)
    config = agent_config.to_config()

    assert config.project_root == launch_cwd
    assert config.runtime_dir == launch_cwd / ".chulk"
    assert config.local_api_key == "local"


def test_public_agent_config_resolves_relative_runtime_dir_against_project_root(monkeypatch, tmp_path):
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)

    config = AgentConfig(project_root=tmp_path).to_config()

    assert config.project_root == tmp_path
    assert config.runtime_dir == tmp_path / ".chulk"
    assert config.store_path == tmp_path / ".chulk" / "store.sqlite"
    assert config.traces_dir == tmp_path / ".chulk" / "traces"
    assert config.skills_dir == tmp_path / ".chulk" / "skills"
    assert config.mcp_config_path == tmp_path / ".chulk" / "mcp.json"
    assert config.permission_profile == "read-only"


def test_public_agent_config_uses_runtime_dir_env_without_sdk_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_RUNTIME_DIR", "env-runtime")

    config = AgentConfig(project_root=tmp_path).to_config()

    assert config.runtime_dir == tmp_path / "env-runtime"
    assert config.store_path == tmp_path / "env-runtime" / "store.sqlite"
    assert config.traces_dir == tmp_path / "env-runtime" / "traces"
    assert config.skills_dir == tmp_path / "env-runtime" / "skills"
    assert config.mcp_config_path == tmp_path / "env-runtime" / "mcp.json"


def test_public_agent_config_uses_runtime_dir_dotenv_without_sdk_override(monkeypatch, tmp_path):
    monkeypatch.delenv("CHULK_RUNTIME_DIR", raising=False)
    (tmp_path / ".env").write_text("CHULK_RUNTIME_DIR=dotenv-runtime\n", encoding="utf-8")

    config = AgentConfig(project_root=tmp_path).to_config()

    assert config.runtime_dir == tmp_path / "dotenv-runtime"
    assert config.store_path == tmp_path / "dotenv-runtime" / "store.sqlite"
    assert config.traces_dir == tmp_path / "dotenv-runtime" / "traces"
    assert config.skills_dir == tmp_path / "dotenv-runtime" / "skills"
    assert config.mcp_config_path == tmp_path / "dotenv-runtime" / "mcp.json"


def test_public_agent_config_explicit_runtime_dir_overrides_env_and_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_RUNTIME_DIR", "env-runtime")
    (tmp_path / ".env").write_text("CHULK_RUNTIME_DIR=dotenv-runtime\n", encoding="utf-8")

    config = AgentConfig(project_root=tmp_path, runtime_dir="sdk-runtime").to_config()

    assert config.runtime_dir == tmp_path / "sdk-runtime"
    assert config.store_path == tmp_path / "sdk-runtime" / "store.sqlite"
    assert config.traces_dir == tmp_path / "sdk-runtime" / "traces"
    assert config.skills_dir == tmp_path / "sdk-runtime" / "skills"
    assert config.mcp_config_path == tmp_path / "sdk-runtime" / "mcp.json"


def test_public_agent_config_uses_dotenv_permission_profile_without_sdk_override(monkeypatch, tmp_path):
    monkeypatch.delenv("CHULK_PERMISSION_PROFILE", raising=False)
    (tmp_path / ".env").write_text("CHULK_PERMISSION_PROFILE=workspace-write\n", encoding="utf-8")

    config = AgentConfig(project_root=tmp_path).to_config()

    assert config.permission_profile == "workspace-write"


def test_public_agent_config_explicit_permission_profile_overrides_env_and_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_PERMISSION_PROFILE", "workspace-write")
    (tmp_path / ".env").write_text("CHULK_PERMISSION_PROFILE=full-access\n", encoding="utf-8")

    config = AgentConfig(project_root=tmp_path, permission_profile="read-only").to_config()

    assert config.permission_profile == "read-only"


def test_public_agent_config_provider_constructors_ignore_cross_provider_env_model(monkeypatch, tmp_path):
    monkeypatch.setenv("CHULK_MODEL", "local-only-model")

    openai_config = AgentConfig.openai(project_root=tmp_path, runtime_dir=tmp_path / "openai", api_key="openai")
    deepseek_config = AgentConfig.deepseek(
        project_root=tmp_path,
        runtime_dir=tmp_path / "deepseek",
        api_key="deepseek",
        base_url="https://deepseek.example",
    )
    local_config = AgentConfig.local(
        project_root=tmp_path,
        runtime_dir=tmp_path / "local",
        model="local-model",
        base_url="http://localhost:1234/v1",
        api_key="local",
    )

    assert openai_config.to_config().llm_provider == "openai"
    assert openai_config.to_config().model == DEFAULT_MODEL
    assert openai_config.to_config().openai_api_key == "openai"
    assert deepseek_config.to_config().llm_provider == "deepseek"
    assert deepseek_config.to_config().model == DEFAULT_DEEPSEEK_MODEL
    assert deepseek_config.to_config().deepseek_api_key == "deepseek"
    assert deepseek_config.to_config().deepseek_base_url == "https://deepseek.example"
    assert local_config.to_config().llm_provider == "local"
    assert local_config.to_config().model == "local-model"
    assert local_config.to_config().local_base_url == "http://localhost:1234/v1"
    assert local_config.to_config().local_api_key == "local"

    default_local = AgentConfig.local(project_root=tmp_path, runtime_dir=tmp_path / "default-local").to_config()
    assert default_local.model == DEFAULT_LOCAL_MODEL


def test_public_agent_config_with_overrides_for_app_agents(tmp_path):
    server = MCP.streamable_http(label="docs", server_url="https://mcp.example.com")
    base_fallback = AgentConfig.fallback_provider("deepseek", DEFAULT_DEEPSEEK_MODEL)
    app_fallback = AgentConfig.fallback_provider("local", DEFAULT_LOCAL_MODEL)
    base_config = AgentConfig.openai(
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime" / "base",
        api_key="openai",
        permission_profile="read-only",
        max_tool_calls_per_turn=1,
        llm_fallback_providers=[base_fallback],
    )
    app_agent_config = base_config.with_overrides(
        model="gpt-4.1",
        runtime_dir=tmp_path / "runtime" / "agent-a",
        permission_profile="workspace-write",
        max_tool_calls_per_turn=7,
        mcp_servers=[server],
        llm_fallback_providers=[app_fallback],
    )

    base_runtime = base_config.to_config()
    app_runtime = app_agent_config.to_config()

    assert base_runtime.model == DEFAULT_MODEL
    assert base_runtime.store_path == tmp_path / "runtime" / "base" / "store.sqlite"
    assert base_runtime.permission_profile == "read-only"
    assert base_runtime.max_tool_calls_per_turn == 1
    assert base_runtime.mcp_servers == ()
    assert base_runtime.llm_fallback_providers == (base_fallback,)
    assert app_runtime.model == "gpt-4.1"
    assert app_runtime.store_path == tmp_path / "runtime" / "agent-a" / "store.sqlite"
    assert app_runtime.permission_profile == "workspace-write"
    assert app_runtime.max_tool_calls_per_turn == 7
    assert app_runtime.mcp_servers == (server,)
    assert app_runtime.llm_fallback_providers == (app_fallback,)


def test_public_agent_config_reuses_programmatic_mcp_iterable(tmp_path):
    server = MCP.streamable_http(label="docs", server_url="https://mcp.example.com")
    config = AgentConfig(
        project_root=tmp_path,
        provider="openai",
        model="gpt-4.1-mini",
        mcp_servers=(configured_server for configured_server in [server]),
    )

    assert config.to_config().mcp_servers == (server,)
    assert config.to_config().mcp_servers == (server,)


def test_public_mcp_builder_uses_hosted_mcp_for_openai(tmp_path):
    config = AgentConfig(project_root=tmp_path, provider="openai", model="gpt-4.1-mini")
    llm = HostedMCPRecordingLLM()
    server = MCP.streamable_http(label="docs", server_url="https://mcp.example.com", allowed_tools=["search_docs"])

    result = Agent(config=config, llm=llm, tools=[], skills=[], mcp=[server]).run_result("search docs")

    assert result.content == "hosted mcp captured"
    assert llm.hosted_mcp_servers == (server,)


def test_public_mcp_builder_resolves_authorization_env(monkeypatch):
    monkeypatch.setenv("DOCS_MCP_TOKEN", "secret-token")

    server = MCP.streamable_http(
        label="docs",
        server_url="https://mcp.example.com",
        authorization_env="DOCS_MCP_TOKEN",
    )

    assert server.authorization == "secret-token"
    assert server.to_openai_tool()["authorization"] == "secret-token"


def test_public_mcp_builder_rejects_missing_authorization_env(monkeypatch):
    monkeypatch.delenv("DOCS_MCP_TOKEN", raising=False)

    with pytest.raises(ValueError, match="DOCS_MCP_TOKEN"):
        MCP.streamable_http(
            label="docs",
            server_url="https://mcp.example.com",
            authorization_env="DOCS_MCP_TOKEN",
        )


def test_public_mcp_builder_rejects_string_allowed_tools():
    with pytest.raises(ValueError, match="allowed_tools"):
        MCP.streamable_http(
            label="docs",
            server_url="https://mcp.example.com",
            allowed_tools="search_docs",
        )


def test_public_mcp_builder_accepts_none_allowed_tools():
    server = MCP.streamable_http(
        label="docs",
        server_url="https://mcp.example.com",
        allowed_tools=None,
    )

    assert server.allowed_tools == ()


def test_public_mcp_empty_list_disables_configured_servers(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime"
    mcp_dir = runtime_dir
    mcp_dir.mkdir()
    (mcp_dir / "mcp.json").write_text(
        json.dumps({"servers": [{"label": "docs", "server_url": "https://mcp.example.com"}]}),
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("bridge discovery should be disabled")

    monkeypatch.setattr("chulk.runtime.create_mcp_bridge_tools", fail_if_called)
    config = AgentConfig(
        project_root=tmp_path,
        runtime_dir=runtime_dir,
        provider="local",
        model="local-model",
        local_api_key="local",
    )
    handle = Agent(
        config=config,
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "no mcp"})]),
        tools=[],
        skills=[],
        mcp=[],
    )

    result = handle.run_result("hello")

    assert result.content == "no mcp"
    assert result.status == "completed"


def test_wheel_install_exposes_sdk_defaults_and_bundled_skills(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            ".",
            "-w",
            str(wheelhouse),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheelhouse.glob("chulkharness-*.whl"))

    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [str(python), "-m", "pip", "install", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )

    app_dir = tmp_path / "my-app"
    app_dir.mkdir()
    script = textwrap.dedent(
        """
        import json
        import os
        from pathlib import Path

        import chulk
        from chulk import Agent, AgentConfig, Skills, Tool, Tools
        from chulk.llm import LLMClient
        from chulk.skills import SkillRegistry, bundled_skills_dir

        class FakeLLM(LLMClient):
            def complete(self, messages, *, max_output_tokens=None):
                return json.dumps({"type": "final_answer", "content": "wheel ok"})

        os.environ.pop("CHULK_RUNTIME_DIR", None)
        os.environ.pop("CHULK_PERMISSION_PROFILE", None)

        handle = Agent(llm=FakeLLM(), tools=[], skills=[])
        result = handle.run_result("hello")
        runtime_dir = Path.cwd() / ".chulk"
        assert result.content == "wheel ok"
        assert Path(result.trace_path).parent == runtime_dir / "traces"
        assert (runtime_dir / "store.sqlite").exists()
        assert handle.runtime.permission_policy.name == "read-only"

        registry = SkillRegistry(runtime_dir / "skills", skills_dirs=(bundled_skills_dir(), runtime_dir / "skills"))
        registry.load_metadata()
        skill_names = {skill.name for skill in registry.list_skills()}
        assert {"files", "shell", "memory"} <= skill_names

        package_root = Path(chulk.__file__).resolve().parent
        assert not (package_root / ".chulk").exists()
        assert not (package_root / "store.sqlite").exists()
        assert AgentConfig and Tool and Tools and Skills
        print("ok")
        """
    )
    completed = subprocess.run(
        [str(python), "-c", script],
        cwd=app_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"


@pytest.mark.parametrize(
    ("provider", "model", "extra_config"),
    [
        ("local", "local-model", {"local_api_key": "local"}),
        ("deepseek", "deepseek-v4-flash", {"deepseek_api_key": "deepseek"}),
    ],
)
def test_public_mcp_bridge_registers_for_non_openai_providers(monkeypatch, tmp_path, provider, model, extra_config):
    bridge_calls = []

    def fake_bridge(servers):
        bridge_calls.append(tuple(servers))

        @Tool(permission_level=ToolPermissionLevel.EXTERNAL_SERVICE, requires_confirmation=True)
        def mcp_docs_search_docs(query: str) -> str:
            """Search docs."""
            return f"found {query}"

        return [mcp_docs_search_docs]

    monkeypatch.setattr("chulk.runtime.create_mcp_bridge_tools", fake_bridge)
    server = MCP.streamable_http(label="docs", server_url="https://mcp.example.com")
    config = AgentConfig(
        project_root=tmp_path,
        provider=provider,
        model=model,
        permission_profile="workspace-write",
        **extra_config,
    )
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "content": None,
                    "tool_name": "mcp_docs_search_docs",
                    "arguments_json": json.dumps({"query": "sdk"}),
                    "plan_json": "{}",
                    "step_update_json": "{}",
                }
            ),
            json.dumps({"type": "final_answer", "content": "bridge ok"}),
        ]
    )

    result = Agent(
        config=config,
        llm=llm,
        tools=[],
        skills=[],
        mcp=[server],
        permission_callback=lambda _request, _record: True,
    ).run_result("search docs")

    assert bridge_calls == [(server,)]
    assert result.content == "bridge ok"
    assert result.tool_calls[0]["tool_name"] == "mcp_docs_search_docs"
    assert result.tool_calls[0]["success"] is True


def test_public_plan_result_approve_result_and_reject_result(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    plan_payload = _plan_payload("Add SDK metadata")
    approve_handle = Agent(
        config=config,
        llm=FakeLLMClient(
            [
                _plan_response(plan_payload),
                json.dumps(
                    {
                        "type": "plan_step_update",
                        "content": None,
                        "tool_name": None,
                        "arguments_json": "{}",
                        "plan_json": "{}",
                        "step_update_json": json.dumps(
                            {"step_id": "1", "status": "completed", "evidence": "Implemented in API."}
                        ),
                    }
                ),
                json.dumps({"type": "final_answer", "content": "plan executed"}),
            ]
        ),
        tools=[],
        skills=[],
    )

    plan_result = approve_handle.plan_result("plan the SDK change")
    approve_result = approve_handle.approve_result()

    assert isinstance(plan_result, PlanResult)
    assert plan_result.status == "waiting_for_approval"
    assert plan_result.plan.summary == "Add SDK metadata"
    plan_payload = plan_result.to_dict()
    assert plan_payload["plan"]["summary"] == "Add SDK metadata"
    assert plan_payload["trace_path"] == str(approve_handle.trace_path)
    assert approve_result.content == "plan executed"
    assert approve_result.status == "completed"
    assert approve_result.plan.status == "completed"
    assert approve_result.to_dict()["plan"]["status"] == "completed"

    reject_root = tmp_path / "reject"
    reject_root.mkdir()
    reject_handle = Agent(
        config=load_config({"CHULK_PROJECT_ROOT": str(reject_root)}),
        llm=FakeLLMClient([_plan_response(_plan_payload("Reject me"))]),
        tools=[],
        skills=[],
    )
    reject_plan = reject_handle.plan_result("plan then reject")
    reject_result = reject_handle.reject_result()

    assert reject_plan.plan.status == "pending_approval"
    assert reject_result.status == "plan_rejected"
    assert reject_result.content == "Plan rejected. No tools were run."
    assert reject_result.plan.status == "rejected"


def test_public_approve_and_reject_result_without_pending_plan_use_neutral_metadata(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    handle = Agent(
        config=config,
        llm=FakeLLMClient([json.dumps({"type": "final_answer", "content": "done"})]),
        tools=[],
        skills=[],
    )

    completed_result = handle.run_result("hello")
    approve_result = handle.approve_result()
    reject_result = handle.reject_result()

    assert completed_result.status == "completed"
    assert approve_result.content == "No plan is waiting for approval."
    assert approve_result.status == "no_pending_plan"
    assert approve_result.turn_id is None
    assert approve_result.tool_calls == []
    assert approve_result.plan is None
    assert reject_result.content == "No plan is waiting for approval."
    assert reject_result.status == "no_pending_plan"
    assert reject_result.turn_id is None
    assert reject_result.tool_calls == []
    assert reject_result.plan is None


@pytest.mark.asyncio
async def test_async_agent_runs_inside_active_event_loop(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    handle = AsyncAgent(
        config=config,
        llm=FakeLLMClient(
            [
                json.dumps({"type": "final_answer", "content": "async ok"}),
                json.dumps({"type": "final_answer", "content": "async ok again"}),
            ]
        ),
        tools=[],
        skills=[],
    )

    result = await handle.run_result("hello async")

    assert result.content == "async ok"
    assert await handle.run("hello again") == "async ok again"


def test_public_tool_schema_supports_richer_annotations():
    class Mode(Enum):
        fast = "fast"
        safe = "safe"

    @dataclass
    class Profile:
        name: str
        priority: int = 1

    class PydanticLike:
        @classmethod
        def model_json_schema(cls):
            return {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            }

    @Tool
    def rich_tool(
        mode: Mode,
        target: Annotated[str, "Target name"],
        maybe_count: int | None = None,
        level: Literal["low", "high"] = "low",
        profile: Profile | None = None,
        payload: PydanticLike | None = None,
        tags: list[str] | None = None,
        scores: dict[str, int] | None = None,
        states: list[dict[str, Literal["active", "paused"]]] | None = None,
    ) -> str:
        """Use rich schema types."""
        return target

    schema = rich_tool.args_schema

    assert schema["properties"]["mode"] == {"type": ["string"], "enum": ["fast", "safe"]}
    assert schema["properties"]["target"] == {"type": "string", "description": "Target name"}
    assert schema["properties"]["maybe_count"]["type"] == ["integer", "null"]
    assert schema["properties"]["level"]["enum"] == ["low", "high"]
    assert schema["properties"]["profile"]["type"] == ["null", "object"]
    assert schema["properties"]["payload"]["type"] == ["null", "object"]
    assert schema["properties"]["tags"]["type"] == ["array", "null"]
    assert schema["properties"]["scores"]["additionalProperties"] == {"type": "integer"}
    assert schema["properties"]["states"]["items"]["additionalProperties"] == {
        "type": ["string"],
        "enum": ["active", "paused"],
    }
    assert schema["required"] == ["mode", "target"]


def test_public_tool_pydantic_schema_uses_enforced_subset():
    pydantic = pytest.importorskip("pydantic")

    class Address(pydantic.BaseModel):
        city: str

    class Profile(pydantic.BaseModel):
        name: str
        address: Address | None = None

    calls = []

    @Tool
    def profile_tool(profile: Profile) -> str:
        """Use a Pydantic profile."""
        calls.append(profile)
        return "ok"

    profile_schema = profile_tool.args_schema["properties"]["profile"]
    assert "$defs" not in profile_schema
    assert "anyOf" not in profile_schema["properties"]["address"]
    assert profile_schema["properties"]["address"]["type"] == ["null", "object"]
    assert profile_schema["properties"]["address"]["properties"]["city"] == {"type": "string"}

    registry = ToolRegistry()
    registry.register(profile_tool)
    result = registry.run("profile_tool", {"profile": {"name": "Ada", "address": 123}})

    assert not result.success
    assert result.error == "invalid_arguments"
    assert calls == []
    assert result.metadata["validation_errors"] == [
        {
            "path": "profile.address",
            "message": "value has the wrong type",
            "expected": "null or object",
            "actual": "integer",
        }
    ]


def _plan_payload(summary: str) -> dict:
    return {
        "summary": summary,
        "steps": [
            {
                "id": "1",
                "title": "Implement",
                "description": "Implement the SDK change.",
                "status": "pending",
                "acceptance_criteria": ["The SDK change is implemented."],
            }
        ],
    }


def _plan_response(payload: dict) -> str:
    return json.dumps(
        {
            "type": "plan",
            "content": None,
            "tool_name": None,
            "arguments_json": "{}",
            "plan_json": json.dumps(payload),
            "step_update_json": "{}",
        }
    )
