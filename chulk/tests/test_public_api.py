"""Tests for the public Chulk API."""

import json

from chulk import Agent, Skills, Tool, Tools, agent, skills, tool, tools
from chulk.config import load_config
from chulk.llm import FallbackChain, LLMCapabilities, LLMClient, LLMError
from chulk.presets import SoftwareEngineer, software_engineer
from chulk.presets.software_engineer import DEFAULT_AGENT_PLAYBOOK, SOFTWARE_ENGINEER_SYSTEM_PROMPT


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


def test_public_api_exports_capitalized_aliases():
    assert Agent is agent
    assert Tool is tool
    assert Tools is tools
    assert Skills is skills
    assert SoftwareEngineer is software_engineer


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


def test_public_agent_can_pin_skill(tmp_path):
    skill_dir = tmp_path / "skills" / "files"
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
