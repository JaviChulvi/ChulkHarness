"""Public programmable API for Chulk."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chulk.config import Config, load_config
from chulk.core import Agent
from chulk.llm import LLMClient
from chulk.runtime import create_agent as create_runtime_agent


@dataclass(frozen=True)
class AgentPreset:
    """Reusable collection of prompt, tools, skills, and default behavior."""

    system_prompt: str | None = None
    tools: tuple[object, ...] = field(default_factory=tuple)
    skills: tuple[object, ...] = field(default_factory=tuple)


class AgentHandle:
    """Small ergonomic wrapper around the explicit core Agent runtime."""

    def __init__(self, runtime: Agent) -> None:
        self.runtime = runtime

    @property
    def state(self):
        return self.runtime.state

    @property
    def conversation_id(self) -> str:
        return self.runtime.state.conversation_id

    @property
    def trace_path(self) -> Path | None:
        trace_logger = getattr(self.runtime, "trace_logger", None)
        return getattr(trace_logger, "path", None)

    @property
    def tool_registry(self):
        return self.runtime.tool_registry

    @property
    def skill_registry(self):
        return self.runtime.skill_registry

    def run(self, message: str) -> str:
        """Run one normal agent turn."""
        return self.runtime.run_turn(message)

    def __call__(self, message: str) -> str:
        return self.run(message)

    def plan(self, message: str) -> str:
        """Run one planned turn that pauses for approval before mutation."""
        return self.runtime.run_planned_turn(message)

    def approve(self) -> str:
        """Approve and continue a pending plan."""
        return self.runtime.approve_plan()

    def reject(self) -> str:
        """Reject a pending plan."""
        return self.runtime.reject_plan()


def agent(
    *,
    config: Config | None = None,
    preset: AgentPreset | None = None,
    llm: LLMClient | Any | None = None,
    tools: Iterable[object] | None = None,
    skills: Iterable[object] | None = None,
    system_prompt: str | None = None,
    conversation_id: str | None = None,
) -> AgentHandle:
    """Create a configured Chulk agent handle."""
    runtime_config = config or load_config()
    selected_tools = tools if tools is not None else (preset.tools if preset is not None else None)
    selected_skills = skills if skills is not None else (preset.skills if preset is not None else None)
    selected_prompt = system_prompt or (preset.system_prompt if preset is not None else None)
    runtime = create_runtime_agent(
        runtime_config,
        conversation_id=conversation_id,
        llm_client=llm,
        tool_specs=selected_tools,
        skill_specs=selected_skills,
        system_prompt=selected_prompt,
    )
    return AgentHandle(runtime)


Agent = agent


__all__ = ["Agent", "AgentHandle", "AgentPreset", "agent"]
