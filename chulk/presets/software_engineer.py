"""Software-engineer agent preset."""

from __future__ import annotations

from importlib.resources import files

from chulk.api import AgentPreset
from chulk.core.prompts import BASE_SYSTEM_PROMPT
import chulk.skills as skills
import chulk.tools as tools


DEFAULT_AGENT_PLAYBOOK = files("chulk.presets").joinpath("AGENT.md").read_text(encoding="utf-8").strip()

SOFTWARE_ENGINEER_SYSTEM_PROMPT = "\n\n".join(
    [
        BASE_SYSTEM_PROMPT,
        DEFAULT_AGENT_PLAYBOOK,
    ]
)


def software_engineer() -> AgentPreset:
    """Return the default Chulk coding-agent preset."""
    return AgentPreset(
        system_prompt=SOFTWARE_ENGINEER_SYSTEM_PROMPT,
        tools=tuple(tools.default_software_engineer()),
        skills=(skills.files, skills.shell, skills.memory),
    )


SoftwareEngineer = software_engineer


__all__ = ["DEFAULT_AGENT_PLAYBOOK", "SOFTWARE_ENGINEER_SYSTEM_PROMPT", "SoftwareEngineer", "software_engineer"]
