"""Software-engineer agent preset."""

from __future__ import annotations

from chulk.api import AgentPreset
from chulk.core.prompts import BASE_SYSTEM_PROMPT
import chulk.skills as skills
import chulk.tools as tools


SOFTWARE_ENGINEER_SYSTEM_PROMPT = "\n\n".join(
    [
        BASE_SYSTEM_PROMPT,
        """You are configured as an agentic software engineer.

Read the code before making claims about it. Prefer small, inspectable changes that fit the existing architecture. Use tools to inspect files, edit with patches, run focused validation, and explain concrete outcomes. Treat tool arguments as untrusted, keep all file and shell work inside the configured project root, and preserve unrelated user changes.""",
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


__all__ = ["SOFTWARE_ENGINEER_SYSTEM_PROMPT", "SoftwareEngineer", "software_engineer"]
