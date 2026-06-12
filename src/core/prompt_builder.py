"""Agent prompt composition helpers."""

from __future__ import annotations

from src.core.prompts import (
    JSON_ACTION_PROMPT,
    format_memories_for_prompt,
    format_skills_for_prompt,
    format_tool_call_rules,
    format_tools_for_prompt,
)
from src.memory import ConversationMemory, MemoryRecord
from src.skills import SkillSelection
from src.tools import ToolRegistry


def build_agent_messages(
    *,
    system_prompt: str,
    memory: ConversationMemory,
    profile_memories: list[MemoryRecord],
    relevant_memories: list[MemoryRecord],
    selected_skills: list[SkillSelection],
    tool_registry: ToolRegistry,
    max_skill_content_chars: int,
    max_tool_calls_per_turn: int,
) -> list[dict[str, str]]:
    """Build the model input from prompt, tools, and short-term history."""
    composed_system_prompt = "\n\n".join(
        [
            system_prompt,
            format_memories_for_prompt(
                profile_memories=profile_memories,
                relevant_memories=relevant_memories,
            ),
            format_skills_for_prompt(
                selected_skills,
                max_chars_per_skill=max_skill_content_chars,
            ),
            JSON_ACTION_PROMPT,
            format_tool_call_rules(max_tool_calls_per_turn),
            format_tools_for_prompt(tool_registry.tool_descriptions_for_prompt()),
        ]
    )
    return [{"role": "system", "content": composed_system_prompt}, *memory.recent()]
