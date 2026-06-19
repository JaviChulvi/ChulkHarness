"""Agent prompt composition helpers."""

from __future__ import annotations

from chulk.core.prompts import (
    JSON_ACTION_PROMPT,
    NATIVE_ACTION_PROMPT,
    format_conversation_summary_for_prompt,
    format_context_sections_for_prompt,
    format_memories_for_prompt,
    format_planning_for_prompt,
    format_prompt_metadata_for_prompt,
    format_skills_for_prompt,
    format_tool_call_rules,
    format_tools_for_prompt,
)
from chulk.core.context import (
    AgentPrompt,
    ContextBudget,
    ContextSection,
    TurnContextSection,
    build_context_report,
    select_messages_for_budget,
)
from chulk.core.state import Plan
from chulk.memory import ConversationMemory, MemoryRecord
from chulk.skills import SkillSelection
from chulk.tools import ToolRegistry


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
    context_sections: list[TurnContextSection] | None = None,
    prompt_profile: str | None = None,
    locale: str | None = None,
    planning_enabled: bool = False,
    active_plan: Plan | None = None,
    plan_approved: bool = False,
    require_plan: bool = False,
    native_action_protocol: bool = False,
    context_budget: ContextBudget | None = None,
) -> list[dict[str, str]]:
    """Build the model input from prompt, tools, and short-term history."""
    return build_agent_prompt(
        system_prompt=system_prompt,
        memory=memory,
        profile_memories=profile_memories,
        relevant_memories=relevant_memories,
        selected_skills=selected_skills,
        tool_registry=tool_registry,
        max_skill_content_chars=max_skill_content_chars,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        context_sections=context_sections,
        prompt_profile=prompt_profile,
        locale=locale,
        planning_enabled=planning_enabled,
        active_plan=active_plan,
        plan_approved=plan_approved,
        require_plan=require_plan,
        native_action_protocol=native_action_protocol,
        context_budget=context_budget,
    ).messages


def build_agent_prompt(
    *,
    system_prompt: str,
    memory: ConversationMemory,
    profile_memories: list[MemoryRecord],
    relevant_memories: list[MemoryRecord],
    selected_skills: list[SkillSelection],
    tool_registry: ToolRegistry,
    max_skill_content_chars: int,
    max_tool_calls_per_turn: int,
    context_sections: list[TurnContextSection] | None = None,
    prompt_profile: str | None = None,
    locale: str | None = None,
    planning_enabled: bool = False,
    active_plan: Plan | None = None,
    plan_approved: bool = False,
    require_plan: bool = False,
    native_action_protocol: bool = False,
    context_budget: ContextBudget | None = None,
) -> AgentPrompt:
    """Build model input and a context report from prompt, tools, and history."""
    tool_descriptions = tool_registry.tool_descriptions_for_prompt()
    memory_prompt = format_memories_for_prompt(
        profile_memories=profile_memories,
        relevant_memories=relevant_memories,
    )
    skills_prompt = format_skills_for_prompt(
        selected_skills,
        max_chars_per_skill=max_skill_content_chars,
    )
    conversation_summary_prompt = format_conversation_summary_for_prompt(memory.conversation_summary)
    prompt_metadata_prompt = format_prompt_metadata_for_prompt(prompt_profile=prompt_profile, locale=locale)
    turn_context_sections = context_sections or []
    external_context_prompt = format_context_sections_for_prompt(turn_context_sections)
    planning_prompt = format_planning_for_prompt(
        planning_enabled=planning_enabled,
        active_plan=active_plan,
        plan_approved=plan_approved,
        require_plan=require_plan,
        max_reconnaissance_tool_calls=max_tool_calls_per_turn,
    )
    tool_rules = format_tool_call_rules(max_tool_calls_per_turn)
    tools_prompt = format_tools_for_prompt(tool_descriptions)
    action_protocol = NATIVE_ACTION_PROMPT if native_action_protocol else JSON_ACTION_PROMPT
    system_parts = [
        ("system_prompt", "Base system prompt", system_prompt, {}),
        (
            "prompt_metadata",
            "Prompt metadata",
            prompt_metadata_prompt,
            {"prompt_profile": prompt_profile, "locale": locale},
        ),
        (
            "external_context",
            "External turn context",
            external_context_prompt,
            {"context_section_ids": [section.id for section in turn_context_sections]},
        ),
        (
            "memories",
            "Selected memories",
            memory_prompt,
            {
                "profile_memory_ids": [memory.id for memory in profile_memories],
                "relevant_memory_ids": [memory.id for memory in relevant_memories],
            },
        ),
        (
            "skills",
            "Selected skills",
            skills_prompt,
            {"skill_names": [selection.skill.name for selection in selected_skills]},
        ),
        (
            "conversation_summary",
            "Conversation summary",
            conversation_summary_prompt,
            {
                "summary_message_count": memory.summary_message_count,
                "has_summary": memory.conversation_summary is not None,
            },
        ),
        ("planning", "Planning instructions", planning_prompt, {"enabled": planning_enabled}),
        (
            "tools",
            "Available tools",
            tools_prompt,
            {"tool_names": [tool.name for tool in tool_registry.list_tools()]},
        ),
        ("tool_rules", "Tool-call rules", tool_rules, {"max_tool_calls_per_turn": max_tool_calls_per_turn}),
        (
            "action_protocol",
            "Action protocol",
            action_protocol,
            {"native_tool_calling": native_action_protocol},
        ),
    ]
    system_sections = [
        ContextSection.from_text(
            name,
            label,
            content,
            item_count=_section_item_count(name, content, metadata),
            metadata=metadata,
        )
        for name, label, content, metadata in system_parts
    ]
    composed_system_prompt = "\n\n".join(content for _, _, content, _ in system_parts)
    system_message = {"role": "system", "content": composed_system_prompt}
    budget = context_budget or ContextBudget()
    history_messages, omitted_messages = select_messages_for_budget(
        system_message=system_message,
        history_messages=memory.recent(),
        budget=budget,
    )
    messages = [system_message, *history_messages]
    context_report = build_context_report(
        system_sections=system_sections,
        history_messages=history_messages,
        omitted_messages=omitted_messages,
        budget=budget,
        sent_messages=messages,
    )
    return AgentPrompt(messages=messages, context_report=context_report, omitted_messages=omitted_messages)


def _section_item_count(name: str, content: str, metadata: dict) -> int:
    if name == "memories":
        return len(metadata.get("profile_memory_ids", [])) + len(metadata.get("relevant_memory_ids", []))
    if name == "skills":
        return len(metadata.get("skill_names", []))
    if name == "external_context":
        return len(metadata.get("context_section_ids", []))
    if name == "tools":
        return len(metadata.get("tool_names", []))
    return 1 if content else 0
