"""Agent prompt composition helpers."""

from __future__ import annotations

from collections.abc import Iterable
import json
from typing import Any, Literal

from chulk.core.prompts import (
    format_action_protocol_for_prompt,
    format_conversation_summary_for_prompt,
    format_context_sections_for_prompt,
    format_memories_for_prompt,
    format_planning_for_prompt,
    format_prompt_metadata_for_prompt,
    format_skills_for_prompt,
    format_system_instructions_for_prompt,
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
from chulk.core.planning import read_only_planning_tool_names
from chulk.memory import ConversationMemory, MemoryRecord
from chulk.skills import SkillSelection
from chulk.tools import ToolRegistry
from chulk.tools.registry import (
    PLAN_STEP_UPDATE_TOOL_NAME,
    PLAN_TOOL_NAME,
    tool_descriptions_for_prompt,
)


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
    native_tool_declarations: list[dict[str, Any]] | None = None,
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
        native_tool_declarations=native_tool_declarations,
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
    native_tool_declarations: list[dict[str, Any]] | None = None,
    context_budget: ContextBudget | None = None,
) -> AgentPrompt:
    """Build model input and a context report from prompt, tools, and history."""
    registered_tools = tool_registry.list_tools()
    planning_tool_names = read_only_planning_tool_names(registered_tools)
    action_tools = (
        [tool for tool in registered_tools if tool.name in planning_tool_names]
        if require_plan
        else registered_tools
    )
    if native_action_protocol and native_tool_declarations is None:
        native_tool_declarations = _registered_native_tool_declarations(
            action_tools
        )
    safe_native_tool_declarations = _safe_native_tool_declarations(
        native_tool_declarations
    )
    if require_plan:
        allowed_native_names = {
            *planning_tool_names,
            PLAN_TOOL_NAME,
            PLAN_STEP_UPDATE_TOOL_NAME,
        }
        safe_native_tool_declarations = [
            declaration
            for declaration in safe_native_tool_declarations
            if declaration.get("name") in allowed_native_names
        ]
    native_declaration_names = _native_tool_declaration_names(
        safe_native_tool_declarations
    )
    action_transport: Literal["provider_native", "chulk_json"] = (
        "provider_native" if native_action_protocol else "chulk_json"
    )
    system_instructions_prompt = format_system_instructions_for_prompt(system_prompt)
    tool_descriptions = tool_descriptions_for_prompt(action_tools)
    turn_context_sections = context_sections or []
    propose_plan = require_plan and active_plan is None
    update_plan_step = bool(
        plan_approved
        and active_plan is not None
        and active_plan.active_step() is not None
    )
    allow_final_answer = not propose_plan and not update_plan_step
    registered_tool_calls_available = bool(
        action_tools
    )
    native_action_names = {
        name
        for name in native_declaration_names
        if name not in {PLAN_TOOL_NAME, PLAN_STEP_UPDATE_TOOL_NAME}
    }
    native_tool_calls_available = bool(native_action_names) and (
        not require_plan or registered_tool_calls_available
    )
    tool_calls_available = (
        native_tool_calls_available
        if native_action_protocol
        else registered_tool_calls_available
    )
    action_protocol = format_action_protocol_for_prompt(
        native=native_action_protocol,
        allow_final_answer=allow_final_answer,
        allow_tool_call=tool_calls_available,
        allow_plan=propose_plan,
        allow_plan_step_update=update_plan_step,
    )
    json_fallback_action_protocol = format_action_protocol_for_prompt(
        native=False,
        allow_final_answer=allow_final_answer,
        allow_tool_call=registered_tool_calls_available,
        allow_plan=propose_plan,
        allow_plan_step_update=update_plan_step,
    )
    tool_metadata = {
        "tool_names": (
            native_declaration_names
            if native_action_protocol
            else [tool.name for tool in action_tools]
        ),
        "delivery": "provider_native" if native_action_protocol else "prompt",
        "schemas_embedded": not native_action_protocol,
    }
    system_parts: list[tuple[str, str, str, dict]] = [
        ("system_prompt", "Base system prompt", system_instructions_prompt, {}),
    ]
    if profile_memories or relevant_memories:
        system_parts.append(
            (
                "memories",
                "Selected memories",
                format_memories_for_prompt(
                    profile_memories=profile_memories,
                    relevant_memories=relevant_memories,
                ),
                {
                    "profile_memory_ids": [memory.id for memory in profile_memories],
                    "relevant_memory_ids": [memory.id for memory in relevant_memories],
                },
            )
        )
    if selected_skills:
        system_parts.append(
            (
                "skills",
                "Selected skills",
                format_skills_for_prompt(
                    selected_skills,
                    max_chars_per_skill=max_skill_content_chars,
                ),
                {"skill_names": [selection.skill.name for selection in selected_skills]},
            )
        )
    if memory.conversation_summary:
        system_parts.append(
            (
                "conversation_summary",
                "Conversation summary",
                format_conversation_summary_for_prompt(memory.conversation_summary),
                {
                    "summary_message_count": memory.summary_message_count,
                    "has_summary": True,
                },
            )
        )
    if planning_enabled:
        system_parts.append(
            (
                "planning",
                "Planning instructions",
                format_planning_for_prompt(
                    planning_enabled=True,
                    active_plan=active_plan,
                    plan_approved=plan_approved,
                    require_plan=require_plan,
                    max_reconnaissance_tool_calls=max_tool_calls_per_turn,
                    read_only_tool_names=planning_tool_names,
                ),
                {"enabled": True},
            )
        )
    if action_tools or (native_action_protocol and safe_native_tool_declarations):
        system_parts.append(
            (
                "tools",
                "Available tools",
                format_tools_for_prompt(
                    tool_descriptions,
                    delivery="provider_native" if native_action_protocol else "prompt",
                    native_tool_count=(
                        len(safe_native_tool_declarations)
                        if native_action_protocol
                        else None
                    ),
                ),
                tool_metadata,
            )
        )
    if action_tools:
        system_parts.append(
            (
                "tool_rules",
                "Tool-call rules",
                format_tool_call_rules(max_tool_calls_per_turn),
                {"max_tool_calls_per_turn": max_tool_calls_per_turn},
            )
        )
    system_parts.append(
        (
            "action_protocol",
            "Action protocol",
            action_protocol,
            {
                "native_tool_calling": native_action_protocol,
                "action_transport": action_transport,
            },
        )
    )
    if prompt_profile or locale:
        prompt_metadata_prompt = format_prompt_metadata_for_prompt(
            prompt_profile=prompt_profile,
            locale=locale,
        )
        system_parts.insert(
            1,
            (
                "prompt_metadata",
                "Prompt metadata",
                prompt_metadata_prompt,
                {"prompt_profile": prompt_profile, "locale": locale},
            ),
        )
    if turn_context_sections:
        external_context_prompt = format_context_sections_for_prompt(
            turn_context_sections
        )
        insert_index = 2 if prompt_profile or locale else 1
        system_parts.insert(
            insert_index,
            (
                "external_context",
                "External turn context",
                external_context_prompt,
                {"context_section_ids": [section.id for section in turn_context_sections]},
            ),
        )
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
    native_tools_section: ContextSection | None = None
    request_overhead_estimated_tokens = 0
    if native_action_protocol and safe_native_tool_declarations:
        serialized_declarations = _serialize_native_tool_declarations(
            safe_native_tool_declarations
        )
        declaration_names = _native_tool_declaration_names(
            safe_native_tool_declarations
        )
        native_tools_section = ContextSection.from_text(
            "native_tools",
            "Provider-native tool declarations",
            serialized_declarations,
            item_count=len(safe_native_tool_declarations),
            metadata={
                "tool_names": declaration_names,
                "delivery": "provider_native",
                "schemas_embedded": False,
            },
        )
        request_overhead_estimated_tokens = native_tools_section.estimated_tokens
        tools_section_index = next(
            (
                index
                for index, section in enumerate(system_sections)
                if section.name == "tools"
            ),
            len(system_sections) - 1,
        )
        system_sections.insert(tools_section_index + 1, native_tools_section)
    composed_system_prompt = _compose_xml_system_prompt(system_parts)
    system_message = {"role": "system", "content": composed_system_prompt}
    alternate_system_message: dict[str, str] | None = None
    if native_action_protocol:
        alternate_system_parts = _json_fallback_system_parts(
            system_parts,
            full_tools_prompt=(
                format_tools_for_prompt(tool_descriptions)
                if action_tools
                else None
            ),
            json_action_protocol=json_fallback_action_protocol,
        )
        alternate_system_message = {
            "role": "system",
            "content": _compose_xml_system_prompt(alternate_system_parts),
        }
    budget = context_budget or ContextBudget()
    history_messages, omitted_messages = select_messages_for_budget(
        system_message=system_message,
        history_messages=memory.recent(),
        budget=budget,
        request_overhead_tokens=request_overhead_estimated_tokens,
        alternate_system_message=alternate_system_message,
    )
    messages = [system_message, *history_messages]
    fallback_sent_messages = (
        [alternate_system_message, *history_messages]
        if alternate_system_message is not None
        else None
    )
    context_report = build_context_report(
        system_sections=system_sections,
        history_messages=history_messages,
        omitted_messages=omitted_messages,
        budget=budget,
        sent_messages=messages,
        request_overhead_estimated_tokens=request_overhead_estimated_tokens,
        fallback_sent_messages=fallback_sent_messages,
    )
    return AgentPrompt(
        messages=messages,
        context_report=context_report,
        omitted_messages=omitted_messages,
        action_transport=action_transport,
        native_tool_declarations=safe_native_tool_declarations,
    )


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


def _compose_xml_system_prompt(system_parts: list[tuple[str, str, str, dict]]) -> str:
    sections = ["<chulk_prompt>"]
    for name, _label, content, _metadata in system_parts:
        clean_content = content.strip()
        if clean_content.startswith(f"<{name}>") and clean_content.endswith(
            f"</{name}>"
        ):
            sections.extend([clean_content, ""])
            continue
        sections.extend(
            [
                f"<{name}>",
                clean_content,
                f"</{name}>",
                "",
            ]
        )
    if sections[-1] == "":
        sections.pop()
    sections.append("</chulk_prompt>")
    return "\n".join(sections)


def _json_fallback_system_parts(
    system_parts: list[tuple[str, str, str, dict]],
    *,
    full_tools_prompt: str | None,
    json_action_protocol: str,
) -> list[tuple[str, str, str, dict]]:
    fallback_parts: list[tuple[str, str, str, dict]] = []
    for name, label, content, metadata in system_parts:
        if name == "tools":
            if full_tools_prompt is None:
                continue
            fallback_parts.append(
                (
                    name,
                    label,
                    full_tools_prompt,
                    {
                        **metadata,
                        "delivery": "prompt",
                        "schemas_embedded": True,
                    },
                )
            )
            continue
        if name == "action_protocol":
            fallback_parts.append(
                (
                    name,
                    label,
                    json_action_protocol,
                    {
                        **metadata,
                        "native_tool_calling": False,
                        "action_transport": "chulk_json",
                    },
                )
            )
            continue
        fallback_parts.append((name, label, content, dict(metadata)))
    return fallback_parts


def _safe_native_tool_declarations(
    declarations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if declarations is None:
        return []
    try:
        safe_value = json.loads(
            json.dumps(declarations, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("native_tool_declarations must be JSON-safe") from exc
    if not isinstance(safe_value, list) or any(
        not isinstance(declaration, dict) for declaration in safe_value
    ):
        raise TypeError("native_tool_declarations must be a list of objects")
    return safe_value


def _registered_native_tool_declarations(
    tools: Iterable[object],
) -> list[dict[str, Any]]:
    return [
        {
            "name": str(getattr(tool, "name")),
            "description": str(getattr(tool, "description", "")),
            "parameters": getattr(tool, "args_schema", {}) or {},
        }
        for tool in tools
    ]


def _serialize_native_tool_declarations(
    declarations: list[dict[str, Any]],
) -> str:
    if not declarations:
        return ""
    return json.dumps(
        declarations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _native_tool_declaration_names(
    declarations: list[dict[str, Any]],
) -> list[str]:
    return [
        name
        for declaration in declarations
        if isinstance(name := declaration.get("name"), str) and name
    ]
