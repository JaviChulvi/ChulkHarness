"""Prompt templates for the agent loop."""

from html import escape
import json

from chulk.core.context import TurnContextSection
from chulk.memory import MemoryRecord
from chulk.skills import Skill, SkillSelection
from chulk.core.planning import format_read_only_planning_tools
from chulk.core.state import Plan

MAX_MEMORY_PROMPT_CONTENT_CHARS = 500
MAX_SKILL_PROMPT_CONTENT_CHARS = 4000
MAX_CONTEXT_SECTION_CHARS = 1500

BASE_SYSTEM_PROMPT = """You are ChulkHarness, a lightweight Python agent harness.

Answer the user's message directly and clearly. Use the recent conversation history for context.
When tools are available, call a tool only when it materially helps answer the user.
"""

JSON_ACTION_PROMPT = """<response_protocol>
<transport>json_object</transport>
<primary_rule>You must respond with exactly one JSON object and no extra prose.</primary_rule>
<formats>
<format>
<label>Direct answer format:</label>
<json_example>
{"type": "final_answer", "content": "...", "tool_name": null, "arguments_json": "{}", "plan_json": "{}", "step_update_json": "{}"}
</json_example>
</format>
<format>
<label>Plan format:</label>
<json_example>
{"type": "plan", "content": null, "tool_name": null, "arguments_json": "{}", "plan_json": "{\\\"summary\\\":\\\"...\\\",\\\"steps\\\":[{\\\"id\\\":\\\"1\\\",\\\"title\\\":\\\"...\\\",\\\"description\\\":\\\"...\\\",\\\"status\\\":\\\"pending\\\",\\\"depends_on\\\":[],\\\"acceptance_criteria\\\":[\\\"...\\\"],\\\"retry_limit\\\":0}]}", "step_update_json": "{}"}
</json_example>
</format>
<format>
<label>Tool call format:</label>
<json_example>
{"type": "tool_call", "content": null, "tool_name": "tool_name", "arguments_json": "{\\\"arg\\\":\\\"value\\\"}", "plan_json": "{}", "step_update_json": "{}"}
</json_example>
</format>
<format>
<label>Plan step update format:</label>
<json_example>
{"type": "plan_step_update", "content": null, "tool_name": null, "arguments_json": "{}", "plan_json": "{}", "step_update_json": "{\\\"step_id\\\":\\\"1\\\",\\\"status\\\":\\\"completed\\\",\\\"evidence\\\":\\\"...\\\",\\\"reason\\\":null}"}
</json_example>
</format>
</formats>
<rules>
<rule>Use a final_answer when you can answer without a tool. Use a tool_call when you need a listed tool.</rule>
<rule>If you intend to use a tool, the type must be tool_call; never put tool_name or arguments_json on a final_answer.</rule>
<rule>If you intend to answer the user, the type must be final_answer, tool_name must be null, and arguments_json must be "{}".</rule>
<rule>For tool calls, arguments_json must be a JSON-encoded object string containing the tool arguments.</rule>
<rule>For tool calls, use only argument fields from the listed tool schema.</rule>
<rule>Use a plan only when the Planning section explicitly tells you to propose a plan.</rule>
<rule>For plans, plan_json must be a JSON-encoded object string with summary and executable steps.</rule>
<rule>Each plan step should include depends_on, acceptance_criteria, and retry_limit 0.</rule>
<rule>When an approved plan is active, work only on the current executable step.</rule>
<rule>After tool evidence satisfies that step's acceptance criteria, return plan_step_update instead of final_answer.</rule>
<rule>Use final_answer only after the approved plan is completed.</rule>
<rule>After an observation is provided, use it to produce the next tool_call, plan_step_update, or final_answer.</rule>
<rule>Some tool observations may contain bounded head/tail previews plus local artifact paths for full output.</rule>
<rule>If the omitted middle may contain information needed to answer correctly, inspect the artifact or run a narrower follow-up tool call before giving a final_answer.</rule>
</rules>
</response_protocol>
"""

NATIVE_ACTION_PROMPT = """<response_protocol>
<transport>provider_native_tool_calling</transport>
<primary_rule>Use the provider-native tool-calling interface for actions.</primary_rule>
<rules>
<rule>Use normal assistant text only for a direct final answer to the user.</rule>
<rule>When you need a listed Chulk tool, call that tool through the native tool interface.</rule>
<rule>For tool calls, use only argument fields from the listed tool schema.</rule>
<rule>When the Planning section tells you to propose a plan, call chulk_propose_plan.</rule>
<rule>For plans, provide a summary and executable steps with depends_on, acceptance_criteria, and retry_limit 0.</rule>
<rule>When an approved plan is active, work only on the current executable step.</rule>
<rule>After tool evidence satisfies that step's acceptance criteria, call chulk_plan_step_update instead of answering directly.</rule>
<rule>Use a final answer only after the approved plan is completed.</rule>
<rule>After an observation is provided, use it to produce the next native tool call, plan step update, or final answer.</rule>
<rule>Some tool observations may contain bounded head/tail previews plus local artifact paths for full output.</rule>
<rule>If the omitted middle may contain information needed to answer correctly, inspect the artifact or run a narrower follow-up tool call before giving a final answer.</rule>
</rules>
</response_protocol>
"""

JSON_REPAIR_PROMPT = """Your previous response could not be parsed as ChulkHarness action JSON.
Return exactly one valid JSON object using one of these shapes:
{"type": "final_answer", "content": "...", "tool_name": null, "arguments_json": "{}", "plan_json": "{}", "step_update_json": "{}"}
{"type": "plan", "content": null, "tool_name": null, "arguments_json": "{}", "plan_json": "{\\\"summary\\\":\\\"...\\\",\\\"steps\\\":[{\\\"id\\\":\\\"1\\\",\\\"title\\\":\\\"...\\\",\\\"description\\\":\\\"...\\\",\\\"status\\\":\\\"pending\\\",\\\"depends_on\\\":[],\\\"acceptance_criteria\\\":[\\\"...\\\"],\\\"retry_limit\\\":0}]}", "step_update_json": "{}"}
{"type": "tool_call", "content": null, "tool_name": "tool_name", "arguments_json": "{\\\"arg\\\":\\\"value\\\"}", "plan_json": "{}", "step_update_json": "{}"}
{"type": "plan_step_update", "content": null, "tool_name": null, "arguments_json": "{}", "plan_json": "{}", "step_update_json": "{\\\"step_id\\\":\\\"1\\\",\\\"status\\\":\\\"completed\\\",\\\"evidence\\\":\\\"...\\\",\\\"reason\\\":null}"}
Do not include Markdown fences, comments, or prose outside the JSON object.
If your previous response included tool_name or tool arguments, return a tool_call action.
If your previous response was a direct answer, return a final_answer action with tool_name null and arguments_json "{}".
If an observation reported invalid_arguments, remove unsupported fields and use only the listed schema fields.
If plan execution feedback says a plan is incomplete, continue the current step or return a plan_step_update.
"""

REFLECTION_PROMPT = """You are ChulkHarness's final-answer reviewer.
Review a proposed final answer before it is shown to the user.

Return only one JSON object with this shape:
{
  "approved": true,
  "reason": "short reason",
  "feedback": null
}

Set approved to false only when the answer has a material issue: it contradicts the user's request, ignores tool output, skips an approved plan step, hides a tool error, claims unverified work, or should take another action before answering.
Do not reject merely for style, wording preference, or optional extra detail.
When approved is false, feedback must be a concise instruction for the next model action.
Do not include Markdown fences, comments, or prose outside the JSON object.
"""


def format_system_instructions_for_prompt(system_prompt: str) -> str:
    """Format host/system instructions as escaped XML prompt content."""
    return "\n".join(
        [
            "<system_instructions>",
            "<instruction_text>",
            _xml_text(system_prompt.strip()),
            "</instruction_text>",
            "</system_instructions>",
        ]
    )


def format_tool_call_rules(max_tool_calls_per_turn: int) -> str:
    """Format per-turn tool-call limits and recovery guidance."""
    return "\n".join(
        [
            "<tool_call_rules>",
            f"<max_tool_calls_per_turn>{max_tool_calls_per_turn}</max_tool_calls_per_turn>",
            (
                "<rule>Tool-call limit: you may request at most "
                f"{max_tool_calls_per_turn} tool calls for this user turn.</rule>"
            ),
            "<rule>If a tool observation reports invalid arguments, retry only when you can correct the arguments from the schema.</rule>",
            "<rule>If a tool is unavailable or still failing, explain the limitation instead of repeatedly calling it.</rule>",
            "</tool_call_rules>",
        ]
    )


def format_tools_for_prompt(tool_descriptions: str) -> str:
    """Format available tools for prompt injection."""
    if not tool_descriptions:
        return "\n".join(["<available_tools>", "<status>Available tools: none.</status>", "</available_tools>"])

    try:
        tools = json.loads(tool_descriptions)
    except json.JSONDecodeError:
        return "\n".join(
            [
                "<available_tools>",
                "<status>Available tools: provided as raw JSON.</status>",
                "<tool_catalog_json>",
                _xml_text(tool_descriptions),
                "</tool_catalog_json>",
                "</available_tools>",
            ]
        )
    tool_entries = [tool for tool in tools if isinstance(tool, dict)] if isinstance(tools, list) else []
    if not tool_entries:
        return "\n".join(["<available_tools>", "<status>Available tools: none.</status>", "</available_tools>"])

    lines = [
        "<available_tools>",
        "<status>Available tools are callable actions for this turn.</status>",
    ]
    for tool in tool_entries:
        arguments_schema = json.dumps(tool.get("arguments", {}), indent=2, sort_keys=True)
        lines.extend(
            [
                "<tool>",
                f"<name>{_xml_text(tool.get('name', ''))}</name>",
                f"<description>{_xml_text(tool.get('description', ''))}</description>",
                f"<requires_confirmation>{_xml_text(tool.get('requires_confirmation', False))}</requires_confirmation>",
                f"<permission_level>{_xml_text(tool.get('permission_level', 'read'))}</permission_level>",
                "<arguments_schema_json>",
                _xml_text(arguments_schema),
                "</arguments_schema_json>",
                "</tool>",
            ]
        )
    lines.append("</available_tools>")
    return "\n".join(lines)


def format_available_skills_for_prompt(available_skills: list[Skill]) -> str:
    """Format registered skill metadata for prompt injection."""
    if not available_skills:
        return "\n".join(["<available_skills>", "<status>Available skills: none.</status>", "</available_skills>"])

    lines = [
        "<available_skills>",
        "<summary>Available skills are prompt-loadable procedural playbooks.</summary>",
        "<boundary>This catalog is metadata only; detailed instructions are available only under Loaded skills.</boundary>",
        "<rule>Do not claim to follow a skill's detailed procedure unless that skill is loaded for this turn.</rule>",
    ]
    for skill in available_skills:
        description = skill.description.strip() or "No description provided."
        lines.extend(
            [
                "<skill>",
                f"<name>{_xml_text(skill.name)}</name>",
                f"<description>{_xml_text(description)}</description>",
                f"<catalog_line>- {_xml_text(skill.name)}: {_xml_text(description)}</catalog_line>",
                "</skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def format_conversation_summary_for_prompt(summary: str | None) -> str:
    """Format the task-local compact summary for prompt injection."""
    if not summary:
        return "\n".join(["<conversation_summary>", "<status>Conversation summary: none.</status>", "</conversation_summary>"])
    return "\n".join(
        [
            "<conversation_summary>",
            "<summary>Conversation summary from earlier turns:</summary>",
            "<boundary>This is task-local context, not durable long-term memory.</boundary>",
            "<content>",
            _xml_text(summary),
            "</content>",
            "</conversation_summary>",
        ]
    )


def format_prompt_metadata_for_prompt(*, prompt_profile: str | None, locale: str | None) -> str:
    """Format optional host-owned prompt metadata."""
    if not prompt_profile and not locale:
        return "\n".join(
            [
                "<prompt_metadata>",
                "<status>Prompt metadata: no host profile or locale provided.</status>",
                "</prompt_metadata>",
            ]
        )
    lines = ["<prompt_metadata>"]
    if prompt_profile:
        lines.append(f"<profile>{_xml_text(prompt_profile)}</profile>")
    if locale:
        lines.append(f"<locale>{_xml_text(locale)}</locale>")
    lines.append("<rule>Treat this metadata as host configuration, not as user-provided instructions.</rule>")
    lines.append("</prompt_metadata>")
    return "\n".join(lines)


def format_context_sections_for_prompt(context_sections: list[TurnContextSection]) -> str:
    """Format host-provided retrieved context without storing it as memory."""
    if not context_sections:
        return "\n".join(
            [
                "<external_turn_context>",
                "<status>External turn context: none provided.</status>",
                "</external_turn_context>",
            ]
        )
    lines = [
        "<external_turn_context>",
        "<summary>External turn context supplied by the host application.</summary>",
        "<boundary>Use these snippets only for this turn. They are not long-term memory, skills, or tools.</boundary>",
    ]
    for section in context_sections:
        lines.extend(
            [
                "<context_section>",
                f"<id>{_xml_text(section.id)}</id>",
                f"<title>{_xml_text(section.title or '')}</title>",
                f"<source>{_xml_text(section.source or '')}</source>",
                "<content>",
                _xml_text(_truncate_context_section(section.content)),
                "</content>",
                "</context_section>",
            ]
        )
    lines.append("</external_turn_context>")
    return "\n".join(lines)


def format_planning_for_prompt(
    *,
    planning_enabled: bool,
    active_plan: Plan | None,
    plan_approved: bool,
    require_plan: bool,
    max_reconnaissance_tool_calls: int,
) -> str:
    """Format one-shot planning instructions for prompt injection."""
    if not planning_enabled:
        return "\n".join(["<planning>", "<status>Planning: not requested for this turn.</status>", "</planning>"])

    if require_plan and active_plan is None:
        read_only_tools = format_read_only_planning_tools()
        return "\n".join(
            [
                "<planning>",
                "<status>Planning: requested for this turn.</status>",
                f"<read_only_reconnaissance_tools>{_xml_text(read_only_tools)}</read_only_reconnaissance_tools>",
                f"<max_reconnaissance_tool_calls>{max_reconnaissance_tool_calls}</max_reconnaissance_tool_calls>",
                f"<rule>Before proposing the plan, you may call only these read-only reconnaissance tools: {_xml_text(read_only_tools)}.</rule>",
                "<rule>Use reconnaissance when codebase details matter; inspect the smallest useful set of files before planning.</rule>",
                "<rule>Use search_files to locate symbols or factory functions instead of guessing which file owns them.</rule>",
                "<rule>Only name files/modules in the plan when they are supported by listed, searched, or read observations.</rule>",
                f"<rule>You have at most {max_reconnaissance_tool_calls} reconnaissance tool calls. Do not spend them all unless necessary.</rule>",
                "<rule>After two or three useful file reads/searches, stop reconnaissance and return the approval plan.</rule>",
                "<rule>Do not call shell, write, memory-mutation, import/export, or other mutating tools before approval.</rule>",
                "<rule>After reconnaissance, return a plan action. Do not execute implementation steps until the user approves the plan.</rule>",
                "<rule>Keep the plan concrete, short, and executable, with specific files/modules when they are known.</rule>",
                "<rule>For each plan step, include depends_on, acceptance_criteria, and retry_limit 0.</rule>",
                "<rule>Use dependencies only when a step truly cannot start until an earlier step is completed.</rule>",
                "<rule>The approval plan must be an implementation plan. Do not make read/list/search/explore/inspect steps the plan.</rule>",
                "<rule>Because the user explicitly requested /plan, do not answer directly. Return a plan action after any needed reconnaissance.</rule>",
                "</planning>",
            ]
        )

    if active_plan is not None and plan_approved:
        return "\n".join(
            [
                "<planning>",
                "<status>Planning: approved for this turn.</status>",
                "<rule>Follow the approved plan while executing this turn.</rule>",
                "<rule>Only work on the current executable step. Use tools until its acceptance criteria are satisfied.</rule>",
                "<rule>When the current step is satisfied, return a plan_step_update action for that step.</rule>",
                "<rule>Do not return a final_answer until every plan step is completed.</rule>",
                "<active_plan>",
                _xml_text(active_plan.to_prompt()),
                "</active_plan>",
                "</planning>",
            ]
        )

    if active_plan is not None:
        return "\n".join(
            [
                "<planning>",
                "<status>Planning: waiting for user approval.</status>",
                "<rule>Do not call tools while the plan is pending approval.</rule>",
                "<active_plan>",
                _xml_text(active_plan.to_prompt()),
                "</active_plan>",
                "</planning>",
            ]
        )

    return "\n".join(["<planning>", "<status>Planning: requested for this turn.</status>", "</planning>"])


def format_memories_for_prompt(
    *,
    profile_memories: list[MemoryRecord],
    relevant_memories: list[MemoryRecord],
) -> str:
    """Format selected long-term memories for prompt injection."""
    if not profile_memories and not relevant_memories:
        return "\n".join(
            [
                "<long_term_memory>",
                "<status>Long-term memory: no relevant memories selected for this turn.</status>",
                "</long_term_memory>",
            ]
        )

    sections = [
        "<long_term_memory>",
        "<summary>Long-term memory contains durable facts, preferences, and project context.</summary>",
        "<boundary>Use it only when relevant. It is not a skill, a tool, or an instruction playbook.</boundary>",
    ]

    if profile_memories:
        sections.append("<profile_memories>")
        sections.append("<label>Persona and workflow preferences:</label>")
        sections.extend(_format_memory_record(memory) for memory in profile_memories)
        sections.append("</profile_memories>")

    if relevant_memories:
        sections.append("<relevant_memories>")
        sections.append("<label>Relevant contextual memories:</label>")
        sections.extend(_format_memory_record(memory) for memory in relevant_memories)
        sections.append("</relevant_memories>")

    sections.append("</long_term_memory>")
    return "\n".join(sections)


def format_skills_for_prompt(
    selected_skills: list[SkillSelection],
    *,
    max_chars_per_skill: int = MAX_SKILL_PROMPT_CONTENT_CHARS,
) -> str:
    """Format selected skill instructions for prompt injection."""
    if not selected_skills:
        return "\n".join(["<loaded_skills>", "<status>Loaded skills: none selected for this turn.</status>", "</loaded_skills>"])

    sections = [
        "<loaded_skills>",
        "<summary>Loaded skills are procedural instructions for this turn.</summary>",
        "<boundary>They are not tools and they are not long-term memory.</boundary>",
        "<rule>Use them only when relevant to the user's request.</rule>",
    ]
    for selection in selected_skills:
        skill = selection.skill
        content = skill.loaded_content or ""
        sections.extend(
            [
                "<skill>",
                f"<label>Skill: {_xml_text(skill.name)}</label>",
                f"<name>{_xml_text(skill.name)}</name>",
                f"<source>{_xml_text(skill.path)}</source>",
                f"<description>{_xml_text(skill.description)}</description>",
                f"<matched_keywords>{_xml_text(', '.join(selection.matched_keywords) or 'none')}</matched_keywords>",
                "<instructions>",
                _xml_text(_truncate_skill_content(content, max_chars_per_skill)),
                "</instructions>",
                "</skill>",
            ]
        )
    sections.append("</loaded_skills>")
    return "\n".join(sections)


def _format_memory_record(memory: MemoryRecord) -> str:
    tag_text = ", ".join(memory.tags) if memory.tags else "untagged"
    return "\n".join(
        [
            "<memory>",
            f"<id>{_xml_text(memory.id)}</id>",
            f"<tags>{_xml_text(tag_text)}</tags>",
            f"<importance>{memory.importance}</importance>",
            "<content>",
            _xml_text(_truncate_memory_content(memory.content)),
            "</content>",
            "</memory>",
        ]
    )


def _truncate_memory_content(content: str) -> str:
    if len(content) <= MAX_MEMORY_PROMPT_CONTENT_CHARS:
        return content
    return content[:MAX_MEMORY_PROMPT_CONTENT_CHARS].rstrip() + "..."


def _truncate_context_section(content: str) -> str:
    if len(content) <= MAX_CONTEXT_SECTION_CHARS:
        return content
    return content[:MAX_CONTEXT_SECTION_CHARS].rstrip() + "..."


def _truncate_skill_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "..."


def _xml_text(value: object) -> str:
    return escape(str(value), quote=False)
