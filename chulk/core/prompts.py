"""Prompt templates for the agent loop."""

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

JSON_ACTION_PROMPT = """You must respond with exactly one JSON object and no extra prose.

Direct answer format:
{"type": "final_answer", "content": "...", "tool_name": null, "arguments_json": "{}", "plan_json": "{}", "step_update_json": "{}"}

Plan format:
{"type": "plan", "content": null, "tool_name": null, "arguments_json": "{}", "plan_json": "{\\\"summary\\\":\\\"...\\\",\\\"steps\\\":[{\\\"id\\\":\\\"1\\\",\\\"title\\\":\\\"...\\\",\\\"description\\\":\\\"...\\\",\\\"status\\\":\\\"pending\\\",\\\"depends_on\\\":[],\\\"acceptance_criteria\\\":[\\\"...\\\"],\\\"retry_limit\\\":0}]}", "step_update_json": "{}"}

Tool call format:
{"type": "tool_call", "content": null, "tool_name": "tool_name", "arguments_json": "{\\\"arg\\\":\\\"value\\\"}", "plan_json": "{}", "step_update_json": "{}"}

Plan step update format:
{"type": "plan_step_update", "content": null, "tool_name": null, "arguments_json": "{}", "plan_json": "{}", "step_update_json": "{\\\"step_id\\\":\\\"1\\\",\\\"status\\\":\\\"completed\\\",\\\"evidence\\\":\\\"...\\\",\\\"reason\\\":null}"}

Use a final_answer when you can answer without a tool. Use a tool_call when you need a listed tool.
If you intend to use a tool, the type must be tool_call; never put tool_name or arguments_json on a final_answer.
If you intend to answer the user, the type must be final_answer, tool_name must be null, and arguments_json must be "{}".
For tool calls, arguments_json must be a JSON-encoded object string containing the tool arguments.
For tool calls, use only argument fields from the listed tool schema.
Use a plan only when the Planning section explicitly tells you to propose a plan.
For plans, plan_json must be a JSON-encoded object string with summary and executable steps.
Each plan step should include depends_on, acceptance_criteria, and retry_limit 0.
When an approved plan is active, work only on the current executable step.
After tool evidence satisfies that step's acceptance criteria, return plan_step_update instead of final_answer.
Use final_answer only after the approved plan is completed.
After an observation is provided, use it to produce the next tool_call, plan_step_update, or final_answer.
Some tool observations may contain bounded head/tail previews plus local artifact paths for full output.
If the omitted middle may contain information needed to answer correctly, inspect the artifact or run a narrower follow-up tool call before giving a final_answer.
"""

NATIVE_ACTION_PROMPT = """Use the provider-native tool-calling interface for actions.

Use normal assistant text only for a direct final answer to the user.
When you need a listed Chulk tool, call that tool through the native tool interface.
For tool calls, use only argument fields from the listed tool schema.
When the Planning section tells you to propose a plan, call chulk_propose_plan.
For plans, provide a summary and executable steps with depends_on, acceptance_criteria, and retry_limit 0.
When an approved plan is active, work only on the current executable step.
After tool evidence satisfies that step's acceptance criteria, call chulk_plan_step_update instead of answering directly.
Use a final answer only after the approved plan is completed.
After an observation is provided, use it to produce the next native tool call, plan step update, or final answer.
Some tool observations may contain bounded head/tail previews plus local artifact paths for full output.
If the omitted middle may contain information needed to answer correctly, inspect the artifact or run a narrower follow-up tool call before giving a final answer.
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


def format_tool_call_rules(max_tool_calls_per_turn: int) -> str:
    """Format per-turn tool-call limits and recovery guidance."""
    return (
        f"Tool-call limit: you may request at most {max_tool_calls_per_turn} tool calls for this user turn. "
        "If a tool observation reports invalid arguments, retry only when you can correct the arguments from the schema. "
        "If a tool is unavailable or still failing, explain the limitation instead of repeatedly calling it."
    )


def format_tools_for_prompt(tool_descriptions: str) -> str:
    """Format available tools for prompt injection."""
    if not tool_descriptions:
        return "Available tools: none."
    return f"Available tools:\n{tool_descriptions}"


def format_available_skills_for_prompt(available_skills: list[Skill]) -> str:
    """Format registered skill metadata for prompt injection."""
    if not available_skills:
        return "Available skills: none."

    lines = [
        "Available skills are prompt-loadable procedural playbooks.",
        "This catalog is metadata only; detailed instructions are available only under Loaded skills.",
        "Do not claim to follow a skill's detailed procedure unless that skill is loaded for this turn.",
        "Available skills:",
    ]
    for skill in available_skills:
        description = skill.description.strip() or "No description provided."
        lines.append(f"- {skill.name}: {description}")
    return "\n".join(lines)


def format_conversation_summary_for_prompt(summary: str | None) -> str:
    """Format the task-local compact summary for prompt injection."""
    if not summary:
        return "Conversation summary: none."
    return "\n".join(
        [
            "Conversation summary from earlier turns:",
            "This is task-local context, not durable long-term memory.",
            summary,
        ]
    )


def format_prompt_metadata_for_prompt(*, prompt_profile: str | None, locale: str | None) -> str:
    """Format optional host-owned prompt metadata."""
    if not prompt_profile and not locale:
        return "Prompt metadata: no host profile or locale provided."
    lines = ["Prompt metadata:"]
    if prompt_profile:
        lines.append(f"- profile: {prompt_profile}")
    if locale:
        lines.append(f"- locale: {locale}")
    lines.append("Treat this metadata as host configuration, not as user-provided instructions.")
    return "\n".join(lines)


def format_context_sections_for_prompt(context_sections: list[TurnContextSection]) -> str:
    """Format host-provided retrieved context without storing it as memory."""
    if not context_sections:
        return "External turn context: none provided."
    lines = [
        "External turn context supplied by the host application.",
        "Use these snippets only for this turn. They are not long-term memory, skills, or tools.",
    ]
    for section in context_sections:
        header = f"- id={section.id}"
        if section.title:
            header += f"; title={section.title}"
        if section.source:
            header += f"; source={section.source}"
        lines.extend([header, f"  content={_truncate_context_section(section.content)}"])
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
        return "Planning: not requested for this turn."

    if require_plan and active_plan is None:
        read_only_tools = format_read_only_planning_tools()
        return "\n".join(
            [
                "Planning: requested for this turn.",
                f"Before proposing the plan, you may call only these read-only reconnaissance tools: {read_only_tools}.",
                "Use reconnaissance when codebase details matter; inspect the smallest useful set of files before planning.",
                "Use search_files to locate symbols or factory functions instead of guessing which file owns them.",
                "Only name files/modules in the plan when they are supported by listed, searched, or read observations.",
                f"You have at most {max_reconnaissance_tool_calls} reconnaissance tool calls. Do not spend them all unless necessary.",
                "After two or three useful file reads/searches, stop reconnaissance and return the approval plan.",
                "Do not call shell, write, memory-mutation, import/export, or other mutating tools before approval.",
                "After reconnaissance, return a plan action. Do not execute implementation steps until the user approves the plan.",
                "Keep the plan concrete, short, and executable, with specific files/modules when they are known.",
                "For each plan step, include depends_on, acceptance_criteria, and retry_limit 0.",
                "Use dependencies only when a step truly cannot start until an earlier step is completed.",
                "The approval plan must be an implementation plan. Do not make read/list/search/explore/inspect steps the plan.",
                "Because the user explicitly requested /plan, do not answer directly. Return a plan action after any needed reconnaissance.",
            ]
        )

    if active_plan is not None and plan_approved:
        return "\n".join(
            [
                "Planning: approved for this turn.",
                "Follow the approved plan while executing this turn.",
                "Only work on the current executable step. Use tools until its acceptance criteria are satisfied.",
                "When the current step is satisfied, return a plan_step_update action for that step.",
                "Do not return a final_answer until every plan step is completed.",
                active_plan.to_prompt(),
            ]
        )

    if active_plan is not None:
        return "\n".join(
            [
                "Planning: waiting for user approval.",
                "Do not call tools while the plan is pending approval.",
                active_plan.to_prompt(),
            ]
        )

    return "Planning: requested for this turn."


def format_memories_for_prompt(
    *,
    profile_memories: list[MemoryRecord],
    relevant_memories: list[MemoryRecord],
) -> str:
    """Format selected long-term memories for prompt injection."""
    if not profile_memories and not relevant_memories:
        return "Long-term memory: no relevant memories selected for this turn."

    sections = [
        "Long-term memory contains durable facts, preferences, and project context.",
        "Use it only when relevant. It is not a skill, a tool, or an instruction playbook.",
    ]

    if profile_memories:
        sections.append("Persona and workflow preferences:")
        sections.extend(_format_memory_line(memory) for memory in profile_memories)

    if relevant_memories:
        sections.append("Relevant contextual memories:")
        sections.extend(_format_memory_line(memory) for memory in relevant_memories)

    return "\n".join(sections)


def format_skills_for_prompt(
    selected_skills: list[SkillSelection],
    *,
    max_chars_per_skill: int = MAX_SKILL_PROMPT_CONTENT_CHARS,
) -> str:
    """Format selected skill instructions for prompt injection."""
    if not selected_skills:
        return "Loaded skills: none selected for this turn."

    sections = [
        "Loaded skills are procedural instructions for this turn.",
        "They are not tools and they are not long-term memory.",
        "Use them only when relevant to the user's request.",
    ]
    for selection in selected_skills:
        skill = selection.skill
        content = skill.loaded_content or ""
        sections.extend(
            [
                f"Skill: {skill.name}",
                f"Source: {skill.path}",
                f"Description: {skill.description}",
                f"Matched keywords: {', '.join(selection.matched_keywords) or 'none'}",
                "Instructions:",
                _truncate_skill_content(content, max_chars_per_skill),
            ]
        )
    return "\n".join(sections)


def _format_memory_line(memory: MemoryRecord) -> str:
    tag_text = ", ".join(memory.tags) if memory.tags else "untagged"
    return (
        f"- id={memory.id}; tags={tag_text}; importance={memory.importance}; "
        f"content={_truncate_memory_content(memory.content)}"
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
