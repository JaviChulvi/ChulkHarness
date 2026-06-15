"""Prompt templates for the agent loop."""

from chulk.memory import MemoryRecord
from chulk.skills import SkillSelection
from chulk.core.planning import format_read_only_planning_tools
from chulk.core.state import Plan

MAX_MEMORY_PROMPT_CONTENT_CHARS = 500
MAX_SKILL_PROMPT_CONTENT_CHARS = 4000

BASE_SYSTEM_PROMPT = """You are ChulkHarness, a lightweight Python agent harness.

Answer the user's message directly and clearly. Use the recent conversation history for context.
When tools are available, call a tool only when it materially helps answer the user.
"""

JSON_ACTION_PROMPT = """You must respond with exactly one JSON object and no extra prose.

Direct answer format:
{"type": "final_answer", "content": "...", "tool_name": null, "arguments_json": "{}", "plan_json": "{}"}

Plan format:
{"type": "plan", "content": null, "tool_name": null, "arguments_json": "{}", "plan_json": "{\\\"summary\\\":\\\"...\\\",\\\"steps\\\":[{\\\"id\\\":\\\"1\\\",\\\"title\\\":\\\"...\\\",\\\"description\\\":\\\"...\\\",\\\"status\\\":\\\"pending\\\"}]}"}

Tool call format:
{"type": "tool_call", "content": null, "tool_name": "tool_name", "arguments_json": "{\\\"arg\\\":\\\"value\\\"}", "plan_json": "{}"}

Use a final_answer when you can answer without a tool. Use a tool_call when you need a listed tool.
If you intend to use a tool, the type must be tool_call; never put tool_name or arguments_json on a final_answer.
If you intend to answer the user, the type must be final_answer, tool_name must be null, and arguments_json must be "{}".
For tool calls, arguments_json must be a JSON-encoded object string containing the tool arguments.
For tool calls, use only argument fields from the listed tool schema.
Use a plan only when the Planning section explicitly tells you to propose a plan.
For plans, plan_json must be a JSON-encoded object string with summary and steps.
After an observation is provided, use it to produce the next tool_call or final_answer.
Some tool observations may contain bounded head/tail previews plus local artifact paths for full output.
If the omitted middle may contain information needed to answer correctly, inspect the artifact or run a narrower follow-up tool call before giving a final_answer.
"""

JSON_REPAIR_PROMPT = """Your previous response could not be parsed as ChulkHarness action JSON.
Return exactly one valid JSON object using one of these shapes:
{"type": "final_answer", "content": "...", "tool_name": null, "arguments_json": "{}", "plan_json": "{}"}
{"type": "plan", "content": null, "tool_name": null, "arguments_json": "{}", "plan_json": "{\\\"summary\\\":\\\"...\\\",\\\"steps\\\":[{\\\"id\\\":\\\"1\\\",\\\"title\\\":\\\"...\\\",\\\"description\\\":\\\"...\\\",\\\"status\\\":\\\"pending\\\"}]}"}
{"type": "tool_call", "content": null, "tool_name": "tool_name", "arguments_json": "{\\\"arg\\\":\\\"value\\\"}", "plan_json": "{}"}
Do not include Markdown fences, comments, or prose outside the JSON object.
If your previous response included tool_name or tool arguments, return a tool_call action.
If your previous response was a direct answer, return a final_answer action with tool_name null and arguments_json "{}".
If an observation reported invalid_arguments, remove unsupported fields and use only the listed schema fields.
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
                "The approval plan must be an implementation plan. Do not make read/list/search/explore/inspect steps the plan.",
                "Because the user explicitly requested /plan, do not answer directly. Return a plan action after any needed reconnaissance.",
            ]
        )

    if active_plan is not None and plan_approved:
        return "\n".join(
            [
                "Planning: approved for this turn.",
                "Follow the approved plan while executing this turn.",
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


def _truncate_skill_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "..."
