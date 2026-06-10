"""Prompt templates for the agent loop."""

from src.memory import MemoryRecord

MAX_MEMORY_PROMPT_CONTENT_CHARS = 500

BASE_SYSTEM_PROMPT = """You are ChulkHarness, a lightweight Python agent harness.

Answer the user's message directly and clearly. Use the recent conversation history for context.
When tools are available, call a tool only when it materially helps answer the user.
"""

JSON_ACTION_PROMPT = """You must respond with exactly one JSON object and no extra prose.

Direct answer format:
{"type": "final_answer", "content": "...", "tool_name": null, "arguments_json": "{}"}

Tool call format:
{"type": "tool_call", "content": null, "tool_name": "tool_name", "arguments_json": "{\\\"arg\\\":\\\"value\\\"}"}

Use a final_answer when you can answer without a tool. Use a tool_call when you need a listed tool.
For tool calls, arguments_json must be a JSON-encoded object string containing the tool arguments.
After an observation is provided, use it to produce the next tool_call or final_answer.
"""

JSON_REPAIR_PROMPT = """Your previous response could not be parsed as ChulkHarness action JSON.
Return exactly one valid JSON object using one of these shapes:
{"type": "final_answer", "content": "...", "tool_name": null, "arguments_json": "{}"}
{"type": "tool_call", "content": null, "tool_name": "tool_name", "arguments_json": "{\\\"arg\\\":\\\"value\\\"}"}
Do not include Markdown fences, comments, or prose outside the JSON object.
"""


def format_tools_for_prompt(tool_descriptions: str) -> str:
    """Format available tools for prompt injection."""
    if not tool_descriptions:
        return "Available tools: none."
    return f"Available tools:\n{tool_descriptions}"


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
