"""Tests for prompt context accounting and budgets."""

import json

from chulk.core.context import ContextBudget, TurnContextSection, estimate_tokens
from chulk.core.prompt_builder import build_agent_prompt
from chulk.memory import ConversationMemory
from chulk.tools import ToolRegistry, calculator_tool


def test_estimate_tokens_is_deterministic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_build_agent_prompt_reports_named_sections():
    memory = ConversationMemory()
    memory.add_user_message("hello")
    registry = ToolRegistry()
    registry.register(calculator_tool())

    prompt = build_agent_prompt(
        system_prompt="Base prompt.",
        memory=memory,
        profile_memories=[],
        relevant_memories=[],
        selected_skills=[],
        tool_registry=registry,
        max_skill_content_chars=1000,
        max_tool_calls_per_turn=3,
    )
    report = prompt.context_report.to_dict()
    section_names = [section["name"] for section in report["sections"]]

    assert prompt.messages[-1] == {"role": "user", "content": "hello"}
    assert report["estimated_tokens"] > 0
    assert report["included_message_count"] == 1
    assert report["omitted_message_count"] == 0
    assert "system_prompt" in section_names
    assert "memories" in section_names
    assert "skills" in section_names
    assert "tools" in section_names
    assert "history" in section_names
    assert "observations" in section_names


def test_build_agent_prompt_injects_external_context_and_prompt_metadata():
    memory = ConversationMemory()
    memory.add_user_message("answer from sources")

    prompt = build_agent_prompt(
        system_prompt="Base prompt.",
        memory=memory,
        profile_memories=[],
        relevant_memories=[],
        selected_skills=[],
        tool_registry=ToolRegistry(),
        max_skill_content_chars=1000,
        max_tool_calls_per_turn=3,
        context_sections=[
            TurnContextSection(
                id="src-1",
                title="Handbook",
                source="drive://handbook",
                content="The handbook says onboarding takes three days.",
            )
        ],
        prompt_profile="polp-search",
        locale="es-ES",
    )
    system_prompt = prompt.messages[0]["content"]
    report = prompt.context_report.to_dict()
    external = next(section for section in report["sections"] if section["name"] == "external_context")
    metadata = next(section for section in report["sections"] if section["name"] == "prompt_metadata")

    assert "External turn context supplied by the host application" in system_prompt
    assert "The handbook says onboarding takes three days." in system_prompt
    assert "profile: polp-search" in system_prompt
    assert "locale: es-ES" in system_prompt
    assert external["metadata"]["context_section_ids"] == ["src-1"]
    assert metadata["metadata"]["prompt_profile"] == "polp-search"
    assert metadata["metadata"]["locale"] == "es-ES"


def test_build_agent_prompt_injects_conversation_summary_section():
    memory = ConversationMemory()
    memory.replace(
        [{"role": "user", "content": "latest question"}],
        conversation_summary="Earlier work chose prompt compaction and ruled out long-term memory.",
        summary_message_count=4,
    )

    prompt = build_agent_prompt(
        system_prompt="Base prompt.",
        memory=memory,
        profile_memories=[],
        relevant_memories=[],
        selected_skills=[],
        tool_registry=ToolRegistry(),
        max_skill_content_chars=1000,
        max_tool_calls_per_turn=3,
    )
    system_prompt = prompt.messages[0]["content"]
    report = prompt.context_report.to_dict()
    summary_section = next(section for section in report["sections"] if section["name"] == "conversation_summary")

    assert "Conversation summary from earlier turns" in system_prompt
    assert "prompt compaction" in system_prompt
    assert summary_section["metadata"]["has_summary"] is True
    assert summary_section["metadata"]["summary_message_count"] == 4


def test_context_budget_trims_old_observations_and_keeps_latest_user():
    memory = ConversationMemory(max_messages=10)
    memory.add_user_message("older question")
    memory.add_observation("large old observation " + ("x" * 5000))
    memory.add_assistant_message("older answer")
    memory.add_user_message("latest question")

    prompt = build_agent_prompt(
        system_prompt="Base prompt.",
        memory=memory,
        profile_memories=[],
        relevant_memories=[],
        selected_skills=[],
        tool_registry=ToolRegistry(),
        max_skill_content_chars=1000,
        max_tool_calls_per_turn=3,
        context_budget=ContextBudget(max_prompt_tokens=950, response_reserve_tokens=0),
    )
    payload = json.dumps(prompt.messages)
    report = prompt.context_report.to_dict()

    assert "latest question" in payload
    assert "large old observation" not in payload
    assert report["trimmed"] is True
    assert report["omitted_message_count"] >= 1
    assert report["omitted_observation_count"] == 1


def test_context_budget_trims_complete_old_history_blocks():
    memory = ConversationMemory(max_messages=10)
    memory.add_user_message("older question " + ("a" * 1000))
    memory.add_assistant_message("older answer " + ("b" * 1000))
    memory.add_user_message("newer question " + ("c" * 1000))
    memory.add_assistant_message("newer answer " + ("d" * 1000))
    memory.add_user_message("latest question")

    prompt = build_agent_prompt(
        system_prompt="Base prompt.",
        memory=memory,
        profile_memories=[],
        relevant_memories=[],
        selected_skills=[],
        tool_registry=ToolRegistry(),
        max_skill_content_chars=1000,
        max_tool_calls_per_turn=3,
        context_budget=ContextBudget(max_prompt_tokens=1200, response_reserve_tokens=0),
    )
    payload = json.dumps(prompt.messages)
    report = prompt.context_report.to_dict()

    assert "latest question" in payload
    assert "older question" not in payload
    assert "older answer" not in payload
    assert report["omitted_message_count"] >= 2
    assert report["section_estimated_tokens"] > 0
