"""Focused contracts for provider-native tool declaration helpers."""

from __future__ import annotations

from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import pytest

from chulk.llm.tools import (
    PLAN_STEP_UPDATE_TOOL_NAME,
    PLAN_TOOL_NAME,
    PlanningToolAvailability,
    chat_completion_tools,
    openai_response_tools,
    provider_action_tools,
    with_json_action_prompt,
)
from chulk.tools import Tool
from chulk.tools.registry import ToolResult


CALCULATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "Arithmetic expression to evaluate.",
        },
        "precision": {"type": "integer", "minimum": 0},
    },
    "required": ["expression"],
    "additionalProperties": False,
}


def _calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description="Evaluate an arithmetic expression safely.",
        args_schema=deepcopy(CALCULATOR_SCHEMA),
        callable=lambda _arguments: ToolResult("calculator", True, "unused"),
        requires_confirmation=True,
        permission_level="write",
        metadata={"private_marker": "must-not-enter-the-prompt"},
    )


@pytest.mark.parametrize("include_regular_tool", [False, True])
@pytest.mark.parametrize(
    ("planning_tools", "expected_planning_names"),
    [
        (PlanningToolAvailability(), []),
        (PlanningToolAvailability(propose_plan=True), [PLAN_TOOL_NAME]),
        (
            PlanningToolAvailability(update_plan_step=True),
            [PLAN_STEP_UPDATE_TOOL_NAME],
        ),
        (
            PlanningToolAvailability(propose_plan=True, update_plan_step=True),
            [PLAN_TOOL_NAME, PLAN_STEP_UPDATE_TOOL_NAME],
        ),
    ],
)
def test_provider_action_tools_respects_exact_planning_availability_matrix(
    include_regular_tool: bool,
    planning_tools: PlanningToolAvailability,
    expected_planning_names: list[str],
) -> None:
    tools = [_calculator_tool()] if include_regular_tool else []

    declarations = provider_action_tools(tools, planning_tools=planning_tools)

    expected_names = (["calculator"] if include_regular_tool else []) + expected_planning_names
    assert [declaration["name"] for declaration in declarations] == expected_names


def test_explicit_empty_planning_availability_produces_no_declarations() -> None:
    assert provider_action_tools(
        [],
        planning_tools=PlanningToolAvailability(),
    ) == []


def test_default_planning_availability_adds_no_pseudo_tools() -> None:
    assert [
        declaration["name"]
        for declaration in provider_action_tools([_calculator_tool()])
    ] == ["calculator"]


def test_plan_declaration_defaults_new_steps_to_pending() -> None:
    declaration = provider_action_tools(
        [],
        planning_tools=PlanningToolAvailability(propose_plan=True),
    )[0]
    step_schema = declaration["parameters"]["properties"]["steps"]["items"]

    assert "status" not in step_schema["properties"]
    assert "status" not in step_schema["required"]


@pytest.mark.parametrize("name", [PLAN_TOOL_NAME, PLAN_STEP_UPDATE_TOOL_NAME])
def test_provider_action_tools_rejects_user_tools_with_internal_action_names(
    name: str,
) -> None:
    tool = Tool(
        name=name,
        description="Must not shadow an internal action.",
        args_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        callable=lambda _arguments: ToolResult(name, True, "unused"),
    )

    with pytest.raises(ValueError, match="reserved for internal Chulk actions"):
        provider_action_tools([tool])


def test_provider_action_tools_deep_copies_source_schema() -> None:
    tool = _calculator_tool()
    original_schema = deepcopy(tool.args_schema)

    declarations = provider_action_tools(
        [tool],
        planning_tools=PlanningToolAvailability(),
    )

    assert declarations == [
        {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression safely.",
            "parameters": original_schema,
        }
    ]
    declarations[0]["parameters"]["properties"]["expression"]["description"] = "mutated"
    assert tool.args_schema == original_schema


def test_openai_response_tools_preserves_exact_function_shape() -> None:
    declarations = openai_response_tools(
        [_calculator_tool()],
        planning_tools=PlanningToolAvailability(),
    )

    assert declarations == [
        {
            "type": "function",
            "name": "calculator",
            "description": "Evaluate an arithmetic expression safely.",
            "parameters": CALCULATOR_SCHEMA,
        }
    ]


def test_chat_completion_tools_preserves_exact_function_shape() -> None:
    declarations = chat_completion_tools(
        [_calculator_tool()],
        planning_tools=PlanningToolAvailability(),
    )

    assert declarations == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression safely.",
                "parameters": CALCULATOR_SCHEMA,
            },
        }
    ]


def test_json_fallback_replaces_native_chulk_sections_with_one_safe_catalog() -> None:
    tool = _calculator_tool()
    messages = [
        {
            "role": "system",
            "content": """<chulk_prompt>
<system_prompt>
<system_instructions>Keep the original instructions.</system_instructions>
</system_prompt>
<tools>
<available_tools>
<status>Definitions are supplied through the provider-native transport.</status>
</available_tools>
</tools>
<action_protocol>
<response_protocol>
<transport>provider_native_tool_calling</transport>
<primary_rule>Use the provider-native tool-calling interface for actions.</primary_rule>
</response_protocol>
</action_protocol>
</chulk_prompt>""",
        },
        {"role": "user", "content": "Calculate 2 + 2."},
    ]
    original_messages = deepcopy(messages)

    fallback = with_json_action_prompt(messages, tools=[tool])
    repeated = with_json_action_prompt(fallback, tools=[tool])

    assert messages == original_messages
    assert repeated == fallback
    assert fallback[1:] == original_messages[1:]
    system_content = fallback[0]["content"]
    assert system_content.count("<response_protocol>") == 1
    assert "provider_native_tool_calling" not in system_content
    assert "provider-native tool-calling interface" not in system_content
    assert "must-not-enter-the-prompt" not in system_content

    root = ET.fromstring(system_content)
    assert root.findtext("system_prompt/system_instructions") == "Keep the original instructions."
    assert root.findtext("action_protocol/response_protocol/transport") == "json_object"
    assert root.findtext("tools/available_tools/tool/name") == "calculator"
    assert (
        root.findtext("tools/available_tools/tool/description")
        == "Evaluate an arithmetic expression safely."
    )
    assert root.findtext("tools/available_tools/tool/requires_confirmation") == "True"
    assert root.findtext("tools/available_tools/tool/permission_level") == "write"
    schema_text = root.findtext("tools/available_tools/tool/arguments_schema_json")
    assert schema_text is not None
    assert json.loads(schema_text) == CALCULATOR_SCHEMA


def test_json_fallback_augments_arbitrary_system_message_idempotently() -> None:
    messages = [
        {"role": "system", "content": "Preserve this arbitrary system instruction."},
        {"role": "user", "content": "hello"},
    ]
    original_messages = deepcopy(messages)

    fallback = with_json_action_prompt(messages, tools=[_calculator_tool()])

    assert messages == original_messages
    assert with_json_action_prompt(fallback, tools=[_calculator_tool()]) == fallback
    assert fallback[0]["content"].startswith("Preserve this arbitrary system instruction.")
    assert fallback[0]["content"].count("<response_protocol>") == 1
    assert fallback[0]["content"].count("<tools>") == 1
    assert "<name>calculator</name>" in fallback[0]["content"]
    assert fallback[1:] == original_messages[1:]


def test_json_fallback_preserves_arbitrary_xml_named_like_chulk_sections() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "Custom instructions.\n"
                "<tools>Do not replace this custom text.</tools>\n"
                "<action_protocol>Keep this custom protocol.</action_protocol>"
            ),
        },
        {"role": "user", "content": "hello"},
    ]

    fallback = with_json_action_prompt(messages, tools=[_calculator_tool()])
    system_content = fallback[0]["content"]

    assert "<tools>Do not replace this custom text.</tools>" in system_content
    assert "<action_protocol>Keep this custom protocol.</action_protocol>" in system_content
    assert system_content.count("<chulk_action_fallback>") == 1
    assert system_content.count("<response_protocol>") == 1
    assert with_json_action_prompt(fallback, tools=[_calculator_tool()]) == fallback


def test_json_fallback_prepends_system_message_to_non_system_messages() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "continue"},
    ]
    original_messages = deepcopy(messages)

    fallback = with_json_action_prompt(messages, tools=[_calculator_tool()])

    assert messages == original_messages
    assert fallback[0]["role"] == "system"
    assert fallback[0]["content"].count("<response_protocol>") == 1
    assert fallback[0]["content"].count("<tools>") == 1
    assert "<name>calculator</name>" in fallback[0]["content"]
    assert fallback[1:] == original_messages
    assert with_json_action_prompt(fallback, tools=[_calculator_tool()]) == fallback
