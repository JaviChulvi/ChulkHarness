"""Tests for model action parsing."""

import json

from chulk.core.actions import (
    STRICT_AGENT_ACTION_JSON_SCHEMA,
    ActionParseError,
    FinalAnswerAction,
    PlanAction,
    PlanStepUpdateAction,
    ToolCallAction,
    parse_model_response,
)


def test_parse_final_answer():
    action = parse_model_response(json.dumps({"type": "final_answer", "content": "hello"}))

    assert action == FinalAnswerAction(type="final_answer", content="hello")


def test_parse_final_answer_rejects_embedded_tool_call_fields():
    raw_response = json.dumps(
        {
            "type": "final_answer",
            "content": "I will search the files.",
            "tool_name": "search_files",
            "arguments_json": '{"query":"agent loop"}',
        }
    )

    try:
        parse_model_response(raw_response)
    except ActionParseError as exc:
        assert "tool call fields" in str(exc)
    else:
        raise AssertionError("Expected final_answer with tool fields to fail")


def test_parse_tool_call():
    action = parse_model_response(
        json.dumps({"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1 + 1"}})
    )

    assert action == ToolCallAction(type="tool_call", tool_name="calculator", arguments={"expression": "1 + 1"})


def test_parse_tool_call_with_arguments_json_transport():
    action = parse_model_response(
        json.dumps(
            {
                "type": "tool_call",
                "content": None,
                "tool_name": "calculator",
                "arguments_json": '{"expression":"1 + 1"}',
            }
        )
    )

    assert action == ToolCallAction(type="tool_call", tool_name="calculator", arguments={"expression": "1 + 1"})


def test_parse_plan_action_with_plan_json_transport():
    plan_json = json.dumps(
        {
            "summary": "Inspect the project and report back.",
            "steps": [
                {
                    "id": "1",
                    "title": "List files",
                    "description": "Use the file listing tool to inspect the project shape.",
                    "status": "pending",
                }
            ],
        }
    )

    action = parse_model_response(
        json.dumps(
            {
                "type": "plan",
                "content": None,
                "tool_name": None,
                "arguments_json": "{}",
                "plan_json": plan_json,
            }
        )
    )

    assert isinstance(action, PlanAction)
    assert action.plan.summary == "Inspect the project and report back."
    assert action.plan.steps[0].title == "List files"
    assert action.plan.steps[0].status == "pending"
    assert action.plan.steps[0].acceptance_criteria == ["Use the file listing tool to inspect the project shape."]


def test_parse_rich_plan_step_fields():
    plan_json = json.dumps(
        {
            "summary": "Implement a feature in two steps.",
            "steps": [
                {
                    "id": "1",
                    "title": "Add state",
                    "description": "Extend the plan state.",
                    "status": "pending",
                    "depends_on": [],
                    "acceptance_criteria": ["State serializes dependencies.", "State serializes evidence."],
                    "retry_limit": 2,
                },
                {
                    "id": "2",
                    "title": "Use state",
                    "description": "Wire the agent loop.",
                    "status": "pending",
                    "depends_on": ["1"],
                    "acceptance_criteria": ["Agent advances after step 1."],
                    "retry_limit": 0,
                },
            ],
        }
    )

    action = parse_model_response(
        json.dumps(
            {
                "type": "plan",
                "content": None,
                "tool_name": None,
                "arguments_json": "{}",
                "plan_json": plan_json,
            }
        )
    )

    assert isinstance(action, PlanAction)
    assert action.plan.steps[0].acceptance_criteria == ["State serializes dependencies.", "State serializes evidence."]
    assert action.plan.steps[0].retry_limit == 0
    assert action.plan.steps[1].depends_on == ["1"]


def test_parse_plan_action_rejects_later_dependency():
    plan_json = json.dumps(
        {
            "summary": "Invalid dependency.",
            "steps": [
                {
                    "id": "1",
                    "title": "First",
                    "description": "Cannot depend on the future.",
                    "depends_on": ["2"],
                },
                {
                    "id": "2",
                    "title": "Second",
                    "description": "Later step.",
                },
            ],
        }
    )

    try:
        parse_model_response(
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": plan_json,
                }
            )
        )
    except ActionParseError as exc:
        assert "depends on unknown or later step" in str(exc)
    else:
        raise AssertionError("Expected invalid dependency to fail")


def test_parse_plan_step_update_action():
    action = parse_model_response(
        json.dumps(
            {
                "type": "plan_step_update",
                "content": None,
                "tool_name": None,
                "arguments_json": "{}",
                "plan_json": "{}",
                "step_update_json": json.dumps(
                    {
                        "step_id": "1",
                        "status": "completed",
                        "evidence": "The calculator returned 4.",
                        "reason": None,
                    }
                ),
            }
        )
    )

    assert action == PlanStepUpdateAction(
        type="plan_step_update",
        step_id="1",
        status="completed",
        evidence="The calculator returned 4.",
        reason=None,
    )


def test_parse_plan_step_update_rejects_blocked_without_reason():
    try:
        parse_model_response(
            json.dumps(
                {
                    "type": "plan_step_update",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": "{}",
                    "step_update_json": json.dumps(
                        {
                            "step_id": "1",
                            "status": "blocked",
                            "evidence": "The tool cannot run.",
                            "reason": None,
                        }
                    ),
                }
            )
        )
    except ActionParseError as exc:
        assert "reason" in str(exc)
    else:
        raise AssertionError("Expected blocked step update without reason to fail")


def test_strict_schema_includes_plan_step_update_transport():
    schema = STRICT_AGENT_ACTION_JSON_SCHEMA

    assert "plan_step_update" in schema["properties"]["type"]["enum"]
    assert "step_update_json" in schema["properties"]
    assert "step_update_json" in schema["required"]


def test_parse_json_markdown_block():
    action = parse_model_response('```json\n{"type": "final_answer", "content": "hello"}\n```')

    assert isinstance(action, FinalAnswerAction)


def test_parse_invalid_json_raises():
    try:
        parse_model_response("not json")
    except ActionParseError as exc:
        assert "not valid JSON" in str(exc)
    else:
        raise AssertionError("Expected invalid JSON to fail")


def test_parse_invalid_tool_call_arguments_raises():
    try:
        parse_model_response(json.dumps({"type": "tool_call", "tool_name": "calculator", "arguments": []}))
    except ActionParseError as exc:
        assert "arguments" in str(exc)
    else:
        raise AssertionError("Expected invalid arguments to fail")


def test_parse_plan_action_rejects_invalid_step_status():
    plan_json = json.dumps(
        {
            "summary": "Invalid plan.",
            "steps": [
                {
                    "id": "1",
                    "title": "Bad step",
                    "description": "This step has an invalid status.",
                    "status": "done",
                }
            ],
        }
    )

    try:
        parse_model_response(
            json.dumps(
                {
                    "type": "plan",
                    "content": None,
                    "tool_name": None,
                    "arguments_json": "{}",
                    "plan_json": plan_json,
                }
            )
        )
    except ActionParseError as exc:
        assert "plan step status" in str(exc)
    else:
        raise AssertionError("Expected invalid plan status to fail")
