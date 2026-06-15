"""Tests for model action parsing."""

import json

from chulk.core.actions import ActionParseError, FinalAnswerAction, PlanAction, ToolCallAction, parse_model_response


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
