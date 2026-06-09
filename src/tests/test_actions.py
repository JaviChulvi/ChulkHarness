"""Tests for model action parsing."""

import json

from src.core.actions import ActionParseError, FinalAnswerAction, ToolCallAction, parse_model_response


def test_parse_final_answer():
    action = parse_model_response(json.dumps({"type": "final_answer", "content": "hello"}))

    assert action == FinalAnswerAction(type="final_answer", content="hello")


def test_parse_tool_call():
    action = parse_model_response(
        json.dumps({"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": "1 + 1"}})
    )

    assert action == ToolCallAction(type="tool_call", tool_name="calculator", arguments={"expression": "1 + 1"})


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
