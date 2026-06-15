"""Tests for final-answer reflection parsing."""

import json

from chulk.core.reflection import ReflectionParseError, parse_reflection_response


def test_parse_reflection_response_accepts_json_fence():
    result = parse_reflection_response(
        "```json\n"
        + json.dumps({"approved": True, "reason": "The answer is grounded.", "feedback": None})
        + "\n```"
    )

    assert result.approved is True
    assert result.reason == "The answer is grounded."
    assert result.feedback is None


def test_parse_reflection_response_requires_feedback_when_rejected():
    try:
        parse_reflection_response({"approved": False, "reason": "Missing evidence.", "feedback": ""})
    except ReflectionParseError as exc:
        assert "feedback" in str(exc)
    else:
        raise AssertionError("Expected rejected reflection without feedback to fail")
