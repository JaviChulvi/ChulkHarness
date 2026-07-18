"""Tests for final-answer reflection parsing."""

import json

from chulk.core.reflection import (
    ReflectionParseError,
    build_reflection_messages,
    parse_reflection_response,
)
from chulk.core.state import MAX_PLAN_PROMPT_EVIDENCE_CHARS, Plan, PlanStep, TurnState


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


def test_reflection_references_plan_evidence_without_copying_its_content() -> None:
    evidence = "RAW_PLAN_EVIDENCE_MARKER " + ("x" * 4000)
    step = PlanStep(id="1", title="Implement change", description="Update the behavior.")
    step.add_evidence(evidence, tool_name="lookup", tool_call_iteration=2)
    step.mark("in_progress")
    plan = Plan(summary="Make the requested change.", steps=[step])
    plan.approve()
    turn = TurnState(
        user_message="Please make the change.",
        active_plan=plan,
        plan_approved=True,
    )

    messages = build_reflection_messages(turn, "The change is complete.")
    reflection_request = messages[-1]["content"]

    assert "RAW_PLAN_EVIDENCE_MARKER" not in reflection_request
    assert "evidence_records=1" in reflection_request
    assert "evidence_sources=lookup" in reflection_request
    assert "Make the requested change." in reflection_request


def test_reflection_includes_bounded_step_description_and_acceptance_criteria() -> None:
    description = "DESCRIPTION_MARKER " + ("d" * 900)
    criterion = "CRITERION_MARKER " + ("c" * 700)
    step = PlanStep(
        id="1",
        title="Generic work",
        description=description,
        acceptance_criteria=[criterion],
    )
    plan = Plan(summary="Complete the work.", steps=[step])
    plan.approve()
    turn = TurnState(
        user_message="Please complete it.",
        active_plan=plan,
        plan_approved=True,
    )

    reflection_request = build_reflection_messages(turn, "Done.")[-1]["content"]

    assert "DESCRIPTION_MARKER" in reflection_request
    assert "CRITERION_MARKER" in reflection_request
    assert description not in reflection_request
    assert criterion not in reflection_request


def test_plan_prompt_bounds_latest_evidence_preview() -> None:
    evidence = "EVIDENCE_PREVIEW_MARKER " + ("x" * 4000)
    step = PlanStep(id="1", title="Implement change", description="Update the behavior.")
    step.add_evidence(evidence, tool_name="lookup", tool_call_iteration=2)
    plan = Plan(summary="Make the requested change.", steps=[step])

    prompt = plan.to_prompt()
    evidence_line = next(line for line in prompt.splitlines() if line.startswith("  Evidence:"))
    preview = evidence_line.rsplit(": ", maxsplit=1)[-1]

    assert "Evidence: 1 record(s); latest from lookup call #2:" in evidence_line
    assert "EVIDENCE_PREVIEW_MARKER" in preview
    assert len(preview) <= MAX_PLAN_PROMPT_EVIDENCE_CHARS
    assert evidence not in prompt
    assert preview.endswith("... [truncated; see recorded observation]")
