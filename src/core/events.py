"""Trace event names shared by the agent and CLI."""

from __future__ import annotations


class TraceEvent:
    """String constants for trace and progress events."""

    TURN_STARTED = "turn_started"
    USER_MESSAGE = "user_message"
    MEMORY_EXTRACTION_COMPLETED = "memory_extraction_completed"
    MEMORY_SEARCH_STARTED = "memory_search_started"
    MEMORY_SEARCH_COMPLETED = "memory_search_completed"
    SKILL_SELECTION_STARTED = "skill_selection_started"
    SKILL_SELECTION_COMPLETED = "skill_selection_completed"
    MODEL_REQUEST_STARTED = "model_request_started"
    MODEL_RESPONSE = "model_response"
    PARSED_ACTION = "parsed_action"
    MODEL_RESPONSE_PARSED = "model_response_parsed"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL = "tool_call"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    TOOL_OBSERVATION = "tool_observation"
    PLAN_CREATED = "plan_created"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"
    PLAN_REVISION_REQUESTED = "plan_revision_requested"
    PLAN_STEP_STARTED = "plan_step_started"
    PLAN_STEP_COMPLETED = "plan_step_completed"
    PLAN_STEP_BLOCKED = "plan_step_blocked"
    FINAL_ANSWER = "final_answer"
    TURN_FAILED = "turn_failed"
    TURN_FINISHED = "turn_finished"
