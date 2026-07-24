"""Tool observation formatting for model feedback."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from chulk.tools.output import TextPreview, preview_text
from chulk.tools.registry import ToolResult


ArtifactWriter = Callable[[str, str], dict[str, Any] | None]
MAX_TOOL_ACTION_CONTEXT_CHARS = 2000


def format_tool_action_context(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    phase: str,
    iteration: int,
    plan_step_id: str | None,
    max_chars: int = MAX_TOOL_ACTION_CONTEXT_CHARS,
) -> tuple[str, dict[str, Any]]:
    """Format a bounded, provider-neutral record of an executed tool action."""
    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero")

    arguments_json = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    argument_limit = max(1, min(len(arguments_json), max_chars))
    context = ""
    argument_preview = preview_text(arguments_json, argument_limit)
    while True:
        argument_preview = preview_text(arguments_json, argument_limit)
        payload = {
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments_json": argument_preview.text,
            "arguments_truncated": argument_preview.truncated,
            "phase": phase,
            "iteration": iteration,
            "plan_step_id": plan_step_id,
        }
        context = (
            "<executed_tool_action>\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n</executed_tool_action>"
        )
        if len(context) <= max_chars or argument_limit == 1:
            break
        overflow = len(context) - max_chars
        argument_limit = max(1, argument_limit - overflow - 1)

    context_preview = preview_text(context, max_chars)
    return context_preview.text, {
        "arguments": argument_preview.to_metadata(),
        "context": context_preview.to_metadata(),
    }


def format_tool_observation(
    *,
    requested_tool_name: str,
    result: ToolResult,
    max_observation_chars: int,
    max_stdout_chars: int,
    max_stderr_chars: int,
    artifact_writer: ArtifactWriter,
) -> tuple[str, dict[str, Any]]:
    """Format one tool result as a bounded model observation plus metadata."""
    status = "success" if result.success else "error"
    parts = [f"Tool {result.tool_name} finished with {status}.", result.observation]
    metadata: dict[str, Any] = {
        "requested_tool_name": requested_tool_name,
        "tool_name": result.tool_name,
        "success": result.success,
        "stdout": None,
        "stderr": None,
        "observation": None,
        "artifacts": [],
    }

    if result.stdout:
        stdout_preview = preview_text(result.stdout, max_stdout_chars)
        metadata["stdout"] = stdout_preview.to_metadata()
        parts.append("stdout:\n" + stdout_preview.text)
        _append_artifact_note(parts, metadata, artifact_writer, result.tool_name, "stdout", result.stdout, stdout_preview)

    if result.stderr:
        stderr_preview = preview_text(result.stderr, max_stderr_chars)
        metadata["stderr"] = stderr_preview.to_metadata()
        parts.append("stderr:\n" + stderr_preview.text)
        _append_artifact_note(parts, metadata, artifact_writer, result.tool_name, "stderr", result.stderr, stderr_preview)

    if result.exit_code is not None:
        parts.append(f"exit_code: {result.exit_code}")
    if result.error:
        parts.append(f"error: {result.error}")

    full_observation = "\n".join(parts)
    observation_preview = preview_text(full_observation, max_observation_chars)
    metadata["observation"] = observation_preview.to_metadata()
    if observation_preview.truncated:
        artifact = artifact_writer(f"{result.tool_name}-observation", full_observation)
        if artifact is not None:
            metadata["artifacts"].append({"field": "observation", **artifact})
            artifact_note = _artifact_note("observation", artifact)
            final_observation = _with_required_suffix(
                full_observation,
                suffix=artifact_note,
                max_chars=max_observation_chars,
            )
        else:
            final_observation = observation_preview.text
    else:
        final_observation = observation_preview.text

    return final_observation, metadata


def _append_artifact_note(
    parts: list[str],
    metadata: dict[str, Any],
    artifact_writer: ArtifactWriter,
    tool_name: str,
    field: str,
    content: str,
    preview: TextPreview,
) -> None:
    if not preview.truncated:
        return
    artifact = artifact_writer(f"{tool_name}-{field}", content)
    if artifact is None:
        parts.append(f"[full {field} omitted from model context; no artifact writer configured]")
        return
    metadata["artifacts"].append({"field": field, **artifact})
    parts.append(_artifact_note(field, artifact))


def _artifact_note(field: str, artifact: dict[str, Any]) -> str:
    return (
        f"[full {field} saved as trace artifact {artifact['artifact_id']}; "
        f"chars={artifact['char_count']}; sha256={artifact['sha256']}]"
    )


def _with_required_suffix(text: str, *, suffix: str, max_chars: int) -> str:
    separator = "\n"
    suffix_block = separator + suffix
    if len(suffix_block) >= max_chars:
        return suffix_block[-max_chars:]
    preview = preview_text(text, max_chars - len(suffix_block))
    return preview.text + suffix_block
