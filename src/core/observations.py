"""Tool observation formatting for model feedback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.tools.output import TextPreview, preview_text
from src.tools.registry import ToolResult


ArtifactWriter = Callable[[str, str], dict[str, Any] | None]


def format_tool_observation(
    *,
    requested_tool_name: str,
    result: ToolResult,
    max_observation_chars: int,
    max_stdout_chars: int,
    max_stderr_chars: int,
    artifact_writer: ArtifactWriter,
) -> tuple[str, dict]:
    """Format one tool result as a bounded model observation plus metadata."""
    status = "success" if result.success else "error"
    parts = [f"Tool {result.tool_name} finished with {status}.", result.observation]
    metadata = {
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
    metadata: dict,
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


def _artifact_note(field: str, artifact: dict) -> str:
    return (
        f"[full {field} saved to {artifact['path']}; "
        f"chars={artifact['char_count']}; sha256={artifact['sha256']}]"
    )


def _with_required_suffix(text: str, *, suffix: str, max_chars: int) -> str:
    separator = "\n"
    suffix_block = separator + suffix
    if len(suffix_block) >= max_chars:
        return suffix_block[-max_chars:]
    preview = preview_text(text, max_chars - len(suffix_block))
    return preview.text + suffix_block
