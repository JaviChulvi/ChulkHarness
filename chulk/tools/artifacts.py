"""Capability-gated bounded reads for trace output artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from typing import Any

from chulk.hosting.async_utils import call_async_service
from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.registry import Tool, ToolResult
from chulk.tracing.artifacts import (
    DEFAULT_ARTIFACT_READ_BYTES,
    MAX_ARTIFACT_READ_BYTES,
    TraceArtifactStore,
)


def read_trace_artifact_tool(artifact_store: TraceArtifactStore) -> Tool:
    """Return the opt-in tool bound to one conversation's artifact store."""
    return Tool(
        name="read_trace_artifact",
        description=(
            "Read a bounded, integrity-checked view of a trace artifact id "
            "recorded for this conversation."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "pattern": "^art_[0-9a-f]{32}$",
                },
                "mode": {
                    "type": "string",
                    "enum": ["slice", "head", "tail", "head_tail"],
                    "default": "head_tail",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ARTIFACT_READ_BYTES,
                    "default": DEFAULT_ARTIFACT_READ_BYTES,
                },
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
        callable=lambda arguments: read_trace_artifact(arguments, artifact_store),
        run_in_executor=True,
        permission_level=ToolPermissionLevel.READ,
    )


def async_read_trace_artifact_tool(artifact_store: object) -> Tool:
    """Return the artifact tool bound to a native async hosted store."""

    async def invoke(arguments: dict[str, Any]) -> ToolResult:
        result = await call_async_service(
            artifact_store,
            "read",
            arguments["artifact_id"],
            mode=arguments.get("mode", "head_tail"),
            offset=arguments.get("offset", 0),
            max_bytes=arguments.get(
                "max_bytes",
                DEFAULT_ARTIFACT_READ_BYTES,
            ),
        )
        return _artifact_read_result(result)

    return replace(
        read_trace_artifact_tool(artifact_store),  # type: ignore[arg-type]
        callable=invoke,
        run_in_executor=False,
    )


def read_trace_artifact(
    arguments: dict[str, Any],
    artifact_store: TraceArtifactStore,
) -> ToolResult:
    result = artifact_store.read(
        arguments["artifact_id"],
        mode=arguments.get("mode", "head_tail"),
        offset=arguments.get("offset", 0),
        max_bytes=arguments.get("max_bytes", DEFAULT_ARTIFACT_READ_BYTES),
    )
    return _artifact_read_result(result)


def _artifact_read_result(result: Any) -> ToolResult:
    if isinstance(result, Mapping):
        payload = dict(result)
    else:
        to_dict = getattr(result, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("artifact store returned an unsupported read")
        payload = dict(to_dict())
    return ToolResult(
        "read_trace_artifact",
        True,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        metadata={
            "artifact_id": payload.get("artifact_id"),
            "byte_count": payload.get("byte_count"),
            "total_byte_count": payload.get("total_byte_count"),
            "truncated": payload.get("truncated"),
        },
        value=payload,
    )


__all__ = [
    "async_read_trace_artifact_tool",
    "read_trace_artifact",
    "read_trace_artifact_tool",
]
