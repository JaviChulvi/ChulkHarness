"""Capability-gated bounded reads for trace output artifacts."""

from __future__ import annotations

import json
from typing import Any

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
    payload = result.to_dict()
    return ToolResult(
        "read_trace_artifact",
        True,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        metadata={
            "artifact_id": result.artifact_id,
            "byte_count": result.byte_count,
            "total_byte_count": result.total_byte_count,
            "truncated": result.truncated,
        },
        value=payload,
    )


__all__ = ["read_trace_artifact", "read_trace_artifact_tool"]
