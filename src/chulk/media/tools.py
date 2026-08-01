"""Optional media tool factories with explicit processing and delivery policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from chulk.media.models import ContentTrust, MediaItem, MediaKind, RetentionPolicy
from chulk.media.processors import MediaProcessorRegistry, TransformKind
from chulk.media.store import ContentStore
from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.registry import Tool, ToolResult


@dataclass(frozen=True, slots=True)
class GeneratedMedia:
    data: bytes
    kind: MediaKind
    mime_type: str
    file_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class MediaGenerator(Protocol):
    name: str
    provider: str
    network_access: bool
    retains_data: bool

    def generate(self, prompt: str) -> GeneratedMedia: ...


MediaDelivery = Callable[[MediaItem, str], dict[str, object] | str]


def media_transform_tool(
    store: ContentStore,
    registry: MediaProcessorRegistry,
    *,
    max_output_chars: int = 12_000,
) -> Tool:
    """Create one read-only tool for transcription/extraction/understanding."""

    def transform(arguments: dict[str, Any]) -> ToolResult:
        item = store.get(arguments["content_ref"], profile_id=store.profile_id)
        requested = TransformKind(arguments["transform"])
        result = registry.transform(
            store,
            item,
            instruction=arguments["instruction"],
            preferred=(requested,),
        )
        text = result.text or ""
        bounded = text[:max_output_chars]
        return ToolResult(
            tool_name="media_transform",
            success=True,
            observation=bounded or "Media processor returned no text.",
            metadata={
                "content_ref": item.content_ref.id,
                "processor": result.processor,
                "transform": result.transform.value,
                "truncated": len(text) > len(bounded),
            },
            value={
                "content_ref": item.content_ref.id,
                "text": bounded,
                "processor": result.processor,
                "transform": result.transform.value,
            },
        )

    return Tool(
        name="media_transform",
        description=(
            "Transcribe audio/video, extract a document, or understand media "
            "already owned by this profile."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "content_ref": {"type": "string", "minLength": 1, "maxLength": 128},
                "instruction": {"type": "string", "minLength": 1, "maxLength": 4000},
                "transform": {
                    "type": "string",
                    "enum": [
                        TransformKind.TRANSCRIPTION.value,
                        TransformKind.EXTRACTION.value,
                        TransformKind.UNDERSTANDING.value,
                    ],
                },
            },
            "required": ["content_ref", "instruction", "transform"],
            "additionalProperties": False,
        },
        callable=transform,
        permission_level=ToolPermissionLevel.READ,
        timeout_seconds=120,
        run_in_executor=True,
    )


def media_generation_tool(
    store: ContentStore,
    generator: MediaGenerator,
    *,
    tool_name: str,
    transform: TransformKind,
) -> Tool:
    """Create an explicit image-generation or text-to-speech tool."""
    if transform not in {TransformKind.GENERATION, TransformKind.TEXT_TO_SPEECH}:
        raise ValueError("generation tools must generate images or speech")

    def generate(arguments: dict[str, Any]) -> ToolResult:
        generated = generator.generate(arguments["prompt"])
        item = store.put(
            generated.data,
            kind=generated.kind,
            mime_type=generated.mime_type,
            file_name=generated.file_name,
            provenance=f"{generator.provider}:{generator.name}",
            trust=ContentTrust.TRUSTED,
            retention=RetentionPolicy.SESSION,
            metadata={
                **generated.metadata,
                "generator": generator.name,
                "provider": generator.provider,
                "network_access": generator.network_access,
                "retains_data": generator.retains_data,
            },
        )
        return ToolResult(
            tool_name=tool_name,
            success=True,
            observation=(
                f"Generated {item.kind.value} content as {item.content_ref.id}. "
                "Delivery requires a separate approved media_delivery call."
            ),
            metadata={"content_ref": item.content_ref.id, "mime_type": item.mime_type},
            value=item.to_dict(),
        )

    return Tool(
        name=tool_name,
        description=(
            "Generate profile-owned media. This does not deliver the result to a channel."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 8000}
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        callable=generate,
        permission_level=(
            ToolPermissionLevel.NETWORK
            if generator.network_access
            else ToolPermissionLevel.WRITE
        ),
        requires_confirmation=generator.network_access,
        timeout_seconds=180,
        run_in_executor=True,
    )


def media_delivery_tool(
    store: ContentStore,
    deliver: MediaDelivery,
) -> Tool:
    """Create a distinct confirmation-gated channel delivery side effect."""

    def send(arguments: dict[str, Any]) -> ToolResult:
        item = store.get(arguments["content_ref"], profile_id=store.profile_id)
        receipt = deliver(item, arguments["target"])
        return ToolResult(
            tool_name="media_delivery",
            success=True,
            observation=f"Delivered {item.content_ref.id} to the approved target.",
            metadata={
                "content_ref": item.content_ref.id,
                "target": arguments["target"],
            },
            value=receipt,
        )

    return Tool(
        name="media_delivery",
        description="Deliver profile-owned media to a channel target after explicit approval.",
        args_schema={
            "type": "object",
            "properties": {
                "content_ref": {"type": "string", "minLength": 1, "maxLength": 128},
                "target": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "required": ["content_ref", "target"],
            "additionalProperties": False,
        },
        callable=send,
        permission_level=ToolPermissionLevel.EXTERNAL_SERVICE,
        requires_confirmation=True,
        timeout_seconds=120,
        run_in_executor=True,
        idempotent=False,
    )


__all__ = [
    "GeneratedMedia",
    "MediaDelivery",
    "MediaGenerator",
    "media_delivery_tool",
    "media_generation_tool",
    "media_transform_tool",
]
