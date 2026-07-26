"""Explicit media processor capabilities and transformation routing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum
from typing import Callable, Protocol

from chulk.media.models import MediaItem, MediaKind, UnsupportedMediaError
from chulk.media.store import ContentStore


class TransformKind(StrEnum):
    NATIVE = "native"
    TRANSCRIPTION = "transcription"
    EXTRACTION = "extraction"
    UNDERSTANDING = "understanding"
    GENERATION = "generation"
    TEXT_TO_SPEECH = "text_to_speech"
    DELIVERY = "delivery"


@dataclass(frozen=True, slots=True)
class ProcessorCapability:
    name: str
    transforms: frozenset[TransformKind]
    media_kinds: frozenset[MediaKind]
    mime_types: frozenset[str] = frozenset()
    max_bytes: int = 25 * 1024 * 1024
    provider: str = "local"
    network_access: bool = False
    retains_data: bool = False
    pricing_per_unit: Decimal | None = None
    unit_name: str = "request"

    def supports(self, item: MediaItem, transform: TransformKind) -> bool:
        return bool(
            transform in self.transforms
            and item.kind in self.media_kinds
            and item.byte_length <= self.max_bytes
            and (not self.mime_types or item.mime_type in self.mime_types)
        )


@dataclass(frozen=True, slots=True)
class MediaTransformResult:
    transform: TransformKind
    processor: str
    text: str | None = None
    media: MediaItem | None = None
    units: int = 1
    metadata: dict[str, object] = field(default_factory=dict)


class MediaProcessor(Protocol):
    capability: ProcessorCapability

    def process(
        self,
        item: MediaItem,
        data: bytes,
        *,
        instruction: str,
        transform: TransformKind,
    ) -> MediaTransformResult: ...


class MediaProcessorRegistry:
    """Select one declared processor without hidden provider fallback."""

    def __init__(self, processors: tuple[MediaProcessor, ...] = ()) -> None:
        self._processors: list[MediaProcessor] = list(processors)

    def register(self, processor: MediaProcessor) -> None:
        if any(item.capability.name == processor.capability.name for item in self._processors):
            raise ValueError(f"media processor already registered: {processor.capability.name}")
        self._processors.append(processor)

    def choose(
        self,
        item: MediaItem,
        *,
        preferred: tuple[TransformKind, ...] | None = None,
    ) -> tuple[MediaProcessor, TransformKind]:
        order = preferred or _default_transforms(item.kind)
        for transform in order:
            for processor in self._processors:
                if processor.capability.supports(item, transform):
                    return processor, transform
        raise UnsupportedMediaError(
            f"No configured media processor supports {item.kind.value} ({item.mime_type})."
        )

    def transform(
        self,
        store: ContentStore,
        item: MediaItem,
        *,
        instruction: str,
        preferred: tuple[TransformKind, ...] | None = None,
        selection: tuple[MediaProcessor, TransformKind] | None = None,
    ) -> MediaTransformResult:
        processor, transform = selection or self.choose(item, preferred=preferred)
        data = store.read(
            item.content_ref,
            profile_id=store.profile_id,
            max_bytes=processor.capability.max_bytes,
        )
        result = processor.process(
            item,
            data,
            instruction=instruction,
            transform=transform,
        )
        capability = processor.capability
        return replace(
            result,
            metadata={
                **result.metadata,
                "provider": capability.provider,
                "network_access": capability.network_access,
                "retains_data": capability.retains_data,
                "pricing_per_unit": (
                    str(capability.pricing_per_unit)
                    if capability.pricing_per_unit is not None
                    else None
                ),
                "unit_name": capability.unit_name,
            },
        )


class LocalTextExtractor:
    """Credential-free bounded UTF-8 document extraction."""

    capability = ProcessorCapability(
        name="local_text_extractor",
        transforms=frozenset({TransformKind.EXTRACTION}),
        media_kinds=frozenset({MediaKind.DOCUMENT}),
        mime_types=frozenset(
            {
                "application/json",
                "application/xml",
                "text/calendar",
                "text/csv",
                "text/html",
                "text/markdown",
                "text/plain",
                "text/rtf",
                "text/vcard",
                "text/xml",
            }
        ),
        max_bytes=2 * 1024 * 1024,
    )

    def process(
        self,
        item: MediaItem,
        data: bytes,
        *,
        instruction: str,
        transform: TransformKind,
    ) -> MediaTransformResult:
        del instruction
        if transform is not TransformKind.EXTRACTION:
            raise UnsupportedMediaError("local text extractor only performs extraction")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnsupportedMediaError("document is not valid UTF-8 text") from exc
        return MediaTransformResult(
            transform=transform,
            processor=self.capability.name,
            text=text,
            metadata={"mime_type": item.mime_type},
        )


class LocalTranscriptionProcessor:
    """Local speech-to-text adapter around an injected offline callable."""

    def __init__(
        self,
        transcribe: Callable[[bytes, str], str],
        *,
        name: str = "local_transcription",
        max_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self._transcribe = transcribe
        self.capability = ProcessorCapability(
            name=name,
            transforms=frozenset({TransformKind.TRANSCRIPTION}),
            media_kinds=frozenset({MediaKind.AUDIO, MediaKind.VIDEO}),
            max_bytes=max_bytes,
            provider="local",
            network_access=False,
            retains_data=False,
        )

    def process(
        self,
        item: MediaItem,
        data: bytes,
        *,
        instruction: str,
        transform: TransformKind,
    ) -> MediaTransformResult:
        del instruction
        if transform is not TransformKind.TRANSCRIPTION:
            raise UnsupportedMediaError("transcription adapter received another transform")
        text = self._transcribe(data, item.mime_type).strip()
        if not text:
            raise RuntimeError("local transcription returned no text")
        return MediaTransformResult(transform, self.capability.name, text=text)


class HostedTranscriptionProcessor(LocalTranscriptionProcessor):
    """Hosted speech-to-text adapter with explicit network/retention metadata."""

    def __init__(
        self,
        transcribe: Callable[[bytes, str], str],
        *,
        provider: str,
        name: str = "hosted_transcription",
        max_bytes: int = 25 * 1024 * 1024,
        retains_data: bool = False,
        pricing_per_minute: Decimal | None = None,
    ) -> None:
        super().__init__(transcribe, name=name, max_bytes=max_bytes)
        self.capability = ProcessorCapability(
            name=name,
            transforms=frozenset({TransformKind.TRANSCRIPTION}),
            media_kinds=frozenset({MediaKind.AUDIO, MediaKind.VIDEO}),
            max_bytes=max_bytes,
            provider=provider,
            network_access=True,
            retains_data=retains_data,
            pricing_per_unit=pricing_per_minute,
            unit_name="minute",
        )


def _default_transforms(kind: MediaKind) -> tuple[TransformKind, ...]:
    if kind in {MediaKind.AUDIO, MediaKind.VIDEO}:
        return (
            TransformKind.NATIVE,
            TransformKind.TRANSCRIPTION,
            TransformKind.UNDERSTANDING,
        )
    if kind is MediaKind.DOCUMENT:
        return (
            TransformKind.NATIVE,
            TransformKind.EXTRACTION,
            TransformKind.UNDERSTANDING,
        )
    return (TransformKind.NATIVE, TransformKind.UNDERSTANDING)


__all__ = [
    "HostedTranscriptionProcessor",
    "LocalTextExtractor",
    "LocalTranscriptionProcessor",
    "MediaProcessor",
    "MediaProcessorRegistry",
    "MediaTransformResult",
    "ProcessorCapability",
    "TransformKind",
]
