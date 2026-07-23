from __future__ import annotations

from types import SimpleNamespace

from chulk.llm.providers.gemini_media import GeminiMediaProcessor
from chulk.telegram.client import TelegramAttachment


def test_gemini_media_processor_shapes_bounded_media_request() -> None:
    calls: list[dict[str, object]] = []

    class Models:
        def generate_content(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(text=" hello world ")

    processor = GeminiMediaProcessor(
        model="gemini-test",
        api_key=None,
        client=SimpleNamespace(models=Models()),
        part_factory=lambda **kwargs: ("part", kwargs),
    )
    result = processor.process(
        TelegramAttachment("id", "voice", "audio/ogg"),
        b"sound",
        instruction="Be exact",
    )

    assert result == "hello world"
    assert calls[0]["model"] == "gemini-test"
    contents = calls[0]["contents"]
    assert "Transcribe" in contents[0]
    assert contents[1] == ("part", {"data": b"sound", "mime_type": "audio/ogg"})
