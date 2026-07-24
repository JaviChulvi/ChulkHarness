from __future__ import annotations

from types import SimpleNamespace

import pytest

from chulk.llm import LLMError
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


def test_gemini_media_inherits_timeout_retry_and_owns_factory_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Client:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    client = Client()

    def create_client(**kwargs: object) -> Client:
        captured.update(kwargs)
        return client

    monkeypatch.setattr("google.genai.Client", create_client)
    processor = GeminiMediaProcessor(
        model="gemini-test",
        api_key="secret",
        timeout_seconds=2.5,
        max_retries=4,
    )

    processor.close()
    processor.close()

    assert captured["http_options"] == {
        "timeout": 2500,
        "retry_options": {"attempts": 5},
    }
    assert client.close_count == 1


def test_injected_gemini_media_client_is_caller_owned_unless_overridden() -> None:
    class Client:
        def __init__(self) -> None:
            self.models = SimpleNamespace()
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    caller_owned = Client()
    GeminiMediaProcessor(
        model="gemini-test",
        api_key=None,
        client=caller_owned,
        part_factory=lambda **kwargs: kwargs,
    ).close()
    wrapper_owned = Client()
    processor = GeminiMediaProcessor(
        model="gemini-test",
        api_key=None,
        client=wrapper_owned,
        part_factory=lambda **kwargs: kwargs,
        owns_client=True,
    )
    processor.close()

    assert caller_owned.close_count == 0
    assert wrapper_owned.close_count == 1


def test_gemini_media_normalizes_timeout_failure() -> None:
    class Models:
        def generate_content(self, **_kwargs: object) -> object:
            raise TimeoutError("provider timeout")

    processor = GeminiMediaProcessor(
        model="gemini-test",
        api_key=None,
        client=SimpleNamespace(models=Models()),
        part_factory=lambda **kwargs: kwargs,
    )

    with pytest.raises(LLMError) as caught:
        processor.process(
            TelegramAttachment("id", "image", "image/png"),
            b"image",
            instruction="Describe",
        )

    assert caught.value.code == "timeout"
