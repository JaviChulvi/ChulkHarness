"""Bounded Gemini media understanding used by channel adapters."""

from __future__ import annotations

from typing import Any, Callable, cast

from chulk.llm.base import LLMConfigurationError, provider_error_from_exception
from chulk.llm.lifecycle import aclose_resources, close_resources
from chulk.telegram.client import TelegramAttachment


class GeminiMediaProcessor:
    """Describe or transcribe one in-memory attachment with Gemini."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        client: Any | None = None,
        part_factory: Callable[..., object] | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        owns_client: bool | None = None,
    ) -> None:
        self.model = model
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self._owns_client = (client is None) if owns_client is None else owns_client
        self._closed = False
        if client is None:
            if not api_key:
                raise LLMConfigurationError("A Gemini API key is required for media processing")
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise LLMConfigurationError(
                    "The google-genai package is required for Gemini media processing"
                ) from exc
            client = genai.Client(
                api_key=api_key,
                http_options=cast(
                    Any,
                    {
                        "timeout": max(1, round(timeout_seconds * 1000)),
                        "retry_options": {"attempts": max_retries + 1},
                    },
                ),
            )
            part_factory = types.Part.from_bytes
        self._client = client
        if part_factory is None:
            raise ValueError("part_factory is required with an injected client")
        self._part_factory = part_factory

    def process(
        self,
        attachment: TelegramAttachment,
        data: bytes,
        *,
        instruction: str,
    ) -> str:
        """Return a transcription or grounded description without retaining bytes."""
        task = (
            "Transcribe this audio accurately. Include relevant non-speech sounds."
            if attachment.kind in {"voice", "audio"}
            else (
                "Describe this video accurately with timestamps for important events."
                if attachment.kind == "video"
                else "Extract and describe the useful content of this attachment accurately."
            )
        )
        prompt = f"{task}\nUser instruction: {instruction or 'Analyze this attachment.'}"
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=cast(
                    Any,
                    [
                        prompt,
                        self._part_factory(data=data, mime_type=attachment.mime_type),
                    ],
                ),
            )
        except Exception as exc:
            raise provider_error_from_exception(
                exc,
                provider="gemini",
                model=self.model,
                message="Gemini media processing failed",
            ) from exc
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini media processing returned no text")
        return text.strip()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            close_resources((self._client,))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await aclose_resources((self._client,))


__all__ = ["GeminiMediaProcessor"]
