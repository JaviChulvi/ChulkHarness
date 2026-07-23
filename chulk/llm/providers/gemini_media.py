"""Bounded Gemini media understanding used by channel adapters."""

from __future__ import annotations

from typing import Any, Callable

from chulk.llm.base import LLMConfigurationError, provider_error_from_exception
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
    ) -> None:
        self.model = model
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
            client = genai.Client(api_key=api_key)
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
            else "Extract and describe the useful content of this attachment accurately."
        )
        prompt = f"{task}\nUser instruction: {instruction or 'Analyze this attachment.'}"
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    self._part_factory(data=data, mime_type=attachment.mime_type),
                ],
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


__all__ = ["GeminiMediaProcessor"]
