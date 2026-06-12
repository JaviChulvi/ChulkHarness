"""Provider capability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LLMCapabilities:
    """Capabilities exposed by one provider implementation."""

    supports_structured_output: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = False
    api_style: Literal["responses", "chat_completions"] = "chat_completions"
