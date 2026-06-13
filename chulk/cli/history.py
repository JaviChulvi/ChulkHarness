"""Prompt history support for interactive input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class PromptHistory:
    """Thin wrapper around readline history for arrow-up/down navigation."""

    readline: Any | None = None
    enabled: bool = False

    @classmethod
    def create(cls, *, enabled: bool = True) -> "PromptHistory":
        """Create a prompt history manager when readline is available."""
        if not enabled:
            return cls(enabled=False)
        try:
            import readline
        except ImportError:
            return cls(enabled=False)
        return cls(readline=readline, enabled=True)

    def replace(self, messages: Iterable[Any]) -> None:
        """Replace readline history with user prompts from the active session."""
        if not self.enabled or self.readline is None:
            return
        self.readline.clear_history()
        for content in _user_prompt_contents(messages):
            self.add(content)

    def add(self, prompt: str) -> None:
        """Add one prompt to history without duplicating the latest item."""
        if not self.enabled or self.readline is None:
            return
        clean_prompt = prompt.strip()
        if not clean_prompt:
            return
        if self._latest() == clean_prompt:
            return
        self.readline.add_history(clean_prompt)

    def _latest(self) -> str | None:
        if self.readline is None:
            return None
        length = self.readline.get_current_history_length()
        if length < 1:
            return None
        latest = self.readline.get_history_item(length)
        return latest if isinstance(latest, str) else None


def _user_prompt_contents(messages: Iterable[Any]) -> list[str]:
    prompts: list[str] = []
    for message in messages:
        role = _message_value(message, "role")
        if role != "user":
            continue
        metadata = _message_value(message, "metadata") or {}
        if isinstance(metadata, dict) and metadata.get("internal"):
            continue
        content = _message_value(message, "content")
        if isinstance(content, str) and content.strip():
            prompts.append(content.strip())
    return prompts


def _message_value(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)
