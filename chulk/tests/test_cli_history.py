"""Tests for interactive prompt history."""

from chulk.cli.history import PromptHistory
from chulk.sessions.models import MessageRecord


class FakeReadline:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.completer = None
        self.delimiters = ""
        self.binding = ""

    def clear_history(self) -> None:
        self.items.clear()

    def add_history(self, item: str) -> None:
        self.items.append(item)

    def get_current_history_length(self) -> int:
        return len(self.items)

    def get_history_item(self, index: int) -> str | None:
        if index < 1 or index > len(self.items):
            return None
        return self.items[index - 1]

    def set_completer_delims(self, delimiters: str) -> None:
        self.delimiters = delimiters

    def set_completer(self, completer) -> None:
        self.completer = completer

    def parse_and_bind(self, binding: str) -> None:
        self.binding = binding


def test_prompt_history_loads_user_messages_only():
    readline = FakeReadline()
    history = PromptHistory(readline=readline, enabled=True)
    messages = [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "first answer"},
        MessageRecord(
            id="message-1",
            conversation_id="conversation-1",
            turn_id="turn-1",
            role="user",
            content="internal approval",
            ordinal=2,
            created_at="2026-01-01T00:00:00+00:00",
            metadata={"internal": True},
        ),
        MessageRecord(
            id="message-2",
            conversation_id="conversation-1",
            turn_id="turn-2",
            role="user",
            content="second prompt",
            ordinal=3,
            created_at="2026-01-01T00:00:01+00:00",
        ),
    ]

    history.replace(messages)

    assert readline.items == ["first prompt", "second prompt"]


def test_prompt_history_does_not_duplicate_latest_prompt():
    readline = FakeReadline()
    history = PromptHistory(readline=readline, enabled=True)

    history.add("same prompt")
    history.add("same prompt")
    history.add("next prompt")

    assert readline.items == ["same prompt", "next prompt"]


def test_prompt_history_configures_command_completion():
    readline = FakeReadline()
    history = PromptHistory(readline=readline, enabled=True)

    history.configure_completion(["/status", "/sessions", "/status"])

    assert readline.delimiters == "\n"
    assert readline.binding == "tab: complete"
    assert readline.completer("/sta", 0) == "/status"
    assert readline.completer("/sta", 1) is None
    assert readline.completer("/s", 1) == "/sessions"
