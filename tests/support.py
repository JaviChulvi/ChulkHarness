"""Private builders shared by repository tests."""

from __future__ import annotations

import json
from collections.abc import Iterable

from chulk.llm import LLMClient


def final_action(content: str = "done") -> str:
    return json.dumps({"type": "final_answer", "content": content})


def tool_action(name: str, **arguments: object) -> str:
    return json.dumps(
        {
            "type": "tool_call",
            "content": None,
            "tool_name": name,
            "arguments_json": json.dumps(arguments),
        }
    )


class RepeatingFakeLLMClient(LLMClient):
    """Consume scripted responses while keeping the final response repeatable."""

    provider = "test"
    model = "fake"

    def __init__(self, responses: Iterable[str]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.requests.append(messages)
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)
