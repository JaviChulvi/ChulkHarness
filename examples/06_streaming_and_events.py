"""Receive streamed final-answer deltas and public SDK events."""

import bootstrap  # noqa: F401
from chulk import AgentEvent, ChatAgent

from common import live_config


INTERESTING_EVENTS = {
    "turn_started",
    "model_request_started",
    "model_stream_delta",
    "final_answer",
    "turn_finished",
}


def on_event(event: AgentEvent) -> None:
    if event.type in INTERESTING_EVENTS:
        print(f"\n[event] {event.type}")


def on_delta(text: str) -> None:
    print(text, end="", flush=True)


def main() -> None:
    assistant = ChatAgent(config=live_config("06-streaming-and-events"), on_event=on_event)
    print("=== Streaming Answer ===")
    result = assistant.run_result(
        "Write a short, practical checklist for embedding Chulk in a Python service.",
        on_delta=on_delta,
    )
    print("\n\n=== Result Metadata ===")
    print(f"status: {result.status}")
    print(f"trace_path: {result.trace_path}")


if __name__ == "__main__":
    main()
