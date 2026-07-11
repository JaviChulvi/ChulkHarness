"""Consume public SDK events with callbacks and sync/async iterators."""

import asyncio
import bootstrap  # noqa: F401
from chulk import AgentEvent, AsyncAgent, ChatAgent, EventName, RunCompletedPayload

from common import live_config


INTERESTING_EVENTS = {
    EventName.RUN_STARTED.value,
    EventName.MODEL_REQUEST_STARTED.value,
    EventName.MODEL_DELTA.value,
    EventName.RUN_COMPLETED.value,
}


def on_event(event: AgentEvent) -> None:
    if event.name in INTERESTING_EVENTS:
        print(f"\n[event] {event.name}")


def on_delta(text: str) -> None:
    print(text, end="", flush=True)


def main() -> None:
    with ChatAgent(config=live_config("06-streaming-and-events"), on_event=on_event) as assistant:
        print("=== Synchronous Event Iterator ===")
        for event in assistant.run_events(
            "Write a short, practical checklist for embedding Chulk in a Python service.",
            on_delta=on_delta,
        ):
            if isinstance(event.payload, RunCompletedPayload):
                print(f"\nstatus: {event.payload.result.status}")

    asyncio.run(async_example())


async def async_example() -> None:
    async with AsyncAgent(config=live_config("06-async-events")) as assistant:
        print("\n=== Asynchronous Event Iterator ===")
        async for event in assistant.run_events_async("Give one concise embedding tip."):
            print(event.name)


if __name__ == "__main__":
    main()
