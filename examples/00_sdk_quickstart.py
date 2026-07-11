"""Credential-free SDK quickstart with an explicit live-provider opt-in."""

from __future__ import annotations

from chulk import Agent, Tool

from common import scripted_or_live


@Tool
def order_status(order_id: str) -> str:
    """Look up a demo order by id."""
    return f"Order {order_id} is packed and ships tomorrow."


def main() -> None:
    config, llm, mode = scripted_or_live(
        "00_sdk_quickstart",
        [
            {
                "type": "tool_call",
                "tool_name": "order_status",
                "arguments": {"order_id": "A-100"},
            },
            {
                "type": "final_answer",
                "content": "Order A-100 is packed and ships tomorrow.",
            },
        ],
    )

    with Agent(config=config, llm=llm, tools=[order_status], skills=[]) as assistant:
        result = assistant.run_result(
            "Use the order_status tool to check order A-100, then reply in one sentence."
        )
    print(f"mode: {mode}")
    print(result.content)
    print(f"runtime_dir: {config.to_config().runtime_dir}")
    print(f"trace_path: {result.trace_path}")


if __name__ == "__main__":
    main()
