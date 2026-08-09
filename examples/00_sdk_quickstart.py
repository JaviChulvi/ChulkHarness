"""Credential-free SDK quickstart with an explicit live-provider opt-in."""

from __future__ import annotations

from chulk import Agent, Tool, ToolContext

from common import scripted_or_live


OrderStore = dict[str, dict[str, str]]
ORDERS: OrderStore = {
    "A-100": {"status": "packed", "estimated_ship_date": "tomorrow"}
}


@Tool
def order_status(order_id: str, context: ToolContext[OrderStore]) -> dict[str, str]:
    """Look up one order in the host application's store."""
    return context.require_deps().get(
        order_id,
        {"error": f"Unknown order: {order_id}"},
    )


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
                "content": "Order A-100 is packed and expected to ship tomorrow.",
            },
        ],
    )

    with Agent(
        config=config,
        llm=llm,
        tools=[order_status],
        skills=[],
        deps=ORDERS,
    ) as assistant:
        result = assistant.run_result(
            "Use the order_status tool to say when order A-100 will ship."
        )
    print(f"mode: {mode}")
    print(result.content)
    print("tool_calls: " + ", ".join(call.tool_name for call in result.tool_calls))
    print(f"runtime_dir: {config.to_config().runtime_dir}")
    print(f"trace_path: {result.trace_path}")


if __name__ == "__main__":
    main()
