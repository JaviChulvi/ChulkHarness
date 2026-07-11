"""Minimal SDK quickstart for an installed ChulkHarness package.

Install the hosted-provider extra first with:
    python -m pip install "chulkharness[openai]"
"""

from __future__ import annotations

import os

from chulk import Agent, AgentConfig, Tool


@Tool
def order_status(order_id: str) -> str:
    """Look up a demo order by id."""
    return f"Order {order_id} is packed and ships tomorrow."


def main() -> None:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("CHULK_LLM_PROVIDER"):
        raise SystemExit(
            "Set OPENAI_API_KEY, or configure CHULK_LLM_PROVIDER plus its "
            "credentials, before running this example."
        )

    assistant = Agent(
        config=AgentConfig.from_env(
            project_root=".",
            runtime_dir=".chulk",
            permission_profile="read-only",
        ),
        tools=[order_status],
        skills=[],
    )
    result = assistant.run_result(
        "Use the order_status tool to check order A-100, then reply in one sentence."
    )
    print(result.content)
    print(f"trace_path: {result.trace_path}")


if __name__ == "__main__":
    main()
