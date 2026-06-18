"""Use a built-in tool and inspect structured tool metadata."""

import bootstrap  # noqa: F401
from chulk import Agent, Tools

from common import live_config, print_run_result


def main() -> None:
    assistant = Agent(
        config=live_config("03-built-in-tools"),
        tools=[Tools.calculator],
        skills=[],
    )
    result = assistant.run_result(
        "Use the calculator tool to compute (128 * 37) + 944, then explain the result briefly."
    )
    print_run_result(result)


if __name__ == "__main__":
    main()
