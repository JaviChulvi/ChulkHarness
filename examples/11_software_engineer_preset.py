"""Use the built-in software engineer preset."""

import bootstrap  # noqa: F401
from chulk import Agent
from chulk.presets import SoftwareEngineer

from common import live_config, print_run_result


def main() -> None:
    assistant = Agent(
        config=live_config("11-software-engineer-preset", permission_profile="read-only"),
        preset=SoftwareEngineer(),
    )
    result = assistant.run_result(
        "Inspect the repository at a high level and explain where the public SDK is defined."
    )
    print_run_result(result)


if __name__ == "__main__":
    main()
