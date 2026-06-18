"""Create an agent with explicit programmatic configuration."""

import bootstrap  # noqa: F401
from chulk import ChatAgent

from common import live_config


def main() -> None:
    config = live_config(
        "02-agent-config",
        permission_profile="read-only",
    )

    assistant = ChatAgent(config=config)
    result = assistant.run_result("Say hello and mention which kind of runtime state this example configured.")
    runtime_config = config.to_config()
    print(result.content)
    print(f"store_path: {runtime_config.store_path}")
    print(f"trace_path: {runtime_config.traces_dir}")


if __name__ == "__main__":
    main()
