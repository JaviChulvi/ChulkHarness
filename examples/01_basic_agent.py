"""Create a basic agent and get a string response."""

import bootstrap  # noqa: F401
from chulk import ChatAgent

from common import live_config


def main() -> None:
    assistant = ChatAgent(config=live_config("01-basic-agent"))
    answer = assistant.run("In one sentence, explain what Chulk is useful for.")
    print(answer)


if __name__ == "__main__":
    main()
