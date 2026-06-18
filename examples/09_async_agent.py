"""Use AsyncAgent from an async application or server handler."""

from __future__ import annotations

import asyncio

import bootstrap  # noqa: F401
from chulk import AsyncChatAgent

from common import live_config


async def main() -> None:
    assistant = AsyncChatAgent(config=live_config("09-async-agent"))
    result = await assistant.run_result("Reply with two concise tips for using AsyncAgent in web apps.")
    print(result.content)
    print(f"status: {result.status}")
    print(f"trace_path: {result.trace_path}")


if __name__ == "__main__":
    asyncio.run(main())
