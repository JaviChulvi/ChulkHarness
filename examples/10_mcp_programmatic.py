"""Configure remote MCP servers directly from Python."""

from __future__ import annotations

import os
import sys

import bootstrap  # noqa: F401
from chulk import ChatAgent, MCP

from common import live_config, print_run_result


def csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    server_url = os.getenv("DOCS_MCP_URL")
    if not server_url:
        print("Set DOCS_MCP_URL to a Streamable HTTP MCP server URL before running this example.", file=sys.stderr)
        raise SystemExit(2)

    server = MCP.streamable_http(
        label=os.getenv("DOCS_MCP_LABEL", "docs"),
        server_url=server_url,
        server_description="Documentation search server",
        allowed_tools=csv_env("DOCS_MCP_ALLOWED_TOOLS"),
        authorization_env="DOCS_MCP_TOKEN" if os.getenv("DOCS_MCP_TOKEN") else None,
        approval=os.getenv("DOCS_MCP_APPROVAL", "always"),
    )
    assistant = ChatAgent(
        config=live_config("10-mcp-programmatic", permission_profile="workspace-write"),
        mcp=[server],
        permission_callback=lambda _request, _record: True,
    )
    result = assistant.run_result("Use the docs MCP server to search for Chulk SDK usage guidance.")
    print_run_result(result)


if __name__ == "__main__":
    main()
