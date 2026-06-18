"""Run Chulk against a local OpenAI-compatible provider."""

from __future__ import annotations

import os

import bootstrap  # noqa: F401
from chulk import AgentConfig, ChatAgent

from common import REPO_ROOT, runtime_dir


def main() -> None:
    base_url = os.getenv("CHULK_LOCAL_BASE_URL", "http://localhost:1234/v1")
    model = os.getenv("CHULK_MODEL", "local-model")
    config = AgentConfig.local(
        project_root=REPO_ROOT,
        runtime_dir=runtime_dir("12-local-provider"),
        model=model,
        base_url=base_url,
        api_key=os.getenv("CHULK_LOCAL_API_KEY"),
    )
    assistant = ChatAgent(config=config)
    result = assistant.run_result("Say hello from the local-provider SDK example.")
    print(result.content)
    print(f"provider: local")
    print(f"model: {model}")
    print(f"base_url: {base_url}")
    print(f"trace_path: {result.trace_path}")


if __name__ == "__main__":
    main()
