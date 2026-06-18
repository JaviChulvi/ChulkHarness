"""Shared helpers for SDK examples."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import bootstrap  # noqa: F401
from chulk import AgentConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_STATE_ROOT = REPO_ROOT / "examples" / "runtime"


def runtime_dir(name: str) -> Path:
    root = EXAMPLE_STATE_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def provider_from_env(explicit: str | None = None) -> str:
    return (explicit or os.getenv("CHULK_LLM_PROVIDER") or "openai").strip().lower()


def require_env(*names: str) -> None:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        print(f"Missing required environment variable(s): {joined}", file=sys.stderr)
        raise SystemExit(2)


def require_provider_credentials(provider: str) -> None:
    if provider == "openai":
        require_env("OPENAI_API_KEY")
    elif provider == "deepseek" and not (os.getenv("CHULK_DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")):
        print("Missing CHULK_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY", file=sys.stderr)
        raise SystemExit(2)


def live_config(
    name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    permission_profile: str | None = None,
    **overrides: Any,
) -> AgentConfig:
    provider_name = provider_from_env(provider)
    require_provider_credentials(provider_name)
    return AgentConfig.from_env(
        project_root=REPO_ROOT,
        runtime_dir=runtime_dir(name),
        provider=provider_name,
        model=model,
        permission_profile=permission_profile,
        **overrides,
    )


def print_run_result(result: Any) -> None:
    print("\n=== Assistant ===")
    print(result.content)
    print("\n=== Metadata ===")
    print(f"status: {result.status}")
    print(f"turn_id: {result.turn_id}")
    print(f"conversation_id: {result.conversation_id}")
    print(f"trace_path: {result.trace_path}")
    if result.usage:
        print(f"usage: {result.usage}")
    if result.cost:
        print(f"cost: {result.cost}")
    if result.loaded_skill_names:
        print(f"skills: {', '.join(result.loaded_skill_names)}")
    if result.loaded_memory_ids:
        print(f"memories: {', '.join(result.loaded_memory_ids)}")
    if result.tool_calls:
        print("\n=== Tool Calls ===")
        for call in result.tool_calls:
            name = call.get("tool_name")
            success = call.get("success")
            print(f"- {name}: success={success}")
    if result.observations:
        print("\n=== Observations ===")
        for observation in result.observations:
            name = observation.get("tool_name")
            preview = str(observation.get("content", "")).replace("\n", " ")[:160]
            print(f"- {name}: {preview}")
