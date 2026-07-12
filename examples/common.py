"""Shared helpers for SDK examples."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import bootstrap  # noqa: F401
from chulk import AgentConfig
from chulk.llm import LLMClient
from chulk.testing import ScriptedLLMClient


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_STATE_ROOT = Path(
    os.getenv("CHULK_EXAMPLE_RUNTIME_DIR", REPO_ROOT / "examples" / "runtime")
).expanduser()
EXAMPLE_MODE_ENV = "CHULK_EXAMPLE_MODE"
EXPLICIT_MODEL_PROVIDERS = frozenset(
    {"openai-compatible", "openrouter", "anthropic", "bedrock", "gemini"}
)


def runtime_dir(name: str) -> Path:
    root = EXAMPLE_STATE_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def provider_from_env(explicit: str | None = None) -> str:
    return (explicit or os.getenv("CHULK_LLM_PROVIDER") or "openai").strip().lower()


def require_env(*names: str) -> None:
    missing = [name for name in names if not _configured_env(name)]
    if missing:
        joined = ", ".join(missing)
        print(f"Missing required environment variable(s): {joined}", file=sys.stderr)
        raise SystemExit(2)


def require_any_env(provider: str, *names: str) -> None:
    if any(_configured_env(name) for name in names):
        return
    joined = ", ".join(names)
    print(f"Missing {provider} configuration; set one of: {joined}", file=sys.stderr)
    raise SystemExit(2)


def require_provider_credentials(provider: str, *, model: str | None = None) -> None:
    if provider in EXPLICIT_MODEL_PROVIDERS and not (
        (model is not None and model.strip()) or _configured_env("CHULK_MODEL")
    ):
        print(f"CHULK_MODEL is required for provider {provider}", file=sys.stderr)
        raise SystemExit(2)

    if provider == "openai":
        require_env("OPENAI_API_KEY")
    elif provider == "deepseek":
        require_any_env(provider, "CHULK_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY")
    elif provider == "openai-compatible":
        require_env(
            "CHULK_OPENAI_COMPATIBLE_API_KEY",
            "CHULK_OPENAI_COMPATIBLE_BASE_URL",
        )
    elif provider == "openrouter":
        require_any_env(provider, "CHULK_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    elif provider == "anthropic":
        require_any_env(provider, "CHULK_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    elif provider == "bedrock":
        require_any_env(
            provider,
            "CHULK_BEDROCK_API_KEY",
            "BEDROCK_API_KEY",
            "AWS_BEARER_TOKEN_BEDROCK",
        )
        require_any_env(provider, "CHULK_BEDROCK_BASE_URL", "CHULK_BASE_URL")
    elif provider == "gemini":
        require_any_env(
            provider,
            "CHULK_GEMINI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        )


def _configured_env(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and bool(value.strip())


def live_config(
    name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    permission_profile: str | None = None,
    **overrides: Any,
) -> AgentConfig:
    provider_name = provider_from_env(provider)
    require_provider_credentials(provider_name, model=model)
    return AgentConfig.from_env(
        project_root=REPO_ROOT,
        runtime_dir=runtime_dir(name),
        provider=provider_name,
        model=model,
        permission_profile=permission_profile,
        **overrides,
    )


def scripted_or_live(
    name: str,
    responses: list[object],
    *,
    permission_profile: str = "read-only",
) -> tuple[AgentConfig, LLMClient | None, str]:
    """Select deterministic execution by default and live execution explicitly."""
    mode = os.getenv(EXAMPLE_MODE_ENV, "scripted").strip().lower()
    if mode == "scripted":
        return (
            AgentConfig(
                project_root=REPO_ROOT,
                runtime_dir=runtime_dir(name),
                permission_profile=permission_profile,
            ),
            ScriptedLLMClient(responses),
            mode,
        )
    if mode == "live":
        return live_config(name, permission_profile=permission_profile), None, mode
    raise SystemExit(f"{EXAMPLE_MODE_ENV} must be 'scripted' or 'live', got {mode!r}")


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
