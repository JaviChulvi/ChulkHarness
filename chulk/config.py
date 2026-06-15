"""Configuration helpers for ChulkHarness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

from chulk.llm import supported_llm_providers


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_PROVIDER = "openai"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_LOCAL_MODEL = "google/gemma-4-12b-qat"
DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MAX_SKILLS_PER_TURN = 3
DEFAULT_MAX_SKILL_CONTENT_CHARS = 4000
DEFAULT_TRACE_MAX_PROMPT_CHARS = 50000
DEFAULT_MAX_OBSERVATION_CHARS = 12000
DEFAULT_MAX_TOOL_STDOUT_CHARS = 8000
DEFAULT_MAX_TOOL_STDERR_CHARS = 4000
SUPPORTED_LLM_PROVIDERS = supported_llm_providers()


@dataclass(frozen=True)
class LLMFallbackProviderConfig:
    """One configured fallback provider after the primary LLM."""

    provider: str
    model: str


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    project_root: Path
    skills_dir: Path
    store_path: Path
    traces_dir: Path
    llm_provider: str
    model: str
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    local_api_key: str | None = None
    local_base_url: str = DEFAULT_LOCAL_BASE_URL
    llm_fallback_providers: tuple[LLMFallbackProviderConfig, ...] = ()
    history_limit: int = 20
    max_tool_calls_per_turn: int = 5
    max_skills_per_turn: int = DEFAULT_MAX_SKILLS_PER_TURN
    max_skill_content_chars: int = DEFAULT_MAX_SKILL_CONTENT_CHARS
    shell_timeout_seconds: int = 10
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    trace_max_prompt_chars: int = DEFAULT_TRACE_MAX_PROMPT_CHARS
    max_observation_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    max_tool_stdout_chars: int = DEFAULT_MAX_TOOL_STDOUT_CHARS
    max_tool_stderr_chars: int = DEFAULT_MAX_TOOL_STDERR_CHARS


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple .env file without adding a runtime dependency."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{key} must be greater than zero")
    return parsed


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{key} must be greater than zero")
    return parsed


def load_config(environ: Mapping[str, str] | None = None) -> Config:
    """Load local development configuration."""
    default_root = Path(__file__).resolve().parent.parent
    process_env = dict(os.environ if environ is None else environ)
    initial_root = Path(process_env.get("CHULK_PROJECT_ROOT", default_root)).resolve()
    dotenv_env = _parse_dotenv(initial_root / ".env")
    env = {**dotenv_env, **process_env}
    project_root = Path(env.get("CHULK_PROJECT_ROOT", initial_root)).resolve()
    llm_provider = (env.get("CHULK_LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
        raise ValueError(f"CHULK_LLM_PROVIDER must be one of: {supported}")

    default_model = _default_model_for_provider(llm_provider)
    model = env.get("CHULK_MODEL") or default_model

    return Config(
        project_root=project_root,
        skills_dir=project_root / "skills",
        store_path=project_root / "chulk" / "store.sqlite",
        traces_dir=project_root / "traces",
        llm_provider=llm_provider,
        model=model,
        openai_api_key=env.get("OPENAI_API_KEY") or None,
        deepseek_api_key=env.get("CHULK_DEEPSEEK_API_KEY") or env.get("DEEPSEEK_API_KEY") or None,
        deepseek_base_url=env.get("CHULK_DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL,
        local_api_key=env.get("CHULK_LOCAL_API_KEY") or None,
        local_base_url=env.get("CHULK_LOCAL_BASE_URL") or DEFAULT_LOCAL_BASE_URL,
        llm_fallback_providers=_parse_fallback_providers(
            env,
            primary_provider=llm_provider,
            primary_model=model,
        ),
        history_limit=_env_int(env, "CHULK_HISTORY_LIMIT", 20),
        max_tool_calls_per_turn=_env_int(env, "CHULK_MAX_TOOL_CALLS_PER_TURN", 5),
        max_skills_per_turn=_env_int(env, "CHULK_MAX_SKILLS_PER_TURN", DEFAULT_MAX_SKILLS_PER_TURN),
        max_skill_content_chars=_env_int(env, "CHULK_MAX_SKILL_CONTENT_CHARS", DEFAULT_MAX_SKILL_CONTENT_CHARS),
        shell_timeout_seconds=_env_int(env, "CHULK_SHELL_TIMEOUT_SECONDS", 10),
        llm_timeout_seconds=_env_float(env, "CHULK_LLM_TIMEOUT_SECONDS", 60.0),
        llm_max_retries=_env_int(env, "CHULK_LLM_MAX_RETRIES", 2),
        trace_max_prompt_chars=_env_int(env, "CHULK_TRACE_MAX_PROMPT_CHARS", DEFAULT_TRACE_MAX_PROMPT_CHARS),
        max_observation_chars=_env_int(env, "CHULK_MAX_OBSERVATION_CHARS", DEFAULT_MAX_OBSERVATION_CHARS),
        max_tool_stdout_chars=_env_int(env, "CHULK_MAX_TOOL_STDOUT_CHARS", DEFAULT_MAX_TOOL_STDOUT_CHARS),
        max_tool_stderr_chars=_env_int(env, "CHULK_MAX_TOOL_STDERR_CHARS", DEFAULT_MAX_TOOL_STDERR_CHARS),
    )


def _default_model_for_provider(provider: str) -> str:
    if provider == "deepseek":
        return DEFAULT_DEEPSEEK_MODEL
    if provider == "local":
        return DEFAULT_LOCAL_MODEL
    return DEFAULT_MODEL


def _parse_fallback_providers(
    env: Mapping[str, str],
    *,
    primary_provider: str,
    primary_model: str,
) -> tuple[LLMFallbackProviderConfig, ...]:
    raw_value = env.get("CHULK_LLM_FALLBACK_PROVIDERS")
    if raw_value is None or raw_value.strip() == "":
        return ()

    providers: list[LLMFallbackProviderConfig] = []
    seen = {(primary_provider.lower(), primary_model.lower())}
    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        provider, separator, raw_model = item.partition(":")
        provider = provider.strip().lower()
        model = raw_model.strip() if separator else _default_model_for_provider(provider)
        if not provider:
            raise ValueError("CHULK_LLM_FALLBACK_PROVIDERS contains an empty provider name")
        if provider not in SUPPORTED_LLM_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
            raise ValueError(f"CHULK_LLM_FALLBACK_PROVIDERS must use providers from: {supported}")
        if not model:
            raise ValueError("CHULK_LLM_FALLBACK_PROVIDERS entries with ':' must include a model")

        key = (provider, model.lower())
        if key in seen:
            continue
        seen.add(key)
        providers.append(LLMFallbackProviderConfig(provider=provider, model=model))

    return tuple(providers)
