"""Configuration helpers for ChulkHarness."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any

from chulk.capabilities import Capabilities, MemoryMode

from chulk.llm.capabilities import (
    LOCAL_DEFAULT_CONTEXT_WINDOW_TOKENS,
    LOCAL_DEFAULT_RESPONSE_RESERVE_TOKENS,
)
from chulk.llm.factory import supported_llm_providers
from chulk.llm.providers.compatible import DEFAULT_OPENROUTER_BASE_URL
from chulk.mcp import MCPServerConfig, load_mcp_servers
from chulk.memory.models import MemoryRetentionPolicy, normalize_memory_namespace
from chulk.tools.permissions import DEFAULT_PERMISSION_PROFILE, normalize_permission_profile


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_PROVIDER = "openai"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MOONSHOT_MODEL = "kimi-k3"
DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_LOCAL_MODEL = "google/gemma-4-12b-qat"
DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LOCAL_CONTEXT_WINDOW_TOKENS = LOCAL_DEFAULT_CONTEXT_WINDOW_TOKENS
DEFAULT_MAX_SKILLS_PER_TURN = 3
DEFAULT_MAX_SKILL_CONTENT_CHARS = 4000
DEFAULT_TRACE_MAX_PROMPT_CHARS = 50000
DEFAULT_MAX_OBSERVATION_CHARS = 12000
DEFAULT_MAX_TOOL_STDOUT_CHARS = 8000
DEFAULT_MAX_TOOL_STDERR_CHARS = 4000
DEFAULT_MAX_REFLECTION_ATTEMPTS = 0
SUPPORTED_LLM_PROVIDERS = supported_llm_providers()


class ConfigValueError(ValueError):
    """Internal validation error retaining the invalid environment field."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


@dataclass(frozen=True)
class LLMFallbackProviderConfig:
    """One configured fallback provider after the primary LLM."""

    provider: str
    model: str


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    project_root: Path
    runtime_dir: Path
    skills_dir: Path
    skills_dirs: tuple[Path, ...]
    store_path: Path
    traces_dir: Path
    mcp_config_path: Path
    llm_provider: str
    model: str
    mcp_servers: tuple[MCPServerConfig, ...] = ()
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    moonshot_api_key: str | None = field(default=None, kw_only=True)
    moonshot_base_url: str = field(default=DEFAULT_MOONSHOT_BASE_URL, kw_only=True)
    local_api_key: str | None = None
    local_base_url: str = DEFAULT_LOCAL_BASE_URL
    local_context_window_tokens: int = field(
        default=DEFAULT_LOCAL_CONTEXT_WINDOW_TOKENS,
        kw_only=True,
    )
    openai_compatible_api_key: str | None = None
    openai_compatible_base_url: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    bedrock_api_key: str | None = None
    bedrock_base_url: str | None = None
    gemini_api_key: str | None = None
    gemini_base_url: str | None = None
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
    max_reflection_attempts: int = DEFAULT_MAX_REFLECTION_ATTEMPTS
    permission_profile: str = DEFAULT_PERMISSION_PROFILE
    profile_id: str = "default"
    memory_retention_policy: MemoryRetentionPolicy | None = None


@dataclass(frozen=True)
class AgentConfig:
    """Programmatic SDK configuration with environment fallback."""

    project_root: str | Path | None = None
    runtime_dir: str | Path | None = None
    provider: str | None = None
    model: str | None = None
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str | None = None
    moonshot_api_key: str | None = field(default=None, kw_only=True)
    moonshot_base_url: str | None = field(default=None, kw_only=True)
    local_api_key: str | None = None
    local_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    openai_compatible_base_url: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    bedrock_api_key: str | None = None
    bedrock_base_url: str | None = None
    gemini_api_key: str | None = None
    gemini_base_url: str | None = None
    permission_profile: str | None = None
    store_path: str | Path | None = None
    traces_dir: str | Path | None = None
    skills_dir: str | Path | None = None
    mcp_servers: Iterable[MCPServerConfig] | None = None
    llm_fallback_providers: Iterable[LLMFallbackProviderConfig] | None = None
    history_limit: int | None = None
    max_tool_calls_per_turn: int | None = None
    max_skills_per_turn: int | None = None
    max_skill_content_chars: int | None = None
    shell_timeout_seconds: int | None = None
    llm_timeout_seconds: float | None = None
    llm_max_retries: int | None = None
    trace_max_prompt_chars: int | None = None
    max_observation_chars: int | None = None
    max_tool_stdout_chars: int | None = None
    max_tool_stderr_chars: int | None = None
    max_reflection_attempts: int | None = None
    capabilities: Capabilities | None = None
    memory_mode: MemoryMode | str | None = None
    memory_namespace: str | None = None
    memory_retention_policy: MemoryRetentionPolicy | None = None
    local_context_window_tokens: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.memory_retention_policy is not None and not isinstance(
            self.memory_retention_policy,
            MemoryRetentionPolicy,
        ):
            raise TypeError("memory_retention_policy must be a MemoryRetentionPolicy")
        if self.mcp_servers is not None:
            object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers))
        if self.llm_fallback_providers is not None:
            object.__setattr__(self, "llm_fallback_providers", tuple(self.llm_fallback_providers))
        if self.memory_mode is not None:
            base = self.capabilities or Capabilities.read_only()
            capabilities = base.with_memory(self.memory_mode)
            object.__setattr__(self, "capabilities", capabilities)
            object.__setattr__(self, "memory_mode", capabilities.memory)
        if self.memory_namespace is not None:
            object.__setattr__(
                self,
                "memory_namespace",
                normalize_memory_namespace(self.memory_namespace),
            )

    def resolved_capabilities(self) -> Capabilities:
        """Return explicit capabilities or the safe SDK default."""
        return self.capabilities or Capabilities.read_only()

    def with_overrides(self, **overrides: Any) -> "AgentConfig":
        """Return a copy with any AgentConfig field overridden."""
        return replace(self, **overrides)

    @staticmethod
    def fallback_provider(provider: str, model: str) -> LLMFallbackProviderConfig:
        """Create one provider fallback entry for AgentConfig."""
        return LLMFallbackProviderConfig(provider=provider, model=model)

    @classmethod
    def from_env(
        cls,
        *,
        project_root: str | Path | None = None,
        runtime_dir: str | Path | None = None,
        provider: str | None = None,
        model: str | None = None,
        permission_profile: str | None = None,
        **overrides: Any,
    ) -> "AgentConfig":
        """Create SDK config from the current environment with optional overrides."""
        env = dict(os.environ)
        resolved_project_root = _project_root_override(env, project_root)
        values: dict[str, Any] = {
            "project_root": resolved_project_root,
            "runtime_dir": runtime_dir or os.getenv("CHULK_RUNTIME_DIR") or None,
            "provider": provider or os.getenv("CHULK_LLM_PROVIDER") or None,
            "model": model or os.getenv("CHULK_MODEL") or None,
            "permission_profile": permission_profile or os.getenv("CHULK_PERMISSION_PROFILE") or None,
        }
        values.update(overrides)
        return cls(**{key: value for key, value in values.items() if value is not None})

    @classmethod
    def openai(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for OpenAI-backed agents."""
        values = dict(kwargs)
        if api_key is not None:
            values["openai_api_key"] = api_key
        return cls.from_env(provider="openai", model=model or DEFAULT_MODEL, **values)

    @classmethod
    def deepseek(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for DeepSeek-backed agents."""
        values = dict(kwargs)
        if api_key is not None:
            values["deepseek_api_key"] = api_key
        if base_url is not None:
            values["deepseek_base_url"] = base_url
        return cls.from_env(provider="deepseek", model=model or DEFAULT_DEEPSEEK_MODEL, **values)

    @classmethod
    def moonshot(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for Moonshot AI-backed agents."""
        values = dict(kwargs)
        if api_key is not None:
            values["moonshot_api_key"] = api_key
        if base_url is not None:
            values["moonshot_base_url"] = base_url
        return cls.from_env(provider="moonshot", model=model or DEFAULT_MOONSHOT_MODEL, **values)

    @classmethod
    def local(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        context_window_tokens: int | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for local OpenAI-compatible providers."""
        values = dict(kwargs)
        if api_key is not None:
            values["local_api_key"] = api_key
        if base_url is not None:
            values["local_base_url"] = base_url
        if context_window_tokens is not None:
            values["local_context_window_tokens"] = context_window_tokens
        return cls.from_env(provider="local", model=model or DEFAULT_LOCAL_MODEL, **values)

    @classmethod
    def openai_compatible(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for a hosted OpenAI-compatible endpoint."""
        values = dict(kwargs)
        if api_key is not None:
            values["openai_compatible_api_key"] = api_key
        if base_url is not None:
            values["openai_compatible_base_url"] = base_url
        return cls.from_env(provider="openai-compatible", model=model, **values)

    @classmethod
    def openrouter(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for OpenRouter-backed agents."""
        values = dict(kwargs)
        if api_key is not None:
            values["openrouter_api_key"] = api_key
        if base_url is not None:
            values["openrouter_base_url"] = base_url
        return cls.from_env(provider="openrouter", model=model, **values)

    @classmethod
    def anthropic(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for Anthropic-backed agents."""
        values = dict(kwargs)
        if api_key is not None:
            values["anthropic_api_key"] = api_key
        if base_url is not None:
            values["anthropic_base_url"] = base_url
        return cls.from_env(provider="anthropic", model=model, **values)

    @classmethod
    def bedrock(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for an AWS Bedrock OpenAI-compatible endpoint."""
        values = dict(kwargs)
        if api_key is not None:
            values["bedrock_api_key"] = api_key
        if base_url is not None:
            values["bedrock_base_url"] = base_url
        return cls.from_env(provider="bedrock", model=model, **values)

    @classmethod
    def gemini(
        cls,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for Google Gemini-backed agents."""
        resolved_model = model.strip()
        if not resolved_model:
            raise ValueError("AgentConfig.gemini requires a non-empty model")
        values = dict(kwargs)
        if api_key is not None:
            values["gemini_api_key"] = api_key
        if base_url is not None:
            values["gemini_base_url"] = base_url
        return cls.from_env(provider="gemini", model=resolved_model, **values)

    def to_config(self) -> Config:
        """Build the internal runtime config."""
        return _resolve_config(dict(os.environ), overrides=self)


def _iter_dotenv(path: Path | None) -> Iterator[tuple[str, str]]:
    """Parse a simple .env file without adding a runtime dependency."""
    if path is None or not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        yield key.strip(), value.strip().strip("'\"")


def _parse_dotenv(path: Path | None) -> dict[str, str]:
    return dict(_iter_dotenv(path))


def _env_value(env: Mapping[str, str], key: str, override: str | None) -> str | None:
    return env.get(key) if override is None else str(override)


def _env_int(
    env: Mapping[str, str], key: str, default: int, override: int | None = None, *, minimum: int = 1,
) -> int:
    value = env.get(key) if override is None else override
    if value is None or value == "":
        return default
    try:
        parsed = int(value) if override is None or type(value) is int else int(str(value))
    except ValueError as exc:
        raise ConfigValueError(key, f"{key} must be an integer") from exc
    if parsed < minimum:
        requirement = "zero or greater" if minimum == 0 else "greater than zero"
        raise ConfigValueError(key, f"{key} must be {requirement}")
    return parsed


def _local_context_window_tokens(env: Mapping[str, str], override: int | None) -> int:
    key = "CHULK_LOCAL_CONTEXT_WINDOW_TOKENS"
    value = _env_int(env, key, DEFAULT_LOCAL_CONTEXT_WINDOW_TOKENS, override)
    if value <= LOCAL_DEFAULT_RESPONSE_RESERVE_TOKENS:
        raise ConfigValueError(
            key,
            f"{key} must be greater than the local response reserve "
            f"({LOCAL_DEFAULT_RESPONSE_RESERVE_TOKENS})",
        )
    return value


def _env_float(env: Mapping[str, str], key: str, default: float, override: float | None) -> float:
    value = env.get(key) if override is None else override
    if value is None or value == "":
        return default
    try:
        parsed = float(value) if override is None or type(value) in (int, float) else float(str(value))
    except ValueError as exc:
        raise ConfigValueError(key, f"{key} must be a number") from exc
    if parsed <= 0:
        raise ConfigValueError(key, f"{key} must be greater than zero")
    return parsed


def _resolve_config_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def load_config(environ: Mapping[str, str] | None = None) -> Config:
    """Load local development configuration."""
    return _resolve_config(dict(os.environ if environ is None else environ))


def _resolve_config(process_env: dict[str, str], *, overrides: AgentConfig | None = None) -> Config:
    sdk = overrides is not None
    options = overrides if overrides is not None else AgentConfig()
    initial_root = (
        _project_root_override(process_env, options.project_root)
        if sdk else Path(process_env.get("CHULK_PROJECT_ROOT", Path.cwd())).resolve()
    )
    runtime_override = _runtime_dir_override(initial_root, process_env, options.runtime_dir) if sdk else None
    env = {**_parse_dotenv(initial_root / ".env"), **process_env}
    project_root = initial_root if sdk else Path(env.get("CHULK_PROJECT_ROOT", initial_root)).resolve()
    llm_provider = (_env_value(env, "CHULK_LLM_PROVIDER", options.provider) or DEFAULT_PROVIDER).lower()
    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
        raise ConfigValueError("CHULK_LLM_PROVIDER", f"CHULK_LLM_PROVIDER must be one of: {supported}")

    model = _configured_model(env, llm_provider, options.model)
    bedrock_base_url = _bedrock_base_url(env, options.bedrock_base_url)
    llm_fallback_providers = _parse_fallback_providers(
        env,
        primary_provider=llm_provider,
        primary_model=model,
    )
    bedrock_is_configured = llm_provider == "bedrock" or any(
        fallback.provider == "bedrock" for fallback in llm_fallback_providers
    )
    if bedrock_is_configured and bedrock_base_url is None:
        raise ConfigValueError(
            "CHULK_BEDROCK_BASE_URL",
            "CHULK_BEDROCK_BASE_URL or CHULK_BASE_URL is required when Bedrock "
            "is configured as a primary or fallback provider",
        )
    runtime_dir = runtime_override or _resolve_config_path(env.get("CHULK_RUNTIME_DIR") or ".chulk", base=project_root)
    mcp_config_path = runtime_dir / "mcp.json"
    mcp_servers = load_mcp_servers(mcp_config_path, env)

    return Config(
        project_root=project_root,
        runtime_dir=runtime_dir,
        mcp_config_path=mcp_config_path,
        mcp_servers=mcp_servers if options.mcp_servers is None else tuple(options.mcp_servers),
        llm_provider=llm_provider,
        model=model,
        openai_api_key=_env_value(env, "OPENAI_API_KEY", options.openai_api_key) or None,
        deepseek_api_key=_env_value(env, "CHULK_DEEPSEEK_API_KEY", options.deepseek_api_key) or env.get("DEEPSEEK_API_KEY") or None,
        deepseek_base_url=_env_value(env, "CHULK_DEEPSEEK_BASE_URL", options.deepseek_base_url) or DEFAULT_DEEPSEEK_BASE_URL,
        moonshot_api_key=_env_value(env, "CHULK_MOONSHOT_API_KEY", options.moonshot_api_key) or env.get("MOONSHOT_API_KEY") or None,
        moonshot_base_url=_env_value(env, "CHULK_MOONSHOT_BASE_URL", options.moonshot_base_url) or DEFAULT_MOONSHOT_BASE_URL,
        local_api_key=_env_value(env, "CHULK_LOCAL_API_KEY", options.local_api_key) or None,
        local_base_url=_env_value(env, "CHULK_LOCAL_BASE_URL", options.local_base_url) or DEFAULT_LOCAL_BASE_URL,
        local_context_window_tokens=_local_context_window_tokens(env, options.local_context_window_tokens),
        openai_compatible_api_key=_env_value(env, "CHULK_OPENAI_COMPATIBLE_API_KEY", options.openai_compatible_api_key) or None,
        openai_compatible_base_url=_env_value(env, "CHULK_OPENAI_COMPATIBLE_BASE_URL", options.openai_compatible_base_url) or None,
        openrouter_api_key=_env_value(env, "CHULK_OPENROUTER_API_KEY", options.openrouter_api_key) or env.get("OPENROUTER_API_KEY") or None,
        openrouter_base_url=_env_value(env, "CHULK_OPENROUTER_BASE_URL", options.openrouter_base_url) or DEFAULT_OPENROUTER_BASE_URL,
        anthropic_api_key=_env_value(env, "CHULK_ANTHROPIC_API_KEY", options.anthropic_api_key) or env.get("ANTHROPIC_API_KEY") or None,
        anthropic_base_url=_env_value(env, "CHULK_ANTHROPIC_BASE_URL", options.anthropic_base_url) or None,
        bedrock_api_key=(
            _env_value(env, "CHULK_BEDROCK_API_KEY", options.bedrock_api_key)
            or env.get("BEDROCK_API_KEY")
            or env.get("AWS_BEARER_TOKEN_BEDROCK")
            or None
        ),
        bedrock_base_url=bedrock_base_url,
        gemini_api_key=_first_nonblank(
            _env_value(env, "CHULK_GEMINI_API_KEY", options.gemini_api_key),
            env.get("GEMINI_API_KEY"), env.get("GOOGLE_API_KEY"),
        ),
        gemini_base_url=_first_nonblank(_env_value(env, "CHULK_GEMINI_BASE_URL", options.gemini_base_url)),
        llm_fallback_providers=(
            llm_fallback_providers if options.llm_fallback_providers is None else tuple(options.llm_fallback_providers)
        ),
        history_limit=_env_int(env, "CHULK_HISTORY_LIMIT", 20, options.history_limit),
        max_tool_calls_per_turn=_env_int(env, "CHULK_MAX_TOOL_CALLS_PER_TURN", 5, options.max_tool_calls_per_turn),
        max_skills_per_turn=_env_int(env, "CHULK_MAX_SKILLS_PER_TURN", DEFAULT_MAX_SKILLS_PER_TURN, options.max_skills_per_turn),
        max_skill_content_chars=_env_int(env, "CHULK_MAX_SKILL_CONTENT_CHARS", DEFAULT_MAX_SKILL_CONTENT_CHARS, options.max_skill_content_chars),
        shell_timeout_seconds=_env_int(env, "CHULK_SHELL_TIMEOUT_SECONDS", 10, options.shell_timeout_seconds),
        llm_timeout_seconds=_env_float(env, "CHULK_LLM_TIMEOUT_SECONDS", 60.0, options.llm_timeout_seconds),
        llm_max_retries=_env_int(env, "CHULK_LLM_MAX_RETRIES", 2, options.llm_max_retries),
        trace_max_prompt_chars=_env_int(env, "CHULK_TRACE_MAX_PROMPT_CHARS", DEFAULT_TRACE_MAX_PROMPT_CHARS, options.trace_max_prompt_chars),
        max_observation_chars=_env_int(env, "CHULK_MAX_OBSERVATION_CHARS", DEFAULT_MAX_OBSERVATION_CHARS, options.max_observation_chars),
        max_tool_stdout_chars=_env_int(env, "CHULK_MAX_TOOL_STDOUT_CHARS", DEFAULT_MAX_TOOL_STDOUT_CHARS, options.max_tool_stdout_chars),
        max_tool_stderr_chars=_env_int(env, "CHULK_MAX_TOOL_STDERR_CHARS", DEFAULT_MAX_TOOL_STDERR_CHARS, options.max_tool_stderr_chars),
        max_reflection_attempts=_env_int(
            env,
            "CHULK_MAX_REFLECTION_ATTEMPTS",
            DEFAULT_MAX_REFLECTION_ATTEMPTS, options.max_reflection_attempts, minimum=0,
        ),
        permission_profile=_permission_profile(env, process_env, project_root, overrides),
        store_path=(
            _resolve_config_path(options.store_path, base=project_root)
            if options.store_path is not None else (runtime_dir if sdk else project_root / "chulk") / "store.sqlite"
        ),
        traces_dir=(
            _resolve_config_path(options.traces_dir, base=project_root)
            if options.traces_dir is not None else (runtime_dir if sdk else project_root) / "traces"
        ),
        skills_dir=(skills_dir := (
            _resolve_config_path(options.skills_dir, base=project_root)
            if options.skills_dir is not None else runtime_dir / "skills"
        )),
        skills_dirs=_default_skills_dirs(skills_dir, resolve=sdk),
        memory_retention_policy=options.memory_retention_policy,
    )


def resolve_cli_environment(
    environ: Mapping[str, str] | None = None,
    *,
    cwd: Path | str | None = None,
) -> dict[str, str]:
    """Return CLI environment values with one explicit project root."""
    env = dict(os.environ if environ is None else environ)
    if not env.get("CHULK_PROJECT_ROOT"):
        project_root = Path.cwd() if cwd is None else Path(cwd)
        env["CHULK_PROJECT_ROOT"] = str(project_root.expanduser().resolve())
    return env


def load_cli_config(
    environ: Mapping[str, str] | None = None,
    *,
    cwd: Path | str | None = None,
) -> Config:
    """Load CLI configuration relative to the current project directory."""
    return load_config(resolve_cli_environment(environ, cwd=cwd))


def bundled_skills_dir() -> Path:
    """Return the installed path for Chulk's bundled skill playbooks."""
    return Path(__file__).resolve().parent / "skills" / "bundled"


def _default_skills_dirs(project_skills_dir: Path, *, resolve: bool = False) -> tuple[Path, ...]:
    paths = (bundled_skills_dir(), project_skills_dir)
    return tuple(dict.fromkeys(path.resolve() for path in paths)) if resolve else paths


def _configured_model(env: Mapping[str, str], provider: str, override: str | None) -> str:
    configured = (_env_value(env, "CHULK_MODEL", override) or "").strip()
    if configured:
        return configured
    default = _default_model_for_provider(provider)
    if default is None:
        raise ConfigValueError(
            "CHULK_MODEL",
            f"CHULK_MODEL is required when CHULK_LLM_PROVIDER={provider}",
        )
    return default


def _default_model_for_provider(provider: str) -> str | None:
    return {
        "openai": DEFAULT_MODEL,
        "deepseek": DEFAULT_DEEPSEEK_MODEL,
        "moonshot": DEFAULT_MOONSHOT_MODEL,
        "local": DEFAULT_LOCAL_MODEL,
    }.get(provider)


def _bedrock_base_url(env: Mapping[str, str], override: str | None) -> str | None:
    value = _env_value(env, "CHULK_BEDROCK_BASE_URL", override) or env.get("CHULK_BASE_URL")
    if value is None or not value.strip():
        return None
    return value.strip()


def _first_nonblank(*values: str | None) -> str | None:
    for value in values:
        if value is not None and value.strip():
            return value.strip()
    return None


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
        if not provider:
            raise ValueError("CHULK_LLM_FALLBACK_PROVIDERS contains an empty provider name")
        if provider not in SUPPORTED_LLM_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
            raise ValueError(f"CHULK_LLM_FALLBACK_PROVIDERS must use providers from: {supported}")
        model = raw_model.strip() if separator else _default_model_for_provider(provider)
        if model is None:
            raise ValueError(
                f"CHULK_LLM_FALLBACK_PROVIDERS entries for {provider} must include an explicit model"
            )
        if not model:
            raise ValueError("CHULK_LLM_FALLBACK_PROVIDERS entries with ':' must include a model")

        key = (provider, model.lower())
        if key in seen:
            continue
        seen.add(key)
        providers.append(LLMFallbackProviderConfig(provider=provider, model=model))

    return tuple(providers)


def _project_root_override(env: dict[str, str], value: str | Path | None) -> Path:
    if value is not None:
        return Path(value).resolve()
    configured = _config_key_value(Path.cwd(), env, "CHULK_PROJECT_ROOT")
    if configured is not None:
        return _resolve_config_path(configured, base=Path.cwd())
    return Path.cwd().resolve()


def _runtime_dir_override(project_root: Path, env: dict[str, str], value: str | Path | None) -> Path | None:
    if value is not None:
        return _resolve_config_path(value, base=project_root)
    configured = _config_key_value(project_root, env, "CHULK_RUNTIME_DIR")
    if configured is None:
        return None
    return _resolve_config_path(configured, base=project_root)


def _config_key_value(project_root: Path, env: dict[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is not None and value != "":
        return value
    for dotenv_key, dotenv_value in _iter_dotenv(project_root / ".env"):
        if dotenv_key == key:
            return dotenv_value or None
    return None


def _permission_profile(
    env: Mapping[str, str], process_env: dict[str, str], project_root: Path, overrides: AgentConfig | None,
) -> str:
    override = overrides.permission_profile if overrides is not None else None
    profile = normalize_permission_profile(_env_value(env, "CHULK_PERMISSION_PROFILE", override))
    if overrides is not None:
        configured = override or _config_key_value(project_root, process_env, "CHULK_PERMISSION_PROFILE")
        if override is None and configured is None:
            return "read-only"
    return profile
