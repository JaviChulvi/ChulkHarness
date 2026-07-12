"""Public SDK configuration values and builders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any

from chulk.capabilities import Capabilities, MemoryMode
from chulk.config import (
    Config,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MODEL,
    LLMFallbackProviderConfig,
    bundled_skills_dir,
    load_config,
)
from chulk.mcp import MCPServerConfig, build_mcp_server_config


SDK_DEFAULT_RUNTIME_DIR = ".chulk"
SDK_DEFAULT_PERMISSION_PROFILE = "read-only"


@dataclass(frozen=True)
class AgentPreset:
    """Reusable collection of prompt, tools, skills, and default behavior."""

    system_prompt: str | None = None
    tools: tuple[object, ...] = field(default_factory=tuple)
    skills: tuple[object, ...] = field(default_factory=tuple)

    @classmethod
    def chat(cls, *, system_prompt: str | None = None) -> "AgentPreset":
        """Return a no-tool, no-skill preset for plain chat embedding."""
        return cls(system_prompt=system_prompt, tools=(), skills=())


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
    local_api_key: str | None = None
    local_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    openai_compatible_base_url: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
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

    def __post_init__(self) -> None:
        if self.mcp_servers is not None:
            object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers))
        if self.llm_fallback_providers is not None:
            object.__setattr__(self, "llm_fallback_providers", tuple(self.llm_fallback_providers))
        if self.memory_mode is not None:
            base = self.capabilities or Capabilities.read_only()
            object.__setattr__(self, "capabilities", base.with_memory(self.memory_mode))
            object.__setattr__(self, "memory_mode", self.capabilities.memory)

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
    def local(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "AgentConfig":
        """Create config for local OpenAI-compatible providers."""
        values = dict(kwargs)
        if api_key is not None:
            values["local_api_key"] = api_key
        if base_url is not None:
            values["local_base_url"] = base_url
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

    def to_config(self) -> Config:
        """Build the internal runtime config."""
        env = dict(os.environ)
        project_root = _project_root_override(env, self.project_root)
        _set_env(env, "CHULK_PROJECT_ROOT", project_root)
        runtime_dir = _runtime_dir_override(project_root, env, self.runtime_dir)
        _set_env(env, "CHULK_RUNTIME_DIR", runtime_dir)
        _set_env(env, "CHULK_LLM_PROVIDER", self.provider)
        _set_env(env, "CHULK_MODEL", self.model)
        _set_env(env, "OPENAI_API_KEY", self.openai_api_key)
        _set_env(env, "CHULK_DEEPSEEK_API_KEY", self.deepseek_api_key)
        _set_env(env, "CHULK_DEEPSEEK_BASE_URL", self.deepseek_base_url)
        _set_env(env, "CHULK_LOCAL_API_KEY", self.local_api_key)
        _set_env(env, "CHULK_LOCAL_BASE_URL", self.local_base_url)
        _set_env(env, "CHULK_OPENAI_COMPATIBLE_API_KEY", self.openai_compatible_api_key)
        _set_env(env, "CHULK_OPENAI_COMPATIBLE_BASE_URL", self.openai_compatible_base_url)
        _set_env(env, "CHULK_OPENROUTER_API_KEY", self.openrouter_api_key)
        _set_env(env, "CHULK_OPENROUTER_BASE_URL", self.openrouter_base_url)
        _set_env(env, "CHULK_ANTHROPIC_API_KEY", self.anthropic_api_key)
        _set_env(env, "CHULK_ANTHROPIC_BASE_URL", self.anthropic_base_url)
        _set_env(env, "CHULK_PERMISSION_PROFILE", self.permission_profile)
        _set_env(env, "CHULK_HISTORY_LIMIT", self.history_limit)
        _set_env(env, "CHULK_MAX_TOOL_CALLS_PER_TURN", self.max_tool_calls_per_turn)
        _set_env(env, "CHULK_MAX_SKILLS_PER_TURN", self.max_skills_per_turn)
        _set_env(env, "CHULK_MAX_SKILL_CONTENT_CHARS", self.max_skill_content_chars)
        _set_env(env, "CHULK_SHELL_TIMEOUT_SECONDS", self.shell_timeout_seconds)
        _set_env(env, "CHULK_LLM_TIMEOUT_SECONDS", self.llm_timeout_seconds)
        _set_env(env, "CHULK_LLM_MAX_RETRIES", self.llm_max_retries)
        _set_env(env, "CHULK_TRACE_MAX_PROMPT_CHARS", self.trace_max_prompt_chars)
        _set_env(env, "CHULK_MAX_OBSERVATION_CHARS", self.max_observation_chars)
        _set_env(env, "CHULK_MAX_TOOL_STDOUT_CHARS", self.max_tool_stdout_chars)
        _set_env(env, "CHULK_MAX_TOOL_STDERR_CHARS", self.max_tool_stderr_chars)
        _set_env(env, "CHULK_MAX_REFLECTION_ATTEMPTS", self.max_reflection_attempts)
        config = load_config(env)
        has_configured_permission_profile = _config_key_has_value(project_root, env, "CHULK_PERMISSION_PROFILE")
        permission_profile = (
            config.permission_profile
            if self.permission_profile is not None or has_configured_permission_profile
            else SDK_DEFAULT_PERMISSION_PROFILE
        )
        runtime_dir = runtime_dir or config.runtime_dir
        store_path = (
            _resolve_path(self.store_path, base=project_root)
            if self.store_path is not None
            else runtime_dir / "store.sqlite"
        )
        traces_dir = (
            _resolve_path(self.traces_dir, base=project_root)
            if self.traces_dir is not None
            else runtime_dir / "traces"
        )
        skills_dir = (
            _resolve_path(self.skills_dir, base=project_root)
            if self.skills_dir is not None
            else runtime_dir / "skills"
        )
        updates: dict[str, object] = {
            "project_root": project_root,
            "runtime_dir": runtime_dir,
            "store_path": store_path,
            "traces_dir": traces_dir,
            "skills_dir": skills_dir,
            "skills_dirs": _skills_dirs(skills_dir),
            "mcp_config_path": config.mcp_config_path,
            "permission_profile": permission_profile,
        }
        if self.mcp_servers is not None:
            updates["mcp_servers"] = tuple(self.mcp_servers)
        if self.llm_fallback_providers is not None:
            updates["llm_fallback_providers"] = tuple(self.llm_fallback_providers)
        return replace(config, **updates)


class MCP:
    """Public MCP server builders."""

    @staticmethod
    def streamable_http(
        *,
        label: str,
        server_url: str,
        server_description: str = "",
        allowed_tools: Iterable[str] = (),
        authorization: str | None = None,
        authorization_env: str | None = None,
        approval: str = "always",
        defer_loading: bool = False,
    ) -> MCPServerConfig:
        return build_mcp_server_config(
            label=label,
            server_url=server_url,
            server_description=server_description,
            allowed_tools=allowed_tools,
            authorization=authorization,
            authorization_env=authorization_env,
            approval=approval,
            defer_loading=defer_loading,
        )


def coerce_config(config: Config | AgentConfig | None) -> Config:
    if config is None:
        return AgentConfig.from_env().to_config()
    if isinstance(config, AgentConfig):
        return config.to_config()
    return config


def ensure_chat_kwargs(kwargs: dict[str, Any]) -> None:
    disallowed = [name for name in ("preset", "tools", "skills") if name in kwargs and kwargs[name] is not None]
    if disallowed:
        joined = ", ".join(disallowed)
        raise ValueError(f"ChatAgent does not accept {joined}; use Agent for configured tools or skills")


def _set_env(env: dict[str, str], key: str, value: object) -> None:
    if value is not None:
        env[key] = str(value)


def _project_root_override(env: dict[str, str], value: str | Path | None) -> Path:
    if value is not None:
        return _resolve_project_root(value)
    configured = _config_key_value(Path.cwd(), env, "CHULK_PROJECT_ROOT")
    if configured is not None:
        return _resolve_path(configured, base=Path.cwd())
    return Path.cwd().resolve()


def _runtime_dir_override(project_root: Path, env: dict[str, str], value: str | Path | None) -> Path | None:
    if value is not None:
        return _resolve_path(value, base=project_root)
    configured = _config_key_value(project_root, env, "CHULK_RUNTIME_DIR")
    if configured is None:
        return None
    return _resolve_path(configured, base=project_root)


def _config_key_has_value(project_root: Path, env: dict[str, str], key: str) -> bool:
    return _config_key_value(project_root, env, key) is not None


def _config_key_value(project_root: Path, env: dict[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is not None and value != "":
        return value
    return _dotenv_key_value(project_root / ".env", key)


def _dotenv_key_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        raw_key, raw_value = line.split("=", 1)
        if raw_key.strip() != key:
            continue
        value = raw_value.strip().strip("'\"")
        return value or None
    return None


def _resolve_project_root(value: str | Path | None) -> Path:
    return (Path.cwd() if value is None else Path(value)).resolve()


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _skills_dirs(project_skills_dir: Path) -> tuple[Path, ...]:
    ordered = [bundled_skills_dir().resolve(), project_skills_dir.resolve()]
    unique: list[Path] = []
    for path in ordered:
        if path not in unique:
            unique.append(path)
    return tuple(unique)


__all__ = ["AgentConfig", "AgentPreset", "MCP"]
