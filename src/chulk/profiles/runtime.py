"""Profile-aware runtime configuration and assembly."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from chulk.config import Config, bundled_skills_dir
from chulk.execution import ExecutionBackend
from chulk.profiles.models import AgentProfile, DEFAULT_PROFILE_ID
from chulk.profiles.store import SQLiteProfileStore


ExecutionBackendFactory = Callable[[AgentProfile, Config], ExecutionBackend]


@dataclass(frozen=True, slots=True)
class ResolvedProfileRuntime:
    """A profile and its isolated effective runtime configuration."""

    profile: AgentProfile
    config: Config


class ProfileRuntimeFactory:
    """Resolve CLI selections and create profile-owned agent runtimes."""

    def __init__(
        self,
        base_config: Config,
        *,
        profile_store: SQLiteProfileStore | None = None,
        backend_factories: dict[str, ExecutionBackendFactory] | None = None,
    ) -> None:
        self.base_config = base_config
        self.profile_store = profile_store or SQLiteProfileStore(
            base_config.runtime_dir / "control.sqlite",
            base_config=base_config,
        )
        self.backend_factories = dict(backend_factories or {})

    def resolve(self, profile_id: str | None = None) -> ResolvedProfileRuntime:
        """Resolve an explicit profile, defaulting safely to the implicit profile."""
        profile = self.profile_store.get(profile_id or DEFAULT_PROFILE_ID).profile
        return self._resolve_profile(profile)

    def resolve_cli(self, profile_id: str | None = None) -> ResolvedProfileRuntime:
        """Resolve an override or the owner-local CLI selection."""
        profile = self.profile_store.resolve(profile_id).profile
        return self._resolve_profile(profile)

    def _resolve_profile(self, profile: AgentProfile) -> ResolvedProfileRuntime:
        if profile.implicit:
            return ResolvedProfileRuntime(profile=profile, config=replace(self.base_config, profile_id=profile.id))
        project_skills = profile.project_root / ".chulk" / "skills"
        config = replace(
            self.base_config,
            profile_id=profile.id,
            project_root=profile.project_root,
            runtime_dir=profile.runtime_dir,
            skills_dir=project_skills,
            skills_dirs=(bundled_skills_dir(), project_skills),
            store_path=profile.store_path,
            traces_dir=profile.traces_dir,
            mcp_config_path=profile.project_root / ".chulk" / "mcp.json",
            permission_profile=profile.permission_profile,
        )
        return ResolvedProfileRuntime(profile=profile, config=config)

    def create_agent(self, profile_id: str | None = None, **kwargs: Any):
        """Create an agent after applying host-owned profile restrictions."""
        from chulk._runtime.request import AgentAssemblyRequest
        from chulk.runtime import create_agent

        resolved = self.resolve(profile_id)
        profile = resolved.profile
        kwargs["memory_namespace"] = profile.memory_namespace
        if profile.system_prompt is not None:
            kwargs["system_prompt"] = profile.system_prompt
        if profile.allowed_mcp_servers is not None:
            allowed = set(profile.allowed_mcp_servers)
            requested_servers = tuple(
                kwargs.get("mcp_servers", resolved.config.mcp_servers)
            )
            kwargs["mcp_servers"] = tuple(
                server for server in requested_servers if server.label in allowed
            )
        requested_skills = kwargs.get("allowed_skill_names")
        if profile.allowed_skills is not None:
            profile_skills = set(profile.allowed_skills)
            kwargs["allowed_skill_names"] = (
                profile.allowed_skills
                if requested_skills is None
                else tuple(name for name in requested_skills if name in profile_skills)
            )
        supplied_backend = kwargs.get("execution_backend")
        if supplied_backend is not None:
            if getattr(supplied_backend, "name", None) != profile.execution_backend_id:
                raise ValueError(
                    "execution backend does not match the profile-owned backend selection"
                )
        elif profile.execution_backend_id != "host":
            try:
                backend_factory = self.backend_factories[profile.execution_backend_id]
            except KeyError as exc:
                raise ValueError(
                    f"execution backend {profile.execution_backend_id!r} is not configured by the host"
                ) from exc
            kwargs["execution_backend"] = backend_factory(profile, resolved.config)
        kwargs["profile_id"] = profile.id
        return create_agent(AgentAssemblyRequest(config=resolved.config, **kwargs))


def profile_control_path(config: Config) -> Path:
    """Return the owner-only control database path for a base configuration."""
    return config.runtime_dir / "control.sqlite"


__all__ = [
    "ExecutionBackendFactory",
    "ProfileRuntimeFactory",
    "ResolvedProfileRuntime",
    "profile_control_path",
]
