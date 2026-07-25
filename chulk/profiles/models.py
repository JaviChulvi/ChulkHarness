"""Immutable agent-profile models and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Literal

from chulk.tools.permissions import DEFAULT_PERMISSION_PROFILE, normalize_permission_profile


DEFAULT_PROFILE_ID = "default"
DEFAULT_MODEL_PROFILE_ID = "default"
DEFAULT_EXECUTION_BACKEND_ID = "host"
CredentialSource = Literal["environment", "keyring", "host"]

_PROFILE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_REFERENCE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class CredentialRef:
    """A reference to host-owned secret storage, never a credential value."""

    name: str
    source: CredentialSource = "environment"

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("credential reference name cannot be empty")
        if self.source == "environment":
            if _ENVIRONMENT_NAME_PATTERN.fullmatch(name) is None:
                raise ValueError("environment credential references must be environment variable names")
        elif self.source not in {"keyring", "host"}:
            raise ValueError("credential source must be environment, keyring, or host")
        elif _REFERENCE_PATTERN.fullmatch(name) is None:
            raise ValueError("credential reference name contains unsupported characters")
        object.__setattr__(self, "name", name)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "source": self.source}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CredentialRef:
        return cls(name=str(value["name"]), source=str(value.get("source", "environment")))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AuxiliaryModelProfiles:
    """Optional narrower model-profile references for specialized work."""

    summarization: str | None = None
    memory_review: str | None = None
    learning_review: str | None = None
    skill_routing: str | None = None
    vision: str | None = None
    background: str | None = None

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _validate_reference(value, field_name))

    def to_dict(self) -> dict[str, str | None]:
        return {
            "summarization": self.summarization,
            "memory_review": self.memory_review,
            "learning_review": self.learning_review,
            "skill_routing": self.skill_routing,
            "vision": self.vision,
            "background": self.background,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> AuxiliaryModelProfiles:
        data = value or {}
        return cls(
            summarization=_optional_string(data.get("summarization")),
            memory_review=_optional_string(data.get("memory_review")),
            learning_review=_optional_string(data.get("learning_review")),
            skill_routing=_optional_string(data.get("skill_routing")),
            vision=_optional_string(data.get("vision")),
            background=_optional_string(data.get("background")),
        )


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """One durable authority boundary for an agent runtime."""

    id: str
    project_root: Path
    runtime_dir: Path
    store_path: Path
    traces_dir: Path
    memory_namespace: str | None = None
    permission_profile: str = DEFAULT_PERMISSION_PROFILE
    model_profile_id: str = DEFAULT_MODEL_PROFILE_ID
    execution_backend_id: str = DEFAULT_EXECUTION_BACKEND_ID
    allowed_skills: tuple[str, ...] | None = None
    allowed_mcp_servers: tuple[str, ...] | None = None
    credential_refs: tuple[CredentialRef, ...] = ()
    auxiliary_models: AuxiliaryModelProfiles = field(default_factory=AuxiliaryModelProfiles)
    system_prompt: str | None = None
    implicit: bool = False

    def __post_init__(self) -> None:
        profile_id = normalize_profile_id(self.id)
        project_root = _absolute_path(self.project_root, "project_root")
        runtime_dir = _absolute_path(self.runtime_dir, "runtime_dir")
        store_path = _absolute_path(self.store_path, "store_path")
        traces_dir = _absolute_path(self.traces_dir, "traces_dir")
        if store_path == runtime_dir or store_path.name in {"", ".", ".."}:
            raise ValueError("profile store_path must name a database file")
        if self.memory_namespace is not None:
            namespace = self.memory_namespace.strip()
            if not namespace:
                raise ValueError("memory_namespace cannot be blank")
            if len(namespace) > 256:
                raise ValueError("memory_namespace cannot exceed 256 characters")
            object.__setattr__(self, "memory_namespace", namespace)
        model_profile_id = _validate_reference(self.model_profile_id, "model_profile_id")
        execution_backend_id = _validate_reference(
            self.execution_backend_id,
            "execution_backend_id",
        )
        system_prompt = self.system_prompt
        if system_prompt is not None:
            system_prompt = system_prompt.strip()
            if not system_prompt:
                system_prompt = None
            elif "\x00" in system_prompt:
                raise ValueError("system_prompt cannot contain NUL characters")
            elif len(system_prompt) > 50_000:
                raise ValueError("system_prompt cannot exceed 50000 characters")

        object.__setattr__(self, "id", profile_id)
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "runtime_dir", runtime_dir)
        object.__setattr__(self, "store_path", store_path)
        object.__setattr__(self, "traces_dir", traces_dir)
        object.__setattr__(
            self,
            "permission_profile",
            normalize_permission_profile(self.permission_profile),
        )
        object.__setattr__(self, "model_profile_id", model_profile_id)
        object.__setattr__(self, "execution_backend_id", execution_backend_id)
        object.__setattr__(self, "allowed_skills", _normalize_allowlist(self.allowed_skills, "skill"))
        object.__setattr__(
            self,
            "allowed_mcp_servers",
            _normalize_allowlist(self.allowed_mcp_servers, "MCP server"),
        )
        object.__setattr__(self, "credential_refs", tuple(self.credential_refs))
        object.__setattr__(self, "system_prompt", system_prompt)
        if self.implicit and profile_id != DEFAULT_PROFILE_ID:
            raise ValueError("only the default profile may be implicit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_root": str(self.project_root),
            "runtime_dir": str(self.runtime_dir),
            "store_path": str(self.store_path),
            "traces_dir": str(self.traces_dir),
            "memory_namespace": self.memory_namespace,
            "permission_profile": self.permission_profile,
            "model_profile_id": self.model_profile_id,
            "execution_backend_id": self.execution_backend_id,
            "allowed_skills": list(self.allowed_skills) if self.allowed_skills is not None else None,
            "allowed_mcp_servers": (
                list(self.allowed_mcp_servers) if self.allowed_mcp_servers is not None else None
            ),
            "credential_refs": [reference.to_dict() for reference in self.credential_refs],
            "auxiliary_models": self.auxiliary_models.to_dict(),
            "system_prompt": self.system_prompt,
            "implicit": self.implicit,
        }


def normalize_profile_id(value: str) -> str:
    """Normalize and validate a stable profile identifier."""
    profile_id = value.strip().lower()
    if _PROFILE_ID_PATTERN.fullmatch(profile_id) is None:
        raise ValueError(
            "profile id must start with a letter and contain only lowercase "
            "letters, digits, underscores, or hyphens (maximum 63 characters)"
        )
    return profile_id


def _absolute_path(value: Path, field_name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    return path.resolve()


def _validate_reference(value: str, field_name: str) -> str:
    normalized = value.strip()
    if _REFERENCE_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return normalized


def _normalize_allowlist(
    values: tuple[str, ...] | None,
    label: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    normalized = tuple(dict.fromkeys(_validate_reference(value, label) for value in values))
    return tuple(sorted(normalized))


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "AgentProfile",
    "AuxiliaryModelProfiles",
    "CredentialRef",
    "CredentialSource",
    "DEFAULT_EXECUTION_BACKEND_ID",
    "DEFAULT_MODEL_PROFILE_ID",
    "DEFAULT_PROFILE_ID",
    "normalize_profile_id",
]
