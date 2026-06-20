"""Skill registry primitives and public skill references."""

from dataclasses import dataclass
from pathlib import Path

from chulk.skills.registry import Skill, SkillRegistry, SkillSelection


def bundled_skills_dir() -> Path:
    """Return the installed directory containing Chulk's bundled skill playbooks."""
    return Path(__file__).resolve().parent / "bundled"


@dataclass(frozen=True)
class SkillRef:
    """Reference to a skill that should be available and pinned for an agent."""

    name: str | None = None
    skill_path: Path | None = None

    def register(self, registry: SkillRegistry) -> str | None:
        if self.skill_path is not None:
            skill = registry.register_path(self.skill_path)
            return skill.name
        if self.name is None:
            raise ValueError("SkillRef must include name or skill_path")
        if registry.get_skill(self.name) is None:
            return None
        return self.name


@dataclass(frozen=True)
class SkillDirectoryRef:
    """Reference to an additional directory of selectable skill playbooks."""

    skills_dir: Path

    def register(self, registry: SkillRegistry) -> str | None:
        registry.register_directory(self.skills_dir)
        return None


@dataclass(frozen=True)
class SkillAllowlistRef:
    """Names of catalog skills that may be selected for one agent."""

    names: tuple[str, ...]


@dataclass(frozen=True)
class SkillPinRef:
    """Names of catalog skills that should always be loaded for one agent."""

    names: tuple[str, ...]


def path(skill_path: str | Path) -> SkillRef:
    """Pin one skill from a SKILL.md path or a directory containing SKILL.md."""
    return SkillRef(skill_path=Path(skill_path))


def from_dir(skills_dir: str | Path) -> SkillDirectoryRef:
    """Register an additional directory of skill folders for selection."""
    return SkillDirectoryRef(Path(skills_dir))


def only(*names: str) -> SkillAllowlistRef:
    """Allow automatic selection only from these catalog skill names."""
    return SkillAllowlistRef(tuple(names))


def pin(*names: str) -> SkillPinRef:
    """Always load these catalog skill names for the agent."""
    return SkillPinRef(tuple(names))


files = SkillRef(name="files")
shell = SkillRef(name="shell")
memory = SkillRef(name="memory")


__all__ = [
    "Skill",
    "SkillAllowlistRef",
    "SkillDirectoryRef",
    "SkillPinRef",
    "SkillRef",
    "SkillRegistry",
    "SkillSelection",
    "bundled_skills_dir",
    "files",
    "from_dir",
    "memory",
    "only",
    "path",
    "pin",
    "shell",
]
