"""Skill registry primitives and public skill references."""

from dataclasses import dataclass
from pathlib import Path

from chulk.skills.registry import Skill, SkillRegistry, SkillSelection


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


def path(skill_path: str | Path) -> SkillRef:
    """Pin one skill from a SKILL.md path or a directory containing SKILL.md."""
    return SkillRef(skill_path=Path(skill_path))


def from_dir(skills_dir: str | Path) -> SkillDirectoryRef:
    """Register an additional directory of skill folders for selection."""
    return SkillDirectoryRef(Path(skills_dir))


files = SkillRef(name="files")
shell = SkillRef(name="shell")
memory = SkillRef(name="memory")


__all__ = [
    "Skill",
    "SkillDirectoryRef",
    "SkillRef",
    "SkillRegistry",
    "SkillSelection",
    "files",
    "from_dir",
    "memory",
    "path",
    "shell",
]
