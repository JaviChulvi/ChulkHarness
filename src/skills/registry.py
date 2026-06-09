"""Lazy-loaded skill registry."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """Procedural instructions that can be injected into the prompt."""

    name: str
    description: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    loaded_content: str | None = None


class SkillRegistry:
    """Registry that loads skill metadata first and full skill content later."""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def load_content(self, name: str) -> str:
        skill = self._skills[name]
        if skill.loaded_content is None:
            skill.loaded_content = skill.path.read_text(encoding="utf-8")
        return skill.loaded_content
