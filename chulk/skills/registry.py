"""Lazy-loaded skill registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from chulk.skills.manifest import (
    SkillManifest,
    load_skill_package,
    resolve_skill_resource,
    split_skill_front_matter,
)

DEFAULT_MAX_SKILLS = 3
DEFAULT_MAX_SKILL_CONTENT_CHARS = 4000

DEFAULT_SKILL_KEYWORDS: dict[str, set[str]] = {
    "files": {
        "create",
        "directory",
        "edit",
        "editar",
        "escribe",
        "file",
        "files",
        "archivo",
        "archivos",
        "lee",
        "organize",
        "patch",
        "path",
        "read",
        "write",
    },
    "memory": {
        "delete",
        "durable",
        "forget",
        "guarda",
        "guardar",
        "memory",
        "memoria",
        "prefer",
        "prefiero",
        "preference",
        "recall",
        "recuerda",
        "recuperar",
        "remember",
        "retrieve",
        "save",
        "summarize",
        "update",
    },
    "shell": {
        "bash",
        "cli",
        "cmd",
        "comando",
        "comandos",
        "command",
        "commands",
        "ejecuta",
        "ejecutar",
        "execute",
        "run",
        "shell",
        "stderr",
        "stdout",
        "terminal",
        "zsh",
    },
}

KEYWORD_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "before",
    "can",
    "for",
    "from",
    "how",
    "into",
    "not",
    "only",
    "should",
    "the",
    "this",
    "use",
    "user",
    "when",
    "with",
}


@dataclass
class Skill:
    """Procedural instructions that can be injected into the prompt."""

    name: str
    description: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    loaded_content: str | None = None
    manifest: SkillManifest | None = None
    digest: str | None = None
    root: Path | None = None
    loaded_resources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkillSelection:
    """A selected skill plus the evidence used to select it."""

    skill: Skill
    score: int
    matched_keywords: list[str]


class SkillRegistry:
    """Registry that loads skill metadata first and full skill content later."""

    def __init__(
        self,
        skills_dir: Path | str,
        *,
        skills_dirs: Iterable[Path | str] | None = None,
        max_skills: int = DEFAULT_MAX_SKILLS,
        max_content_chars: int = DEFAULT_MAX_SKILL_CONTENT_CHARS,
    ) -> None:
        if max_skills < 1:
            raise ValueError("max_skills must be greater than zero")
        if max_content_chars < 1:
            raise ValueError("max_content_chars must be greater than zero")
        self.skills_dir = Path(skills_dir)
        self.skills_dirs = (
            tuple(Path(path) for path in skills_dirs)
            if skills_dirs is not None
            else (self.skills_dir,)
        )
        self.max_skills = max_skills
        self.max_content_chars = max_content_chars
        self._skills: dict[str, Skill] = {}

    def load_metadata(self) -> None:
        """Scan skill folders and register metadata without loading prompt content."""
        self._skills = {}
        for skills_dir in self.skills_dirs:
            self._register_directory(skills_dir, replace=True)

    def register(self, skill: Skill, *, replace: bool = False) -> None:
        """Register one skill metadata record."""
        clean_name = _normalize_skill_name(skill.name)
        if clean_name in self._skills and not replace:
            raise ValueError(f"Skill already registered: {clean_name}")
        skill.name = clean_name
        skill.keywords = _normalize_keywords([skill.name, *skill.keywords, *DEFAULT_SKILL_KEYWORDS.get(skill.name, [])])
        self._skills[clean_name] = skill

    def register_path(self, path: Path | str) -> Skill:
        """Register one skill from a SKILL.md file or a directory containing one."""
        skill_path = Path(path)
        if skill_path.is_dir():
            skill_path = skill_path / "SKILL.md"
        skill = _skill_from_markdown(skill_path)
        self.register(skill)
        return skill

    def register_directory(self, skills_dir: Path | str) -> list[Skill]:
        """Register all skills under a directory of skill folders."""
        return self._register_directory(skills_dir, replace=False)

    def _register_directory(self, skills_dir: Path | str, *, replace: bool) -> list[Skill]:
        root = Path(skills_dir)
        registered: list[Skill] = []
        if not root.exists():
            return registered
        for skill_path in sorted(root.glob("*/SKILL.md")):
            skill = _skill_from_markdown(skill_path)
            self.register(skill, replace=replace)
            registered.append(skill)
        return registered

    def clear(self) -> None:
        """Remove all registered skill metadata."""
        self._skills = {}

    def restrict_to(self, names: Iterable[str]) -> None:
        """Keep only registered skill metadata with the given names."""
        allowed_names = {_normalize_skill_name(name) for name in names}
        self._skills = {name: skill for name, skill in self._skills.items() if name in allowed_names}

    def list_skills(self) -> list[Skill]:
        """Return registered skill metadata sorted by name."""
        return [self._skills[name] for name in sorted(self._skills)]

    def get_skill(self, name: str) -> Skill | None:
        """Return a registered skill by name."""
        return self._skills.get(_normalize_skill_name(name))

    def select_skills(self, user_request: str, *, limit: int | None = None) -> list[SkillSelection]:
        """Select relevant skills using deterministic keyword matching."""
        request_text = user_request.strip().lower()
        request_terms = _tokenize(user_request)
        if not request_terms and not request_text:
            return []

        selected: list[SkillSelection] = []
        for skill in self.list_skills():
            searchable_terms = set(skill.keywords)
            searchable_terms.update(_tokenize(skill.name))
            searchable_terms.update(_tokenize(skill.description))
            matched = sorted(request_terms & searchable_terms)
            score = len(matched) * 10

            if skill.name in request_text:
                score += 25
                if skill.name not in matched:
                    matched.append(skill.name)

            phrase_hits = [keyword for keyword in skill.keywords if " " in keyword and keyword in request_text]
            if phrase_hits:
                score += len(phrase_hits) * 15
                matched.extend(phrase_hits)

            if score > 0:
                selected.append(
                    SkillSelection(
                        skill=skill,
                        score=score,
                        matched_keywords=sorted(set(matched)),
                    )
                )

        selected.sort(key=lambda selection: (-selection.score, selection.skill.name))
        return selected[: self._selection_limit(limit)]

    def load_selected_skills(self, user_request: str, *, limit: int | None = None) -> list[SkillSelection]:
        """Select relevant skills and lazy-load only those selected skill files."""
        selections = self.select_skills(user_request, limit=limit)
        for selection in selections:
            self.load_content(selection.skill.name)
        return selections

    def load_content(self, name: str) -> str:
        """Progressively load entrypoint and reference text under one budget."""
        skill = self._skills[_normalize_skill_name(name)]
        if skill.loaded_content is None:
            root = skill.root or skill.path.parent.resolve(strict=True)
            manifest = skill.manifest
            entrypoint = (
                resolve_skill_resource(root, manifest.entrypoint)
                if manifest is not None
                else skill.path
            )
            text = entrypoint.read_text(encoding="utf-8")
            if entrypoint == skill.path:
                _front_matter, text = split_skill_front_matter(text)
            content = text.strip()[: self.max_content_chars]
            loaded_resources = [entrypoint.relative_to(root).as_posix()]
            for reference in manifest.references if manifest is not None else ():
                if len(content) >= self.max_content_chars:
                    break
                reference_path = resolve_skill_resource(root, reference)
                reference_text = reference_path.read_text(encoding="utf-8").strip()
                header = f"\n\nReference: {reference}\n"
                remaining = self.max_content_chars - len(content)
                addition = f"{header}{reference_text}"[:remaining]
                if addition:
                    content += addition
                    loaded_resources.append(reference)
            skill.loaded_content = content
            skill.loaded_resources = loaded_resources
        return skill.loaded_content

    def _selection_limit(self, limit: int | None) -> int:
        if limit is None:
            return self.max_skills
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        return min(limit, self.max_skills)


def _skill_from_markdown(path: Path) -> Skill:
    package = load_skill_package(path)
    manifest = package.manifest
    text = package.manifest_path.read_text(encoding="utf-8")
    _front_matter, body = split_skill_front_matter(text)
    name = _normalize_skill_name(manifest.name)
    description = manifest.description
    keywords = _normalize_keywords(
        [
            name,
            *manifest.keywords,
            *manifest.tags,
            *DEFAULT_SKILL_KEYWORDS.get(name, []),
            *_tokenize(description),
            *_heading_terms(body),
        ]
    )
    return Skill(
        name=name,
        description=description,
        path=package.manifest_path,
        metadata=dict(manifest.extensions),
        keywords=keywords,
        manifest=manifest,
        digest=package.digest,
        root=package.root,
    )


def _heading_terms(body: str) -> list[str]:
    terms: list[str] = []
    for line in body.splitlines():
        if line.startswith("#"):
            terms.extend(_tokenize(line.lstrip("#").strip()))
    return terms


def _normalize_skill_name(name: str) -> str:
    clean_name = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower())
    clean_name = clean_name.strip("-_")
    if not clean_name:
        raise ValueError("Skill name cannot be empty")
    return clean_name


def _normalize_keywords(keywords: list[Any]) -> list[str]:
    normalized: set[str] = set()
    for keyword in keywords:
        if isinstance(keyword, str):
            if " " in keyword.strip():
                clean_phrase = " ".join(_tokenize(keyword))
                if clean_phrase:
                    normalized.add(clean_phrase)
            normalized.update(_tokenize(keyword))
            continue
        normalized.update(_tokenize(str(keyword)))
    return sorted(normalized)


def _tokenize(text: str) -> set[str]:
    tokens = set()
    for raw_token in re.findall(r"[a-zA-Z0-9_/-]+", text.lower()):
        token = raw_token.strip("_-/")
        if len(token) < 3 or token in KEYWORD_STOPWORDS:
            continue
        tokens.add(token)
    return tokens
