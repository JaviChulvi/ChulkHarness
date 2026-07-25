"""Lazy-loaded skill registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
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
_EXACT_SKILL_NAME_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?$"
)


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
    reason: str = "keyword_match"
    stage: str = "keyword"


SkillReranker = Callable[[str, tuple[SkillSelection, ...]], Iterable[str]]


@dataclass(frozen=True)
class SkillRouteDecision:
    """Explain why one skill was selected, rejected, or omitted."""

    skill_name: str
    status: str
    stage: str
    reason: str
    score: int = 0
    matched_keywords: tuple[str, ...] = ()
    version: str | None = None
    digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "status": self.status,
            "stage": self.stage,
            "reason": self.reason,
            "score": self.score,
            "matched_keywords": list(self.matched_keywords),
            "version": self.version,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class SkillRoutingResult:
    """Selections and complete deterministic routing evidence for one request."""

    selections: tuple[SkillSelection, ...]
    decisions: tuple[SkillRouteDecision, ...]
    explicit_skill_names: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[SkillRouteDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.status == "rejected"
            and decision.stage in {"explicit", "include"}
        )


class SkillRegistry:
    """Registry that loads skill metadata first and full skill content later."""

    def __init__(
        self,
        skills_dir: Path | str,
        *,
        skills_dirs: Iterable[Path | str] | None = None,
        max_skills: int = DEFAULT_MAX_SKILLS,
        max_content_chars: int = DEFAULT_MAX_SKILL_CONTENT_CHARS,
        reranker: SkillReranker | None = None,
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
        self.reranker = reranker
        self._skills: dict[str, Skill] = {}
        self._platform: str | None = None
        self._available_tools: frozenset[str] | None = None
        self._capabilities: frozenset[str] | None = None
        self.last_routing_result = SkillRoutingResult((), ())

    def configure_environment(
        self,
        *,
        platform: str | None = None,
        available_tools: Iterable[str] | None = None,
        capabilities: Iterable[str] | None = None,
    ) -> None:
        """Set the runtime environment used for pre-selection visibility checks."""
        self._platform = _normalize_platform(platform or sys.platform)
        self._available_tools = (
            frozenset(available_tools) if available_tools is not None else None
        )
        self._capabilities = (
            frozenset(capabilities) if capabilities is not None else None
        )

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

    def list_visible_skills(self) -> list[Skill]:
        """Return skills that satisfy the configured platform and capabilities."""
        return [
            skill
            for skill in self.list_skills()
            if self.skill_visibility(skill)[0]
        ]

    def get_skill(self, name: str, *, visible_only: bool = False) -> Skill | None:
        """Return a registered skill by name."""
        skill = self._skills.get(_normalize_skill_name(name))
        if skill is not None and visible_only and not self.skill_visibility(skill)[0]:
            return None
        return skill

    def skill_visibility(self, skill: Skill | str) -> tuple[bool, str]:
        """Return whether a skill is usable and the deterministic reason."""
        resolved = self.get_skill(skill) if isinstance(skill, str) else skill
        if resolved is None:
            return False, "unknown_skill"
        manifest = resolved.manifest
        if manifest is None:
            return True, "compatible"
        if (
            self._platform is not None
            and manifest.platforms
            and self._platform not in manifest.platforms
        ):
            return False, f"platform_unavailable:{self._platform}"
        if self._available_tools is not None:
            missing_tools = sorted(
                set(manifest.required_tools) - self._available_tools
            )
            if missing_tools:
                return False, f"missing_tools:{','.join(missing_tools)}"
        if self._capabilities is not None:
            missing_capabilities = sorted(
                set(manifest.required_capabilities) - self._capabilities
            )
            if missing_capabilities:
                return (
                    False,
                    f"missing_capabilities:{','.join(missing_capabilities)}",
                )
            forbidden = sorted(
                set(manifest.forbidden_capabilities) & self._capabilities
            )
            if forbidden:
                return False, f"forbidden_capabilities:{','.join(forbidden)}"
        return True, "compatible"

    def select_skills(self, user_request: str, *, limit: int | None = None) -> list[SkillSelection]:
        """Select relevant skills using deterministic keyword matching."""
        return list(self.route_skills(user_request, limit=limit).selections)

    def route_skills(
        self,
        user_request: str,
        *,
        pinned_names: Iterable[str] = (),
        limit: int | None = None,
    ) -> SkillRoutingResult:
        """Apply visibility, explicit, pinned, keyword, and budget stages."""
        request_text = user_request.strip().lower()
        request_terms = _tokenize(user_request)
        explicit_names = explicit_skill_names(user_request)
        decisions: list[SkillRouteDecision] = []
        candidates: list[SkillSelection] = []
        candidate_names: set[str] = set()
        visible: dict[str, Skill] = {}
        for skill in self.list_skills():
            is_visible, reason = self.skill_visibility(skill)
            if not is_visible:
                decisions.append(
                    self._decision(
                        skill,
                        status="rejected",
                        stage="capability_filter",
                        reason=reason,
                    )
                )
                continue
            visible[skill.name] = skill

        def add_named(name: str, *, stage: str, score: int, reason: str) -> None:
            normalized = _normalize_skill_name(name)
            skill = self._skills.get(normalized)
            if skill is None:
                decisions.append(
                    SkillRouteDecision(
                        skill_name=normalized,
                        status="rejected",
                        stage=stage,
                        reason="unknown_skill",
                    )
                )
                return
            is_visible, visibility_reason = self.skill_visibility(skill)
            if not is_visible:
                decisions.append(
                    self._decision(
                        skill,
                        status="rejected",
                        stage=stage,
                        reason=visibility_reason,
                    )
                )
                return
            if skill.name in candidate_names:
                return
            candidate_names.add(skill.name)
            candidates.append(
                SkillSelection(
                    skill=skill,
                    score=score,
                    matched_keywords=[reason],
                    reason=reason,
                    stage=stage,
                )
            )
            for included_name in (
                skill.manifest.includes if skill.manifest is not None else ()
            ):
                add_named(
                    included_name,
                    stage="include",
                    score=score - 1,
                    reason=f"included_by:{skill.name}",
                )

        for name in explicit_names:
            add_named(name, stage="explicit", score=20_000, reason="explicit")
        for name in dict.fromkeys(pinned_names):
            add_named(name, stage="pinned", score=10_000, reason="pinned")

        if request_terms or request_text:
            keyword_candidates: list[SkillSelection] = []
            for skill in visible.values():
                if skill.name in candidate_names:
                    continue
                selection = self._keyword_selection(
                    skill,
                    request_text=request_text,
                    request_terms=request_terms,
                )
                if selection is None:
                    decisions.append(
                        self._decision(
                            skill,
                            status="omitted",
                            stage="keyword",
                            reason="no_keyword_match",
                        )
                    )
                    continue
                keyword_candidates.append(selection)
            keyword_candidates.sort(
                key=lambda selection: (-selection.score, selection.skill.name)
            )
            keyword_candidates = self._rerank_keywords(
                user_request,
                keyword_candidates,
            )
            candidates.extend(keyword_candidates)

        selection_limit = self._selection_limit(limit)
        selected = candidates[:selection_limit]
        selected_names = {selection.skill.name for selection in selected}
        for selection in selected:
            decisions.append(
                self._decision(
                    selection.skill,
                    status="selected",
                    stage=selection.stage,
                    reason=selection.reason,
                    score=selection.score,
                    matched_keywords=selection.matched_keywords,
                )
            )
        for selection in candidates[selection_limit:]:
            decisions.append(
                self._decision(
                    selection.skill,
                    status="omitted",
                    stage="budget",
                    reason="skill_count_limit",
                    score=selection.score,
                    matched_keywords=selection.matched_keywords,
                )
            )
        for skill in visible.values():
            if (
                skill.name not in selected_names
                and skill.name not in candidate_names
                and not any(
                    decision.skill_name == skill.name for decision in decisions
                )
            ):
                decisions.append(
                    self._decision(
                        skill,
                        status="omitted",
                        stage="keyword",
                        reason="no_keyword_match",
                    )
                )
        result = SkillRoutingResult(
            selections=tuple(selected),
            decisions=tuple(decisions),
            explicit_skill_names=explicit_names,
        )
        self.last_routing_result = result
        return result

    def _rerank_keywords(
        self,
        user_request: str,
        selections: list[SkillSelection],
    ) -> list[SkillSelection]:
        """Apply an optional reorder-only reranker with deterministic fallback."""
        if self.reranker is None or len(selections) < 2:
            return selections
        by_name = {selection.skill.name: selection for selection in selections}
        try:
            reranked_names = tuple(self.reranker(user_request, tuple(selections)))
        except Exception:
            return selections
        if (
            len(reranked_names) != len(by_name)
            or len(set(reranked_names)) != len(reranked_names)
            or set(reranked_names) != set(by_name)
        ):
            return selections
        return [
            SkillSelection(
                skill=by_name[name].skill,
                score=by_name[name].score,
                matched_keywords=by_name[name].matched_keywords,
                reason="reranked",
                stage="rerank",
            )
            for name in reranked_names
        ]

    def _keyword_selection(
        self,
        skill: Skill,
        *,
        request_text: str,
        request_terms: set[str],
    ) -> SkillSelection | None:
        searchable_terms = set(skill.keywords)
        searchable_terms.update(_tokenize(skill.name))
        searchable_terms.update(_tokenize(skill.description))
        matched = sorted(request_terms & searchable_terms)
        score = len(matched) * 10

        if skill.name in request_text:
            score += 25
            if skill.name not in matched:
                matched.append(skill.name)

        phrase_hits = [
            keyword
            for keyword in skill.keywords
            if " " in keyword and keyword in request_text
        ]
        if phrase_hits:
            score += len(phrase_hits) * 15
            matched.extend(phrase_hits)

        if score > 0:
            return SkillSelection(
                skill=skill,
                score=score,
                matched_keywords=sorted(set(matched)),
            )
        return None

    def load_selected_skills(
        self,
        user_request: str,
        *,
        pinned_names: Iterable[str] = (),
        limit: int | None = None,
    ) -> list[SkillSelection]:
        """Select relevant skills and lazy-load only those selected skill files."""
        selections = list(
            self.route_skills(
                user_request,
                pinned_names=pinned_names,
                limit=limit,
            ).selections
        )
        for selection in selections:
            self.load_content(selection.skill.name)
        return selections

    def _decision(
        self,
        skill: Skill,
        *,
        status: str,
        stage: str,
        reason: str,
        score: int = 0,
        matched_keywords: Iterable[str] = (),
    ) -> SkillRouteDecision:
        manifest = skill.manifest
        return SkillRouteDecision(
            skill_name=skill.name,
            status=status,
            stage=stage,
            reason=reason,
            score=score,
            matched_keywords=tuple(matched_keywords),
            version=manifest.version if manifest is not None else None,
            digest=skill.digest,
        )

    def load_content(self, name: str) -> str:
        """Progressively load entrypoint and reference text under one budget."""
        skill = self._skills[_normalize_skill_name(name)]
        if skill.loaded_content is None:
            if skill.digest is not None:
                load_skill_package(skill.path, expected_digest=skill.digest)
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


def explicit_skill_names(user_request: str) -> tuple[str, ...]:
    """Parse one or more leading `/skill` invocations without consuming the task."""
    tokens = user_request.strip().split()
    names: list[str] = []
    for token in tokens:
        if not token.startswith("/") or token == "/":
            break
        raw_names = token[1:].replace("+", ",").replace("/", ",").split(",")
        for raw_name in raw_names:
            candidate = raw_name.removeprefix("/").strip().lower()
            if not candidate:
                continue
            if not _EXACT_SKILL_NAME_PATTERN.fullmatch(candidate):
                continue
            if candidate not in names:
                names.append(candidate)
    return tuple(names)


def _normalize_platform(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("win"):
        return "windows"
    if normalized.startswith("linux"):
        return "linux"
    if normalized in {"darwin", "mac", "macos"}:
        return "darwin"
    return normalized
