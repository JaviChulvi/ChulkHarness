"""Parity coverage for runtime skill-spec assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from chulk._runtime.skills import resolve_skill_specs, resolve_skill_specs_async
from chulk.skills import SkillAllowlistRef, SkillPinRef, SkillRef


@dataclass(frozen=True)
class _Skill:
    name: str


class _SyncRegistry:
    def __init__(self) -> None:
        self.skills = {name: _Skill(name) for name in ("review", "pinned", "raw")}
        self.restricted_to: list[str] | None = None

    def clear(self) -> None:
        self.skills.clear()

    def register_path(self, path: Path | str) -> _Skill:
        skill = _Skill(Path(path).stem)
        self.skills[skill.name] = skill
        return skill

    def get_skill(self, name: str) -> _Skill | None:
        return self.skills.get(name)

    def restrict_to(self, names: list[str]) -> None:
        self.restricted_to = names


class _AsyncRegistry(_SyncRegistry):
    async def clear(self) -> None:  # type: ignore[override]
        self.skills.clear()

    async def register_path(self, path: Path | str) -> _Skill:  # type: ignore[override]
        return super().register_path(path)

    async def get_skill(self, name: str) -> _Skill | None:  # type: ignore[override]
        return super().get_skill(name)

    async def restrict_to(self, names: list[str]) -> None:  # type: ignore[override]
        self.restricted_to = names


def _specs() -> list[object]:
    return [
        SkillAllowlistRef(("review", "missing")),
        SkillPinRef(("pinned",)),
        SkillRef(skill_path=Path("path-skill")),
        "raw",
    ]


@pytest.mark.asyncio
async def test_sync_and_async_skill_specs_share_normalized_resolution():
    sync_registry = _SyncRegistry()
    async_registry = _AsyncRegistry()

    sync = resolve_skill_specs(sync_registry, _specs())  # type: ignore[arg-type]
    async_result = await resolve_skill_specs_async(async_registry, _specs())

    assert async_result == sync
    assert sync.pinned_skill_names == ["pinned", "path-skill", "raw"]
    assert sync_registry.restricted_to == ["review", "pinned", "path-skill", "raw"]
    assert async_registry.restricted_to == sync_registry.restricted_to
