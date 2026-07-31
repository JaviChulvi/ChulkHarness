"""Skill configuration for runtime assembly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from chulk.capabilities import Capabilities
from chulk.skills import (
    SkillAllowlistRef,
    SkillDirectoryRef,
    SkillPinRef,
    SkillRef,
    SkillRegistry,
)


@dataclass(frozen=True)
class SkillSpecResolution:
    """Resolved SDK skill configuration for one agent runtime."""

    pinned_skill_names: list[str]
    warnings: list[dict[str, str]]


def skill_capability_names(capabilities: Capabilities) -> set[str]:
    """Flatten runtime capabilities into manifest-facing capability names."""
    values = capabilities.to_dict()
    names = {
        name
        for name in ("shell", "network", "external_services", "utilities")
        if values[name] is True
    }
    file_access = str(values["files"])
    if file_access in {"read", "write"}:
        names.update({"files", "files:read"})
    if file_access == "write":
        names.add("files:write")
    memory_mode = str(values["memory"])
    if memory_mode != "off":
        names.update({"memory", "memory:read"})
    if memory_mode in {"manual", "automatic"}:
        names.add("memory:write")
    if memory_mode == "automatic":
        names.add("memory:automatic")
    return names


def resolve_skill_specs(
    registry: SkillRegistry,
    skill_specs: object | Iterable[object] | None,
) -> SkillSpecResolution:
    """Apply configured skill references and return pinned names and warnings."""
    specs = _coerce_skill_specs(skill_specs)
    if specs is None:
        return SkillSpecResolution(pinned_skill_names=[], warnings=[])
    if not specs:
        registry.clear()
        return SkillSpecResolution(pinned_skill_names=[], warnings=[])

    allowlist_requests: list[str] = []
    pin_requests: list[str] = []
    warning_payloads: list[dict[str, str]] = []
    has_allowlist = False

    for spec in specs:
        if isinstance(spec, SkillAllowlistRef):
            has_allowlist = True
            allowlist_requests.extend(spec.names)
            continue
        if isinstance(spec, SkillPinRef):
            pin_requests.extend(spec.names)
            continue
        if isinstance(spec, SkillDirectoryRef):
            spec.register(registry)
            continue
        if isinstance(spec, SkillRef):
            if spec.skill_path is not None:
                skill = registry.register_path(spec.skill_path)
                pin_requests.append(skill.name)
                continue
            if spec.name is not None:
                pin_requests.append(spec.name)
                continue
            raise ValueError("SkillRef must include name or skill_path")
        if hasattr(spec, "register"):
            pinned_name = spec.register(registry)  # type: ignore[attr-defined]
            if pinned_name:
                pin_requests.append(str(pinned_name))
            continue
        if isinstance(spec, str):
            pin_requests.append(spec)
            continue
        raise TypeError(f"Unsupported skill spec: {spec!r}")

    allowlisted_names = _resolve_existing_skill_names(
        registry,
        allowlist_requests,
        kind="allowlist",
        warning_payloads=warning_payloads,
    )
    pinned_skill_names = _resolve_existing_skill_names(
        registry,
        pin_requests,
        kind="pin",
        warning_payloads=warning_payloads,
    )

    if has_allowlist:
        registry.restrict_to([*allowlisted_names, *pinned_skill_names])

    return SkillSpecResolution(
        pinned_skill_names=pinned_skill_names,
        warnings=warning_payloads,
    )


def _coerce_skill_specs(
    skill_specs: object | Iterable[object] | None,
) -> list[object] | None:
    if skill_specs is None:
        return None
    if isinstance(
        skill_specs,
        (str, SkillAllowlistRef, SkillDirectoryRef, SkillPinRef, SkillRef),
    ):
        return [skill_specs]
    try:
        return list(cast(Iterable[object], skill_specs))
    except TypeError:
        return [skill_specs]


def _resolve_existing_skill_names(
    registry: SkillRegistry,
    names: Iterable[str],
    *,
    kind: str,
    warning_payloads: list[dict[str, str]],
) -> list[str]:
    resolved_names: list[str] = []
    for requested_name in names:
        skill = registry.get_skill(requested_name)
        if skill is None:
            _append_missing_skill_warning(kind, requested_name, warning_payloads)
            continue
        if skill.name not in resolved_names:
            resolved_names.append(skill.name)
    return resolved_names


def _append_missing_skill_warning(
    kind: str,
    requested_name: str,
    warnings_list: list[dict[str, str]],
) -> None:
    if any(
        payload["kind"] == kind and payload["name"] == requested_name
        for payload in warnings_list
    ):
        return
    warnings_list.append(
        {
            "kind": kind,
            "name": requested_name,
            "message": (
                f"Skill '{requested_name}' requested by {kind} configuration "
                "is not registered; skipping."
            ),
        }
    )
