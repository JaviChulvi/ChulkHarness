"""Skill configuration for runtime assembly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from chulk.capabilities import Capabilities
from chulk.hosting.async_utils import call_async_service
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


@dataclass(frozen=True)
class SkillSpecOperation:
    """One normalized skill configuration operation."""

    kind: Literal["allowlist", "pin", "directory", "path", "custom"]
    value: object


@dataclass(frozen=True)
class SkillSpecPlan:
    """Normalized skill configuration applied by sync or async registry drivers."""

    operations: tuple[SkillSpecOperation, ...]


@dataclass
class _SkillSpecRequests:
    allowlist_names: list[str]
    pin_names: list[str]
    has_allowlist: bool = False

    @classmethod
    def create(cls) -> "_SkillSpecRequests":
        return cls(allowlist_names=[], pin_names=[])

    def record(
        self,
        operation: SkillSpecOperation,
        *,
        registered_name: str | None = None,
    ) -> None:
        if operation.kind == "allowlist":
            self.has_allowlist = True
            self.allowlist_names.extend(cast(tuple[str, ...], operation.value))
        elif operation.kind == "pin":
            self.pin_names.extend(cast(tuple[str, ...], operation.value))
        elif registered_name:
            self.pin_names.append(registered_name)


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
    plan = _skill_spec_plan(skill_specs)
    if plan is None:
        return SkillSpecResolution(pinned_skill_names=[], warnings=[])
    if not plan.operations:
        registry.clear()
        return SkillSpecResolution(pinned_skill_names=[], warnings=[])

    requests = _SkillSpecRequests.create()
    warning_payloads: list[dict[str, str]] = []

    for operation in plan.operations:
        requests.record(
            operation,
            registered_name=_apply_skill_spec_operation(registry, operation),
        )

    allowlisted_names = _resolve_existing_skill_names(
        registry,
        requests.allowlist_names,
        kind="allowlist",
        warning_payloads=warning_payloads,
    )
    pinned_skill_names = _resolve_existing_skill_names(
        registry,
        requests.pin_names,
        kind="pin",
        warning_payloads=warning_payloads,
    )

    if requests.has_allowlist:
        registry.restrict_to([*allowlisted_names, *pinned_skill_names])

    return SkillSpecResolution(
        pinned_skill_names=pinned_skill_names,
        warnings=warning_payloads,
    )


async def resolve_skill_specs_async(
    registry: object,
    skill_specs: object | Iterable[object] | None,
) -> SkillSpecResolution:
    """Apply skill references through the native async registry contract."""
    plan = _skill_spec_plan(skill_specs)
    if plan is None:
        return SkillSpecResolution(pinned_skill_names=[], warnings=[])
    if not plan.operations:
        await call_async_service(registry, "clear")
        return SkillSpecResolution(pinned_skill_names=[], warnings=[])

    requests = _SkillSpecRequests.create()
    warning_payloads: list[dict[str, str]] = []

    for operation in plan.operations:
        requests.record(
            operation,
            registered_name=await _apply_skill_spec_operation_async(
                registry,
                operation,
            ),
        )

    allowlisted_names = await _resolve_existing_skill_names_async(
        registry,
        requests.allowlist_names,
        kind="allowlist",
        warning_payloads=warning_payloads,
    )
    pinned_skill_names = await _resolve_existing_skill_names_async(
        registry,
        requests.pin_names,
        kind="pin",
        warning_payloads=warning_payloads,
    )
    if requests.has_allowlist:
        await call_async_service(
            registry,
            "restrict_to",
            [*allowlisted_names, *pinned_skill_names],
        )
    return SkillSpecResolution(
        pinned_skill_names=pinned_skill_names,
        warnings=warning_payloads,
    )


def _skill_spec_plan(
    skill_specs: object | Iterable[object] | None,
) -> SkillSpecPlan | None:
    specs = _coerce_skill_specs(skill_specs)
    if specs is None:
        return None
    return SkillSpecPlan(tuple(_skill_spec_operation(spec) for spec in specs))


def _skill_spec_operation(spec: object) -> SkillSpecOperation:
    if isinstance(spec, SkillAllowlistRef):
        return SkillSpecOperation("allowlist", tuple(spec.names))
    if isinstance(spec, SkillPinRef):
        return SkillSpecOperation("pin", tuple(spec.names))
    if isinstance(spec, SkillDirectoryRef):
        return SkillSpecOperation("directory", spec)
    if isinstance(spec, SkillRef):
        if spec.skill_path is not None:
            return SkillSpecOperation("path", spec.skill_path)
        if spec.name is not None:
            return SkillSpecOperation("pin", (spec.name,))
        raise ValueError("SkillRef must include name or skill_path")
    if isinstance(spec, str):
        return SkillSpecOperation("pin", (spec,))
    return SkillSpecOperation("custom", spec)


def _apply_skill_spec_operation(
    registry: SkillRegistry,
    operation: SkillSpecOperation,
) -> str | None:
    if operation.kind in {"allowlist", "pin"}:
        return None
    if operation.kind == "directory":
        cast(SkillDirectoryRef, operation.value).register(registry)
        return None
    if operation.kind == "path":
        return registry.register_path(cast(Path | str, operation.value)).name
    custom = operation.value
    register = getattr(custom, "register", None)
    if not callable(register):
        raise TypeError(f"Unsupported skill spec: {custom!r}")
    pinned_name = register(registry)
    return str(pinned_name) if pinned_name else None


async def _apply_skill_spec_operation_async(
    registry: object,
    operation: SkillSpecOperation,
) -> str | None:
    if operation.kind in {"allowlist", "pin"}:
        return None
    if operation.kind == "directory":
        await call_async_service(
            registry,
            "register_directory",
            cast(SkillDirectoryRef, operation.value).skills_dir,
        )
        return None
    if operation.kind == "path":
        skill = await call_async_service(
            registry,
            "register_path",
            operation.value,
        )
        return str(skill.name)
    custom = operation.value
    register_async = getattr(custom, "register_async", None)
    if not callable(register_async):
        raise TypeError(
            "native async hosted skills require SkillRef values or an "
            "async register_async(registry) implementation"
        )
    pinned_name = await register_async(registry)
    return str(pinned_name) if pinned_name else None


async def _resolve_existing_skill_names_async(
    registry: object,
    names: Iterable[str],
    *,
    kind: str,
    warning_payloads: list[dict[str, str]],
) -> list[str]:
    resolved_names: list[str] = []
    for requested_name in names:
        skill = await call_async_service(
            registry,
            "get_skill",
            requested_name,
        )
        if skill is None:
            _append_missing_skill_warning(
                kind,
                requested_name,
                warning_payloads,
            )
            continue
        if skill.name not in resolved_names:
            resolved_names.append(skill.name)
    return resolved_names


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
