"""CLI operations for durable agent profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from chulk.profiles import (
    CredentialRef,
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    ProfileOwnershipError,
    SQLiteProfileStore,
)


def run_profile_command(
    command: str,
    *,
    store: SQLiteProfileStore,
    profile_id: str | None,
    project_root: Path | str | None,
    permission_profile: str | None,
    model_profile_id: str,
    execution_backend_id: str,
    allowed_skills: tuple[str, ...] | None,
    allowed_mcp_servers: tuple[str, ...] | None,
    credential_environment_names: tuple[str, ...],
    system_prompt: str | None,
    json_output: bool,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> int:
    """Execute one deterministic profile control operation."""
    try:
        if command == "create":
            if profile_id is None or project_root is None:
                raise ValueError("profile create requires an id and project root")
            stored = store.create_profile(
                profile_id,
                project_root=project_root,
                permission_profile=permission_profile,
                model_profile_id=model_profile_id,
                execution_backend_id=execution_backend_id,
                allowed_skills=allowed_skills,
                allowed_mcp_servers=allowed_mcp_servers,
                credential_refs=tuple(
                    CredentialRef(name=name) for name in credential_environment_names
                ),
                system_prompt=system_prompt,
            )
            payload = {
                "ok": True,
                "status": "created",
                "profile": stored.to_dict(),
            }
            output_func(
                _json(payload)
                if json_output
                else f"Created profile {stored.profile.id} at {stored.profile.runtime_dir}"
            )
            return 0
        if command == "list":
            selected_id = store.selected_cli_profile_id()
            profiles = [
                {
                    **stored.to_dict(),
                    "selected_for_cli": stored.profile.id == selected_id,
                }
                for stored in store.list()
            ]
            output_func(
                _json({"ok": True, "profiles": profiles})
                if json_output
                else _format_profile_list(profiles)
            )
            return 0
        if command == "use":
            if profile_id is None:
                raise ValueError("profile use requires an id")
            selected = store.use(profile_id)
            payload = {
                "ok": True,
                "status": "selected",
                "profile_id": selected.profile.id,
            }
            output_func(
                _json(payload)
                if json_output
                else f"CLI default profile: {selected.profile.id}"
            )
            return 0
        if command == "inspect":
            if profile_id is None:
                raise ValueError("profile inspect requires an id")
            stored = store.get(profile_id)
            profile_payload: dict[str, object] = {
                **stored.to_dict(),
                "selected_for_cli": stored.profile.id == store.selected_cli_profile_id(),
            }
            payload = {
                "ok": True,
                "profile": profile_payload,
            }
            output_func(_json(payload) if json_output else _format_profile(profile_payload))
            return 0
        raise ValueError(f"unknown profile command: {command}")
    except (
        OSError,
        ProfileAlreadyExistsError,
        ProfileNotFoundError,
        ProfileOwnershipError,
        ValueError,
    ) as exc:
        if json_output:
            output_func(_json({"ok": False, "status": "profile_error", "error": str(exc)}))
        else:
            error_func(f"profile error: {exc}")
        return 2


def _format_profile_list(profiles: list[dict[str, object]]) -> str:
    lines = ["Profiles:"]
    for profile in profiles:
        marker = "*" if profile["selected_for_cli"] else " "
        lines.append(f"  {marker} {profile['id']}  {profile['project_root']}")
    return "\n".join(lines)


def _format_profile(profile: dict[str, object]) -> str:
    lines = [f"Profile {profile['id']}:"]
    for key in (
        "project_root",
        "runtime_dir",
        "store_path",
        "traces_dir",
        "memory_namespace",
        "permission_profile",
        "model_profile_id",
        "execution_backend_id",
        "allowed_skills",
        "allowed_mcp_servers",
        "credential_refs",
        "auxiliary_models",
        "system_prompt",
        "implicit",
        "selected_for_cli",
    ):
        lines.append(f"  {key}: {profile.get(key)}")
    return "\n".join(lines)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


__all__ = ["run_profile_command"]
