"""Operator commands for statically inspected local plugins."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from chulk.plugins import (
    LocalPluginRegistry,
    PluginArtifactError,
    PluginCatalogError,
    PluginInspectionError,
    PluginLifecycleError,
    PluginLoadError,
    PluginLockError,
    PluginMigrationError,
    PluginRegistrationError,
    PluginVerificationError,
)


def run_plugin_command(
    command: str,
    *,
    registry: LocalPluginRegistry,
    path: Path | str | None,
    approved_by: str | None,
    plugin_name: str | None,
    repository_url: str | None,
    commit_sha: str | None,
    allowed_git_hosts: tuple[str, ...],
    approve_authority_changes: bool,
    reason: str | None,
    revoked_by: str | None,
    catalog_path: Path | str | None,
    query: str | None,
    category: str | None,
    version: str | None,
    limit: int,
    acknowledge_host_authority: bool,
    granted_capabilities: tuple[str, ...] | None,
    json_output: bool,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> int:
    """Execute one host-only plugin registry operation."""
    try:
        if command == "inspect":
            if path is None:
                raise ValueError("plugins inspect requires a path")
            inspection = registry.inspect(path)
            payload = {
                "ok": True,
                "status": "inspected",
                "inspection": inspection.to_dict(),
            }
            output_func(
                _json(payload)
                if json_output
                else _format_inspection(inspection.to_dict())
            )
            return 0
        if command == "install":
            if path is None or approved_by is None:
                raise ValueError(
                    "plugins install requires a path and --approved-by"
                )
            grants = granted_capabilities or ()
            if bool(repository_url) != bool(commit_sha):
                raise ValueError(
                    "trusted Git install requires both --repository-url "
                    "and --commit"
                )
            if repository_url is not None and commit_sha is not None:
                receipt = registry.install_trusted_git(
                    path,
                    repository_url=repository_url,
                    commit_sha=commit_sha,
                    allowed_hosts=allowed_git_hosts,
                    approved_by=approved_by,
                    acknowledge_host_authority=(
                        acknowledge_host_authority
                    ),
                    granted_capabilities=grants,
                )
            else:
                receipt = registry.install(
                    path,
                    approved_by=approved_by,
                    acknowledge_host_authority=(
                        acknowledge_host_authority
                    ),
                    granted_capabilities=grants,
                )
            output_func(
                _json({"ok": True, "receipt": receipt.to_dict()})
                if json_output
                else _format_receipt(receipt.to_dict())
            )
            return 0
        if command == "plan-update":
            if path is None:
                raise ValueError("plugins plan-update requires a path")
            plan = registry.plan_update(path)
            payload = {"ok": True, "plan": plan.to_dict()}
            output_func(
                _json(payload)
                if json_output
                else _format_update_plan(plan.to_dict())
            )
            return 0
        if command == "update":
            if path is None or approved_by is None:
                raise ValueError(
                    "plugins update requires a path and --approved-by"
                )
            if bool(repository_url) != bool(commit_sha):
                raise ValueError(
                    "trusted Git update requires both --repository-url "
                    "and --commit"
                )
            if repository_url is not None and commit_sha is not None:
                receipt = registry.update_trusted_git(
                    path,
                    repository_url=repository_url,
                    commit_sha=commit_sha,
                    allowed_hosts=allowed_git_hosts,
                    approved_by=approved_by,
                    acknowledge_host_authority=(
                        acknowledge_host_authority
                    ),
                    granted_capabilities=granted_capabilities,
                    approve_authority_changes=approve_authority_changes,
                )
            else:
                receipt = registry.update(
                    path,
                    approved_by=approved_by,
                    acknowledge_host_authority=(
                        acknowledge_host_authority
                    ),
                    granted_capabilities=granted_capabilities,
                    approve_authority_changes=approve_authority_changes,
                )
            output_func(
                _json({"ok": True, "receipt": receipt.to_dict()})
                if json_output
                else _format_receipt(receipt.to_dict())
            )
            return 0
        if command in {"uninstall", "rollback"}:
            if plugin_name is None or approved_by is None:
                raise ValueError(
                    f"plugins {command} requires a plugin name and "
                    "--approved-by"
                )
            operation = (
                registry.uninstall
                if command == "uninstall"
                else registry.rollback
            )
            receipt = operation(plugin_name, approved_by=approved_by)
            output_func(
                _json({"ok": True, "receipt": receipt.to_dict()})
                if json_output
                else _format_receipt(receipt.to_dict())
            )
            return 0
        if command == "revoke":
            if (
                plugin_name is None
                or reason is None
                or revoked_by is None
            ):
                raise ValueError(
                    "plugins revoke requires a plugin name, --reason, "
                    "and --revoked-by"
                )
            receipt = registry.revoke(
                plugin_name,
                reason=reason,
                revoked_by=revoked_by,
            )
            output_func(
                _json({"ok": True, "receipt": receipt.to_dict()})
                if json_output
                else _format_receipt(receipt.to_dict())
            )
            return 0
        if command == "register":
            if path is None:
                raise ValueError("plugins register requires a path")
            if approved_by is None:
                raise ValueError("plugins register requires --approved-by")
            entry = registry.register_local(
                path,
                approved_by=approved_by,
                acknowledge_host_authority=acknowledge_host_authority,
                granted_capabilities=granted_capabilities or (),
            )
            payload = {
                "ok": True,
                "status": "registered",
                "plugin": entry.to_dict(),
            }
            output_func(
                _json(payload)
                if json_output
                else (
                    f"Registered {entry.name} {entry.version} "
                    f"({entry.digest})"
                )
            )
            return 0
        if command == "list":
            entries = registry.list()
            payload = {
                "ok": True,
                "plugins": [entry.to_dict() for entry in entries],
            }
            output_func(
                _json(payload)
                if json_output
                else _format_list(
                    tuple(entry.to_dict() for entry in entries)
                )
            )
            return 0
        if command == "audit":
            report = registry.audit()
            output_func(
                _json(report.to_dict())
                if json_output
                else _format_audit(report.to_dict())
            )
            return 0 if report.ok else 2
        if command in {"catalog-search", "catalog-inspect"}:
            if catalog_path is None:
                raise ValueError(
                    f"plugins {command} requires a catalog path"
                )
            catalog = registry.load_catalog(
                catalog_path,
                allowed_git_hosts=allowed_git_hosts,
            )
            if command == "catalog-search":
                if query is None:
                    raise ValueError(
                        "plugins catalog-search requires a query"
                    )
                catalog_entries = catalog.search(
                    query,
                    category=category,
                    limit=limit,
                )
                payload = {
                    "ok": True,
                    "catalog_id": catalog.snapshot.catalog_id,
                    "entries": [
                        item.to_dict() for item in catalog_entries
                    ],
                }
            else:
                if plugin_name is None:
                    raise ValueError(
                        "plugins catalog-inspect requires a plugin name"
                    )
                catalog_entry = catalog.inspect(
                    plugin_name,
                    version=version,
                )
                payload = {
                    "ok": True,
                    "catalog_id": catalog.snapshot.catalog_id,
                    "entry": catalog_entry.to_dict(),
                }
            output_func(
                _json(payload)
                if json_output
                else _format_catalog(payload)
            )
            return 0
        raise ValueError(f"unknown plugins command: {command}")
    except (
        OSError,
        PluginArtifactError,
        PluginCatalogError,
        PluginInspectionError,
        PluginLifecycleError,
        PluginLoadError,
        PluginLockError,
        PluginMigrationError,
        PluginRegistrationError,
        PluginVerificationError,
        ValueError,
    ) as exc:
        if json_output:
            output_func(
                _json(
                    {
                        "ok": False,
                        "status": "plugin_error",
                        "error": str(exc),
                    }
                )
            )
        else:
            error_func(f"plugin error: {exc}")
        return 2


def _format_inspection(value: dict[str, object]) -> str:
    manifest = value["manifest"]
    assert isinstance(manifest, dict)
    lines = [
        f"Plugin {manifest['name']} {manifest['version']}:",
        f"  compatible: {value['compatible']}",
        f"  digest: {value['digest']}",
        f"  source: {manifest['source']}",
        f"  trust: {manifest['trust']}",
        f"  capabilities: {manifest['capabilities']}",
        f"  secret_refs: {manifest['secret_refs']}",
        f"  network_domains: {manifest['network_domains']}",
        "  entry_points:",
    ]
    entry_points = manifest["entry_points"]
    assert isinstance(entry_points, dict)
    for category, entries in sorted(entry_points.items()):
        assert isinstance(entries, dict)
        for name, definition in sorted(entries.items()):
            assert isinstance(definition, dict)
            lines.append(
                f"    {category}:{name} -> {definition['target']}"
            )
    for field_name in (
        "missing_python_dependencies",
        "incompatible_python_dependencies",
    ):
        if value[field_name]:
            lines.append(f"  {field_name}: {value[field_name]}")
    return "\n".join(lines)


def _format_list(values: tuple[dict[str, object], ...]) -> str:
    if not values:
        return "No registered plugins."
    lines = ["Plugins:"]
    for value in values:
        lines.append(
            f"  {value['name']} {value['version']} "
            f"{value['status']} {value['digest']}"
        )
    return "\n".join(lines)


def _format_audit(value: dict[str, object]) -> str:
    status = "passed" if value["ok"] else "failed"
    lines = [
        f"Plugin audit {status} for profile {value['profile_id']}.",
        f"  verified: {value['verified_plugins']}",
    ]
    findings = value["findings"]
    assert isinstance(findings, list)
    for finding in findings:
        assert isinstance(finding, dict)
        lines.append(
            f"  {finding['severity']} {finding['plugin_name']} "
            f"{finding['code']}: {finding['message']}"
        )
    return "\n".join(lines)


def _format_receipt(value: dict[str, object]) -> str:
    verbs = {
        "install": "Installed",
        "update": "Updated",
        "uninstall": "Uninstalled",
        "rollback": "Rolled back",
        "revoke": "Revoked",
    }
    action = str(value["action"])
    return (
        f"{verbs.get(action, action.title())} "
        f"{value['plugin_name']} {value['version']} "
        f"({value['digest']})"
    )


def _format_update_plan(value: dict[str, object]) -> str:
    lines = [
        (
            f"Plugin update {value['plugin_name']}: "
            f"{value['current_version']} -> {value['candidate_version']}"
        ),
        f"  compatible: {value['compatible']}",
        f"  current digest: {value['current_digest']}",
        f"  candidate digest: {value['candidate_digest']}",
        f"  requires reapproval: {value['requires_reapproval']}",
    ]
    authority = value["authority_diff"]
    assert isinstance(authority, dict)
    for field_name, items in authority.items():
        if field_name != "changed" and items:
            lines.append(f"  {field_name}: {items}")
    return "\n".join(lines)


def _format_catalog(value: dict[str, object]) -> str:
    if "entry" in value:
        entry = value["entry"]
        assert isinstance(entry, dict)
        return (
            f"{entry['name']} {entry['version']} "
            f"{entry['audit_state']} {entry['package_digest']}"
        )
    entries = value["entries"]
    assert isinstance(entries, list)
    if not entries:
        return "No catalog matches."
    lines = [f"Catalog {value['catalog_id']}:"]
    for entry in entries:
        assert isinstance(entry, dict)
        lines.append(
            f"  {entry['name']} {entry['version']} "
            f"{entry['audit_state']} {entry['package_digest']}"
        )
    return "\n".join(lines)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


__all__ = ["run_plugin_command"]
