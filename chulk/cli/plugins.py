"""Operator commands for statically inspected local plugins."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from chulk.plugins import (
    LocalPluginRegistry,
    PluginInspectionError,
    PluginLoadError,
    PluginLockError,
    PluginRegistrationError,
    PluginVerificationError,
)


def run_plugin_command(
    command: str,
    *,
    registry: LocalPluginRegistry,
    path: Path | str | None,
    approved_by: str | None,
    acknowledge_host_authority: bool,
    granted_capabilities: tuple[str, ...],
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
        if command == "register":
            if path is None:
                raise ValueError("plugins register requires a path")
            if approved_by is None:
                raise ValueError("plugins register requires --approved-by")
            entry = registry.register_local(
                path,
                approved_by=approved_by,
                acknowledge_host_authority=acknowledge_host_authority,
                granted_capabilities=granted_capabilities,
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
        raise ValueError(f"unknown plugins command: {command}")
    except (
        OSError,
        PluginInspectionError,
        PluginLoadError,
        PluginLockError,
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


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


__all__ = ["run_plugin_command"]
