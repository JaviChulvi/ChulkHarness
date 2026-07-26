"""Tests for managed plugin supply-chain and recovery lifecycle."""

from __future__ import annotations

from base64 import urlsafe_b64encode
from contextlib import closing
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import zipfile

import pytest

from chulk.main import main
from chulk.plugins import (
    LocalPluginRegistry,
    PluginArtifactError,
    PluginCatalogError,
    PluginCategory,
    PluginLifecycleAction,
    PluginRegistrationError,
    PluginRegistrationStatus,
    PluginSourceKind,
    PluginVerificationError,
    ReviewedPluginCatalog,
)
from chulk.tests.test_plugin_manifests import (
    plugin_manifest,
    write_plugin,
)
from chulk.tools import create_default_tool_registry


def registry(tmp_path: Path) -> LocalPluginRegistry:
    return LocalPluginRegistry(
        tmp_path / "runtime",
        profile_id="default",
    )


def versioned_plugin(
    root: Path,
    version: str,
    *,
    authority_change: bool = False,
    second_migration: str | None = None,
) -> Path:
    manifest = plugin_manifest(version=version)
    if authority_change:
        manifest = manifest.replace(
            "  - network\nsecret_refs:",
            "  - network\n  - shell\nsecret_refs:",
        ).replace(
            "  - env:SAMPLE_TOKEN\nnetwork_domains:",
            "  - env:SAMPLE_TOKEN\n  - host:NEW_TOKEN\nnetwork_domains:",
        ).replace(
            "  - api.example.com\nfilesystem:",
            "  - api.example.com\n  - uploads.example.com\nfilesystem:",
        )
    if second_migration is not None:
        manifest = manifest.replace(
            "migrations:\n  - migrations/001.sql\n",
            "migrations:\n"
            "  - migrations/001.sql\n"
            "  - migrations/002.sql\n",
        )
    package = write_plugin(root, manifest=manifest)
    if second_migration is not None:
        (package / "migrations" / "002.sql").write_text(
            second_migration,
            encoding="utf-8",
        )
    return package


def test_managed_install_quarantines_and_decouples_local_source(tmp_path):
    package = versioned_plugin(tmp_path / "source", "1.2.3")
    plugins = registry(tmp_path)

    receipt = plugins.install(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )

    entry = plugins.list()[0]
    assert receipt.action is PluginLifecycleAction.INSTALL
    assert entry.source_path != package
    assert entry.source_path.is_relative_to(
        tmp_path / "runtime" / "plugins" / "installed"
    )
    assert entry.source_reference == str(package.resolve())
    assert entry.artifact_digest == entry.digest
    assert entry.source_kind is PluginSourceKind.LOCAL_DIRECTORY
    assert (tmp_path / "runtime" / "plugins" / "quarantine").is_dir()
    (package / "sample_plugin" / "tools.py").write_text(
        "raise RuntimeError('source changed')\n",
        encoding="utf-8",
    )
    assert plugins.audit().ok is True
    assert plugins.verify_startup().verified_plugins == ("sample-plugin",)


def test_update_requires_authority_reapproval_and_rolls_back_data(
    tmp_path,
):
    plugins = registry(tmp_path)
    first = versioned_plugin(tmp_path / "v1", "1.2.3")
    plugins.install(
        first,
        approved_by="operator",
        acknowledge_host_authority=True,
        granted_capabilities=("files:read",),
    )
    second = versioned_plugin(
        tmp_path / "v2",
        "2.0.0",
        authority_change=True,
        second_migration=(
            "CREATE TABLE second_version "
            "(id TEXT PRIMARY KEY);\n"
        ),
    )

    plan = plugins.plan_update(second)

    assert plan.requires_reapproval is True
    assert plan.authority_diff.added_capabilities == ("shell",)
    assert plan.authority_diff.added_secret_refs == ("host:NEW_TOKEN",)
    assert plan.authority_diff.added_network_domains == (
        "uploads.example.com",
    )
    with pytest.raises(
        PluginRegistrationError,
        match="authority changed",
    ):
        plugins.update(
            second,
            approved_by="operator",
            acknowledge_host_authority=True,
        )

    updated = plugins.update(
        second,
        approved_by="operator",
        acknowledge_host_authority=True,
        approve_authority_changes=True,
    )

    assert updated.action is PluginLifecycleAction.UPDATE
    assert updated.previous_version == "1.2.3"
    assert plugins.list()[0].version == "2.0.0"
    database = plugins.migrations.database_path("sample-plugin")
    with closing(sqlite3.connect(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "second_version" in tables

    rolled_back = plugins.rollback(
        "sample-plugin",
        approved_by="operator",
    )

    assert rolled_back.action is PluginLifecycleAction.ROLLBACK
    assert plugins.list()[0].version == "1.2.3"
    with closing(sqlite3.connect(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "second_version" not in tables
    assert plugins.audit().ok is True


def test_failed_migration_restores_lock_and_validated_database(tmp_path):
    plugins = registry(tmp_path)
    first = versioned_plugin(tmp_path / "v1", "1.2.3")
    plugins.install(
        first,
        approved_by="operator",
        acknowledge_host_authority=True,
    )
    second = versioned_plugin(
        tmp_path / "v2",
        "2.0.0",
        second_migration=(
            "CREATE TABLE should_rollback (id TEXT);\n"
            "THIS IS NOT SQL;\n"
        ),
    )

    with pytest.raises(
        PluginRegistrationError,
        match="migration failed",
    ):
        plugins.update(
            second,
            approved_by="operator",
            acknowledge_host_authority=True,
        )

    assert plugins.list()[0].version == "1.2.3"
    database = plugins.migrations.database_path("sample-plugin")
    with closing(sqlite3.connect(database)) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert connection.execute(
            "PRAGMA integrity_check"
        ).fetchone() == ("ok",)
    assert "should_rollback" not in names
    assert plugins.audit().ok is True


def test_update_rejects_modified_forward_only_migration(tmp_path):
    plugins = registry(tmp_path)
    first = versioned_plugin(tmp_path / "v1", "1.2.3")
    plugins.install(
        first,
        approved_by="operator",
        acknowledge_host_authority=True,
    )
    second = versioned_plugin(tmp_path / "v2", "2.0.0")
    (second / "migrations" / "001.sql").write_text(
        "CREATE TABLE changed_history (id TEXT);\n",
        encoding="utf-8",
    )

    with pytest.raises(
        PluginRegistrationError,
        match="existing migration changed",
    ):
        plugins.plan_update(second)


def test_update_plan_hashes_instruction_content_changes(tmp_path):
    plugins = registry(tmp_path)
    manifest_v1 = plugin_manifest(version="1.2.3") + (
        "instructions:\n  - instructions/guide.md\n"
    )
    first = write_plugin(tmp_path / "v1", manifest=manifest_v1)
    (first / "instructions").mkdir()
    (first / "instructions" / "guide.md").write_text(
        "Original instructions.\n",
        encoding="utf-8",
    )
    plugins.install(
        first,
        approved_by="operator",
        acknowledge_host_authority=True,
    )
    manifest_v2 = plugin_manifest(version="2.0.0") + (
        "instructions:\n  - instructions/guide.md\n"
    )
    second = write_plugin(tmp_path / "v2", manifest=manifest_v2)
    (second / "instructions").mkdir()
    (second / "instructions" / "guide.md").write_text(
        "Changed authority-bearing instructions.\n",
        encoding="utf-8",
    )

    plan = plugins.plan_update(second)

    assert plan.requires_reapproval is True
    assert plan.authority_diff.added_instructions[0].startswith(
        "instructions/guide.md@sha256:"
    )
    assert plan.authority_diff.removed_instructions[0].startswith(
        "instructions/guide.md@sha256:"
    )


def test_uninstall_checks_dependents_and_retains_recovery(tmp_path):
    plugins = registry(tmp_path)
    base = versioned_plugin(tmp_path / "base", "1.2.3")
    plugins.install(
        base,
        approved_by="operator",
        acknowledge_host_authority=True,
    )
    dependent_manifest = plugin_manifest(
        name="dependent-plugin",
        version="1.0.0",
    ).replace(
        "dependencies: {}",
        'dependencies:\n  sample-plugin: ">=1,<2"',
    )
    dependent = write_plugin(
        tmp_path / "dependent",
        manifest=dependent_manifest,
    )
    dependent = dependent.rename(
        dependent.parent / "dependent-plugin"
    )
    plugins.install(
        dependent,
        approved_by="operator",
        acknowledge_host_authority=True,
    )

    with pytest.raises(
        PluginRegistrationError,
        match="enabled dependents",
    ):
        plugins.uninstall(
            "sample-plugin",
            approved_by="operator",
        )

    receipt = plugins.uninstall(
        "dependent-plugin",
        approved_by="operator",
    )
    assert receipt.recovery_id is not None
    assert {
        entry.name: entry.status for entry in plugins.list()
    }["dependent-plugin"] is PluginRegistrationStatus.DISABLED

    plugins.rollback("dependent-plugin", approved_by="operator")
    assert {
        entry.name: entry.status for entry in plugins.list()
    }["dependent-plugin"] is PluginRegistrationStatus.ENABLED


def test_revocation_is_persistent_and_startup_fails_closed(tmp_path):
    plugins = registry(tmp_path)
    package = versioned_plugin(tmp_path / "source", "1.2.3")
    plugins.install(
        package,
        approved_by="operator",
        acknowledge_host_authority=True,
    )

    receipt = plugins.revoke(
        "sample-plugin",
        reason="compromised signing account",
        revoked_by="security",
    )

    assert receipt.action is PluginLifecycleAction.REVOKE
    report = plugins.audit()
    assert report.ok is False
    assert report.findings[0].code == "revoked_digest"
    with pytest.raises(
        PluginVerificationError,
        match="compromised signing account",
    ):
        plugins.verify_startup()
    with pytest.raises(
        PluginRegistrationError,
        match="revoked plugin digest",
    ):
        plugins.rollback("sample-plugin", approved_by="operator")


def _wheel_from_package(package: Path, destination: Path) -> Path:
    files = {
        path.relative_to(package).as_posix(): path.read_bytes()
        for path in package.rglob("*")
        if path.is_file()
    }
    dist_info = "sample_plugin-1.2.3.dist-info"
    files[f"{dist_info}/METADATA"] = (
        "Metadata-Version: 2.3\n"
        "Name: sample-plugin\n"
        "Version: 1.2.3\n"
        "Requires-Dist: packaging (>=24)\n"
        "\n"
    ).encode()
    files[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: chulk-tests\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
        "\n"
    ).encode()
    rows: list[tuple[str, str, str]] = []
    for name, content in sorted(files.items()):
        encoded = urlsafe_b64encode(
            sha256(content).digest()
        ).decode().rstrip("=")
        rows.append((name, f"sha256={encoded}", str(len(content))))
    record_name = f"{dist_info}/RECORD"
    rows.append((record_name, "", ""))
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    files[record_name] = stream.getvalue().encode()
    wheel = destination / "sample_plugin-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(
        wheel,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, content in sorted(files.items()):
            archive.writestr(name, content)
    return wheel


def test_exact_prebuilt_wheel_is_verified_retained_and_audited(tmp_path):
    package = versioned_plugin(tmp_path / "package", "1.2.3")
    wheel = _wheel_from_package(package, tmp_path)
    plugins = registry(tmp_path)

    plugins.install(
        wheel,
        approved_by="operator",
        acknowledge_host_authority=True,
    )

    entry = plugins.list()[0]
    assert entry.source_kind is PluginSourceKind.PREBUILT_WHEEL
    retained = (
        tmp_path
        / "runtime"
        / "plugins"
        / "quarantine"
        / entry.artifact_digest.removeprefix("sha256:")
        / "artifact.whl"
    )
    assert retained.is_file()
    assert plugins.audit().ok is True
    retained.write_bytes(b"tampered wheel")
    report = plugins.audit()
    assert report.ok is False
    assert report.findings[0].code == "package_invalid"


def test_wheel_inspection_rejects_archive_escape(tmp_path):
    wheel = tmp_path / "bad-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("../escape.py", b"bad")

    with pytest.raises(PluginArtifactError, match="escapes"):
        registry(tmp_path).packages.prepare(wheel)
    assert not (tmp_path / "escape.py").exists()


def catalog_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "catalog_id": "reviewed",
        "source": {
            "repository_url": "https://github.com/example/catalog.git",
            "commit_sha": "a" * 40,
            "digest": "sha256:" + "b" * 64,
            "reviewed_by": "operator",
        },
        "entries": [
            {
                "name": "sample-plugin",
                "version": "1.2.3",
                "description": "Safe MCP and skill metadata.",
                "package_source": (
                    "https://github.com/example/sample-plugin.git"
                ),
                "package_digest": "sha256:" + "c" * 64,
                "source_kind": "trusted-git",
                "trust": "operator-trusted",
                "capabilities": ["network"],
                "secret_refs": ["env:SAMPLE_TOKEN"],
                "network_domains": ["api.example.com"],
                "categories": [
                    "mcp_configuration",
                    "skill_registry",
                ],
                "audit_state": "reviewed",
            }
        ],
    }


def test_reviewed_catalog_is_metadata_only_and_host_pinned(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(catalog_payload()),
        encoding="utf-8",
    )
    catalog = ReviewedPluginCatalog.load(
        catalog_path,
        allowed_git_hosts=("github.com",),
    )

    matches = catalog.search(
        "MCP",
        category=PluginCategory.MCP_CONFIGURATION,
    )

    assert [entry.name for entry in matches] == ["sample-plugin"]
    assert catalog.inspect("sample-plugin").secret_refs == (
        "env:SAMPLE_TOKEN",
    )
    assert not hasattr(catalog, "install")
    with pytest.raises(PluginCatalogError, match="not explicitly trusted"):
        ReviewedPluginCatalog.load(
            catalog_path,
            allowed_git_hosts=("gitlab.com",),
        )


def test_trusted_git_install_requires_clean_exact_allowed_checkout(
    tmp_path,
):
    package = versioned_plugin(tmp_path, "1.2.3")
    subprocess.run(
        ("git", "init", str(package)),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "-C", str(package), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(package), "config", "user.name", "Test"),
        check=True,
    )
    remote = "https://github.com/example/sample-plugin.git"
    subprocess.run(
        ("git", "-C", str(package), "remote", "add", "origin", remote),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(package), "add", "."),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(package), "commit", "-m", "initial"),
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(package), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plugins = registry(tmp_path)

    plugins.install_trusted_git(
        package,
        repository_url=remote,
        commit_sha=commit,
        allowed_hosts=("github.com",),
        approved_by="operator",
        acknowledge_host_authority=True,
    )

    assert plugins.list()[0].source_kind is PluginSourceKind.TRUSTED_GIT
    (package / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(
        PluginRegistrationError,
        match="must be clean",
    ):
        plugins.update_trusted_git(
            package,
            repository_url=remote,
            commit_sha=commit,
            allowed_hosts=("github.com",),
            approved_by="operator",
            acknowledge_host_authority=True,
        )


def test_cli_lifecycle_is_host_only_and_not_registered_as_model_tools(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    package = versioned_plugin(tmp_path / "source", "1.2.3")

    assert (
        main(
            [
                "plugins",
                "install",
                str(package),
                "--approved-by",
                "operator",
                "--acknowledge-host-authority",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["receipt"]["action"] == "install"
    assert (
        main(
            [
                "plugins",
                "uninstall",
                "sample-plugin",
                "--approved-by",
                "operator",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    model_tool_names = {
        tool.name
        for tool in create_default_tool_registry(tmp_path).list_tools()
    }
    assert not {
        "plugins_install",
        "plugins_update",
        "plugins_uninstall",
    } & model_tool_names
