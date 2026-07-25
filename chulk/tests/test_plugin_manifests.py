"""Tests for static plugin manifest and package inspection."""

from __future__ import annotations

from pathlib import Path

import pytest

from chulk.plugins import (
    PluginCategory,
    PluginInspectionError,
    PluginManifestError,
    inspect_plugin_directory,
    load_plugin_manifest,
)


def plugin_manifest(
    *,
    name: str = "sample-plugin",
    version: str = "1.2.3",
    extra: str = "",
) -> str:
    return f"""\
schema_version: 1
name: {name}
version: {version}
description: A statically inspected test plugin.
source: local
trust: local
requires_chulk: ">=0.1,<1"
requires_python: ">=3.11"
capabilities:
  - files:read
  - network
secret_refs:
  - env:SAMPLE_TOKEN
network_domains:
  - api.example.com
filesystem:
  - path: workspace
    access: read
entry_points:
  tool:
    sample:
      target: sample_plugin.tools:create_tool
      required_capabilities:
        - files:read
  provider:
    sample-provider: sample_plugin.provider:Provider
dependencies: {{}}
python_dependencies:
  packaging: ">=24"
migrations:
  - migrations/001.sql
{extra}"""


def write_plugin(
    root: Path,
    *,
    manifest: str | None = None,
) -> Path:
    package = root / "sample-plugin"
    (package / "sample_plugin").mkdir(parents=True)
    (package / "migrations").mkdir()
    (package / "chulk-plugin.yaml").write_text(
        manifest or plugin_manifest(),
        encoding="utf-8",
    )
    (package / "sample_plugin" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (package / "sample_plugin" / "tools.py").write_text(
        "def create_tool():\n    return object()\n",
        encoding="utf-8",
    )
    (package / "sample_plugin" / "provider.py").write_text(
        "class Provider:\n    pass\n",
        encoding="utf-8",
    )
    (package / "migrations" / "001.sql").write_text(
        "CREATE TABLE example (id TEXT PRIMARY KEY);\n",
        encoding="utf-8",
    )
    return package


def test_static_inspection_validates_authority_and_compatibility(tmp_path):
    package = write_plugin(tmp_path)

    inspected = inspect_plugin_directory(package)

    assert inspected.compatible is True
    assert inspected.package.manifest.name == "sample-plugin"
    assert inspected.package.digest.startswith("sha256:")
    assert inspected.package.file_count == 5
    assert inspected.package.total_bytes > 0
    assert {
        entry.category for entry in inspected.package.manifest.entry_points
    } == {PluginCategory.PROVIDER, PluginCategory.TOOL}
    assert inspected.package.manifest.secret_refs == ("env:SAMPLE_TOKEN",)
    assert inspected.package.manifest.network_domains == (
        "api.example.com",
    )


def test_static_inspection_does_not_import_plugin_code(tmp_path):
    package = write_plugin(tmp_path)
    sentinel = tmp_path / "imported"
    (package / "sample_plugin" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n",
        encoding="utf-8",
    )

    inspect_plugin_directory(package)

    assert not sentinel.exists()


@pytest.mark.parametrize(
    "replacement, match",
    [
        (
            "entry_points:\n  unsupported:\n    sample: sample_plugin.tools:create_tool\n",
            "unsupported plugin category",
        ),
        (
            "entry_points:\n  tool:\n    sample: os:path\n",
            "not present in the plugin package",
        ),
        (
            "capabilities: []\nentry_points:\n  tool:\n    sample:\n"
            "      target: sample_plugin.tools:create_tool\n"
            "      required_capabilities: [network]\n",
            "undeclared capabilities",
        ),
        (
            "requires_python: definitely-not-a-specifier\n",
            "valid version specifier",
        ),
    ],
)
def test_manifest_rejects_unsupported_or_unsafe_contracts(
    tmp_path,
    replacement,
    match,
):
    base = plugin_manifest()
    if replacement.startswith("requires_python"):
        manifest = base.replace('requires_python: ">=3.11"\n', replacement)
    elif replacement.startswith("capabilities"):
        start = base.index("capabilities:")
        end = base.index("secret_refs:")
        entry_start = base.index("entry_points:")
        entry_end = base.index("dependencies:")
        manifest = (
            base[:start]
            + replacement.split("entry_points:")[0]
            + base[end:entry_start]
            + "entry_points:"
            + replacement.split("entry_points:", 1)[1]
            + base[entry_end:]
        )
    else:
        start = base.index("entry_points:")
        end = base.index("dependencies:")
        manifest = base[:start] + replacement + base[end:]
    package = write_plugin(tmp_path, manifest=manifest)

    with pytest.raises(
        (PluginManifestError, PluginInspectionError),
        match=match,
    ):
        inspect_plugin_directory(package)


@pytest.mark.parametrize(
    "manifest",
    [
        "schema_version: 1\nname: sample-plugin\nname: duplicate\n",
        "schema_version: 1\nname: &name sample-plugin\nversion: 1.0.0\n",
        "schema_version: 1\nname: !python/object sample-plugin\n",
    ],
)
def test_manifest_rejects_duplicate_keys_anchors_and_custom_tags(
    tmp_path,
    manifest,
):
    path = tmp_path / "chulk-plugin.yaml"
    path.write_text(manifest, encoding="utf-8")

    with pytest.raises(PluginManifestError):
        load_plugin_manifest(path)


def test_inspection_rejects_symlinks_and_path_escape(tmp_path):
    package = write_plugin(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    link = package / "sample_plugin" / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(PluginInspectionError, match="symlinks"):
        inspect_plugin_directory(package)


@pytest.mark.parametrize(
    ("relative_path", "match"),
    [
        ("sample_plugin/__pycache__/tools.cpython-312.pyc", "bytecode caches"),
        ("sample_plugin/tools.pyc", "Python bytecode"),
        ("sample_plugin/tools.pyo", "Python bytecode"),
    ],
)
def test_inspection_rejects_unreviewed_python_bytecode(
    tmp_path,
    relative_path,
    match,
):
    package = write_plugin(tmp_path)
    bytecode = package / relative_path
    bytecode.parent.mkdir(parents=True, exist_ok=True)
    bytecode.write_bytes(b"unreviewed executable bytes")

    with pytest.raises(PluginInspectionError, match=match):
        inspect_plugin_directory(package)


def test_inspection_reports_incompatible_runtime_without_importing(tmp_path):
    manifest = plugin_manifest().replace(
        'requires_chulk: ">=0.1,<1"',
        'requires_chulk: ">=99"',
    )
    package = write_plugin(tmp_path, manifest=manifest)

    inspected = inspect_plugin_directory(package)

    assert inspected.compatible is False
    assert inspected.chulk_compatible is False


def test_inspection_rejects_missing_declared_migration(tmp_path):
    package = write_plugin(tmp_path)
    (package / "migrations" / "001.sql").unlink()

    with pytest.raises(PluginInspectionError, match="missing"):
        inspect_plugin_directory(package)
