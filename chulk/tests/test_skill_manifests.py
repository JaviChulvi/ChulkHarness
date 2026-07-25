"""Tests for strict, backwards-compatible skill package manifests."""

from pathlib import Path

import pytest

from chulk.skills import (
    LEGACY_SKILL_VERSION,
    SkillManifestError,
    SkillRegistry,
    load_skill_package,
    skill_package_digest,
)


def write_package(tmp_path: Path, content: str) -> Path:
    root = tmp_path / "example"
    root.mkdir()
    path = root / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_legacy_skill_uses_compatibility_defaults(tmp_path):
    path = write_package(
        tmp_path,
        "# Example\n\nUse this skill when compatibility is required.\n",
    )

    package = load_skill_package(path)

    assert package.manifest.name == "example"
    assert package.manifest.version == LEGACY_SKILL_VERSION
    assert package.manifest.compatibility_mode is True
    assert package.manifest.entrypoint == "SKILL.md"
    assert package.digest.startswith("sha256:")


def test_explicit_manifest_loads_all_supported_fields(tmp_path):
    path = write_package(
        tmp_path,
        """\
---
schema_version: 1
name: example
version: 1.2.3
description: Example workflow.
platforms: [darwin, linux]
tags: [python]
keywords: [pytest]
required_tools: [read_file]
optional_tools: [shell]
required_capabilities: [tools, vision:image]
forbidden_capabilities: [unsafe]
configuration:
  timeout: 30
secret_refs: [EXAMPLE_TOKEN]
entrypoint: instructions.md
references: [references/guide.md]
scripts: [scripts/check.py]
templates: [templates/report.md]
examples: [examples/example.md]
tests: [tests/test_example.py]
includes: [helper]
source: project
trust: reviewed
---
Package metadata.
""",
    )
    root = path.parent
    resources = (
        "instructions.md",
        "references/guide.md",
        "scripts/check.py",
        "templates/report.md",
        "examples/example.md",
        "tests/test_example.py",
    )
    for resource in resources:
        resource_path = root / resource
        resource_path.parent.mkdir(parents=True, exist_ok=True)
        resource_path.write_text(resource, encoding="utf-8")

    manifest = load_skill_package(path).manifest

    assert manifest.version == "1.2.3"
    assert manifest.platforms == ("darwin", "linux")
    assert manifest.required_tools == ("read_file",)
    assert manifest.required_capabilities == ("tools", "vision:image")
    assert manifest.configuration == {"timeout": 30}
    assert manifest.includes == ("helper",)
    assert manifest.compatibility_mode is False


@pytest.mark.parametrize(
    ("front_matter", "message"),
    [
        ("name: example\nname: duplicate", "duplicate key"),
        ("name: &value example\ndescription: *value", "anchors are not allowed"),
        ("name: !custom example", "custom YAML tag"),
        ("- name\n- example", "must be a mapping"),
        ("schema_version: 2\nname: example", "unsupported skill schema_version"),
        (
            "schema_version: 1\nname: example\nowner: core",
            "unsupported skill manifest fields",
        ),
    ],
)
def test_manifest_rejects_unsafe_or_unsupported_yaml(
    tmp_path,
    front_matter,
    message,
):
    path = write_package(
        tmp_path,
        f"---\n{front_matter}\n---\n# Example\n\nInstructions.\n",
    )

    with pytest.raises(SkillManifestError, match=message):
        load_skill_package(path)


def test_legacy_front_matter_preserves_unknown_fields(tmp_path):
    path = write_package(
        tmp_path,
        """\
---
name: example
description: Example workflow.
owner: core
---
# Example
""",
    )

    package = load_skill_package(path)

    assert package.manifest.compatibility_mode is True
    assert package.manifest.extensions == {"owner": "core"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", "latest", "semantic version"),
        ("platforms", "[plan9]", "unsupported skill platforms"),
        ("required_tools", "[read-file]", "invalid skill required_tools"),
        (
            "required_capabilities",
            "[Vision]",
            "invalid skill required_capabilities",
        ),
        ("secret_refs", "[lowercase]", "invalid skill secret_refs"),
        ("configuration", "[]", "configuration must be a string-keyed mapping"),
        ("references", "[../outside.md]", "unsafe skill references path"),
    ],
)
def test_manifest_validates_typed_fields(tmp_path, field, value, message):
    version = "" if field == "version" else "version: 1.0.0\n"
    path = write_package(
        tmp_path,
        f"""\
---
schema_version: 1
name: example
{version}\
description: Example workflow.
{field}: {value}
---
# Example
""",
    )

    with pytest.raises(SkillManifestError, match=message):
        load_skill_package(path)


def test_manifest_rejects_missing_and_escaped_resources(tmp_path):
    missing = write_package(
        tmp_path,
        """\
---
schema_version: 1
name: example
version: 1.0.0
description: Example workflow.
references: [missing.md]
---
# Example
""",
    )
    with pytest.raises(SkillManifestError, match="escapes or is missing"):
        load_skill_package(missing)

    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    link = missing.parent / "linked.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    missing.write_text(
        missing.read_text(encoding="utf-8").replace("missing.md", "linked.md"),
        encoding="utf-8",
    )
    with pytest.raises(SkillManifestError, match="escapes or is missing"):
        load_skill_package(missing)


def test_package_digest_is_stable_and_changes_with_resources(tmp_path):
    path = write_package(
        tmp_path,
        "# Example\n\nUse this skill when an example is needed.\n",
    )
    reference = path.parent / "guide.md"
    reference.write_text("first", encoding="utf-8")

    first = skill_package_digest(path.parent)
    second = skill_package_digest(path.parent)
    reference.write_text("second", encoding="utf-8")

    assert first == second
    assert skill_package_digest(path.parent) != first


def test_registry_progressively_loads_references_within_budget(tmp_path):
    path = write_package(
        tmp_path,
        """\
---
schema_version: 1
name: example
version: 1.0.0
description: Example workflow.
references: [first.md, second.md]
---
Entry.
""",
    )
    (path.parent / "first.md").write_text("first reference", encoding="utf-8")
    (path.parent / "second.md").write_text("second reference", encoding="utf-8")
    registry = SkillRegistry(tmp_path, max_content_chars=40)
    registry.load_metadata()

    content = registry.load_content("example")
    skill = registry.get_skill("example")

    assert len(content) == 40
    assert "Entry." in content
    assert "first.md" in content
    assert skill is not None
    assert skill.loaded_resources == ["SKILL.md", "first.md"]
