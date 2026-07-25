"""Strict, backwards-compatible local skill package manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
from typing import Any

import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node


SKILL_MANIFEST_SCHEMA_VERSION = 1
LEGACY_SKILL_VERSION = "0.0.0"
SUPPORTED_SKILL_PLATFORMS = frozenset({"darwin", "linux", "windows"})
SUPPORTED_SKILL_TRUST = frozenset(
    {"bundled", "local", "reviewed", "external", "agent-proposed"}
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "version",
        "description",
        "platforms",
        "tags",
        "keywords",
        "required_tools",
        "optional_tools",
        "required_capabilities",
        "forbidden_capabilities",
        "configuration",
        "secret_refs",
        "entrypoint",
        "references",
        "scripts",
        "templates",
        "examples",
        "tests",
        "includes",
        "source",
        "trust",
    }
)
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?$")
_VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_CAPABILITY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?::[a-z0-9_-]+)?$")
_SECRET_REF_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SkillManifestError(ValueError):
    """Raised when a local skill manifest is unsafe or invalid."""


class _StrictSkillLoader(yaml.SafeLoader):
    """Safe loader that also rejects aliases, anchors, and duplicate keys."""

    def compose_node(self, parent: Node | None, index: int) -> Node:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ComposerError(
                None,
                None,
                "YAML aliases are not allowed in skill manifests",
                event.start_mark,
            )
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise ComposerError(
                None,
                None,
                "YAML anchors are not allowed in skill manifests",
                event.start_mark,
            )
        node = super().compose_node(parent, index)
        if node is None:
            raise ComposerError(None, None, "skill YAML node is missing", None)
        return node

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                "expected a mapping",
                node.start_mark,
            )
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _reject_custom_tag(
    _loader: _StrictSkillLoader,
    tag_suffix: str,
    node: Node,
) -> object:
    raise ConstructorError(
        None,
        None,
        f"custom YAML tag is not allowed: {tag_suffix or node.tag}",
        node.start_mark,
    )


_StrictSkillLoader.add_multi_constructor("!", _reject_custom_tag)


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Validated metadata describing one local skill package."""

    name: str
    version: str
    description: str
    schema_version: int = SKILL_MANIFEST_SCHEMA_VERSION
    platforms: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    optional_tools: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    forbidden_capabilities: tuple[str, ...] = ()
    configuration: dict[str, Any] = field(default_factory=dict)
    secret_refs: tuple[str, ...] = ()
    entrypoint: str = "SKILL.md"
    references: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()
    templates: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    includes: tuple[str, ...] = ()
    source: str = "local"
    trust: str = "local"
    compatibility_mode: bool = False
    extensions: dict[str, Any] = field(default_factory=dict)

    @property
    def declared_resources(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.entrypoint,
                    *self.references,
                    *self.scripts,
                    *self.templates,
                    *self.examples,
                    *self.tests,
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "platforms": list(self.platforms),
            "tags": list(self.tags),
            "keywords": list(self.keywords),
            "required_tools": list(self.required_tools),
            "optional_tools": list(self.optional_tools),
            "required_capabilities": list(self.required_capabilities),
            "forbidden_capabilities": list(self.forbidden_capabilities),
            "configuration": dict(self.configuration),
            "secret_refs": list(self.secret_refs),
            "entrypoint": self.entrypoint,
            "references": list(self.references),
            "scripts": list(self.scripts),
            "templates": list(self.templates),
            "examples": list(self.examples),
            "tests": list(self.tests),
            "includes": list(self.includes),
            "source": self.source,
            "trust": self.trust,
            "compatibility_mode": self.compatibility_mode,
            "extensions": dict(self.extensions),
        }


@dataclass(frozen=True, slots=True)
class SkillPackage:
    """One validated manifest tied to an immutable package digest."""

    root: Path
    manifest_path: Path
    manifest: SkillManifest
    digest: str


def load_skill_package(path: Path | str) -> SkillPackage:
    """Load and validate one `SKILL.md` package without executing resources."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "SKILL.md"
    manifest_path = manifest_path.resolve(strict=True)
    root = manifest_path.parent.resolve(strict=True)
    text = manifest_path.read_text(encoding="utf-8")
    front_matter, body = split_skill_front_matter(text)
    compatibility_mode = "schema_version" not in front_matter
    manifest = _manifest_from_mapping(
        front_matter,
        body=body,
        root=root,
        manifest_path=manifest_path,
        compatibility_mode=compatibility_mode,
    )
    _validate_declared_resources(root, manifest)
    return SkillPackage(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        digest=skill_package_digest(root),
    )


def split_skill_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Return strict YAML metadata and Markdown body."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        ),
        None,
    )
    if closing_index is None:
        raise SkillManifestError("skill YAML front matter is not terminated")
    yaml_text = "\n".join(lines[1:closing_index])
    try:
        payload = yaml.load(yaml_text, Loader=_StrictSkillLoader)
    except yaml.YAMLError as exc:
        raise SkillManifestError(f"invalid skill YAML front matter: {exc}") from exc
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise SkillManifestError("skill YAML front matter must be a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise SkillManifestError("skill manifest keys must be strings")
    normalized = {
        key.strip().lower().replace("-", "_"): value
        for key, value in payload.items()
    }
    return normalized, "\n".join(lines[closing_index + 1 :])


def skill_package_digest(root: Path | str) -> str:
    """Hash all regular package files in stable relative-path order."""
    package_root = Path(root).resolve(strict=True)
    digest = sha256()
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
    )
    for path in files:
        resolved = _resolve_inside_root(package_root, path)
        relative = resolved.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def resolve_skill_resource(root: Path, value: str) -> Path:
    """Resolve a declared resource without allowing absolute or escaped paths."""
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise SkillManifestError(f"unsafe skill resource path: {value}")
    return _resolve_inside_root(root.resolve(strict=True), root.joinpath(*pure.parts))


def _manifest_from_mapping(
    values: dict[str, Any],
    *,
    body: str,
    root: Path,
    manifest_path: Path,
    compatibility_mode: bool,
) -> SkillManifest:
    data = dict(values)
    schema_version = data.pop("schema_version", SKILL_MANIFEST_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise SkillManifestError("skill schema_version must be an integer")
    if schema_version != SKILL_MANIFEST_SCHEMA_VERSION:
        raise SkillManifestError(
            f"unsupported skill schema_version: {schema_version}"
        )
    unknown = sorted(set(data) - (_MANIFEST_FIELDS - {"schema_version"}))
    if unknown and not compatibility_mode:
        raise SkillManifestError(
            f"unsupported skill manifest fields: {', '.join(unknown)}"
        )
    extensions = {
        key: data.pop(key)
        for key in unknown
    }
    name = _skill_name(data.pop("name", root.name))
    description = _required_text(
        "description",
        data.pop("description", None)
        or _extract_legacy_description(body, name),
    )
    version = _required_text(
        "version",
        data.pop("version", LEGACY_SKILL_VERSION),
    )
    if not _VERSION_PATTERN.fullmatch(version):
        raise SkillManifestError("skill version must use semantic version syntax")
    platforms = _string_tuple(data.pop("platforms", ()), "platforms")
    invalid_platforms = sorted(set(platforms) - SUPPORTED_SKILL_PLATFORMS)
    if invalid_platforms:
        raise SkillManifestError(
            f"unsupported skill platforms: {', '.join(invalid_platforms)}"
        )
    required_tools = _validated_names(
        data.pop("required_tools", ()),
        "required_tools",
        _TOOL_NAME_PATTERN,
    )
    optional_tools = _validated_names(
        data.pop("optional_tools", ()),
        "optional_tools",
        _TOOL_NAME_PATTERN,
    )
    required_capabilities = _validated_names(
        data.pop("required_capabilities", ()),
        "required_capabilities",
        _CAPABILITY_PATTERN,
    )
    forbidden_capabilities = _validated_names(
        data.pop("forbidden_capabilities", ()),
        "forbidden_capabilities",
        _CAPABILITY_PATTERN,
    )
    if set(required_capabilities) & set(forbidden_capabilities):
        raise SkillManifestError(
            "required_capabilities and forbidden_capabilities cannot overlap"
        )
    configuration = data.pop("configuration", {})
    if not isinstance(configuration, dict) or any(
        not isinstance(key, str) for key in configuration
    ):
        raise SkillManifestError("skill configuration must be a string-keyed mapping")
    secret_refs = _validated_names(
        data.pop("secret_refs", ()),
        "secret_refs",
        _SECRET_REF_PATTERN,
    )
    entrypoint = _resource_text(
        data.pop(
            "entrypoint",
            manifest_path.relative_to(root).as_posix(),
        ),
        "entrypoint",
    )
    source = _required_text("source", data.pop("source", "local"))
    trust = _required_text("trust", data.pop("trust", "local"))
    if trust not in SUPPORTED_SKILL_TRUST:
        raise SkillManifestError(f"unsupported skill trust value: {trust}")
    return SkillManifest(
        schema_version=schema_version,
        name=name,
        version=version,
        description=description,
        platforms=platforms,
        tags=_string_tuple(data.pop("tags", ()), "tags"),
        keywords=_string_tuple(data.pop("keywords", ()), "keywords"),
        required_tools=required_tools,
        optional_tools=optional_tools,
        required_capabilities=required_capabilities,
        forbidden_capabilities=forbidden_capabilities,
        configuration=dict(configuration),
        secret_refs=secret_refs,
        entrypoint=entrypoint,
        references=_resource_tuple(data.pop("references", ()), "references"),
        scripts=_resource_tuple(data.pop("scripts", ()), "scripts"),
        templates=_resource_tuple(data.pop("templates", ()), "templates"),
        examples=_resource_tuple(data.pop("examples", ()), "examples"),
        tests=_resource_tuple(data.pop("tests", ()), "tests"),
        includes=_validated_names(
            data.pop("includes", ()),
            "includes",
            _SKILL_NAME_PATTERN,
        ),
        source=source,
        trust=trust,
        compatibility_mode=compatibility_mode,
        extensions=extensions,
    )


def _validate_declared_resources(root: Path, manifest: SkillManifest) -> None:
    for resource in manifest.declared_resources:
        path = resolve_skill_resource(root, resource)
        if not path.is_file():
            raise SkillManifestError(
                f"declared skill resource is not a file: {resource}"
            )


def _resolve_inside_root(root: Path, path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise SkillManifestError(
            f"skill resource escapes or is missing from package root: {path}"
        ) from exc
    return resolved


def _skill_name(value: object) -> str:
    name = _required_text("name", value).lower()
    if not _SKILL_NAME_PATTERN.fullmatch(name):
        raise SkillManifestError(
            "skill name must contain only lowercase letters, numbers, hyphens, "
            "or underscores"
        )
    return name


def _required_text(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SkillManifestError(f"skill {label} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = tuple(value)
    else:
        raise SkillManifestError(f"skill {label} must be a string list")
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise SkillManifestError(f"skill {label} entries must be non-empty strings")
        clean = item.strip()
        if clean not in normalized:
            normalized.append(clean)
    return tuple(normalized)


def _validated_names(
    value: object,
    label: str,
    pattern: re.Pattern[str],
) -> tuple[str, ...]:
    values = _string_tuple(value, label)
    invalid = [item for item in values if not pattern.fullmatch(item)]
    if invalid:
        raise SkillManifestError(
            f"invalid skill {label}: {', '.join(invalid)}"
        )
    return values


def _resource_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_resource_text(item, label) for item in _string_tuple(value, label))


def _resource_text(value: object, label: str) -> str:
    text = _required_text(label, value).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise SkillManifestError(f"unsafe skill {label} path: {text}")
    return pure.as_posix()


def _extract_legacy_description(body: str, name: str) -> str:
    lines = [line.strip() for line in body.splitlines()]
    for line in lines:
        if line.lower().startswith("use this skill when"):
            return line
    for line in lines:
        if line and not line.startswith("#") and not line.startswith("-"):
            return line
    return f"Procedural instructions for {name} requests."


__all__ = [
    "LEGACY_SKILL_VERSION",
    "SKILL_MANIFEST_SCHEMA_VERSION",
    "SUPPORTED_SKILL_PLATFORMS",
    "SUPPORTED_SKILL_TRUST",
    "SkillManifest",
    "SkillManifestError",
    "SkillPackage",
    "load_skill_package",
    "resolve_skill_resource",
    "skill_package_digest",
    "split_skill_front_matter",
]
