"""Strict static plugin manifest parsing without importing plugin code."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import re
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node

from chulk.profiles import CredentialRef
from chulk.plugins.models import (
    FilesystemAccess,
    PLUGIN_MANIFEST_FILENAME,
    PLUGIN_MANIFEST_SCHEMA_VERSION,
    PluginCategory,
    PluginDependency,
    PluginEntryPoint,
    PluginFilesystemRequirement,
    PluginManifest,
    PluginTrustState,
)


_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "version",
        "description",
        "source",
        "trust",
        "requires_chulk",
        "requires_python",
        "entry_points",
        "capabilities",
        "secret_refs",
        "network_domains",
        "filesystem",
        "dependencies",
        "python_dependencies",
        "migrations",
    }
)
_PLUGIN_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$"
)
_ENTRY_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_TARGET_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*$"
)
_CAPABILITY_PATTERN = re.compile(
    r"^[a-z][a-z0-9_-]*(?::[a-z0-9_-]+)?$"
)
_DOMAIN_PATTERN = re.compile(
    r"^(?:\*\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_DEPENDENCY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_MANIFEST_BYTES = 256_000


class PluginManifestError(ValueError):
    """Raised when plugin metadata cannot be safely inspected."""


class _StrictPluginLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects aliases, anchors, and duplicate keys."""

    def compose_node(self, parent: Node | None, index: int) -> Node:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ComposerError(
                None,
                None,
                "YAML aliases are not allowed in plugin manifests",
                event.start_mark,
            )
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            raise ComposerError(
                None,
                None,
                "YAML anchors are not allowed in plugin manifests",
                event.start_mark,
            )
        node = super().compose_node(parent, index)
        if node is None:
            raise ComposerError(None, None, "plugin YAML node is missing", None)
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
    _loader: _StrictPluginLoader,
    tag_suffix: str,
    node: Node,
) -> object:
    raise ConstructorError(
        None,
        None,
        f"custom YAML tag is not allowed: {tag_suffix or node.tag}",
        node.start_mark,
    )


_StrictPluginLoader.add_multi_constructor("!", _reject_custom_tag)


def load_plugin_manifest(path: Path | str) -> PluginManifest:
    """Parse one manifest without importing or executing plugin code."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / PLUGIN_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PluginManifestError(
            f"plugin manifest is not a regular file: {manifest_path}"
        )
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise PluginManifestError(
            f"plugin manifest exceeds {_MAX_MANIFEST_BYTES} bytes"
        )
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PluginManifestError("plugin manifest must be UTF-8") from exc
    try:
        payload = yaml.load(text, Loader=_StrictPluginLoader)
    except yaml.YAMLError as exc:
        raise PluginManifestError(f"invalid plugin YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise PluginManifestError("plugin manifest must contain a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise PluginManifestError("plugin manifest keys must be strings")
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        clean_key = key.strip().lower().replace("-", "_")
        if clean_key in normalized:
            raise PluginManifestError(
                f"duplicate normalized plugin manifest key: {clean_key}"
            )
        normalized[clean_key] = value
    return plugin_manifest_from_mapping(normalized)


def plugin_manifest_from_mapping(
    value: Mapping[str, Any],
) -> PluginManifest:
    """Validate a manifest mapping from YAML or a reviewed lock snapshot."""
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise PluginManifestError("plugin manifest keys must be strings")
        clean_key = key.strip().lower().replace("-", "_")
        if clean_key in normalized:
            raise PluginManifestError(
                f"duplicate normalized plugin manifest key: {clean_key}"
            )
        normalized[clean_key] = item
    unknown = sorted(set(normalized) - _MANIFEST_FIELDS)
    if unknown:
        raise PluginManifestError(
            f"unsupported plugin manifest fields: {', '.join(unknown)}"
        )
    return _manifest_from_mapping(normalized)


def resolve_plugin_resource(root: Path, value: str) -> Path:
    """Resolve a declared package resource without allowing escape."""
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise PluginManifestError(f"unsafe plugin resource path: {value}")
    try:
        resolved = root.joinpath(*pure.parts).resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise PluginManifestError(
            f"plugin resource escapes or is missing from package root: {value}"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise PluginManifestError(
            f"plugin resource is not a regular file: {value}"
        )
    return resolved


def _manifest_from_mapping(values: dict[str, Any]) -> PluginManifest:
    data = dict(values)
    schema_version = data.pop("schema_version", None)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise PluginManifestError("plugin schema_version must be an integer")
    if schema_version != PLUGIN_MANIFEST_SCHEMA_VERSION:
        raise PluginManifestError(
            f"unsupported plugin schema_version: {schema_version}"
        )
    name = _plugin_name(data.pop("name", None))
    version = _version(data.pop("version", None))
    description = _required_text(
        data.pop("description", None),
        "description",
        max_chars=2_000,
    )
    source = _required_text(
        data.pop("source", "local"),
        "source",
        max_chars=512,
    )
    try:
        trust = PluginTrustState(
            _required_text(data.pop("trust", "untrusted"), "trust")
        )
    except ValueError as exc:
        raise PluginManifestError(
            "plugin trust must be untrusted, local, operator-trusted, or revoked"
        ) from exc
    requires_chulk = _specifier(
        data.pop("requires_chulk", ">=0"),
        "requires_chulk",
    )
    requires_python = _specifier(
        data.pop("requires_python", ">=3.11"),
        "requires_python",
    )
    capabilities = _capabilities(data.pop("capabilities", ()))
    entry_points = _entry_points(
        data.pop("entry_points", None),
        capabilities=capabilities,
    )
    secret_refs = tuple(
        sorted(
            {
                CredentialRef.parse(
                    _required_text(item, "secret_refs item", max_chars=256)
                ).uri
                for item in _sequence(
                    data.pop("secret_refs", ()),
                    "secret_refs",
                )
            }
        )
    )
    network_domains = tuple(
        sorted(
            {
                _domain(item)
                for item in _sequence(
                    data.pop("network_domains", ()),
                    "network_domains",
                )
            }
        )
    )
    filesystem = _filesystem(data.pop("filesystem", ()))
    dependencies = _dependencies(
        data.pop("dependencies", {}),
        "dependencies",
    )
    python_dependencies = _dependencies(
        data.pop("python_dependencies", {}),
        "python_dependencies",
    )
    migrations = tuple(
        dict.fromkeys(
            _resource_path(item, "migrations item", suffix=".sql")
            for item in _sequence(
                data.pop("migrations", ()),
                "migrations",
            )
        )
    )
    return PluginManifest(
        schema_version=schema_version,
        name=name,
        version=version,
        description=description,
        source=source,
        trust=trust,
        requires_chulk=requires_chulk,
        requires_python=requires_python,
        entry_points=entry_points,
        capabilities=capabilities,
        secret_refs=secret_refs,
        network_domains=network_domains,
        filesystem=filesystem,
        dependencies=dependencies,
        python_dependencies=python_dependencies,
        migrations=migrations,
    )


def _entry_points(
    value: object,
    *,
    capabilities: tuple[str, ...],
) -> tuple[PluginEntryPoint, ...]:
    if not isinstance(value, Mapping) or not value:
        raise PluginManifestError("entry_points must be a non-empty mapping")
    entries: list[PluginEntryPoint] = []
    seen: set[tuple[PluginCategory, str]] = set()
    allowed_capabilities = set(capabilities)
    for raw_category, raw_entries in value.items():
        try:
            category = PluginCategory(str(raw_category))
        except ValueError as exc:
            raise PluginManifestError(
                f"unsupported plugin category: {raw_category}"
            ) from exc
        if not isinstance(raw_entries, Mapping) or not raw_entries:
            raise PluginManifestError(
                f"entry_points.{category.value} must be a non-empty mapping"
            )
        for raw_name, raw_entry in raw_entries.items():
            name = _entry_name(raw_name)
            required: tuple[str, ...]
            if isinstance(raw_entry, str):
                target = _target(raw_entry)
                required = ()
            elif isinstance(raw_entry, Mapping):
                unknown = set(raw_entry) - {"target", "required_capabilities"}
                if unknown:
                    raise PluginManifestError(
                        f"unsupported entry point fields for {name}: "
                        f"{', '.join(sorted(str(item) for item in unknown))}"
                    )
                target = _target(raw_entry.get("target"))
                required = _capabilities(
                    raw_entry.get("required_capabilities", ())
                )
            else:
                raise PluginManifestError(
                    f"entry point {name} must be a target string or mapping"
                )
            undeclared: list[str] = sorted(
                set(required) - allowed_capabilities
            )
            if undeclared:
                raise PluginManifestError(
                    f"entry point {name} uses undeclared capabilities: "
                    f"{', '.join(undeclared)}"
                )
            identity = (category, name)
            if identity in seen:
                raise PluginManifestError(
                    f"duplicate plugin entry point: {category.value}:{name}"
                )
            seen.add(identity)
            entries.append(
                PluginEntryPoint(
                    name=name,
                    category=category,
                    target=target,
                    required_capabilities=required,
                )
            )
    return tuple(
        sorted(
            entries,
            key=lambda item: (item.category.value, item.name),
        )
    )


def _filesystem(value: object) -> tuple[PluginFilesystemRequirement, ...]:
    requirements: list[PluginFilesystemRequirement] = []
    for item in _sequence(value, "filesystem"):
        if not isinstance(item, Mapping):
            raise PluginManifestError("filesystem entries must be mappings")
        if set(item) != {"path", "access"}:
            raise PluginManifestError(
                "filesystem entries require only path and access"
            )
        path = _required_text(item["path"], "filesystem path", max_chars=512)
        if "\x00" in path:
            raise PluginManifestError("filesystem path cannot contain NUL")
        try:
            access = FilesystemAccess(str(item["access"]))
        except ValueError as exc:
            raise PluginManifestError(
                "filesystem access must be read or write"
            ) from exc
        requirements.append(
            PluginFilesystemRequirement(path=path, access=access)
        )
    return tuple(
        sorted(
            set(requirements),
            key=lambda item: (item.path, item.access.value),
        )
    )


def _dependencies(
    value: object,
    field_name: str,
) -> tuple[PluginDependency, ...]:
    if not isinstance(value, Mapping):
        raise PluginManifestError(f"{field_name} must be a mapping")
    dependencies: list[PluginDependency] = []
    for raw_name, raw_spec in value.items():
        name = _required_text(raw_name, f"{field_name} name", max_chars=128)
        if not _DEPENDENCY_PATTERN.fullmatch(name):
            raise PluginManifestError(
                f"invalid {field_name} name: {name}"
            )
        optional = False
        if isinstance(raw_spec, Mapping):
            if set(raw_spec) - {"version", "optional"}:
                raise PluginManifestError(
                    f"unsupported {field_name} fields for {name}"
                )
            specifier = _specifier(
                raw_spec.get("version", ""),
                f"{field_name}.{name}.version",
            )
            if not isinstance(raw_spec.get("optional", False), bool):
                raise PluginManifestError(
                    f"{field_name}.{name}.optional must be a boolean"
                )
            optional = bool(raw_spec.get("optional", False))
        else:
            specifier = _specifier(
                raw_spec,
                f"{field_name}.{name}",
            )
        dependencies.append(
            PluginDependency(
                name=name.lower(),
                version_spec=specifier,
                optional=optional,
            )
        )
    return tuple(sorted(dependencies, key=lambda item: item.name))


def _plugin_name(value: object) -> str:
    name = _required_text(value, "name", max_chars=128).lower()
    if not _PLUGIN_NAME_PATTERN.fullmatch(name):
        raise PluginManifestError(
            "plugin name must use lowercase letters, numbers, hyphens, "
            "or underscores"
        )
    return name


def _entry_name(value: object) -> str:
    name = _required_text(value, "entry point name", max_chars=128)
    if not _ENTRY_NAME_PATTERN.fullmatch(name):
        raise PluginManifestError(f"invalid plugin entry point name: {name}")
    return name


def _target(value: object) -> str:
    target = _required_text(value, "entry point target", max_chars=512)
    if not _TARGET_PATTERN.fullmatch(target):
        raise PluginManifestError(
            "entry point target must use module.path:attribute.path"
        )
    return target


def _version(value: object) -> str:
    raw = _required_text(value, "version", max_chars=128)
    try:
        return str(Version(raw))
    except InvalidVersion as exc:
        raise PluginManifestError(
            "plugin version must be a valid PEP 440 version"
        ) from exc


def _specifier(value: object, field_name: str) -> str:
    raw = _required_text(value, field_name, max_chars=256)
    try:
        return str(SpecifierSet(raw))
    except InvalidSpecifier as exc:
        raise PluginManifestError(
            f"{field_name} must be a valid version specifier"
        ) from exc


def _capabilities(value: object) -> tuple[str, ...]:
    capabilities: set[str] = set()
    for item in _sequence(value, "capabilities"):
        capability = _required_text(item, "capability", max_chars=128)
        if not _CAPABILITY_PATTERN.fullmatch(capability):
            raise PluginManifestError(
                f"invalid plugin capability: {capability}"
            )
        capabilities.add(capability)
    return tuple(sorted(capabilities))


def _domain(value: object) -> str:
    domain = _required_text(value, "network domain", max_chars=253).lower()
    if (
        "://" in domain
        or "/" in domain
        or "@" in domain
        or ":" in domain
        or not _DOMAIN_PATTERN.fullmatch(domain)
    ):
        raise PluginManifestError(
            f"network domain must be a hostname without scheme or port: {domain}"
        )
    return domain


def _resource_path(
    value: object,
    field_name: str,
    *,
    suffix: str,
) -> str:
    raw = _required_text(value, field_name, max_chars=512)
    pure = PurePosixPath(raw.replace("\\", "/"))
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or not pure.parts
        or pure.suffix.lower() != suffix
    ):
        raise PluginManifestError(f"unsafe {field_name}: {raw}")
    return pure.as_posix()


def _required_text(
    value: object,
    field_name: str,
    *,
    max_chars: int = 1_000,
) -> str:
    if not isinstance(value, str):
        raise PluginManifestError(f"{field_name} must be a string")
    clean = value.strip()
    if not clean:
        raise PluginManifestError(f"{field_name} cannot be empty")
    if "\x00" in clean:
        raise PluginManifestError(f"{field_name} cannot contain NUL")
    if len(clean) > max_chars:
        raise PluginManifestError(
            f"{field_name} cannot exceed {max_chars} characters"
        )
    return clean


def _sequence(value: object, field_name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise PluginManifestError(f"{field_name} must be a list")
    return tuple(value)


__all__ = [
    "PluginManifestError",
    "load_plugin_manifest",
    "plugin_manifest_from_mapping",
    "resolve_plugin_resource",
]
