"""Metadata-only reviewed plugin catalogs with pinned Git provenance."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

from packaging.version import InvalidVersion, Version

from chulk.plugins.models import (
    PluginCatalogEntry,
    PluginCatalogSnapshot,
    PluginCatalogSource,
    PluginCategory,
    PluginSourceKind,
    PluginTrustState,
)


_CATALOG_ID = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
_PLUGIN_NAME = re.compile(
    r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_AUDIT_STATES = frozenset({"reviewed", "warning", "revoked"})
_MAX_CATALOG_BYTES = 5_000_000
_MAX_ENTRIES = 10_000


class PluginCatalogError(ValueError):
    """Reviewed catalog metadata is invalid or untrusted."""


class ReviewedPluginCatalog:
    """Search an explicitly supplied local snapshot without installing."""

    def __init__(self, snapshot: PluginCatalogSnapshot) -> None:
        self.snapshot = snapshot

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        allowed_git_hosts: tuple[str, ...],
    ) -> ReviewedPluginCatalog:
        catalog_path = Path(path).expanduser()
        if catalog_path.is_symlink() or not catalog_path.is_file():
            raise PluginCatalogError(
                "plugin catalog must be a regular local file"
            )
        if catalog_path.stat().st_size > _MAX_CATALOG_BYTES:
            raise PluginCatalogError(
                f"plugin catalog exceeds {_MAX_CATALOG_BYTES} bytes"
            )
        try:
            payload = json.loads(
                catalog_path.read_text(encoding="utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PluginCatalogError(
                "plugin catalog must be valid UTF-8 JSON"
            ) from exc
        snapshot = _catalog_from_payload(
            payload,
            allowed_git_hosts=allowed_git_hosts,
        )
        return cls(snapshot)

    def search(
        self,
        query: str,
        *,
        category: PluginCategory | str | None = None,
        limit: int = 50,
    ) -> tuple[PluginCatalogEntry, ...]:
        """Return bounded static matches and never install package code."""
        text = query.strip().lower()
        if not text:
            raise PluginCatalogError("catalog query cannot be empty")
        if not 1 <= limit <= 200:
            raise PluginCatalogError("catalog limit must be between 1 and 200")
        selected_category: PluginCategory | None = None
        if category is not None:
            try:
                selected_category = PluginCategory(category)
            except ValueError as exc:
                raise PluginCatalogError(
                    f"unsupported plugin category: {category}"
                ) from exc
        matches: list[PluginCatalogEntry] = []
        for entry in self.snapshot.entries:
            haystack = " ".join(
                (
                    entry.name,
                    entry.version,
                    entry.description,
                    entry.package_source,
                    *entry.capabilities,
                    *entry.network_domains,
                    *(item.value for item in entry.categories),
                )
            ).lower()
            if text not in haystack:
                continue
            if (
                selected_category is not None
                and selected_category not in entry.categories
            ):
                continue
            matches.append(entry)
            if len(matches) == limit:
                break
        return tuple(matches)

    def inspect(
        self,
        name: str,
        *,
        version: str | None = None,
    ) -> PluginCatalogEntry:
        """Return exact static catalog metadata without resolving a package."""
        matches = [
            entry
            for entry in self.snapshot.entries
            if entry.name == name and (
                version is None or entry.version == version
            )
        ]
        if not matches:
            raise PluginCatalogError(
                f"plugin catalog entry does not exist: {name}"
            )
        if len(matches) > 1:
            versions = ", ".join(item.version for item in matches)
            raise PluginCatalogError(
                f"plugin catalog entry is ambiguous; choose a version: "
                f"{versions}"
            )
        return matches[0]


def _catalog_from_payload(
    payload: object,
    *,
    allowed_git_hosts: tuple[str, ...],
) -> PluginCatalogSnapshot:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "catalog_id",
        "source",
        "entries",
    }:
        raise PluginCatalogError(
            "plugin catalog has unsupported top-level fields"
        )
    if payload["schema_version"] != 1:
        raise PluginCatalogError(
            "unsupported plugin catalog schema version"
        )
    catalog_id = _text(payload["catalog_id"], "catalog_id", 128).lower()
    if not _CATALOG_ID.fullmatch(catalog_id):
        raise PluginCatalogError("plugin catalog id is invalid")
    raw_source = payload["source"]
    if not isinstance(raw_source, dict) or set(raw_source) != {
        "repository_url",
        "commit_sha",
        "digest",
        "reviewed_by",
    }:
        raise PluginCatalogError(
            "plugin catalog source metadata is invalid"
        )
    repository_url = _trusted_url(
        _text(raw_source["repository_url"], "repository_url", 2_000),
        allowed_git_hosts=allowed_git_hosts,
    )
    commit = _text(raw_source["commit_sha"], "commit_sha", 40).lower()
    digest = _text(raw_source["digest"], "source digest", 71).lower()
    if not _COMMIT.fullmatch(commit) or not _DIGEST.fullmatch(digest):
        raise PluginCatalogError(
            "catalog source requires an exact commit and sha256 digest"
        )
    source = PluginCatalogSource(
        repository_url=repository_url,
        commit_sha=commit,
        digest=digest,
        reviewed_by=_text(
            raw_source["reviewed_by"],
            "reviewed_by",
            200,
        ),
    )
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > _MAX_ENTRIES:
        raise PluginCatalogError(
            f"plugin catalog entries must be a list of at most "
            f"{_MAX_ENTRIES} items"
        )
    entries = tuple(_entry(item) for item in raw_entries)
    identities = {(item.name, item.version) for item in entries}
    if len(identities) != len(entries):
        raise PluginCatalogError(
            "plugin catalog contains duplicate name/version entries"
        )
    return PluginCatalogSnapshot(
        catalog_id=catalog_id,
        source=source,
        entries=tuple(
            sorted(entries, key=lambda item: (item.name, Version(item.version)))
        ),
    )


def _entry(value: object) -> PluginCatalogEntry:
    fields = {
        "name",
        "version",
        "description",
        "package_source",
        "package_digest",
        "source_kind",
        "trust",
        "capabilities",
        "secret_refs",
        "network_domains",
        "categories",
        "audit_state",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise PluginCatalogError(
            "plugin catalog entry has unsupported fields"
        )
    name = _text(value["name"], "entry name", 128).lower()
    if not _PLUGIN_NAME.fullmatch(name):
        raise PluginCatalogError("plugin catalog entry name is invalid")
    raw_version = _text(value["version"], "entry version", 128)
    try:
        version = str(Version(raw_version))
    except InvalidVersion as exc:
        raise PluginCatalogError(
            "plugin catalog entry version is invalid"
        ) from exc
    digest = _text(
        value["package_digest"],
        "package digest",
        71,
    ).lower()
    if not _DIGEST.fullmatch(digest):
        raise PluginCatalogError(
            "plugin catalog package digest is invalid"
        )
    try:
        source_kind = PluginSourceKind(value["source_kind"])
        trust = PluginTrustState(value["trust"])
    except ValueError as exc:
        raise PluginCatalogError(
            "plugin catalog source kind or trust state is invalid"
        ) from exc
    audit_state = _text(
        value["audit_state"],
        "audit_state",
        32,
    ).lower()
    if audit_state not in _AUDIT_STATES:
        raise PluginCatalogError(
            "plugin catalog audit state is invalid"
        )
    categories: list[PluginCategory] = []
    for item in _strings(value["categories"], "categories"):
        try:
            categories.append(PluginCategory(item))
        except ValueError as exc:
            raise PluginCatalogError(
                f"unsupported plugin category: {item}"
            ) from exc
    return PluginCatalogEntry(
        name=name,
        version=version,
        description=_text(value["description"], "description", 2_000),
        package_source=_text(
            value["package_source"],
            "package_source",
            2_000,
        ),
        package_digest=digest,
        source_kind=source_kind,
        trust=trust,
        capabilities=_strings(value["capabilities"], "capabilities"),
        secret_refs=_strings(value["secret_refs"], "secret_refs"),
        network_domains=_strings(
            value["network_domains"],
            "network_domains",
        ),
        categories=tuple(sorted(set(categories), key=lambda item: item.value)),
        audit_state=audit_state,
    )


def _trusted_url(
    value: str,
    *,
    allowed_git_hosts: tuple[str, ...],
) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "ssh"} or not parsed.hostname:
        raise PluginCatalogError(
            "catalog repository must use https or ssh"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PluginCatalogError(
            "catalog repository cannot contain credentials or query data"
        )
    allowed = {host.lower() for host in allowed_git_hosts}
    if parsed.hostname.lower() not in allowed:
        raise PluginCatalogError(
            "catalog Git host is not explicitly trusted"
        )
    return value


def _strings(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise PluginCatalogError(f"{field_name} must be a string list")
    return tuple(
        sorted(
            {
                _text(item, f"{field_name} item", 512)
                for item in value
            }
        )
    )


def _text(value: Any, field_name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise PluginCatalogError(f"{field_name} must be a string")
    clean = value.strip()
    if (
        not clean
        or "\x00" in clean
        or len(clean) > limit
    ):
        raise PluginCatalogError(f"{field_name} is invalid")
    return clean


__all__ = [
    "PluginCatalogError",
    "ReviewedPluginCatalog",
]
