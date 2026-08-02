"""Typed host resources and application-event publication contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import json
import re
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse

from chulk.redaction import redact_data, redact_text


RESOURCE_SCHEMA_VERSION = 1
MAX_RESOURCE_EXCERPT_CHARS = 2_000
MAX_RESOURCE_METADATA_BYTES = 16_384
MAX_APPLICATION_EVENT_PAYLOAD_BYTES = 16_384
_NAMESPACE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9_-]*)+$")
_EVENT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_RESOURCE_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class ResourcePersistence(StrEnum):
    """Which host lifecycle applies to a resource reference."""

    EPHEMERAL = "ephemeral"
    HOST_MANAGED = "host_managed"
    REFERENCE_ONLY = "reference_only"


@dataclass(frozen=True, slots=True)
class HostResource:
    """Immutable public reference to host evidence or generated output."""

    id: str
    kind: str
    title: str
    source: str
    uri: str | None = None
    excerpt: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    relevance: Mapping[str, Any] = field(default_factory=dict)
    persistence: ResourcePersistence | str = ResourcePersistence.HOST_MANAGED
    schema_version: int = RESOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        resource_id = _required_text(self.id, "resource id", max_chars=200)
        kind = _required_text(self.kind, "resource kind", max_chars=64).lower()
        if _RESOURCE_KIND_PATTERN.fullmatch(kind) is None:
            raise ValueError("resource kind must use lowercase letters, digits, '_' or '-'")
        title = redact_text(_required_text(self.title, "resource title", max_chars=500))
        source = redact_text(_required_text(self.source, "resource source", max_chars=500))
        if self.schema_version != RESOURCE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported resource schema_version: {self.schema_version}"
            )
        uri = _public_uri(self.uri)
        excerpt = redact_text(self.excerpt) if self.excerpt is not None else None
        if excerpt is not None and len(excerpt) > MAX_RESOURCE_EXCERPT_CHARS:
            raise ValueError(
                f"resource excerpt exceeds {MAX_RESOURCE_EXCERPT_CHARS} characters"
            )
        provenance = _safe_mapping(
            self.provenance,
            name="resource provenance",
            max_bytes=MAX_RESOURCE_METADATA_BYTES,
        )
        relevance = _safe_mapping(
            self.relevance,
            name="resource relevance",
            max_bytes=MAX_RESOURCE_METADATA_BYTES,
        )
        object.__setattr__(self, "id", resource_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "excerpt", excerpt)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "relevance", relevance)
        object.__setattr__(self, "persistence", ResourcePersistence(self.persistence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "source": self.source,
            "uri": self.uri,
            "excerpt": self.excerpt,
            "provenance": _plain_mapping(self.provenance),
            "relevance": _plain_mapping(self.relevance),
            "persistence": ResourcePersistence(self.persistence).value,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HostResource":
        return cls(
            id=str(value.get("id") or ""),
            kind=str(value.get("kind") or ""),
            title=str(value.get("title") or ""),
            source=str(value.get("source") or ""),
            uri=value.get("uri") if isinstance(value.get("uri"), str) else None,
            excerpt=(
                value.get("excerpt")
                if isinstance(value.get("excerpt"), str)
                else None
            ),
            provenance=_mapping(value.get("provenance")),
            relevance=_mapping(value.get("relevance")),
            persistence=str(value.get("persistence") or "host_managed"),
            schema_version=int(value.get("schema_version") or 1),
        )


@dataclass(frozen=True, slots=True)
class ApplicationEventSchema:
    """One application event contract registered on a tool definition."""

    namespace: str
    name: str
    version: int
    payload_schema: Mapping[str, Any]
    max_payload_bytes: int = MAX_APPLICATION_EVENT_PAYLOAD_BYTES

    def __post_init__(self) -> None:
        namespace = _namespace(self.namespace)
        name = _event_name(self.name)
        if isinstance(self.version, bool) or self.version < 1:
            raise ValueError("application event schema version must be positive")
        if (
            isinstance(self.max_payload_bytes, bool)
            or self.max_payload_bytes < 1
            or self.max_payload_bytes > MAX_APPLICATION_EVENT_PAYLOAD_BYTES
        ):
            raise ValueError(
                "application event max_payload_bytes must be between 1 and "
                f"{MAX_APPLICATION_EVENT_PAYLOAD_BYTES}"
            )
        schema = _safe_mapping(
            self.payload_schema,
            name="application event JSON schema",
            max_bytes=MAX_APPLICATION_EVENT_PAYLOAD_BYTES,
            redact=False,
        )
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "payload_schema", schema)

    @property
    def key(self) -> tuple[str, str, int]:
        return self.namespace, self.name, self.version


@dataclass(frozen=True, slots=True)
class ApplicationEventIntent:
    """A tool-produced request for ordered application event publication."""

    namespace: str
    name: str
    schema_version: int
    payload: Mapping[str, Any]
    idempotency_key: str

    def __post_init__(self) -> None:
        namespace = _namespace(self.namespace)
        name = _event_name(self.name)
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("application event schema_version must be positive")
        idempotency_key = _required_text(
            self.idempotency_key,
            "application event idempotency_key",
            max_chars=300,
        )
        payload = _safe_mapping(
            self.payload,
            name="application event payload",
            max_bytes=MAX_APPLICATION_EVENT_PAYLOAD_BYTES,
        )
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "idempotency_key", idempotency_key)

    @property
    def schema_key(self) -> tuple[str, str, int]:
        return self.namespace, self.name, self.schema_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "name": self.name,
            "schema_version": self.schema_version,
            "payload": _plain_mapping(self.payload),
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApplicationEventIntent":
        return cls(
            namespace=str(value.get("namespace") or ""),
            name=str(value.get("name") or ""),
            schema_version=int(value.get("schema_version") or 0),
            payload=_mapping(value.get("payload")),
            idempotency_key=str(value.get("idempotency_key") or ""),
        )


def deduplicate_resources(resources: Iterable[HostResource]) -> tuple[HostResource, ...]:
    """Deduplicate identical ids and reject conflicting projections."""
    seen: dict[str, HostResource] = {}
    for resource in resources:
        existing = seen.get(resource.id)
        if existing is not None and existing != resource:
            raise ValueError(f"conflicting host resource id: {resource.id}")
        seen.setdefault(resource.id, resource)
    return tuple(seen.values())


def _namespace(value: object) -> str:
    namespace = _required_text(value, "application event namespace", max_chars=200).lower()
    if _NAMESPACE_PATTERN.fullmatch(namespace) is None:
        raise ValueError("application event namespace must be a dotted lowercase name")
    return namespace


def _event_name(value: object) -> str:
    name = _required_text(value, "application event name", max_chars=200).lower()
    if _EVENT_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("application event name must be lowercase and dot-separated")
    return name


def _required_text(value: object, name: str, *, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    clean = value.strip()
    if len(clean) > max_chars:
        raise ValueError(f"{name} exceeds {max_chars} characters")
    return clean


def _public_uri(value: str | None) -> str | None:
    if value is None:
        return None
    uri = _required_text(value, "resource uri", max_chars=2_000)
    scheme = urlparse(uri).scheme.lower()
    if scheme not in {"https", "http", "urn"}:
        raise ValueError("resource uri must use https, http, or urn")
    return redact_text(uri)


def _safe_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
    max_bytes: int,
    redact: bool = True,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    plain = {str(key): item for key, item in value.items()}
    try:
        encoded = json.dumps(
            plain,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-safe values") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{name} exceeds {max_bytes} bytes")
    safe = redact_data(plain) if redact else plain
    return _freeze_mapping(safe)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): (
                _freeze_mapping(item)
                if isinstance(item, Mapping)
                else tuple(
                    _freeze_mapping(child) if isinstance(child, Mapping) else child
                    for child in item
                )
                if isinstance(item, (list, tuple))
                else item
            )
            for key, item in value.items()
        }
    )


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[str(key)] = _plain_mapping(item)
        elif isinstance(item, tuple):
            result[str(key)] = [
                _plain_mapping(child) if isinstance(child, Mapping) else child
                for child in item
            ]
        else:
            result[str(key)] = item
    return result


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "ApplicationEventIntent",
    "ApplicationEventSchema",
    "HostResource",
    "MAX_APPLICATION_EVENT_PAYLOAD_BYTES",
    "MAX_RESOURCE_EXCERPT_CHARS",
    "RESOURCE_SCHEMA_VERSION",
    "ResourcePersistence",
    "deduplicate_resources",
]
