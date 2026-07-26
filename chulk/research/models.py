"""Shared trust, provenance, and budget contracts for external research tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import ipaddress
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import SplitResult, urlsplit


DEFAULT_ALLOWED_PORTS = frozenset({80, 443})
DEFAULT_FETCH_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/pdf",
        "application/xhtml+xml",
        "application/xml",
        "text/html",
        "text/plain",
        "text/xml",
    }
)


class ExternalTrust(StrEnum):
    """Trust assigned to content outside the configured project/profile."""

    UNTRUSTED = "untrusted"
    REVIEWED = "reviewed"


class ResearchPolicyError(PermissionError):
    """A request was rejected before an external side effect."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ResearchLimitError(RuntimeError):
    """A bounded research operation exhausted a configured limit."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DomainPolicy:
    """Host-owned URL policy applied to fetches and every browser request."""

    allowed_domains: tuple[str, ...] = ()
    denied_domains: tuple[str, ...] = ()
    allowed_ports: frozenset[int] = DEFAULT_ALLOWED_PORTS
    allow_http: bool = True
    allow_https: bool = True
    block_private_networks: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_domains",
            tuple(_normalize_domain_pattern(item) for item in self.allowed_domains),
        )
        object.__setattr__(
            self,
            "denied_domains",
            tuple(_normalize_domain_pattern(item) for item in self.denied_domains),
        )
        ports = frozenset(self.allowed_ports)
        if not ports or any(
            isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
            for port in ports
        ):
            raise ValueError("allowed_ports must contain ports between 1 and 65535")
        object.__setattr__(self, "allowed_ports", ports)
        if not self.allow_http and not self.allow_https:
            raise ValueError("at least one HTTP scheme must be allowed")

    def validate_url(self, value: str) -> SplitResult:
        """Validate syntax and host policy before DNS resolution."""
        if not isinstance(value, str) or not value.strip():
            raise ResearchPolicyError("URL cannot be empty", code="invalid_url")
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ResearchPolicyError(
                "URL contains unsafe control characters",
                code="invalid_url",
            )
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"}:
            raise ResearchPolicyError(
                "Only HTTP and HTTPS URLs are allowed",
                code="scheme_denied",
            )
        if parsed.scheme == "http" and not self.allow_http:
            raise ResearchPolicyError("HTTP URLs are denied", code="scheme_denied")
        if parsed.scheme == "https" and not self.allow_https:
            raise ResearchPolicyError("HTTPS URLs are denied", code="scheme_denied")
        if parsed.username is not None or parsed.password is not None:
            raise ResearchPolicyError(
                "URLs containing credentials are denied",
                code="credentials_denied",
            )
        try:
            host = (parsed.hostname or "").encode("idna").decode("ascii").lower()
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except (UnicodeError, ValueError) as exc:
            raise ResearchPolicyError("URL host or port is invalid", code="invalid_url") from exc
        if not host:
            raise ResearchPolicyError("URL host cannot be empty", code="invalid_url")
        if port not in self.allowed_ports:
            raise ResearchPolicyError(
                f"Network port {port} is not allowed",
                code="port_denied",
            )
        if any(_domain_matches(host, pattern) for pattern in self.denied_domains):
            raise ResearchPolicyError("URL domain is denied", code="domain_denied")
        if self.allowed_domains and not any(
            _domain_matches(host, pattern) for pattern in self.allowed_domains
        ):
            raise ResearchPolicyError(
                "URL domain is outside the allowlist",
                code="domain_not_allowed",
            )
        return parsed

    def validate_addresses(self, addresses: tuple[str, ...]) -> tuple[str, ...]:
        """Reject local, private, reserved, and otherwise non-global DNS results."""
        if not addresses:
            raise ResearchPolicyError(
                "URL host did not resolve to an address",
                code="dns_resolution_failed",
            )
        normalized: list[str] = []
        for value in addresses:
            try:
                address = ipaddress.ip_address(value)
            except ValueError as exc:
                raise ResearchPolicyError(
                    "URL host resolved to an invalid address",
                    code="dns_resolution_failed",
                ) from exc
            if self.block_private_networks and not address.is_global:
                raise ResearchPolicyError(
                    "URL resolves to a private or non-public network",
                    code="private_network_denied",
                )
            normalized.append(address.compressed)
        return tuple(dict.fromkeys(normalized))


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    """Safety and extraction limits for one bounded HTTP client."""

    domains: DomainPolicy = field(default_factory=DomainPolicy)
    allowed_content_types: frozenset[str] = DEFAULT_FETCH_CONTENT_TYPES
    timeout_seconds: float = 15.0
    max_redirects: int = 5
    max_response_bytes: int = 2 * 1024 * 1024
    max_decompressed_bytes: int = 8 * 1024 * 1024
    max_extracted_chars: int = 40_000

    def __post_init__(self) -> None:
        content_types = frozenset(
            value.strip().lower().split(";", 1)[0]
            for value in self.allowed_content_types
            if value.strip()
        )
        if not content_types:
            raise ValueError("allowed_content_types cannot be empty")
        object.__setattr__(self, "allowed_content_types", content_types)
        if self.timeout_seconds <= 0 or self.timeout_seconds > 120:
            raise ValueError("timeout_seconds must be between 0 and 120")
        if not 0 <= self.max_redirects <= 20:
            raise ValueError("max_redirects must be between 0 and 20")
        for name in (
            "max_response_bytes",
            "max_decompressed_bytes",
            "max_extracted_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Stable evidence attached to one fetched or browser-derived source."""

    requested_url: str
    final_url: str
    retrieved_at: str
    sha256: str
    content_type: str
    byte_length: int
    trust: ExternalTrust = ExternalTrust.UNTRUSTED
    redirects: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        digest = self.sha256.strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("sha256 must be a hexadecimal SHA-256 digest")
        object.__setattr__(self, "sha256", digest)
        if self.byte_length < 0:
            raise ValueError("byte_length cannot be negative")
        parsed_time = datetime.fromisoformat(self.retrieved_at)
        if parsed_time.tzinfo is None:
            raise ValueError("retrieved_at must include a timezone")
        object.__setattr__(self, "trust", ExternalTrust(self.trust))
        object.__setattr__(self, "redirects", tuple(self.redirects))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def now(
        cls,
        *,
        requested_url: str,
        final_url: str,
        sha256: str,
        content_type: str,
        byte_length: int,
        redirects: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> SourceProvenance:
        return cls(
            requested_url=requested_url,
            final_url=final_url,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            sha256=sha256,
            content_type=content_type,
            byte_length=byte_length,
            redirects=redirects,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "retrieved_at": self.retrieved_at,
            "sha256": self.sha256,
            "content_type": self.content_type,
            "byte_length": self.byte_length,
            "trust": self.trust.value,
            "redirects": list(self.redirects),
            "metadata": dict(self.metadata),
        }


def _normalize_domain_pattern(value: str) -> str:
    clean = value.strip().lower().rstrip(".")
    wildcard = clean.startswith("*.")
    host = clean[2:] if wildcard else clean
    if not host or "/" in host or ":" in host or any(char.isspace() for char in host):
        raise ValueError("domain patterns must be plain hostnames or *.host wildcards")
    try:
        normalized = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("domain pattern is invalid") from exc
    return f"*.{normalized}" if wildcard else normalized


def _domain_matches(host: str, pattern: str) -> bool:
    if pattern.startswith("*."):
        suffix = pattern[2:]
        return host.endswith(f".{suffix}") and host != suffix
    return host == pattern or host.endswith(f".{pattern}")


__all__ = [
    "DEFAULT_ALLOWED_PORTS",
    "DEFAULT_FETCH_CONTENT_TYPES",
    "DomainPolicy",
    "ExternalTrust",
    "FetchPolicy",
    "ResearchLimitError",
    "ResearchPolicyError",
    "SourceProvenance",
]
