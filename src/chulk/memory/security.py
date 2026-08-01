"""Credential detection for durable memory writes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any


class MemorySecretError(ValueError):
    """Raised when a durable memory payload contains credential-like data."""


_REJECTION_MESSAGE = "Credential-like data is not allowed in durable memory."
_SECRET_FIELD_RE = re.compile(
    r"(?ix)^(?:"
    r"api[_-]?key|api[_-]?keys|access[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"auth[_-]?token|authorization|bearer[_-]?token|client[_-]?secret|cookie|credentials?|"
    r"password|passwd|pwd|private[_-]?key|secret|secret[_-]?key(?:[_-]?base)?|token|"
    r"[a-z0-9]+(?:[_-][a-z0-9]+)*[_-](?:api[_-]?key|access[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|auth[_-]?token|client[_-]?secret|credential|password|passwd|pwd|"
    r"private[_-]?key|secret|secret[_-]?key(?:[_-]?base)?|token)"
    r")$"
)
_SECRET_LABEL = (
    r"(?:(?:[a-z0-9]+[_-])*(?:api[_ -]?key|access[_ -]?key|access[_ -]?token|"
    r"refresh[_ -]?token|auth[_ -]?token|authorization|bearer[_ -]?token|"
    r"client[_ -]?secret|cookie|credential|password|passwd|pwd|private[_ -]?key|"
    r"secret|secret[_ -]?key(?:[_ -]?base)?|token)|[a-z0-9]*(?:apikey|accesskey|"
    r"accesstoken|refreshtoken|authtoken|clientsecret|privatekey|secretkeybase))"
)
_ASSIGNMENT_RE = re.compile(
    rf"(?ix)\b(?P<label>{_SECRET_LABEL})\s*(?P<operator>[:=]|\bis\b)\s*"
    r"(?P<value>'[^'\n]{1,512}'|\"[^\"\n]{1,512}\"|[^\s,;}]{1,512})"
)
_URL_CREDENTIAL_RE = re.compile(
    r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:(?P<password>[^@\s/]+)@"
)
_KNOWN_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{12,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{12,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{12,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)
_SAFE_LITERAL_VALUES = {
    "[redacted]",
    "***",
    "available",
    "bearer",
    "changed",
    "changeme",
    "configured",
    "disabled",
    "encrypted",
    "example",
    "false",
    "generated",
    "hashed",
    "injected",
    "invalid",
    "loaded",
    "managed",
    "masked",
    "missing",
    "needed",
    "none",
    "no",
    "not",
    "not set",
    "null",
    "optional",
    "redacted",
    "refreshed",
    "required",
    "rotated",
    "secure",
    "set",
    "stored",
    "strong",
    "unique",
    "unset",
    "valid",
    "weak",
    "yes",
    "true",
}
_SAFE_REFERENCE_MARKERS = (
    "environment variable",
    "env var",
    "os.getenv(",
    "secret manager",
    "keychain",
    "vault",
)


def ensure_memory_payload_safe(**fields: Any) -> None:
    """Reject a memory payload without including the detected value in errors."""
    if _contains_secret(fields, seen=set()):
        raise MemorySecretError(_REJECTION_MESSAGE)


def _contains_secret(value: Any, *, seen: set[int]) -> bool:
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        try:
            for raw_key, item in value.items():
                key = str(raw_key)
                if _text_contains_secret(key):
                    return True
                if _is_secret_field(key) and _secret_field_value_is_set(item, seen=seen):
                    return True
                if _contains_secret(item, seen=seen):
                    return True
        finally:
            seen.remove(identity)
        return False

    if isinstance(value, (list, tuple, set, frozenset)):
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        try:
            return any(_contains_secret(item, seen=seen) for item in value)
        finally:
            seen.remove(identity)

    if isinstance(value, Path):
        return _text_contains_secret(str(value))
    if isinstance(value, str):
        return _text_contains_secret(value)
    return False


def _secret_field_value_is_set(value: Any, *, seen: set[int]) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Path):
        return not _is_safe_reference(str(value))
    if isinstance(value, str):
        return not _is_safe_reference(value)
    if isinstance(value, Mapping):
        return _contains_secret(value, seen=seen)
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_secret_field_value_is_set(item, seen=seen) for item in value)
    return False


def _text_contains_secret(text: str) -> bool:
    if any(pattern.search(text) for pattern in _KNOWN_SECRET_PATTERNS):
        return True

    for match in _ASSIGNMENT_RE.finditer(text):
        if not _is_safe_reference(match.group("value")):
            return True

    for match in _URL_CREDENTIAL_RE.finditer(text):
        if not _is_safe_reference(match.group("password")):
            return True
    return False


def _is_secret_field(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9_-]+", "_", key.lower()).strip("_-")
    if _SECRET_FIELD_RE.fullmatch(normalized):
        return True
    compact = normalized.replace("_", "").replace("-", "")
    return any(
        compact.endswith(suffix)
        for suffix in (
            "apikey",
            "accesskey",
            "accesstoken",
            "refreshtoken",
            "authtoken",
            "clientsecret",
            "password",
            "privatekey",
        )
    )


def _is_safe_reference(value: str) -> bool:
    stripped = value.strip().strip("'\"").strip(".,!?")
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered in _SAFE_LITERAL_VALUES:
        return True
    if lowered.startswith("$"):
        return True
    if lowered.startswith(("{{", "<")) and stripped.endswith(("}", ">")):
        return True
    if lowered.startswith(("env[", "env.", "env:", "settings.", "config.")):
        return True
    return any(
        lowered == marker
        or lowered.startswith(f"{marker}:")
        or f" {marker}" in lowered
        for marker in _SAFE_REFERENCE_MARKERS
    )


__all__ = ["MemorySecretError", "ensure_memory_payload_safe"]
