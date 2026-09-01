"""Shared secret redaction for public errors and trace payloads."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any


REDACTED = "[redacted]"
_SECRET_KEY_MARKERS = (
    "access_key",
    "accesskey",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "passwd",
    "password",
    "private_key",
    "privatekey",
    "pwd",
    "secret",
    "ssh_key",
    "sshkey",
)
_SECRET_LABEL = (
    r"(?:api[_ -]?key|access[_ -]?key|authorization|cookie|credential|password|"
    r"passwd|pwd|private[_ -]?key|secret|ssh[_ -]?key|token|apikey|accesskey|"
    r"privatekey|sshkey)"
)
_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?P<label>(?:[A-Z0-9]+ )*PRIVATE KEY)-----.*?"
    r"(?:-----END (?P=label)-----|\Z)",
    re.DOTALL,
)
_URL_CREDENTIAL_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://[^/\s:@]+:)[^@\s/]+@"
)


def redact_text(text: str) -> str:
    """Return free-form text with common credential shapes removed."""
    redacted = _PEM_PRIVATE_KEY_RE.sub(REDACTED, text)
    redacted = _URL_CREDENTIAL_RE.sub(rf"\1{REDACTED}@", redacted)
    redacted = re.sub(
        r"(?i)\bbearer\s+[a-z0-9._~+/=-]+",
        f"Bearer {REDACTED}",
        redacted,
    )
    redacted = re.sub(
        rf"(?i)\b([a-z0-9_-]*{_SECRET_LABEL}[a-z0-9_-]*)"
        r"\s*([:=])\s*['\"]?[^'\"\s,;}]+",
        lambda match: f"{match.group(1)}{match.group(2)} {REDACTED}",
        redacted,
    )
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", REDACTED, redacted)
    redacted = re.sub(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", REDACTED, redacted)
    return redacted


def redact_data(value: Any) -> Any:
    """Recursively convert a value to redacted, plain log-safe data."""
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            safe[key] = REDACTED if _is_secret_key(key) else redact_data(item)
        return safe
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_data(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Path):
        return redact_text(str(value))
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))


def _is_secret_key(key: str) -> bool:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.casefold()).strip("_")
    if any(marker in normalized for marker in _SECRET_KEY_MARKERS):
        return True
    return normalized == "token" or normalized.endswith("_token")
