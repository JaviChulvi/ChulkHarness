"""Shared secret redaction for public errors and trace payloads."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any


REDACTED = "[redacted]"
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)


def redact_text(text: str) -> str:
    """Return free-form text with common credential shapes removed."""
    redacted = re.sub(
        r"(?i)\b([a-z0-9_-]*(?:api[_-]?key|authorization|cookie|credential|password|secret|token)[a-z0-9_-]*)"
        r"\s*([:=])\s*['\"]?[^'\"\s,;}]+",
        lambda match: f"{match.group(1)}{match.group(2)} {REDACTED}",
        text,
    )
    redacted = re.sub(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+", f"Bearer {REDACTED}", redacted)
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", REDACTED, redacted)
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
    lowered = key.lower().replace("-", "_")
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)
