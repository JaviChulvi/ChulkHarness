"""Clock seam for durable run persistence."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time for durable run transitions."""
    return datetime.now(timezone.utc)


__all__ = ["utc_now"]
