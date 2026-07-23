"""Scheduled-job data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ScheduledJob:
    id: str
    adapter: str
    destination_id: str
    prompt: str
    next_run_at: datetime
    interval_seconds: int | None
    status: str
    last_run_at: datetime | None = None
    last_error: str | None = None


__all__ = ["ScheduledJob"]
