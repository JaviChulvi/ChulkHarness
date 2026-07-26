"""Pure recurrence calculation for durable automation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from zoneinfo import ZoneInfo

from croniter import CroniterBadCronError, croniter
from dateutil.rrule import rrulestr

from chulk.scheduling.models import (
    AmbiguousTimePolicy,
    MisfirePolicy,
    NonexistentTimePolicy,
    RecurrenceKind,
    RecurrenceSpec,
)


@dataclass(frozen=True, slots=True)
class DueOccurrences:
    """Bounded work selected for one due definition."""

    occurrences: tuple[datetime, ...]
    next_at: datetime | None
    skipped: int = 0


class RecurrenceCalculator:
    """Calculate occurrences without reading SQLite or the wall clock."""

    def validate(self, recurrence: RecurrenceSpec, *, anchor: datetime) -> None:
        anchor = _utc(anchor)
        if recurrence.kind is RecurrenceKind.CRON:
            assert recurrence.cron is not None
            if not croniter.is_valid(recurrence.cron, strict=True):
                raise ValueError("invalid cron expression")
            self.next_after(recurrence, anchor - timedelta(minutes=1), anchor=anchor)
        elif recurrence.kind is RecurrenceKind.RRULE:
            self._rrule(recurrence, anchor)

    def next_after(
        self,
        recurrence: RecurrenceSpec,
        after: datetime,
        *,
        anchor: datetime,
        identity: str = "",
    ) -> datetime | None:
        """Return the first jittered occurrence strictly after ``after``."""
        after = _utc(after)
        anchor = _utc(anchor)
        candidate_after = after
        for _ in range(256):
            base = self._next_base(recurrence, candidate_after, anchor=anchor)
            if base is None:
                return None
            candidate = self._jitter(recurrence, base, identity=identity)
            if recurrence.starts_at is not None and candidate < recurrence.starts_at:
                candidate_after = recurrence.starts_at - timedelta(microseconds=1)
                continue
            if recurrence.ends_at is not None and candidate > recurrence.ends_at:
                return None
            if candidate > after:
                return candidate
            candidate_after = base
        raise ValueError("recurrence did not advance after 256 occurrences")

    def due(
        self,
        recurrence: RecurrenceSpec,
        scheduled_for: datetime,
        now: datetime,
        *,
        anchor: datetime,
        identity: str = "",
    ) -> DueOccurrences:
        """Apply a job's explicit misfire policy to its due occurrences."""
        scheduled_for = _utc(scheduled_for)
        now = _utc(now)
        anchor = _utc(anchor)
        if scheduled_for > now:
            return DueOccurrences((), scheduled_for)
        if recurrence.ends_at is not None and scheduled_for > recurrence.ends_at:
            return DueOccurrences((), None, skipped=1)
        candidates = [scheduled_for]
        cursor = scheduled_for
        scan_limit = max(2, recurrence.max_catch_up + 2)
        while len(candidates) < scan_limit:
            next_at = self.next_after(
                recurrence,
                cursor,
                anchor=anchor,
                identity=identity,
            )
            if next_at is None or next_at > now:
                break
            candidates.append(next_at)
            cursor = next_at
        next_at = self.next_after(
            recurrence,
            candidates[-1],
            anchor=anchor,
            identity=identity,
        )
        lateness = (now - scheduled_for).total_seconds()
        if lateness <= recurrence.misfire_grace_seconds:
            return DueOccurrences((scheduled_for,), next_at)
        if recurrence.misfire_policy is MisfirePolicy.SKIP:
            future = self.next_after(
                recurrence,
                now,
                anchor=anchor,
                identity=identity,
            )
            return DueOccurrences((), future, skipped=len(candidates))
        if recurrence.misfire_policy is MisfirePolicy.RUN_ONCE:
            future = self.next_after(
                recurrence,
                now,
                anchor=anchor,
                identity=identity,
            )
            return DueOccurrences(
                (candidates[-1],), future, skipped=len(candidates) - 1
            )
        selected = tuple(candidates[: recurrence.max_catch_up])
        if len(candidates) > len(selected):
            next_at = candidates[len(selected)]
        return DueOccurrences(
            selected,
            next_at,
            skipped=max(0, len(candidates) - len(selected)),
        )

    def _next_base(
        self,
        recurrence: RecurrenceSpec,
        after: datetime,
        *,
        anchor: datetime,
    ) -> datetime | None:
        if recurrence.kind is RecurrenceKind.ONCE:
            return anchor if anchor > after else None
        if recurrence.kind is RecurrenceKind.INTERVAL:
            assert recurrence.interval_seconds is not None
            if after < anchor:
                return anchor
            elapsed = (after - anchor).total_seconds()
            steps = int(elapsed // recurrence.interval_seconds) + 1
            return anchor + timedelta(seconds=steps * recurrence.interval_seconds)
        local_timezone = ZoneInfo(recurrence.timezone_name)
        local_after = after.astimezone(local_timezone)
        if recurrence.kind is RecurrenceKind.CRON:
            assert recurrence.cron is not None
            cursor = local_after
            for _ in range(8):
                try:
                    value = croniter(recurrence.cron, cursor).get_next(datetime)
                except CroniterBadCronError as exc:
                    raise ValueError("invalid cron expression") from exc
                aware = _apply_ambiguous_policy(
                    _aware(value, local_timezone),
                    recurrence.ambiguous_time_policy,
                )
                if (
                    recurrence.nonexistent_time_policy is NonexistentTimePolicy.SKIP
                    and not croniter.match(
                        recurrence.cron,
                        aware.replace(tzinfo=None),
                    )
                ):
                    cursor = aware
                    continue
                return aware.astimezone(timezone.utc)
            raise ValueError("cron recurrence could not resolve a DST occurrence")
        rule = self._rrule(recurrence, anchor)
        cursor = local_after
        for _ in range(8):
            value = rule.after(cursor, inc=False)
            if value is None:
                return None
            aware = _aware(value, local_timezone)
            normalized = aware.astimezone(timezone.utc).astimezone(local_timezone)
            imaginary = normalized.replace(tzinfo=None) != aware.replace(tzinfo=None)
            if imaginary:
                if recurrence.nonexistent_time_policy is NonexistentTimePolicy.SKIP:
                    cursor = aware
                    continue
                aware = normalized
            aware = _apply_ambiguous_policy(
                aware,
                recurrence.ambiguous_time_policy,
            )
            return aware.astimezone(timezone.utc)
        raise ValueError("RRULE could not resolve a DST occurrence")

    def _rrule(self, recurrence: RecurrenceSpec, anchor: datetime):
        assert recurrence.rrule is not None
        value = recurrence.rrule.strip()
        if "\n" not in value and not value.upper().startswith("RRULE:"):
            value = f"RRULE:{value}"
        local_timezone = ZoneInfo(recurrence.timezone_name)
        try:
            return rrulestr(
                value,
                dtstart=anchor.astimezone(local_timezone),
                forceset=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid RFC 5545 RRULE") from exc

    @staticmethod
    def _jitter(
        recurrence: RecurrenceSpec,
        value: datetime,
        *,
        identity: str,
    ) -> datetime:
        if recurrence.jitter_seconds == 0:
            return value
        digest = sha256(f"{identity}:{value.isoformat()}".encode()).digest()
        seconds = int.from_bytes(digest[:8], "big") % (recurrence.jitter_seconds + 1)
        return value + timedelta(seconds=seconds)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("recurrence datetimes must be timezone-aware")
    return value.astimezone(timezone.utc)


def _aware(value: datetime, timezone_value: ZoneInfo) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone_value)


def _apply_ambiguous_policy(
    value: datetime,
    policy: AmbiguousTimePolicy,
) -> datetime:
    earliest = value.replace(fold=0)
    latest = value.replace(fold=1)
    if earliest.utcoffset() == latest.utcoffset():
        return value
    return latest if policy is AmbiguousTimePolicy.LATEST else earliest


__all__ = ["DueOccurrences", "RecurrenceCalculator"]
