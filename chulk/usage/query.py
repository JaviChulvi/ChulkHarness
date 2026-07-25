"""Profile-owned usage queries, aggregation, and bounded exports."""

from __future__ import annotations

import csv
from datetime import datetime, time, timedelta, timezone, tzinfo
import io
import json
import os
from pathlib import Path
import tempfile

from chulk.usage.models import (
    ResourceKind,
    UsageAggregate,
    UsageEntry,
    UsageGroupBy,
    UsagePage,
    UsageQuery,
)
from chulk.usage.store import SQLiteUsageStore


MAX_EXPORT_ENTRIES = 10_000


class UsageLedger:
    """Safe query facade bound to one profile-owned runtime database."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        profile_id: str = "default",
    ) -> None:
        clean_profile_id = profile_id.strip()
        if not clean_profile_id:
            raise ValueError("usage ledger profile_id cannot be empty")
        self.profile_id = clean_profile_id
        self.store = SQLiteUsageStore(db_path)

    def query(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resource_kind: ResourceKind | None = None,
        channel: str | None = None,
        conversation_id: str | None = None,
        goal_id: str | None = None,
        job_id: str | None = None,
        child_task_id: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> UsagePage:
        return self.store.query(
            UsageQuery(
                profile_id=self.profile_id,
                start=start,
                end=end,
                resource_kind=resource_kind,
                channel=channel,
                conversation_id=conversation_id,
                goal_id=goal_id,
                job_id=job_id,
                child_task_id=child_task_id,
                limit=limit,
                cursor=cursor,
            )
        )

    def today(
        self,
        *,
        now: datetime | None = None,
        timezone_info: tzinfo | None = None,
        resource_kind: ResourceKind | None = None,
        channel: str | None = None,
        limit: int = 100,
    ) -> UsagePage:
        current = now or datetime.now().astimezone()
        if current.tzinfo is None:
            raise ValueError("usage clock must be timezone-aware")
        zone = timezone_info or current.tzinfo
        local_now = current.astimezone(zone)
        start = datetime.combine(local_now.date(), time.min, tzinfo=zone)
        return self.query(
            start=start,
            end=start + timedelta(days=1),
            resource_kind=resource_kind,
            channel=channel,
            limit=limit,
        )

    def group(
        self,
        group_by: UsageGroupBy,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        resource_kind: ResourceKind | None = None,
        channel: str | None = None,
        limit: int = MAX_EXPORT_ENTRIES,
    ) -> tuple[UsageAggregate, ...]:
        return self.store.aggregate(
            UsageQuery(
                profile_id=self.profile_id,
                start=start,
                end=end,
                resource_kind=resource_kind,
                channel=channel,
                limit=limit,
            ),
            group_by=group_by,
        )

    def export(
        self,
        destination: Path | str,
        *,
        format: str,
        start: datetime | None = None,
        end: datetime | None = None,
        resource_kind: ResourceKind | None = None,
        channel: str | None = None,
        max_entries: int = MAX_EXPORT_ENTRIES,
        force: bool = False,
    ) -> Path:
        """Write a bounded credential-free CSV or JSON export atomically."""
        export_format = format.strip().lower()
        if export_format not in {"csv", "json"}:
            raise ValueError("usage export format must be csv or json")
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 1
            or max_entries > MAX_EXPORT_ENTRIES
        ):
            raise ValueError(
                f"max_entries must be between 1 and {MAX_EXPORT_ENTRIES}"
            )
        target = Path(destination).expanduser().resolve()
        if target.exists() and not force:
            raise FileExistsError(target)
        entries = self._bounded_entries(
            start=start,
            end=end,
            resource_kind=resource_kind,
            channel=channel,
            max_entries=max_entries,
        )
        content = (
            _json_export(entries)
            if export_format == "json"
            else _csv_export(entries)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                temporary_path = Path(handle.name)
            temporary_path.chmod(0o600)
            os.replace(temporary_path, target)
            target.chmod(0o600)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return target

    def _bounded_entries(
        self,
        *,
        start: datetime | None,
        end: datetime | None,
        resource_kind: ResourceKind | None,
        channel: str | None,
        max_entries: int,
    ) -> tuple[UsageEntry, ...]:
        page = self.query(
            start=start,
            end=end,
            resource_kind=resource_kind,
            channel=channel,
            limit=max_entries,
        )
        if page.next_cursor is not None:
            raise ValueError(
                "usage export exceeds max_entries; narrow the range or filters"
            )
        return page.entries


def _json_export(entries: tuple[UsageEntry, ...]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "entries": [entry.to_public_dict() for entry in entries],
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _csv_export(entries: tuple[UsageEntry, ...]) -> str:
    output = io.StringIO()
    fieldnames = (
        "id",
        "resource_kind",
        "source_event_id",
        "profile_id",
        "channel",
        "conversation_id",
        "turn_id",
        "goal_id",
        "job_id",
        "child_task_id",
        "purpose",
        "occurred_at",
        "billing_period",
        "provider",
        "model",
        "tool_or_service",
        "model_profile_id",
        "model_calls",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_amount",
        "currency",
        "pricing_known",
        "cost_estimated",
        "cost_reported",
        "usage_estimated",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for entry in entries:
        payload = entry.to_public_dict()
        cost = payload["cost"]
        units = payload["units"]
        writer.writerow(
            {
                key: (
                    units.get(key)
                    if key
                    in {
                        "model_calls",
                        "tool_calls",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    }
                    else cost.get(key.removeprefix("cost_"))
                    if key
                    in {
                        "cost_amount",
                        "cost_estimated",
                        "cost_reported",
                    }
                    else cost.get("currency")
                    if key == "currency"
                    else cost.get("pricing_known")
                    if key == "pricing_known"
                    else payload.get(key)
                )
                for key in fieldnames
            }
        )
    return output.getvalue()


def parse_usage_boundary(value: str, *, end: bool = False) -> datetime:
    """Parse an ISO date or datetime into an aware query boundary."""
    clean = value.strip()
    if not clean:
        raise ValueError("usage date boundary cannot be empty")
    try:
        parsed = datetime.fromisoformat(clean)
    except ValueError as exc:
        raise ValueError(f"invalid ISO usage date or datetime: {value}") from exc
    if "T" not in clean and " " not in clean:
        parsed = datetime.combine(
            parsed.date() + (timedelta(days=1) if end else timedelta()),
            time.min,
            tzinfo=timezone.utc,
        )
    elif parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


__all__ = ["MAX_EXPORT_ENTRIES", "UsageLedger", "parse_usage_boundary"]
