"""Deterministic usage-ledger CLI output and exports."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from chulk.cli.entrypoints import (
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    json_text,
)
from chulk.usage import (
    ResourceKind,
    UsageGroupBy,
    UsageLedger,
    UsagePage,
    parse_usage_boundary,
)


def run_usage_command(
    command: str,
    *,
    ledger: UsageLedger,
    start: str | None = None,
    end: str | None = None,
    group_by: str | None = None,
    resource_kind: str | None = None,
    channel: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
    output_path: Path | str | None = None,
    export_format: str = "json",
    max_entries: int = 10_000,
    force: bool = False,
    json_output: bool = False,
    output_func: Callable[[str], None] = print,
    error_func: Callable[[str], None] = print,
    clock: Callable[[], datetime] | None = None,
) -> int:
    """Run one profile-owned usage query without exposing credential refs."""
    try:
        start_at = parse_usage_boundary(start) if start is not None else None
        end_at = (
            parse_usage_boundary(end, end=True) if end is not None else None
        )
        kind = ResourceKind(resource_kind) if resource_kind is not None else None
        if command == "today":
            page = ledger.today(
                now=clock() if clock is not None else None,
                resource_kind=kind,
                channel=channel,
                limit=limit,
            )
            return _emit_page(
                "today",
                page,
                json_output=json_output,
                output_func=output_func,
            )
        if command == "range":
            if start_at is None or end_at is None:
                raise ValueError("usage range requires --from and --to")
            page = ledger.query(
                start=start_at,
                end=end_at,
                resource_kind=kind,
                channel=channel,
                limit=limit,
                cursor=cursor,
            )
            return _emit_page(
                "range",
                page,
                json_output=json_output,
                output_func=output_func,
            )
        if command == "group":
            if group_by is None:
                raise ValueError("usage group requires --by")
            groups = ledger.group(
                UsageGroupBy(group_by),
                start=start_at,
                end=end_at,
                resource_kind=kind,
                channel=channel,
                limit=limit,
            )
            payload = {
                "ok": True,
                "profile_id": ledger.profile_id,
                "group_by": group_by,
                "groups": [group.to_dict() for group in groups],
            }
            output_func(
                json_text(payload)
                if json_output
                else _format_groups(group_by, payload["groups"])
            )
            return EXIT_OK
        if command == "export":
            if output_path is None:
                raise ValueError("usage export requires --output")
            destination = ledger.export(
                output_path,
                format=export_format,
                start=start_at,
                end=end_at,
                resource_kind=kind,
                channel=channel,
                max_entries=max_entries,
                force=force,
            )
            payload = {
                "ok": True,
                "profile_id": ledger.profile_id,
                "format": export_format,
                "output_path": str(destination),
            }
            output_func(
                json_text(payload)
                if json_output
                else f"Usage exported to {destination}"
            )
            return EXIT_OK
        raise ValueError(f"Unknown usage command: {command}")
    except (FileExistsError, OSError, ValueError) as exc:
        if json_output:
            output_func(
                json_text(
                    {
                        "ok": False,
                        "status": "usage_error",
                        "error": str(exc),
                    }
                )
            )
        else:
            error_func(f"usage error: {exc}")
        return EXIT_RUNTIME_ERROR


def _emit_page(
    label: str,
    page: UsagePage,
    *,
    json_output: bool,
    output_func: Callable[[str], None],
) -> int:
    summary = _page_summary(page)
    payload = {
        "ok": True,
        "period": label,
        "summary": summary,
        **page.to_dict(),
    }
    output_func(
        json_text(payload)
        if json_output
        else _format_page(label, summary, page)
    )
    return EXIT_OK


def _page_summary(page: UsagePage) -> dict[str, object]:
    known_cost = sum(
        (
            entry.cost.amount
            for entry in page.entries
            if entry.cost.amount is not None
        ),
        start=Decimal(0),
    )
    currencies = {entry.cost.currency for entry in page.entries}
    return {
        "entry_count": len(page.entries),
        "model_calls": sum(
            int(entry.units.get("model_calls", Decimal(0)))
            for entry in page.entries
        ),
        "tool_calls": sum(
            int(entry.units.get("tool_calls", Decimal(0)))
            for entry in page.entries
        ),
        "total_tokens": sum(
            int(entry.units.get("total_tokens", Decimal(0)))
            for entry in page.entries
        ),
        "known_cost": str(known_cost),
        "currency": next(iter(currencies)) if len(currencies) == 1 else "mixed",
        "unknown_cost_entries": sum(
            1
            for entry in page.entries
            if not entry.cost.pricing_known or entry.cost.amount is None
        ),
    }


def _format_page(
    label: str,
    summary: dict[str, object],
    page: UsagePage,
) -> str:
    lines = [
        f"Usage {label}:",
        f"  entries       {summary['entry_count']}",
        f"  model calls   {summary['model_calls']}",
        f"  tool calls    {summary['tool_calls']}",
        f"  tokens        {summary['total_tokens']}",
        f"  known cost    {summary['known_cost']} {summary['currency']}",
        f"  unknown cost  {summary['unknown_cost_entries']} entries",
    ]
    if page.next_cursor is not None:
        lines.append(f"  next cursor   {page.next_cursor}")
    return "\n".join(lines)


def _format_groups(group_by: str, groups: object) -> str:
    values = groups if isinstance(groups, list) else []
    lines = [f"Usage grouped by {group_by}:"]
    if not values:
        lines.append("  no usage")
        return "\n".join(lines)
    for value in values:
        if not isinstance(value, dict):
            continue
        cost = value.get("cost")
        cost_payload = cost if isinstance(cost, dict) else {}
        amount = cost_payload.get("amount")
        currency = cost_payload.get("currency") or "USD"
        lines.append(
            f"  {value.get('key')}: {value.get('entry_count')} entries, "
            f"{value.get('total_tokens')} tokens, "
            f"{amount if amount is not None else 'unknown'} {currency}"
        )
    return "\n".join(lines)


__all__ = ["run_usage_command"]
