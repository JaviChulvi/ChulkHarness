"""Owner commands for channel gateway lifecycle, routes, and pairing."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json

from chulk.gateway import SQLiteGatewayLedger, SQLiteGatewayRouter
from chulk.profiles import SQLiteProfileStore


def run_gateway_command(
    command: str,
    *,
    ledger: SQLiteGatewayLedger,
    router: SQLiteGatewayRouter,
    profile_store: SQLiteProfileStore,
    start_func: Callable[[], int],
    adapter: str,
    account_id: str,
    route_command: str | None,
    route_id: str | None,
    profile_id: str | None,
    principal_id: str | None,
    destination_id: str | None,
    thread_id: str | None,
    pairing_ttl_seconds: int,
    include_disabled: bool,
    json_output: bool,
    output_func: Callable[[str], None],
    error_func: Callable[[str], None],
) -> int:
    """Execute one deterministic owner-side gateway operation."""
    try:
        if command == "start":
            if adapter != "telegram":
                raise ValueError("only the telegram adapter is currently available")
            if account_id != "primary":
                raise ValueError("the Telegram gateway currently uses account 'primary'")
            return start_func()
        if command == "status":
            statuses = [
                _status_payload(status)
                for status in ledger.list_adapter_statuses()
                if status.adapter == adapter and status.account_id == account_id
            ]
            payload = {"ok": True, "adapters": statuses}
            output_func(
                _json(payload)
                if json_output
                else _format_status(statuses)
            )
            return 0
        if command == "stop":
            requested = ledger.request_adapter_stop(adapter, account_id)
            payload = {
                "ok": requested,
                "status": "stop_requested" if requested else "not_running",
                "adapter": adapter,
                "account_id": account_id,
            }
            output_func(
                _json(payload)
                if json_output
                else (
                    f"Stop requested for {adapter}/{account_id}."
                    if requested
                    else f"{adapter}/{account_id} is not running."
                )
            )
            return 0 if requested else 2
        if command == "routes":
            return _run_routes(
                router,
                profile_store=profile_store,
                route_command=route_command,
                route_id=route_id,
                adapter=adapter,
                account_id=account_id,
                profile_id=profile_id,
                principal_id=principal_id,
                destination_id=destination_id,
                thread_id=thread_id,
                include_disabled=include_disabled,
                json_output=json_output,
                output_func=output_func,
            )
        if command == "pair":
            if profile_id is None:
                raise ValueError("gateway pair requires --profile")
            profile_store.get(profile_id)
            challenge = router.create_pairing(
                adapter=adapter,
                account_id=account_id,
                profile_id=profile_id,
                principal_id=principal_id,
                ttl_seconds=pairing_ttl_seconds,
            )
            payload = {
                "ok": True,
                "status": "pairing_created",
                "pairing_id": challenge.id,
                "code": challenge.code,
                "adapter": challenge.adapter,
                "account_id": challenge.account_id,
                "profile_id": challenge.profile_id,
                "principal_id": challenge.principal_id,
                "expires_at": challenge.expires_at.isoformat(),
            }
            output_func(
                _json(payload)
                if json_output
                else (
                    f"Pair {adapter}/{account_id} with {profile_id} by sending:\n"
                    f"{challenge.code}\n"
                    f"Expires at {challenge.expires_at.isoformat()}."
                )
            )
            return 0
        raise ValueError(f"unknown gateway command: {command}")
    except (LookupError, OSError, RuntimeError, ValueError) as exc:
        if json_output:
            output_func(
                _json(
                    {
                        "ok": False,
                        "status": "gateway_error",
                        "error": str(exc),
                    }
                )
            )
        else:
            error_func(f"gateway error: {exc}")
        return 2


def _run_routes(
    router: SQLiteGatewayRouter,
    *,
    profile_store: SQLiteProfileStore,
    route_command: str | None,
    route_id: str | None,
    adapter: str,
    account_id: str,
    profile_id: str | None,
    principal_id: str | None,
    destination_id: str | None,
    thread_id: str | None,
    include_disabled: bool,
    json_output: bool,
    output_func: Callable[[str], None],
) -> int:
    if route_command == "list":
        routes: list[dict[str, object]] = [
            {
                "id": route.id,
                "adapter": route.adapter,
                "account_id": route.account_id,
                "profile_id": route.profile_id,
                "principal_id": route.principal_id,
                "destination_id": route.destination_id,
                "thread_id": route.thread_id,
                "enabled": route.enabled,
            }
            for route in router.list_routes(include_disabled=include_disabled)
        ]
        output_func(
            _json({"ok": True, "routes": routes})
            if json_output
            else _format_routes(routes)
        )
        return 0
    if route_command == "add":
        if profile_id is None:
            raise ValueError("gateway routes add requires --profile")
        profile_store.get(profile_id)
        route = router.add_route(
            adapter=adapter,
            account_id=account_id,
            profile_id=profile_id,
            principal_id=principal_id,
            destination_id=destination_id,
            thread_id=thread_id,
        )
        payload = {
            "ok": True,
            "status": "route_added",
            "route_id": route.id,
            "profile_id": route.profile_id,
        }
        output_func(
            _json(payload)
            if json_output
            else f"Route {route.id} now selects profile {route.profile_id}."
        )
        return 0
    if route_command == "remove":
        if route_id is None:
            raise ValueError("gateway routes remove requires a route id")
        removed = router.remove_route(route_id)
        if not removed:
            raise ValueError(f"active route {route_id!r} does not exist")
        output_func(
            _json({"ok": True, "status": "route_removed", "route_id": route_id})
            if json_output
            else f"Disabled route {route_id}."
        )
        return 0
    raise ValueError("gateway routes requires list, add, or remove")


def _status_payload(status) -> dict[str, object]:
    lease_expired = (
        status.lease_until is not None
        and status.lease_until <= datetime.now(timezone.utc)
    )
    return {
        "adapter": status.adapter,
        "account_id": status.account_id,
        "state": "stale" if status.state == "running" and lease_expired else status.state,
        "cursor": status.cursor,
        "lease_until": (
            status.lease_until.isoformat()
            if status.lease_until is not None
            else None
        ),
        "stop_requested": status.stop_requested,
        "legacy_adopted_at": (
            status.legacy_adopted_at.isoformat()
            if status.legacy_adopted_at is not None
            else None
        ),
    }


def _format_status(statuses: list[dict[str, object]]) -> str:
    if not statuses:
        return "No matching gateway adapter state."
    lines = ["Gateway adapters:"]
    for status in statuses:
        lines.append(
            f"  {status['adapter']}/{status['account_id']} "
            f"{status['state']} cursor={status['cursor']} "
            f"stop_requested={status['stop_requested']}"
        )
    return "\n".join(lines)


def _format_routes(routes: list[dict[str, object]]) -> str:
    if not routes:
        return "No gateway routes."
    lines = ["Gateway routes:"]
    for route in routes:
        lines.append(
            f"  {route['id']} {route['adapter']}/{route['account_id']} "
            f"principal={route['principal_id']} "
            f"destination={route['destination_id']} "
            f"thread={route['thread_id']} -> {route['profile_id']} "
            f"enabled={route['enabled']}"
        )
    return "\n".join(lines)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


__all__ = ["run_gateway_command"]
