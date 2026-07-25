"""Owner-managed identity routing and one-time channel pairing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import secrets
import sqlite3
from uuid import uuid4

from chulk.gateway.models import (
    AuthenticationState,
    ChannelScope,
    InboundEnvelope,
)
from chulk.profiles.store import CONTROL_MIGRATIONS
from chulk.storage import initialize_sqlite_database, sqlite_connection


@dataclass(frozen=True, slots=True)
class GatewayRoute:
    id: str
    adapter: str
    account_id: str
    profile_id: str
    principal_id: str | None = None
    destination_id: str | None = None
    thread_id: str | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class PairingChallenge:
    id: str
    code: str
    adapter: str
    account_id: str
    profile_id: str
    principal_id: str | None
    expires_at: datetime


class SQLiteGatewayRouter:
    """Resolve authenticated channel identities without profile self-selection."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        initialize_sqlite_database(self.db_path, migrations=CONTROL_MIGRATIONS)

    def add_route(
        self,
        *,
        adapter: str,
        account_id: str,
        profile_id: str,
        principal_id: str | None = None,
        destination_id: str | None = None,
        thread_id: str | None = None,
    ) -> GatewayRoute:
        """Add or replace one explicit owner-approved route."""
        adapter = _required(adapter, "adapter")
        account_id = _required(account_id, "account_id")
        profile_id = _required(profile_id, "profile_id")
        principal = _optional(principal_id)
        destination = _optional(destination_id)
        thread = _optional(thread_id)
        if not principal and not destination:
            raise ValueError("a route requires a principal_id or destination_id")
        observed = _utc_now()
        route_id = uuid4().hex
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO gateway_routes (
                    id, adapter, account_id, principal_id, destination_id,
                    thread_id, profile_id, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(
                    adapter, account_id, principal_id, destination_id, thread_id
                ) DO UPDATE SET
                    profile_id = excluded.profile_id,
                    enabled = 1,
                    updated_at = excluded.updated_at
                """,
                (
                    route_id,
                    adapter,
                    account_id,
                    principal,
                    destination,
                    thread,
                    profile_id,
                    _encode(observed),
                    _encode(observed),
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM gateway_routes
                WHERE adapter = ? AND account_id = ? AND principal_id = ?
                  AND destination_id = ? AND thread_id = ?
                """,
                (adapter, account_id, principal, destination, thread),
            ).fetchone()
        assert row is not None
        return _row_to_route(row)

    def remove_route(self, route_id: str) -> bool:
        """Disable a route while retaining owner audit evidence."""
        with sqlite_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE gateway_routes
                SET enabled = 0, updated_at = ?
                WHERE id = ? AND enabled = 1
                """,
                (_encode(_utc_now()), route_id),
            )
        return cursor.rowcount == 1

    def list_routes(self, *, include_disabled: bool = False) -> tuple[GatewayRoute, ...]:
        clause = "" if include_disabled else "WHERE enabled = 1"
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM gateway_routes
                {clause}
                ORDER BY adapter, account_id, principal_id, destination_id, thread_id
                """
            ).fetchall()
        return tuple(_row_to_route(row) for row in rows)

    def resolve(self, envelope: InboundEnvelope) -> GatewayRoute | None:
        """Resolve only trusted, authenticated identities through owner routes."""
        if envelope.authentication is not AuthenticationState.AUTHENTICATED:
            return None
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM gateway_routes
                WHERE adapter = ? AND account_id = ? AND enabled = 1
                  AND principal_id IN ('', ?)
                  AND destination_id IN ('', ?)
                  AND thread_id IN ('', ?)
                ORDER BY
                    (principal_id != '') DESC,
                    (destination_id != '') DESC,
                    (thread_id != '') DESC,
                    id
                """,
                (
                    envelope.identity.adapter,
                    envelope.identity.account_id,
                    envelope.identity.principal_id,
                    envelope.destination_id,
                    envelope.thread_id or "",
                ),
            ).fetchall()
        for row in rows:
            route = _row_to_route(row)
            if envelope.scope is ChannelScope.GROUP and route.destination_id is None:
                continue
            if route.principal_id is None:
                continue
            return route
        return None

    def create_pairing(
        self,
        *,
        adapter: str,
        account_id: str,
        profile_id: str,
        principal_id: str | None = None,
        ttl_seconds: int = 600,
        now: datetime | None = None,
    ) -> PairingChallenge:
        """Create a high-entropy, short-lived owner-issued pairing challenge."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        observed = _observed(now)
        code = secrets.token_urlsafe(24)
        challenge = PairingChallenge(
            id=uuid4().hex,
            code=code,
            adapter=_required(adapter, "adapter"),
            account_id=_required(account_id, "account_id"),
            profile_id=_required(profile_id, "profile_id"),
            principal_id=_optional(principal_id) or None,
            expires_at=observed + timedelta(seconds=ttl_seconds),
        )
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO gateway_pairings (
                    id, code_digest, adapter, account_id, principal_id,
                    profile_id, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    challenge.id,
                    _digest(code),
                    challenge.adapter,
                    challenge.account_id,
                    challenge.principal_id,
                    challenge.profile_id,
                    _encode(challenge.expires_at),
                    _encode(observed),
                ),
            )
        return challenge

    def consume_pairing(
        self,
        code: str,
        envelope: InboundEnvelope,
        *,
        now: datetime | None = None,
    ) -> GatewayRoute | None:
        """Consume a matching challenge once and bind the authenticated principal."""
        if envelope.authentication is not AuthenticationState.AUTHENTICATED:
            return None
        if envelope.scope is not ChannelScope.DIRECT:
            return None
        observed = _observed(now)
        with sqlite_connection(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM gateway_pairings
                WHERE code_digest = ? AND consumed_at IS NULL
                """,
                (_digest(code),),
            ).fetchone()
            if row is None:
                return None
            if (
                str(row["adapter"]) != envelope.identity.adapter
                or str(row["account_id"]) != envelope.identity.account_id
                or _decode(str(row["expires_at"])) <= observed
                or (
                    row["principal_id"] is not None
                    and str(row["principal_id"]) != envelope.identity.principal_id
                )
            ):
                return None
            consumed = conn.execute(
                """
                UPDATE gateway_pairings
                SET consumed_at = ?
                WHERE id = ? AND consumed_at IS NULL
                """,
                (_encode(observed), row["id"]),
            )
            if consumed.rowcount != 1:
                return None
            route_id = uuid4().hex
            conn.execute(
                """
                INSERT INTO gateway_routes (
                    id, adapter, account_id, principal_id, destination_id,
                    thread_id, profile_id, enabled, created_at, updated_at
                ) VALUES (?, ?, ?, ?, '', '', ?, 1, ?, ?)
                ON CONFLICT(
                    adapter, account_id, principal_id, destination_id, thread_id
                ) DO UPDATE SET
                    profile_id = excluded.profile_id,
                    enabled = 1,
                    updated_at = excluded.updated_at
                """,
                (
                    route_id,
                    envelope.identity.adapter,
                    envelope.identity.account_id,
                    envelope.identity.principal_id,
                    row["profile_id"],
                    _encode(observed),
                    _encode(observed),
                ),
            )
            route_row = conn.execute(
                """
                SELECT * FROM gateway_routes
                WHERE adapter = ? AND account_id = ? AND principal_id = ?
                  AND destination_id = '' AND thread_id = ''
                """,
                (
                    envelope.identity.adapter,
                    envelope.identity.account_id,
                    envelope.identity.principal_id,
                ),
            ).fetchone()
        assert route_row is not None
        return _row_to_route(route_row)


def _row_to_route(row: sqlite3.Row) -> GatewayRoute:
    return GatewayRoute(
        id=str(row["id"]),
        adapter=str(row["adapter"]),
        account_id=str(row["account_id"]),
        profile_id=str(row["profile_id"]),
        principal_id=str(row["principal_id"]) or None,
        destination_id=str(row["destination_id"]) or None,
        thread_id=str(row["thread_id"]) or None,
        enabled=bool(row["enabled"]),
    )


def _required(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned or "\x00" in cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def _optional(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip()
    if "\x00" in cleaned:
        raise ValueError("route identities cannot contain NUL characters")
    return cleaned


def _digest(code: str) -> str:
    cleaned = code.strip()
    if not cleaned:
        return ""
    return sha256(cleaned.encode("utf-8")).hexdigest()


def _observed(value: datetime | None) -> datetime:
    observed = value or _utc_now()
    if observed.tzinfo is None:
        raise ValueError("now must include a timezone")
    return observed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _encode(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _decode(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


__all__ = [
    "GatewayRoute",
    "PairingChallenge",
    "SQLiteGatewayRouter",
]
