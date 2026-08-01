"""Security and identity tests for gateway routing and pairing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from chulk.gateway import (
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    InboundEnvelope,
    SQLiteGatewayRouter,
    TextPart,
    TrustLevel,
)


NOW = datetime(2026, 7, 25, 11, tzinfo=timezone.utc)


def _envelope(
    *,
    principal: str = "user-7",
    destination: str = "chat-9",
    scope: ChannelScope = ChannelScope.DIRECT,
    authentication: AuthenticationState = AuthenticationState.AUTHENTICATED,
) -> InboundEnvelope:
    return InboundEnvelope(
        event_id="1",
        idempotency_key="telegram:primary:1",
        identity=ChannelIdentity("telegram", "primary", principal),
        destination_id=destination,
        parts=(TextPart("hello"),),
        scope=scope,
        authentication=authentication,
        trust=TrustLevel.UNTRUSTED,
    )


def test_unknown_or_unauthenticated_identity_cannot_select_a_profile(tmp_path) -> None:
    router = SQLiteGatewayRouter(tmp_path / "control.sqlite")
    router.add_route(
        adapter="telegram",
        account_id="primary",
        principal_id="user-7",
        profile_id="work",
    )

    assert router.resolve(_envelope(principal="unknown")) is None
    assert (
        router.resolve(
            _envelope(authentication=AuthenticationState.UNAUTHENTICATED)
        )
        is None
    )


def test_specific_destination_route_wins_and_groups_require_destination_policy(
    tmp_path,
) -> None:
    router = SQLiteGatewayRouter(tmp_path / "control.sqlite")
    router.add_route(
        adapter="telegram",
        account_id="primary",
        principal_id="user-7",
        profile_id="personal",
    )
    group_route = router.add_route(
        adapter="telegram",
        account_id="primary",
        principal_id="user-7",
        destination_id="group-10",
        profile_id="work",
    )

    assert router.resolve(_envelope()).profile_id == "personal"
    assert (
        router.resolve(
            _envelope(destination="group-9", scope=ChannelScope.GROUP)
        )
        is None
    )
    resolved = router.resolve(
        _envelope(destination="group-10", scope=ChannelScope.GROUP)
    )
    assert resolved is not None
    assert resolved.id == group_route.id


def test_pairing_is_short_lived_bound_to_transport_and_consumed_once(tmp_path) -> None:
    router = SQLiteGatewayRouter(tmp_path / "control.sqlite")
    challenge = router.create_pairing(
        adapter="telegram",
        account_id="primary",
        profile_id="work",
        now=NOW,
        ttl_seconds=10,
    )

    assert router.consume_pairing(
        challenge.code,
        _envelope(principal="user-8"),
        now=NOW + timedelta(seconds=1),
    )
    assert (
        router.consume_pairing(
            challenge.code,
            _envelope(principal="user-9"),
            now=NOW + timedelta(seconds=2),
        )
        is None
    )
    resolved = router.resolve(_envelope(principal="user-8"))
    assert resolved is not None
    assert resolved.profile_id == "work"

    expired = router.create_pairing(
        adapter="telegram",
        account_id="primary",
        profile_id="personal",
        principal_id="user-9",
        now=NOW,
        ttl_seconds=10,
    )
    assert (
        router.consume_pairing(
            expired.code,
            _envelope(principal="user-9"),
            now=NOW + timedelta(seconds=10),
        )
        is None
    )


def test_routes_are_disabled_without_destroying_audit_identity(tmp_path) -> None:
    router = SQLiteGatewayRouter(tmp_path / "control.sqlite")
    route = router.add_route(
        adapter="telegram",
        account_id="primary",
        principal_id="user-7",
        profile_id="work",
    )

    assert router.remove_route(route.id)
    assert router.resolve(_envelope()) is None
    assert router.list_routes() == ()
    retained = router.list_routes(include_disabled=True)
    assert len(retained) == 1
    assert retained[0].id == route.id
    assert not retained[0].enabled
