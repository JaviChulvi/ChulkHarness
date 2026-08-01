"""Hosted gateway definition resolution and durable submission contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chulk import ExecutionScope
from chulk.authoring import DefinitionStatus
from chulk.gateway import (
    AsyncDurableGatewayRunSubmitter,
    AsyncPublishedDefinitionGatewayResolver,
    AuthenticationState,
    ChannelIdentity,
    ChannelScope,
    DurableGatewayRunSubmitter,
    GatewayRunTarget,
    InboundEnvelope,
    PublishedDefinitionGatewayResolver,
    TextPart,
    TrustLevel,
)
from chulk.runs import AsyncInMemoryRunStore, InMemoryRunStore


def _scope(run_id: str = "run-1") -> ExecutionScope:
    return ExecutionScope(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        actor_id="actor-a",
        agent_id="support",
        agent_version="1.0.0",
        run_id=run_id,
    )


def _envelope() -> InboundEnvelope:
    return InboundEnvelope(
        event_id="event-1",
        idempotency_key="gateway:event-1",
        identity=ChannelIdentity("fake", "primary", "actor-a"),
        destination_id="chat-1",
        parts=(TextPart("hello"),),
        scope=ChannelScope.DIRECT,
        authentication=AuthenticationState.AUTHENTICATED,
        trust=TrustLevel.TRUSTED,
    )


def _record(status: DefinitionStatus = DefinitionStatus.PUBLISHED):
    return SimpleNamespace(
        status=status,
        definition=SimpleNamespace(
            agent_id="support",
            version="1.0.0",
            digest="sha256:definition",
        ),
    )


def test_definition_resolver_requires_one_published_revision() -> None:
    class Definitions:
        def get(self, scope, agent_id, version):
            assert (agent_id, version) == ("support", "1.0.0")
            return _record()

    resolver = PublishedDefinitionGatewayResolver(
        Definitions(),
        lambda route, envelope: _scope(),
    )
    target = resolver.resolve(SimpleNamespace(profile_id="work"), _envelope())

    assert target.scope == _scope()
    assert target.definition_digest == "sha256:definition"

    class RevokedDefinitions:
        def get(self, scope, agent_id, version):
            return _record(DefinitionStatus.REVOKED)

    with pytest.raises(ValueError, match="not published"):
        PublishedDefinitionGatewayResolver(
            RevokedDefinitions(),
            lambda route, envelope: _scope(),
        ).resolve(SimpleNamespace(profile_id="work"), _envelope())


def test_gateway_submission_is_idempotent_at_the_durable_run_owner() -> None:
    store = InMemoryRunStore()
    submitter = DurableGatewayRunSubmitter(store)
    target = GatewayRunTarget(
        scope=_scope(),
        definition_id="support",
        definition_version="1.0.0",
        definition_digest="sha256:definition",
    )

    first = submitter.submit(
        target,
        _envelope(),
        idempotency_key="gateway:event-1",
    )
    duplicate = submitter.submit(
        target,
        _envelope(),
        idempotency_key="gateway:event-1",
    )

    assert first == duplicate == "run-1"
    assert store.get(_scope(), "run-1").metadata["source_event_id"] == "event-1"


@pytest.mark.asyncio
async def test_async_gateway_resolver_and_submitter_are_native() -> None:
    calls: list[str] = []

    class Definitions:
        async def get(self, scope, agent_id, version):
            calls.append("definition")
            return _record()

    async def scope_factory(route, envelope):
        calls.append("scope")
        return _scope("run-async")

    resolver = AsyncPublishedDefinitionGatewayResolver(
        Definitions(),
        scope_factory,
    )
    target = await resolver.resolve(
        SimpleNamespace(profile_id="work"),
        _envelope(),
    )
    store = AsyncInMemoryRunStore()
    submitter = AsyncDurableGatewayRunSubmitter(store)

    run_id = await submitter.submit(
        target,
        _envelope(),
        idempotency_key="gateway:event-1",
    )

    assert calls == ["scope", "definition"]
    assert run_id == "run-async"
