"""Hosted gateway durable submission contracts."""
from __future__ import annotations
import pytest
from chulk import ExecutionScope
from chulk.gateway import AsyncDurableGatewayRunSubmitter, AuthenticationState, ChannelIdentity, ChannelScope, DurableGatewayRunSubmitter, GatewayRunTarget, InboundEnvelope, TextPart, TrustLevel
from chulk.runs import AsyncInMemoryRunStore, InMemoryRunStore

def _scope(run_id: str='run-1') -> ExecutionScope:
    return ExecutionScope(tenant_id='tenant-a', workspace_id='workspace-a', actor_id='actor-a', agent_id='support', agent_version='1.0.0', run_id=run_id)

def _envelope() -> InboundEnvelope:
    return InboundEnvelope(event_id='event-1', idempotency_key='gateway:event-1', identity=ChannelIdentity('fake', 'primary', 'actor-a'), destination_id='chat-1', parts=(TextPart('hello'),), scope=ChannelScope.DIRECT, authentication=AuthenticationState.AUTHENTICATED, trust=TrustLevel.TRUSTED)


def test_gateway_submission_is_idempotent_at_the_durable_run_owner() -> None:
    store = InMemoryRunStore()
    submitter = DurableGatewayRunSubmitter(store)
    target = GatewayRunTarget(scope=_scope(), definition_id='support', definition_version='1.0.0', definition_digest='sha256:definition')
    first = submitter.submit(target, _envelope(), idempotency_key='gateway:event-1')
    duplicate = submitter.submit(target, _envelope(), idempotency_key='gateway:event-1')
    assert first == duplicate == 'run-1'
    assert store.get(_scope(), 'run-1').metadata['source_event_id'] == 'event-1'


@pytest.mark.asyncio
async def test_async_gateway_submitter_is_native_and_idempotent() -> None:
    scope = _scope('run-async')
    target = GatewayRunTarget(scope=scope, definition_id='support', definition_version='1.0.0', definition_digest='sha256:definition')
    store = AsyncInMemoryRunStore()
    submitter = AsyncDurableGatewayRunSubmitter(store)
    first = await submitter.submit(target, _envelope(), idempotency_key='gateway:event-1')
    duplicate = await submitter.submit(target, _envelope(), idempotency_key='gateway:event-1')
    assert first == duplicate == 'run-async'
    assert (await store.get(scope, first)).metadata['source_event_id'] == 'event-1'
