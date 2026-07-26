# Hosted gateway

`GatewayRuntime` keeps provider adapters at the edge while the host owns
routing, durable inbox/outbox state, execution scope, published definitions,
and run submission. Existing adapters remain compatible with the SQLite
reference ledger.

## Host contracts

The public sync and async boundaries are:

- `GatewayStore` and `AsyncGatewayStore` for inbox ownership, execution claims,
  outbox delivery, dead letters, and reconciliation.
- `GatewayRouter` and `AsyncGatewayRouter` for authenticated application
  routing.
- `GatewayScopeResolver` and `AsyncGatewayScopeResolver` for selecting one
  immutable `ExecutionScope` and published agent-definition revision.
- `GatewayRunSubmitter` and `AsyncGatewayRunSubmitter` for idempotent durable
  run submission.

`SQLiteGatewayLedger` and `SQLiteGatewayRouter` are reference adapters, not
required storage choices. A third-party store can implement these protocols
without importing a concrete SQLite type.

`PublishedDefinitionGatewayResolver` rejects draft, revoked, or missing
revisions. `DurableGatewayRunSubmitter` records the definition digest, source
event, correlation ID, and a stable input digest in `RunSubmission`. Async
variants await definition and run stores directly.

## Ownership and acknowledgement

The runtime resolves a route and immutable run target, durably ingests the
envelope, and submits the idempotent run before acknowledging the provider.
Duplicate inbound deliveries reuse the same inbox record and durable run. A
host whose inbox and run ledger share a transaction can implement both
operations atomically; otherwise the stable idempotency key makes the
persist-then-submit boundary restart-safe.

Per-destination and thread FIFO ordering remains in the store. Execution and
delivery claims use opaque lease tokens, and stale workers cannot commit.
Provider authentication, transport credentials, and rate limiting remain
inside the adapter.

## Delivery outcomes

`GatewayLimits.max_delivery_attempts` bounds retryable delivery. Exhausted
delivery and poison input become visible `dead_letter` records. An adapter
returning `accepted` or `unknown` creates an ambiguous outcome: Chulk does not
resend it. If the adapter implements `DeliveryReconciler`,
`reconcile_available()` asks the provider for evidence and records the result.

Hosted deliveries emit schema-v3 `delivery.started`, `delivery.completed`,
`delivery.failed`, `delivery.unknown`, `delivery.reconciled`, and
`delivery.dead_lettered` events with run scope, source-event identity, attempt,
and checkpoint metadata. No provider credential or raw secret enters the
event.

## Contract gate

Third-party stores should run
`chulk.testing.assert_gateway_store_contract(...)` and its async counterpart.
The gate checks deduplication, stale-claim rejection, durable outbox ownership,
ambiguous-outcome suppression, and reconciliation. The offline
[resilience example](../examples/hosted_runtime/resilience_contract.py) runs
the reference gate without credentials or external infrastructure.
