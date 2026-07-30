# Hosted runtime embedding

Use `HostedRuntime` or `AsyncHostedRuntime` when an application—not Chulk—owns
persistence, tenancy, authorization, credentials, audit, and resource
lifecycle. Hosted mode is explicit and fail-closed: it never fills a missing
service with SQLite or a local directory.

The credential-free [reference application](../examples/hosted_runtime/hosted_app.py)
runs sync and async agents with tenant-aware in-memory services, a versioned
read-only tool, and host-owned event collection. It asserts that the configured
project root stays empty. The companion
[resilience contract](../examples/hosted_runtime/resilience_contract.py)
checks isolation, duplicate runs and effects, cross-process approval,
reconciliation, event ordering, and gateway delivery.

## Execution scope

Every hosted construction requires an immutable `ExecutionScope` supplied by
the application:

```python
from chulk import ExecutionScope

scope = ExecutionScope(
    tenant_id="tenant-42",
    workspace_id="support",
    actor_id="user-7",
    agent_id="support-assistant",
    agent_version="2.3.0",
    run_id="run-018",
    grants=frozenset({"tickets:read"}),
)
```

Tenant, workspace, agent, agent version, and run identifiers are mandatory. The
runtime binds a generated conversation id when the host omits one. The complete
canonical encoding has a SHA-256 `scope.key`; hosts can use that opaque key for
collision-safe partitioning. Service factories receive the bound scope before
model or tool execution.

`scope.child(...)` preserves tenant, workspace, and actor authority, records
the parent run, and permits only a subset of the parent's grants. Resume and
cross-process code should compare persisted scopes with
`scope.assert_same_authority(...)` before loading state.

Local `Agent` and CLI construction remains unchanged. Local mode continues to
use its documented SQLite and filesystem defaults.

## Complete service boundary

`RuntimeServices` and `AsyncRuntimeServices` require bindings for every runtime
resource:

| Binding | Responsibility |
|---|---|
| `memory` | scoped long-term memory reads and writes |
| `sessions` | conversation/turn persistence and scoped session search |
| `skills` | skill selection plus optional lifecycle services |
| `traces` | internal trace events |
| `artifacts` | opaque, bounded trace artifacts |
| `usage` | reservations, usage, cost, and budget accounting |
| `audit` | host-visible sensitive access and side-effect audit |
| `execution` | turn-scoped execution sessions |
| `plugins` | reviewed plugin startup verification |
| `content` and `media` | typed input content and media processing |
| `tool_policy` | authorization, credential, effect, and redaction hooks |
| `runs` | durable run, step, attempt, checkpoint, lease, and effect state |
| `approvals` | restart-safe approval requests and single-use decisions |
| `events` | application-owned schema-v3 public event delivery |

Wrap each resource in `ServiceBinding.host(...)`,
`ServiceBinding.runtime(...)`, or `ServiceBinding.scoped(...)`. Host-owned
resources are never closed by Chulk. Runtime-owned resources are finalized once
in reverse resolution order. Scoped factories receive the immutable execution
scope and default to runtime ownership; pass
`ownership=ResourceOwnership.HOST` for shared application resources.

`SessionRuntimeServices` groups the session store and its read/search facade.
`SkillRuntimeServices` groups the registry with optional lifecycle, proposal,
and review services. An explicit bundle is complete: invalid bindings and
factories that return no resource fail during construction, before model or tool
work.

Async hosts may use `AsyncServiceBinding.host(...)`,
`AsyncServiceBinding.runtime(...)`, or `AsyncServiceBinding.scoped(...)`.
Await native async factories with `await AsyncHostedRuntime.create(...)`.
Factories and runtime-owned `aclose()` methods run on the caller event loop;
synchronous compatibility bindings are isolated from it. Direct
`AsyncHostedRuntime(...)` construction remains available when every binding is
synchronous.

The protocols exported from `chulk.hosting` do not expose SQLite paths or
filesystem implementation types. The in-memory objects under
`chulk.hosting.reference` are a reference implementation and contract-test
fixture, not durable production storage.

## Durable hosted execution

`DurableHostedExecutor` and `AsyncDurableHostedExecutor` run an SDK turn
through the shared run owner. The host submits an immutable definition/input
digest and named steps, then Chulk claims a lease before invoking the model:

```python
from chulk import (
    DurableHostedExecutor,
    RunSubmission,
    StepDefinition,
)

submission = RunSubmission(
    idempotency_key="webhook:evt-42",
    input_digest="sha256:...",
    definition_digest="sha256:published-agent-v7",
    steps=(StepDefinition(id="agent", name="Run agent turn"),),
)
outcome = DurableHostedExecutor(
    runtime,
    runtime.runtime.run_store,
).execute(
    "Update ticket 42.",
    submission,
    worker_id="worker-a",
    step_id="agent",
)
```

Duplicate idempotency keys return the existing run. Claims use expiring leases
and every checkpoint uses compare-and-set ownership, so a stale worker cannot
commit. Lease reconciliation requeues work only when no committed checkpoint
or possible effect exists. Otherwise the run becomes `unknown`.

Before a tool transport begins, durable execution stores the logical effect
key, tool and input-schema versions, and arguments digest. Mutating tools must
receive a stable effect key from `ToolPolicyHooks.derive_effect_key`. A
transport failure after dispatch becomes `unknown`; replay is blocked until
`reconcile_effect(...)` records an operator decision. Cancellation during an
uncertain effect records the request but cannot claim the effect did not
happen.

The sync and async `RunStore` and `ApprovalStore` protocols are host-facing
contracts. `SQLiteRunStore`, `AsyncSQLiteRunStore`, `SQLiteApprovalStore`, and
`AsyncSQLiteApprovalStore` are reference adapters built on the shared
forward-only migration and transaction policy. In-memory adapters are
filesystem-free test and local-host fixtures.

For a production relational reference, install `chulkharness[postgres]` and
follow the [PostgreSQL operations guide](postgres.md). The optional adapters
reuse these same run, approval, gateway, and schedule state owners and add
forward-only Alembic migrations plus atomic inbox/run and run/outbox ownership
transfers.

## Durable parent and child runs

Parent/child execution is an explicit host operation. A child-shaped
`ExecutionScope` or model output never creates lineage by itself.
`submit_parent(...)` atomically submits the parent in
`waiting_for_children` and freezes a `ParentRunPolicy` containing the required
child count, fan-out ceiling, and aggregate `RunBudget`. The parent stays out
of the ordinary worker claim queue.

```python
from chulk import (
    BudgetScope,
    ParentRunPolicy,
    RunBudget,
)

policy = ParentRunPolicy(
    required_children=2,
    max_children=4,
    budget=RunBudget(
        scope=BudgetScope.CHILD_TASK,
        max_model_calls=8,
        max_tool_calls=16,
        max_tokens=20_000,
    ),
)
parent = runs.submit_parent(
    parent_scope,
    parent_submission,
    policy=policy,
)

child_scope = parent_scope.child(
    run_id="child-1",
    agent_id="reviewer",
    agent_version="published-7",
    grants=frozenset({"repository:read"}),
)
child = runs.submit_child(
    parent_scope,
    child_scope,
    child_submission,
    definition_revision="published-7",
)
```

Child creation fails closed unless tenant, workspace, and actor match the
parent, grants are a subset, the published definition revision is explicit,
matches `child_scope.agent_version`, and the cumulative child allocations fit
the parent budget. Parent-row serialization makes the fan-out and budget checks
safe across multiple workers. A child can inspect only its exact run; siblings
remain invisible.

Workers use the ordinary child run lease for steps, approvals, effects,
retry/resume, and reconciliation. `record_child_progress(...)` additionally
requires that live lease and commits strictly increasing progress sequences to
both the child and parent event streams. Stale workers cannot append progress.

`aggregate_children(...)` refuses nonterminal, approval-waiting, or unknown
children. After every required outcome is completed, failed, cancelled, or
reconciled, it atomically:

1. snapshots terminal events, results, errors, effect reconciliation, and the
   last progress sequence as child evidence;
2. terminalizes the parent once; and
3. creates one `ParentCompletion` outbox record.

Parent cancellation is explicit through `request_parent_cancellation(...)`;
it marks running children for cooperative cancellation and immediately
cancels children that have not started. Aggregation emits a cancelled parent
only after all children settle. A failed or cancelled child otherwise makes
the parent aggregate fail rather than claim unsupported success.

Claim parent completions with `claim_parent_completion(...)`. A definite
pre-delivery failure can be released with `fail_parent_completion(...)`.
Ambiguous delivery must be quarantined with
`mark_parent_completion_unknown(...)` and resolved through
`reconcile_parent_completion(...)`; expired claims also become `unknown`
instead of being blindly replayed. See the credential-free
[`parent_child_app.py`](../examples/hosted_runtime/parent_child_app.py).

## Durable approvals

When the permission policy returns `ASK`, the durable executors first commit
the logical effect intent, then `DurableApprovalService` (or
`AsyncDurableApprovalService`) creates an immutable effect-linked request and
checkpoint, clears the worker lease, moves the run to
`waiting_for_approval`, and invokes the optional budget-release hook. The SDK
turn is left waiting rather than failed. Another process can record a decision
and resume the run later.

Resume revalidates the exact execution scope, authority, credentials, tool and
schema versions, arguments digest, and policy version. Changed facts invalidate
the request. Approvals are consumed with a revision compare-and-set exactly
once; a restart between consumption and run release safely finishes the
resume. Denial, expiry, cancellation, revoked authority, and unavailable
credentials return typed outcomes.

`ImmediateApprovalAdapter` is available to local integrations that resolve the
decision immediately while recording the same durable request, decision, and
consumption trail. `DurablePermissionBroker` adapts the existing control-server
inbox to the same run-scoped ledger. A hosted run that bypasses
`DurableHostedExecutor` is denied rather than falling back to an in-process
wait.

## Events, audit, traces, and artifacts

These are separate contracts:

- `EventSink` receives redacted schema-v3 application events.
- `AuditSink` receives durable security and effect metadata, never raw prompts,
  arguments, or credentials.
- `TraceSink` receives sensitive diagnostic detail under host retention.
- `ArtifactStore` owns large opaque content independently of trace retention.

`RunEventPublisher` projects append-only run transitions into typed public
payloads with stable event IDs and deterministic causation. A host can rebuild
state from the durable run ledger even after detailed traces expire.
`CallbackEventSink` supports application-owned streams and defaults to
fail-closed delivery; set `fail_closed=False` only when dropping a public event
is an explicit host policy.

## Versioned tools and host hooks

Every registered tool receives a `ToolIdentity` and `ToolPolicy`. Simple tools
get version `1.0.0`, schema digests, serial execution, and a conservative policy
derived from their existing permission level. Applications can declare a
stronger contract:

```python
from chulk import Tool, ToolEffect, ToolPolicy

@Tool(
    policy=ToolPolicy(
        version="2.0.0",
        required_grants=frozenset({"catalog:read"}),
        effect=ToolEffect.READ,
    )
)
def catalog_lookup(item_id: str) -> str:
    return application_catalog[item_id]
```

Identity includes the tool, input schema, and output schema versions and
digests plus a stable implementation digest for Python callables. Registration
rejects a mismatched name or digest. Permission requests, tool results, and
trace records carry the identity and policy versions and digests, so changed
arguments, code contracts, or schemas cannot reuse an old approval silently.

`ToolPolicyHooks` supports authorization, credential resolution, preview,
effect-key derivation, reconciliation, compensation, and host redaction.
Required grants and the optional authorizer run before credential resolution.
Credentials are injected only into `ToolContext.credentials` immediately before
execution. They are omitted from context serialization, prompts, definitions,
events, traces, results, and errors.

Mutating and unknown-effect tools default to serial execution. A tool may
declare `parallel_safe` only with a read effect.
The async tool-batch path runs concurrently only when every selected tool is
both read-only and parallel-safe; any mutating or serial tool makes the whole
batch deterministic and serial. Results retain input order.
Retry eligibility follows an explicit idempotency strategy; the legacy
`idempotent=True` flag remains compatible.

Async tool authorization and credential hooks are awaited directly by
`AsyncHostedRuntime`. Sync construction rejects awaitable hooks rather than
opening a nested event loop.

See [SDK errors](sdk-errors.md) for the stable public failure categories,
[permissions](permissions.md) for the existing coarse local policy, and
[events](events.md) for the public event contract. Third-party stores can run
the reusable contract functions in `chulk.testing`; see
[hosted gateway](gateway.md#contract-gate).
