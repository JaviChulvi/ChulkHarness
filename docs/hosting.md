# Hosted runtime embedding

Use `HostedRuntime` or `AsyncHostedRuntime` when an application—not Chulk—owns
persistence, tenancy, authorization, credentials, audit, and resource
lifecycle. Hosted mode is explicit and fail-closed: it never fills a missing
service with SQLite or a local directory.

The credential-free [reference application](../examples/hosted_runtime/app.py)
runs sync and async agents with tenant-aware in-memory services, a versioned
read-only tool, and host-owned event collection. It asserts that the configured
project root stays empty.

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

The protocols exported from `chulk.hosting` do not expose SQLite paths or
filesystem implementation types. The in-memory objects under
`chulk.hosting.reference` are a reference implementation and contract-test
fixture, not durable production storage.

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
declare `parallel_safe` only with a read effect. Retry eligibility follows an
explicit idempotency strategy; the legacy `idempotent=True` flag remains
compatible.

Async tool authorization and credential hooks are awaited directly by
`AsyncHostedRuntime`. Sync construction rejects awaitable hooks rather than
opening a nested event loop.

See [SDK errors](sdk-errors.md) for the stable public failure categories,
[permissions](permissions.md) for the existing coarse local policy, and
[events](events.md) for the public event contract.
