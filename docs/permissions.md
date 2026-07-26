# Capabilities, permissions, and approval

Chulk uses three separate safety decisions. They answer different questions and
none bypasses the next layer.

| Layer | Question | Result |
|---|---|---|
| Capabilities | Should this kind of tool exist for this agent? | Disabled tools are never registered or shown to the model. |
| Permission policy | May this registered tool run under the selected profile? | The call is allowed, denied, or sent to approval. |
| Approval callback | Does the host approve this specific call? | The call proceeds or returns a recoverable denial. |

```python
from chulk import Agent, AgentConfig, Capabilities

agent = Agent(
    config=AgentConfig(permission_profile="read-only"),
    capabilities=Capabilities(
        files="read",
        shell=False,
        memory="read-only",
        network=False,
        external_services=False,
    ),
    llm=client,
)
```

The safe SDK default is `Capabilities.read_only()`: read-oriented file and
memory tools plus utility tools, with writes, shell, network, and external
services disabled. `Capabilities.none()` creates a valid tool-free agent.
`Capabilities.coding()` and `Capabilities.full()` are explicit broader presets.

File modes are `off`, `read`, and `write`. Memory modes are documented in
[memory.md](memory.md). Hosted or bridged MCP servers require
`external_services=True`; configuring a server alone does not expose it.

Explicit caller-supplied custom tools remain authoritative for backward
compatibility. They still pass through their declared permission level and
confirmation requirement. Retries repeat the permission and approval decision
for every attempt, and the public `ToolAttempt` records the outcome.

Approval callbacks receive a redacted request with the tool name, declared
capability category, arguments, and policy context. Treat approval as a host UI
decision, not as model consent. MCP is an external-service boundary even for
read-looking methods; see [MCP](mcp.md) and [safety](safety.md).

Hosted mode adds a prior application-authorization layer based on
`ExecutionScope.grants` and `ToolPolicy.required_grants`. Permission requests
include tool/schema versions, policy versions, schema digests, and an exact
argument digest. Credential resolution happens only after host authorization
and Chulk permission/approval succeed. See [hosted runtime](hosting.md).

For restart-safe execution, `DurableHostedExecutor` and
`AsyncDurableHostedExecutor` route an `ASK` decision through
`DurableApprovalService` or `AsyncDurableApprovalService` instead of waiting
in-process. The effect-linked approval and checkpoint are committed before the
worker lease and optional budget reservation are released. A different process
may decide it, but resume rechecks scope, grants, credentials, tool and schema
versions, the arguments digest, and the policy version before consuming the
approval once. Credentials are resolved only after that durable approval has
been consumed. Denied, expired, cancelled, revoked-authority, and
unavailable-integration outcomes are explicit and do not execute the tool.

`ImmediateApprovalAdapter` is the local ergonomic adapter over this same
ledger; it does not introduce a second approval engine.
