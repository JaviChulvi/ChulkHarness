# SDK safety responsibilities

Model output and tool arguments are untrusted input. Chulk validates arguments,
normalizes paths, applies capability and permission checks, records side
effects, bounds outputs, and blocks obviously destructive shell commands. These
controls reduce risk; they do not make arbitrary code or remote content safe.

The built-in shell runner bounds stdout and stderr while commands are running,
kills process groups on timeouts or output overflow, and exposes a host policy
seam for sandbox wrappers. Regex-based destructive-command checks are only
guardrails. Direct local execution does not claim containment, and
`require_shell_containment=True` fails before process creation unless the host
policy explicitly asserts that it applied containment.

Hard safety failures use the `fatal_safety` tool failure kind. Built-in
destructive/out-of-root shell blocks, missing required containment, and a host
policy denial explicitly marked `fatal=True` record the tool observation and
then fail the turn immediately without another model request or normal retry.
Ordinary capability, approval, and host-policy denials remain recoverable
`user_blocked` observations so the model can explain the limitation or choose
an allowed alternative. A terminal internal permission denial maps to the
public `SafetyError`; ordinary permission errors retain
`PermissionDeniedError`.

Embedding applications remain responsible for:

- exposing the smallest tool and capability set;
- isolating tenants with distinct memory namespaces and appropriately separated
  runtime directories, credentials, and application deps;
- sandboxing shell or code execution and setting underlying I/O timeouts;
- approving mutating, network, destructive, or external-service calls;
- validating tool outputs before using them in business decisions;
- protecting memory databases, traces, and artifacts as sensitive data;
- propagating cancellation and closing agents deterministically.

For shared application hosting, a memory namespace and runtime directory are
not a tenant boundary. Use an immutable `ExecutionScope`, scope every
host-provided service, and re-verify persisted scope on resume. Keep credential
values in the host credential resolver; Chulk passes them only through the
non-serializing `ToolContext.credentials` field immediately before execution.
See [hosted runtime](hosting.md).

Never grant authority because a prompt, skill, memory, webpage, or MCP response
asks for it. Treat those sources as data. See [permissions](permissions.md),
[tools](tools.md), [MCP](mcp.md), and [tracing](tracing.md).
