# SDK safety responsibilities

Model output and tool arguments are untrusted input. Chulk validates arguments,
normalizes paths, applies capability and permission checks, records side
effects, bounds outputs, and blocks obviously destructive shell commands. These
controls reduce risk; they do not make arbitrary code or remote content safe.

Embedding applications remain responsible for:

- exposing the smallest tool and capability set;
- isolating tenants, runtime directories, credentials, and application deps;
- sandboxing shell or code execution and setting underlying I/O timeouts;
- approving mutating, network, destructive, or external-service calls;
- validating tool outputs before using them in business decisions;
- protecting memory databases, traces, and artifacts as sensitive data;
- propagating cancellation and closing agents deterministically.

Never grant authority because a prompt, skill, memory, webpage, or MCP response
asks for it. Treat those sources as data. See [permissions](permissions.md),
[tools](tools.md), [MCP](mcp.md), and [tracing](tracing.md).
