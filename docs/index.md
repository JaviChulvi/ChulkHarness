# ChulkHarness documentation

Choose the path that matches what you are doing:

- **Embedding the SDK:** start with the credential-free [quickstart](quickstart.md),
  then use the [SDK guide](sdk.md), [configuration](configuration.md), and
  [release policy](release-policy.md).
- **Operating the CLI:** use the [repository README](../README.md#cli) for
  commands, configuration, and local runtime paths.
- **Talking to a remote agent:** use the private, allowlisted
  [Telegram adapter](telegram.md).
- **Developing ChulkHarness:** use the [contributor setup](../README.md#development),
  [roadmap](../TODO.md), and [security policy](../SECURITY.md).

## Topic guides

- [Quickstart](quickstart.md): install to first deterministic result.
- [SDK](sdk.md): facades, lifecycle, results, exceptions, and async boundaries.
- [Hosted runtime](hosting.md): service injection, execution scope, and tool policy.
- [Portable authoring](authoring.md): immutable definitions, constrained compilation, publication, and revocation.
- [Configuration](configuration.md): precedence, paths, and host ownership.
- [Providers](providers.md): injected, scripted, hosted, and local clients.
- [Tools](tools.md): custom tools, application context, outputs, and retries.
- [Permissions](permissions.md): capabilities, policy, and approvals.
- [Skills](skills.md): bundled and project procedural instructions.
- [Memory](memory.md): modes, review, provenance, and sensitive data.
- [Events](events.md): stable event streams, ordering, and cancellation.
- [MCP](mcp.md): external service trust and approval boundaries.
- [Tracing](tracing.md): sensitive internal diagnostics and walkthroughs.
- [Safety](safety.md): responsibilities around untrusted model input.
- [SDK errors](sdk-errors.md): stable exception categories and diagnostics.
- [Release policy](release-policy.md): stability labels and compatibility.
- [Telegram](telegram.md): private phone access, credentials, and deployment.

The SDK, CLI, and repository internals share one runtime builder, but their
defaults differ. The SDK starts read-only, the CLI coding workflow may enable
workspace writes, and internal modules are not a supported embedding surface.
