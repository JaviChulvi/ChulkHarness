# MCP integrations

MCP servers are external services. Configuring a server is not enough to expose
it: the agent must opt into `Capabilities(external_services=True)`, and every
bridged call still passes the selected permission and approval policy.

Hosted OpenAI MCP definitions are sent through the provider's native transport.
Other providers use Chulk-managed bridge tools. Both paths normalize results
into the same agent loop, but remote server behavior, authentication, latency,
availability, and data handling remain outside Chulk's trust boundary.

Keep authorization values in environment variables referenced by
`authorization_env`, never in `.chulk/mcp.json`. Restrict `allowed_tools`, use
approval for side effects, validate returned content, set network timeouts, and
assume prompts and tool arguments may leave the local machine.

MCP transport internals and raw traces are not stable public APIs. See
[permissions](permissions.md), [safety](safety.md), and [tracing](tracing.md).
