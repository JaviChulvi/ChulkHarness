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

Secret-free `.chulk/mcp.json` is declarative project configuration and may be
committed for review. File configuration rejects literal credential-bearing
fields such as `authorization`, `token`, `api_key`, and request headers; only
the name in `authorization_env` belongs in JSON. The referenced value stays in
the process environment or an ignored `.env`.

All other `.chulk/` data is runtime state except reviewable project playbooks
under `.chulk/skills/`. `chulk init` writes narrow ignore/allow rules, and
`chulk doctor` fails if a credential file, database, sidecar, backup, trace, or
artifact is tracked.

MCP transport internals and raw traces are not stable public APIs. See
[permissions](permissions.md), [safety](safety.md), and [tracing](tracing.md).
