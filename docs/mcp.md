# MCP integrations

MCP servers are external services. Configuring a server is not enough to expose
it: the agent must opt into `Capabilities(external_services=True)`, and every
bridged call still passes the selected permission and approval policy.

Hosted OpenAI MCP definitions are sent through the provider's native transport.
Other providers use Chulk-managed bridge tools. Both paths normalize results
into the same agent loop, but remote server behavior, authentication, latency,
availability, and data handling remain outside Chulk's trust boundary.

Project MCP configuration must use a non-empty `allowed_tools` list. Chulk
defers bridge discovery and requires approval before the first project-declared
remote call. Remote descriptions, schemas, errors, and results remain untrusted
content; a mutating follow-on action requires a fresh owner approval.

Secret-free `.chulk/mcp.json` is declarative project configuration and may be
committed for review. Project configuration cannot select `authorization_env`
or set `approval` to `never`; those authority-bearing options are available
only through programmatic `MCPServerConfig` construction by the embedding host.
File configuration also rejects literal credential-bearing fields such as
`authorization`, `token`, `api_key`, and request headers.

All other `.chulk/` data is runtime state except reviewable project playbooks
under `.chulk/skills/`. `chulk init` writes narrow ignore/allow rules, and
`chulk doctor` fails if a credential file, database, sidecar, backup, trace, or
artifact is tracked.

MCP transport internals and raw traces are not stable public APIs. See
[permissions](permissions.md), [safety](safety.md), and [tracing](tracing.md).
