# Tools in embedded applications

Declare tools with `@Tool`, expose only the smallest necessary list, and give
each tool an accurate permission level. Model arguments are untrusted and are
validated against the generated schema before the Python callable starts.

## Trusted host dependencies

`ToolContext[Deps]` carries application-owned state that the model cannot
supply. Chulk removes the injected parameter from the model-facing JSON schema
and fills it from `Agent(..., deps=...)` or a per-run `deps=` override.

```python
from dataclasses import dataclass
from chulk import Agent, Tool, ToolContext

@dataclass(frozen=True)
class Deps:
    tenant_id: str

@Tool
def lookup_invoice(invoice_id: str, context: ToolContext[Deps]) -> str:
    return database.lookup(context.deps.tenant_id, invoice_id)

agent = Agent(llm=client, tools=[lookup_invoice], deps=Deps("tenant-1"))
agent.run("Find invoice 42", deps=Deps("tenant-2"))
```

If required dependencies are missing, the attempt fails before the decorated
function begins. Dependency values are not written into turn snapshots or
model-visible arguments. Use `ToolContext.metadata` for non-secret request
metadata; Chulk adds the current conversation and turn ids.

## Built-in file-read policy

The built-in `read_file`, `list_files`, and `search_files` tools deny sensitive
paths by default. The deny list covers local `.env` variants, credential and
private-key files, Git internals, trace directories, SQLite state, and `.chulk`
runtime state. Committable `.env.example`/`.env.sample`/`.env.template` files,
normal project source, and `.chulk/skills` playbooks remain readable.

An embedding host can deliberately construct a file tool with sensitive access
for a trusted workflow:

```python
from chulk.tools import FileReadPolicy, read_file_tool

read_sensitive_file = read_file_tool(
    project_root,
    read_policy=FileReadPolicy(allow_sensitive_paths=True),
)
```

The policy is captured when the host creates the tool and is absent from its
model-facing argument schema. Opting in never relaxes the project-root boundary.
This policy does not sandbox shell commands or custom tools; applications must
disable or separately constrain those capabilities when file confidentiality is
required.

## Structured output

Declare an object output schema with `output_schema=` or `ToolOutputPolicy`.
The successful Python value is normalized and validated before it is accepted.
Invalid output becomes an `invalid_output` tool failure with field-level issues.

```python
@Tool(
    output_schema={
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": False,
    }
)
def count_items() -> dict:
    return {"count": 3}
```

## Timeouts and retries

`timeout_seconds` applies to each attempt. Async timeouts cancel the awaiting
task. Python cannot force-kill a running synchronous thread, so synchronous
tools must still use underlying I/O timeouts and cooperative cancellation for
strong side-effect guarantees.

Retries are disabled by default. Opt in with `ToolRetryPolicy`, declare the tool
`idempotent=True`, and list the failure kinds that may retry.

```python
from chulk import ToolRetryPolicy

@Tool(
    timeout_seconds=2,
    retry_policy=ToolRetryPolicy(
        max_attempts=3,
        retryable_failure_kinds=("timeout", "environment_failure"),
    ),
    idempotent=True,
)
def fetch_status() -> dict:
    ...
```

Each retry repeats argument validation, capability availability, permission,
approval, execution, timeout, output validation, and tracing. Cancellation,
permission denial, unknown tools, invalid model arguments, and async/sync misuse
are never retried. Non-idempotent tools are held to one attempt when the policy
requires idempotency. `RunResult.tool_calls[*].attempts` exposes immutable
`ToolAttempt` records with timing, permission outcome, failure, and retry
disposition.

Application dependencies injected through `ToolContext` are host-owned, but
their methods can still produce side effects. Keep secrets out of `metadata`,
enforce tenant scope inside the dependency, and return only the data the model
needs. See [permissions](permissions.md) and [safety](safety.md).
