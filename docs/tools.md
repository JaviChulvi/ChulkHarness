# Tools in embedded applications

Declare tools with `@Tool`, expose only the smallest necessary list, and give
each tool an accurate permission level. Model arguments are untrusted and are
validated against the generated schema before the Python callable starts.

## Built-in shell execution

`run_cmd` keeps the existing shell capability, permission-profile, and approval
checks. Once approved, it captures stdout and stderr through separate bounded
byte buffers while the child is running. `CHULK_MAX_TOOL_STDOUT_CHARS` and
`CHULK_MAX_TOOL_STDERR_CHARS` remain the configuration keys for compatibility;
their numeric values are also enforced as the raw byte ceilings for shell
capture. Crossing either ceiling stops the command, kills its process group,
and returns `output_limit_exceeded` with a bounded head/tail preview and
inspectable byte counts. A timeout follows the same process-group cleanup path
and retains any bounded partial output.

The built-in destructive-command checks are guardrails, not a sandbox. Direct
local execution is still the default for the CLI. Embedding hosts that require
containment can fail closed and inject a policy that wraps the model command in
their own sandbox:

```python
from chulk import Agent, ShellExecutionDecision, ShellExecutionRequest, Tools

class SandboxPolicy:
    def prepare(self, request: ShellExecutionRequest) -> ShellExecutionDecision:
        sandbox_argv = build_sandbox_command(
            command=request.command,
            project_root=request.cwd,
        )
        return ShellExecutionDecision.allow(
            sandbox_argv,
            policy_name="application-sandbox",
            shell=False,
            containment_applied=True,
        )

agent = Agent(
    tools=[Tools.run_cmd],
    shell_execution_policy=SandboxPolicy(),
    require_shell_containment=True,
)
```

`containment_applied=True` is a host assertion, not something Chulk can verify.
Set it only after the wrapper establishes the filesystem, network, process, and
credential boundaries your application requires. If containment is required
and the policy does not assert it, Chulk returns `containment_required` before
starting a child. A policy can also deny a request explicitly with
`ShellExecutionDecision.deny(...)`.

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

When `apply_patch` finds stale context, its failure identifies the file, hunk,
target line, and whether the mismatch was in the hunk location, context, or
removal. It also supplies a bounded nearby line range to reread before rebuilding
the hunk from current text. The diagnostic never includes the file contents.

## Trace artifact reader

`Tools.read_trace_artifact` is a separate, opt-in read capability for output
that Chulk truncated and stored beside the active trace. It accepts only an
opaque artifact id, never a path, and is bound by runtime assembly to the
agent's current conversation. Reads validate ownership, file type, size, and
hash, then return a bounded head, tail, head/tail, or byte slice.

The tool is intentionally absent from default tool registries. A host must add
it explicitly and its normal read permission policy still applies:

```python
agent = Agent(
    llm=client,
    tools=[Tools.read_trace_artifact],
)
```

This capability does not grant general trace access and does not weaken the
built-in file-read deny list.

## Structured output

Declare an object output schema with `output_schema=` or `ToolOutputPolicy`.
The model also sees the declared schema in the tool description, whether the
provider uses Chulk's JSON fallback catalog or native function declarations.
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

The action loop also stops before executing an unchanged call that already
ended in a non-retryable cancellation, permission denial, unknown tool,
invalid arguments, async/sync misuse, or fatal safety failure. A changed tool
or argument set remains available, while environment failures and timeouts may
still be retried or polled. Each model request receives a harness-derived late
status section with the tool calls used and remaining, the current unchanged
failure sequence, and the active plan step.

Application dependencies injected through `ToolContext` are host-owned, but
their methods can still produce side effects. Keep secrets out of `metadata`,
enforce tenant scope inside the dependency, and return only the data the model
needs. See [permissions](permissions.md) and [safety](safety.md).

## Hosted tool contracts

Hosted applications can add semantic `ToolIdentity` and `ToolPolicy` contracts
for required grants, risk, effects, approval, concurrency, idempotency, dry-run
and compensation support, and input/output classification. Chulk derives safe
version `1.0.0` identity and schema digests for simple existing tools. Explicit
identity/schema mismatches fail registration.

Host authorization runs before the ordinary permission profile. Credentials
are resolved only after both layers allow the exact call and are available only
from `ToolContext.credentials`. Secret-classified output is withheld before it
can reach an observation, event, trace, or result. See the
[hosted runtime guide](hosting.md) for the complete contract and hooks.

A tool may return `ToolResult.resources` and `ToolResult.application_events`.
Application events are accepted only when the tool registers a matching
`ApplicationEventSchema` through `application_event_schemas`. Chulk validates
the namespace, schema version, JSON payload, schema, size, redaction, and
idempotency key before adding the event to the public ordered stream. Invalid
publications turn the tool result into `invalid_output`; they are never sent to
the event sink.

For transports that return several independent calls, the async batch path
permits concurrency only when every tool declares both `ToolEffect.READ` and
`ToolConcurrency.PARALLEL_SAFE`. The presence of one write, unknown effect, or
serial policy makes the complete batch serial. Returned results always preserve
the model call order.
