# SDK embedding and result contract

Use `Agent` for synchronous hosts and `AsyncAgent` inside an async event loop.
Both own their runtime resources; close them explicitly or use `with`/`async
with`. A closed facade rejects further work. Construction, provider, safety,
tool, memory, and trace failures map to the stable [SDK exception
family](sdk-errors.md).

For multi-tenant or filesystem-free application hosting, use
`HostedRuntime`/`AsyncHostedRuntime` with a complete service bundle and
`ExecutionScope`. See the [hosted runtime guide](hosting.md). A memory namespace
alone is not an authorization or tenant-isolation boundary.

When several logical users or workspaces share a `store_path`, set
`AgentConfig(memory_namespace="tenant:workspace-key")` (or pass
`memory_namespace` directly to `Agent`). Omitting it preserves the
single-project compatibility namespace `default`; it is not a safe
multi-tenant boundary. See [memory](memory.md) for the normalization and
isolation contract.

```python
from chulk import Agent, AgentConfig
from chulk.testing import ScriptedLLMClient

client = ScriptedLLMClient([{"type": "final_answer", "content": "Ready."}])
with Agent(config=AgentConfig(project_root="."), llm=client, tools=[], skills=[]) as agent:
    result = agent.run_result("Check readiness")
```

Chulk returns immutable public snapshots from `run_result`, `plan_result`,
`approve_result`, and `reject_result`. Common records are available from `chulk`;
the complete contract is available from `chulk.results`.

```python
from chulk import Agent, RunStatus
from chulk.results import Cost, RunResult, ToolCall, Usage

result: RunResult = agent.run_result("Summarize the project")
if result.status is RunStatus.COMPLETED:
    print(result.content)

for call in result.tool_calls:
    print(call.tool_name, call.success)
```

`RunStatus`, `PlanStatus`, and `PlanStepStatus` are finite string enums. Unknown
future values convert to their explicit `UNKNOWN` member instead of pretending
to be a current lifecycle state.

## Stable records

`RunResult` and `PlanResult` compose typed records rather than unstructured
containers:

- `Usage` and `Cost` preserve provider-neutral accounting.
- `ToolCall` and `Observation` describe tool activity and returned evidence.
- `ContextReport`, `ContextBudget`, and `ContextSection` describe prompt input.
- `Plan`, `PlanStep`, and `PlanStepEvidence` describe approval and execution.

Sequences are tuples and mappings are recursively read-only. Each result is a
detached snapshot: later runtime state changes cannot alter a result already
returned to the caller.

Unknown pricing is represented by `Cost(amount=None, pricing_known=False)`.
That is distinct from a known free operation whose amount is `Decimal("0")`.
When a provider reports prompt-cache writes separately, `Usage` exposes
`cache_write_input_tokens` and `Cost` exposes `cache_write_input_cost` instead
of folding that billed bucket into ordinary input.

## Serialization

Every public record has `to_dict()`. It returns fresh JSON-oriented lists,
dictionaries, strings, numbers, booleans, and null values. Decimal monetary
values and trace paths serialize as strings. The returned containers are mutable
for adapter convenience and never mutate the source snapshot.

```python
payload = result.to_dict()
payload["errors"].append("adapter annotation")  # result.errors is unchanged
```

`extension_metadata` is recursively read-only on the result and serialized as a
fresh plain dictionary. Stable fields remain separate from extension data so an
adapter can preserve unknown metadata without weakening the typed contract.

## Async and concurrency boundary

One facade serializes work-starting calls so one conversation cannot interleave
turn state or callbacks. Use separate agent instances for true parallel runs.
`AsyncAgent` uses native async model requests for every built-in provider,
including planning, action repair, context summaries, reflection, and fallback
chains. Sync-only custom clients and some compatibility operations use a
thread-backed path. Cancellation propagates as `asyncio.CancelledError`; hosts
should cancel and await tasks, then close the agent. A synchronous Python worker
thread cannot be force-killed, so custom clients and tools need cooperative
cancellation and I/O timeouts.

See [events](events.md) for generator cleanup and ordering,
[configuration](configuration.md) for runtime ownership, and the
[release policy](release-policy.md) before importing advanced modules.
