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

Applications that store, review, and publish agent behavior independently from
deployment paths should use `AgentDefinition`, `AgentCompiler`, and
`AgentDefinitionRuntime`. See [portable authoring](authoring.md).

For a local application that keeps its prompt and declared tool names together,
`Agent.from_directory(...)` and `AsyncAgent.from_directory(...)` load
`agent.toml` plus `instructions.md`. Pass the declared tool objects explicitly;
the normal SDK capability and permission checks remain in force. This local
shortcut has no publication authority. Hosted applications should compile and
publish the same directory through `AgentDirectory`.

When several logical users or workspaces share a `store_path`, set
`AgentConfig(memory_namespace="tenant:workspace-key")` (or pass
`memory_namespace` directly to `Agent`). Omitting it preserves the
single-project compatibility namespace `default`; it is not a safe
multi-tenant boundary. See [memory](memory.md) for the normalization and
isolation contract.

For local SQLite memory, `AgentConfig` also accepts an opt-in
`MemoryRetentionPolicy`; see [memory](memory.md) for its archive-first limits
and hosted-service boundary.

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
- `HostResource` values in `RunResult.resources` identify retrieved evidence
  and generated outputs without requiring observation or trace parsing.
- `ContextReport`, `ContextBudget`, and `ContextSection` describe prompt input.
- `Plan`, `PlanStep`, and `PlanStepEvidence` describe approval and execution.
- `FinalAnswerDelivery` distinguishes a complete answer from safe truncation,
  policy blocking, or failure after partial delivery.

## Verified plan-step completion

Plan-step completion remains model-asserted by default for compatibility. Hosts
that need deterministic termination can pass `plan_step_verifier` or
`async_plan_step_verifier`. The callback receives a
`PlanStepVerificationRequest` containing the step description, acceptance
criteria, prior evidence, and the model's completion claim. It must return a
`PlanStepVerification` with a boolean decision and non-empty evidence.

```python
from chulk import PlanStepVerification

def verify_step(request):
    passed = run_acceptance_checks(request.acceptance_criteria)
    return PlanStepVerification(
        passed=passed,
        evidence="Focused acceptance checks passed." if passed else "Tests still fail.",
    )

agent = Agent(..., plan_step_verifier=verify_step)
```

Only a passing decision marks the step complete, and both the model assertion
and host verification are retained as step evidence. A rejection keeps the step
`in_progress` and returns the verifier evidence to the model as recovery
feedback. Async execution awaits the async callback; when only the sync callback
is configured, it runs outside the event loop. Calling the synchronous plan
path with only an async verifier fails explicitly instead of bypassing it.

## Incremental final answers

Validated-final-answer replay remains the default. Hosted applications can opt
into true incremental delivery explicitly:

```python
from chulk import Agent, FinalAnswerStreamingMode

agent = Agent(
    config=config,
    llm=client,
    final_answer_streaming=FinalAnswerStreamingMode.INCREMENTAL,
)
```

Incremental mode first completes the normal structured action request. Once a
validated final-answer intent is legal, Chulk starts a separate plain-text
provider request through `stream_final_answer` or `astream_final_answer`. Action
JSON, tool arguments, repair prompts, and invalid partial actions are never
projected as `model.delta`.

Pass `output_policy` for sync hosts or `async_output_policy` for async hosts.
Each policy receives `FinalAnswerChunk` and returns
`FinalAnswerPolicyDecision`, so it can transform, buffer, block, or safely stop
delivery. The decision runs after mandatory redaction and before callbacks,
event sinks, traces, session persistence, or result reconstruction. Policy
exceptions fail closed by default; `OutputPolicyFailureMode.OPEN` must be an
explicit host choice. Buffering policies implement `reset()` so Chulk can
discard uncommitted text safely before a pre-delta provider fallback.

Concatenating public deltas produces `RunResult.content`. The typed
`final_answer_delivery` field records `complete`, `safely_truncated`, `blocked`,
or `failed_after_partial`, the public delta count, provider completion, and a
bounded error. A fallback provider may take over only before streaming emits a
chunk; after output begins, failure terminalizes the partial answer and never
mixes providers.

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
Synchronous hosts can call `agent.cancel()` from another thread to request
cooperative cancellation of the active turn; it returns `False` when no turn is
running. Cancellation closes the active model transport and is observed between
model, tool, and reflection operations. Custom blocking tools must still provide
their own bounded I/O or react to resource closure.
`AsyncAgent` uses native async model requests for every built-in provider,
including planning, action repair, context summaries, reflection, and fallback
chains. Sync-only custom clients and some compatibility operations use a
thread-backed path. Cancellation propagates as `asyncio.CancelledError`; hosts
should cancel and await tasks, then close the agent. A synchronous Python worker
thread cannot be force-killed, so custom clients and tools need cooperative
cancellation and I/O timeouts.
Incremental final answers use the provider's async iterator directly. OpenAI
Responses and OpenAI-compatible Chat Completions clients implement native async
streaming; providers without it retain a one-shot async compatibility stream.
The shared LLM timeout is reapplied while waiting for every async stream chunk.
A stalled iterator is closed, active accounting is settled, and partial public
content terminalizes as `failed_after_partial`; internal delivery evidence and
the stream failure trace identify `stream_idle_timeout` separately from provider
errors.

Native async hosted factories are resolved with
`await AsyncHostedRuntime.create(...)`. Runtime-owned async resources are
closed with their `aclose()` method on the active event loop. Native async
construction exposes a closed, typed keyword-only option set; unknown or
misspelled options raise `TypeError` before service or runtime assembly.
Service methods and policy hooks are awaited directly; explicit synchronous
bindings are isolated in worker threads. Ordered persistence and sink journals
flush before model, tool, approval, and terminal boundaries. Cancellation and
timeouts release active usage reservations and flush terminal evidence; close
the runtime with `async with` or `await runtime.close()` to finalize all
runtime-owned resources. The direct constructor remains the compatibility path
for entirely synchronous service bindings.

Application-owned service and gateway implementations can run the published
offline gates in `chulk.testing`:

- `assert_hosted_services_contract` and
  `assert_async_hosted_services_contract`;
- `assert_durable_execution_contract` and
  `assert_async_durable_execution_contract`; and
- `assert_parent_child_run_contract` and
  `assert_async_parent_child_run_contract`;
- `assert_gateway_store_contract` and
  `assert_async_gateway_store_contract`.

They check the complete hosted service surface, scope isolation, duplicate
triggers and logical effects, approval restart behavior, unknown-effect
reconciliation, deterministic event order, bounded parent/child fan-out and
progress, durable outbox ownership, and ambiguous delivery reconciliation.

See [events](events.md) for generator cleanup and ordering,
[configuration](configuration.md) for runtime ownership, and the
[release policy](release-policy.md) before importing advanced modules.
