# SDK events

Chulk projects selected runtime activity into one versioned public envelope.
Internal JSONL trace names and payloads are diagnostic implementation details;
they are not automatically exposed as SDK events.

`chulk.server.ai_sdk_ui_chunks(...)` is an optional adapter for Vercel AI SDK
clients. It projects safe final-answer text into UI message chunks and retains
tool/approval records as `data-chulk-*` chunks. Chulk's event stream remains
the stable source of truth for durable state and reconnection.

```python
from chulk import Agent, EventName, RunCompletedPayload

with Agent(config=config, llm=client) as agent:
    for event in agent.run_events("Summarize this project"):
        print(event.name, event.turn_id)
        if event.name == EventName.RUN_COMPLETED.value:
            assert isinstance(event.payload, RunCompletedPayload)
            print(event.payload.result.content)
```

`AgentEvent` schema v3 contains `event_id`, `name`, an ISO-8601 `timestamp`,
`conversation_id`, optional turn/run/step IDs, an `ExecutionScope`,
correlation, causation, source-event and idempotency IDs, a typed `payload`,
and read-only `extensions`. Readers continue to accept schema v1 and v2;
legacy envelopes receive a deterministic `legacy_...` event ID. Dotted
lowercase names are the compatibility-stable catalog:

| Event | Payload | Meaning |
|---|---|---|
| `run.started` | `RunStartedPayload` | A turn acquired the agent run gate. |
| `model.request.started` | `ModelRequestPayload` | A provider request is starting. |
| `model.delta` | `ModelDeltaPayload` | A final-answer text fragment is available. |
| `model.response.completed` | `ModelResponsePayload` | A provider response completed. |
| `tool.call.started` | `ToolCallPayload` | A selected tool is starting. |
| `tool.call.completed` | `ToolCallPayload` | Tool execution succeeded. |
| `tool.call.failed` | `ToolCallPayload` | Tool execution returned a recoverable failure. |
| `permission.requested` | `PermissionPayload` | A tool permission decision is required. |
| `permission.resolved` | `PermissionPayload` | The permission decision was resolved. |
| `memory.loaded` | `ResourcesLoadedPayload` | Durable memories were selected. |
| `skill.loaded` | `ResourcesLoadedPayload` | Skill playbooks were selected. |
| `resource.available` | `ResourceAvailablePayload` | A redacted host resource reference is available before dependent output. |
| `application.event` | `ApplicationEventPayload` | A tool-produced, schema-validated application event is available. |
| `plan.created` | `PlanPayload` | Plan mode created an approval-gated plan. |
| `plan.approved` | `PlanPayload` | A pending plan was approved. |
| `run.completed` | `RunCompletedPayload` | The terminal structured result is available. |
| `run.failed` | `RunFailedPayload` | The run failed before a normal result. |
| durable `run.*` transitions | `RunLifecyclePayload` | Queue, pause, resume, retry, cancellation, unknown, and dead-letter state. |
| durable `step.*` transitions | `StepLifecyclePayload` | Step attempts and committed checkpoints. |
| durable `effect.*` transitions | `EffectLifecyclePayload` or `ReconciliationPayload` | Effect intent, dispatch, outcome, retry, and operator reconciliation. |
| durable `approval.*` transitions | `ApprovalLifecyclePayload` | Request, decision, consumption, invalidation, and expiry. |
| durable `child.*` and `parent.completion.*` transitions | `SerializedEventPayload` | Explicit child submission/progress/cancellation and parent completion outbox state. |
| `delivery.started`, `delivery.completed`, `delivery.failed` | `DeliveryPayload` | An outbound delivery was claimed and reached a known outcome. |
| `delivery.unknown`, `delivery.reconciled`, `delivery.dead_lettered` | `DeliveryPayload` | An ambiguous outcome awaits evidence, was reconciled, or became terminal. |

Unknown future trace events are excluded until Chulk explicitly adds a public
projection. Permission payloads deliberately omit raw tool arguments.

Hosted runtime events include the redacted execution scope and its canonical
key in the envelope. Tool completion and permission payload extensions
include tool/schema identity, policy versions, and digests, but never resolved
credential values.

Context resources are published before the first dependent model request. Tool
resources and application events follow `tool.call.completed` and precede the
terminal event. Their event IDs and idempotency keys are deterministic, while
normal causation chaining preserves their exact position in the run stream.
Only the bounded, redacted `HostResource` projection is public; private context
content remains available to the model without appearing in the event.

## Iteration and callbacks

Use `agent.run_events(...)` for a synchronous iterator and
`async_agent.run_events_async(...)` for an async iterator. Events retain
execution order and end with exactly one `run.completed` or `run.failed` event.
The completion payload contains the same `RunResult` contract returned by
`run_result(...)`. See [SDK result contract](sdk.md) for its finite statuses,
immutable nested records, and serialization rules. Model response events also
reuse the public `Usage` and `Cost` snapshots, while plan events carry `Plan`.

Constructor and per-run `on_event` callbacks receive the same public envelope.
`on_delta` remains supported and is driven by `model.delta` events.

In incremental mode, every `model.delta` is already redacted and accepted by
the configured host output policy. Its sequence is deterministic and its
causation link points at the preceding public event. The corresponding terminal
result reconstructs exactly the concatenated permitted deltas and records the
delivery outcome in `RunResult.final_answer_delivery`. Structured action bytes,
tool-call arguments, and repair responses never enter this event.

Durable transitions are first committed to `RunStore` as append-only
`RunEvent` records. `RunEventPublisher` projects them to schema-v3 envelopes
using the durable event ID and sequence, chaining each event to its predecessor.
Application sinks can therefore deduplicate and reconstruct deterministically.
`AsyncRunEventPublisher` awaits async sinks directly. Hosted gateway events use
the same scope, source-event, correlation, and idempotency fields as their
durable run.

Configured parent runs add `run.waiting_for_children` and
`run.children_aggregated` lifecycle events. Child progress is recorded on both
the child and parent streams with the child run ID and monotonic sequence.
Terminal parent delivery events remain separate from `run.completed` so an
application can distinguish durable execution from external notification.

```python
async for event in agent.run_events_async("Explain the change"):
    render(event)
```

Cancelling an incremental async iteration cancels the native provider iterator,
releases active accounting reservations, records partial-delivery evidence, and
terminalizes the turn as cancelled. Validated compatibility runs retain their
historical in-flight unwind behavior. Closing a synchronous iterator
also waits for the operation to finish; Python threads are not force-killed.

Consumers should always close partially consumed synchronous generators and
`aclose()` partially consumed async generators. Cancellation is propagated, not
translated into an SDK error. Cleanup removes event-channel ownership; it does
not promise to terminate an already running synchronous tool thread.

## Concurrency policy

One facade serializes work-starting operations in submission order. A second
call waits for the current turn, so per-run callbacks cannot intercept another
turn's events. Use separate `Agent` or `AsyncAgent` instances when turns should
run concurrently. Each instance retains its own conversation, run gate, and
event routing state.

The public event catalog is stable and versioned; internal trace ordering is
trace-only. See [release policy](release-policy.md) and [tracing](tracing.md).
