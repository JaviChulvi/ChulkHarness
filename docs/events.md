# SDK events

Chulk projects selected runtime activity into one versioned public envelope.
Internal JSONL trace names and payloads are diagnostic implementation details;
they are not automatically exposed as SDK events.

```python
from chulk import Agent, EventName, RunCompletedPayload

with Agent(config=config, llm=client) as agent:
    for event in agent.run_events("Summarize this project"):
        print(event.name, event.turn_id)
        if event.name == EventName.RUN_COMPLETED.value:
            assert isinstance(event.payload, RunCompletedPayload)
            print(event.payload.result.content)
```

`AgentEvent` contains `name`, an ISO-8601 `timestamp`, `schema_version` (currently
`1`), `conversation_id`, an optional `turn_id`, a typed `payload`, and read-only
`extensions`. Dotted lowercase names are the compatibility-stable catalog:

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
| `plan.created` | `PlanPayload` | Plan mode created an approval-gated plan. |
| `plan.approved` | `PlanPayload` | A pending plan was approved. |
| `run.completed` | `RunCompletedPayload` | The terminal structured result is available. |
| `run.failed` | `RunFailedPayload` | The run failed before a normal result. |

Unknown future trace events are excluded until Chulk explicitly adds a public
projection. Permission payloads deliberately omit raw tool arguments.

## Iteration and callbacks

Use `agent.run_events(...)` for a synchronous iterator and
`async_agent.run_events_async(...)` for an async iterator. Events retain
execution order and end with exactly one `run.completed` or `run.failed` event.
The completion payload contains the same `RunResult` contract returned by
`run_result(...)`.

Constructor and per-run `on_event` callbacks receive the same public envelope.
`on_delta` remains supported and is driven by `model.delta` events.

```python
async for event in agent.run_events_async("Explain the change"):
    render(event)
```

Cancelling an async iteration cancels its worker task and releases callback
ownership. Closing a synchronous iterator waits for the in-flight operation to
finish, then removes its channel; Python threads are not force-killed.

## Concurrency policy

One facade serializes work-starting operations in submission order. A second
call waits for the current turn, so per-run callbacks cannot intercept another
turn's events. Use separate `Agent` or `AsyncAgent` instances when turns should
run concurrently. Each instance retains its own conversation, run gate, and
event routing state.
