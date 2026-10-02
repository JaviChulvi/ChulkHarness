# Durable goal context

`GoalService` owns revisioned goal controls and evidence. `GoalService.run()` and
the current `goal run` CLI command change a goal to `running`; they do not yet
schedule autonomous execution. A host can start a step, claim its execution and
pass `goal_execution` to `Agent`, `AsyncAgent`, `HostedRuntime`, or
`AsyncHostedRuntime`. See [the offline example](../examples/goal_context.py).

## Authoritative context

Goals persist an operational `description` and explicit `constraints` alongside
criteria, steps, progress, budget and steering. Description defaults to the
title. SQLite migration 22 backfills that default and an empty constraint list
for older snapshots; it never interprets historical summaries as restrictions.

Before each action, reflection, or incremental final-answer request, the runtime
reads the current goal and checks its claim, active step, pause and cancellation.
These controls operate independently of the local or hosted usage service.
The mandatory prompt section includes the description, constraints, criteria,
active step, compact progress, active instructions and evidence references.
At most six recent step evidence references are navigation hints; the complete
evidence remains in the goal store. Raw evidence output belongs in sessions or
artifacts, not the mandatory section. Summary requests compress history and do
not acknowledge steering.

Active instructions appear once per mandatory section, even after incorporation
or fulfillment. `GoalService.steer(..., supersedes=(instruction_id,))` explicitly
replaces selected active instructions while retaining their history. Unknown or
already superseded IDs are rejected. Neither the model nor the summarizer can
retire restrictions. Ordinary agents without `goal_execution` are unchanged.

## Incorporation and fulfillment

Each work-directing request has a `GoalModelRequest` receipt with the goal
revision, instruction IDs, conversation, turn, request index and purpose.
Admission checks the revision and live claim transactionally. Steering admitted
before that boundary is included; steering arriving during the request is read
at the next boundary.

The runtime acknowledges incorporation only after its response recorder
succeeds. Async execution flushes the recorder before acknowledgment. A
failed response write leaves the receipt pending and the steering unincorporated.
The host must honor the session or journal durability contract; an in-memory
service fixture does not survive process failure.

`GoalStore.model_requests(goal_id, pending_only=True)` exposes pending receipts.
If a crash occurs after the response was durably recorded but before its receipt
was acknowledged, a claim owner may reconcile it with
`GoalExecutionContext.acknowledge_response(request_id, response_ref=...)` after
checking the response in its session or journal. References use
`model-response:<conversation-id>:<turn-id>:<request-index>`. Acknowledgment is
idempotent for the same reference and rejects a different one. A pending receipt
does not authorize replaying its provider request or tool action.

Incorporation means the instruction was included in a request with a durably
recorded response. It does not prove the instruction was obeyed.
`GoalService.fulfill_steering()` records a separate host decision, requiring
nonempty references to known `GoalEvidence` IDs and an audit actor. It neither
completes a step nor retires the instruction. Prompt markers expose incorporation
and fulfillment separately.

## Size and authority

Mandatory state is allocated before optional history. Compaction may shorten
history and evidence text but never the mandatory goal section. If required
context cannot fit, `context_budget_exceeded` stops execution before any provider
request, including a summary request. The context report identifies the overage;
the operator must explicitly change the goal or configured window.

Goal text does not grant tools, broaden `ExecutionScope`, load credentials, or
override host policy and approvals. Evidence references are data, not authority.
See [permissions](permissions.md), [configuration](configuration.md),
[SDK errors](sdk-errors.md), and [hosted services](hosting.md).
