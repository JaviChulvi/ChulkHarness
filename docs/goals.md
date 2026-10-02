# Durable goal context

`GoalService` owns revisioned goal controls and evidence. `GoalService.run()`
remains an audited state transition. `GoalRunner.run()` and the `goal run` CLI
coordinate foreground execution through bounded turns. `run_slice()` and CLI
`--single-slice` stop at the next slice or selected-step boundary. See the
[offline runner example](../examples/goal_runner.py). Hosts may also bind a
claimed `GoalExecutionContext` directly to an agent.

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

## Bounded goal slices

Set `execution.slice_limits = GoalSliceLimits()` on a claimed
`GoalExecutionContext` to enable continuation. Defaults are five tool attempts,
twenty provider calls and sixty cooperative seconds per slice. The existing
goal ledger still enforces its cumulative limits; a new slice does not reset it.
Repairs, fallback attempts, reflection and history summaries consume provider
allowance. Conservative reservations can yield before the nominal call limit.
An operation larger than a fresh slice raises `ConfigurationError`.

`Agent.continue_goal_slice()` and `await AsyncAgent.continue_goal_slice()`
resume the latest yielded turn in the same conversation, including after reopen.
They allocate a new turn and refresh host tool/context bindings without inserting
a user instruction. Pending validated actions and reflection outcomes survive
a yield. Slice exhaustion returns `RunStatus.YIELDED`, empty answer content and
`run.yielded`; it neither completes nor retries the authoritative goal step.
The foreground runner invokes this same continuation boundary.

A configured host verifier's decisions are persisted by migration 23. Rejection
counts are keyed by the selected step, authoritative criteria/instructions and
observed result content. The third rejection without changed evidence blocks
the goal with `verification_stagnation`. New IDs and model assertions are not
evidence. Raw result output remains in the owning session/artifact store; goal
checkpoints retain digests. Ordinary unbound turns retain their current behavior.


## Foreground coordinator

```console
chulk goal run GOAL_ID --revision REVISION --verifier project.verification:verify
chulk goal run GOAL_ID --revision REVISION --single-slice --verifier project.verification:verify
```

The project supplies a Python `PlanStepVerifier`; `CHULK_GOAL_VERIFIER` can name
its importable `module:callable`. This is host configuration, never model input.
The CLI preserves profile/model selection, tools and permission policy through
the shared runtime builder. Missing verification or a missing finite global
model-call budget stops before model construction or dispatch.

`GoalRunner(store, agent_factory=..., verifier=...)` takes an explicit
`GoalExecutionStore`. `AsyncGoalRunner` supports native `AsyncGoalExecutionStore`
and `async_verifier`; explicitly synchronous bindings use the cancellation-safe
service adapter. The factory receives the execution context and the conversation
ID to reopen (or `None` for first creation). It must bind that exact context and
use durable session recording and budget accounting. Custom accounting must
explicitly declare `enforces_goal_budgets = True` and implement cumulative goal
and slice enforcement; the in-memory demonstration service is insufficient.
Other hosted agents do not need a goal store.

Admission transactionally claims and starts the first eligible persisted step,
checking dependencies and selected-step approval. The execution conversation is
separate from its source and retains source conversation/turn provenance. The
turn plan projects the selected goal step. The host verifier receives an immutable
current `goal` snapshot alongside the step request; completion writes require
its persisted decision, current revision and live claim. A turn ending alone
cannot complete goal work. Evidence and completion are applied atomically once.

Claims last 120 seconds and renew every 40 seconds during model, tool and verifier
waits. Losing ownership stops dispatch and fences progress writes. Pause and
cancellation are observed at admission boundaries; known in-flight results drain
before release. Approving a step never resumes a user-paused goal.

`GoalExecutionResult` exposes the goal, conversation/turn/continuation identity,
cumulative usage and `GoalStopReason`. Reasons distinguish completion, a completed
step, yield, pause, cancellation, approval wait, blocking, budget exhaustion,
lease loss, required-context overflow and recovery required. `run()` continues
only yielded slices and completed-step boundaries. `run_slice()` returns both.
Global exhaustion records the exhausted budget and consumption. Use
`GoalService.update_budget()` (CLI `goal budget`) to explicitly replace the
budget, then resume the paused work. Counters and the absolute deadline are
never reset by a slice, pause or process restart.

Migration 24 persists execution conversations, preallocated slice identities and
verification application receipts. Migration 25 links goal checkpoints to durable
effects, stores recoverable effect results and marks dispatched usage reservations.
No hidden daemon is started. Hosts own durable scheduling, credentials, scope,
authorization, external reconciliation and process availability.

## Recovery and hosted execution

The runner acquires the goal claim, then the durable-run claim, in that order.
Both renew during execution and are checked before dispatch. SQLite goals use the
same database's `SQLiteRunStore` by default. Custom goal stores must supply `runs=`;
`AsyncGoalRunner` accepts either native `AsyncRunStore` or synchronous bindings.
Pass an explicit `execution_scope=` for hosted work, and bind the factory's runtime
to `context.execution_scope`. A durable run is an execution envelope, not another
independently evolving plan. Goal steps and criteria remain authoritative.

Hosted session stores must declare `supports_goal_recovery = True` and atomically
store the `turn` snapshot with model requests and responses. Summary persistence
must atomically apply `metadata.recovery_turn` with the summary. This is a storage
guarantee, not a flag to enable on an in-memory example. Accounting must implement
`mark_model_dispatched`, `mark_tool_dispatched` and `recover_unsent_model_request`,
in addition to cumulative goal/slice reservations. Native async methods are awaited;
explicit synchronous bindings use the cancellation-safe adapter.

Recovery restores the same interrupted turn, its validated action and execution
phase. Normal yields allocate a new turn. Stable operation identities survive both.
A known provider response is reused and its steering receipt acknowledged; a known
tool result is integrity-checked, reauthorized and reconstructed without executing
the original tool. Verification decisions, observations, evidence and progress are
idempotent. Terminal goal recovery finishes the durable envelope if a crash occurred
after the authoritative goal commit. An optional runner `event_sink=` replays pending
durable public events with stable event/idempotency identities; the host sink must
deduplicate delivery. Ordinary runtime event publication uses the same durable owner.

A dispatched mutation without a result remains uncertain and returns
`recovery_required`. `retry-step` cannot bypass an unresolved linked effect. The
host must reconcile through the run/effect store and supply trustworthy external
evidence. A digest alone cannot reconstruct a missing result. There is no exactly-once
promise for external systems that cannot confirm what happened.

Possibly billed model calls retain their reserved allowance across lease expiry,
TTL expiry and process restarts. Only durably unsent requests can release allowance
automatically. Provider repairs and fallback attempts recheck ownership before
calling another provider. Global deadlines remain absolute; model waits, streams
and tool timeouts are bounded by the remaining deadline. A timeout may leave remote
work in progress, so its unknown result remains subject to reconciliation.

Durable approval waits release the worker. Resolving approval revalidates scope,
tool identity, schema, arguments and policy; it does not resume a user-paused goal.
A yielded durable envelope returns to the worker queue without consuming another
logical attempt. Hosts use the same runner for local and hosted slices and schedule
subsequent invocations themselves. Background acceptance requires a separately
persisted execution request.
