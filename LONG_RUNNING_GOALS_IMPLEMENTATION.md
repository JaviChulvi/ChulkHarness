# Long-running goals implementation

## Authorized delivery

Implement the four dependent draft PRs requested in this task. Merging and deployment
are separate actions. Work remains in `/private/tmp/chulk-long-running-goals`;
the original analysis checkout and unrelated user changes are untouched.

| Stage | Branch | Base | Draft PR | Validation |
| --- | --- | --- | --- | --- |
| Authoritative context and steering | feat/goal-authoritative-context | main | [#128](https://github.com/JaviChulvi/ChulkHarness/pull/128) | Full Linux/PostgreSQL CI passed at 598fce5, run 36930051500 |
| Bounded slices and stagnation | feat/goal-bounded-slices | feat/goal-authoritative-context | [#129](https://github.com/JaviChulvi/ChulkHarness/pull/129) | Full Linux/PostgreSQL CI passed at b46906b, run 36933258603 |
| Foreground CLI and SDK runner | feat/goal-local-runner | feat/goal-bounded-slices | [#130](https://github.com/JaviChulvi/ChulkHarness/pull/130) | Full Linux/PostgreSQL CI passed at 613f106, run 36935301936 |
| Durable recovery and hosted execution | feat/goal-hosted-execution | feat/goal-local-runner | Ready to publish | Final local checks passed apart from documented macOS baseline |

## Ownership and contracts

- `GoalService.run()` remains an audited transition. Sync/async `GoalRunner` owns
  foreground continuation through the existing action loop; hosts own scheduling.
- Mandatory objective, constraints, criteria, active instructions and progress are
  separate from historical summaries. Oversized mandatory state rejects admission.
- Steering receipts distinguish inclusion, durable-response incorporation and
  host-evidenced fulfillment. Supersession is explicit; fulfillment retains rules.
- Slice defaults are five tool attempts, twenty provider calls and sixty cooperative
  seconds. Yields preserve pending actions and do not consume logical step retries.
  Goal budget and absolute deadline remain cumulative across pauses and restarts.
- Three same-context, same-evidence verification rejections block persistently.
  No additional LLM judge or text-based stagnation heuristic is introduced.
- Automatic goals require a verifier, durable session/usage bindings and finite
  global model-call allowance. Goal steps are authoritative; turn plans project them.
- Goal claim is acquired before the durable-run claim. Both renew at 40 seconds
  with 120-second leases; dispatch and progress check ownership. Hosted approvals
  release workers and never override a subsequent user pause.
- Stable operation IDs link goal checkpoints to existing durable effect records.
  Known results are integrity checked and reconstructed after current authorization.
  Uncertain effects require host reconciliation; retry-step cannot bypass them.
- Possibly billed requests keep reservations until reconciled. Only durably unsent
  work releases automatically. Known model responses, verification decisions and
  goal progress recover idempotently, without repeating original external actions.
- Provider-hosted MCP execution is rejected for automatic goals because it bypasses
  the effect journal. Registered harness tools retain current host authority.
- No exactly-once external-effect guarantee, hidden daemon or automatic merge.

## Persistence

SQLite migrations 22 (context/receipts), 23 (verification), 24 (execution slices)
and 25 (effect receipts, dispatch holds, slice-start markers) are forward-only.
PostgreSQL migration 0006 extends the existing run/effect adapters with recoverable
result references and digests. Legacy snapshots receive explicit safe defaults;
mandatory restrictions are never inferred from summaries.

## Validation checkpoint

Stage 4 deterministic recovery tests cover crashes before the first turn, before
and after dispatch, after result/observation/response storage, around verification
and goal-progress commits, and around reflection/final-answer/summary responses.
They also cover withheld unknown usage, native async hosted approval recovery,
user pause, global deadline repair admission, independent durable lease loss,
async heartbeat while a synchronous tool waits, and unsupported opaque effects.

The final focused recovery/provider/profile/streaming run passed 141 cases,
including 44 goal-recovery cases. Full local coverage passed 2,117 tests, skipped
45 PostgreSQL/optional cases, and reached 81.97%. Ten existing macOS filesystem,
replay-export and platform-error-message tests fail on this host; the preceding
stage passed all required Linux/PostgreSQL CI checks.

Ruff, Linux-platform mypy (336 source files), compileall, docs (20 topics),
quickstart, review bot, goal-runner example and deterministic evaluation (2/2)
passed. Clean-wheel installation/import/example smoke passed. A final rebuild
includes the last model-profile forwarding fix. Required PR CI must still verify
the final head, including PostgreSQL migration 0006 and shared adapter coverage.

Stage 4 consists of a persistence/effect/accounting commit and a goal recovery,
coordination, provider-boundary and regression-test commit. All publication uses
follow-up commits; published history is preserved.
