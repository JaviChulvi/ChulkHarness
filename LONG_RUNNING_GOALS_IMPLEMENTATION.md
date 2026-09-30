# Long-running goals implementation checkpoint

## Objective and authority

Implement the four stacked draft PRs requested in the attached implementation
brief. No merge or deployment. Ordinary agents keep their default behavior.
The study at `4fea281` is orientation only; implementation starts from freshly
fetched `origin/main` at `c5c6aa7` in `/private/tmp/chulk-long-running-goals`.
The original study checkout is untouched.

## Decisions

- Keep `GoalService.run()` as a state transition; add a small goals coordinator.
- Reuse the existing action loop, plan verifier, sessions, usage ledger, effects,
  approvals, and runtime assembly. No separate action loop or LLM judge.
- Mandatory goal context is authoritative persistent state, never summary text.
- Incorporation receipts acknowledge a durable model response, not fulfillment.
- Active steering remains context until explicitly superseded; retain history.
- Slice yields are recoverable, never step completion or failure. Step attempts
  do not increment when continuing a slice.
- Three unchanged-evidence verification rejections block a step persistently.
- Automatic execution requires an explicit finite global model-call budget.
- Uncertain dispatched mutations require reconciliation before retry.
- A possibly billed provider request retains its reservation after interruption.

## Stack

| PR | Branch | Base | State | Commit | URL |
| --- | --- | --- | --- | --- | --- |
| 1: authoritative context and steering | feat/goal-authoritative-context | main | validated locally; publication pending | pending | pending |
| 2: bounded slices and stagnation | feat/goal-bounded-slices | feat/goal-authoritative-context | pending | pending | pending |
| 3: local runner, CLI, SDK | feat/goal-local-runner | feat/goal-bounded-slices | pending | pending | pending |
| 4: hosted durable execution | feat/goal-hosted-execution | feat/goal-local-runner | pending | pending | pending |

## Contracts and migrations

Base SQLite schema: 21. PR 1 adds forward-only migration 22.
Keep legacy readers compatible through explicit defaults, without interpreting
historical conversation summaries as authoritative restrictions.
Update public exports, typed results/events/errors, sync/async behavior, docs,
typing tests, and examples together with each affected delivery.

## Evidence and validation

- PR 1: 113 focused tests passed, including 24 new context/receipt tests,
  migration rollback, concurrent initializers, local/hosted sync/async controls,
  successive compactions, reflection and incremental answer receipts.
- Ruff passed. Mypy `--platform linux src/chulk examples typing_tests` passed
  (355 files); native macOS mypy hits existing Linux-only `os.O_TMPFILE` typing.
- Documentation check passed (21 topics), compileall passed, goal context,
  quickstart and review bot examples passed; deterministic eval passed 2/2 cases.
- Wheel build and clean installation/import/example smoke passed. Rebuild after
  final migration transaction fix before publishing.
- Full local suite before final fixes: 2113 passed, 44 skipped, 15 failed.
  Two export inventory failures were repaired and rechecked. Twelve require
  Linux atomic-publication primitives; one wheel dependency install needed
  network access (subsequent clean-wheel check passed with access).
- Docker daemon is unavailable and localhost PostgreSQL is not serving. Full
  Linux/PostgreSQL verification must be obtained from required GitHub CI.
- Required final checks: full coverage suite, Ruff, mypy (including examples and
  typing_tests), compileall, docs, credential-free examples, deterministic eval,
  clean wheel build and installation. PostgreSQL adapters as required by CI.
- Historical migration scripts now preserve the writer transaction instead of
  `executescript` implicitly committing; adversarial rollback and concurrent
  initialization tests pass without hiding duplicate-table errors.

## Pending risks

- Persist model-response acknowledgment only after the response is durable,
  including async flush and recovery after a crash between the two writes.
- Goal controls must operate independently of hosted usage services.
- Heartbeat must cover provider/tool/verifier waits and drain paused work safely.
- Preserve output and accounting across the tool/session/ledger/goal boundaries.
- Draft PR publication and CI must be verified rather than assumed.

## Next action

Rebuild final PR 1 wheel, publish draft PR and verify required CI. Then branch
PR 2 from PR 1: opt-in bounded slices and persistent stagnation detection.
