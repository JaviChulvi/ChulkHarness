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
| 1: authoritative context and steering | feat/goal-authoritative-context | main | draft; full CI green | 598fce5 | https://github.com/JaviChulvi/ChulkHarness/pull/128 |
| 2: bounded slices and stagnation | feat/goal-bounded-slices | feat/goal-authoritative-context | implemented; validating | pending | pending |
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
- PR 1 published and attached. GitHub had no CI run because new main `247c9e7`
  conflicted with API exports. Integrate it with a follow-up merge commit (no
  history rewrite), preserve removed surfaces and revalidate the changed owners.
- Final pre-main-merge wheel clean installation/import/example smoke passed.

## Pending risks

- Persist model-response acknowledgment only after the response is durable,
  including async flush and recovery after a crash between the two writes.
- Goal controls must operate independently of hosted usage services.
- Heartbeat must cover provider/tool/verifier waits and drain paused work safely.
- Preserve output and accounting across the tool/session/ledger/goal boundaries.
- Draft PR publication and CI must be verified rather than assumed.

## Current implementation progress

- The user explicitly approved the complete four-stage implementation plan.
- PR 1 head `a86ac05` CI ran the Linux/PostgreSQL suite: 2087 passed, one
  skipped, three failed. The failures were the documentation topic count and
  the replay transport missing the new asynchronous prompt method.
- Both owning fixes are implemented. Local documentation/context/replay tests
  passed 44 cases; two CLI replay export cases require Linux atomic-publication
  primitives. Ruff, Linux-platform mypy (331 files), and docs checks passed.
- Next: confirm repaired PR 1 CI, then implement PR 2 bounded slices and
  persistent stagnation on `feat/goal-bounded-slices`. Stages 2-4 are still
  required; publication of PR 1 alone does not complete the objective.

## Stage 2 execution checkpoint (2026-10-02)

- PR 1 full required CI passed: run 36930051500, head 598fce5.
- Stage 2 adds immutable GoalSliceLimits, ledger-backed dynamic turn constraints,
  YIELDED results/public events (schema 4; readers retain 1-3), explicit sync/async
  continuation, and pending action/reflection/attempt checkpoints in sessions.
- Tool admission precedes counters; refused summary admission preserves history.
  JSON repairs now commit all provider calls. The prior tool failure guard
  survives slices. Native async goal persistence currently uses drained sync
  bindings; native goal-store protocol work belongs to stage 3.
- Migration 23 persists fenced/idempotent verification decisions and blocks on
  three same-context/same-evidence rejections. Result digests exclude new IDs.
- 19 deterministic slice tests pass, covering 5/5/2, reopen, cumulative global
  budgets, reflection/final streaming, retry phases, stagnation and migration.
- Full macOS coverage run: 2056 passed, 44 skipped, 10 platform failures; coverage
  81.68%. Failures are existing Linux atomic-publication/replay export behavior
  and platform-specific embedded-NUL error wording. See /tmp/chulk-stage2-pytest.log.
  An earlier run was interrupted after an outdated Goal test double caused a
  cancellation test to wait indefinitely; fixed the double and bounded the wait.
- Ruff, Linux-platform mypy (331 files), compileall, docs (20 topics), quickstart
  and review-bot examples passed. Clean wheel installation/import/example smoke
  passed after moving stale generated build output aside.
- Latest changes refresh continuation prompt guidance and release a demonstrably
  unsent reservation when goal receipt admission rejects it. Focused recheck in
  progress; required Linux/PostgreSQL CI will run on the published PR.
- Still required: publish/attach draft PR 2, then implement and publish stages
  3 and 4. Do not claim the four-stage objective complete after this slice.

## Stage 3 design notes for continuation

Use a small sync/async GoalRunner around the existing Agent loop; require an
explicit verifier, durable recorder/accounting and finite goal model-call limit
before admission. Keep GoalService.run audited. Add an owner transaction for
claim + ordered eligible-step selection/start + persisted conversation/slice IDs.
Use a dedicated execution conversation and a projection of the chosen goal step.
Fence progress with token and revision; persist verification/evidence/completion
idempotently. Heartbeat 120/40 through model/tool/verifier waits, preserving owned
in-flight draining after pause/cancel. CLI goal run invokes runner foreground;
single-slice flag and explicit project verifier configuration fail before model
use if missing. Native async store operations must be awaited; sync bindings use
call_async_service. Do not introduce a second model/action loop.

Stage 4 must extend durable effect/run owners and PostgreSQL adapters, preserve
unknown model reservations and recover known responses/results before usage,
observations, verification, progress and events. Stable operation IDs cannot be
just turn IDs/argument hashes; dispatched mutations without known results block
for reconciliation and retry-step must not bypass them. The current stage-2
turn checkpoints support known slice yields, not arbitrary external-effect crash
replay. Existing InMemoryUsageService does not enforce slice budgets: automatic
hosted execution must require a genuine durable usage binding.


## Stage 3 execution checkpoint (2026-10-02)

- PR2 published as draft #129, head b46906b, base PR128; attached to this task.
  Required Linux/PostgreSQL CI passed (run 36933258603).
- Current branch feat/goal-local-runner adds foreground GoalRunner/AsyncGoalRunner,
  explicit sync/async store protocols and immutable result/stop types. Shared loop
  performs selected-step projections, current-goal verification, cumulative usage,
  pause/cancel draining, claim renewal and fenced progress. Native async bindings
  are awaited; synchronous verifier/store calls use the draining adapter.
- Migration24 persists goal execution conversations, slices and applied verification
  receipts. GoalService.run remains a transition; CLI goal run now coordinates work
  with project --verifier/CHULK_GOAL_VERIFIER and --single-slice. Budget replacement
  is explicit via GoalService.update_budget/goal budget and requires resume.
- 18 runner tests plus 5 CLI tests cover multi-slice/step completion, dependencies,
  approval/pause, cancellation, lease loss/renewal, native async admission, budget
  recovery, verification context, final-answer continuation, mandatory overflow,
  migrations and stale/concurrent writes. Public exports count548; typing/example
  and documentation updated. Interrupted slices still report recovery_required;
  effect reconciliation is deliberately the next coherent stage.
- Full local run:2070 passed/44 skipped/14 failed,81.28% coverage. Ten failures are
  the documented macOS atomic-publication baseline. Four were outdated hosted
  GoalContext test doubles missing assert_boundary_async; corrected and retesting.
  Latest focused runner tests18 pass; Ruff/mypy334 pass. Clean wheel validation
  running. Must obtain PR3 CI green and finish stage4 before claiming completion.
